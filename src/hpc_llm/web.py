"""Approved public retrieval; no model, implicit preview, or provider fallback.

DDGS 9.16's DuckDuckGo payload/extractor is used with this module's HTTP
transport. The upstream random-impersonating/proxy-aware client is not used.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import ssl
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urljoin, urlsplit, urlunsplit
import zlib

import aiohttp
from aiohttp.abc import AbstractResolver

from .contracts import AppError, SourceRecord

MAX_WIRE = 2 * 1024 * 1024
MAX_BODY = 4 * 1024 * 1024
MAX_TEXT = 80000
MAX_REDIRECTS = 3
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DDGS_ENDPOINT = "https://html.duckduckgo.com/html/"
DEMO_URL = "https://example.org/hpc-llm-demo/research"
DEMO_HTML = b"""<!doctype html><html><head><title>Synthetic allocation study</title>
<meta property="article:published_time" content="2026-01-01"></head><body>
<h1>Synthetic allocation study</h1><p>This controlled fixture is not a real study.
In 12 synthetic sessions, persistent allocations survived terminal detach.
Application-managed search works without native model tools.</p></body></html>"""


def clean_text(value: str, limit: int = MAX_TEXT) -> str:
    """Keep source data as data; terminal controls never become terminal output."""
    return "".join(c for c in value if (ord(c) >= 32 or c in "\n\t") and not 127 <= ord(c) <= 159 and c not in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")[:limit]


def _public_ip(value: str) -> str:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        raise AppError("permission", "The hostname returned an invalid network address.") from None
    if (not ip.is_global or ip.is_multicast or ip.is_unspecified or ip.is_reserved
            or getattr(ip, "ipv4_mapped", None) is not None
            or (isinstance(ip, ipaddress.IPv6Address) and (
                ip in ipaddress.ip_network("64:ff9b::/96")
                or ip in ipaddress.ip_network("64:ff9b:1::/48")
                or ip in ipaddress.ip_network("2002::/16")
                or ip in ipaddress.ip_network("2001::/32")))):
        raise AppError("permission", "Only public internet addresses may be fetched; private, local and metadata addresses are blocked.")
    return str(ip)


def normalize_public_url(url: str) -> str:
    """Validate syntax before any DNS, including unusual encoded host forms."""
    if not isinstance(url, str) or len(url) > 4096 or not url:
        raise AppError("validation", "Enter a public HTTP or HTTPS URL (at most 4096 characters).")
    if any(ord(c) < 33 or ord(c) == 127 for c in url) or "\\" in url:
        raise AppError("permission", "URL controls, whitespace and backslashes are not allowed.")
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in {"https", "http"} or not parts.hostname:
            raise ValueError
        if parts.username is not None or parts.password is not None:
            raise ValueError
        host = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        port = parts.port
    except (ValueError, UnicodeError):
        raise AppError("permission", "Use a public HTTP(S) URL without embedded credentials.") from None
    if "%" in host or not host or port not in {None, 80, 443}:
        raise AppError("permission", "Encoded hosts and nonstandard public ports are not allowed.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if re.fullmatch(r"[0-9.]+", host) or "." not in host or not re.fullmatch(r"[a-z0-9.-]+", host) or any(not label or len(label) > 63 or label.startswith("-") or label.endswith("-") for label in host.split(".")):
            raise AppError("permission", "A fully qualified public hostname is required.") from None
        blocked = (".localhost", ".local", ".internal", ".intranet", ".lan", ".home.arpa", ".cluster", ".hpc.wehi.edu.au")
        if any(host == suffix[1:] or host.endswith(suffix) for suffix in blocked):
            raise AppError("permission", "Internal and cluster hostnames are not public sources.")
    else:
        _public_ip(str(ip))
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    return urlunsplit((parts.scheme.lower(), authority, parts.path or "/", parts.query, ""))


@dataclass(frozen=True)
class ResolvedURL:
    url: str
    host: str
    port: int
    addresses: tuple[str, ...]


class PinnedResolver(AbstractResolver):
    """The connector receives only the addresses validated for this request.

    The URL hostname is unchanged, so aiohttp performs hostname-aware TLS
    verification against the requested hostname, not the pinned IP.
    """
    def __init__(self, target: ResolvedURL):
        self.target = target

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        if host.lower().rstrip(".") != self.target.host or port != self.target.port:
            raise OSError("Pinned connection hostname mismatch")
        return [{"hostname": host, "host": address, "port": port,
                 "family": socket.AF_INET6 if ":" in address else socket.AF_INET,
                 "proto": socket.IPPROTO_TCP, "flags": socket.AI_NUMERICHOST}
                for address in self.target.addresses]

    async def close(self):
        pass


@dataclass
class HTTPResult:
    status: int
    headers: dict[str, str]
    body: bytes


class BoundedBody:
    """Streaming wire and decompressed limits, before parsers see any bytes."""
    def __init__(self, encoding: str):
        self.wire = 0
        self.output = bytearray()
        encoding = encoding.strip().lower()
        if encoding in {"", "identity"}:
            self.decoder = None
        elif encoding in {"gzip", "deflate"}:
            self.decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
        else:
            raise AppError("unsupported", "The server used an unsupported content encoding.")

    def feed(self, chunk: bytes):
        self.wire += len(chunk)
        if self.wire > MAX_WIRE:
            raise AppError("parser_limit", "Public response exceeds the 2 MiB transfer limit.")
        try:
            output = self.decoder.decompress(chunk, MAX_BODY - len(self.output) + 1) if self.decoder else chunk
        except zlib.error:
            raise AppError("provider", "The server returned invalid compressed content.") from None
        self.output.extend(output)
        if len(self.output) > MAX_BODY or (self.decoder and self.decoder.unconsumed_tail):
            raise AppError("parser_limit", "Public response exceeds the 4 MiB decompression limit.")

    def finish(self) -> bytes:
        if self.decoder and (not self.decoder.eof or self.decoder.unused_data):
            raise AppError("provider", "The server returned incomplete or concatenated compressed content.")
        return bytes(self.output)


def _private_directory(path: Path):
    path = path.absolute()
    for ancestor in (*reversed(path.parents), path):
        if ancestor.is_symlink():
            raise AppError("permission", "Private web settings must not pass through symbolic links.")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise AppError("permission", "Web settings directory must be owned by you with permissions 0700.")


def _private_read(path: Path) -> str | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError:
        raise AppError("permission", "Cannot read private web settings safely.") from None
    with os.fdopen(fd, "r") as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4096:
            raise AppError("permission", "Web settings file must be a private regular file owned by you.")
        return stream.read(4097)


def _private_write(path: Path, text: str):
    _private_read(path)  # Reject symlinks, hard links and unsafe existing files.
    temporary = path.with_name(path.name + "." + os.urandom(8).hex() + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class WebService:
    def __init__(self, config_dir: Path, demo: bool = False):
        self.config_dir = Path(config_dir)
        _private_directory(self.config_dir)
        self.demo = demo
        self._sequence = int(_private_read(self.config_dir / "source-sequence") or "0")
        # Narrow dependency seams for controlled tests; not exposed as user/model settings.
        self._resolver = self._dns_addresses
        self._transport = self._network_request

    def status(self, provider: str = "ddgs") -> dict[str, Any]:
        if self.demo:
            return {"provider": "fixture", "configured": True, "available": True, "label": "DEMO — controlled fixtures, no internet", "external": False}
        if provider == "brave":
            configured = bool(_private_read(self.config_dir / "brave.key"))
            return {"provider": provider, "configured": configured, "available": configured, "label": "Brave Search API", "external": True,
                    "message": "Ready for an approved query; compute-node egress is untested." if configured else "Add your Brave Search API key in Web settings. No account or plan is created automatically."}
        if provider != "ddgs":
            raise AppError("validation", "Choose DuckDuckGo or Brave Search.")
        return {"provider": "ddgs", "configured": True, "available": True, "label": "DuckDuckGo (no key, best effort)", "external": True,
                "message": "Fixed DuckDuckGo backend via DDGS 9.16.0; blocking and throttling are possible. No automatic provider fallback."}

    def set_brave_key(self, key: str):
        if not isinstance(key, str) or not 8 <= len(key.strip()) <= 512 or not key.isascii() or any(ord(c) < 33 or ord(c) == 127 for c in key.strip()):
            raise AppError("validation", "Enter a valid Brave key without whitespace or control characters.")
        _private_write(self.config_dir / "brave.key", key.strip())
        return self.status("brave")

    async def _dns_addresses(self, host: str, port: int) -> list[str]:
        try:
            addresses = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM), 8)
        except (OSError, TimeoutError):
            raise AppError("network", "Public DNS lookup failed. Check compute-node egress; offline chat is still available.") from None
        return list(dict.fromkeys(entry[4][0] for entry in addresses))

    async def _resolve(self, url: str) -> ResolvedURL:
        url = normalize_public_url(url)
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            addresses = [str(ipaddress.ip_address(host))]
        except ValueError:
            addresses = await self._resolver(host, port)
        if not addresses:
            raise AppError("network", "The public hostname did not resolve to an address.")
        return ResolvedURL(url, host, port, tuple(_public_ip(address) for address in addresses))

    async def _network_request(self, target: ResolvedURL, method: str, headers: dict, payload: Any = None) -> HTTPResult:
        connector = aiohttp.TCPConnector(resolver=PinnedResolver(target), use_dns_cache=False,
                                        ssl=ssl.create_default_context(), force_close=True)
        timeout = aiohttp.ClientTimeout(total=20, connect=8, sock_read=10)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False,
                                         auto_decompress=False, cookie_jar=aiohttp.DummyCookieJar()) as session:
            kwargs = {"json": payload} if headers.get("Content-Type") == "application/json" else {"data": payload}
            async with session.request(method, target.url, headers=headers, allow_redirects=False, **kwargs) as response:
                result_headers = {k.lower(): v for k, v in response.headers.items()}
                if response.status in {301, 302, 303, 307, 308}:
                    return HTTPResult(response.status, result_headers, b"")
                length = result_headers.get("content-length", "")
                if length.isdigit() and int(length) > MAX_WIRE:
                    raise AppError("parser_limit", "Public response exceeds the 2 MiB transfer limit.")
                body = BoundedBody(result_headers.get("content-encoding", ""))
                async for chunk in response.content.iter_chunked(32768):
                    body.feed(chunk)
                return HTTPResult(response.status, result_headers, body.finish())

    async def _request(self, url: str, method: str = "GET", headers: dict | None = None, payload: Any = None, redirects: bool = True) -> tuple[HTTPResult, str]:
        headers = {"User-Agent": "hpc-llm/0.1 (public research retrieval)", "Accept-Encoding": "identity", **(headers or {})}
        try:
            async with asyncio.timeout(35):
                for step in range(MAX_REDIRECTS + 1):
                    target = await self._resolve(url)
                    response = await self._transport(target, method, headers, payload)
                    if response.status in {301, 302, 303, 307, 308}:
                        if not redirects or step >= MAX_REDIRECTS:
                            raise AppError("provider", "The provider redirected unexpectedly or the public redirect limit was reached.")
                        location = response.headers.get("location")
                        if not location:
                            raise AppError("provider", "The public page returned a redirect without a destination.")
                        url = urljoin(target.url, location)
                        continue
                    if response.status in {401, 403}:
                        raise AppError("provider", "The provider refused access (HTTP %s). Check the key or site access policy; offline chat is available." % response.status)
                    if response.status == 429:
                        raise AppError("provider", "The provider is rate limiting requests (HTTP 429). Try later; no provider was switched.")
                    if not 200 <= response.status < 300:
                        raise AppError("provider", f"The public service returned HTTP {response.status}; no results were invented.")
                    if len(response.body) > MAX_BODY:
                        raise AppError("parser_limit", "Public response exceeds the 4 MiB decompression limit.")
                    return response, target.url
        except AppError:
            raise
        except (TimeoutError, aiohttp.ClientError, OSError):
            raise AppError("network", "Public connection failed or timed out. Check compute-node egress/TLS; offline chat remains available.") from None
        raise AppError("provider", "Public redirect limit reached.")

    def _source(self, **kwargs) -> SourceRecord:
        self._sequence += 1
        _private_write(self.config_dir / "source-sequence", str(self._sequence))
        text = clean_text(kwargs.pop("text"))
        title = clean_text(kwargs.pop("title"), 512)
        fingerprint = hashlib.sha256((kwargs.get("url", "") + "\0" + kwargs.get("locator", "") + "\0" + text).encode()).hexdigest()
        return SourceRecord(id=f"W{self._sequence}", title=title, text=text, fingerprint=fingerprint, **kwargs)

    async def search(self, query: str, provider: str = "ddgs") -> list[SourceRecord]:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000 or any(ord(c) < 32 for c in query):
            raise AppError("validation", "Enter a search query of 1–2000 characters without control characters.")
        if self.demo:
            rows = [{"title": "Synthetic allocation study (DEMO)", "url": DEMO_URL,
                     "description": "Controlled fixture: 12 synthetic sessions retained their allocation after detach. No external search was performed."}]
            label = "fixture"
        elif provider == "brave":
            key = _private_read(self.config_dir / "brave.key")
            if not key:
                raise AppError("provider", "Brave Search needs your API key. Add it in Web settings or explicitly select the no-key provider.")
            response, _ = await self._request(BRAVE_ENDPOINT, "POST", {"Accept": "application/json", "Content-Type": "application/json", "X-Subscription-Token": key}, {"q": query, "count": 6}, redirects=False)
            try:
                data = json.loads(response.body)
                rows = data.get("web", {}).get("results", [])
                if not isinstance(rows, list):
                    raise ValueError
            except (ValueError, AttributeError, RecursionError):
                raise AppError("provider", "Brave returned malformed search results. Retry later or use offline chat.") from None
            label = "brave"
        elif provider == "ddgs":
            response, _ = await self._request(DDGS_ENDPOINT, "POST", {"Content-Type": "application/x-www-form-urlencoded"}, {"q": query, "b": "", "l": "us-en"}, redirects=False)
            rows = await _parse_isolated("ddgs", response.body)
            label = "DuckDuckGo via DDGS 9.16.0"
        else:
            raise AppError("validation", "Unknown web provider; choose DuckDuckGo or Brave.")
        sources = []
        for row in rows[:6]:
            if not isinstance(row, dict):
                raise AppError("provider", "Search returned a malformed result entry.")
            url, title, description = row.get("url", row.get("href")), row.get("title"), row.get("description", row.get("body"))
            if not all(isinstance(value, str) for value in (url, title, description)):
                raise AppError("provider", "Search returned missing or malformed source fields.")
            try:
                if provider == "ddgs":
                    # The HTML provider sometimes wraps outbound links; decode
                    # the destination locally without following the tracker.
                    if url.startswith("//"):
                        url = "https:" + url
                    split = urlsplit(url)
                    if split.hostname in {"duckduckgo.com", "www.duckduckgo.com"} and split.path == "/l/":
                        destinations = parse_qs(split.query).get("uddg", [])
                        if len(destinations) != 1:
                            continue
                        url = destinations[0]
                url = normalize_public_url(url)
            except AppError:
                continue  # A malicious/internal result is never a fetchable source.
            sources.append(self._source(title=title, kind="web_snippet", text=description[:5000], url=url,
                                        locator=f"Search snippet · {label} · page not fetched", partial=True))
        return sources

    async def fetch(self, url: str) -> SourceRecord:
        original = normalize_public_url(url)
        if self.demo:
            if original != DEMO_URL:
                raise AppError("permission", "DEMO can fetch only the controlled example.org fixture; no internet requests are made.")
            response, fetched = HTTPResult(200, {"content-type": "text/html"}, DEMO_HTML), original
        else:
            response, fetched = await self._request(original)
        mime = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if mime == "application/pdf":
            from .files import parse_public_pdf
            parsed = await asyncio.to_thread(parse_public_pdf, response.body)
            chunks = parsed.get("sources", [])
            text = "\n\n".join(f"[{chunk['locator']}]\n{chunk['text']}" for chunk in chunks)
            locator = "; ".join(chunk["locator"] for chunk in chunks)
            partial = parsed.get("status") != "ready" or len(text) > MAX_TEXT
            title, published = urlsplit(fetched).path.rsplit("/", 1)[-1] or "Public PDF", None
        elif mime in {"text/html", "application/xhtml+xml"}:
            parsed = await _parse_isolated("html", response.body)
            title, text, published = parsed["title"], parsed["text"], parsed["published_at"]
            locator = "Fetched HTML text (scripts, styles and forms excluded)"
            partial = len(text) > MAX_TEXT
        elif mime in {"text/plain", "text/markdown"}:
            text = response.body.decode("utf-8", "replace")
            title, published = urlsplit(fetched).path.rsplit("/", 1)[-1] or "Public text", None
            locator, partial = "Fetched text", len(text) > MAX_TEXT
        else:
            raise AppError("unsupported", "This public content type is unsupported. Use an HTML, text or text-based PDF source.")
        if not text.strip():
            raise AppError("unsupported", "No readable text was extracted. JavaScript and OCR are not executed.")
        if partial:
            locator += "; partial coverage (bounded excerpt)"
        return self._source(title=title, kind="web_page", text=text, url=original, fetched_url=fetched,
                            locator=locator[:2000], published_at=published, partial=partial)


def _parse_html(body: bytes) -> dict:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(body, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Public webpage"
    published = None
    for key in ("article:published_time", "datePublished"):
        element = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if element and isinstance(element.get("content"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T[0-9:.+Z-]+)?", element["content"]):
            try:
                datetime.fromisoformat(element["content"].replace("Z", "+00:00"))
            except ValueError:
                continue
            published = element["content"][:64]
            break
    for item in soup(["script", "style", "noscript", "form", "iframe", "svg", "template"]):
        item.decompose()
    return {"title": title, "text": soup.get_text("\n", strip=True)[:MAX_TEXT + 1], "published_at": published}


def _parse_ddgs(body: bytes) -> list[dict]:
    from ddgs.engines.duckduckgo import Duckduckgo
    # Only parser methods are used: no upstream HTTP client (random impersonation,
    # proxy environment inheritance, or implicit provider fallback) is created.
    class FixedDuckDuckGo(Duckduckgo):
        def __init__(self):
            self.results = []
    engine = FixedDuckDuckGo()
    try:
        text = body.decode("utf-8", "replace")
        results = engine.post_extract_results(engine.extract_results(text))
        if not results and ("anomaly" in text.lower() or "captcha" in text.lower()):
            raise AppError("provider", "DuckDuckGo blocked automated access. No CAPTCHA bypass was attempted; use offline chat or explicitly select Brave.")
        return [{"title": r.title[:512], "href": r.href[:4097], "body": r.body[:5000]} for r in results[:6]]
    except AppError:
        raise
    except Exception:
        raise AppError("provider", "DuckDuckGo returned an unreadable search response; use offline chat or try later.") from None


async def _parse_isolated(kind: str, body: bytes) -> Any:
    """Untrusted markup is parsed in a killable CPU/memory/wall-time worker."""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "hpc_llm.web", "--parse", kind,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(8):
            stdout, _ = await process.communicate(body)
        if process.returncode or len(stdout) > 1024 * 1024:
            raise AppError("parser_limit", "Public markup could not be parsed within resource limits.")
        result = json.loads(stdout)
        if "error" in result:
            raise AppError(result["code"], result["error"])
        return result["result"]
    except (TimeoutError, ValueError):
        raise AppError("parser_limit", "Public markup parser reached its time or output limit.") from None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def _worker():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    data = sys.stdin.buffer.read(MAX_BODY + 1)
    try:
        if len(data) > MAX_BODY:
            raise AppError("parser_limit", "Public markup exceeds the parsing limit.")
        if sys.argv[-1] == "html":
            result = _parse_html(data)
        elif sys.argv[-1] == "ddgs":
            result = _parse_ddgs(data)
        else:
            raise AppError("validation", "Unknown parser.")
        sys.stdout.write(json.dumps({"result": result}))
    except AppError as error:
        sys.stdout.write(json.dumps({"error": error.message, "code": error.code}))
    except Exception:
        sys.stdout.write(json.dumps({"error": "Public markup parser could not read this content.", "code": "parser_limit"}))


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--parse":
    _worker()
