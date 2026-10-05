"""Authenticated compute-local API client; no inference stream lives in this client."""
from __future__ import annotations
import ipaddress
import os
import stat
from pathlib import Path
from urllib.parse import urlsplit
import aiohttp
from .contracts import AppError, SessionManifest


class SessionClient:
    def __init__(self, manifest: SessionManifest):
        self.manifest = manifest
        if not manifest.endpoint:
            raise AppError("startup", "This session has no running service endpoint. Return to the launcher; resume a running session or restore saved chats from an ended session.")
        endpoint = urlsplit(manifest.endpoint)
        try:
            local = ipaddress.ip_address(endpoint.hostname or "").is_loopback
        except ValueError:
            local = False
        if not local or endpoint.scheme != "http" or not endpoint.port or endpoint.username or endpoint.query or endpoint.fragment:
            raise AppError("auth", "Session endpoint is not a valid compute-local service. Reconnect through the launcher.")
        self.endpoint = manifest.endpoint.rstrip("/")
        key = Path(manifest.directory) / "supervisor.key"
        try:
            fd = os.open(key, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                    raise AppError("auth", "Session credentials must be private and owned by you.")
                self.token = os.read(fd, 1024).decode().strip()
            finally:
                os.close(fd)
        except OSError as exc:
            raise AppError("auth", "Cannot read private session credentials; inspect session permissions.") from exc
        if not self.token:
            raise AppError("auth", "Session credentials are empty.")
        self._http: aiohttp.ClientSession | None = None
        self._verified = False

    async def _session(self):
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=660, connect=5), headers={"Authorization": f"Bearer {self.token}", "X-Session-Nonce": self.manifest.nonce})
        return self._http

    async def _request(self, method, path, json=None, params=None):
        if not path.startswith("/") or path.startswith("//"):
            raise AppError("validation", "Invalid local API path")
        try:
            session = await self._session()
            async with session.request(method, self.endpoint + path, json=json, params=params, allow_redirects=False) as response:
                try:
                    result = await response.json()
                except (ValueError, aiohttp.ContentTypeError) as exc:
                    raise AppError("backend", "Session returned an invalid response.") from exc
                if response.status >= 300:
                    error = result.get("error", result)
                    if isinstance(error, dict):
                        raise AppError(error.get("code", "backend"), error.get("message", "Session request failed"))
                    raise AppError(result.get("code", "backend"), str(error))
                return result
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise AppError("network", "The compute session is unreachable. Your saved chat remains available; reconnect or check session status.") from exc

    async def request(self, method, path, json=None, params=None):
        if not self._verified:
            identity = await self._request("GET", "/identity")
            if identity.get("session_id") != self.manifest.id or identity.get("nonce") != self.manifest.nonce:
                raise AppError("auth", "This endpoint belongs to a different session. No request was sent.")
            self._verified = True
        return await self._request(method, path, json=json, params=params)

    async def state(self):
        return await self.request("GET", "/state")

    async def close(self):
        if self._http:
            await self._http.close()
