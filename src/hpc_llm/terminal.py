"""Preserve the calling terminal when a scheduler attach step exits."""
from __future__ import annotations

import subprocess
import sys
import time
import os
from urllib.parse import urlsplit


def hyperlink_text(url: str, *, enabled: bool = False) -> str:
    """Only our fixed OSC 8 controls bypass clean_text; URL text is validated.

    The complete URL is always printed as a fallback, including the private
    fragment. Never pass arbitrary provider text or filenames through this API.
    """
    if not isinstance(url, str) or any(c.isspace() or ord(c) < 33 or 127 <= ord(c) <= 159 for c in url):
        raise ValueError('Browser URL contains terminal controls or whitespace.')
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Browser URL must be an HTTP(S) address without credentials in its authority.')
    parsed.port  # Reject invalid port text before emitting a terminal sequence.
    label = f'\x1b]8;;{url}\x1b\\Open chat\x1b]8;;\x1b\\' if enabled else 'Open chat'
    return label + '\n' + url


def print_browser_link(url: str, *, stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    preference = os.environ.get('HPC_LLM_HYPERLINKS', 'auto').lower()
    enabled = preference == 'on' or (preference != 'off' and stream.isatty() and os.environ.get('TERM', '') != 'dumb')
    # Intentionally bypass cli.display/clean_text only after strict URL validation;
    # sending OSC 8 through the general untrusted-text sanitizer would remove it.
    print(hyperlink_text(url, enabled=enabled), file=stream, flush=True)


def _wait_attached(argv: list[str], env: dict[str, str]) -> int:
    process = subprocess.Popen(argv, env=env)
    try:
        return process.wait()
    except BaseException:
        # Popen's context manager can skip its final wait after Ctrl+C. Reap
        # explicitly before another screen or the shell takes this terminal.
        deadline = time.monotonic() + 5
        while True:
            try:
                process.kill()
                process.wait(timeout=max(0, deadline - time.monotonic()))
                break
            except KeyboardInterrupt:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(argv, 5)
        raise


def run_attached(argv: list[str], *, env: dict[str, str]) -> int:
    """Wait for the attach client and restore tty modes even after interruption.

    Kill and reap only the attach client on an exception. Never signal a
    process group: it can include the user's shell or allocation owner.
    Textual owns display escape sequences; this guard only restores tty modes.
    """
    try:
        import termios
    except ImportError:
        return _wait_attached(argv, env)

    fd = None
    attrs = None
    try:
        fd = sys.stdin.fileno()
        attrs = termios.tcgetattr(fd)
    except (AttributeError, OSError, ValueError, termios.error):
        pass  # Redirected stdin and headless invocations have no tty to restore.
    try:
        return _wait_attached(argv, env)
    finally:
        if fd is not None and attrs is not None:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
            except (OSError, termios.error):
                pass  # The terminal may have disconnected during attach.
