"""Preserve the calling terminal when a scheduler attach step exits."""
from __future__ import annotations

import subprocess
import sys
import time


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
