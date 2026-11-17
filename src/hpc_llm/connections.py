"""Portable browser connection plans and compute-owned reverse SSH lifecycle.

No SSH configuration, keys or known_hosts files are changed by this module.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import re
import shlex
import socket
import tempfile
import time
from urllib.parse import urlsplit

import aiohttp

from .contracts import AppError


def ssh_destination(value: str, *, node: str | None = None) -> str:
    if node is not None:
        value = value.replace('{node}', node)
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('SSH destination must be a hostname or SSH alias, optionally user@host; no options or shell syntax.')
    return value


def port_number(value: int, *, dynamic: bool = False) -> int:
    value = int(value)
    if not (dynamic and value == 0) and not 1024 <= value <= 65535:
        raise ValueError('Ports must be between 1024 and 65535 (relay_port may be 0 for automatic allocation).')
    return value


def validate_origin(value: str) -> str:
    value = value.strip().rstrip('/')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path or any(ord(c) < 33 or ord(c) == 127 for c in value)):
        raise ValueError('Enter your Open OnDemand HTTPS address without a path, for example https://ondemand.example.edu')
    parsed.port  # Validate port syntax before allocating a job.
    return value


def ood_prefix(template: str, node: str, port: int) -> str:
    ssh_destination(node)
    port_number(port)
    # Only these substitutions are accepted, never Python format expressions.
    if template.replace('{node}', '').replace('{port}', '').find('{') >= 0 or '}' in template.replace('{node}', '').replace('{port}', ''):
        raise ValueError('OOD path permits only {node} and {port} placeholders.')
    if '{node}' not in template or '{port}' not in template:
        raise ValueError('OOD path must contain both {node} and {port}.')
    value = template.replace('{node}', node).replace('{port}', str(port)).rstrip('/')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', value) or '//' in value or any(p in {'.', '..'} for p in value.split('/')):
        raise ValueError('OOD path must be an absolute URL path without traversal, queries or terminal controls.')
    return value


def connection_options(profile) -> dict:
    connection = getattr(profile, 'connection', None)
    return connection.model_dump() if connection is not None else {'mode': 'ood'}


def ssh_base(host: str, jump_host: str = '') -> list[str]:
    host = ssh_destination(host)
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            '-o', 'ForwardAgent=no']
    if jump_host:
        argv += ['-J', ssh_destination(jump_host)]
    return argv + [host]


def local_forward_command(options: dict, node: str, service_port: int, *, relay_port: int | None = None) -> list[str]:
    local_port = port_number(options.get('local_port', 8765))
    mode = options.get('mode', 'ood')
    if mode == 'reverse-ssh':
        host = ssh_destination(options.get('relay_host', ''))
        target_port = port_number(relay_port)
    elif mode == 'local-forward':
        host = ssh_destination(options.get('ssh_host') or '{node}', node=node)
        target_port = port_number(service_port)
    else:
        raise ValueError('A local tunnel is only needed for SSH connection modes.')
    argv = ssh_base(host, options.get('jump_host', ''))
    return argv[:-1] + ['-N', '-L', f'127.0.0.1:{local_port}:127.0.0.1:{target_port}', argv[-1]]


async def authenticated_probe(url: str, password: str, *, prefix: str = '', origin: str | None = None) -> None:
    """Verify cookie login and the backend API, not just an SSH listening socket."""
    async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as client:
        headers = {'Origin': origin} if origin else {}
        async with client.post(url + prefix + '/_hpc/login', json={'key': password}, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                raise AppError('network', 'Browser gateway sign-in readiness check failed.')
            # Secure OOD cookies cannot be sent over this compute-local HTTP probe.
            cookie = '; '.join(f'{item.key}={item.value}' for item in response.cookies.values())
        async with client.get(url + prefix + '/v1/models', headers={**headers, 'Cookie': cookie},
                              timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                raise AppError('network', 'Browser gateway model API readiness check failed.')
            await response.read()


class ReverseTunnel:
    """One compute-owned master; dynamically allocate a loopback relay port."""
    def __init__(self, options: dict, gateway_port: int, log_path: Path):
        self.options = options
        self.gateway_port = port_number(gateway_port)
        self.log_path = log_path
        self.process = None
        self._temp = None
        self._log = None
        self.relay_port = None

    async def control(self, *args: str) -> str:
        base = ssh_base(self.options['relay_host'], self.options.get('relay_jump_host', ''))
        argv = base[:-1] + ['-S', self.socket, *args, base[-1]]
        process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode:
            if stderr and self._log:
                self._log.write(stderr)
                self._log.flush()
            raise AppError('network', 'SSH forwarding failed. Inspect logs/reverse-ssh.log; check relay reachability, known_hosts, batch authentication, forwarding permissions and port conflicts.')
        return stdout.decode().strip()

    async def start(self, password: str) -> int:
        self._temp = tempfile.TemporaryDirectory(prefix='hpc-ssh-')
        self.socket = str(Path(self._temp.name) / 'ctl')
        self._log = self.log_path.open('ab')
        base = ssh_base(self.options['relay_host'], self.options.get('relay_jump_host', ''))
        argv = base[:-1] + ['-N', '-M', '-S', self.socket, '-o', 'ControlPersist=no', base[-1]]
        self.process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
                                                          stdout=self._log, stderr=self._log)
        deadline = time.monotonic() + 20
        while not Path(self.socket).exists():
            if self.process.returncode is not None or time.monotonic() >= deadline:
                raise AppError('network', 'Compute-to-relay batch SSH did not start. Inspect logs/reverse-ssh.log. Use OOD or local-forward if compute nodes cannot SSH to this relay.')
            await asyncio.sleep(.1)
        await self.control('-O', 'check')
        requested = port_number(self.options.get('relay_port', 0), dynamic=True)
        assigned = await self.control('-O', 'forward', '-R', f'127.0.0.1:{requested}:127.0.0.1:{self.gateway_port}')
        self.relay_port = port_number(int(assigned) if requested == 0 else requested)
        # Verify the complete compute -> relay -> authenticated compute gateway path.
        for attempt in range(5):
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                probe_port = probe.getsockname()[1]
            spec = f'127.0.0.1:{probe_port}:127.0.0.1:{self.relay_port}'
            try:
                await self.control('-O', 'forward', '-L', spec)
                break
            except AppError:
                if attempt == 4:
                    raise
        try:
            await authenticated_probe(f'http://127.0.0.1:{probe_port}', password)
        finally:
            await self.control('-O', 'cancel', '-L', spec)
        return self.relay_port

    async def close(self):
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self._log:
            self._log.close()
        if self._temp:
            self._temp.cleanup()


def tunnel_text(options: dict, node: str, port: int, relay_port: int | None = None) -> str:
    return shlex.join(local_forward_command(options, node, port, relay_port=relay_port))
