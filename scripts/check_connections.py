"""Connection regressions; --real-ssh runs sshd with temporary fixture keys/config."""
import argparse
import asyncio
from contextlib import redirect_stdout
import io
import gzip
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
from unittest.mock import patch
from aiohttp import web, ClientSession
from hpc_llm.browser import gateway_app, run_native
from hpc_llm.compute_web import prepare_browser, show_browser
from hpc_llm.connections import ReverseTunnel, local_forward_command, ood_prefix, ssh_destination
from hpc_llm.contracts import ConnectionConfig, InferenceSettings, ModelSpec, ResourceRequest, SiteProfile, AppError
from hpc_llm.lifecycle import SessionManager, atomic_json, read_private_json
from hpc_llm.terminal import hyperlink_text


def rejects(call):
    try:
        call()
    except (ValueError, AppError):
        return
    raise AssertionError('Unsafe input accepted')


def plans():
    options = ConnectionConfig(mode='local-forward', ssh_host='researcher@{node}', jump_host='login.example').model_dump()
    command = local_forward_command(options, 'gpu42.example', 30123)
    assert command[-1] == 'researcher@gpu42.example'
    assert command[command.index('-J') + 1] == 'login.example'
    assert command[command.index('-L') + 1] == '127.0.0.1:8765:127.0.0.1:30123'
    for flag in ('BatchMode=yes', 'StrictHostKeyChecking=yes', 'ExitOnForwardFailure=yes', 'ForwardAgent=no'):
        assert flag in command
    options.update(mode='reverse-ssh', relay_host='researcher@relay02.example')
    reverse = local_forward_command(options, 'gpu42.example', 30123, relay_port=25001)
    assert reverse[-1] == 'researcher@relay02.example'
    assert reverse[reverse.index('-L') + 1].endswith(':25001')
    for host in ('-bad', 'host;command', 'user@host extra', 'host\x1b]8;;bad', ''):
        rejects(lambda: ssh_destination(host))
    for route in ('//bad/{node}/{port}', '/rnode/{node!r}/{port}', '/rnode/{node}/../{port}', '/rnode/{node}/{port}?secret'):
        rejects(lambda: ood_prefix(route, 'gpu42', 30000))
    url = 'https://ood.example/rnode/gpu42/30000/#access=private'
    assert hyperlink_text(url) == 'Open chat\n' + url
    linked = hyperlink_text(url, enabled=True)
    assert '\x1b]8;;' in linked and linked.endswith('\n' + url)
    for bad in ('https://ood.example/\x1b]8;;bad', 'https://ood.example/\ncommand', 'file:///etc/passwd', 'https://ood.example/\x9bcontrol'):
        rejects(lambda: hyperlink_text(bad, enabled=True))


class FixtureBackend:
    latest = None
    def __init__(self, manifest):
        type(self).latest = self
        self.manifest = manifest
        self.token = 'fixture-backend-private'
        self.pid = 123
        self.process = type('Process', (), {'returncode': None})()
        self.runner = None
        self.closed = False

    async def start(self):
        assert self.native_host == '127.0.0.1' and self.native_webui
        async def handle(request):
            if request.headers.get('Authorization') != 'Bearer ' + self.token:
                raise web.HTTPUnauthorized()
            if request.path == '/v1/models':
                return web.json_response({'data': [{'id': 'fixture'}]})
            return web.Response(text='<html><body>Native fixture</body></html>', content_type='text/html')
        app = web.Application()
        app.router.add_route('*', '/{path:.*}', handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        self.endpoint = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
        return self.manifest.capabilities

    async def close(self):
        self.closed = True
        if self.runner:
            await self.runner.cleanup()


async def lifecycle(root, *, mode='local-forward', relay=None, disconnect=False):
    manager = SessionManager(root / ('state-' + mode), root / 'hpc-llm')
    profile = SiteProfile(connection=ConnectionConfig(mode=mode, ood_origin='https://ood.example', relay_host=relay or ''), workspaces=[str(root / 'workspace')])
    (root / 'workspace').mkdir(exist_ok=True)
    manifest = manager.create(ModelSpec(name='Fixture', path='/fixture.gguf'), ResourceRequest(), profile, InferenceSettings())
    manifest.job_id = 'fixture-job'
    prepare_browser(manifest, manager.root)
    folder = Path(manifest.directory)
    handlers = {}
    loop = asyncio.get_running_loop()
    with patch('hpc_llm.backend.Backend', FixtureBackend), patch.object(loop, 'add_signal_handler', side_effect=lambda sig, fn: handlers.update({sig: fn})), patch('hpc_llm.browser.current_job_id', return_value='fixture-job'):
        task = asyncio.create_task(run_native(manifest))
        try:
            async with asyncio.timeout(30):
                while not (folder / 'browser-access.json').exists():
                    if task.done():
                        await task
                    await asyncio.sleep(.02)
            saved = manager.load(manifest.id)
            assert saved.backend_state == 'ready'
            access = read_private_json(folder / 'browser-access.json')
            assert access['mode'] == mode
            with patch.object(manager, 'submit', side_effect=AssertionError('Reconnect submitted job')), redirect_stdout(io.StringIO()) as output:
                show_browser(saved)
                show_browser(saved)
            assert output.getvalue().count('#access=') == 2
            assert not task.done()
            if mode != 'ood':
                assert 'leave that local SSH process running' in output.getvalue()
            if disconnect:
                TunnelFixture.latest.process.returncode = 255
                try:
                    await asyncio.wait_for(task, 5)
                except AppError as exc:
                    assert exc.code == 'network'
                else:
                    raise AssertionError('Tunnel disconnect did not fail the job')
            else:
                handlers[signal.SIGTERM]()
                await asyncio.wait_for(task, 10)
            assert FixtureBackend.latest.closed
            assert manager.load(manifest.id).backend_state == ('failed' if disconnect else 'stopped')
            assert not (folder / 'browser-access.json').exists()
            assert not (folder / 'browser-backend.json').exists()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def native_tools_blocked(root):
    token = root / 'backend.json'
    atomic_json(token, {'token': 'secret'})
    runner = web.AppRunner(gateway_app('http://127.0.0.1:1', token, 'key', cookie_auth=True))
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    try:
        async with ClientSession() as client:
            for path in ('/tools', '/tools/', '/%74ools', '/tools%2f', '/tools/shell', '/%2574ools'):
                async with client.post(url + path) as response:
                    assert response.status in {400, 403}, (path, response.status)
    finally:
        await runner.cleanup()


async def compressed_ui(root):
    """Pinned llama.cpp serves gzip HTML; inject panel on index and SPA reloads."""
    seen = []
    async def upstream(request):
        assert 'gzip' in request.headers.get('Accept-Encoding', '')
        seen.append(request.path)
        body = b'<html><body>Native compressed UI</body></html>'
        if request.path == '/oversized':
            body = b'x' * (16 * 1024 * 1024 + 1)
        if request.path == '/asset.js':
            return web.Response(body=gzip.compress(b'asset fixture'), headers={'Content-Encoding': 'gzip', 'Content-Type': 'text/javascript'})
        return web.Response(body=gzip.compress(body), headers={'Content-Encoding': 'gzip', 'Content-Type': 'text/html', 'ETag': 'old'})
    app = web.Application()
    app.router.add_route('*', '/{path:.*}', upstream)
    backend = web.AppRunner(app)
    await backend.setup()
    site = web.TCPSite(backend, '127.0.0.1', 0)
    await site.start()
    token = root / 'gzip-token.json'
    atomic_json(token, {'token': 'backend-key'})
    class Panel:
        closed = False
        async def handle(self, request, path):
            return None
        def transform_html(self, body, prefix):
            return body.replace(b'</body>', b'<button>Files &amp; tools</button></body>')
        async def close(self):
            self.closed = True
    panel = Panel()
    gateway = web.AppRunner(gateway_app(f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}', token, 'browser-key', cookie_auth=True, workspace=panel))
    await gateway.setup()
    exposed = web.TCPSite(gateway, '127.0.0.1', 0)
    await exposed.start()
    base = f'http://127.0.0.1:{exposed._server.sockets[0].getsockname()[1]}'
    try:
        async with ClientSession(auto_decompress=False) as client:
            async with client.post(base + '/_hpc/login', json={'key': 'browser-key'}) as response:
                cookie = '; '.join(f'{c.key}={c.value}' for c in response.cookies.values())
            for path in ('/', '/index.html', '/chat/fixture'):
                async with client.get(base + path, headers={'Cookie': cookie}) as response:
                    assert response.status == 200
                    assert 'Content-Encoding' not in response.headers and 'ETag' not in response.headers
                    assert b'Files &amp; tools' in await response.read()
            async with client.get(base + '/asset.js', headers={'Cookie': cookie}) as response:
                assert response.headers['Content-Encoding'] == 'gzip'
                assert gzip.decompress(await response.read()) == b'asset fixture'
            async with client.get(base + '/oversized', headers={'Cookie': cookie}) as response:
                assert response.status == 503
        assert seen == ['/', '/index.html', '/chat/fixture', '/asset.js', '/oversized']
    finally:
        await gateway.cleanup()
        await backend.cleanup()
    assert panel.closed


async def reverse_control(root):
    """Deterministic control protocol tests; this does not claim a real SSH path."""
    calls = []
    class Process:
        returncode = None
        def terminate(self):
            self.returncode = 0
        def kill(self):
            self.returncode = -9
        async def wait(self):
            return self.returncode
    class Tunnel(ReverseTunnel):
        fail = False
        async def control(self, *args):
            calls.append(args)
            if '-R' in args:
                if self.fail:
                    raise AppError('network', 'fixture collision')
                return '25001'
            return ''
    async def create(*argv, **kwargs):
        assert 'BatchMode=yes' in argv and 'StrictHostKeyChecking=yes' in argv
        assert '-M' in argv and argv[-1] == 'relay02.example'
        Path(argv[argv.index('-S') + 1]).touch()
        return Process()
    async def probe(url, password):
        assert url.startswith('http://127.0.0.1:') and password == 'browser-key'
    tunnel = Tunnel({'relay_host': 'relay02.example'}, 30123, root / 'reverse.log')
    with patch('asyncio.create_subprocess_exec', create), patch('hpc_llm.connections.authenticated_probe', probe):
        try:
            assert await tunnel.start('browser-key') == 25001
            assert ('-O', 'forward', '-R', '127.0.0.1:0:127.0.0.1:30123') in calls
            assert any(call[:3] == ('-O', 'cancel', '-L') for call in calls)
        finally:
            await tunnel.close()
        assert tunnel.process.returncode == 0 and not Path(tunnel.socket).exists()
        tunnel = Tunnel({'relay_host': 'relay02.example', 'relay_port': 25002}, 30123, root / 'collision.log')
        tunnel.fail = True
        try:
            try:
                await tunnel.start('browser-key')
            except AppError:
                pass
            else:
                raise AssertionError('Collision did not fail')
        finally:
            await tunnel.close()


class TunnelFixture:
    latest = None
    fail = False
    def __init__(self, options, gateway_port, log_path):
        type(self).latest = self
        self.process = type('Process', (), {'returncode': None})()
        self.closed = False
    async def start(self, password):
        if self.fail:
            raise AppError('network', 'fixture tunnel startup failure')
        return 25001
    async def close(self):
        self.closed = True


async def reverse_lifecycle(root):
    with patch('hpc_llm.connections.ReverseTunnel', TunnelFixture):
        await lifecycle(root, mode='reverse-ssh', relay='relay02.example')
        assert TunnelFixture.latest.closed
        await lifecycle(root, mode='reverse-ssh', relay='relay02.example', disconnect=True)
        TunnelFixture.fail = True
        try:
            try:
                await lifecycle(root, mode='reverse-ssh', relay='relay02.example')
            except AppError as exc:
                assert exc.code == 'network'
            else:
                raise AssertionError('Tunnel failure did not fail startup')
            assert TunnelFixture.latest.closed and FixtureBackend.latest.closed
        finally:
            TunnelFixture.fail = False


async def real_ssh(root):
    """Actual forwarding; fixture-only keys never touch user or system SSH config."""
    sshd = shutil.which('sshd')
    if not sshd or not shutil.which('ssh-keygen'):
        raise RuntimeError('--real-ssh requires sshd, ssh and ssh-keygen')
    folder = root / 'ssh-fixture'
    folder.mkdir(mode=0o700)
    for name in ('host', 'client'):
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(folder / name)], check=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        ssh_port = sock.getsockname()[1]
    config = folder / 'sshd_config'
    config.write_text(f'Port {ssh_port}\nListenAddress 127.0.0.1\nHostKey {folder / "host"}\nPidFile {folder / "pid"}\nAuthorizedKeysFile {folder / "client.pub"}\nStrictModes no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nUsePAM no\nAllowTcpForwarding yes\nGatewayPorts no\nLogLevel ERROR\n')
    (folder / 'known_hosts').write_text(f'[127.0.0.1]:{ssh_port} ' + (folder / 'host.pub').read_text())
    (folder / 'config').write_text(f'Host fixture-relay\n HostName 127.0.0.1\n Port {ssh_port}\n IdentityFile {folder / "client"}\n IdentitiesOnly yes\n UserKnownHostsFile {folder / "known_hosts"}\n GlobalKnownHostsFile /dev/null\n')
    bindir = folder / 'bin'
    bindir.mkdir()
    wrapper = bindir / 'ssh'
    wrapper.write_text('#!/bin/sh\nexec /usr/bin/ssh -F ' + shlex.quote(str(folder / 'config')) + ' "$@"\n')
    wrapper.chmod(0o700)
    log = (folder / 'sshd.log').open('wb')
    server = subprocess.Popen([sshd, '-D', '-e', '-f', str(config)], stdout=log, stderr=log)
    try:
        async with asyncio.timeout(10):
            while True:
                if server.poll() is not None:
                    log.flush()
                    raise RuntimeError((folder / 'sshd.log').read_text())
                try:
                    reader, writer = await asyncio.open_connection('127.0.0.1', ssh_port)
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    await asyncio.sleep(.05)
        with patch.dict(os.environ, {'PATH': str(bindir) + os.pathsep + os.environ['PATH']}):
            try:
                await lifecycle(root, mode='reverse-ssh', relay='fixture-relay')
            except Exception:
                for path in (root / 'state-reverse-ssh').glob('sessions/*/logs/reverse-ssh.log'):
                    print(path.read_text())
                log.flush()
                print((folder / 'sshd.log').read_text())
                raise
            with socket.socket() as occupied:
                occupied.bind(('127.0.0.1', 0))
                occupied.listen()
                tunnel = ReverseTunnel({'relay_host': 'fixture-relay', 'relay_port': occupied.getsockname()[1]}, 30000, folder / 'collision.log')
                try:
                    try:
                        await tunnel.start('not-used')
                    except AppError:
                        pass
                    else:
                        raise AssertionError('Remote collision was accepted')
                finally:
                    await tunnel.close()
    finally:
        server.terminate()
        server.wait(timeout=5)
        log.close()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-ssh', action='store_true', help='Create temporary fixture keys and run isolated sshd')
    args = parser.parse_args()
    plans()
    with tempfile.TemporaryDirectory(prefix='hpc-connection-check-') as temporary:
        root = Path(temporary)
        await lifecycle(root)
        await lifecycle(root, mode='ood')
        await native_tools_blocked(root)
        await compressed_ui(root)
        await reverse_control(root)
        await reverse_lifecycle(root)
        if args.real_ssh:
            await real_ssh(root)
    print('Connections passed: plans/hyperlinks, OOD/local lifecycle, cookie/API readiness, gzip HTML injection, asset passthrough, reconnect, native-tool blocking, simulated reverse allocation/collision/disconnect and cleanup' + ('; real OpenSSH reverse roundtrip and collision.' if args.real_ssh else '; real SSH not requested.'))


if __name__ == '__main__':
    asyncio.run(main())
