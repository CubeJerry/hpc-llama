"""Offline regression checks for embedded MTP and native job transports.

Uses a real local HTTP readiness fixture; GPU and SSH execution are stubbed.
Run: PYTHONPATH=src python scripts/check_mtp_browser.py
"""
import asyncio
import getpass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from hpc_llm.browser import run_native
from hpc_llm.contracts import AppError, BackendCapabilities, InferenceSettings, ModelSpec, SessionManifest
from hpc_llm.lifecycle import atomic_json, read_private_json
from hpc_llm.runtime import acceleration_command


def check_mtp():
    caps = BackendCapabilities(mtp_supported=True, flags=['--spec-type', '--spec-draft-model'])
    model = ModelSpec(name='Qwen', path='/models/Qwen-Q4_K_P.gguf', mtp_layers=1,
                      mtp_path='/models/Qwen-FastMTP-32K.gguf')
    for mode in ('auto', 'embedded'):
        argv = acceleration_command(caps, model, InferenceSettings(acceleration=mode))
        assert argv == ['--spec-type', 'draft-mtp'], argv
        assert caps.acceleration_status == 'Requested'
    argv = acceleration_command(caps, model, InferenceSettings(acceleration='mtp'))
    assert argv == ['--spec-type', 'draft-mtp', '--spec-draft-model', model.mtp_path]
    assert acceleration_command(caps, model, InferenceSettings(acceleration='off')) == []
    model.mtp_layers = 0
    assert acceleration_command(caps, model, InferenceSettings(acceleration='auto')) == []
    try:
        acceleration_command(caps, model, InferenceSettings(acceleration='embedded'))
    except AppError as exc:
        assert exc.code == 'unsupported'
    else:
        raise AssertionError('Explicit embedded mode accepted a model without embedded MTP')
    model.mtp_path = '/models/Qwen-MTP.gguf'
    assert '--spec-draft-model' in acceleration_command(caps, model, InferenceSettings(acceleration='auto'))
    caps.mtp_supported = False
    assert acceleration_command(caps, model, InferenceSettings(acceleration='auto')) == []


async def check_transport(transport, fail=False):
    with TemporaryDirectory() as temporary:
        folder = Path(temporary) / 'session'
        (folder / 'logs').mkdir(parents=True)
        manifest = SessionManifest(owner=getpass.getuser(), directory=str(folder),
                                   model=ModelSpec(name='fixture', path='/models/fixture.gguf'))
        atomic_json(folder / 'browser.json', {'transport': 'compute', 'connection': {
            'mode': transport, 'ood_origin': 'https://ood.example', 'relay_host': 'relay02.example'}})
        calls, closed, observed = [], [], []
        class BackendFixture:
            def __init__(self, session):
                self.token = 'backend-test-token'
                self.pid = 12345
                self.process = SimpleNamespace(returncode=1 if fail else None)
            async def start(self):
                async def home(request):
                    assert request.headers['Authorization'] == 'Bearer backend-test-token'
                    return web.Response(text='<html>Native UI fixture</html>', content_type='text/html')
                app = web.Application()
                app.router.add_get('/', home)
                app.router.add_get('/v1/models', home)
                self.runner = web.AppRunner(app)
                await self.runner.setup()
                site = web.TCPSite(self.runner, '127.0.0.1', 0)
                await site.start()
                self.endpoint = 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])
                observed.append((self.native_webui, self.native_host))
                return BackendCapabilities()
            async def close(self):
                observed.append(read_private_json(folder / 'browser-backend.json'))
                await self.runner.cleanup()
                closed.append(True)
        class ProcessFixture:
            returncode = None
            async def wait(self):
                self.returncode = 0
                return 0
            def terminate(self):
                self.returncode = 0
            def kill(self):
                self.returncode = -9
        async def subprocess_fixture(*argv, **kwargs):
            raise AssertionError('Unexpected direct subprocess; reverse tunnel fixture owns its process')
        class ReverseFixture:
            def __init__(self, options, port, log_path):
                self.process = ProcessFixture()
                calls.append((options, port))
            async def start(self, password):
                return 25000
            async def close(self):
                self.process.terminate()
        loop = asyncio.get_running_loop()
        timers = []
        def signal_fixture(sig, callback):
            timers.append(loop.call_later(0.15, callback))
        try:
            with patch('hpc_llm.backend.Backend', BackendFixture), \
                 patch('hpc_llm.browser.asyncio.create_subprocess_exec', subprocess_fixture), \
                 patch('hpc_llm.connections.ReverseTunnel', ReverseFixture), \
                 patch.object(loop, 'add_signal_handler', signal_fixture):
                try:
                    await run_native(manifest)
                except AppError as exc:
                    assert fail and exc.code == 'startup'
                else:
                    assert not fail
        finally:
            for timer in timers:
                timer.cancel()
        assert closed == [True]
        assert observed[0] == (True, '127.0.0.1')
        assert not (folder / 'browser-backend.json').exists(), 'Backend credential was not cleaned up'
        assert read_private_json(folder / 'manifest.json')['backend_state'] == ('failed' if fail else 'stopped')
        assert observed[1]['endpoint'].startswith('http://127.0.0.1:')
        if transport != 'reverse-ssh':
            assert not calls, 'OOD/local transport unexpectedly invoked compute SSH'
        else:
            assert len(calls) == 1 and calls[0][0]['relay_host'] == 'relay02.example'


async def main():
    check_mtp()
    await check_transport('ood')
    await check_transport('local-forward')
    await check_transport('reverse-ssh')
    await check_transport('ood', fail=True)
    print('MTP selection, OOD/local/reverse transport lifecycle, readiness, failure and credential cleanup passed.')


if __name__ == '__main__':
    asyncio.run(main())
