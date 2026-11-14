"""Offline integration: real HTTP, simulated GPU and scheduler, headless Textual."""
import asyncio
from contextlib import redirect_stdout
import io
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import BasicAuth, ClientSession, web
from textual.widgets import Button, Input, Select, Static

from hpc_llm.browser import run_native
from hpc_llm.cli import attach, parse_args, launcher
from hpc_llm.compute_web import prepare_browser, saved_origin, validate_origin
from hpc_llm.contracts import AppError, InferenceSettings, ModelSpec, ResourceRequest, SiteProfile
from hpc_llm.lifecycle import SessionManager, read_private_json
from hpc_llm.ui import LauncherApp


class FakeBackend:
    latest = None
    fail = False
    block = False

    def __init__(self, manifest):
        type(self).latest = self
        self.manifest = manifest
        self.process = SimpleNamespace(returncode=None)
        self.pid = 123
        self.token = 'private-backend-key'
        self.runner = None
        self.closed = False

    async def start(self):
        assert self.native_webui and self.native_host == '127.0.0.1'
        if self.block:
            await asyncio.Event().wait()
        if self.fail:
            raise AppError('startup', 'fixture loading failure')
        async def handle(request):
            if request.headers.get('Authorization') != 'Bearer ' + self.token:
                raise web.HTTPUnauthorized()
            if request.path == '/v1/chat/completions':
                response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                await response.prepare(request)
                await response.write(b'data: fixture\n\n')
                await response.write(b'data: [DONE]\n\n')
                return response
            return web.Response(text='<html>native UI fixture</html>', content_type='text/html')
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


async def lifecycle_check(root):
    manager = SessionManager(root / 'state', root / 'hpc-llm')
    manifest = manager.create(ModelSpec(id='fixture', name='Fixture', path='/fixture.gguf'),
                              ResourceRequest(), SiteProfile(), InferenceSettings())
    manifest.job_id = '12345'
    prepare_browser(manifest, manager.root, 'https://ood.example.edu/')
    assert saved_origin(manager.root, root) == 'https://ood.example.edu'
    folder = Path(manifest.directory)
    handlers = {}
    loop = asyncio.get_running_loop()
    with patch('hpc_llm.backend.Backend', FakeBackend), \
         patch.object(loop, 'add_signal_handler', side_effect=lambda sig, fn: handlers.update({sig: fn})), \
         patch('hpc_llm.browser.socket.getfqdn', return_value='gpu01.example.edu'), \
         patch('hpc_llm.browser.current_job_id', return_value='12345'), \
         patch('asyncio.create_subprocess_exec', side_effect=AssertionError('No SSH or extra job allowed')):
        task = asyncio.create_task(run_native(manifest))
        try:
            async with asyncio.timeout(10):
                while not (folder / 'browser-access.json').exists():
                    if task.done():
                        await task
                    await asyncio.sleep(.01)
            access = read_private_json(folder / 'browser-access.json')
            assert access['url'] == f'https://ood.example.edu/rnode/gpu01.example.edu/{access["port"]}/'
            password = (folder / 'browser.password').read_text().strip()
            base = f'http://127.0.0.1:{access["port"]}'
            async with ClientSession() as client:
                async with client.get(base) as response:
                    assert response.status == 401
                auth = BasicAuth('hpc', password)
                async with client.get(base + f'/rnode/gpu01.example.edu/{access["port"]}/', auth=auth) as response:
                    assert response.status == 200 and 'native UI' in await response.text()
                async with client.post(base + '/v1/chat/completions', auth=auth,
                                       headers={'Origin': 'https://ood.example.edu'}, json={}) as response:
                    assert response.status == 200 and '[DONE]' in await response.text()
                async with client.post(base + '/v1/chat/completions', auth=auth,
                                       headers={'Origin': 'https://other.example.edu'}, json={}) as response:
                    assert response.status == 403
            with redirect_stdout(io.StringIO()) as output:
                assert attach(manager, manager.load(manifest.id)) == 0
            assert access['url'] in output.getvalue() and 'close this terminal' in output.getvalue()
            assert not task.done()  # attach returned; batch-owned server remains alive.
            handlers[signal.SIGTERM]()
            await asyncio.wait_for(task, 5)
            assert FakeBackend.latest.closed
            assert manager.load(manifest.id).backend_state == 'stopped'
            assert not (folder / 'browser-access.json').exists()
            assert not (folder / 'browser-backend.json').exists()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        FakeBackend.fail = True
        try:
            await run_native(manifest)
        except AppError:
            pass
        else:
            raise AssertionError('Loading failure was swallowed')
        finally:
            FakeBackend.fail = False
        assert manager.load(manifest.id).backend_state == 'failed'
        assert not (folder / 'browser-access.json').exists()
        assert FakeBackend.latest.closed

        FakeBackend.block = True
        try:
            task = asyncio.create_task(run_native(manifest))
            await asyncio.sleep(.05)
            handlers[signal.SIGTERM]()
            await asyncio.wait_for(task, 2)
            assert FakeBackend.latest.closed
            assert manager.load(manifest.id).backend_state == 'stopped'
        finally:
            FakeBackend.block = False


async def ui_check():
    model = ModelSpec(id='fixture', name='Fixture', path='/fixture.gguf',
                     defaults={'context': 8192, 'acceleration': 'auto'})
    app = LauncherApp([model], [], SiteProfile(), ood_origin='https://ood.example.edu')
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        assert app.query_one('#launch-context', Input).value == '8192'
        app.query_one('#launch-ood-origin', Input).value = 'not-an-origin'
        await pilot.click('#start')
        await pilot.pause()
        assert 'HTTPS' in str(app.query_one('#launch-error', Static).render())
        app.query_one('#launch-ood-origin', Input).value = 'https://ood.example.edu'
        app.query_one('#launch-context', Input).value = '16384'
        app.query_one('#launch-acceleration', Select).value = 'off'
        await pilot.click('#start')
    result = app.return_value
    assert result['chat_mode'] == 'browser' and result['settings']['context'] == 16384
    assert result['settings']['acceleration'] == 'off'
    app = LauncherApp([model], [], SiteProfile())
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        app.query_one('#launch-chat-mode', Select).value = 'terminal'
        await pilot.pause()
        assert app.query_one('#launch-ood-origin', Input).disabled
        await pilot.click('#start')
    assert app.return_value['chat_mode'] == 'terminal'


def launcher_check(root):
    manager = SessionManager(root / 'launcher-state', root / 'hpc-llm')
    profile = SiteProfile()
    model = ModelSpec(id='test', name='Fixture', path='/fixture.gguf')
    choice = {'action': 'start', 'model_id': model.id, 'chat_mode': 'browser',
              'ood_origin': 'https://ood.example.edu', 'resources': profile.resources.model_dump()}
    def submit(manifest):
        assert read_private_json(Path(manifest.directory) / 'browser.json')['transport'] == 'compute'
        manifest.job_id = '12345'
        return manifest
    with patch('hpc_llm.cli.profile_for', return_value=profile), \
         patch('hpc_llm.cli._library', return_value=SimpleNamespace(list=lambda: [model])), \
         patch.object(manager, 'inventory', return_value={}), \
         patch.object(LauncherApp, 'run', return_value=choice), \
         patch.object(manager, 'submit', side_effect=submit) as submission, \
         patch('hpc_llm.cli._wait_ready', side_effect=lambda manager, manifest: manifest), \
         patch('hpc_llm.cli.attach', return_value=0) as handoff, redirect_stdout(io.StringIO()):
        assert launcher(manager, manager.root) == 0
        submission.assert_called_once()
        handoff.assert_called_once()


async def main():
    for bad in ('http://ood.example.edu', 'https://ood.example.edu/path',
                'https://user:pass@ood.example.edu', 'https://ood.example.edu?x=1', ''):
        try:
            validate_origin(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid origin: {bad}')
    assert parse_args(['launch', '--ood-url', 'https://ood.example.edu']).command == 'launch'
    with tempfile.TemporaryDirectory() as folder:
        await lifecycle_check(Path(folder))
        launcher_check(Path(folder))
    await ui_check()
    print('Compute browser: lifecycle, private backend, auth, prefix, SSE, origin, failure/stop cleanup, reattach and TUI launch passed.')


if __name__ == '__main__':
    asyncio.run(main())
