"""Offline HTTP integration checks for the browser launcher and confined file IO."""
import asyncio
from pathlib import Path
import socket
import tempfile
import threading
import time
from unittest.mock import patch

from aiohttp import web, ClientSession, BasicAuth
from hpc_llm.contracts import ModelSpec, ResourceRequest, InferenceSettings, SiteProfile
from hpc_llm.dashboard import dashboard_app, Workspace
from hpc_llm.lifecycle import SessionManager, atomic_json, read_private_json


async def serve(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


async def check():
    seen = []
    async def upstream(request):
        assert request.headers.get('Authorization') == 'Bearer backend-private-key'
        assert 'Cookie' not in request.headers
        if request.method == 'POST':
            data = await request.json()
            seen.append(data)
            if not data.get('stream', False):
                return web.json_response({'choices': [{'message': {'content': 'generated file text'}, 'finish_reason': 'stop'}]})
            out = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await out.prepare(request)
            await out.write(b'data: first\n\n')
            await asyncio.sleep(.01)
            await out.write(b'data: [DONE]\n\n')
            return out
        return web.Response(text='<html>native chat</html>', content_type='text/html')
    app = web.Application()
    app.router.add_route('*', '/{path:.*}', upstream)
    upstream_runner, upstream_port = await serve(app)
    try:
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            state = base / 'state'
            work = base / 'work'; work.mkdir()
            (work / 'input.txt').write_text('user-selected data')
            (work / 'secret-link').symlink_to(state)
            (work / 'escape').symlink_to(base)
            manager = SessionManager(state, base / 'launcher')
            model = ModelSpec(name='Fixture', path=str(work / 'model.gguf'))
            manifest = manager.create(model, ResourceRequest(), SiteProfile(), InferenceSettings())
            manifest.backend_state = 'ready'
            manifest.node = '127.0.0.1'
            manifest.scheduler_state = 'RUNNING'
            atomic_json(Path(manifest.directory) / 'manifest.json', manifest.model_dump())
            atomic_json(Path(manifest.directory) / 'browser.json', {'transport': 'direct', 'gateway_host': socket.getfqdn(), 'tunnel_port': upstream_port})
            atomic_json(Path(manifest.directory) / 'browser-backend.json', {'token': 'backend-private-key', 'endpoint': f'http://127.0.0.1:{upstream_port}'})
            gateway = dashboard_app(manager, state, 'browser-private-password', prefix='/rnode/login/8765', public_origin='https://ondemand.example.edu', workspace=work)
            runner, port = await serve(gateway)
            url = f'http://127.0.0.1:{port}/rnode/login/8765'
            try:
                async with ClientSession() as client:
                    async with client.get(url + '/api/files') as r: assert r.status == 401
                    auth = BasicAuth('hpc', 'browser-private-password')
                    headers = {'X-HPC-Request': 'dashboard', 'Origin': 'https://ondemand.example.edu'}
                    async def post(path, body, custom=headers):
                        async with client.post(url + path, auth=auth, json=body, headers=custom) as r:
                            return r.status, await r.json()
                    status, _ = await post('/api/files/write', {'path': 'new.txt', 'content': 'hello'}, {})
                    assert status == 403
                    status, _ = await post('/api/files/write', {'path': 'new.txt', 'content': 'hello'}, {**headers, 'Origin': 'https://evil.example'})
                    assert status == 403
                    status, _ = await post('/api/files/write', {'path': 'new.txt', 'content': 'hello'})
                    assert status == 200
                    status, _ = await post('/api/files/write', {'path': 'new.txt', 'content': 'overwrite'})
                    assert status == 400 and (work / 'new.txt').read_text() == 'hello'
                    status, _ = await post('/api/files/write', {'path': 'new.txt', 'content': 'overwrite', 'overwrite': True})
                    assert status == 200 and (work / 'new.txt').read_text() == 'overwrite'
                    for path in ('../outside.txt', str(state / 'browser.password'), 'escape/outside.txt', '.hidden', 'bad.key'):
                        status, _ = await post('/api/files/write', {'path': path, 'content': 'not allowed'})
                        assert status == 400, path
                    status, _ = await post('/api/files/mkdir', {'path': 'newdir'})
                    assert status == 200 and (work / 'newdir').is_dir()
                    async with client.get(url + '/api/files', auth=auth) as r:
                        listing = await r.json()
                        assert r.status == 200 and 'escape' not in [e['name'] for e in listing['entries']]
                    async with client.get(url + '/api/files/read', params={'path': 'input.txt'}, auth=auth) as r:
                        assert (await r.json())['content'] == 'user-selected data'
                    async with client.post(url + '/api/files/upload?path=uploaded.bin', data=b'\x00\x01test', auth=auth, headers=headers) as r:
                        assert r.status == 200
                    async with client.get(url + '/api/files/download?path=uploaded.bin', auth=auth) as r:
                        assert await r.read() == b'\x00\x01test'
                    async with client.get(url + '/api/sessions', auth=auth) as r:
                        sessions = (await r.json())['sessions']
                        assert sessions[0]['is_web'] and sessions[0]['can_open']
                    async with client.get(url + '/chat/' + manifest.id + '/', auth=auth) as r:
                        assert r.status == 200 and 'native chat' in await r.text()
                    async with client.post(url + '/chat/' + manifest.id + '/v1/chat/completions', auth=auth, headers={'Cookie': 'secret=never-forward'}, json={'stream': True}) as r:
                        assert r.status == 200
                        assert await r.content.readline() == b'data: first\n'
                        assert '[DONE]' in await r.text()
                    status, data = await post('/api/files/generate', {'session_id': manifest.id, 'prompt': 'write a file', 'paths': ['input.txt']})
                    assert status == 200 and data['content'] == 'generated file text', data
                    assert 'user-selected data' in seen[-1]['messages'][-1]['content']
                    assert not (work / 'generated.txt').exists()
                    status, _ = await post('/api/files/generate', {'session_id': manifest.id, 'prompt': 'write a file', 'paths': ['../outside']})
                    assert status == 400
                    # Public state never serves a token-bearing private document.
                    async with client.get(url + '/api/files/read', params={'path': str(state / 'sessions' / manifest.id / 'browser-backend.json')}, auth=auth) as r:
                        assert r.status == 400
                    # If workspace encompasses the installation, code remains inaccessible.
                    guarded = Workspace(base, state, base)
                    (base / 'src').mkdir()
                    try: guarded.write('src/injected.py', b'not allowed')
                    except ValueError: pass
                    else: raise AssertionError('Application source was writable')
            finally:
                await runner.cleanup()
    finally:
        await upstream_runner.cleanup()
    print('Dashboard: auth, CSRF, prefix, confined files, explicit overwrite, upload/download, native SSE proxy, session metadata and generation preview passed.')




async def management_check():
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        work = base / 'work'; work.mkdir()
        manager = SessionManager(base / 'state', base / 'launcher')
        model = ModelSpec(name='Fixture', path=str(work / 'model.gguf'))
        profile = SiteProfile(model_cache=str(base / 'cache'))
        released = threading.Event()
        stopped = []
        defaults = []
        installs = []
        class Library:
            def list(self): return [model]
            def _target(self, key):
                if key != model.id: raise ValueError('Unknown model')
                return model
            def register(self, path, **kwargs):
                assert path == model.path
                return model
            def save_defaults(self, mid, value):
                assert mid == model.id
                defaults.append(InferenceSettings.model_validate(value))
            def list_install_choices(self, source, revision):
                return {'models': [{'filename': 'main.gguf'}], 'projectors': [], 'mtp_heads': [], 'revision': 'a'*40}
            def plan_install(self, **kw):
                installs.append(kw)
                return {'filename': 'main.gguf', 'revision': 'a'*40}
            def execute_install(self, plan, cancel, progress):
                progress(10, 20)
                while not released.wait(.01):
                    if cancel.is_set(): raise ValueError('cancelled')
                return model
        class Profiles:
            def __init__(self, *args): pass
            def load(self, name=None): return profile
            def list(self): return [profile]
            def select(self, name):
                assert name == profile.name
                return profile
            def set_cache(self, path): profile.model_cache = path
        def submit(manifest):
            manifest.job_id = '123'
            manifest.scheduler_state = 'PENDING'
            atomic_json(Path(manifest.directory) / 'manifest.json', manifest.model_dump())
            (Path(manifest.directory) / 'logs' / 'allocation-123.log').write_text('scheduler startup output')
            return manifest
        manager.submit = submit
        manager.status = lambda manifest: manager.load(manifest.id)
        manager.stop = lambda manifest: stopped.append(manifest.id)
        with patch('hpc_llm.cli._library', lambda root: Library()), patch('hpc_llm.cli.profile_for', lambda root: profile), patch('hpc_llm.profiles.ProfileStore', Profiles):
            app = dashboard_app(manager, base / 'state', 'browser-private-password', workspace=work)
        runner, port = await serve(app)
        url = f'http://127.0.0.1:{port}'
        try:
            async with ClientSession(auth=BasicAuth('hpc', 'browser-private-password'), headers={'X-HPC-Request': 'dashboard'}) as client:
                async def post(path, data):
                    async with client.post(url + path, json=data) as r:
                        return r.status, await r.json()
                async with client.get(url + '/api/state') as r:
                    state = await r.json()
                    assert state['models'][0]['id'] == model.id
                status, session = await post('/api/sessions', {'model': model.id, 'settings': {'context': 4096}, 'resources': {'memory_gb': 32}})
                assert status == 200 and session['resources']['memory_gb'] == 32 and session['settings']['context'] == 4096, session
                assert 'nonce' not in session and session['is_web']
                config = read_private_json(Path(manager.load(session['id']).directory) / 'browser.json')
                assert config['transport'] == 'direct'
                async with client.get(url + '/api/sessions/' + session['id'] + '/logs') as r:
                    assert (await r.json())['text'] == 'scheduler startup output'
                async with client.get(url + '/api/models/choices?source=owner/repo') as r:
                    assert (await r.json())['models'][0]['filename'] == 'main.gguf'
                status, job = await post('/api/models/install', {'source': 'owner/repo', 'quant': 'Q4_K_P', 'mtp': 'auto'})
                assert status == 202
                for _ in range(100):
                    async with client.get(url + '/api/jobs') as r:
                        jobs = (await r.json())['jobs']
                    if jobs[0]['downloaded_bytes'] == 10: break
                    await asyncio.sleep(.01)
                else: raise AssertionError('Download progress did not reach browser')
                # Long downloads do not delay GPU stop or file saving.
                status, _ = await asyncio.wait_for(post('/api/sessions/' + session['id'] + '/stop', {}), 1)
                assert status == 200 and stopped == [session['id']]
                status, _ = await asyncio.wait_for(post('/api/files/write', {'path': 'during-download.txt', 'content': 'saved'}), 1)
                assert status == 200
                status, _ = await post('/api/models/register', {'path': model.path})
                assert status == 409
                released.set()
                for _ in range(100):
                    async with client.get(url + '/api/jobs') as r:
                        jobs = (await r.json())['jobs']
                    if jobs[0]['state'] == 'done': break
                    await asyncio.sleep(.01)
                else: raise AssertionError('Install did not complete')
                assert installs == [{'source': 'owner/repo', 'quant': 'Q4_K_P', 'mtp': 'auto'}]
                status, _ = await post('/api/models/register', {'path': model.path})
                assert status == 200
                status, _ = await post('/api/models/' + model.id + '/defaults', {'settings': {'context': 8192}})
                assert status == 200 and defaults[-1].context == 8192
                status, _ = await post('/api/profile', {'name': profile.name, 'model_cache': str(base / 'new-cache')})
                assert status == 200 and profile.model_cache.endswith('new-cache')
                released.clear()
                status, job = await post('/api/models/install', {'source': 'owner/repo'})
                assert status == 202
                status, _ = await post('/api/jobs/' + job['id'] + '/cancel', {})
                assert status == 200
                for _ in range(100):
                    async with client.get(url + '/api/jobs') as r:
                        jobs = (await r.json())['jobs']
                    if jobs[-1]['state'] == 'cancelled': break
                    await asyncio.sleep(.01)
                else: raise AssertionError('Download cancellation did not complete')
        finally:
            released.set()
            await runner.cleanup()
    print('Dashboard management: start/configure/log/stop, install choices/progress/registration/cancel, profile/default persistence and downloads do not block stop/files passed.')


if __name__ == '__main__':
    asyncio.run(check())
    asyncio.run(management_check())
