"""Offline checks: real HTTP gateway, authentication, SSE, prefix and SSH argv."""
import asyncio
from pathlib import Path
import tempfile
from aiohttp import web, ClientSession, BasicAuth
from hpc_llm.browser import gateway_app, ssh_command
from hpc_llm.lifecycle import atomic_json

async def check():
    seen = []
    async def backend(request):
        assert request.headers['Authorization'] == 'Bearer backend-secret'
        assert 'Cookie' not in request.headers
        seen.append((request.path, await request.read()))
        if request.path == '/v1/chat/completions':
            stream = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await stream.prepare(request)
            await stream.write(b'data: first\n\n')
            await asyncio.sleep(.1)
            await stream.write(b'data: [DONE]\n\n')
            return stream
        return web.Response(text='<html>stock UI fixture</html>', content_type='text/html')
    upstream = web.Application(); upstream.router.add_route('*', '/{p:.*}', backend)
    runner = web.AppRunner(upstream); await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    with tempfile.TemporaryDirectory() as d:
        key = Path(d)/'key.json'; atomic_json(key, {'token':'backend-secret'})
        gateway = web.AppRunner(gateway_app(f'http://127.0.0.1:{port}', key, 'browser-secret', '/rnode/login/8765', 'https://ondemand.example.edu'))
        await gateway.setup(); exposed = web.TCPSite(gateway, '127.0.0.1', 0); await exposed.start()
        url = f'http://127.0.0.1:{exposed._server.sockets[0].getsockname()[1]}'
        try:
            async with ClientSession() as client:
                async with client.get(url) as r: assert r.status == 401
                async with client.get(url, auth=BasicAuth('hpc','wrong')) as r: assert r.status == 401
                auth = BasicAuth('hpc','browser-secret')
                async with client.get(url+'/rnode/login/8765/', auth=auth) as r:
                    assert r.status == 200 and 'stock UI' in await r.text()
                async with client.post(url+'/v1/chat/completions', auth=auth, json={'messages':[]}, headers={'Origin':'https://ondemand.example.edu'}) as r:
                    assert r.status == 200
                    assert await r.content.readline() == b'data: first\n'
                    assert '[DONE]' in await r.text()
                async with client.post(url+'/completion', auth=auth, headers={'Origin':'https://evil.example'}) as r: assert r.status == 403
                assert len(seen)==2 and seen[0][0]=='/'
                key.unlink()
                async with client.get(url, auth=auth) as r: assert r.status == 503
        finally:
            await gateway.cleanup()
    await runner.cleanup()
    argv=ssh_command('login.example.edu', 25000, 25001)
    assert '-R' in argv and '127.0.0.1:25000:127.0.0.1:25001' in argv
    assert 'StrictHostKeyChecking=yes' in argv and 'ExitOnForwardFailure=yes' in argv
    for host in ('-oProxyCommand=bad', 'host;touch x', 'x y'):
        try: ssh_command(host,25000,25001)
        except ValueError: pass
        else: raise AssertionError('Invalid SSH destination accepted')
    print('Browser gateway: authentication, prefix, streaming, POST, origin checks, missing backend and SSH validation passed.')

if __name__=='__main__': asyncio.run(check())
