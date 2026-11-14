"""Exercise cookie login through a proxy that strips Authorization like OOD can."""
import asyncio
import json
from pathlib import Path
import re
import subprocess
import tempfile

from aiohttp import ClientSession, CookieJar, web
from hpc_llm.browser import gateway_app
from hpc_llm.lifecycle import atomic_json


async def start(app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    return runner, f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'


def check_script(page):
    script = re.search(r'<script nonce="[^"]+">(.*?)</script>', page, re.S).group(1)
    harness = '''const vm = require('node:vm');
const assert = require('node:assert/strict');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const elements = {};
let cleaned = false, opened = '', calls = [];
const context = {
  URLSearchParams,
  document: {getElementById: id => elements[id] ||= {addEventListener(){}}},
  history: {replaceState: (a,b,url) => {assert(!url.includes('secret')); cleaned = true;}},
  location: {hash:'#access=secret',pathname:input.root,search:'',replace:url=>opened=url},
  fetch: async (url, options) => {calls.push({url,options}); return {ok:true};}
};
vm.runInNewContext(input.script, context);
setImmediate(() => {
  assert(cleaned);
  assert.equal(opened,input.root);
  assert.equal(calls[0].url,input.root+'_hpc/login');
  assert.equal(JSON.parse(calls[0].options.body).key,'secret');
  assert.equal(calls[1].url,input.root+'_hpc/session');
  assert.equal(calls[0].options.credentials,'same-origin');
});'''
    subprocess.run(['node', '-e', harness], input=json.dumps({'script': script, 'root': '/rnode/gpu01/43210/'}),
                   text=True, check=True)


async def main():
    prefix = '/rnode/gpu01/43210'
    upstream_calls = []
    async def backend(request):
        assert request.headers.get('Authorization') == 'Bearer internal-secret'
        assert 'Cookie' not in request.headers
        upstream_calls.append(request.path)
        return web.Response(text='native chat', content_type='text/html')
    upstream = web.Application()
    upstream.router.add_route('*', '/{path:.*}', backend)
    backend_runner, backend_url = await start(upstream)
    with tempfile.TemporaryDirectory() as folder:
        token = Path(folder) / 'token.json'
        atomic_json(token, {'token': 'internal-secret'})
        gateway = gateway_app(backend_url, token, 'private-link-key', prefix, 'https://ood.example.edu', cookie_auth=True)
        gateway_runner, gateway_url = await start(gateway)
        async with ClientSession() as bridge:
            async def proxy(request):
                assert request.path.startswith(prefix + '/')
                headers = {k: v for k, v in request.headers.items()
                           if k.lower() not in {'authorization', 'host', 'content-length'}}
                async with bridge.request(request.method, gateway_url + request.path[len(prefix):],
                                          headers=headers, data=await request.read()) as response:
                    output = web.Response(status=response.status, body=await response.read(),
                                          headers={k: v for k,v in response.headers.items()
                                                   if k.lower() not in {'content-length', 'transfer-encoding', 'set-cookie'}})
                    for cookie in response.headers.getall('Set-Cookie', []):
                        # OOD replaces cookie Path with the public rnode prefix.
                        output.headers.add('Set-Cookie', re.sub(r'Path=[^;]*', 'Path=' + prefix, cookie))
                    return output
            fake_ood = web.Application()
            fake_ood.router.add_route('*', '/{path:.*}', proxy)
            proxy_runner, proxy_url = await start(fake_ood)
            jar = CookieJar(unsafe=True, treat_as_secure_origin=proxy_url)
            try:
                async with ClientSession(cookie_jar=jar) as client:
                    root = proxy_url + prefix + '/'
                    async with client.get(root, headers={'Authorization': 'Basic deliberately-stripped'}) as r:
                        assert r.status == 200 and 'WWW-Authenticate' not in r.headers
                        page = await r.text()
                        assert 'private-link-key' not in page
                        check_script(page)
                    async with client.get(root + 'v1/models') as r:
                        assert r.status == 401 and 'WWW-Authenticate' not in r.headers
                    async with client.post(root + '_hpc/login', json={'key': 'wrong'}) as r:
                        assert r.status == 403 and not r.cookies
                    async with client.post(root + '_hpc/login', json={'key': 'private-link-key'},
                                           headers={'Origin': 'https://evil.example'}) as r:
                        assert r.status == 403 and not r.cookies
                    async with client.post(root + '_hpc/login', data=b'x' * 4097) as r:
                        assert r.status == 413
                    async with client.post(root + '_hpc/login', json={'key': 'private-link-key'},
                                           headers={'Origin': 'https://ood.example.edu'}) as r:
                        assert r.status == 200
                        cookie = r.headers['Set-Cookie']
                        assert 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=Strict' in cookie
                    async with client.get(root + '_hpc/session') as r:
                        assert r.status == 200
                    async with client.get(root) as r:
                        assert r.status == 200 and await r.text() == 'native chat'
                    async with client.post(root + 'v1/chat/completions', json={}, headers={'Origin':'https://ood.example.edu'}) as r:
                        assert r.status == 200
                    assert upstream_calls == ['/', '/v1/chat/completions']
                    jar.clear()
                    async with client.get(root + '_hpc/session') as r:
                        assert r.status == 401 and 'WWW-Authenticate' not in r.headers
            finally:
                await proxy_runner.cleanup()
                await gateway_runner.cleanup()
    await backend_runner.cleanup()
    print('Cookie login passed through Authorization-stripping/prefix-rewriting proxy; auto-login JS, private cookies, CSRF, limits and API protection passed.')


if __name__ == '__main__':
    asyncio.run(main())
