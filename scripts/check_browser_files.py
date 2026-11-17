"""Real HTTP proxy + optional Chromium protocol-fixture regression checks.
Run PYTHONPATH=src python scripts/check_browser_files.py --browser.
This is deterministic backend validation, not cluster/GPU validation.
"""
import asyncio
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
from aiohttp import ClientSession,CookieJar,web
from hpc_llm.browser import gateway_app
from hpc_llm.browser_files import BrowserFiles,Workspace,select_workspace
from hpc_llm.contracts import AppError
from hpc_llm.lifecycle import atomic_json

async def start(app):
    runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    return runner,f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

async def until(client,url,predicate):
    for _ in range(100):
        async with client.get(url) as response:result=await response.json()
        if predicate(result):return result
        await asyncio.sleep(.02)
    raise AssertionError(result)

def check_root_ancestor_race(folder):
    allowed=folder/'allowed';allowed.mkdir();root=allowed/'workspace';root.mkdir()
    outside=folder/'outside';outside.mkdir();(outside/'workspace').mkdir()
    (root/'secret.txt').write_text('AUTHORIZED')
    (outside/'workspace'/'secret.txt').write_text('OUTSIDE')
    files=Workspace([str(root)],[],262144)
    preview=files.preview(0,'write_workspace',{'path':str(root/'new.txt'),'content':'approved content'})
    allowed.rename(folder/'original');allowed.symlink_to(outside,target_is_directory=True)
    for operation in [lambda:files.read(0,str(root/'secret.txt')),lambda:files.commit(0,preview)]:
        try:operation()
        except AppError:pass
        else:raise AssertionError('A swapped workspace ancestor escaped into another tree')
    assert not (outside/'workspace'/'new.txt').exists()
    external_app=folder/'external-app';external_app.mkdir()
    unsafe_app=folder/'unsafe-app';unsafe_app.symlink_to(external_app,target_is_directory=True)
    try:select_workspace(SimpleNamespace(workspaces=[]),unsafe_app)
    except AppError:pass
    else:raise AssertionError('Default workspace creation followed a symbolic link')
    assert not (external_app/'workspace').exists()

async def main(browser=False):
    prefix='/rnode/gpu/43210'
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary);root=folder/'workspace';root.mkdir();state=folder/'state';state.mkdir()
        check_root_ancestor_race(folder)
        token=state/'backend.json';atomic_json(token,{'token':'internal-secret'})
        assert select_workspace(SimpleNamespace(workspaces=[str(root)]),folder,str(root))==root
        (folder/'app').mkdir()
        assert select_workspace(SimpleNamespace(workspaces=[]),folder/'app')==(folder/'app'/'workspace')
        try:select_workspace(SimpleNamespace(workspaces=[str(root)]),folder,str(state))
        except ValueError:pass
        else:raise AssertionError('Workspace escape accepted')
        secret=folder/'outside.txt';secret.write_text('private outside')
        (root/'link.txt').symlink_to(secret);(root/'linked').symlink_to(folder,target_is_directory=True)
        (root/'original.txt').write_text('original');os.link(root/'original.txt',root/'hard.txt')
        (root/'.env').write_text('secret');(root/'config.yaml').write_text('credential')
        (root/'odd<script>.txt').write_text('<script>bad()</script>')
        async def backend(request):
            assert request.headers.get('Authorization')=='Bearer internal-secret'
            assert 'Cookie' not in request.headers
            if request.path=='/v1/models':return web.json_response({'data':[{'id':'fixture-model'}]})
            if request.path=='/v1/chat/completions':
                payload=await request.json();prompt=payload['messages'][1]['content']
                results=[m for m in payload['messages'] if m['role']=='tool']
                if prompt=='endless':
                    call={'id':'read-'+str(len(results)),'type':'function','function':{'name':'list_workspace','arguments':json.dumps({'path':str(root)})}}
                elif results:
                    assert json.loads(results[-1]['content']).get('saved') is True
                    return web.json_response({'choices':[{'message':{'role':'assistant','content':'Saved the CSV after successful file execution.'}}]})
                else:
                    call={'id':'save','type':'function','function':{'name':'write_workspace','arguments':json.dumps({'path':str(root/('cancelled.csv' if prompt=='cancel me' else 'model.csv')),'content':'name,value\nalpha,42\n'})}}
                # Deliberately split JSON across network chunks: reads must reach EOF.
                response=web.StreamResponse(headers={'Content-Type':'application/json'});await response.prepare(request)
                value=json.dumps({'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[call]}}]}).encode()
                await response.write(value[:17]);await asyncio.sleep(.01);await response.write(value[17:]);return response
            return web.Response(body=gzip.compress(b'<html><body><h1>Native chat fixture</h1></body></html>'),content_type='text/html',headers={'Content-Encoding':'gzip'})
        upstream=web.Application();upstream.router.add_route('*','/{path:.*}',backend)
        backend_runner,backend_url=await start(upstream)
        workspace=BrowserFiles(backend_url,token,{'roots':[str(root)],'state_dir':str(state),'max_tool_calls':2})
        for name,args in [('shell',{}),('write_workspace',{'path':'x.txt','content':'x','extra':True}),('read_workspace','{"path":"a.txt","path":"b.txt"}')]:
            try:workspace.workspace.validate(name,args)
            except AppError:pass
            else:raise AssertionError('Invalid tool accepted')
        (root/'large.txt').write_text('x'*(workspace.limit+1))
        gateway=gateway_app(backend_url,token,'browser-secret',prefix,cookie_auth=True,workspace=workspace)
        gateway_runner,gateway_url=await start(gateway)
        async with ClientSession(auto_decompress=False) as bridge:
            async def proxy(request):
                headers={k:v for k,v in request.headers.items() if k.lower() not in {'authorization','content-length'}}
                target=gateway_url+request.path[len(prefix):]+('?' + request.query_string if request.query_string else '')
                async with bridge.request(request.method,target,headers=headers,data=await request.read(),allow_redirects=False) as response:
                    output=web.Response(status=response.status,body=await response.read(),headers={k:v for k,v in response.headers.items() if k.lower() not in {'content-length','transfer-encoding','set-cookie'}})
                    for cookie in response.headers.getall('Set-Cookie',[]):output.headers.add('Set-Cookie',re.sub(r'Path=[^;]*','Path='+prefix,cookie))
                    return output
            proxy_app=web.Application();proxy_app.router.add_route('*','/{path:.*}',proxy)
            proxy_runner,proxy_url=await start(proxy_app);url=proxy_url+prefix
            try:
                async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as client:
                    async def post(route,data,status=200):
                        async with client.post(url+'/_hpc/files'+route,json=data,headers={'Origin':proxy_url}) as response:
                            value=await response.json();assert response.status==status,(route,response.status,value);return value
                    async with client.get(url+'/_hpc/files/info') as response:assert response.status==401
                    async with client.post(url+'/_hpc/login',json={'key':'browser-secret'},headers={'Origin':proxy_url}) as response:assert response.status==200
                    async with client.get(url+'/') as response:assert b'Files &amp; tools' in await response.read()
                    info=await post('/list',{'path':str(root)})
                    assert all(e['name'] not in {'link.txt','linked','hard.txt','original.txt','.env','config.yaml'} for e in info['entries'])
                    for path in ['../outside.txt',str(secret),str(root/'link.txt'),str(root/'linked'/'outside.txt'),str(root/'.env'),str(root/'hard.txt'),str(root/'config.yaml'),str(root/'large.txt')]:await post('/read',{'path':path},400)
                    preview=await post('/preview',{'arguments':{'path':str(root/'manual.txt'),'content':'hello <script>world</script>'}})
                    assert not (root/'manual.txt').exists();await post('/approve',{'id':preview['id'],'approved':False});assert not (root/'manual.txt').exists()
                    preview=await post('/preview',{'arguments':{'path':str(root/'manual.txt'),'content':'first'}})
                    saved=await post('/approve',{'id':preview['id'],'approved':True});assert saved['saved'];await post('/approve',{'id':preview['id'],'approved':True},400)
                    preview=await post('/preview',{'arguments':{'path':str(root/'manual.txt'),'content':'second'}})
                    assert preview['overwrite'] and '-first' in preview['diff'];(root/'manual.txt').write_text('changed externally')
                    await post('/approve',{'id':preview['id'],'approved':True},400);assert (root/'manual.txt').read_text()=='changed externally'
                    preview=await post('/preview',{'operation':'mkdir_workspace','arguments':{'path':str(root/'results')}})
                    await post('/approve',{'id':preview['id'],'approved':True});assert (root/'results').is_dir()
                    for name in ['tools','%74ools','tools%2f','tools/run']:
                        async with client.post(url+'/'+name,json={'tool':'shell','params':{}}) as response:assert response.status==403
                    async with client.post(url+'/_hpc/files/preview',json={'arguments':{'path':str(root/'x.txt'),'content':'x'}},headers={'Origin':'https://evil.example'}) as response:assert response.status==403
                    task=await post('/tasks',{'prompt':'make csv'});current=await until(client,url+'/_hpc/files/tasks/'+task['id'],lambda t:t['status']=='approval')
                    assert not (root/'model.csv').exists();await post('/decision',{'id':task['id'],'approval_id':current['pending']['id'],'approved':True})
                    current=await until(client,url+'/_hpc/files/tasks/'+task['id'],lambda t:t['status']=='completed')
                    assert (root/'model.csv').read_text()=='name,value\nalpha,42\n' and current['events'][0]['result']['saved'] is True
                    async with client.get(url+'/_hpc/files/download',params={'root':0,'path':str(root/'model.csv')}) as response:assert response.status==200 and await response.text()=='name,value\nalpha,42\n' and 'attachment;' in response.headers['Content-Disposition']
                    task=await post('/tasks',{'prompt':'cancel me'});await until(client,url+'/_hpc/files/tasks/'+task['id'],lambda t:t['status']=='approval')
                    await post('/cancel',{'id':task['id']});assert not (root/'cancelled.csv').exists()
                    task=await post('/tasks',{'prompt':'endless'});current=await until(client,url+'/_hpc/files/tasks/'+task['id'],lambda t:t['status']=='failed')
                    assert len(current['events'])==2 and 'limit' in current['answer']
                if '--serve' in sys.argv:
                    print('FIXTURE_URL='+url+'/#access=browser-secret',flush=True)
                    print('FIXTURE_WORKSPACE='+str(root),flush=True)
                    await asyncio.Event().wait()
                if browser:await check_browser(url,root)
            finally:await proxy_runner.cleanup();await gateway_runner.cleanup();await backend_runner.cleanup()
    print('Browser files: real HTTP proxy/auth, UTF-8/CSV saves, approvals, mkdir/download, escapes/secrets, split-JSON model tool execution/results, limit and cancellation passed.')

async def check_browser(url,root):
    from playwright.async_api import async_playwright
    async with async_playwright() as playwright:
        browser=await playwright.chromium.launch(headless=True,executable_path=shutil.which('chromium'),args=['--no-sandbox'])
        try:
            page=await browser.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            await page.goto(url+'/#access=browser-secret');await page.get_by_text('Native chat fixture').wait_for()
            await page.locator('#hpc-files summary').click()
            panel=page.frame_locator('iframe[title="Cluster workspace files and tools"]')
            await panel.locator('#root option').wait_for(state='attached')
            await panel.locator('#destination').fill(str(root/'browser.txt'))
            await panel.locator('#content').fill('Created by the actual browser\n<script>untrusted()</script>')
            await panel.locator('#save').click();await panel.locator('#preview[open]').wait_for();assert not (root/'browser.txt').exists()
            await panel.locator('#approve').click();await panel.locator('#status').filter(has_text='Saved '+str(root/'browser.txt')).wait_for()
            assert (root/'browser.txt').read_text().startswith('Created by the actual browser')
            await panel.locator('#content').fill('overwrite that must be denied')
            await panel.locator('#save').click();await panel.locator('#preview[open]').wait_for();await panel.locator('#deny').click()
            assert (root/'browser.txt').read_text().startswith('Created by the actual browser')
            await panel.locator('#destination').fill(str(root/'browser.csv'));await panel.locator('#content').fill('x,y\n1,2\n')
            await panel.locator('#save').click();await panel.locator('#approve').click();await panel.locator('#status').filter(has_text='Saved '+str(root/'browser.csv')).wait_for()
            assert (root/'browser.csv').read_text()=='x,y\n1,2\n'
            async with page.expect_download() as download_info:await panel.locator('#download').click()
            assert (await download_info.value).suggested_filename=='browser.csv'
            await panel.locator('#browse').click();await panel.locator('#listing').get_by_text('File: browser.csv',exact=True).wait_for()
            await panel.locator('#prompt').fill('make csv');await panel.locator('#run').click();await panel.locator('#preview[open]').wait_for()
            await panel.locator('#approve').click();await panel.locator('#activity').filter(has_text='Status: completed').wait_for()
            assert not errors,errors
            print('Actual Chromium: private fragment login through stripping proxy, injected panel, exact .txt/.csv saves, content escaping, overwrite denial, download/refresh, model CSV approval/result passed.')
        finally:await browser.close()

if __name__=='__main__':asyncio.run(main('--browser' in sys.argv))
