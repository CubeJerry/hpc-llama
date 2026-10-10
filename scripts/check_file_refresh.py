#!/usr/bin/env python3
"""Exercise live filesystem refresh through the real supervisor and Textual UI."""
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Button, Input, TextArea, Tree
from hpc_llm.client import SessionClient
from hpc_llm.contracts import InferenceSettings, ModelSpec, ResourceRequest, SiteProfile
from hpc_llm.lifecycle import SessionManager
from hpc_llm.supervisor import Supervisor
from hpc_llm.ui import ChatApp


async def check():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        os.chdir(root)
        service = None
        client = None
        try:
            folder = root / 'documents'
            folder.mkdir()
            (root / 'old.txt').write_text('old')
            manager = SessionManager(root / 'state', root / 'hpc-llm')
            manifest = manager.create(ModelSpec(name='Demo', path='demo.gguf'),
                ResourceRequest(), SiteProfile(modules=[]), InferenceSettings(), demo=True)
            service = Supervisor(manifest)
            await service.start()
            client = SessionClient(service.manifest)
            app = ChatApp(client)
            async with app.run_test(size=(118, 36)) as pilot:
                await pilot.pause(.2)
                tree = app.query_one('#file-tree', Tree)
                names = lambda: {str(node.label) for node in tree.root.children}
                documents = next(n for n in tree.root.children if str(n.label) == 'documents')
                documents.expand()
                await pilot.pause()
                assert str(folder) in app._tree_loaded
                app.query_one('#composer', TextArea).load_text('Keep this unsent draft')
                (root / 'new.txt').write_text('uploaded during the session')
                (folder / 'nested.txt').write_text('new nested upload')
                (root / 'old.txt').unlink()
                assert 'new.txt' not in names() and 'old.txt' in names()
                await pilot.click('#refresh-files')
                await pilot.pause()
                assert 'new.txt' in names() and 'old.txt' not in names()
                assert str(folder) not in app._tree_loaded
                documents = next(n for n in tree.root.children if str(n.label) == 'documents')
                documents.expand()
                await pilot.pause()
                assert 'nested.txt' in {str(n.label) for n in documents.children}
                app.query_one('#file-filter', Input).value = 'new'
                await pilot.click('#refresh-files')
                await pilot.pause()
                assert 'new.txt' in names() and 'documents' not in names()
                assert app.query_one('#file-filter', Input).value == 'new'
                assert app.query_one('#composer', TextArea).text == 'Keep this unsent draft'
                assert not app.query_one('#refresh-files', Button).disabled
                original_api = app.api
                async def failing_api(method, path, **kwargs):
                    if path == '/files':
                        raise OSError('Directory temporarily unavailable')
                    return await original_api(method, path, **kwargs)
                with patch.object(app, 'api', failing_api):
                    await pilot.click('#refresh-files')
                    await pilot.pause()
                    assert 'Directory temporarily unavailable' in app.last_error
                    assert 'new.txt' in names()
                    assert not app.query_one('#refresh-files', Button).disabled
                await pilot.resize_terminal(80, 24)
                app.query_one('#drawer').display = True
                await pilot.pause()
                assert await pilot.click('#refresh-files')
        finally:
            if client:
                await client.close()
            if service:
                await service.close()
            os.chdir(previous)
    print('File refresh: uploads, removals, nested cache, filter, draft and compact layout passed')


if __name__ == '__main__':
    asyncio.run(check())
