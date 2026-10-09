#!/usr/bin/env python3
"""Offline check: embedded-MTP variants and repeated companion selection."""
import asyncio
import io
import json
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

from textual.app import App
from textual.widgets import Input, Select, Button
from hpc_llm.models import ModelLibrary
from hpc_llm.ui import ImportDialog

BASE = 'Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX'
MAIN = BASE + '-MTP-Q4_K_M.gguf'
NAMES = [BASE + '-Q4_K_M.gguf', MAIN, BASE + '-LOW-MTP-Q4_K_M.gguf',
         'mmproj-F16.gguf', 'mmproj-other-F16.gguf']


def response(*args, **kwargs):
    return io.BytesIO(json.dumps({'sha': 'a' * 40, 'siblings': [
        {'rfilename': name, 'size': 100} for name in NAMES]}).encode())


def gguf(path, arch, embedded=False):
    def string(s):
        b = s.encode()
        return struct.pack('<Q', len(b)) + b
    fields = string('general.architecture') + struct.pack('<I', 8) + string(arch)
    if embedded:
        fields += string(arch + '.nextn_predict_layers') + struct.pack('<II', 4, 1)
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1 + embedded) + fields)


async def check():
    with tempfile.TemporaryDirectory() as tmp, patch('hpc_llm.models.urlopen', response):
        library = ModelLibrary(Path(tmp) / 'registry')
        choices = library.list_install_choices('owner/repo')
        assert MAIN in [row['filename'] for row in choices['models']]
        assert BASE + '-LOW-MTP-Q4_K_M.gguf' in [row['filename'] for row in choices['models']]
        assert not choices['mtp_heads'], 'Full MTP models must not be offered as heads'
        plan = library.plan_install('owner/repo', quant=MAIN, projector='mmproj-F16.gguf', mtp='auto')
        assert plan['mtp_filename'] is None
        assert [row['filename'] for row in plan['files']] == [MAIN, 'mmproj-F16.gguf']
        # The existing registration gate must still verify embedded MTP metadata.
        main = Path(tmp) / MAIN
        vision = Path(tmp) / 'mmproj-F16.gguf'
        gguf(main, 'qwen35', True)
        gguf(vision, 'clip')
        model = library.register(main, projector_path=str(vision))
        assert model.mtp_layers == 1 and model.projector_path == str(vision)
        NAMES.append('qwen-mtp-head-Q8_0.gguf')
        try:
            assert [r['filename'] for r in library.list_install_choices('owner/repo')['mtp_heads']] == [NAMES[-1]]
        finally:
            NAMES.pop()
        app = App()
        async with app.run_test() as pilot:
            dialog = ImportDialog(library)
            await app.push_screen(dialog)
            await pilot.pause()
            dialog.query_one('#model-repo', Input).value = 'owner/repo'
            await dialog.perform('model-list')  # Ambiguous main: reveal choices.
            dialog.query_one('#model-file', Select).value = MAIN
            await dialog.perform('model-list')  # Ambiguous projector: preserve main.
            assert dialog.query_one('#model-file', Select).value == MAIN
            dialog.query_one('#model-remote-projector', Select).value = 'mmproj-F16.gguf'
            dialog.query_one('#model-remote-mtp', Select).value = 'auto'
            await dialog.perform('model-list')
            assert dialog.install_plan['filename'] == MAIN
            assert not dialog.query_one('#model-download', Button).disabled
            assert dialog.preview_selection == dialog.install_selection()
    print('MTP choices, registration and repeated TUI preview passed')


if __name__ == '__main__':
    asyncio.run(check())
