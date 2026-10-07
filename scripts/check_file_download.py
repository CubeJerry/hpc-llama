#!/usr/bin/env python3
"""Offline regression check for explicit non-GGUF Hugging Face downloads."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

from hpc_llm.contracts import AppError
from hpc_llm.models import ModelLibrary


class Response(io.BytesIO):
    status = 200
    headers = {}


def main():
    commit = 'a' * 40
    payload = b'non-GGUF model bytes'
    filename = 'weights/encoder.safetensors'
    transfers = []

    def fetch(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        if '/api/models/' in url:
            return Response(json.dumps({'sha': commit, 'siblings': [
                {'rfilename': filename, 'lfs': {'size': len(payload),
                    'sha256': hashlib.sha256(payload).hexdigest()}}
            ]}).encode())
        assert unquote(urlsplit(url).path).endswith(f'/resolve/{commit}/{filename}')
        transfers.append(url)
        offset = int(request.get_header('Range', 'bytes=0-')[6:-1])
        result = Response(payload[offset:])
        if offset:
            result.status = 206
            result.headers = {'Content-Range': f'bytes {offset}-{len(payload)-1}/{len(payload)}'}
        return result

    with tempfile.TemporaryDirectory() as directory, patch('hpc_llm.models.urlopen', fetch):
        library = ModelLibrary(Path(directory) / 'registry')
        for route in ('blob', 'resolve'):
            source = f'https://huggingface.co/owner/repo/{route}/main/{filename}?download=true'
            plan = library.plan_install(source)
            assert plan['download_only'] and len(plan['files']) == 1
            if not transfers:
                partial = Path(plan['destination']) / (filename + '.part')
                Path(plan['destination']).mkdir(parents=True, mode=0o700)
                partial.parent.mkdir(parents=True, mode=0o700)
                partial.write_bytes(payload[:5])
            result = library.execute_install(plan)
            assert isinstance(result, Path) and result.read_bytes() == payload
            assert library.list() == []
        assert len(transfers) == 1
        assert library.list_remote('owner/repo') == []
        assert library.plan_install(source, force_verify=True)['cached_bytes'] == len(payload)
        result.write_bytes(b'x' * len(payload))
        try:
            library.execute_install(plan)
        except AppError:
            pass
        else:
            raise AssertionError('Changed cached file passed verification')
        for bad in ('../escape.bin', '%2Ftmp/escape.bin', 'dir%5Cescape.bin'):
            try:
                library.plan_install(f'https://huggingface.co/owner/repo/blob/main/{bad}')
            except AppError:
                pass
            else:
                raise AssertionError('Unsafe path accepted')
    print('Non-GGUF download, resume, reuse, registry isolation and validation checks passed.')


if __name__ == '__main__':
    main()
