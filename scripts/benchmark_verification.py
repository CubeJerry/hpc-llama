#!/usr/bin/env python3
"""Actual ModelLibrary verification benchmark; synthetic bytes, no network/GPU.

Measures a locally cached sparse file. It is not a WEHI disk benchmark.
"""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time

from hpc_llm.models import ModelLibrary


def main():
    with tempfile.TemporaryDirectory(prefix='hpc-verification-benchmark-') as directory:
        root = Path(directory)
        model_path = root / 'bench.gguf'
        with model_path.open('wb') as handle:
            handle.write(b'GGUF' + struct.pack('<IQQ', 3, 0, 0))
            handle.truncate(256 * 1024 * 1024)
        with model_path.open('rb') as handle:
            checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
        library = ModelLibrary(root / 'registry')
        model = library.register(model_path, repo_id='fixture/bench', revision='b' * 40)
        item = dict(filename='bench.gguf', size=model_path.stat().st_size,
                    sha256=checksum, revision='b' * 40, projector=False, mtp=False)
        library.list_remote = lambda *args: [dict(item)]
        digest = hashlib.file_digest
        reads = []
        def counted(handle, algorithm):
            import os
            reads.append(os.fstat(handle.fileno()).st_size)
            return digest(handle, algorithm)
        hashlib.file_digest = counted
        try:
            results = []
            for label, force in [('first', False), ('unchanged', False), ('full_recheck', True), ('unchanged_again', False)]:
                reads.clear()
                start = time.perf_counter()
                plan = library.plan_install(model_id=model.id, force_verify=force)
                library.execute_install(plan)
                results.append(dict(phase=label, seconds=round(time.perf_counter()-start, 6),
                                    checksum_reads=len(reads), checksum_bytes=sum(reads)))
            print(json.dumps(results, indent=2))
        finally:
            hashlib.file_digest = digest


if __name__ == '__main__':
    main()
