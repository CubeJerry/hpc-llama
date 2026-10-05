"""Staged, checksum-pinned private runtime installation (standard library only)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import uuid

MAMBA = {
    'name': 'micromamba-linux-64-2.9.0-0',
    'url': 'https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64',
    'sha256': '366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3',
    'size': 20000000,
}


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fetch(asset: dict, cache: Path, offline: bool = False) -> Path:
    """Never execute an unverified download or follow a cache-file symlink."""
    name = asset['name']
    if Path(name).name != name:
        raise RuntimeError('Invalid package filename in runtime lock.')
    target = cache / name
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.is_symlink():
        raise RuntimeError(f'Refusing package-cache symlink: {target}')
    if target.is_file() and digest(target) == asset['sha256']:
        return target
    if offline:
        raise RuntimeError(f'Offline package missing or checksum mismatch: {target}')
    print(f'Downloading pinned package: {name}', flush=True)
    fd, temporary_name = tempfile.mkstemp(prefix='.download-', dir=cache)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'wb') as output:
            with urllib.request.urlopen(asset['url'], timeout=60) as response:
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > asset['size']:
                        raise RuntimeError(f'Package exceeded its pinned size: {name}')
                    output.write(block)
        if digest(temporary) != asset['sha256']:
            raise RuntimeError(f'Package checksum mismatch: {name}')
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def run(command: list[str], *, env: dict | None = None) -> None:
    subprocess.run(command, env=env, check=True)


def install_runtime(app: Path, target: Path, backend: str, cache: Path, offline: bool) -> dict:
    lock = json.loads((app / 'scripts' / 'install' / 'runtime-locks' / f'{backend}-linux-64.json').read_text())
    mamba = fetch(MAMBA, cache, offline)
    mamba.chmod(0o700)
    packages = lock['packages']
    with ThreadPoolExecutor(max_workers=4) as pool:
        archives = list(pool.map(lambda package: fetch(package, cache, offline), packages))
    target.mkdir(parents=True)
    explicit = target / 'packages.explicit.txt'
    explicit.write_text('@EXPLICIT\n' + ''.join(
        f'{archive.as_uri()}#{package["sha256"]}\n' for archive, package in zip(archives, packages)))
    environment = target / 'env'
    env = os.environ.copy()
    env['MAMBA_ROOT_PREFIX'] = str(app / 'cache' / 'mamba')
    run([str(mamba), '--no-rc', 'create', '--yes', '--offline', '--prefix', str(environment),
         '--file', str(explicit)], env=env)
    executable = environment / 'bin' / 'llama-server'
    if not executable.is_file():
        raise RuntimeError('Pinned runtime did not install llama-server.')
    (target / 'bin').mkdir()
    wrapper = target / 'bin' / 'llama-server'
    wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
runtime_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Private compiler/CUDA/user-space libraries; no module or shell-profile changes.
export LD_LIBRARY_PATH="$runtime_dir/env/lib"
exec "$runtime_dir/env/bin/llama-server" "$@"
''')
    wrapper.chmod(0o700)
    (target / 'provenance.json').write_text(json.dumps(lock, indent=2) + '\n')
    return lock


def publish_links(app: Path, targets: dict[str, Path]) -> None:
    """Publish only verified stages; rollback pointer changes on any failure."""
    old: dict[str, Path | None] = {}
    new_paths: dict[str, Path] = {}
    suffix = uuid.uuid4().hex
    try:
        for name, target in targets.items():
            link = app / f'.{name}.new-{suffix}'
            link.symlink_to(target.relative_to(app), target_is_directory=True)
            new_paths[name] = link
        for name, temporary in new_paths.items():
            current = app / name
            backup = app / f'.{name}.previous-{suffix}'
            if current.exists() or current.is_symlink():
                current.rename(backup)
                old[name] = backup
            else:
                old[name] = None
            temporary.replace(current)
    except BaseException:
        for name, previous in reversed(list(old.items())):
            (app / name).unlink(missing_ok=True)
            if previous is not None:
                previous.rename(app / name)
        raise
    finally:
        for path in new_paths.values():
            path.unlink(missing_ok=True)
    # Retain previous installations for running processes and manual rollback.


def only_missing_host_driver(output: str) -> bool:
    missing = [line.strip() for line in output.splitlines() if 'not found' in line]
    return bool(missing) and all(re.fullmatch(r'libcuda\.so\.1\s+=>\s+not found', line) for line in missing)


LEGACY_ROOT_ITEMS = {
    'requirements.lock': 'scripts/install/requirements.lock',
    'runtime-locks': 'scripts/install/runtime-locks',
    'previews': 'assets',
    **{name: None for name in (
        'SECURITY.md', 'TEST_REPORT.md', 'WEHI_SMOKE_TEST.md', 'KNOWN_ISSUES.md',
        'AGENTS.md', 'SOURCE_CHECKSUMS.sha256')},
}


def archive_legacy_root(app: Path) -> Path | None:
    """Preserve old release files in a unique archive after successful install."""
    archive = None
    moved = 0
    try:
        base = app / '.install' / 'legacy-root'
        if (app / '.install').is_symlink() or base.is_symlink():
            raise OSError('Refusing a symbolic-link archive directory')
        for name, replacement_name in LEGACY_ROOT_ITEMS.items():
            source = app / name
            replacement = app / replacement_name if replacement_name is not None else None
            if not source.exists() and not source.is_symlink():
                continue
            try:
                if source.is_symlink():
                    raise OSError('Refusing a symbolic-link legacy item')
                directory_item = name in ('runtime-locks', 'previews')
                if not (source.is_dir() if directory_item else source.is_file()):
                    raise OSError('Legacy item does not have the expected file type')
                if source.is_dir():
                    for directory, directories, files in os.walk(source, followlinks=False):
                        if any((Path(directory) / item).is_symlink() for item in directories + files):
                            raise OSError('Refusing a legacy directory containing symbolic links')
                if replacement is not None and (not replacement.exists() or any(
                    path.is_symlink() for path in [replacement, *replacement.parents]
                    if path != app and app in path.parents
                )):
                    raise OSError('The replacement release file is missing or contains a symbolic link')
                if archive is None:
                    base.mkdir(parents=True, exist_ok=True)
                    archive = Path(tempfile.mkdtemp(prefix='previous-', dir=base))
                source.rename(archive / name)
                moved += 1
            except OSError as exc:
                print(f'Installation is active; left legacy item {name} in place: {exc}. '
                      'Move it manually after reviewing its contents.', file=sys.stderr)
        if moved:
            print(f'Previous root files preserved in {archive}.', flush=True)
    except OSError as exc:
        print(f'Installation is active; legacy-root cleanup was skipped: {exc}. '
              'Existing files were retained for manual review.', file=sys.stderr)
    return archive


def verify_runtime(python: Path, runtime: Path, backend: str) -> dict:
    imports = 'import aiohttp,textual,pydantic,pypdf,docx,openpyxl,PIL,bs4,ddgs'
    run([str(python), '-c', imports])
    probe = ('from hpc_llm.runtime import probe_runtime; import sys,json; '
             'print(json.dumps(probe_runtime(sys.argv[1]).model_dump()))')
    result = subprocess.run([str(python), '-c', probe, str(runtime / 'bin' / 'llama-server')],
                            capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        capabilities = json.loads(result.stdout)
        print(capabilities['runtime_identity'], flush=True)
        return {'status': 'capabilities_verified', 'capabilities': capabilities,
                'gpu_inference_verified': False}
    if backend == 'cuda12' and shutil.which('ldd'):
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(runtime / 'env' / 'lib')
        dependencies = subprocess.run(['ldd', str(runtime / 'env' / 'bin' / 'llama-server')],
                                      capture_output=True, text=True, timeout=30, env=env)
        if dependencies.returncode == 0 and only_missing_host_driver(dependencies.stdout + dependencies.stderr):
            print('CUDA packages installed and user-space dependencies verified; host libcuda.so.1 is absent here. '
                  'Runtime capability and GPU checks are deferred to the allocated GPU node.', flush=True)
            return {'status': 'deferred_host_driver', 'missing_host_library': 'libcuda.so.1',
                    'gpu_inference_verified': False}
    raise RuntimeError('Managed runtime capability check failed. All user-space libraries must resolve; '
                       'check this host OS/CPU compatibility. ' + result.stderr[-1200:])


def install(args) -> None:
    app = args.app.resolve()
    minimum = (2, 28) if args.backend == 'cuda12' else (2, 17)
    libc, version = platform.libc_ver()
    if libc != 'glibc' or tuple(int(part) for part in version.split('.')) < minimum:
        raise RuntimeError(f'The pinned {args.backend} runtime requires glibc >= {".".join(map(str, minimum))}. This host reports {libc} {version}.')
    build = app / '.install' / 'builds' / uuid.uuid4().hex
    build.mkdir(parents=True, mode=0o700)
    environment = build / 'venv'
    published = False
    try:
        print('Creating managed Python application environment…', flush=True)
        run([str(args.uv), 'venv', '--python', str(args.python), str(environment)])
        command = [str(args.uv), 'pip', 'install', '--python', str(environment / 'bin' / 'python'),
                   '--require-hashes', '--requirement', str(app / 'scripts' / 'install' / 'requirements.lock')]
        if args.wheelhouse:
            command += ['--no-index', '--find-links', str(args.wheelhouse.resolve())]
        if args.offline:
            command += ['--offline']
        run(command)
        run([str(args.uv), 'pip', 'install', '--python', str(environment / 'bin' / 'python'),
             '--no-deps', '--no-build-isolation', '-e', str(app)])
        lock = install_runtime(app, build / 'runtime', args.backend,
                               app / 'cache' / 'downloads', args.offline)
        print('Verifying installed imports and managed runtime capabilities…', flush=True)
        receipt = verify_runtime(environment / 'bin' / 'python', build / 'runtime', args.backend)
        (build / 'runtime' / 'installation-status.json').write_text(json.dumps(receipt, indent=2) + '\n')
        # Profile parsing must succeed before replacing the existing installation.
        configure = '''import sys
from pathlib import Path
from hpc_llm.profiles import ProfileStore
store=ProfileStore(Path(sys.argv[1]),Path(sys.argv[2]))
store.configure_install(profile=sys.argv[3] or None, cache=sys.argv[4] or None)
'''
        run([str(environment / 'bin' / 'python'), '-c', configure,
             str(Path(os.environ.get('HPC_LLM_HOME', app / 'state')).resolve()), str(app),
             args.profile or '', args.cache_dir or ''])
        publish_links(app, {'.venv': environment, 'runtime': build / 'runtime'})
        published = True
        (app / 'hpc-llm').chmod(0o700)
        archive_legacy_root(app)
        print(f'Installation complete: managed Python + pinned llama.cpp {lock["version"]} ({args.backend}).')
        print('Model weights are selected/downloaded separately. Run ./hpc-llm --check, then ./hpc-llm.')
        if args.backend != 'cpu':
            print('CUDA libraries are installed. GPU execution still requires a compatible host NVIDIA driver and an allocated GPU.')
    finally:
        if not published:
            shutil.rmtree(build, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--uv', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--profile')
    parser.add_argument('--cache-dir')
    parser.add_argument('--backend', choices=('cuda12', 'cpu'), default='cuda12')
    parser.add_argument('--wheelhouse', type=Path)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    try:
        install(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'Installation failed; existing environment/runtime were preserved: {exc}') from exc


if __name__ == '__main__':
    main()
