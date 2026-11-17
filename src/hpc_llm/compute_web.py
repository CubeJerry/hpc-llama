"""TUI-to-browser handoff; the GPU allocation owns the whole web service."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .contracts import AppError, session_is_ended
from .lifecycle import atomic_json, private_token, read_private_json
from .connections import validate_origin, connection_options, ood_prefix, ssh_destination, port_number, tunnel_text


def saved_origin(root: Path, app_root: Path) -> str:
    for path in (root / 'browser-settings.json', root / 'dashboard-settings.json',
                 app_root.parent / 'wehi-comfyui/state/settings.json'):
        try:
            return validate_origin(json.loads(path.read_text())['ood_origin'])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return ''


def prepare_browser(manifest, root: Path, origin: str | None = None, *, workspace=None) -> None:
    options = connection_options(manifest.profile)
    mode = options.get('mode', 'ood')
    if mode == 'ood':
        origin = validate_origin(origin or options.get('ood_origin', ''))
        options['ood_origin'] = origin
        ood_prefix(options.get('ood_path', '/rnode/{node}/{port}'), 'compute.example.edu', 8765)
        atomic_json(root / 'browser-settings.json', {'ood_origin': origin})
    elif mode == 'reverse-ssh':
        if not options.get('relay_host') or '{node}' in options['relay_host']:
            raise ValueError('Reverse SSH requires an explicit stable relay_host reachable from compute nodes; use the same concrete host from your computer.')
        ssh_destination(options['relay_host'])
        port_number(options.get('relay_port', 0), dynamic=True)
        if options.get('relay_jump_host'):
            ssh_destination(options['relay_jump_host'])
    elif mode == 'local-forward':
        ssh_destination(options.get('ssh_host') or '{node}', node='compute.example.edu')
    else:
        raise ValueError('Connection mode must be ood, local-forward or reverse-ssh.')
    port_number(options.get('local_port', 8765))
    if options.get('jump_host'):
        ssh_destination(options['jump_host'])
    config = {'transport': 'compute', 'connection': options}
    if mode == 'ood':
        config['ood_origin'] = origin  # Existing compute job snapshots remain readable.
    from .browser_files import select_workspace
    from .cli import APP_ROOT
    selected = select_workspace(manifest.profile, APP_ROOT, workspace)
    tools = manifest.profile.tools.model_dump()
    protected = [APP_ROOT / part for part in ('src', 'runtime', 'scripts', 'profiles', 'install.sh', 'hpc-llm', 'pyproject.toml')]
    protected.extend(Path(value) for value in (manifest.model.path, manifest.model.projector_path, manifest.model.mtp_path, manifest.profile.runtime, manifest.profile.model_cache) if value)
    if os.environ.get('HPC_LLM_CONFIG'):
        protected.append(Path(os.environ['HPC_LLM_CONFIG']).expanduser())
    config['files'] = {**tools, 'roots': [str(selected)], 'state_dir': str(Path(root).resolve()),
                       'excluded': [str(path.resolve()) for path in protected]}
    atomic_json(Path(manifest.directory) / 'browser.json', config)
    private_token(Path(manifest.directory) / 'browser.password')


def is_compute_browser(manifest) -> bool:
    path = Path(manifest.directory) / 'browser.json'
    return path.exists() and read_private_json(path).get('transport') in {'compute', 'portable'}


def show_browser(manifest) -> int:
    from .cli import display
    if session_is_ended(manifest) or manifest.backend_state in {'failed', 'stopped'}:
        raise AppError('startup', 'This browser job has ended. Start a new session in the launcher.')
    folder = Path(manifest.directory)
    if manifest.backend_state != 'ready' or not (folder / 'browser-access.json').exists():
        raise AppError('startup', 'The browser job is still queued or loading. Use attach again when ready.')
    access = read_private_json(folder / 'browser-access.json')
    if access.get('auth') != 'cookie':
        raise AppError('startup', f'This running job still uses the old sign-in code. Stop it with ./hpc-llm stop {manifest.id}, then start a new browser session in the launcher.')
    password = private_token(folder / 'browser.password')
    config = read_private_json(folder / 'browser.json')
    options = config.get('connection', {'mode': 'ood'})
    display(f'Session: {manifest.id}\nJob: {manifest.job_id}', flush=True)
    if options['mode'] != 'ood':
        display('On your computer, run this command and leave that local SSH process running:\n' +
                tunnel_text(options, access['node'], access['port'], access.get('relay_port')) + '\n'
                'If the local port is busy, close the conflicting tunnel first. Then open the private link below.\n'
                'The compute service is ready; browser access also requires this workstation tunnel.', flush=True)
    else:
        display('The compute service is ready. Browser access uses your site-approved Open OnDemand proxy.', flush=True)
    from .terminal import print_browser_link
    print_browser_link(f'{access["url"]}#access={password}')
    display('The link signs you in automatically. Keep it private; no extra username or password is needed.\n'
            'Leave the llama.cpp API key setting empty.\n'
            'You can close this terminal. The batch job keeps the chat running until its time limit.\n'
            f'Show this link again: ./hpc-llm attach {manifest.id}\n'
            f'Stop the GPU job: ./hpc-llm stop {manifest.id}', flush=True)
    return 0
