"""TUI-to-browser handoff; the GPU allocation owns the whole web service."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import AppError, session_is_ended
from .lifecycle import atomic_json, private_token, read_private_json


def validate_origin(value: str) -> str:
    value = value.strip().rstrip('/')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path
            or any(c.isspace() for c in value)):
        raise ValueError('Enter your Open OnDemand HTTPS address without a path, for example https://ondemand.example.edu')
    return value


def saved_origin(root: Path, app_root: Path) -> str:
    for path in (root / 'browser-settings.json', root / 'dashboard-settings.json',
                 app_root.parent / 'wehi-comfyui/state/settings.json'):
        try:
            return validate_origin(json.loads(path.read_text())['ood_origin'])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return ''


def prepare_browser(manifest, root: Path, origin: str) -> None:
    origin = validate_origin(origin)
    atomic_json(root / 'browser-settings.json', {'ood_origin': origin})
    atomic_json(Path(manifest.directory) / 'browser.json',
                {'transport': 'compute', 'ood_origin': origin})
    private_token(Path(manifest.directory) / 'browser.password')


def is_compute_browser(manifest) -> bool:
    path = Path(manifest.directory) / 'browser.json'
    return path.exists() and read_private_json(path).get('transport') == 'compute'


def show_browser(manifest) -> int:
    from .cli import display
    if session_is_ended(manifest) or manifest.backend_state in {'failed', 'stopped'}:
        raise AppError('startup', 'This browser job has ended. Start a new session in the launcher.')
    folder = Path(manifest.directory)
    if manifest.backend_state != 'ready' or not (folder / 'browser-access.json').exists():
        raise AppError('startup', 'The browser job is still queued or loading. Use attach again when ready.')
    access = read_private_json(folder / 'browser-access.json')
    password = private_token(folder / 'browser.password')
    display(f'Session: {manifest.id}\nJob: {manifest.job_id}\n'
            f'Open chat in a browser tab: {access["url"]}\n'
            f'Username: hpc\nPassword: {password}\n'
            'Leave the llama.cpp API key setting empty.\n'
            'You can close this terminal. The batch job keeps the chat running until its time limit.\n'
            f'Show this link again: ./hpc-llm attach {manifest.id}\n'
            f'Stop the GPU job: ./hpc-llm stop {manifest.id}', flush=True)
    return 0
