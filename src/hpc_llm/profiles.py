"""Editable site profiles and explicit model-cache selection.

Relative import, runtime and cache paths are relative to the installation root.
Listing profiles never creates cache directories; loading the selected profile does.
"""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import re

from pydantic import ValidationError

from .contracts import AppError, SiteProfile
from .lifecycle import atomic_json


class ProfileStore:
    def __init__(self, state_root: Path, app_root: Path):
        self.state_root = Path(state_root).absolute()
        self.app_root = Path(app_root).absolute()
        self.bundled = self.app_root / 'profiles'
        self.user_profiles = self.state_root / 'profiles'
        self.config_path = self.state_root / 'config.json'

    @staticmethod
    def _name(value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
            raise AppError('validation', 'Profile names must use 1–64 letters, numbers, underscores or hyphens, starting with a letter or number.')
        return value

    @staticmethod
    def _read(path: Path) -> dict:
        if any(parent.is_symlink() for parent in [path, *path.parents]):
            raise AppError('permission', 'Profile and configuration paths cannot contain symbolic links.')
        try:
            if path.stat().st_size > 1024 * 1024:
                raise AppError('validation', 'Profile or configuration file exceeds 1 MiB.')
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError, RecursionError) as exc:
            raise AppError('validation', f'Cannot read valid JSON from {path.name}. Fix the file before continuing.') from exc
        if not isinstance(data, dict):
            raise AppError('validation', f'{path.name} must contain a JSON object.')
        return data

    def config(self) -> dict:
        if not self.config_path.exists() and not self.config_path.is_symlink():
            return {}
        data = self._read(self.config_path)
        for key in ('profile', 'model_cache'):
            if key in data and (not isinstance(data[key], str) or not data[key].strip()):
                raise AppError('validation', f'Configuration {key} must be a nonempty string.')
        return data

    def _path(self, value: str) -> Path:
        if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
            raise AppError('validation', 'Path must be a nonempty string without control characters.')
        value = value.replace('${USER}', os.environ.get('USER') or getpass.getuser())
        value = re.sub(r'\$USER\b', lambda _: os.environ.get('USER') or getpass.getuser(), value)
        value = os.path.expandvars(os.path.expanduser(value))
        if re.search(r'\$(?:\w+|\{[^}]*\})', value):
            raise AppError('validation', 'Path contains an unset environment variable.')
        path = Path(value)
        return Path(os.path.abspath(path if path.is_absolute() else self.app_root / path))

    def _parse(self, data: dict, expected_name: str | None = None) -> SiteProfile:
        data = dict(data)
        if expected_name:
            data.setdefault('name', expected_name)
            if data['name'] != expected_name:
                raise AppError('validation', 'Profile name must match its filename.')
        if not expected_name and 'name' not in data:
            raise AppError('validation', 'Imported profiles must declare a valid name.')
        # Legacy module setup is superseded by the self-contained runtime.
        data.update(modules=[], module_init='')
        try:
            profile = SiteProfile.model_validate(data)
        except (ValidationError, ValueError, TypeError) as exc:
            raise AppError('validation', 'Invalid site profile. Check scheduler, resources and preset values.') from exc
        self._name(profile.name)
        for name in profile.resource_presets:
            if not name.strip() or len(name) > 100 or any(ord(c) < 32 for c in name):
                raise AppError('validation', 'Resource preset names must be 1–100 printable characters.')
        return profile

    def _raw(self, name_or_path: str | Path) -> SiteProfile:
        selector = str(name_or_path)
        if selector.endswith('.json') or '/' in selector or '\\' in selector:
            return self._parse(self._read(self._path(selector)))
        name = self._name(selector)
        data = None
        for directory in (self.bundled, self.user_profiles):
            path = directory / f'{name}.json'
            if path.exists() or path.is_symlink():
                update = self._read(path)
                if data is None:
                    data = update
                else:
                    try:
                        resources = {**data.get('resources', {}), **update.get('resources', {})}
                    except TypeError:
                        raise AppError('validation', 'Profile resources must be a JSON object.') from None
                    data = {**data, **update, 'resources': resources}
        if data is None:
            raise AppError('validation', f'Unknown site profile: {name}. Choose an installed profile or import a JSON file.')
        return self._parse(data, name)

    def list(self) -> list[SiteProfile]:
        names = set()
        for directory in (self.bundled, self.user_profiles):
            if directory.is_symlink():
                raise AppError('permission', 'Profile directories cannot be symbolic links.')
            if directory.exists():
                names.update(self._name(path.stem) for path in directory.glob('*.json'))
        return [self._raw(name) for name in sorted(names)]

    def _cache(self, value: str) -> Path:
        path = self._path(value)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AppError('storage', f'Cannot create selected model cache: {path}. Choose a writable location.') from exc
        if not path.is_dir():
            raise AppError('storage', 'Selected model cache is not a directory.')
        return path

    def load(self, name_or_path: str | Path | None = None) -> SiteProfile:
        config = self.config()
        profile = self._raw(name_or_path or config.get('profile', 'wehi'))
        return self._effective(profile, config)

    def _effective(self, profile: SiteProfile, config: dict) -> SiteProfile:
        profile = profile.model_copy(deep=True)
        cache = os.environ.get('HPC_LLM_MODEL_CACHE') or config.get('model_cache') or profile.model_cache or str(self.app_root / 'cache' / 'models')
        profile.model_cache = str(self._cache(cache))
        runtime = os.environ.get('HPC_LLM_RUNTIME') or profile.runtime
        profile.runtime = str(self._path(runtime)) if runtime else str(self.app_root / 'runtime' / 'bin' / 'llama-server')
        return profile

    def save(self, profile: SiteProfile, select: bool = True, *, model_cache: str | Path | None = None) -> SiteProfile:
        raw = self._parse(profile.model_dump() if isinstance(profile, SiteProfile) else profile)
        config = self.config()  # Do not overwrite a malformed existing config.
        if model_cache is not None:
            self._cache(str(model_cache))
            config['model_cache'] = str(model_cache)
        effective = self._effective(raw, config) if select else raw
        atomic_json(self.user_profiles / f'{raw.name}.json', raw.model_dump())
        if select:
            config['profile'] = raw.name
        if select or model_cache is not None:
            atomic_json(self.config_path, config)
        return effective

    def select(self, name_or_path: str | Path) -> SiteProfile:
        raw = self._raw(name_or_path)
        selector = str(name_or_path)
        if selector.endswith('.json') or '/' in selector or '\\' in selector:
            return self.save(raw, select=True)
        # Validate cache creation before making selection persistent.
        resolved = self.load(raw.name)
        config = self.config()
        config['profile'] = raw.name
        atomic_json(self.config_path, config)
        return resolved

    def set_cache(self, path: str | Path) -> Path:
        config = self.config()
        resolved = self._cache(str(path))
        config['model_cache'] = str(path)
        atomic_json(self.config_path, config)
        return resolved

    def configure_install(self, profile: str | Path | None = None, cache: str | Path | None = None) -> SiteProfile:
        """Validate requested installation settings before writing one config update."""
        config = self.config()
        selector = str(profile) if profile is not None else config.get('profile', 'wehi')
        raw = self._raw(selector)
        if cache is not None:
            self._cache(str(cache))
            config['model_cache'] = str(cache)
        effective = self._effective(raw, config)
        if selector.endswith('.json') or '/' in selector or '\\' in selector:
            atomic_json(self.user_profiles / f'{raw.name}.json', raw.model_dump())
        config['profile'] = raw.name
        atomic_json(self.config_path, config)
        return effective

    def initialize(self, default_profile: str = 'wehi') -> SiteProfile:
        """Installer entry point: preserve existing selection and cache settings."""
        config = self.config()
        if 'profile' not in config:
            return self.select(default_profile)
        return self.load()
