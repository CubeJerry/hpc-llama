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
import shutil
import secrets

from pydantic import ValidationError

from .contracts import AppError, SiteProfile
from .config import AppConfig, load_config, save_config, validation_error, read_mapping


class ProfileStore:
    def __init__(self, state_root: Path, app_root: Path, config_path: Path | None = None):
        self.state_root = Path(state_root).absolute()
        self.app_root = Path(app_root).absolute()
        self.bundled = self.app_root / 'profiles'
        self.user_profiles = self.state_root / 'profiles'
        self.legacy_config_path = self.state_root / 'config.json'
        self.config_path = Path(config_path or os.environ.get('HPC_LLM_CONFIG') or self.state_root / 'config.yaml').absolute()

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

    def document(self, default_profile: str = 'generic') -> AppConfig:
        if self.config_path.exists() or self.config_path.is_symlink():
            return load_config(self.config_path)
        return self._migrate(default_profile)

    def config(self) -> dict:
        document = self.document()
        result = {'profile': document.selected_profile, 'state_directory': document.state_directory}
        if document.model_cache:
            result['model_cache'] = document.model_cache
        return result

    def _migrate(self, default_profile: str) -> AppConfig:
        legacy = self._read(self.legacy_config_path) if self.legacy_config_path.exists() else {}
        unknown = set(legacy) - {'profile', 'model_cache'}
        if unknown:
            raise AppError('validation', 'Legacy config.json has unknown keys. Preserve the file and move its profile/model_cache settings into config.yaml manually.')
        names = set()
        for directory in (self.bundled, self.user_profiles):
            if directory.is_symlink():
                raise AppError('permission', 'Profile directories cannot be symbolic links.')
            if directory.exists():
                names.update(self._name(path.stem) for path in directory.glob('*.json'))
        profiles = {name: self._legacy_raw(name) for name in sorted(names)}
        if not profiles:
            profiles['generic'] = SiteProfile()
        selected = legacy.get('profile', default_profile)
        if selected not in profiles:
            raise AppError('validation', 'The selected legacy profile is missing; restore its JSON file before migration.')
        try:
            document = AppConfig(selected_profile=selected, model_cache=legacy.get('model_cache', ''), profiles=profiles)
        except ValidationError as exc:
            raise validation_error(exc, 'legacy configuration') from exc
        originals = ([self.legacy_config_path] if self.legacy_config_path.exists() else [])
        originals += list(self.user_profiles.glob('*.json')) if self.user_profiles.exists() else []
        if originals:
            backup = self.state_root / 'migration-backup-v1'
            if backup.exists() or backup.is_symlink():
                # Never overwrite an earlier rollback snapshot.
                backup = self.state_root / ('migration-backup-v1-' + secrets.token_hex(4))
            backup.mkdir(parents=True, mode=0o700)
            for path in originals:
                self._read(path)
                relative = Path('profiles') / path.name if path.parent == self.user_profiles else Path(path.name)
                target = backup / relative
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(path, target)
                target.chmod(0o600)
        # Originals stay intact. Models, defaults and session snapshots live in
        # other files and are deliberately never touched during migration.
        save_config(self.config_path, document)
        return document

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
        try:
            profile = SiteProfile.model_validate(data)
        except (ValidationError, ValueError, TypeError) as exc:
            raise validation_error(exc, 'site profile') from exc
        self._name(profile.name)
        for name in profile.resource_presets:
            if not name.strip() or len(name) > 100 or any(ord(c) < 32 for c in name):
                raise AppError('validation', 'Resource preset names must be 1–100 printable characters.')
        return profile

    def _legacy_raw(self, name_or_path: str | Path) -> SiteProfile:
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

    def _raw(self, name_or_path: str | Path) -> SiteProfile:
        selector = str(name_or_path)
        if selector.endswith(('.json', '.yaml', '.yml')) or '/' in selector or '\\' in selector:
            path = self._path(selector)
            return self._parse(self._read(path) if path.suffix == '.json' else read_mapping(path))
        name = self._name(selector)
        profile = self.document().profiles.get(name)
        if profile is None:
            raise AppError('validation', f'Unknown site profile: {name}. Choose a configured profile or import a profile file.')
        return profile.model_copy(deep=True)

    def list(self) -> list[SiteProfile]:
        return [profile.model_copy(deep=True) for _, profile in sorted(self.document().profiles.items())]

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
        profile = self._raw(name_or_path or config.get('profile', 'generic'))
        return self._effective(profile, config)

    def _effective(self, profile: SiteProfile, config: dict) -> SiteProfile:
        profile = profile.model_copy(deep=True)
        cache = os.environ.get('HPC_LLM_MODEL_CACHE') or config.get('model_cache') or profile.model_cache or str(self.app_root / 'cache' / 'models')
        profile.model_cache = str(self._cache(cache))
        runtime = os.environ.get('HPC_LLM_RUNTIME') or profile.runtime
        profile.runtime = str(self._path(runtime)) if runtime else str(self.app_root / 'runtime' / 'bin' / 'llama-server')
        profile.workspaces = [str(self._path(value)) for value in profile.workspaces]
        return profile

    def save(self, profile: SiteProfile, select: bool = True, *, model_cache: str | Path | None = None) -> SiteProfile:
        raw = self._parse(profile.model_dump() if isinstance(profile, SiteProfile) else profile)
        document = self.document()
        config = self.config()
        if model_cache is not None:
            self._cache(str(model_cache))
            document.model_cache = str(model_cache)
            config['model_cache'] = str(model_cache)
        effective = self._effective(raw, config) if select else raw
        document.profiles[raw.name] = raw
        if select:
            document.selected_profile = raw.name
        save_config(self.config_path, document)
        return effective

    def select(self, name_or_path: str | Path) -> SiteProfile:
        raw = self._raw(name_or_path)
        return self.save(raw, select=True)

    def set_cache(self, path: str | Path) -> Path:
        resolved = self._cache(str(path))
        document = self.document()
        document.model_cache = str(path)
        save_config(self.config_path, document)
        return resolved

    def configure_install(self, profile: str | Path | None = None, cache: str | Path | None = None) -> SiteProfile:
        """Validate requested settings before writing one canonical YAML update."""
        document = self.document()
        raw = self._raw(profile or document.selected_profile)
        return self.save(raw, select=True, model_cache=cache)

    def initialize(self, default_profile: str = 'generic') -> SiteProfile:
        """Installer entry point: preserve existing selection and cache settings."""
        self.document(default_profile)
        return self.load()
