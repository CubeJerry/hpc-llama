"""Versioned YAML configuration; no shell expansion, custom tags or YAML aliases.

Resource precedence (lowest first): persisted resources, named/default preset,
one-session edits, explicit CLI overrides. Resolving never saves a preset.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import secrets
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from .contracts import AppError, Record, ResourceRequest, SiteProfile

MAX_CONFIG_BYTES = 1024 * 1024


def validation_error(exc: ValidationError, label: str = "configuration") -> AppError:
    details = []
    for error in exc.errors(include_input=False, include_url=False)[:8]:
        details.append(f"{'.'.join(str(v) for v in error['loc'])}: {error['msg']}")
    return AppError("validation", f"Invalid {label}: " + "; ".join(details))


class AppConfig(Record):
    schema_version: Literal[1] = 1
    selected_profile: str = "generic"
    model_cache: str = ""
    state_directory: str = ""
    profiles: dict[str, SiteProfile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_profiles(self):
        if self.selected_profile not in self.profiles:
            raise ValueError("selected_profile must name an entry in profiles")
        for name, profile in self.profiles.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or profile.name != name:
                raise ValueError("Profile keys must be 1–64 letters/numbers/_/- and match profile.name")
        for value in (self.model_cache, self.state_directory):
            if any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError("Configuration paths cannot contain control characters")
        return self


class _SafeLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise yaml.YAMLError("YAML aliases are unsupported; repeat the small mapping explicitly")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in keys:
                raise yaml.YAMLError("Configuration keys must be unique strings")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def read_mapping(path: Path) -> dict:
    path = Path(path)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise AppError("permission", "Configuration paths cannot contain symbolic links")
    try:
        if path.stat().st_size > MAX_CONFIG_BYTES:
            raise AppError("validation", "Configuration exceeds 1 MiB")
        text = path.read_text(encoding="utf-8")
        value = yaml.load(text, Loader=_SafeLoader)
    except (OSError, ValueError, yaml.YAMLError, RecursionError) as exc:
        # Parser error strings can include config content; never echo secrets.
        raise AppError("validation", f"Cannot read {path.name}. Use valid YAML with unique keys, no aliases/custom tags, and UTF-8 encoding") from exc
    if not isinstance(value, dict):
        raise AppError("validation", f"{path.name} must contain a YAML mapping")
    return value


def load_config(path: Path) -> AppConfig:
    data = read_mapping(path)
    # Omitted names are convenient in hand-written YAML, but canonical saves
    # include them for backwards-compatible SiteProfile exports.
    if isinstance(data.get("profiles"), dict):
        data["profiles"] = {name: dict(value, name=value.get("name", name)) if isinstance(value, dict) else value for name, value in data["profiles"].items()}
    if "schema_version" not in data:
        raise AppError("validation", "Set schema_version: 1 in config.yaml")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise AppError("validation", "Unsupported schema_version; this release supports integer version 1")
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise validation_error(exc) from exc


def save_config(path: Path, config: AppConfig) -> None:
    path = Path(path).absolute()
    try:
        config = AppConfig.model_validate(config.model_dump())
    except ValidationError as exc:
        raise validation_error(exc) from exc
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise AppError("permission", "Configuration paths cannot contain symbolic links")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() and path.stat().st_uid != os.getuid():
        raise AppError("permission", "Configuration must belong to the current user")
    text = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    if len(text.encode()) > MAX_CONFIG_BYTES:
        raise AppError("validation", "Configuration exceeds 1 MiB")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise AppError("storage", "Cannot save config.yaml; check permissions and free space") from exc
    finally:
        temporary.unlink(missing_ok=True)


def resolve_resources(profile: SiteProfile, preset: str | None = None,
                      session: dict | ResourceRequest | None = None,
                      cli: dict | None = None) -> ResourceRequest:
    result = profile.resources.model_dump()
    selected = profile.default_preset if preset is None else preset
    if selected:
        if selected not in profile.resource_presets:
            raise AppError("validation", f"Unknown resource preset: {selected}")
        result.update(profile.resource_presets[selected].model_dump())
    for overrides in (session, cli):
        if isinstance(overrides, ResourceRequest):
            overrides = overrides.model_dump()
        if overrides:
            result.update({key: value for key, value in overrides.items() if value is not None})
    try:
        return ResourceRequest.model_validate(result)
    except ValidationError as exc:
        raise validation_error(exc, "resource request") from exc
