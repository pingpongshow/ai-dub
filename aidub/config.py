"""Configuration: packaged defaults, merged with /config/config.yaml and secrets."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_FILE = Path(__file__).with_name("default_config.yaml")
_VAR = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _substitute(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _VAR.sub(lambda m: env.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _substitute(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, env) for v in value]
    return value


class Config:
    """Settings for one process. `raw` keeps the user's file for the settings page."""

    def __init__(self, data: dict, config_path: Path | None, secrets: dict[str, str]):
        self.data = data
        self.config_path = config_path
        self.secrets = secrets

    # Convenience accessors -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def appdata(self) -> Path:
        return Path(self.data["paths"]["appdata"])

    @property
    def models(self) -> Path:
        return Path(self.data["paths"]["models"])

    @property
    def work(self) -> Path:
        return Path(self.data["paths"]["work"])

    @property
    def media_roots(self) -> list[dict]:
        return list(self.data.get("media") or [])

    def secrets_path(self) -> Path:
        return self.appdata / "secrets.yaml"

    # Settings page --------------------------------------------------------
    def user_overrides(self) -> dict:
        if self.config_path and self.config_path.exists():
            return yaml.safe_load(self.config_path.read_text()) or {}
        return {}

    def save_overrides(self, overrides: dict) -> None:
        if not self.config_path:
            raise RuntimeError("no config file path")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(overrides, sort_keys=False))
        tmp.replace(self.config_path)

    def save_secret(self, name: str, value: str) -> None:
        path = self.secrets_path()
        current = {}
        if path.exists():
            current = yaml.safe_load(path.read_text()) or {}
        if value:
            current[name] = value
        else:
            current.pop(name, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(current))
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        self.secrets = {k: str(v) for k, v in current.items()}

    def secret_names(self) -> list[str]:
        return sorted(k for k, v in self.secrets.items() if v)


def load(config_path: str | Path | None = None) -> Config:
    defaults = yaml.safe_load(DEFAULTS_FILE.read_text())
    path = Path(config_path or os.environ.get("AIDUB_CONFIG", "/config/config.yaml"))
    user = yaml.safe_load(path.read_text()) if path.exists() else {}
    merged = _merge(defaults, user or {})
    for key in ("appdata", "models", "work"):
        env = os.environ.get(f"AIDUB_{key.upper()}")
        if env:
            merged["paths"][key] = env

    secrets: dict[str, str] = {}
    secrets_file = Path(merged["paths"]["appdata"]) / "secrets.yaml"
    if secrets_file.exists():
        secrets = {k: str(v) for k, v in (yaml.safe_load(secrets_file.read_text()) or {}).items()}
    env = {**secrets, **{k: v for k, v in os.environ.items() if k.isupper()}}
    return Config(_substitute(merged, env), path, secrets)
