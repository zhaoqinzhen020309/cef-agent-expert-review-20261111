"""Read local credentials without copying secrets into experiment artifacts."""
from __future__ import annotations

import os
from pathlib import Path


def load_deepseek_env(path: str | Path = ".env") -> dict[str, str]:
    values = {}
    source = Path(path)
    if source.is_file():
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError("local environment file requires KEY=value lines")
            values[key.strip().upper()] = value.strip().strip("\"'")
    key = (os.environ.get("DEEPSEEK_API_KEY") or values.get("DEEPSEEK_API_KEY")
           or values.get("APIKEY") or values.get("API_KEY"))
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is not configured")
    endpoint = (os.environ.get("DEEPSEEK_BASE_URL") or values.get("DEEPSEEK_BASE_URL")
                or values.get("BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
    return {"api_key": key, "base_url": endpoint,
            "model": os.environ.get("DEEPSEEK_MODEL") or values.get("DEEPSEEK_MODEL") or values.get("MODEL", "")}
