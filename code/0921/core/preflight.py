from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Iterable

from agent_ahd.core.protocol_versions import PROTOCOL_VERSIONS


def file_sha256(path: str | Path) -> str:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_root_contract(
    *,
    config: object,
    source_files: object,
    non_blind_manifest: object,
    environment: object,
    blind_commitment: str | None = None,
) -> dict:
    contract = {
        "protocol_versions": dict(PROTOCOL_VERSIONS),
        "config": config,
        "source_files": source_files,
        "non_blind_manifest": non_blind_manifest,
        "environment": environment,
    }
    if blind_commitment is not None:
        contract["blind_commitment"] = str(blind_commitment)
    contract["root_hash"] = _payload_hash(contract)
    return contract


def validate_root_contract(contract: dict) -> bool:
    if not isinstance(contract, dict):
        raise ValueError("root contract must be an object")
    versions = contract.get("protocol_versions")
    if versions != dict(PROTOCOL_VERSIONS):
        raise ValueError("root contract protocol versions are unsupported")
    root_hash = contract.get("root_hash")
    if not isinstance(root_hash, str) or root_hash != _payload_hash(
        {key: value for key, value in contract.items() if key != "root_hash"}
    ):
        raise ValueError("root contract hash mismatch")
    return True


def assert_new_output_root(
    output_root: str | Path,
    *,
    historical_roots: Iterable[str | Path],
) -> Path:
    output = Path(output_root).resolve()
    historical = tuple(Path(item).resolve() for item in historical_roots)
    for root in historical:
        if _overlaps(output, root):
            raise ValueError(
                f"new output root overlaps protected historical root: {output}"
            )
    if output.exists() and not output.is_dir():
        raise ValueError(f"output root is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()):
        raise FileExistsError(f"new output root must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def collect_environment_fingerprint() -> dict[str, str | int | None]:
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": numpy_version,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def verify_dependency_lock(path: str | Path) -> dict[str, str]:
    lock_path = Path(path).resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    expected = {}
    for line_number, raw in enumerate(
        lock_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError(
                f"dependency lock line {line_number} is not exactly pinned"
            )
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            raise ValueError(f"dependency lock line {line_number} is incomplete")
        expected[name] = version
    if not expected:
        raise ValueError("dependency lock is empty")
    mismatches = {}
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != version:
            mismatches[name] = f"expected={version}, actual={actual}"
    if mismatches:
        raise RuntimeError(f"dependency lock mismatch: {mismatches}")
    return expected


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)
