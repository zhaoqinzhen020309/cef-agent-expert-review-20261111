from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


QUICK_SCREEN_STATUS = "skipped_protocol_disabled"
NONINFERIORITY_TOLERANCE_PP = 0.01


@dataclass(frozen=True)
class EvaluationFingerprint:
    source_sha256: str
    evaluator_sha256: str
    manifest_sha256: str
    split: str
    instance_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    time_limit_sec: float
    iteration_limit: int
    metric_name: str
    metric_unit: str
    diagnostics_schema_sha256: str
    stage: str

    @property
    def cache_key(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EvaluationCache:
    """Content-addressed evaluation cache; mismatched protocols never reuse."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def load(self, fingerprint: EvaluationFingerprint) -> dict | None:
        path = self.root / f"{fingerprint.cache_key}.json"
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = json.loads(json.dumps(asdict(fingerprint)))
        if record.get("fingerprint") != expected:
            raise RuntimeError("evaluation cache fingerprint mismatch")
        return record["result"]

    def store(self, fingerprint: EvaluationFingerprint, result: dict) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{fingerprint.cache_key}.json"
        # JSON normalizes tuples to lists; compare the same representation on reuse.
        payload = json.loads(json.dumps({"fingerprint": asdict(fingerprint), "result": result}))
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError("evaluation cache key collision")
            return path
        _write_json(path, payload)
        return path


def quick_screen_record() -> dict[str, str]:
    return {
        "stage": "quick_screen",
        "status": QUICK_SCREEN_STATUS,
        "reason": (
            "The frozen retrospective found no deterministic subset with at least "
            "95% top-decile retention; no candidate may be eliminated here."
        ),
    }


def _write_json(path: Path, data: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
