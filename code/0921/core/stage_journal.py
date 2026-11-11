from __future__ import annotations

import json
import time
from pathlib import Path


STAGE_ORDER = (
    "generation",
    "static_check",
    "quick_screen",
    "development_evaluation",
    "evidence_validation",
)
TERMINAL_STAGE_STATUSES = {"complete", "skipped", "rejected"}
CLOSURE_GATE_ORDER = (
    "usage_completeness",
    "planned_budget",
    "evaluation",
    "evidence",
    "invalidity_resource_gate",
    "round_complete",
)
ROUND_ATTEMPT_STAGE_ORDER = (
    "round_created",
    "controller_returned",
    "plan_validated",
    "candidates_materialized",
    "hypotheses_locked",
    "round_closed",
)
_REPLACE_RETRIES = 8
_REPLACE_DELAY_SECONDS = 0.05


class CandidateStageJournal:
    """Atomic, fingerprint-checked candidate stage recovery ledger."""

    def __init__(self, path: str | Path, *, candidate_id: str) -> None:
        self.path = Path(path).resolve()
        self.candidate_id = candidate_id
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("candidate_id") != candidate_id:
                raise RuntimeError("stage journal candidate id mismatch")
        else:
            self.data = {"candidate_id": candidate_id, "stages": {}}

    def begin(self, stage: str, *, input_fingerprint: str) -> bool:
        self._require_stage(stage)
        existing = self.data["stages"].get(stage)
        if existing is not None:
            if existing.get("input_fingerprint") != input_fingerprint:
                raise RuntimeError(f"{stage} input fingerprint changed")
            if existing.get("status") in TERMINAL_STAGE_STATUSES:
                return False
        self._require_predecessors(stage)
        attempts = int(existing.get("attempts", 0)) + 1 if existing else 1
        self.data["stages"][stage] = {
            "status": "running",
            "input_fingerprint": input_fingerprint,
            "output_fingerprint": None,
            "attempts": attempts,
            "metadata": {},
        }
        self._save()
        return True

    def finish(
        self,
        stage: str,
        *,
        input_fingerprint: str,
        output_fingerprint: str,
        status: str = "complete",
        metadata: dict | None = None,
    ) -> None:
        if status not in TERMINAL_STAGE_STATUSES:
            raise ValueError("stage terminal status is invalid")
        existing = self.data["stages"].get(stage)
        if existing is not None and existing.get("status") in TERMINAL_STAGE_STATUSES:
            expected = {
                "status": status,
                "input_fingerprint": input_fingerprint,
                "output_fingerprint": output_fingerprint,
                "metadata": dict(metadata or {}),
            }
            if all(existing.get(key) == value for key, value in expected.items()):
                return
            raise RuntimeError(f"{stage} terminal record changed")
        if existing is None or existing.get("status") != "running":
            raise RuntimeError(f"{stage} was not started")
        if existing.get("input_fingerprint") != input_fingerprint:
            raise RuntimeError(f"{stage} input fingerprint changed")
        existing.update({
            "status": status,
            "output_fingerprint": output_fingerprint,
            "metadata": dict(metadata or {}),
        })
        self._save()

    def record(self, stage: str) -> dict | None:
        value = self.data["stages"].get(stage)
        return dict(value) if value is not None else None

    def _require_stage(self, stage: str) -> None:
        if stage not in STAGE_ORDER:
            raise ValueError(f"unknown candidate stage: {stage}")

    def _require_predecessors(self, stage: str) -> None:
        for predecessor in STAGE_ORDER[: STAGE_ORDER.index(stage)]:
            record = self.data["stages"].get(predecessor)
            if record is None or record.get("status") not in TERMINAL_STAGE_STATUSES:
                raise RuntimeError(
                    f"cannot start {stage} before terminal {predecessor}"
                )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _replace_with_transient_windows_retry(temporary, self.path)


class RoundClosureJournal:
    """Atomic ordered ledger for restart-safe round closure gates."""

    def __init__(self, path: str | Path, *, round_index: int) -> None:
        self.path = Path(path).resolve()
        self.round_index = round_index
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("round_index") != round_index:
                raise RuntimeError("round closure journal index mismatch")
        else:
            self.data = {"round_index": round_index, "gates": {}}

    def begin(self, gate: str, *, input_fingerprint: str) -> bool:
        self._require_gate(gate)
        existing = self.data["gates"].get(gate)
        if existing is not None:
            if existing.get("input_fingerprint") != input_fingerprint:
                raise RuntimeError(f"{gate} input fingerprint changed")
            if existing.get("status") == "complete":
                return False
            if existing.get("status") != "running":
                raise RuntimeError(f"{gate} gate status is invalid")
            return True
        self._require_predecessors(gate)
        self.data["gates"][gate] = {
            "status": "running",
            "input_fingerprint": input_fingerprint,
            "output_fingerprint": None,
            "metadata": {},
        }
        self._save()
        return True

    def finish(
        self,
        gate: str,
        *,
        input_fingerprint: str,
        output_fingerprint: str,
        metadata: dict | None = None,
    ) -> None:
        self._require_gate(gate)
        existing = self.data["gates"].get(gate)
        expected = {
            "status": "complete",
            "input_fingerprint": input_fingerprint,
            "output_fingerprint": output_fingerprint,
            "metadata": dict(metadata or {}),
        }
        if existing is not None and existing.get("status") == "complete":
            if all(existing.get(key) == value for key, value in expected.items()):
                return
            raise RuntimeError(f"{gate} terminal record changed")
        if existing is None or existing.get("status") != "running":
            raise RuntimeError(f"{gate} was not started")
        if existing.get("input_fingerprint") != input_fingerprint:
            raise RuntimeError(f"{gate} input fingerprint changed")
        existing.update(expected)
        self._save()

    def record(self, gate: str) -> dict | None:
        self._require_gate(gate)
        value = self.data["gates"].get(gate)
        return dict(value) if value is not None else None

    def records(self) -> list[dict]:
        return [
            {"name": gate, **self.data["gates"][gate]}
            for gate in CLOSURE_GATE_ORDER
            if gate in self.data["gates"]
        ]

    def _require_gate(self, gate: str) -> None:
        if gate not in CLOSURE_GATE_ORDER:
            raise ValueError(f"unknown round closure gate: {gate}")

    def _require_predecessors(self, gate: str) -> None:
        for predecessor in CLOSURE_GATE_ORDER[: CLOSURE_GATE_ORDER.index(gate)]:
            record = self.data["gates"].get(predecessor)
            if record is None or record.get("status") != "complete":
                raise RuntimeError(
                    f"cannot start {gate} before terminal {predecessor}"
                )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _replace_with_transient_windows_retry(temporary, self.path)


def _replace_with_transient_windows_retry(source: Path, target: Path) -> None:
    """Replace a journal atomically, tolerating short Windows sharing locks."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            error_number = getattr(exc, "winerror", None) or getattr(exc, "errno", None)
            if error_number != 5 or attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_DELAY_SECONDS * (attempt + 1))


class RoundAttemptJournal:
    """Append-only transaction milestones for one planner round."""

    def __init__(self, path: str | Path, *, round_index: int) -> None:
        self.path = Path(path).resolve()
        self.round_index = round_index
        if self.path.is_file():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("round_index") != round_index:
                raise RuntimeError("round attempt journal index mismatch")
            if not isinstance(self.data.get("events"), list):
                raise RuntimeError("round attempt journal events are invalid")
        else:
            self.data = {"round_index": round_index, "events": []}

    def append(
        self,
        stage: str,
        *,
        fingerprint: str,
        metadata: dict | None = None,
    ) -> None:
        if stage not in ROUND_ATTEMPT_STAGE_ORDER:
            raise ValueError(f"unknown round attempt stage: {stage}")
        expected = {
            "stage": stage,
            "fingerprint": fingerprint,
            "metadata": dict(metadata or {}),
        }
        existing = next(
            (event for event in self.data["events"] if event.get("stage") == stage),
            None,
        )
        if existing is not None:
            if existing != expected:
                raise RuntimeError(f"round attempt stage changed: {stage}")
            return
        required = ROUND_ATTEMPT_STAGE_ORDER[: ROUND_ATTEMPT_STAGE_ORDER.index(stage)]
        recorded = {event.get("stage") for event in self.data["events"]}
        missing = [item for item in required if item not in recorded]
        if missing:
            raise RuntimeError(
                f"round attempt stage {stage} is missing predecessors: {missing}"
            )
        self.data["events"].append(expected)
        self._save()

    def stages(self) -> tuple[str, ...]:
        return tuple(str(event["stage"]) for event in self.data["events"])

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _replace_with_transient_windows_retry(temporary, self.path)
