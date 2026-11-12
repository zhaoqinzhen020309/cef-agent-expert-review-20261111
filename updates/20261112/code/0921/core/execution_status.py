from __future__ import annotations

import re
import json
from collections.abc import Mapping
from pathlib import Path


ARM_EXECUTION_STATUSES = {
    "completed",
    "terminated_by_protocol_gate",
    "failed",
    "interrupted",
    "stopped_without_selection",
}
COMPARISON_COMPLETENESS = {"complete", "partial", "failed"}
DISPATCHER_LIFECYCLES = {
    "initialized",
    "running",
    "all_arms_frozen",
    "failed",
}
_LEGAL_PROTOCOL_GATE_CATEGORIES = {"quality", "resource"}
_FAILURE_PROTOCOL_GATE_TOKENS = {"accounting", "integrity"}
_CANONICAL_REASON_KIND = re.compile(
    r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
)


def derive_execution_status(
    *,
    completed_round_count: int,
    total_rounds: int,
    protocol_gate: dict | None,
    stopped_reason: str | None,
    failure: dict | None = None,
    abandoned: bool = False,
) -> str | None:
    if type(completed_round_count) is not int or type(total_rounds) is not int:
        raise TypeError("completed_round_count and total_rounds must be integers")
    if total_rounds <= 0:
        raise ValueError("total_rounds must be positive")
    if completed_round_count < 0:
        raise ValueError("completed_round_count must be nonnegative")
    if completed_round_count > total_rounds:
        raise ValueError("completed_round_count cannot exceed total_rounds")

    if failure is not None:
        return "failed"
    if protocol_gate is not None:
        _validate_protocol_gate(protocol_gate)
        return "terminated_by_protocol_gate"
    if stopped_reason is not None:
        return "stopped_without_selection"
    if completed_round_count == total_rounds:
        return "completed"
    if abandoned:
        return "interrupted"
    return None


def _validate_protocol_gate(protocol_gate: dict) -> None:
    if not isinstance(protocol_gate, dict):
        raise ValueError("protocol gate must be a dictionary")

    markers = {}
    for field in ("category", "reason_kind"):
        if field not in protocol_gate:
            continue
        value = protocol_gate[field]
        if type(value) is not str or not value:
            raise ValueError(f"protocol gate {field} must be a non-empty string")
        markers[field] = value

    category = markers.get("category")
    if category is not None and category not in _LEGAL_PROTOCOL_GATE_CATEGORIES:
        raise ValueError(f"unknown protocol gate category: {category!r}")

    reason_kind = markers.get("reason_kind")
    if reason_kind is not None and _CANONICAL_REASON_KIND.fullmatch(reason_kind) is None:
        raise ValueError(
            "protocol gate reason_kind must be canonical lowercase snake_case"
        )

    if any(
        _FAILURE_PROTOCOL_GATE_TOKENS.intersection(marker.split("_"))
        for marker in markers.values()
    ):
        raise ValueError("protocol gate cannot record accounting or integrity failure")

    reason_category = next(
        (
            candidate
            for candidate in _LEGAL_PROTOCOL_GATE_CATEGORIES
            if reason_kind == candidate
            or (reason_kind is not None and reason_kind.startswith(f"{candidate}_"))
        ),
        None,
    )
    if category is None and reason_category is None:
        raise ValueError("protocol gate must identify a quality or resource termination")
    if category is not None and reason_category is not None and category != reason_category:
        raise ValueError("protocol gate category conflicts with reason_kind")


def legacy_comparison_completeness(
    statuses: tuple[str | None, ...],
) -> str | None:
    unknown = {status for status in statuses if status is not None} - ARM_EXECUTION_STATUSES
    if unknown:
        raise ValueError(f"unknown arm execution statuses: {sorted(unknown)}")
    if not statuses:
        return "failed"
    if any(status is None for status in statuses):
        return None
    if "failed" in statuses:
        return "failed"
    if all(status == "completed" for status in statuses):
        return "complete"
    return "partial"


def comparison_completeness(
    *,
    required_arms: tuple[str, ...],
    sealed_arm_statuses: Mapping[str, str],
) -> str | None:
    if not required_arms:
        raise ValueError("required arms must not be empty")
    for arm in required_arms:
        _validate_arm_name(arm, field="required arm names")
    if len(set(required_arms)) != len(required_arms):
        raise ValueError("required arms must be unique")

    required = set(required_arms)
    for arm in sealed_arm_statuses:
        _validate_arm_name(arm, field="sealed arm names")
    extra = set(sealed_arm_statuses) - required
    if extra:
        raise ValueError(f"extra sealed arms: {sorted(extra)}")

    for arm, status in sealed_arm_statuses.items():
        if status not in ARM_EXECUTION_STATUSES:
            raise ValueError(f"invalid sealed arm status for {arm!r}: {status!r}")

    if required - set(sealed_arm_statuses):
        return None

    statuses_by_required_arm = tuple(
        sealed_arm_statuses[arm] for arm in required_arms
    )
    if "failed" in statuses_by_required_arm:
        return "failed"
    if all(status == "completed" for status in statuses_by_required_arm):
        return "complete"
    return "partial"


def dispatcher_lifecycle(record: Mapping[str, object]) -> str | None:
    """Read strict lifecycle state, falling back only for legacy records."""
    if "dispatcher_lifecycle" in record:
        value = record["dispatcher_lifecycle"]
        if type(value) is not str or value not in DISPATCHER_LIFECYCLES:
            raise ValueError("invalid dispatcher lifecycle")
        return value
    legacy = record.get("status")
    if legacy is None:
        return None
    if type(legacy) is not str:
        raise ValueError("legacy comparison status must be a string")
    return legacy


def write_execution_status_seal(
    path: str | Path,
    *,
    execution_status: str,
    rounds_complete: int,
    total_rounds: int,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict:
    """Publish one immutable terminal arm status record."""
    if execution_status not in ARM_EXECUTION_STATUSES:
        raise ValueError("invalid arm execution status")
    if type(rounds_complete) is not int or type(total_rounds) is not int:
        raise TypeError("arm round counts must be integers")
    if total_rounds <= 0 or not 0 <= rounds_complete <= total_rounds:
        raise ValueError("arm round counts are invalid")
    if execution_status == "completed" and rounds_complete != total_rounds:
        raise ValueError("completed arm seal requires every round to be closed")
    if execution_status == "failed":
        if not error_type or not error_message:
            raise ValueError("failed arm seal requires error details")
    elif error_type is not None or error_message is not None:
        raise ValueError("nonfailed arm seal cannot contain error details")
    record = {
        "version": 1,
        "execution_status": execution_status,
        "rounds_complete": rounds_complete,
        "total_rounds": total_rounds,
    }
    if execution_status == "failed":
        record.update({"error_type": error_type, "error_message": error_message})
    target = Path(path)
    if target.is_file():
        existing = load_execution_status_seal(target)
        if existing != record:
            raise RuntimeError("immutable arm execution status changed")
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)
    return record


def load_execution_status_seal(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("arm execution status must be an object")
    required = {"version", "execution_status", "rounds_complete", "total_rounds"}
    status = record.get("execution_status")
    allowed = set(required)
    if status == "failed":
        allowed.update({"error_type", "error_message"})
    if set(record) != allowed or not required.issubset(record):
        raise ValueError("arm execution status fields are invalid")
    if record.get("version") != 1 or status not in ARM_EXECUTION_STATUSES:
        raise ValueError("arm execution status identity is invalid")
    rounds_complete = record.get("rounds_complete")
    total_rounds = record.get("total_rounds")
    if type(rounds_complete) is not int or type(total_rounds) is not int:
        raise TypeError("arm round counts must be integers")
    if total_rounds <= 0 or not 0 <= rounds_complete <= total_rounds:
        raise ValueError("arm round counts are invalid")
    if status == "completed" and rounds_complete != total_rounds:
        raise ValueError("completed arm seal requires every round to be closed")
    if status == "failed":
        if not isinstance(record.get("error_type"), str) or not record["error_type"]:
            raise ValueError("failed arm error type is invalid")
        if not isinstance(record.get("error_message"), str) or not record["error_message"]:
            raise ValueError("failed arm error message is invalid")
    return record


def _validate_arm_name(arm: object, *, field: str) -> None:
    if type(arm) is not str or not arm or arm != arm.strip():
        raise ValueError(f"{field} must be non-empty strings without surrounding whitespace")
