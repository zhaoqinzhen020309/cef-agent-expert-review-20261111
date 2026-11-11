from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


ATTEMPT_LEDGER_FILENAME = "attempt_ledger.jsonl"
LEDGER_VERSION = 1
USAGE_STATUSES = {"reported", "reported_zero", "unavailable", "estimated"}
TERMINAL_STATUSES = {"returned", "failed"}
ATTEMPT_KINDS = {"primary", "recovery", "repair"}


class LedgerIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    model_calls: int
    model_tokens: int
    wall_clock_sec: float

    def __post_init__(self) -> None:
        _require_positive_int(self.model_calls, "model_calls")
        _require_positive_int(self.model_tokens, "model_tokens")
        _require_positive_float(self.wall_clock_sec, "wall_clock_sec")


@dataclass(frozen=True)
class BudgetReservation:
    model_calls: int
    model_tokens: int
    wall_clock_sec: float

    def __post_init__(self) -> None:
        _require_positive_int(self.model_calls, "model_calls")
        _require_positive_int(self.model_tokens, "model_tokens")
        _require_positive_float(self.wall_clock_sec, "wall_clock_sec")


@dataclass(frozen=True)
class LedgerSummary:
    used_model_calls: int
    used_model_tokens: int
    used_wall_clock_sec: float
    active_reserved_model_calls: int
    active_reserved_model_tokens: int
    active_reserved_wall_clock_sec: float


_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}


class AttemptLedger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / ATTEMPT_LEDGER_FILENAME
        self.lock_path = self.run_dir / ".attempt-ledger.lock"

    def records(self) -> list[dict]:
        with self._locked():
            return self._read_records_unlocked()

    def reserve_group(
        self,
        *,
        reservation_group_id: str,
        reservation: BudgetReservation,
        limits: BudgetLimits,
    ) -> None:
        _require_text(reservation_group_id, "reservation_group_id")
        with self._locked():
            state = self._state_unlocked()
            expected = {
                "reservation_group_id": reservation_group_id,
                "reservation": _budget_dict(reservation),
                "limits": _budget_dict(limits),
            }
            existing = state["groups"].get(reservation_group_id)
            if existing is not None:
                if existing["reservation_payload"] != expected:
                    raise LedgerIntegrityError("reservation group changed")
                return
            summary = state["summary"]
            committed = (
                summary.used_model_calls + summary.active_reserved_model_calls,
                summary.used_model_tokens + summary.active_reserved_model_tokens,
                summary.used_wall_clock_sec + summary.active_reserved_wall_clock_sec,
            )
            requested = (
                reservation.model_calls,
                reservation.model_tokens,
                reservation.wall_clock_sec,
            )
            maximum = (limits.model_calls, limits.model_tokens, limits.wall_clock_sec)
            if any(
                used + added > limit
                for used, added, limit in zip(committed, requested, maximum)
            ):
                raise RuntimeError("model budget cannot reserve the call group")
            self._append_unlocked("reservation_group_created", expected)

    def reserve(
        self,
        *,
        logical_call_id: str,
        role: str,
        reservation: BudgetReservation,
        limits: BudgetLimits,
        reservation_group_id: str | None = None,
    ) -> None:
        _require_text(logical_call_id, "logical_call_id")
        _require_text(role, "role")
        if reservation_group_id is not None:
            _require_text(reservation_group_id, "reservation_group_id")
        with self._locked():
            state = self._state_unlocked()
            existing = state["calls"].get(logical_call_id)
            expected = {
                "logical_call_id": logical_call_id,
                "role": role,
                "reservation": _budget_dict(reservation),
                "limits": _budget_dict(limits),
            }
            if reservation_group_id is not None:
                expected["reservation_group_id"] = reservation_group_id
            if existing is not None:
                if existing["reservation_payload"] != expected:
                    raise LedgerIntegrityError("logical call reservation changed")
                return
            if reservation_group_id is None:
                summary = state["summary"]
                committed = (
                    summary.used_model_calls + summary.active_reserved_model_calls,
                    summary.used_model_tokens + summary.active_reserved_model_tokens,
                    summary.used_wall_clock_sec + summary.active_reserved_wall_clock_sec,
                )
                requested = (
                    reservation.model_calls,
                    reservation.model_tokens,
                    reservation.wall_clock_sec,
                )
                maximum = (limits.model_calls, limits.model_tokens, limits.wall_clock_sec)
                if any(
                    used + added > limit
                    for used, added, limit in zip(committed, requested, maximum)
                ):
                    raise RuntimeError("model budget cannot reserve the logical call")
            else:
                group = state["groups"].get(reservation_group_id)
                if group is None or group["released"]:
                    raise RuntimeError("reservation group is unavailable")
                requested = _budget_tuple(reservation)
                claimed = group["claimed"]
                maximum = _budget_tuple(group["reservation"])
                if any(
                    used + added > limit
                    for used, added, limit in zip(claimed, requested, maximum)
                ):
                    raise RuntimeError("reservation group has insufficient capacity")
            self._append_unlocked("reservation_created", expected)

    def start_attempt(
        self,
        *,
        logical_call_id: str,
        attempt: int,
        attempt_kind: str | None = None,
    ) -> None:
        with self._locked():
            state = self._state_unlocked()
            call = _require_call(state, logical_call_id)
            if call["completed"] or call["released"]:
                raise RuntimeError("logical call is already closed")
            if call["active_attempt"] is not None:
                raise RuntimeError("logical call already has an active attempt")
            if attempt != len(call["terminals"]):
                raise RuntimeError("attempt sequence is not contiguous")
            kind = _resolve_attempt_kind(attempt, attempt_kind)
            _validate_attempt_sequence(call["terminals"], attempt, kind)
            if attempt >= call["reservation"].model_calls:
                raise RuntimeError("attempt exceeds reserved model-call budget")
            self._append_unlocked(
                "attempt_dispatched",
                {
                    "logical_call_id": logical_call_id,
                    "attempt": attempt,
                    "attempt_kind": kind,
                },
            )

    def finish_attempt(
        self,
        *,
        logical_call_id: str,
        attempt: int,
        terminal_status: str,
        usage_status: str,
        model_tokens: int,
        wall_clock_sec: float,
        failure_class: str | None = None,
        reason_kind: str | None = None,
        usage: dict | None = None,
    ) -> None:
        if terminal_status not in TERMINAL_STATUSES:
            raise ValueError("invalid attempt terminal status")
        if usage_status not in USAGE_STATUSES:
            raise ValueError("invalid usage status")
        if type(model_tokens) is not int or model_tokens < 0:
            raise ValueError("model_tokens must be a nonnegative integer")
        if not isinstance(wall_clock_sec, (int, float)) or isinstance(wall_clock_sec, bool):
            raise ValueError("wall_clock_sec must be numeric")
        if not math.isfinite(wall_clock_sec) or wall_clock_sec < 0:
            raise ValueError("wall_clock_sec must be finite and nonnegative")
        if terminal_status == "failed":
            _require_text(failure_class, "failure_class")
            _require_text(reason_kind, "reason_kind")
        elif failure_class is not None or reason_kind is not None:
            raise ValueError("returned attempt cannot record a failure")
        with self._locked():
            state = self._state_unlocked()
            call = _require_call(state, logical_call_id)
            if call["active_attempt"] != attempt:
                raise RuntimeError("attempt terminal does not match active dispatch")
            attempt_kind = call["active_attempt_kind"]
            used_tokens = sum(item["model_tokens"] for item in call["terminals"])
            used_wall = sum(item["wall_clock_sec"] for item in call["terminals"])
            if used_tokens + model_tokens > call["reservation"].model_tokens:
                raise RuntimeError("attempt exceeds reserved model-token budget")
            if used_wall + wall_clock_sec > call["reservation"].wall_clock_sec:
                raise RuntimeError("attempt exceeds reserved wall-clock budget")
            self._append_unlocked(
                "attempt_terminal",
                {
                    "logical_call_id": logical_call_id,
                    "attempt": attempt,
                    "attempt_kind": attempt_kind,
                    "terminal_status": terminal_status,
                    "usage_status": usage_status,
                    "model_tokens": model_tokens,
                    "wall_clock_sec": float(wall_clock_sec),
                    "usage": _normalize_usage(usage, model_tokens=model_tokens),
                    "failure_class": failure_class,
                    "reason_kind": reason_kind,
                },
            )

    def complete_logical_call(self, *, logical_call_id: str, outcome: str) -> None:
        _require_text(outcome, "outcome")
        with self._locked():
            state = self._state_unlocked()
            call = _require_call(state, logical_call_id)
            if call["completed"]:
                return
            if call["active_attempt"] is not None or not call["terminals"]:
                raise RuntimeError("logical call has no terminal attempt")
            self._append_unlocked(
                "logical_call_completed",
                {"logical_call_id": logical_call_id, "outcome": outcome},
            )

    def release_reservation(self, *, logical_call_id: str) -> None:
        with self._locked():
            state = self._state_unlocked()
            call = _require_call(state, logical_call_id)
            if call["released"]:
                return
            if not call["completed"]:
                raise RuntimeError("cannot release an incomplete logical call")
            self._append_unlocked(
                "reservation_released",
                {"logical_call_id": logical_call_id},
            )

    def release_group(self, *, reservation_group_id: str) -> None:
        _require_text(reservation_group_id, "reservation_group_id")
        with self._locked():
            state = self._state_unlocked()
            try:
                group = state["groups"][reservation_group_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"reservation group does not exist: {reservation_group_id}"
                ) from exc
            if group["released"]:
                return
            unfinished = [
                logical_call_id
                for logical_call_id in group["logical_call_ids"]
                if not state["calls"][logical_call_id]["released"]
            ]
            if unfinished:
                raise RuntimeError(
                    "cannot release reservation group with unfinished logical calls"
                )
            self._append_unlocked(
                "reservation_group_released",
                {"reservation_group_id": reservation_group_id},
            )

    def summary(self) -> LedgerSummary:
        with self._locked():
            return self._state_unlocked()["summary"]

    def record_dispatcher_concurrency(self, payload: dict) -> None:
        _validate_dispatcher_concurrency_payload(payload)
        with self._locked():
            self._state_unlocked()
            self._append_unlocked("dispatcher_concurrency_observed", payload)

    def call_state(self, logical_call_id: str) -> dict | None:
        with self._locked():
            call = self._state_unlocked()["calls"].get(logical_call_id)
            if call is None:
                return None
            return {
                "role": call["role"],
                "attempts": len(call["terminals"]),
                "active_attempt": call["active_attempt"],
                "active_attempt_kind": call["active_attempt_kind"],
                "attempt_kinds": [
                    item["attempt_kind"] for item in call["terminals"]
                ],
                "terminals": [dict(item) for item in call["terminals"]],
                "completed": call["completed"],
                "released": call["released"],
                "outcome": call["outcome"],
            }

    def _state_unlocked(self) -> dict:
        calls: dict[str, dict] = {}
        groups: dict[str, dict] = {}
        for record in self._read_records_unlocked():
            payload = record["payload"]
            event_type = record["event_type"]
            logical_call_id = payload.get("logical_call_id")
            if event_type == "dispatcher_concurrency_observed":
                _validate_dispatcher_concurrency_payload(payload)
                continue
            if event_type == "reservation_group_created":
                reservation_group_id = payload.get("reservation_group_id")
                _require_text(reservation_group_id, "reservation_group_id")
                if reservation_group_id in groups:
                    raise LedgerIntegrityError("duplicate reservation group")
                reservation = BudgetReservation(**payload["reservation"])
                BudgetLimits(**payload["limits"])
                groups[reservation_group_id] = {
                    "reservation": reservation,
                    "reservation_payload": payload,
                    "claimed": (0, 0, 0.0),
                    "logical_call_ids": [],
                    "released": False,
                }
                continue
            if event_type == "reservation_group_released":
                reservation_group_id = payload.get("reservation_group_id")
                try:
                    group = groups[reservation_group_id]
                except (KeyError, TypeError) as exc:
                    raise LedgerIntegrityError(
                        "reservation group release has no matching group"
                    ) from exc
                if group["released"]:
                    raise LedgerIntegrityError("duplicate reservation group release")
                if any(
                    not calls[logical_call_id]["released"]
                    for logical_call_id in group["logical_call_ids"]
                ):
                    raise LedgerIntegrityError(
                        "reservation group released with unfinished logical calls"
                    )
                group["released"] = True
                continue
            if event_type == "reservation_created":
                if logical_call_id in calls:
                    raise LedgerIntegrityError("duplicate logical call reservation")
                reservation = BudgetReservation(**payload["reservation"])
                BudgetLimits(**payload["limits"])
                reservation_group_id = payload.get("reservation_group_id")
                if reservation_group_id is not None:
                    try:
                        group = groups[reservation_group_id]
                    except (KeyError, TypeError) as exc:
                        raise LedgerIntegrityError(
                            "logical call references an unknown reservation group"
                        ) from exc
                    if group["released"]:
                        raise LedgerIntegrityError(
                            "logical call references a released reservation group"
                        )
                    claimed = tuple(
                        used + added
                        for used, added in zip(
                            group["claimed"],
                            _budget_tuple(reservation),
                        )
                    )
                    if any(
                        used > limit
                        for used, limit in zip(
                            claimed,
                            _budget_tuple(group["reservation"]),
                        )
                    ):
                        raise LedgerIntegrityError(
                            "logical calls exceed reservation group capacity"
                        )
                    group["claimed"] = claimed
                    group["logical_call_ids"].append(logical_call_id)
                calls[logical_call_id] = {
                    "role": payload["role"],
                    "reservation": reservation,
                    "reservation_payload": payload,
                    "active_attempt": None,
                    "active_attempt_kind": None,
                    "terminals": [],
                    "completed": False,
                    "released": False,
                    "outcome": None,
                    "reservation_group_id": reservation_group_id,
                }
                continue
            call = _require_call({"calls": calls}, logical_call_id)
            if event_type == "attempt_dispatched":
                attempt = payload["attempt"]
                if call["completed"] or call["released"] or call["active_attempt"] is not None:
                    raise LedgerIntegrityError("invalid attempt dispatch transition")
                if attempt != len(call["terminals"]):
                    raise LedgerIntegrityError("invalid attempt sequence")
                try:
                    attempt_kind = _resolve_attempt_kind(
                        attempt,
                        payload.get("attempt_kind"),
                    )
                    _validate_attempt_sequence(
                        call["terminals"],
                        attempt,
                        attempt_kind,
                    )
                except (RuntimeError, ValueError) as exc:
                    raise LedgerIntegrityError(str(exc)) from exc
                call["active_attempt"] = attempt
                call["active_attempt_kind"] = attempt_kind
            elif event_type == "attempt_terminal":
                if call["active_attempt"] != payload["attempt"]:
                    raise LedgerIntegrityError("attempt terminal has no matching dispatch")
                terminal_kind = payload.get(
                    "attempt_kind",
                    call["active_attempt_kind"],
                )
                if terminal_kind != call["active_attempt_kind"]:
                    raise LedgerIntegrityError("attempt terminal kind changed")
                terminal = dict(payload)
                terminal["attempt_kind"] = terminal_kind
                call["terminals"].append(terminal)
                call["active_attempt"] = None
                call["active_attempt_kind"] = None
            elif event_type == "logical_call_completed":
                if call["completed"] or call["active_attempt"] is not None or not call["terminals"]:
                    raise LedgerIntegrityError("invalid logical call completion")
                call["completed"] = True
                call["outcome"] = payload["outcome"]
            elif event_type == "reservation_released":
                if call["released"] or not call["completed"]:
                    raise LedgerIntegrityError("invalid reservation release")
                call["released"] = True
            else:
                raise LedgerIntegrityError(f"unknown ledger event type: {event_type}")

        used_calls = sum(len(call["terminals"]) for call in calls.values())
        used_tokens = sum(
            terminal["model_tokens"]
            for call in calls.values()
            for terminal in call["terminals"]
        )
        used_wall = sum(
            terminal["wall_clock_sec"]
            for call in calls.values()
            for terminal in call["terminals"]
        )
        active_calls = active_tokens = 0
        active_wall = 0.0
        for call in calls.values():
            if call["reservation_group_id"] is not None:
                continue
            if call["released"]:
                continue
            reservation = call["reservation"]
            call_used = len(call["terminals"])
            token_used = sum(item["model_tokens"] for item in call["terminals"])
            wall_used = sum(item["wall_clock_sec"] for item in call["terminals"])
            active_calls += reservation.model_calls - call_used
            active_tokens += reservation.model_tokens - token_used
            active_wall += reservation.wall_clock_sec - wall_used
        for group in groups.values():
            if group["released"]:
                continue
            reservation = group["reservation"]
            group_calls = [calls[item] for item in group["logical_call_ids"]]
            active_calls += reservation.model_calls - sum(
                len(call["terminals"]) for call in group_calls
            )
            active_tokens += reservation.model_tokens - sum(
                terminal["model_tokens"]
                for call in group_calls
                for terminal in call["terminals"]
            )
            active_wall += reservation.wall_clock_sec - sum(
                terminal["wall_clock_sec"]
                for call in group_calls
                for terminal in call["terminals"]
            )
        return {
            "calls": calls,
            "groups": groups,
            "summary": LedgerSummary(
                used_model_calls=used_calls,
                used_model_tokens=used_tokens,
                used_wall_clock_sec=used_wall,
                active_reserved_model_calls=active_calls,
                active_reserved_model_tokens=active_tokens,
                active_reserved_wall_clock_sec=active_wall,
            ),
        }

    def _append_unlocked(self, event_type: str, payload: dict) -> None:
        records = self._read_records_unlocked()
        previous = records[-1]["record_sha256"] if records else None
        record = {
            "ledger_version": LEDGER_VERSION,
            "sequence": len(records) + 1,
            "event_type": event_type,
            "recorded_at_epoch_sec": time.time(),
            "previous_sha256": previous,
            "payload": payload,
        }
        record["record_sha256"] = _record_sha256(record)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _read_records_unlocked(self) -> list[dict]:
        if not self.path.is_file():
            return []
        records = []
        previous = None
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise LedgerIntegrityError(f"cannot read attempt ledger: {exc}") from exc
        for index, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerIntegrityError(f"attempt ledger line {index} is invalid JSON") from exc
            if not isinstance(record, dict):
                raise LedgerIntegrityError("attempt ledger record must be an object")
            expected_fields = {
                "ledger_version",
                "sequence",
                "event_type",
                "recorded_at_epoch_sec",
                "previous_sha256",
                "payload",
                "record_sha256",
            }
            if set(record) != expected_fields:
                raise LedgerIntegrityError("attempt ledger record fields changed")
            if record["ledger_version"] != LEDGER_VERSION or record["sequence"] != index:
                raise LedgerIntegrityError("attempt ledger sequence or version changed")
            if record["previous_sha256"] != previous:
                raise LedgerIntegrityError("attempt ledger hash chain changed")
            if record["record_sha256"] != _record_sha256(record):
                raise LedgerIntegrityError("attempt ledger record hash changed")
            previous = record["record_sha256"]
            records.append(record)
        return records

    @contextmanager
    def _locked(self):
        key = os.path.normcase(str(self.path))
        with _LOCKS_GUARD:
            process_lock = _PROCESS_LOCKS.setdefault(key, threading.RLock())
        with process_lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            handle = self.lock_path.open("a+b")
            if self.lock_path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()


def _record_sha256(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _budget_dict(value: BudgetLimits | BudgetReservation) -> dict:
    return {
        "model_calls": value.model_calls,
        "model_tokens": value.model_tokens,
        "wall_clock_sec": value.wall_clock_sec,
    }


def _budget_tuple(value: BudgetLimits | BudgetReservation) -> tuple[int, int, float]:
    return value.model_calls, value.model_tokens, value.wall_clock_sec


def _validate_dispatcher_concurrency_payload(payload: dict) -> None:
    expected = {
        "policy",
        "action",
        "previous_concurrency",
        "current_concurrency",
        "success_streak",
        "cooldown_until_epoch_sec",
        "successful_attempts",
        "capacity_request_ids",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("dispatcher concurrency ledger payload fields are invalid")
    if not isinstance(payload["policy"], dict):
        raise ValueError("dispatcher concurrency policy must be an object")
    if payload["action"] not in {
        "initialized",
        "capacity_decrease",
        "success_observed",
        "success_increase",
    }:
        raise ValueError("dispatcher concurrency action is invalid")
    for field in (
        "previous_concurrency",
        "current_concurrency",
        "success_streak",
        "successful_attempts",
    ):
        value = payload[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"dispatcher concurrency {field} is invalid")
    cooldown = payload["cooldown_until_epoch_sec"]
    if (
        not isinstance(cooldown, (int, float))
        or isinstance(cooldown, bool)
        or not math.isfinite(cooldown)
        or cooldown < 0
    ):
        raise ValueError("dispatcher concurrency cooldown is invalid")
    request_ids = payload["capacity_request_ids"]
    if not isinstance(request_ids, list) or any(
        not isinstance(item, str) or not item for item in request_ids
    ):
        raise ValueError("dispatcher concurrency request ids are invalid")


def _resolve_attempt_kind(attempt: int, attempt_kind: str | None) -> str:
    if type(attempt) is not int or attempt < 0:
        raise ValueError("attempt must be a nonnegative integer")
    if attempt_kind is None:
        # Version-1 ledgers predate explicit kinds: attempt 0 was primary and
        # attempt 1 was the single repair. Preserve that read contract.
        if attempt == 0:
            return "primary"
        if attempt == 1:
            return "repair"
        raise RuntimeError("legacy attempt sequence allows only one repair")
    if attempt_kind not in ATTEMPT_KINDS:
        raise ValueError("invalid attempt kind")
    return attempt_kind


def _validate_attempt_sequence(
    terminals: list[dict],
    attempt: int,
    attempt_kind: str,
) -> None:
    if attempt > 2:
        raise RuntimeError("at most one recovery and one repair attempt are allowed")
    prior = [item["attempt_kind"] for item in terminals]
    if attempt == 0:
        if attempt_kind != "primary":
            raise RuntimeError("first attempt must be primary")
        return
    if not prior or prior[0] != "primary":
        raise RuntimeError("attempt sequence has no primary attempt")
    if attempt_kind == "primary":
        raise RuntimeError("primary attempt cannot be repeated")
    if attempt_kind == "recovery":
        if prior != ["primary"]:
            raise RuntimeError("recovery must immediately follow the primary attempt")
        return
    if attempt_kind == "repair":
        if "repair" in prior:
            raise RuntimeError("at most one repair attempt is allowed")
        if prior not in (["primary"], ["primary", "recovery"]):
            raise RuntimeError("repair attempt sequence is invalid")
        return
    raise RuntimeError("invalid attempt kind")


def _normalize_usage(usage: dict | None, *, model_tokens: int) -> dict:
    usage = usage if isinstance(usage, dict) else {}

    def number(*names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
        return None

    cached = number("cached_input_tokens")
    details = usage.get("prompt_tokens_details")
    if cached is None and isinstance(details, dict):
        cached = _number_from_mapping(details, "cached_tokens")
    return {
        "input_tokens": number("input_tokens", "prompt_tokens"),
        "cached_input_tokens": cached,
        "output_tokens": number("output_tokens", "completion_tokens"),
        "total_tokens": model_tokens,
    }


def _number_from_mapping(mapping: dict, name: str) -> int | None:
    value = mapping.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    return None


def _require_call(state: dict, logical_call_id: object) -> dict:
    _require_text(logical_call_id, "logical_call_id")
    try:
        return state["calls"][logical_call_id]
    except KeyError as exc:
        raise RuntimeError(f"logical call is not reserved: {logical_call_id}") from exc


def _require_text(value: object, field: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_positive_int(value: object, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _require_positive_float(value: object, field: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be positive and finite")
