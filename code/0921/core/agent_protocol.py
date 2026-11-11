from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from agent_ahd.agents.agent_runner import AgentResponse, AgentResponseError, AgentRunner
from agent_ahd.agents.adaptive_concurrency import is_capacity_failure
from agent_ahd.core.backend_factory import backend_descriptor
from agent_ahd.core.attempt_ledger import (
    AttemptLedger,
    BudgetLimits,
    BudgetReservation,
)
from agent_ahd.core.prompt_projection import (
    compact_repair_context,
    prompt_contract,
    validate_prompt_input,
)


class FailureClass(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    AGENT_CONTRACT = "agent_contract"
    CANDIDATE_IMPLEMENTATION = "candidate_implementation"
    RESEARCH_RESULT = "research_result"


class AgentContractViolation(ValueError):
    def __init__(self, message: str, *, reason_kind: str = "schema_violation") -> None:
        super().__init__(message)
        self.reason_kind = reason_kind


class InfrastructureFailure(RuntimeError):
    def __init__(self, message: str, *, reason_kind: str = "infrastructure_error") -> None:
        super().__init__(message)
        self.reason_kind = reason_kind


class CandidateImplementationFailure(RuntimeError):
    def __init__(self, message: str, *, reason_kind: str) -> None:
        super().__init__(message)
        self.reason_kind = reason_kind


class ResearchResult(RuntimeError):
    def __init__(self, message: str, *, reason_kind: str) -> None:
        super().__init__(message)
        self.reason_kind = reason_kind


@dataclass(frozen=True)
class FailureDisposition:
    failure_class: FailureClass
    reason_kind: str
    repairable: bool


class ProtocolAgentRunner:
    """Apply the local contract, repair policy, and attempt ledger to a backend."""

    supports_durable_attempt_recovery = True
    protocol_runner = True

    def __init__(
        self,
        *,
        run_dir: str | Path,
        runner: AgentRunner,
        role: str,
        limits: BudgetLimits,
        reservation: BudgetReservation,
        max_input_bytes: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if not role:
            raise ValueError("protocol runner role is required")
        self.run_dir = Path(run_dir).resolve()
        self.runner = runner
        self.role = role
        self.model = runner.model
        self.reasoning_effort = getattr(runner, "reasoning_effort", None)
        self.backend_descriptor = backend_descriptor(runner)
        self.limits = limits
        self.reservation = reservation
        self.prompt_contract = prompt_contract(
            role,
            max_input_bytes=max_input_bytes,
            max_output_tokens=max_output_tokens,
        )
        self.ledger = AttemptLedger(self.run_dir)
        self._reservation_group_id: str | None = None

    def begin_call_group(self, reservation_group_id: str, *, logical_calls: int) -> None:
        if type(logical_calls) is not int or logical_calls <= 0:
            raise ValueError("logical_calls must be a positive integer")
        if (
            self._reservation_group_id is not None
            and self._reservation_group_id != reservation_group_id
        ):
            raise RuntimeError("protocol runner is already bound to another call group")
        self.ledger.reserve_group(
            reservation_group_id=reservation_group_id,
            reservation=BudgetReservation(
                model_calls=self.reservation.model_calls * logical_calls,
                model_tokens=self.reservation.model_tokens * logical_calls,
                wall_clock_sec=self.reservation.wall_clock_sec * logical_calls,
            ),
            limits=self.limits,
        )
        self._reservation_group_id = reservation_group_id

    def end_call_group(self, reservation_group_id: str) -> None:
        if self._reservation_group_id != reservation_group_id:
            raise RuntimeError("protocol runner is not bound to the call group")
        self.ledger.release_group(reservation_group_id=reservation_group_id)
        self._reservation_group_id = None

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any] | None = None,
        logical_call_id: str | None = None,
        response_validator: Callable[[AgentResponse], None] | None = None,
    ) -> AgentResponse:
        if not logical_call_id:
            raise ValueError("protocol calls require logical_call_id")
        schema = output_schema or {"type": "object"}
        _check_schema_definition(schema)
        self._prompt_record(system_prompt, user_prompt, schema)
        self.ledger.reserve(
            logical_call_id=logical_call_id,
            role=self.role,
            reservation=self.reservation,
            limits=self.limits,
            reservation_group_id=self._reservation_group_id,
        )
        state = self.ledger.call_state(logical_call_id)
        if state is None:
            raise RuntimeError("attempt ledger lost the logical call reservation")
        if state["completed"]:
            return self._load_completed_response(
                logical_call_id=logical_call_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=schema,
            )
        already_dispatched = state["active_attempt"] is not None
        attempt = state["active_attempt"] if already_dispatched else state["attempts"]
        attempt_kind = (
            state["active_attempt_kind"]
            if already_dispatched
            else self._next_attempt_kind(state)
        )
        prompt = user_prompt
        if attempt_kind == "repair":
            prompt = self._repair_prompt_from_ledger(
                logical_call_id,
                schema,
                repair_attempt=attempt,
                original_user_prompt=user_prompt,
            )
            self._prompt_record(system_prompt, prompt, schema)
        response_path = self._response_path(logical_call_id, attempt)
        if (
            already_dispatched
            and not response_path.is_file()
            and not getattr(self.runner, "supports_durable_attempt_recovery", False)
        ):
            self._seal_ambiguous_direct_attempt(
                logical_call_id=logical_call_id,
                attempt=attempt,
                attempt_kind=attempt_kind,
                system_prompt=system_prompt,
                user_prompt=prompt,
                output_schema=schema,
            )
            if attempt_kind != "primary":
                self.ledger.complete_logical_call(
                    logical_call_id=logical_call_id,
                    outcome="failed:infrastructure:ambiguous_dispatched_attempt",
                )
                self.ledger.release_reservation(logical_call_id=logical_call_id)
                raise InfrastructureFailure(
                    "direct recovery attempt is ambiguous; no further attempt is allowed",
                    reason_kind="ambiguous_dispatched_attempt",
                )
            attempt += 1
            attempt_kind = "recovery"
            prompt = user_prompt
            already_dispatched = False
        return self._execute_attempt(
            logical_call_id=logical_call_id,
            attempt=attempt,
            attempt_kind=attempt_kind,
            system_prompt=system_prompt,
            user_prompt=prompt,
            original_user_prompt=user_prompt,
            output_schema=schema,
            already_dispatched=already_dispatched,
            response_validator=response_validator,
        )

    def assert_exhausted(self) -> None:
        active = [
            logical_call_id
            for logical_call_id in self._logical_call_ids()
            if not (self.ledger.call_state(logical_call_id) or {}).get("released")
        ]
        if active:
            raise RuntimeError(f"protocol runner has unfinished logical calls: {active}")
        assert_exhausted = getattr(self.runner, "assert_exhausted", None)
        if assert_exhausted is not None:
            assert_exhausted()

    def _execute_attempt(
        self,
        *,
        logical_call_id: str,
        attempt: int,
        attempt_kind: str,
        system_prompt: str,
        user_prompt: str,
        original_user_prompt: str,
        output_schema: dict,
        already_dispatched: bool,
        response_validator: Callable[[AgentResponse], None] | None,
    ) -> AgentResponse:
        if not already_dispatched:
            self.ledger.start_attempt(
                logical_call_id=logical_call_id,
                attempt=attempt,
                attempt_kind=attempt_kind,
            )
        response_path = self._response_path(logical_call_id, attempt)
        response = None
        started = time.monotonic()
        try:
            if response_path.is_file():
                response = self._read_cached_response(
                    response_path,
                    logical_call_id=logical_call_id,
                    attempt=attempt,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=output_schema,
                )
            else:
                response = self.runner.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=output_schema,
                    logical_call_id=self._backend_call_id(
                        logical_call_id,
                        attempt,
                        attempt_kind,
                    ),
                )
                validate_response_schema(response.payload, output_schema)
                self._write_cached_response(
                    response_path,
                    response=response,
                    logical_call_id=logical_call_id,
                    attempt=attempt,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=output_schema,
                )
            if response_validator is not None:
                response_validator(response)
            usage_status, model_tokens = _normalized_usage(response.usage)
            self.ledger.finish_attempt(
                logical_call_id=logical_call_id,
                attempt=attempt,
                terminal_status="returned",
                usage_status=usage_status,
                model_tokens=model_tokens,
                wall_clock_sec=time.monotonic() - started,
                usage=response.usage,
            )
            self.ledger.complete_logical_call(
                logical_call_id=logical_call_id,
                outcome=(
                    "returned"
                    if attempt_kind == "primary"
                    else f"returned_after_{attempt_kind}"
                ),
            )
            self.ledger.release_reservation(logical_call_id=logical_call_id)
            return response
        except Exception as exc:
            disposition = classify_failure(exc)
            retryable_network = (
                isinstance(exc, AgentResponseError) and exc.stage == "network"
            )
            usage = (
                response.usage
                if response is not None
                else getattr(exc, "usage", {})
            )
            usage_status, model_tokens = _normalized_usage(usage)
            self._write_attempt_error(
                self._error_path(logical_call_id, attempt),
                exc=exc,
                response=response,
                logical_call_id=logical_call_id,
                attempt=attempt,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
            )
            current = self.ledger.call_state(logical_call_id)
            if current is not None and current["active_attempt"] == attempt:
                self.ledger.finish_attempt(
                    logical_call_id=logical_call_id,
                    attempt=attempt,
                    terminal_status="failed",
                    usage_status=usage_status,
                    model_tokens=model_tokens,
                    wall_clock_sec=time.monotonic() - started,
                    failure_class=disposition.failure_class.value,
                    reason_kind=disposition.reason_kind,
                    usage=usage,
                )
            if retryable_network and attempt_kind == "primary":
                if getattr(
                    self.runner,
                    "supports_durable_attempt_recovery",
                    False,
                ):
                    if getattr(
                        self.runner,
                        "automatic_capacity_recovery",
                        False,
                    ) and is_capacity_failure(exc):
                        return self._execute_attempt(
                            logical_call_id=logical_call_id,
                            attempt=attempt + 1,
                            attempt_kind="recovery",
                            system_prompt=system_prompt,
                            user_prompt=original_user_prompt,
                            original_user_prompt=original_user_prompt,
                            output_schema=output_schema,
                            already_dispatched=False,
                            response_validator=response_validator,
                        )
                    # The failed transport is terminal and billed here.
                    # Resume starts one recovery ledger attempt while reusing
                    # the backend's durable logical-call identity.
                    raise
                return self._execute_attempt(
                    logical_call_id=logical_call_id,
                    attempt=attempt + 1,
                    attempt_kind="recovery",
                    system_prompt=system_prompt,
                    user_prompt=original_user_prompt,
                    original_user_prompt=original_user_prompt,
                    output_schema=output_schema,
                    already_dispatched=False,
                    response_validator=response_validator,
                )
            if disposition.repairable and attempt_kind != "repair":
                repair_prompt = _build_repair_prompt(
                    error=exc,
                    raw_content=(
                        response.raw_content
                        if response is not None
                        else str(getattr(exc, "raw_content", ""))
                    ),
                    output_schema=output_schema,
                    role=self.role,
                    original_user_prompt=original_user_prompt,
                )
                try:
                    self._prompt_record(system_prompt, repair_prompt, output_schema)
                except Exception:
                    self.ledger.complete_logical_call(
                        logical_call_id=logical_call_id,
                        outcome="failed:agent_contract:repair_prompt_limit",
                    )
                    self.ledger.release_reservation(logical_call_id=logical_call_id)
                    raise
                return self._execute_attempt(
                    logical_call_id=logical_call_id,
                    attempt=attempt + 1,
                    attempt_kind="repair",
                    system_prompt=system_prompt,
                    user_prompt=repair_prompt,
                    original_user_prompt=original_user_prompt,
                    output_schema=output_schema,
                    already_dispatched=False,
                    response_validator=response_validator,
                )
            self.ledger.complete_logical_call(
                logical_call_id=logical_call_id,
                outcome=f"failed:{disposition.failure_class.value}:{disposition.reason_kind}",
            )
            self.ledger.release_reservation(logical_call_id=logical_call_id)
            raise

    @staticmethod
    def _next_attempt_kind(state: dict) -> str:
        terminals = state["terminals"]
        if not terminals:
            return "primary"
        kinds = state["attempt_kinds"]
        latest = terminals[-1]
        if (
            latest.get("reason_kind") == "ambiguous_dispatched_attempt"
            and kinds == ["primary"]
        ):
            return "recovery"
        if (
            latest.get("reason_kind") == "agent_network_failure"
            and kinds == ["primary"]
        ):
            return "recovery"
        if (
            latest.get("failure_class") == FailureClass.AGENT_CONTRACT.value
            and "repair" not in kinds
        ):
            return "repair"
        raise RuntimeError("logical call has no permitted next attempt")

    def _seal_ambiguous_direct_attempt(
        self,
        *,
        logical_call_id: str,
        attempt: int,
        attempt_kind: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
    ) -> None:
        error = InfrastructureFailure(
            f"direct {attempt_kind} attempt was dispatched without a durable response",
            reason_kind="ambiguous_dispatched_attempt",
        )
        self._write_attempt_error(
            self._error_path(logical_call_id, attempt),
            exc=error,
            response=None,
            logical_call_id=logical_call_id,
            attempt=attempt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )
        self.ledger.finish_attempt(
            logical_call_id=logical_call_id,
            attempt=attempt,
            terminal_status="failed",
            usage_status="unavailable",
            model_tokens=0,
            wall_clock_sec=0.0,
            failure_class=FailureClass.INFRASTRUCTURE.value,
            reason_kind="ambiguous_dispatched_attempt",
            usage={},
        )

    def _load_completed_response(
        self,
        *,
        logical_call_id: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
    ) -> AgentResponse:
        state = self.ledger.call_state(logical_call_id)
        if state is None:
            raise RuntimeError("completed logical call is missing from the ledger")
        for terminal in reversed(state["terminals"]):
            if terminal["terminal_status"] != "returned":
                continue
            attempt = terminal["attempt"]
            attempt_kind = terminal["attempt_kind"]
            path = self._response_path(logical_call_id, attempt)
            if not path.is_file():
                continue
            prompt = (
                self._repair_prompt_from_ledger(
                    logical_call_id,
                    output_schema,
                    repair_attempt=attempt,
                    original_user_prompt=user_prompt,
                )
                if attempt_kind == "repair"
                else user_prompt
            )
            return self._read_cached_response(
                path,
                logical_call_id=logical_call_id,
                attempt=attempt,
                system_prompt=system_prompt,
                user_prompt=prompt,
                output_schema=output_schema,
            )
        raise RuntimeError("completed logical call has no durable response")

    def _repair_prompt_from_ledger(
        self,
        logical_call_id: str,
        output_schema: dict,
        *,
        repair_attempt: int,
        original_user_prompt: str,
    ) -> str:
        records = [
            record["payload"]
            for record in self.ledger.records()
            if record["event_type"] == "attempt_terminal"
            and record["payload"].get("logical_call_id") == logical_call_id
            and record["payload"].get("attempt", repair_attempt) < repair_attempt
            and record["payload"].get("failure_class")
            == FailureClass.AGENT_CONTRACT.value
        ]
        if not records:
            raise RuntimeError("repair attempt has no prior contract failure")
        source = max(records, key=lambda item: item["attempt"])
        source_attempt = source["attempt"]
        cached = self._response_path(logical_call_id, source_attempt)
        raw_content = ""
        if cached.is_file():
            raw_content = str(json.loads(cached.read_text(encoding="utf-8"))["raw_content"])
        else:
            error_path = self._error_path(logical_call_id, source_attempt)
            if error_path.is_file():
                raw_content = str(
                    json.loads(error_path.read_text(encoding="utf-8")).get(
                        "raw_content",
                        "",
                    )
                )
        return _build_repair_prompt(
            error=AgentContractViolation(source["reason_kind"]),
            raw_content=raw_content,
            output_schema=output_schema,
            role=self.role,
            original_user_prompt=original_user_prompt,
        )

    def _prompt_record(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
    ) -> dict:
        return validate_prompt_input(
            role=self.role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
            max_input_bytes=self.prompt_contract["max_input_bytes"],
            max_output_tokens=self.prompt_contract["max_output_tokens"],
        )

    def _logical_call_ids(self) -> tuple[str, ...]:
        return tuple(
            record["payload"]["logical_call_id"]
            for record in self.ledger.records()
            if record["event_type"] == "reservation_created"
        )

    def _response_path(self, logical_call_id: str, attempt: int) -> Path:
        digest = hashlib.sha256(
            f"{logical_call_id}\0{attempt}".encode("utf-8")
        ).hexdigest()
        return self.run_dir / "agent_attempts" / f"{digest}.response.json"

    def _error_path(self, logical_call_id: str, attempt: int) -> Path:
        digest = hashlib.sha256(
            f"{logical_call_id}\0{attempt}".encode("utf-8")
        ).hexdigest()
        return self.run_dir / "agent_attempts" / f"{digest}.error.json"

    def _backend_call_id(
        self,
        logical_call_id: str,
        attempt: int,
        attempt_kind: str,
    ) -> str:
        backend_attempt = attempt
        if attempt_kind == "recovery" and getattr(
            self.runner,
            "supports_durable_attempt_recovery",
            False,
        ):
            backend_attempt = 0
        return f"{logical_call_id}:protocol-attempt:{backend_attempt}"

    def _write_cached_response(
        self,
        path: Path,
        *,
        response: AgentResponse,
        logical_call_id: str,
        attempt: int,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
    ) -> None:
        record = {
            "version": 1,
            "logical_call_id": logical_call_id,
            "attempt": attempt,
            "system_prompt_sha256": _sha256(system_prompt),
            "user_prompt_sha256": _sha256(user_prompt),
            "output_schema_sha256": _sha256(
                json.dumps(output_schema, sort_keys=True)
            ),
            "prompt": self._prompt_record(
                system_prompt,
                user_prompt,
                output_schema,
            ),
            "payload": response.payload,
            "raw_content": response.raw_content,
            "usage": response.usage,
        }
        _write_immutable_json(path, record)

    def _write_attempt_error(
        self,
        path: Path,
        *,
        exc: BaseException,
        response: AgentResponse | None,
        logical_call_id: str,
        attempt: int,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
    ) -> None:
        raw_content = (
            response.raw_content
            if response is not None
            else str(getattr(exc, "raw_content", ""))
        )
        usage = (
            response.usage
            if response is not None
            else dict(getattr(exc, "usage", {}) or {})
        )
        payload = (
            response.payload
            if response is not None
            else getattr(exc, "payload", None)
        )
        record = {
            "version": 1,
            "logical_call_id": logical_call_id,
            "attempt": attempt,
            "system_prompt_sha256": _sha256(system_prompt),
            "user_prompt_sha256": _sha256(user_prompt),
            "output_schema_sha256": _sha256(
                json.dumps(output_schema, sort_keys=True)
            ),
            "prompt": self._prompt_record(
                system_prompt,
                user_prompt,
                output_schema,
            ),
            "replay_error": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "failure_stage": str(getattr(exc, "stage", "request")),
            "payload": payload,
            "raw_content": raw_content,
            "usage": usage,
        }
        _write_immutable_json(path, record)

    def _read_cached_response(
        self,
        path: Path,
        *,
        logical_call_id: str,
        attempt: int,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict,
    ) -> AgentResponse:
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "logical_call_id": logical_call_id,
            "attempt": attempt,
            "system_prompt_sha256": _sha256(system_prompt),
            "user_prompt_sha256": _sha256(user_prompt),
            "output_schema_sha256": _sha256(
                json.dumps(output_schema, sort_keys=True)
            ),
        }
        if any(record.get(field) != value for field, value in expected.items()):
            raise RuntimeError("durable protocol response identity changed")
        if record.get("prompt") != self._prompt_record(
            system_prompt,
            user_prompt,
            output_schema,
        ):
            raise RuntimeError("durable protocol prompt contract changed")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError("durable protocol response payload is invalid")
        validate_response_schema(payload, output_schema)
        return AgentResponse(
            payload=payload,
            raw_content=str(record.get("raw_content") or ""),
            usage=dict(record.get("usage") or {}),
        )


def wrap_agent_runner(*, run_dir: str | Path, runner: AgentRunner, config, role: str) -> AgentRunner:
    """Wrap only v4 runs; legacy fixtures and historical replay stay untouched."""
    if getattr(runner, "protocol_runner", False):
        return runner
    if getattr(config, "protocol_mode", "compatibility") != "v4":
        return runner
    attempts_per_logical_call = 3
    max_calls = int(config.resolved_model_call_budget) * attempts_per_logical_call
    per_attempt_tokens = int(getattr(config, "agent_call_max_tokens", 50_000))
    per_attempt_wall = float(
        getattr(config, "agent_call_max_wall_clock_sec", 3_600.0)
    )
    token_limit = config.model_token_budget
    if token_limit is None:
        token_limit = max_calls * per_attempt_tokens
    wall_limit = max_calls * per_attempt_wall
    return ProtocolAgentRunner(
        run_dir=run_dir,
        runner=runner,
        role=role,
        limits=BudgetLimits(max_calls, int(token_limit), float(wall_limit)),
        reservation=BudgetReservation(
            model_calls=attempts_per_logical_call,
            model_tokens=per_attempt_tokens * attempts_per_logical_call,
            wall_clock_sec=per_attempt_wall * attempts_per_logical_call,
        ),
        max_output_tokens=per_attempt_tokens,
    )


def validate_response_schema(payload: object, schema: dict) -> None:
    _check_schema_definition(schema)
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as exc:
        path = ".".join(str(item) for item in exc.absolute_path)
        location = path or "response"
        raise AgentContractViolation(
            f"response schema violation at {location}: {exc.message}",
            reason_kind="schema_violation",
        ) from exc


def _check_schema_definition(schema: dict) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise AgentContractViolation(
            f"invalid local schema definition: {exc.message}",
            reason_kind="invalid_schema_definition",
        ) from exc


def classify_failure(exc: BaseException) -> FailureDisposition:
    if isinstance(exc, InfrastructureFailure):
        return FailureDisposition(
            FailureClass.INFRASTRUCTURE,
            exc.reason_kind,
            False,
        )
    if isinstance(exc, OSError):
        return FailureDisposition(
            FailureClass.INFRASTRUCTURE,
            "filesystem_error",
            False,
        )
    if isinstance(exc, AgentResponseError):
        if exc.stage in {"network", "orphan", "dispatch", "infrastructure"}:
            return FailureDisposition(
                FailureClass.INFRASTRUCTURE,
                f"agent_{exc.stage}_failure",
                False,
            )
        return FailureDisposition(
            FailureClass.AGENT_CONTRACT,
            _agent_contract_reason(exc.stage),
            True,
        )
    if isinstance(exc, AgentContractViolation):
        return FailureDisposition(
            FailureClass.AGENT_CONTRACT,
            exc.reason_kind,
            True,
        )
    if isinstance(exc, (SyntaxError, TypeError, ValueError)):
        return FailureDisposition(
            FailureClass.AGENT_CONTRACT,
            "response_domain_violation",
            True,
        )
    if isinstance(exc, CandidateImplementationFailure):
        return FailureDisposition(
            FailureClass.CANDIDATE_IMPLEMENTATION,
            exc.reason_kind,
            False,
        )
    if isinstance(exc, ResearchResult):
        return FailureDisposition(
            FailureClass.RESEARCH_RESULT,
            exc.reason_kind,
            False,
        )
    return FailureDisposition(
        FailureClass.INFRASTRUCTURE,
        "unclassified_runtime_error",
        False,
    )


def is_explicit_infrastructure_failure(exc: BaseException) -> bool:
    return (
        isinstance(exc, (InfrastructureFailure, OSError))
        or isinstance(exc, AgentResponseError)
        and exc.stage in {"network", "orphan", "dispatch", "infrastructure"}
    )


def _agent_contract_reason(stage: str) -> str:
    return {
        "response_parse": "invalid_json",
        "response_validation": "response_domain_violation",
        "response_application": "response_application_violation",
    }.get(stage, "agent_contract_failure")


def _build_repair_prompt(
    *,
    error: BaseException,
    raw_content: str,
    output_schema: dict,
    role: str,
    original_user_prompt: str,
) -> str:
    return json.dumps(
        {
            "task": "repair_the_previous_response_contract_only",
            "validation_error": f"{type(error).__name__}: {error}",
            "previous_response": raw_content,
            "required_output_schema": output_schema,
            "repair_context": compact_repair_context(
                role,
                original_user_prompt,
            ),
            "rules": [
                "Return exactly one JSON object.",
                "Do not add commentary or markdown fences.",
                "Preserve the intended change; repair only the reported contract error.",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_usage(usage: object) -> tuple[str, int]:
    if not isinstance(usage, dict) or not usage:
        return "unavailable", 0
    for field in ("total_tokens", "total_token_count"):
        value = usage.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            tokens = max(0, int(value))
            return ("reported" if tokens else "reported_zero"), tokens
    for input_field, output_field in (
        ("input_tokens", "output_tokens"),
        ("prompt_tokens", "completion_tokens"),
    ):
        values = usage.get(input_field), usage.get(output_field)
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            tokens = max(0, int(values[0])) + max(0, int(values[1]))
            return ("reported" if tokens else "reported_zero"), tokens
    return "unavailable", 0


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_immutable_json(path: Path, record: dict) -> None:
    serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"immutable agent attempt artifact changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    try:
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_text(encoding="utf-8") != serialized:
                raise RuntimeError(f"immutable agent attempt artifact changed: {path}")
            return
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
