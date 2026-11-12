"""Create the locked model backends used by every experiment entry point."""

from __future__ import annotations

import os
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

from agent_ahd.agents.agent_runner import (
    AgentRunner,
    OpenAICompatibleAgentRunner,
    QueuedAgentRunner,
    ScriptedAgentRunner,
)
from agent_ahd.agents.adaptive_concurrency import AdaptiveConcurrencyPolicy
from agent_ahd.agents.rule_backend import (
    RULE_MUTATION_CATALOG_FILENAME,
    RuleControllerRunner,
    RuleMutationBackend,
    freeze_clean_room_rule_catalog,
)
from agent_ahd.core.prompt_projection import prompt_contract


BACKEND_DESCRIPTOR_VERSION = "agent_ahd_backend_descriptor_v1"
BACKEND_ADAPTER_VERSION = "agent_ahd_agent_runner_v1"
SUPPORTED_BACKENDS = ("api", "codex_queue", "scripted", "http_mock", "rule")
SUPPORTED_PROTOCOL_MODES = ("compatibility", "v4")
REQUIRED_DESCRIPTOR_FIELDS = {
    "descriptor_version",
    "backend",
    "provider",
    "adapter",
    "adapter_version",
    "adapter_source_sha256",
    "factory_source_sha256",
    "transport",
    "role",
    "model",
    "reasoning_effort",
    "model_revision",
    "quantization",
    "service_version",
    "protocol_mode",
    "response_format",
    "schema_dialect",
    "constrained_decoding",
    "prompt_transport",
    "sampling",
    "limits",
    "role_limits",
    "concurrency",
    "retry_policy",
    "tool_permissions",
    "request_identity",
    "idempotency",
    "permissions",
    "usage_contract",
}


def build_agent_runner(
    *,
    backend: str,
    run_dir: str | Path,
    role: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
    queue_dir: str | Path | None = None,
    tape_path: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    timeout: float | None = None,
    protocol_mode: str = "compatibility",
    max_input_bytes: int | None = None,
    max_output_tokens: int | None = None,
    concurrency: int = 1,
    adaptive_concurrency_policy: dict[str, Any] | None = None,
) -> AgentRunner:
    """Build an existing runner and attach its immutable protocol descriptor."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown agent backend: {backend}")
    if role not in {"controller", "code"}:
        raise ValueError(f"unknown agent role: {role}")
    if protocol_mode not in SUPPORTED_PROTOCOL_MODES:
        raise ValueError(f"unknown protocol mode: {protocol_mode}")
    if backend == "rule" and protocol_mode != "v4":
        raise ValueError("rule backend requires protocol mode v4")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError("backend concurrency must be a positive integer")
    adaptive_policy = (
        AdaptiveConcurrencyPolicy.from_dict(adaptive_concurrency_policy)
        if adaptive_concurrency_policy is not None
        else None
    )
    if adaptive_policy is not None and backend != "codex_queue":
        raise ValueError("adaptive concurrency is supported only by codex_queue")
    if adaptive_policy is not None and adaptive_policy.maximum_concurrency != concurrency:
        raise ValueError("adaptive maximum concurrency must match backend concurrency")

    resolved_provider = (
        "local_http_mock"
        if backend == "http_mock"
        else provider
        or os.environ.get("AGENT_AHD_API_PROVIDER", "openai_compatible")
        or "openai_compatible"
    )
    deepseek = backend == "api" and resolved_provider == "deepseek"
    max_repairs = 1 if protocol_mode == "v4" else 0
    limits = prompt_contract(
        role,
        max_input_bytes=max_input_bytes,
        max_output_tokens=max_output_tokens,
    )
    role_names = (
        ("controller",)
        if role == "controller"
        else ("hypothesis", "implementation", "code")
    )
    retry_policy = {
        "max_backend_retries": 0,
        "max_recovery_attempts_per_logical_call": max_repairs,
        "max_repair_attempts_per_logical_call": max_repairs,
        "ambiguous_attempt_recovery": "new_attempt",
    }
    common = {
        "descriptor_version": BACKEND_DESCRIPTOR_VERSION,
        "role": role,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "model_revision": _optional_environment_value(
            "AGENT_AHD_MODEL_REVISION"
        ),
        "quantization": _optional_environment_value(
            "AGENT_AHD_MODEL_QUANTIZATION"
        ),
        "service_version": _optional_environment_value(
            "AGENT_AHD_SERVICE_VERSION"
        ),
        "protocol_mode": protocol_mode,
        "response_format": "json_object" if deepseek else "json_schema",
        "schema_dialect": "json-schema-draft-2020-12",
        "constrained_decoding": not deepseek,
        "sampling": {
            name: {"status": "not_sent", "value": None}
            for name in (
                "temperature",
                "top_p",
                "seed",
                "max_output_tokens",
                "stop",
                "tool_choice",
            )
        },
        "limits": limits,
        "role_limits": {
            name: prompt_contract(
                name,
                max_output_tokens=max_output_tokens,
            )
            for name in role_names
        },
        "concurrency": concurrency,
        "retry_policy": retry_policy,
        "tool_permissions": [],
        "request_identity": "logical_call_id_and_attempt",
        "usage_contract": {
            "normalizer": "agent_ahd_normalized_usage_v1",
            "statuses": [
                "reported",
                "reported_zero",
                "unavailable",
                "estimated",
            ],
            "provider_raw_usage": "recorded",
        },
        "adapter_version": BACKEND_ADAPTER_VERSION,
        "adapter_source_sha256": _source_sha256(
            Path(__file__).parent.parent / "agents" / "agent_runner.py"
        ),
        "factory_source_sha256": _source_sha256(Path(__file__)),
    }
    if deepseek:
        common["schema_enforcement"] = "local_jsonschema_draft_2020_12"
    if deepseek and max_output_tokens is not None:
        common["sampling"]["max_output_tokens"] = {
            "status": "sent",
            "value": max_output_tokens,
        }

    if backend in {"api", "http_mock"}:
        if backend == "http_mock":
            resolved_endpoint = endpoint or os.environ.get("AGENT_AHD_HTTP_MOCK_ENDPOINT", "")
            resolved_key = api_key or os.environ.get("AGENT_AHD_HTTP_MOCK_API_KEY", "http-mock")
            runner = OpenAICompatibleAgentRunner(
                endpoint=resolved_endpoint,
                api_key=resolved_key,
                model=model or "http-mock-model",
                reasoning_effort=reasoning_effort,
                timeout=(
                    float(os.environ.get("AGENT_AHD_API_TIMEOUT", "60"))
                    if timeout is None
                    else timeout
                ),
                provider="openai_compatible",
                max_output_tokens=max_output_tokens,
            )
        else:
            runner = OpenAICompatibleAgentRunner(
                endpoint=(
                    endpoint
                    if endpoint is not None
                    else os.environ.get("AGENT_AHD_API_ENDPOINT", "")
                ),
                api_key=(
                    api_key
                    if api_key is not None
                    else os.environ.get("AGENT_AHD_API_KEY", "")
                ),
                model=model or os.environ.get("AGENT_AHD_MODEL", ""),
                reasoning_effort=(
                    reasoning_effort
                    or os.environ.get("AGENT_AHD_REASONING_EFFORT")
                    or None
                ),
                timeout=(
                    float(os.environ.get("AGENT_AHD_API_TIMEOUT", "600"))
                    if timeout is None
                    else timeout
                ),
                provider=resolved_provider,
                max_output_tokens=max_output_tokens,
            )
        descriptor = {
            **common,
            "backend": "http_mock" if backend == "http_mock" else "openai_compatible_api",
            "provider": resolved_provider,
            "adapter": "OpenAICompatibleAgentRunner",
            "prompt_transport": (
                "native_messages_plus_canonical_schema_message"
                if deepseek
                else "native_messages"
            ),
            "transport": {
                "kind": "http_sse" if deepseek else "http_json",
                **({"stream": True, "include_usage": True,
                    "completion_policy": "finish_stop_and_done_and_usage"} if deepseek else {}),
                "endpoint": runner.endpoint,
                "timeout_sec": runner.timeout,
                "proxy_policy": (
                    "bypass_environment_proxy" if deepseek else "environment"
                ),
            },
            "idempotency": {
                "local_scope": "logical_call_id_and_attempt",
                "provider_scope": "unsupported",
            },
            "permissions": {
                "tools": "none",
                "filesystem": "none",
                "network": "provider_endpoint_only",
            },
        }
    elif backend == "scripted":
        if tape_path is None:
            raise ValueError("scripted backend requires tape_path")
        if not model:
            raise ValueError("scripted backend requires model")
        runner = ScriptedAgentRunner(
            tape_path=tape_path,
            role=role,
            model=model,
            reasoning_effort=reasoning_effort,
            run_dir=run_dir,
        )
        descriptor = {
            **common,
            "backend": "scripted",
            "provider": "local_scripted_tape",
            "adapter": "ScriptedAgentRunner",
            "prompt_transport": "in_process_tape",
            "transport": {
                "kind": "immutable_json_tape",
                "tape_sha256": _source_sha256(Path(tape_path).resolve()),
            },
            "idempotency": {
                "local_scope": "logical_call_id",
                "provider_scope": "record_index",
            },
            "permissions": {
                "tools": "none",
                "filesystem": "tape_read_and_run_ledger_write",
                "network": "none",
            },
        }
    elif backend == "rule":
        catalog_descriptor = None
        if role == "controller":
            runner = RuleControllerRunner(run_dir=run_dir, role=role)
        else:
            catalog_path, catalog_sha256, catalog = freeze_clean_room_rule_catalog(run_dir)
            runner = RuleMutationBackend(
                run_dir=run_dir,
                role=role,
                catalog_path=catalog_path,
                catalog_sha256=catalog_sha256,
                catalog=catalog,
            )
            catalog_descriptor = {
                "path": RULE_MUTATION_CATALOG_FILENAME,
                "sha256": catalog_sha256,
                "catalog_id": catalog["catalog_id"],
                "version": catalog["version"],
            }
        descriptor = {
            **common,
            "backend": "rule",
            "provider": "local_rule_engine",
            "adapter": type(runner).__name__,
            "prompt_transport": "canonical_local_projection",
            "transport": {
                "kind": "in_process_rule",
                "history_inputs": [],
            },
            "idempotency": {
                "local_scope": "logical_call_id",
                "provider_scope": "deterministic_rule_sequence",
            },
            "permissions": {
                "tools": "none",
                "filesystem": "run_ledger_and_frozen_catalog",
                "network": "none",
            },
        }
        if catalog_descriptor is not None:
            descriptor["clean_room_catalog"] = catalog_descriptor
        descriptor["adapter_source_sha256"] = _source_sha256(
            Path(__file__).parent.parent / "agents" / "rule_backend.py"
        )
    else:
        if not model or not reasoning_effort:
            raise ValueError("codex_queue requires --model and --reasoning-effort")
        resolved_queue_dir = Path(queue_dir or Path(run_dir) / "agent_queue").resolve()
        resolved_timeout = 3600.0 if timeout is None else timeout
        runner = QueuedAgentRunner(
            queue_dir=resolved_queue_dir,
            role=role,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=resolved_timeout,
            automatic_capacity_recovery=adaptive_policy is not None,
        )
        descriptor = {
            **common,
            "backend": "codex_queue",
            "provider": "codex",
            "adapter": "QueuedAgentRunner",
            "prompt_transport": "concatenated_text",
            "transport": {
                "kind": "filesystem_queue",
                "queue_dir": str(resolved_queue_dir),
                "timeout_sec": resolved_timeout,
            },
            "idempotency": {
                "local_scope": "logical_call_id_and_attempt",
                "provider_scope": "queue_attempt_file",
            },
            "permissions": {
                "tools": "codex_cli_internal_tools_prompt_forbidden",
                "filesystem": "queue_read_write_and_dispatcher_read_only",
                "network": "model_provider_only",
            },
        }
        if adaptive_policy is not None:
            descriptor["concurrency_policy"] = adaptive_policy.to_dict()
            descriptor["retry_policy"]["capacity_recovery"] = "automatic_once"
        else:
            descriptor["retry_policy"]["capacity_recovery"] = "resume_only"

    descriptor["model"] = runner.model
    descriptor["reasoning_effort"] = getattr(runner, "reasoning_effort", None)
    descriptor["retry_policy"]["ambiguous_attempt_recovery"] = (
        "durable_backend_identity_with_billed_recovery_attempt"
        if getattr(runner, "supports_durable_attempt_recovery", False)
        else "new_billed_protocol_attempt"
    )
    setattr(runner, "backend_descriptor", descriptor)
    return runner


def backend_descriptor(runner: Any) -> dict[str, Any] | None:
    """Return a copy of a runner descriptor, if the runner has one."""
    descriptor = getattr(runner, "backend_descriptor", None)
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        raise TypeError("backend_descriptor must be a dictionary")
    _validate_descriptor(descriptor)
    return deepcopy(descriptor)


def backend_protocol_identity(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Return the run-stable portion of a descriptor for cross-run checks."""
    _validate_descriptor(descriptor)
    identity = deepcopy(descriptor)
    identity.pop("role", None)
    identity.pop("limits", None)
    transport = identity.get("transport")
    if isinstance(transport, dict):
        transport.pop("queue_dir", None)
    return identity


def _validate_descriptor(descriptor: dict[str, Any]) -> None:
    missing = REQUIRED_DESCRIPTOR_FIELDS - set(descriptor)
    if missing:
        raise ValueError(
            f"backend descriptor missing fields: {sorted(missing)}"
        )
    if not isinstance(descriptor["transport"], dict):
        raise ValueError("backend descriptor transport must be an object")
    if not isinstance(descriptor["retry_policy"], dict):
        raise ValueError("backend descriptor retry_policy must be an object")
    if not isinstance(descriptor["tool_permissions"], list):
        raise ValueError("backend descriptor tool_permissions must be a list")


def _optional_environment_value(name: str) -> dict[str, Any]:
    value = os.environ.get(name)
    if value:
        return {"status": "reported", "value": value}
    return {"status": "unavailable", "value": None}


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
