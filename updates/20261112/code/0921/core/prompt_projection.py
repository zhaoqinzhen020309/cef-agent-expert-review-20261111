from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROMPT_PROJECTION_VERSION = "agent_ahd_prompt_projection_v1"

ROLE_PROMPT_LIMITS = {
    "controller": {"max_input_bytes": 256 * 1024},
    "hypothesis": {"max_input_bytes": 256 * 1024},
    "implementation": {"max_input_bytes": 512 * 1024},
    "code": {"max_input_bytes": 512 * 1024},
}


class PromptLimitError(ValueError):
    pass


def canonical_prompt(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def prompt_contract(
    role: str,
    *,
    max_input_bytes: int | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    default = ROLE_PROMPT_LIMITS.get(role)
    if default is None:
        raise ValueError(f"unknown prompt role: {role}")
    input_limit = (
        int(max_input_bytes)
        if max_input_bytes is not None
        else int(default["max_input_bytes"])
    )
    if input_limit <= 0:
        raise ValueError("max_input_bytes must be positive")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    return {
        "projection_version": PROMPT_PROJECTION_VERSION,
        "role": role,
        "max_input_bytes": input_limit,
        "max_output_tokens": max_output_tokens,
        "output_token_enforcement": "attempt_ledger_usage",
    }


def projection_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_prompt_input(
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: dict,
    max_input_bytes: int | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    contract = prompt_contract(
        role,
        max_input_bytes=max_input_bytes,
        max_output_tokens=max_output_tokens,
    )
    schema_prompt = canonical_prompt(output_schema)
    system_bytes = len(system_prompt.encode("utf-8"))
    user_bytes = len(user_prompt.encode("utf-8"))
    schema_bytes = len(schema_prompt.encode("utf-8"))
    input_bytes = system_bytes + user_bytes + schema_bytes
    if input_bytes > contract["max_input_bytes"]:
        raise PromptLimitError(
            f"{role} prompt exceeds max_input_bytes: "
            f"actual={input_bytes}, limit={contract['max_input_bytes']}"
        )
    return {
        **contract,
        "system_prompt_bytes": system_bytes,
        "user_prompt_bytes": user_bytes,
        "output_schema_bytes": schema_bytes,
        "input_bytes": input_bytes,
        "input_token_upper_bound": input_bytes,
        "system_prompt_sha256": _sha256(system_prompt),
        "user_prompt_sha256": _sha256(user_prompt),
        "output_schema_sha256": _sha256(schema_prompt),
    }


def compact_repair_context(role: str, original_user_prompt: str) -> object:
    try:
        request = json.loads(original_user_prompt)
    except (TypeError, json.JSONDecodeError):
        return {"original_request": original_user_prompt}
    if not isinstance(request, dict):
        return {"original_request": request}
    keys = {
        "controller": ("budget", "decision_catalog", "evidence", "resource_facts", "training_episode_memory"),
        "hypothesis": (
            "resource_facts", "research_guidance",
            "problem_facts",
            "target_failures",
            "edit_contract",
            "source_bundle",
        ),
        "implementation": (
            "resource_facts", "implementation_guidance",
            "problem_facts",
            "locked_hypothesis",
            "edit_contract",
            "source_bundle",
            "auxiliary_mechanism",
            "output_contract",
        ),
        "code": (
            "resource_facts", "implementation_guidance",
            "problem_facts",
            "locked_hypothesis",
            "edit_contract",
            "source_bundle",
            "auxiliary_mechanism",
            "output_contract",
        ),
    }.get(role)
    if keys is None:
        raise ValueError(f"unknown prompt role: {role}")
    return {
        key: request[key]
        for key in keys
        if key in request
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
