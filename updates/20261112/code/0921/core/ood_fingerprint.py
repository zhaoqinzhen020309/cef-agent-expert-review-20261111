from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from agent_ahd.core.preflight import collect_environment_fingerprint, file_sha256
from agent_ahd.core.protocol_versions import PROTOCOL_VERSIONS


def fingerprint_payload(
    *,
    shortlist_policy: str,
    search_state_sha256: str,
    candidate_sources: dict[str, str],
    evaluator_sha256: str,
    timeout_sha256: str,
    data_manifest_sha256: str,
    seeds: tuple[int, ...],
    instance_ids: tuple[str, ...],
    budget: dict,
    environment: dict,
) -> dict:
    if not shortlist_policy or not search_state_sha256:
        raise ValueError("OOD fingerprint requires shortlist policy and search state")
    if not evaluator_sha256 or not timeout_sha256 or not data_manifest_sha256:
        raise ValueError("OOD fingerprint requires evaluator, timeout, and data hashes")
    if not seeds or not instance_ids:
        raise ValueError("OOD fingerprint requires seeds and instances")
    return {
        "schema_version": PROTOCOL_VERSIONS["ood"],
        "shortlist_policy": shortlist_policy,
        "search_state_sha256": search_state_sha256,
        "candidate_sources": dict(sorted(candidate_sources.items())),
        "evaluator_sha256": evaluator_sha256,
        "timeout_sha256": timeout_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "seeds": list(seeds),
        "instance_ids": list(instance_ids),
        "budget": budget,
        "environment": environment,
    }


def load_matching_ood(path: str | Path, expected_fingerprint: dict) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    actual = payload.get("fingerprint")
    if not isinstance(actual, dict):
        raise RuntimeError("OOD result has no complete fingerprint")
    if actual != expected_fingerprint:
        raise RuntimeError("OOD fingerprint mismatch; refusing cached result")
    if actual.get("schema_version") != PROTOCOL_VERSIONS["ood"]:
        raise RuntimeError("OOD fingerprint schema is unsupported")
    return payload


def build_run_ood_fingerprint(
    run_dir: str | Path,
    *,
    adapter,
    shortlist,
    budget,
    shortlist_policy: str,
    evaluator_module: str,
) -> dict:
    root = Path(run_dir).resolve()
    module_spec = importlib.util.find_spec(evaluator_module)
    if module_spec is None or module_spec.origin is None:
        raise RuntimeError(f"cannot resolve evaluator module: {evaluator_module}")
    manifest = adapter.validation_manifest
    data_record = {
        "split": manifest.split,
        "instance_ids": list(manifest.instance_ids),
        "source": _data_hashes(manifest.source_path, manifest.instance_ids),
    }
    return fingerprint_payload(
        shortlist_policy=shortlist_policy,
        search_state_sha256=file_sha256(root / "search_state.json"),
        candidate_sources={
            node.node_id: _node_source_hash(node) for node in shortlist
        },
        evaluator_sha256=file_sha256(module_spec.origin),
        timeout_sha256=file_sha256(
            Path(__file__).resolve().with_name("eoh_timeout.py")
        ),
        data_manifest_sha256=_json_hash(data_record),
        seeds=tuple(budget.resolved_seeds),
        instance_ids=tuple(manifest.instance_ids),
        budget={
            "n_instances": budget.n_instances,
            "time_limit": budget.time_limit,
            "ite_max": budget.ite_max,
        },
        environment=collect_environment_fingerprint(),
    )


def _node_source_hash(node) -> str:
    if node.solver_sha256:
        return str(node.solver_sha256)
    if node.solver_file_sha256:
        return _json_hash(node.solver_file_sha256)
    raise RuntimeError(f"shortlisted candidate has no source hash: {node.node_id}")


def _data_hashes(source_path, instance_ids: tuple[str, ...]) -> object:
    if source_path is None:
        raise RuntimeError("OOD validation data path is missing")
    source = Path(source_path).resolve()
    if source.is_file():
        return {"file_sha256": file_sha256(source)}
    if not source.is_dir():
        raise FileNotFoundError(source)
    return {
        instance_id: file_sha256(source / instance_id)
        for instance_id in instance_ids
    }


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
