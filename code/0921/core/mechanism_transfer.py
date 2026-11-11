from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping

from agent_ahd.core.research_records import (
    EvaluationRecord,
    EvidenceRecord,
    MechanismHypothesis,
    ScopeSpec,
)
from agent_ahd.core.staged_evaluation import NONINFERIORITY_TOLERANCE_PP
from agent_ahd.core.protocol_versions import PROTOCOL_VERSIONS


_EPSILON = 1e-12


@dataclass(frozen=True)
class MechanismTransferRecord:
    evidence_id: str
    hypothesis_id: str
    parent_id: str
    candidate_id: str
    target_problem: str
    planned_change: str
    effective_instance_ids: tuple[str, ...]
    harmed_instance_ids: tuple[str, ...]
    target_failure_types: tuple[str, ...]
    ablation_id: str | None
    ablation_summary: dict[str, float]
    known_failure_distributions: tuple[str, ...]
    validation_candidate_mean_gap: float | None
    validation_control_mean_gap: float | None
    allowed_interfaces: tuple[str, ...]
    modification_boundary: ScopeSpec
    evidence_status: str
    transfer_eligible: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MechanismTransferRecord":
        return cls(
            evidence_id=str(data["evidence_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            parent_id=str(data["parent_id"]),
            candidate_id=str(data["candidate_id"]),
            target_problem=str(data["target_problem"]),
            planned_change=str(data["planned_change"]),
            effective_instance_ids=tuple(data.get("effective_instance_ids", ())),
            harmed_instance_ids=tuple(data.get("harmed_instance_ids", ())),
            target_failure_types=tuple(data.get("target_failure_types", ())),
            ablation_id=data.get("ablation_id"),
            ablation_summary={
                str(key): float(value)
                for key, value in data.get("ablation_summary", {}).items()
            },
            known_failure_distributions=tuple(
                data.get("known_failure_distributions", ())
            ),
            validation_candidate_mean_gap=(
                float(data["validation_candidate_mean_gap"])
                if data.get("validation_candidate_mean_gap") is not None
                else None
            ),
            validation_control_mean_gap=(
                float(data["validation_control_mean_gap"])
                if data.get("validation_control_mean_gap") is not None
                else None
            ),
            allowed_interfaces=tuple(data.get("allowed_interfaces", ())),
            modification_boundary=ScopeSpec.from_dict(
                data["modification_boundary"]
            ),
            evidence_status=str(data["evidence_status"]),
            transfer_eligible=bool(data["transfer_eligible"]),
            rejection_reasons=tuple(data.get("rejection_reasons", ())),
        )


def create_mechanism_transfer_registry(
    records: Iterable[MechanismTransferRecord],
    *,
    registry_context: dict | None = None,
) -> dict:
    eligible = tuple(records)
    if any(not record.transfer_eligible for record in eligible):
        raise ValueError("registry accepts only transfer-eligible records")
    payload = {
        "schema_version": PROTOCOL_VERSIONS["mechanism_evidence"],
        "status": (
            "ready" if eligible else "no_transfer_eligible_donor"
        ),
        "records": [
            record.to_dict()
            for record in sorted(
                eligible,
                key=lambda item: (item.candidate_id, item.hypothesis_id),
            )
        ],
    }
    if registry_context is not None:
        payload["registry_context"] = json.loads(json.dumps(registry_context))
    payload["registry_hash"] = _registry_hash(payload)
    return payload


def partition_mechanism_transfer_records(
    records: Iterable[MechanismTransferRecord],
    *,
    legacy_unverified: Iterable[Mapping[str, object]] = (),
) -> tuple[tuple[MechanismTransferRecord, ...], dict]:
    record_list = tuple(records)
    legacy_items = tuple(dict(item) for item in legacy_unverified)
    eligible = tuple(record for record in record_list if record.transfer_eligible)
    rejected = tuple(record for record in record_list if not record.transfer_eligible)
    audit = {
        "eligible_record_count": len(eligible),
        "rejected_record_count": len(rejected),
        "legacy_unverified_count": len(legacy_items),
        "rejected_records": [record.to_dict() for record in rejected],
        "legacy_unverified": list(legacy_items),
    }
    return eligible, audit


def registry_compatibility_context(
    *,
    locked_artifacts: Mapping[str, str],
    train_manifest: Mapping[str, object],
    validation_manifest: Mapping[str, object],
) -> dict:
    return {
        "evaluator_hash": _json_hash(dict(locked_artifacts)),
        "data_hash": _json_hash(
            {
                "train": dict(train_manifest),
                "validation": dict(validation_manifest),
            }
        ),
        "protocol_versions": dict(PROTOCOL_VERSIONS),
    }


def load_mechanism_transfer_registry(
    path: str | Path,
    *,
    expected_hash: str | None = None,
    expected_context: Mapping[str, object] | None = None,
) -> tuple[MechanismTransferRecord, ...]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    actual_hash = data.get("registry_hash")
    if actual_hash != _registry_hash(
        {key: value for key, value in data.items() if key != "registry_hash"}
    ):
        raise RuntimeError("mechanism transfer registry hash mismatch")
    if expected_hash is not None and actual_hash != expected_hash:
        raise RuntimeError("mechanism transfer registry differs from locked hash")
    if data.get("schema_version") != PROTOCOL_VERSIONS["mechanism_evidence"]:
        raise RuntimeError("mechanism transfer registry schema is unsupported")
    context = data.get("registry_context")
    if context is not None:
        if (
            not isinstance(context, dict)
            or any(
                not isinstance(context.get(key), str)
                for key in ("evaluator_hash", "data_hash")
            )
            or not isinstance(context.get("protocol_versions"), dict)
            or any(
                key in context and not isinstance(context[key], str)
                for key in (
                    "root_protocol_hash",
                    "evidence_hash",
                    "source_state_sha256",
                    "planner_variant",
                )
            )
        ):
            raise RuntimeError("mechanism transfer registry context is incomplete")
    if expected_context is not None:
        if not isinstance(context, dict):
            raise RuntimeError("mechanism transfer registry context is required")
        mismatches = sorted(
            key
            for key, expected in expected_context.items()
            if context.get(key) != expected
        )
        if mismatches:
            raise RuntimeError(
                "mechanism transfer registry context mismatch: "
                + ", ".join(mismatches)
            )
    records = tuple(
        MechanismTransferRecord.from_dict(item)
        for item in data.get("records", ())
    )
    if any(not record.transfer_eligible for record in records):
        raise RuntimeError("registry contains an ineligible transfer record")
    expected_status = "ready" if records else "no_transfer_eligible_donor"
    if data.get("status") != expected_status:
        raise RuntimeError("mechanism transfer registry status mismatch")
    return records


def _registry_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_mechanism_transfer_record(
    hypothesis: MechanismHypothesis,
    evidence: EvidenceRecord,
    *,
    evaluations: Iterable[EvaluationRecord],
    allowed_interfaces: tuple[str, ...],
    validation_family_by_instance: dict[str, str],
    stable_control_id: str,
    noninferiority_tolerance_pp: float = NONINFERIORITY_TOLERANCE_PP,
) -> MechanismTransferRecord:
    records = tuple(evaluations)
    parent = _find_primary_evaluation(records, evidence.parent_id, split="train")
    candidate = _find_primary_evaluation(records, evidence.candidate_id, split="train")
    effective, harmed = _paired_instance_directions(parent, candidate)

    validation_candidate = _find_validation_evaluation(
        records,
        evidence.candidate_id,
    )
    validation_control = _find_validation_evaluation(records, stable_control_id)
    validation_candidate_mean = None
    validation_control_mean = None
    failure_distributions: tuple[str, ...] = ()
    validation_aligned = False
    if validation_candidate is not None and validation_control is not None:
        validation_aligned = _records_are_paired(
            validation_candidate,
            validation_control,
        )
        if validation_aligned:
            validation_candidate_mean = _mean_gap(validation_candidate)
            validation_control_mean = _mean_gap(validation_control)
            failure_distributions = _degraded_validation_families(
                validation_candidate,
                validation_control,
                validation_family_by_instance,
                tolerance=noninferiority_tolerance_pp,
            )

    reasons = []
    if evidence.status != "supported":
        reasons.append("development_evidence_not_supported")
    if evidence.ablation_id is None:
        reasons.append("controlled_ablation_missing")
    if not (set(hypothesis.predicted_effects) - {"runtime_sec"}):
        reasons.append("quality_mechanism_missing")
    if not hypothesis.target_problem.strip():
        reasons.append("target_problem_missing")
    if not hypothesis.target_cluster_ids:
        reasons.append("target_failure_types_missing")
    if not effective and not hypothesis.target_cluster_ids:
        reasons.append("effective_cases_missing")
    if not allowed_interfaces:
        reasons.append("allowed_interfaces_missing")
    if validation_candidate is None or validation_control is None:
        reasons.append("validation_evidence_missing")
    elif not validation_aligned:
        reasons.append("validation_protocol_mismatch")
    elif (
        validation_candidate_mean is not None
        and validation_control_mean is not None
        and validation_candidate_mean
        > validation_control_mean + noninferiority_tolerance_pp + _EPSILON
    ):
        reasons.append("validation_noninferiority_failed")

    paired = evidence.paired_statistics
    ablation_summary = {
        key: float(paired[key])
        for key in (
            "parent_mean_gap",
            "candidate_mean_gap",
            "ablation_mean_gap",
            "parent_minus_candidate_mean_gap",
            "ablation_minus_candidate_mean_gap",
        )
        if key in paired
    }
    return MechanismTransferRecord(
        evidence_id=evidence.evidence_id,
        hypothesis_id=evidence.hypothesis_id,
        parent_id=evidence.parent_id,
        candidate_id=evidence.candidate_id,
        target_problem=hypothesis.target_problem,
        planned_change=hypothesis.planned_change,
        effective_instance_ids=effective,
        harmed_instance_ids=harmed,
        target_failure_types=hypothesis.target_cluster_ids,
        ablation_id=evidence.ablation_id,
        ablation_summary=ablation_summary,
        known_failure_distributions=failure_distributions,
        validation_candidate_mean_gap=validation_candidate_mean,
        validation_control_mean_gap=validation_control_mean,
        allowed_interfaces=tuple(sorted(set(allowed_interfaces))),
        modification_boundary=hypothesis.ablation_scope,
        evidence_status=evidence.status,
        transfer_eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def _find_primary_evaluation(
    records: tuple[EvaluationRecord, ...],
    candidate_id: str,
    *,
    split: str,
) -> EvaluationRecord | None:
    matches = [
        record
        for record in records
        if record.candidate_id == candidate_id
        and record.split == split
        and record.valid
        and record.evaluation_id.startswith("eval_")
    ]
    return matches[0] if matches else None


def _find_validation_evaluation(
    records: tuple[EvaluationRecord, ...],
    candidate_id: str,
) -> EvaluationRecord | None:
    matches = [
        record
        for record in records
        if record.candidate_id == candidate_id
        and record.split == "validation"
        and record.valid
        and record.evaluation_id.startswith("final_confirm_")
    ]
    return matches[-1] if matches else None


def _paired_instance_directions(
    parent: EvaluationRecord | None,
    candidate: EvaluationRecord | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if parent is None or candidate is None or not _records_are_paired(parent, candidate):
        return (), ()
    parent_means = _per_instance_means(parent)
    candidate_means = _per_instance_means(candidate)
    effective = tuple(
        instance_id
        for instance_id, before, after in zip(
            parent.instance_ids,
            parent_means,
            candidate_means,
        )
        if before - after > _EPSILON
    )
    harmed = tuple(
        instance_id
        for instance_id, before, after in zip(
            parent.instance_ids,
            parent_means,
            candidate_means,
        )
        if after - before > _EPSILON
    )
    return effective, harmed


def _degraded_validation_families(
    candidate: EvaluationRecord,
    control: EvaluationRecord,
    family_by_instance: dict[str, str],
    *,
    tolerance: float,
) -> tuple[str, ...]:
    candidate_means = _per_instance_means(candidate)
    control_means = _per_instance_means(control)
    candidate_by_family: dict[str, list[float]] = {}
    control_by_family: dict[str, list[float]] = {}
    for instance_id, candidate_gap, control_gap in zip(
        candidate.instance_ids,
        candidate_means,
        control_means,
    ):
        family = family_by_instance.get(instance_id, "unknown")
        candidate_by_family.setdefault(family, []).append(candidate_gap)
        control_by_family.setdefault(family, []).append(control_gap)
    return tuple(sorted(
        family
        for family in candidate_by_family
        if fmean(candidate_by_family[family])
        > fmean(control_by_family[family]) + tolerance + _EPSILON
    ))


def _records_are_paired(
    first: EvaluationRecord,
    second: EvaluationRecord,
) -> bool:
    return (
        (first.metric_name, first.metric_unit) == (second.metric_name, second.metric_unit)
        and first.split == second.split
        and first.seeds == second.seeds
        and first.instance_ids == second.instance_ids
    )


def _per_instance_means(record: EvaluationRecord) -> tuple[float, ...]:
    return tuple(
        fmean(row[index] for row in record.score_matrix)
        for index in range(len(record.instance_ids))
    )


def _mean_gap(record: EvaluationRecord) -> float:
    return fmean(value for row in record.score_matrix for value in row)
