from __future__ import annotations

import math
from statistics import fmean

from agent_ahd.core.research_records import (
    EvaluationRecord,
    EvidenceRecord,
    MechanismHypothesis,
)


_EPSILON = 1e-12
_RUNTIME_ABSOLUTE_TOLERANCE_SEC = 0.1
_RUNTIME_RELATIVE_TOLERANCE = 1e-3


def update_hypothesis_evidence(
    hypothesis: MechanismHypothesis,
    parent: EvaluationRecord,
    candidate: EvaluationRecord,
    *,
    ablation: EvaluationRecord | None = None,
    target_instance_ids: tuple[str, ...] = (),
    source_action: str | None = None,
) -> EvidenceRecord:
    _require_paired(parent, candidate)
    parent_values = _selected_values(parent, target_instance_ids)
    candidate_values = _selected_values(candidate, target_instance_ids)
    parent_metrics = _metrics(parent, parent_values)
    candidate_metrics = _metrics(candidate, candidate_values)
    statistics = _comparison_statistics(
        "parent",
        parent_metrics,
        "candidate",
        candidate_metrics,
    )
    differences = [
        first - second
        for first, second in zip(parent_values, candidate_values)
    ]
    statistics.update({
        "paired_wins": float(sum(value > _EPSILON for value in differences)),
        "paired_ties": float(sum(abs(value) <= _EPSILON for value in differences)),
        "paired_losses": float(sum(value < -_EPSILON for value in differences)),
    })
    non_target = _non_target_values(parent, candidate, target_instance_ids)
    if non_target is not None:
        parent_non_target, candidate_non_target = non_target
        statistics["non_target_parent_minus_candidate"] = float(
            fmean(parent_non_target) - fmean(candidate_non_target)
        )
    quality_predictions = {
        metric: direction
        for metric, direction in hypothesis.predicted_effects.items()
        if metric != "runtime_sec"
    }
    direction_matches = _predictions_match(
        quality_predictions,
        parent_metrics,
        candidate_metrics,
    )
    runtime_prediction = hypothesis.predicted_effects.get("runtime_sec")
    if runtime_prediction is not None:
        statistics["runtime_prediction_matches"] = float(
            _predictions_match(
                {"runtime_sec": runtime_prediction},
                parent_metrics,
                candidate_metrics,
            )
        )
    if not direction_matches:
        status = "refuted"
    elif ablation is None:
        status = "pending"
    else:
        _require_paired(candidate, ablation)
        ablation_values = _selected_values(ablation, target_instance_ids)
        ablation_metrics = _metrics(ablation, ablation_values)
        statistics.update(
            _comparison_statistics(
                "ablation",
                ablation_metrics,
                "candidate",
                candidate_metrics,
            )
        )
        ablation_reverses = _ablation_removes_quality_benefit(
            quality_predictions,
            parent_metrics,
            candidate_metrics,
            ablation_metrics,
        )
        status = "supported" if ablation_reverses else "inconclusive"

    return EvidenceRecord(
        evidence_id=f"evidence:{hypothesis.hypothesis_id}:{candidate.candidate_id}",
        hypothesis_id=hypothesis.hypothesis_id,
        parent_id=parent.candidate_id,
        candidate_id=candidate.candidate_id,
        ablation_id=ablation.candidate_id if ablation is not None else None,
        status=status,
        paired_statistics=statistics,
        target_instance_ids=target_instance_ids,
        target_cluster_ids=hypothesis.target_cluster_ids,
        source_action=source_action,
    )


def _require_paired(first: EvaluationRecord, second: EvaluationRecord) -> None:
    if not first.valid or not second.valid:
        raise ValueError("paired evidence requires valid evaluations")
    if (first.metric_name, first.metric_unit) != (second.metric_name, second.metric_unit):
        raise ValueError("paired evidence requires the same metric")
    if first.split != second.split:
        raise ValueError("paired evidence requires the same split")
    if first.seeds != second.seeds:
        raise ValueError("paired evidence requires the same seeds")
    if first.instance_ids != second.instance_ids:
        raise ValueError("paired evidence requires the same instances")


def _selected_values(
    record: EvaluationRecord,
    target_instance_ids: tuple[str, ...],
) -> list[float]:
    if not target_instance_ids:
        selected_columns = range(len(record.instance_ids))
    else:
        position = {instance_id: index for index, instance_id in enumerate(record.instance_ids)}
        missing = set(target_instance_ids) - set(position)
        if missing:
            raise ValueError(f"unknown target instances: {sorted(missing)}")
        selected_columns = [position[item] for item in target_instance_ids]
    return [
        row[column]
        for row in record.score_matrix
        for column in selected_columns
    ]


def _non_target_values(
    parent: EvaluationRecord,
    candidate: EvaluationRecord,
    target_instance_ids: tuple[str, ...],
) -> tuple[list[float], list[float]] | None:
    if not target_instance_ids:
        return None
    target = set(target_instance_ids)
    columns = [
        index
        for index, instance_id in enumerate(parent.instance_ids)
        if instance_id not in target
    ]
    if not columns:
        return None
    return (
        [row[index] for row in parent.score_matrix for index in columns],
        [row[index] for row in candidate.score_matrix for index in columns],
    )


def _metrics(record: EvaluationRecord, values: list[float]) -> dict[str, float]:
    tail_count = max(1, math.ceil(len(values) * 0.1))
    suffix = "gap" if record.metric_name == "optimality_gap" else "score"
    return {
        "mean_"+suffix: float(fmean(values)),
        "tail_"+suffix: float(fmean(sorted(values, reverse=True)[:tail_count])),
        "worst_"+suffix: float(max(values)),
        "runtime_sec": float(record.total_runtime_sec),
    }


def _predictions_match(
    predictions: dict[str, str],
    before: dict[str, float],
    after: dict[str, float],
) -> bool:
    for metric, direction in predictions.items():
        if metric not in before or metric not in after:
            raise ValueError("prediction metric differs from the evaluation metric")
        delta = before[metric] - after[metric]
        tolerance = _metric_tolerance(metric, before[metric], after[metric])
        if direction == "decrease" and delta <= tolerance:
            return False
        if direction == "increase" and delta >= -tolerance:
            return False
    return True


def _ablation_removes_quality_benefit(
    predictions: dict[str, str],
    parent: dict[str, float],
    candidate: dict[str, float],
    ablation: dict[str, float],
) -> bool:
    if not predictions:
        return False
    for metric, direction in predictions.items():
        tolerance = _metric_tolerance(metric, parent[metric], ablation[metric])
        if direction == "decrease":
            if ablation[metric] < candidate[metric] + tolerance:
                return False
            if ablation[metric] < parent[metric] - tolerance:
                return False
        elif direction == "increase":
            if ablation[metric] > candidate[metric] - tolerance:
                return False
            if ablation[metric] > parent[metric] + tolerance:
                return False
    return True


def _metric_tolerance(metric: str, before: float, after: float) -> float:
    if metric != "runtime_sec":
        return _EPSILON
    return max(
        _RUNTIME_ABSOLUTE_TOLERANCE_SEC,
        _RUNTIME_RELATIVE_TOLERANCE * max(abs(before), abs(after)),
    )


def _comparison_statistics(
    first_name: str,
    first: dict[str, float],
    second_name: str,
    second: dict[str, float],
) -> dict[str, float]:
    statistics = {}
    for metric in sorted(first):
        statistics[f"{first_name}_{metric}"] = float(first[metric])
        statistics[f"{second_name}_{metric}"] = float(second[metric])
        statistics[f"{first_name}_minus_{second_name}_{metric}"] = float(
            first[metric] - second[metric]
        )
    return statistics
