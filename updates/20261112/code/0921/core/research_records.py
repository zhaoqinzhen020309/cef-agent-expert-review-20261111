from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from statistics import fmean
from typing import Any


class SearchAction(str, Enum):
    EXPLOIT = "EXPLOIT"
    REPAIR = "REPAIR"
    CROSS = "CROSS"
    DIVERSIFY = "DIVERSIFY"
    EXPAND_SCOPE = "EXPAND_SCOPE"
    NARROW_SCOPE = "NARROW_SCOPE"
    ABLATE = "ABLATE"
    RETEST = "RETEST"


class InheritanceChannel(str, Enum):
    FULL_SOURCE = "full_source"
    LOCAL_SOURCE = "local_source"
    MECHANISM_ONLY = "mechanism_only"
    INDEPENDENT = "independent"


@dataclass(frozen=True)
class ActionCost:
    """Deterministic upper-bound cost attached to one catalog option."""

    model_calls: int = 0
    candidate_evaluation_units: int = 0
    diagnostic_evaluation_units: int = 0

    def __post_init__(self) -> None:
        if min(
            self.model_calls,
            self.candidate_evaluation_units,
            self.diagnostic_evaluation_units,
        ) < 0:
            raise ValueError("action costs cannot be negative")

    def __add__(self, other: "ActionCost") -> "ActionCost":
        if not isinstance(other, ActionCost):
            return NotImplemented
        return ActionCost(
            model_calls=self.model_calls + other.model_calls,
            candidate_evaluation_units=(
                self.candidate_evaluation_units
                + other.candidate_evaluation_units
            ),
            diagnostic_evaluation_units=(
                self.diagnostic_evaluation_units
                + other.diagnostic_evaluation_units
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ActionCost":
        data = data or {}
        _reject_unknown_fields(
            data,
            {
                "model_calls",
                "candidate_evaluation_units",
                "diagnostic_evaluation_units",
            },
            "action cost",
        )
        return cls(
            model_calls=int(data.get("model_calls", 0)),
            candidate_evaluation_units=int(
                data.get("candidate_evaluation_units", 0)
            ),
            diagnostic_evaluation_units=int(
                data.get("diagnostic_evaluation_units", 0)
            ),
        )


SCOPE_KINDS = {"full_solver", "symbol_set"}
PREDICTABLE_METRICS = {"mean_gap", "tail_gap", "worst_gap", "mean_score", "tail_score", "worst_score", "runtime_sec"}
PREDICTION_DIRECTIONS = {"decrease", "increase"}


@dataclass(frozen=True)
class ScopeSpec:
    kind: str
    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in SCOPE_KINDS:
            raise ValueError(f"unknown scope kind: {self.kind}")
        if self.kind == "symbol_set" and not self.symbols:
            raise ValueError(f"{self.kind} scope requires at least one symbol")
        if self.kind == "full_solver" and self.symbols:
            raise ValueError("full_solver scope cannot list symbols")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScopeSpec":
        _reject_unknown_fields(data, {"kind", "symbols"}, "scope")
        return cls(kind=str(data["kind"]), symbols=tuple(data.get("symbols", ())))


SCOPE_BOUNDARY_REASON_KINDS = {
    "symbol_dependency",
    "interface_dependency",
    "supported_mechanism_boundary",
}
SCOPE_BOUNDARY_PRODUCERS = {"engine", "static_check", "ast_dependency"}


@dataclass(frozen=True)
class ScopeBoundaryEvidence:
    evidence_id: str
    candidate_id: str
    source_scope: ScopeSpec
    approved_scope: ScopeSpec
    reason_kind: str
    supporting_evaluation_ids: tuple[str, ...]
    producer: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.candidate_id.strip():
            raise ValueError("scope boundary evidence requires identifiers")
        if self.reason_kind not in SCOPE_BOUNDARY_REASON_KINDS:
            raise ValueError(f"unknown scope boundary reason: {self.reason_kind}")
        if self.producer not in SCOPE_BOUNDARY_PRODUCERS:
            raise ValueError(f"unauthorized scope boundary producer: {self.producer}")
        if not self.supporting_evaluation_ids:
            raise ValueError("scope boundary evidence requires evaluation support")
        if not _scope_is_strictly_broader(self.approved_scope, self.source_scope):
            raise ValueError("approved scope must be strictly broader than source scope")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScopeBoundaryEvidence":
        _reject_unknown_fields(
            data,
            {
                "evidence_id",
                "candidate_id",
                "source_scope",
                "approved_scope",
                "reason_kind",
                "supporting_evaluation_ids",
                "producer",
            },
            "scope boundary evidence",
        )
        return cls(
            evidence_id=str(data["evidence_id"]),
            candidate_id=str(data["candidate_id"]),
            source_scope=ScopeSpec.from_dict(data["source_scope"]),
            approved_scope=ScopeSpec.from_dict(data["approved_scope"]),
            reason_kind=str(data["reason_kind"]),
            supporting_evaluation_ids=tuple(
                str(value) for value in data["supporting_evaluation_ids"]
            ),
            producer=str(data["producer"]),
        )


@dataclass(frozen=True)
class MechanismHypothesis:
    hypothesis_id: str
    target_cluster_ids: tuple[str, ...]
    planned_change: str
    predicted_effects: dict[str, str]
    regression_risks: tuple[str, ...]
    ablation_scope: ScopeSpec
    target_problem: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be empty")
        if not self.planned_change.strip():
            raise ValueError("planned_change cannot be empty")
        if not self.predicted_effects:
            raise ValueError("predicted_effects cannot be empty")
        unknown_metrics = set(self.predicted_effects) - PREDICTABLE_METRICS
        if unknown_metrics:
            raise ValueError(f"unknown predicted metrics: {sorted(unknown_metrics)}")
        unknown_directions = set(self.predicted_effects.values()) - PREDICTION_DIRECTIONS
        if unknown_directions:
            raise ValueError(
                f"unknown prediction directions: {sorted(unknown_directions)}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MechanismHypothesis":
        _reject_unknown_fields(
            data,
            {
                "hypothesis_id",
                "target_cluster_ids",
                "target_problem",
                "planned_change",
                "predicted_effects",
                "regression_risks",
                "ablation_scope",
            },
            "hypothesis",
        )
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            target_cluster_ids=tuple(data.get("target_cluster_ids", ())),
            target_problem=str(data.get("target_problem", "")),
            planned_change=str(data["planned_change"]),
            predicted_effects={
                str(key): str(value)
                for key, value in data.get("predicted_effects", {}).items()
            },
            regression_risks=tuple(data.get("regression_risks", ())),
            ablation_scope=ScopeSpec.from_dict(data["ablation_scope"]),
        )


@dataclass(frozen=True)
class PlannerDecision:
    decision_id: str
    action: SearchAction
    scope: ScopeSpec
    hypothesis: MechanismHypothesis | None = None
    primary_parent_id: str | None = None
    secondary_parent_id: str | None = None
    target_cluster_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    generation_context_ids: tuple[str, ...] = ()
    target_hypothesis_id: str | None = None
    retest_split: str | None = None
    retest_seeds: tuple[int, ...] = ()
    retest_instance_ids: tuple[str, ...] = ()
    cost: ActionCost = field(default_factory=ActionCost)
    inheritance_channel: InheritanceChannel = InheritanceChannel.FULL_SOURCE
    scope_evidence_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannerDecision":
        _reject_unknown_fields(
            data,
            {
                "decision_id",
                "action",
                "scope",
                "hypothesis",
                "primary_parent_id",
                "secondary_parent_id",
                "target_cluster_ids",
                "observation_ids",
                "generation_context_ids",
                "target_hypothesis_id",
                "retest_split",
                "retest_seeds",
                "retest_instance_ids",
                "cost",
                "inheritance_channel",
                "scope_evidence_id",
            },
            "planner decision",
        )
        hypothesis = data.get("hypothesis")
        return cls(
            decision_id=str(data["decision_id"]),
            action=SearchAction(data["action"]),
            scope=ScopeSpec.from_dict(data["scope"]),
            hypothesis=(
                MechanismHypothesis.from_dict(hypothesis) if hypothesis else None
            ),
            primary_parent_id=data.get("primary_parent_id"),
            secondary_parent_id=data.get("secondary_parent_id"),
            target_cluster_ids=tuple(data.get("target_cluster_ids", ())),
            observation_ids=tuple(data.get("observation_ids", ())),
            generation_context_ids=tuple(
                data.get("generation_context_ids", ())
            ),
            target_hypothesis_id=data.get("target_hypothesis_id"),
            retest_split=data.get("retest_split"),
            retest_seeds=tuple(int(value) for value in data.get("retest_seeds", ())),
            retest_instance_ids=tuple(
                str(value) for value in data.get("retest_instance_ids", ())
            ),
            cost=ActionCost.from_dict(data.get("cost")),
            inheritance_channel=InheritanceChannel(
                data.get("inheritance_channel", InheritanceChannel.FULL_SOURCE.value)
            ),
            scope_evidence_id=data.get("scope_evidence_id"),
        )


def _scope_is_strictly_broader(requested: ScopeSpec, source: ScopeSpec) -> bool:
    if source.kind == "full_solver":
        return False
    if requested.kind == "full_solver":
        return True
    return set(requested.symbols) > set(source.symbols)


@dataclass(frozen=True)
class RoundPlan:
    round_index: int
    decisions: tuple[PlannerDecision, ...] = ()
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        stopped = self.stop_reason is not None
        if stopped and not self.stop_reason.strip():
            raise ValueError("STOP requires a non-empty reason")
        if stopped and self.decisions:
            raise ValueError("STOP is exclusive with all decisions")
        if not stopped and not self.decisions:
            raise ValueError("a round plan requires decisions or STOP")

    @property
    def is_stop(self) -> bool:
        return self.stop_reason is not None

    @property
    def total_cost(self) -> ActionCost:
        total = ActionCost()
        for decision in self.decisions:
            total += decision.cost
        return total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoundPlan":
        _reject_unknown_fields(
            data,
            {"round_index", "decisions", "stop_reason"},
            "round plan",
        )
        return cls(
            round_index=int(data["round_index"]),
            decisions=tuple(
                PlannerDecision.from_dict(item) for item in data.get("decisions", ())
            ),
            stop_reason=data.get("stop_reason"),
        )

    @classmethod
    def from_controller_dict(
        cls,
        data: dict[str, Any],
        *,
        candidate_count: int,
    ) -> "RoundPlan":
        _reject_unknown_fields(
            data,
            {"round_index", "candidate_decisions", "diagnostic_decisions"},
            "controller round plan",
        )
        candidates = tuple(
            PlannerDecision.from_dict(item)
            for item in data.get("candidate_decisions", ())
        )
        diagnostics = tuple(
            PlannerDecision.from_dict(item)
            for item in data.get("diagnostic_decisions", ())
        )
        if len(candidates) > candidate_count:
            raise ValueError(
                f"controller returned {len(candidates)} candidate decisions, "
                f"maximum is {candidate_count}"
            )
        if any(item.action == SearchAction.RETEST for item in candidates):
            raise ValueError("candidate_decisions cannot contain RETEST")
        if any(item.action != SearchAction.RETEST for item in diagnostics):
            raise ValueError("diagnostic_decisions may contain only RETEST")
        return cls(
            round_index=int(data["round_index"]),
            decisions=candidates + diagnostics,
        )


@dataclass(frozen=True)
class ControllerSelection:
    round_index: int
    candidate_option_ids: tuple[str, ...]
    diagnostic_option_ids: tuple[str, ...] = ()
    stop: bool = False
    stop_reason: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        candidate_count: int,
    ) -> "ControllerSelection":
        _reject_unknown_fields(
            data,
            {
                "round_index",
                "candidate_option_ids",
                "diagnostic_option_ids",
                "stop",
                "stop_reason",
            },
            "controller selection",
        )
        candidate_option_ids = tuple(
            str(value) for value in data.get("candidate_option_ids", ())
        )
        if len(candidate_option_ids) > candidate_count:
            raise ValueError(
                f"controller returned {len(candidate_option_ids)} candidate "
                f"options, maximum is {candidate_count}"
            )
        diagnostic_option_ids = tuple(
            str(value) for value in data.get("diagnostic_option_ids", ())
        )
        stop = bool(data.get("stop", False))
        stop_reason = data.get("stop_reason")
        if stop:
            if candidate_option_ids or diagnostic_option_ids:
                raise ValueError("STOP is exclusive with all option ids")
            if not isinstance(stop_reason, str) or not stop_reason.strip():
                raise ValueError("STOP requires a non-empty reason")
        elif stop_reason is not None:
            raise ValueError("stop_reason requires stop=true")
        elif not candidate_option_ids and not diagnostic_option_ids:
            raise ValueError("selection requires at least one option or STOP")
        return cls(
            round_index=int(data["round_index"]),
            candidate_option_ids=candidate_option_ids,
            diagnostic_option_ids=diagnostic_option_ids,
            stop=stop,
            stop_reason=stop_reason,
        )


def _reject_unknown_fields(
    data: dict[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: str
    candidate_id: str
    split: str
    seeds: tuple[int, ...]
    instance_ids: tuple[str, ...]
    gap_matrix: tuple[tuple[float, ...], ...]
    runtime_matrix: tuple[tuple[float, ...], ...]
    valid: bool
    total_runtime_sec: float = 0.0
    error_type: str | None = None
    error_message: str | None = None
    timeout_timing: dict[str, object] | None = None
    objective_matrix: tuple[tuple[float, ...], ...] = ()
    objective_name: str | None = None
    objective_unit: str | None = None
    metric_name: str = "optimality_gap"
    metric_unit: str = "percent"

    def __post_init__(self) -> None:
        if self.valid:
            matrix = self.score_matrix
            if not self.seeds or not self.instance_ids or not matrix:
                raise ValueError("valid evaluations require seeds, instances, and scores")
            if self.metric_name == "optimality_gap":
                if self.metric_unit != "percent":
                    raise ValueError("optimality gap must remain in percent")
            elif (self.metric_name != self.objective_name or self.metric_unit != self.objective_unit
                  or not self.objective_name or not self.objective_unit or self.gap_matrix):
                raise ValueError("raw scores require matching objective units and no gap matrix")
            if len(matrix) != len(self.seeds):
                raise ValueError("score matrix row count must match seeds")
            if any(len(row) != len(self.instance_ids) for row in matrix):
                raise ValueError("score matrix columns must match instance_ids")
            from math import isfinite
            if any(not isfinite(value) for row in matrix for value in row):
                raise ValueError("score matrix must be finite")
            if self.runtime_matrix:
                if len(self.runtime_matrix) != len(self.seeds):
                    raise ValueError("runtime_matrix row count must match seeds")
                if any(
                    len(row) != len(self.instance_ids) for row in self.runtime_matrix
                ):
                    raise ValueError("runtime_matrix columns must match instance_ids")

    @property
    def mean_gap(self) -> float | None:
        if not self.valid or self.metric_name != "optimality_gap":
            return None
        return fmean(value for row in self.gap_matrix for value in row)

    @property
    def score_matrix(self) -> tuple[tuple[float, ...], ...]:
        return self.gap_matrix if self.metric_name == "optimality_gap" else self.objective_matrix

    @property
    def mean_score(self) -> float | None:
        return fmean(value for row in self.score_matrix for value in row) if self.valid else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRecord":
        return cls(
            evaluation_id=str(data["evaluation_id"]),
            candidate_id=str(data["candidate_id"]),
            split=str(data.get("split", "train")),
            seeds=tuple(int(value) for value in data.get("seeds", ())),
            instance_ids=tuple(str(value) for value in data.get("instance_ids", ())),
            gap_matrix=tuple(
                tuple(float(value) for value in row)
                for row in data.get("gap_matrix", ())
            ),
            runtime_matrix=tuple(
                tuple(float(value) for value in row)
                for row in data.get("runtime_matrix", ())
            ),
            valid=bool(data["valid"]),
            total_runtime_sec=float(data.get("total_runtime_sec", 0.0)),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            objective_matrix=tuple(tuple(float(value) for value in row)
                                   for row in data.get("objective_matrix") or ()),
            objective_name=data.get("objective_name"),
            objective_unit=data.get("objective_unit"),
            metric_name=str(data.get("metric_name", "optimality_gap")),
            metric_unit=str(data.get("metric_unit", "percent")),
            timeout_timing=(
                dict(data["timeout_timing"])
                if isinstance(data.get("timeout_timing"), dict)
                else None
            ),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    hypothesis_id: str
    parent_id: str
    candidate_id: str
    status: str
    paired_statistics: dict[str, float]
    ablation_id: str | None = None
    target_instance_ids: tuple[str, ...] = ()
    target_cluster_ids: tuple[str, ...] = ()
    source_action: str | None = None
    failure_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRecord":
        return cls(
            evidence_id=str(data["evidence_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            parent_id=str(data["parent_id"]),
            candidate_id=str(data["candidate_id"]),
            status=str(data["status"]),
            paired_statistics={
                str(key): float(value)
                for key, value in data.get("paired_statistics", {}).items()
            },
            ablation_id=data.get("ablation_id"),
            target_instance_ids=tuple(data.get("target_instance_ids", ())),
            target_cluster_ids=tuple(data.get("target_cluster_ids", ())),
            source_action=data.get("source_action"),
            failure_reason=data.get("failure_reason"),
        )


@dataclass(frozen=True)
class Observation:
    observation_id: str
    kind: str
    statement: str
    values: dict[str, Any] = field(default_factory=dict)
