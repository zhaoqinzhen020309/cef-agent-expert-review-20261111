from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_ahd.core.research_records import (
    EvaluationRecord,
    EvidenceRecord,
    MechanismHypothesis,
    PlannerDecision,
)

STATE_VERSION = 2
GENERATION_MODES = {
    "fresh",
    "continue",
    "redesign",
    "cross_reference_continue",
    "cross_reference_redesign",
}


@dataclass(frozen=True)
class SearchConfig:
    batch_size: int = 5
    initial_size: int = 5
    random_seed: int = 2024
    initial_temperature: float = 1.0
    temperature_decay: float = 0.95
    reheat_increment: float = 0.2
    global_stagnation_rounds: int = 3
    branch_stagnation_attempts: int = 3
    progressive_widening_exponent: float = 0.6
    negative_reward_scale: float = 0.8

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.initial_size <= 0:
            raise ValueError("batch and initial sizes must be positive")
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if not 0 < self.temperature_decay <= 1:
            raise ValueError("temperature_decay must be in (0, 1]")
        if self.global_stagnation_rounds <= 0 or self.branch_stagnation_attempts <= 0:
            raise ValueError("stagnation thresholds must be positive")
        if not 0 < self.progressive_widening_exponent <= 1:
            raise ValueError("progressive_widening_exponent must be in (0, 1]")


@dataclass
class SolverNode:
    node_id: str
    round_index: int
    root_id: str
    solver_relpath: str
    generation_mode: str
    primary_parent_id: str | None = None
    reference_node_id: str | None = None
    parent_selection: str = "unspecified"
    reference_reason: str | None = None
    status: str = "pending"
    valid: bool | None = None
    score: float | None = None
    gap_unit: str = "percent"
    metric_name: str = "optimality_gap"
    metric_unit: str = "percent"
    instance_scores: list[float] | None = None
    instance_gaps: list[float] | None = None
    runtime_sec: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    solver_sha256: str | None = None
    solver_file_sha256: dict[str, str] | None = None
    mode_weights: dict[str, float] = field(
        default_factory=lambda: {"continue": 0.0, "redesign": 0.0}
    )
    decision_id: str | None = None
    action: str | None = None
    scope: dict[str, Any] | None = None
    hypothesis_id: str | None = None
    evaluation_ids: list[str] = field(default_factory=list)


@dataclass
class SearchState:
    version: int
    run_id: str
    baseline_path: str
    baseline_sha256: str
    config: SearchConfig
    nodes: list[SolverNode] = field(default_factory=list)
    next_node_index: int = 0
    current_round: int = -1
    completed_rounds: list[int] = field(default_factory=list)
    pending_node_ids: list[str] = field(default_factory=list)
    temperature: float = 1.0
    rounds_without_global_improvement: int = 0
    force_fresh_next_round: bool = False
    best_node_id: str | None = None
    rng_state: Any = None
    decisions: list[PlannerDecision] = field(default_factory=list)
    evaluations: list[EvaluationRecord] = field(default_factory=list)
    hypotheses: list[MechanismHypothesis] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    stopped_reason: str | None = None
    stopped_at_round: int | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        baseline_path: str,
        baseline_sha256: str,
        config: SearchConfig,
    ) -> "SearchState":
        rng = random.Random(config.random_seed)
        return cls(
            version=STATE_VERSION,
            run_id=run_id,
            baseline_path=baseline_path,
            baseline_sha256=baseline_sha256,
            config=config,
            temperature=config.initial_temperature,
            rng_state=_state_to_json(rng.getstate()),
        )

    def require_node(self, node_id: str) -> SolverNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def allocate_node(
        self,
        *,
        round_index: int,
        generation_mode: str,
        solver_relpath: str,
        primary_parent_id: str | None = None,
        reference_node_id: str | None = None,
        parent_selection: str = "unspecified",
        reference_reason: str | None = None,
        decision: PlannerDecision | None = None,
    ) -> SolverNode:
        if generation_mode not in GENERATION_MODES:
            raise ValueError(f"unknown generation mode: {generation_mode}")
        if primary_parent_id is None and reference_node_id is not None:
            raise ValueError("fresh nodes cannot have a reference")

        node_id = f"node_{self.next_node_index:04d}"
        self.next_node_index += 1
        if primary_parent_id is None:
            root_id = node_id
            mode_weights = {"continue": 0.0, "redesign": 0.0}
        else:
            parent = self.require_node(primary_parent_id)
            root_id = parent.root_id
            mode_weights = dict(parent.mode_weights)
        if reference_node_id is not None:
            self.require_node(reference_node_id)

        node = SolverNode(
            node_id=node_id,
            round_index=round_index,
            root_id=root_id,
            solver_relpath=solver_relpath,
            generation_mode=generation_mode,
            primary_parent_id=primary_parent_id,
            reference_node_id=reference_node_id,
            parent_selection=parent_selection,
            reference_reason=reference_reason,
            mode_weights=mode_weights,
            decision_id=decision.decision_id if decision is not None else None,
            action=decision.action.value if decision is not None else None,
            scope=(
                {
                    "kind": decision.scope.kind,
                    "symbols": list(decision.scope.symbols),
                }
                if decision is not None
                else None
            ),
            hypothesis_id=(
                decision.hypothesis.hypothesis_id
                if decision is not None and decision.hypothesis is not None
                else decision.target_hypothesis_id if decision is not None else None
            ),
        )
        self.nodes.append(node)
        self.pending_node_ids.append(node_id)
        if decision is not None:
            self.decisions.append(decision)
            if decision.hypothesis is not None:
                self.hypotheses.append(decision.hypothesis)
        return node

    def record_evaluation(
        self,
        node_id: str,
        *,
        valid: bool,
        score: float | None,
        gap_unit: str,
        instance_gaps: list[float] | None,
        runtime_sec: float,
        solver_sha256: str | None = None,
        solver_file_sha256: dict[str, str] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        node = self.require_node(node_id)
        if node.status != "pending":
            raise RuntimeError(f"node {node_id} is already evaluated")
        if valid and score is None:
            raise ValueError("valid nodes require a score")
        if not valid and score is not None:
            raise ValueError("invalid nodes cannot have a score")

        node.status = "valid" if valid else "invalid"
        node.valid = valid
        node.score = score
        node.gap_unit = gap_unit
        node.instance_gaps = list(instance_gaps) if instance_gaps is not None else None
        node.runtime_sec = runtime_sec
        node.solver_sha256 = solver_sha256
        node.solver_file_sha256 = (
            dict(solver_file_sha256) if solver_file_sha256 is not None else None
        )
        node.error_type = error_type
        node.error_message = error_message
        evaluation_id = f"eval_{node_id}_{len(node.evaluation_ids):03d}"
        record = EvaluationRecord(
            evaluation_id=evaluation_id,
            candidate_id=node_id,
            split="train",
            seeds=(self.config.random_seed,) if valid else (),
            instance_ids=(
                tuple(f"train_{index:04d}" for index in range(len(instance_gaps)))
                if instance_gaps is not None
                else ()
            ),
            gap_matrix=(tuple(instance_gaps),) if instance_gaps is not None else (),
            runtime_matrix=(),
            valid=valid,
            total_runtime_sec=runtime_sec,
            error_type=error_type,
            error_message=error_message,
        )
        self.evaluations.append(record)
        node.evaluation_ids.append(evaluation_id)
        self.pending_node_ids.remove(node_id)

    def record_evaluation_record(
        self, record: EvaluationRecord, *, update_node: bool = True
    ) -> None:
        node = self.require_node(record.candidate_id)
        if any(item.evaluation_id == record.evaluation_id for item in self.evaluations):
            raise RuntimeError(f"duplicate evaluation id: {record.evaluation_id}")
        self.evaluations.append(record)
        node.evaluation_ids.append(record.evaluation_id)
        if not update_node:
            return
        if record.split == "train" and record.valid and record.mean_score is not None:
            node.status = "valid"
            node.valid = True
            node.score = record.mean_score
            node.metric_name, node.metric_unit = record.metric_name, record.metric_unit
            node.gap_unit = "percent" if record.metric_name == "optimality_gap" else "not_applicable"
            node.instance_gaps = _mean_instance_scores(record) if record.metric_name == "optimality_gap" else None
            node.instance_scores = _mean_instance_scores(record) if record.metric_name != "optimality_gap" else None
            node.runtime_sec = record.total_runtime_sec
            node.error_type = None
            node.error_message = None
            if node.node_id in self.pending_node_ids:
                self.pending_node_ids.remove(node.node_id)
        elif record.split == "train" and not record.valid:
            node.status = "invalid"
            node.valid = False
            node.score = None
            node.instance_gaps = None
            node.runtime_sec = record.total_runtime_sec
            node.error_type = record.error_type
            node.error_message = record.error_message
            if node.node_id in self.pending_node_ids:
                self.pending_node_ids.remove(node.node_id)

    def finish_round(self, round_index: int, *, require_nodes: bool = True) -> None:
        pending = [
            node.node_id
            for node in self.nodes
            if node.round_index == round_index and node.status == "pending"
        ]
        if pending:
            raise RuntimeError(f"round {round_index} has pending nodes: {pending}")
        if round_index in self.completed_rounds:
            raise RuntimeError(f"round {round_index} is already complete")
        if require_nodes and not any(node.round_index == round_index for node in self.nodes):
            raise RuntimeError(f"round {round_index} has no nodes")
        self.completed_rounds.append(round_index)
        self.current_round = max(self.current_round, round_index)

    def stop_search(self, *, round_index: int, reason: str) -> None:
        if self.stopped_reason is not None:
            raise RuntimeError("search is already stopped")
        if not reason.strip():
            raise ValueError("stop reason cannot be empty")
        if self.pending_node_ids:
            raise RuntimeError("cannot stop search with pending candidates")
        self.stopped_reason = reason
        self.stopped_at_round = round_index

    def require_search_open(self) -> None:
        if self.stopped_reason is not None:
            raise RuntimeError(f"search is stopped: {self.stopped_reason}")

    def restore_rng(self) -> random.Random:
        rng = random.Random()
        rng.setstate(_state_from_json(self.rng_state))
        return rng

    def capture_rng(self, rng: random.Random) -> None:
        self.rng_state = _state_to_json(rng.getstate())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchState":
        version = data.get("version")
        if version not in {1, STATE_VERSION}:
            raise ValueError(f"unsupported search state version: {version}")
        payload = dict(data)
        payload["version"] = STATE_VERSION
        payload["config"] = SearchConfig(**payload["config"])
        payload["nodes"] = [SolverNode(**node) for node in payload.get("nodes", [])]
        payload["decisions"] = [
            PlannerDecision.from_dict(item) for item in payload.get("decisions", [])
        ]
        payload["evaluations"] = [
            EvaluationRecord.from_dict(item) for item in payload.get("evaluations", [])
        ]
        payload["hypotheses"] = [
            MechanismHypothesis.from_dict(item)
            for item in payload.get("hypotheses", [])
        ]
        payload["evidence"] = [
            EvidenceRecord.from_dict(item) for item in payload.get("evidence", [])
        ]
        payload.setdefault("stopped_reason", None)
        payload.setdefault("stopped_at_round", None)
        if version == 1:
            for node in payload["nodes"]:
                node.evaluation_ids = []
            payload["evaluations"] = _migrate_v1_evaluations(
                payload["nodes"], payload["config"]
            )
            for node, record in zip(
                (item for item in payload["nodes"] if item.status != "pending"),
                payload["evaluations"],
            ):
                node.evaluation_ids.append(record.evaluation_id)
        return cls(**payload)


def save_search_state(path: str | Path, state: SearchState) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)


def load_search_state(path: str | Path) -> SearchState:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return SearchState.from_dict(data)


def _state_to_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_state_to_json(item) for item in value]
    return value


def _state_from_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_state_from_json(item) for item in value)
    return value


def _mean_instance_scores(record: EvaluationRecord) -> list[float]:
    return [
        sum(row[column] for row in record.score_matrix) / len(record.score_matrix)
        for column in range(len(record.instance_ids))
    ]


def _migrate_v1_evaluations(
    nodes: list[SolverNode], config: SearchConfig
) -> list[EvaluationRecord]:
    records = []
    for node in nodes:
        if node.status == "pending":
            continue
        gaps = tuple(node.instance_gaps or ())
        valid = bool(node.valid and gaps)
        records.append(
            EvaluationRecord(
                evaluation_id=f"legacy_eval_{node.node_id}",
                candidate_id=node.node_id,
                split="train",
                seeds=(config.random_seed,) if valid else (),
                instance_ids=tuple(
                    f"train_{index:04d}" for index in range(len(gaps))
                ),
                gap_matrix=(gaps,) if valid else (),
                runtime_matrix=(),
                valid=valid,
                total_runtime_sec=float(node.runtime_sec or 0.0),
                error_type=node.error_type,
                error_message=node.error_message,
            )
        )
    return records
