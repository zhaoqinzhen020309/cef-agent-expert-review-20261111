from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_ahd.core.research_records import EvaluationRecord
from agent_ahd.core.search_state import SearchState, load_search_state
from agent_ahd.problems.base import CandidateTask, EvaluationResult


_PREFLIGHT_ERRORS = {
    "generation_error",
    "unchanged_candidate",
    "duplicate_source",
    "scope_violation",
    "invalid_ablation",
}


class RecordedEvaluationRunner:
    def __init__(
        self,
        source_state: SearchState,
        *,
        required_evaluation_ids: set[str] | None = None,
    ) -> None:
        self.source_state = source_state
        self._used: set[str] = set()
        self._required = (
            set(required_evaluation_ids)
            if required_evaluation_ids is not None
            else {
                record.evaluation_id
                for record in source_state.evaluations
                if record.error_type not in _PREFLIGHT_ERRORS
            }
        )

    @classmethod
    def from_run(cls, run_dir: str | Path) -> "RecordedEvaluationRunner":
        root = Path(run_dir).resolve()
        state = load_search_state(root / "search_state.json")
        required = {
            record.evaluation_id
            for record in state.evaluations
            if record.error_type not in _PREFLIGHT_ERRORS
            and not _is_source_cache_hit(root, record)
        }
        return cls(state, required_evaluation_ids=required)

    def __call__(
        self,
        tasks: list[CandidateTask],
        *,
        max_workers: int,
    ) -> list[EvaluationResult]:
        del max_workers
        return [self._result_for(task) for task in tasks]

    def assert_exhausted(self) -> None:
        remaining = sorted(self._required - self._used)
        if remaining:
            raise RuntimeError(
                f"replay has {len(remaining)} unconsumed evaluations; "
                f"first_unconsumed={remaining[:10]}; consumed={len(self._used)}"
            )

    def _result_for(self, task: CandidateTask) -> EvaluationResult:
        candidate_id = self._source_candidate_id(task.candidate_id)
        records = [
            record
            for record in self.source_state.evaluations
            if record.evaluation_id in self._required
            and record.evaluation_id not in self._used
            and record.candidate_id == candidate_id
            and record.split == task.split
            and record.error_type not in _PREFLIGHT_ERRORS
        ]
        exact = [
            record
            for record in records
            if record.seeds == task.budget.resolved_seeds
            and record.instance_ids == task.instance_ids
        ]
        if len(exact) != 1:
            raise RuntimeError(
                f"found {len(exact)} exact recorded evaluations for replay task "
                f"{task.candidate_id}; expected 1"
            )
        record = exact[0]
        self._used.add(record.evaluation_id)
        return _to_result(record, task)

    def _source_candidate_id(self, task_id: str) -> str:
        if task_id.startswith("evidence_node_"):
            return task_id.removeprefix("evidence_")
        if not task_id.startswith("retest_"):
            return task_id
        decision_id = task_id.removeprefix("retest_")
        for decision in self.source_state.decisions:
            if decision.decision_id == decision_id:
                if decision.primary_parent_id is None:
                    break
                return decision.primary_parent_id
        raise RuntimeError(f"unknown recorded retest decision: {decision_id}")


def _is_source_cache_hit(root: Path, record: EvaluationRecord) -> bool:
    journal_path = root / "nodes" / record.candidate_id / "stage_journal.json"
    if not journal_path.is_file():
        return False
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    stage = journal.get("stages", {}).get("development_evaluation", {})
    if stage.get("metadata", {}).get("cache_hit") is not True:
        return False
    record_fingerprint = hashlib.sha256(
        json.dumps(record.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return stage.get("output_fingerprint") == record_fingerprint


def _to_result(record: EvaluationRecord, task: CandidateTask) -> EvaluationResult:
    if (record.metric_name, record.metric_unit) != (task.metric_name, task.metric_unit):
        raise ValueError("replay metric differs from the submitted task")
    instance_gaps = None
    if record.valid and record.metric_name == "optimality_gap":
        instance_gaps = [
            sum(row[column] for row in record.gap_matrix) / len(record.gap_matrix)
            for column in range(len(record.instance_ids))
        ]
    return EvaluationResult(
        candidate_id=task.candidate_id,
        valid=record.valid,
        score=record.mean_score,
        metric_name=record.metric_name,
        metric_unit=record.metric_unit,
        objective_matrix=[list(row) for row in record.objective_matrix] if record.objective_matrix else None,
        objective_name=record.objective_name,
        objective_unit=record.objective_unit,
        gap_unit=task.gap_unit,
        runtime_sec=record.total_runtime_sec,
        contract_id=task.contract_id,
        error_type=record.error_type,
        error_message=record.error_message,
        timeout_timing=(
            dict(record.timeout_timing)
            if record.timeout_timing is not None
            else None
        ),
        instance_gaps=instance_gaps,
        split=record.split,
        seeds=list(record.seeds) if record.valid else None,
        instance_ids=list(record.instance_ids) if record.valid else None,
        gap_matrix=[list(row) for row in record.gap_matrix] if record.valid and record.gap_matrix else None,
        runtime_matrix=(
            [list(row) for row in record.runtime_matrix]
            if record.runtime_matrix
            else None
        ),
    )
