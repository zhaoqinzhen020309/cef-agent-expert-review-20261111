from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from agent_ahd.core.eoh_timeout import CandidateProcessCleanupError
from agent_ahd.problems.base import CandidateTask, EvaluationResult


def evaluate_tasks(
    tasks: Iterable[CandidateTask],
    max_workers: int = 1,
    per_candidate_timeout: float | None = None,
) -> list[EvaluationResult]:
    task_list = list(tasks)
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if not task_list:
        return []

    module_names = {task.evaluator_module for task in task_list}
    if len(module_names) == 1:
        module = importlib.import_module(task_list[0].evaluator_module)
        batch_evaluate = getattr(module, "evaluate_task_batch", None)
        if batch_evaluate is not None:
            return batch_evaluate(
                task_list,
                max_workers=max_workers,
                per_candidate_timeout=per_candidate_timeout,
            )

    if max_workers == 1:
        return [_evaluate_task(task, per_candidate_timeout or _default_timeout(task)) for task in task_list]

    indexed_results: dict[int, EvaluationResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_evaluate_task, task, per_candidate_timeout or _default_timeout(task)): index
            for index, task in enumerate(task_list)
        }
        for future in as_completed(futures):
            indexed_results[futures[future]] = future.result()
    return [indexed_results[index] for index in range(len(task_list))]


def _evaluate_task(task: CandidateTask, timeout: float) -> EvaluationResult:
    try:
        module = importlib.import_module(task.evaluator_module)
        evaluate = getattr(module, task.evaluator_function)
        kwargs = {
            "candidate_timeout": timeout,
            "split": task.split,
            "instance_ids": task.instance_ids,
            "data_path": task.data_path,
        }
        if task.collect_diagnostics:
            kwargs["collect_diagnostics"] = True
        return evaluate(
            task.candidate_id,
            task.candidate_path,
            task.budget,
            **kwargs,
        )
    except CandidateProcessCleanupError:
        raise
    except Exception as exc:
        return EvaluationResult(
            candidate_id=task.candidate_id,
            valid=False,
            score=None,
            gap_unit=task.gap_unit,
            runtime_sec=0.0,
            contract_id=task.contract_id,
            metric_name=task.metric_name,
            metric_unit=task.metric_unit,
            error_type="evaluator_error",
            error_message=f"{type(exc).__name__}: {exc}",
        )


def _default_timeout(task: CandidateTask) -> float:
    per_instance = max(float(task.budget.time_limit), 0.1)
    return (
        per_instance
        * max(task.budget.n_instances, 1)
        * len(task.budget.resolved_seeds)
        + 30.0
    )
