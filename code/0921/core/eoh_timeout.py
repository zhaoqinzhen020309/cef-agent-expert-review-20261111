from __future__ import annotations

import importlib.abc
import copy
import random
import multiprocessing
import os
import queue as queue_module
import signal
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from agent_ahd.core.candidate_loader import load_candidate_strategy
from agent_ahd.core.code_checks import DEFAULT_ENTRY_CONTRACT
from agent_ahd.core.diagnostic_instrumentation import (
    finish_diagnostics as _finish_diagnostics,
    instrument_diagnostic_tree as _instrument_diagnostic_tree,
)
from agent_ahd.core.windows_job import WindowsJob, WindowsJobError
from agent_ahd.core.solution_contract import normalize_structured_solution
from agent_ahd.problems.base import (
    CandidateExecutionResult,
    EntryContract,
    EvaluationBudget,
    LazyInstance,
)


_MP_CTX = multiprocessing.get_context("spawn")
_PROCESS_START_LOCK = threading.Lock()
_CANDIDATE_EXECUTION_MODE = "direct_process_v2"
_TIMEOUT_TIMING_VERSION = "timeout_timing_v1"
_SOLVER_KILL_GRACE_SEC = 0.05
_PROCESS_CLEANUP_WALL_CAP_SEC = 5.0
_PROCESS_TREE_ASSIGNMENT_TIMEOUT_SEC = 5.0


def candidate_execution_contract() -> dict[str, object]:
    return {
        "mode": _CANDIDATE_EXECUTION_MODE,
        "sandbox_required": False,
        "timeout_contract": {
            "id": "strict_parent_deadline_v2_constructor_included",
            "solver_budget_scope": "per_seed_per_instance",
            "solver_budget_source": "EvaluationBudget.time_limit",
            "budget_start": "parent_start_release",
            "deadline_owner": "parent",
            "enforcement_clock": "monotonic",
            "solver_api_deadline": "parent_wall_clock_projection",
            "deadline_rule": "now_gte_deadline_is_timeout",
            "verdict_precedence": "deadline_before_result",
            "startup_timeout_source": "candidate_timeout",
            "non_solver_timeout_source": "candidate_timeout",
            "cleanup_after_verdict": True,
            "cleanup_excluded_from_solver_runtime": True,
            "kill_grace_sec": _SOLVER_KILL_GRACE_SEC,
            "cleanup_cap_sec": _PROCESS_CLEANUP_WALL_CAP_SEC,
            "timing_record_version": _TIMEOUT_TIMING_VERSION,
        },
        "process_tree_contract": {
            "backend": (
                "windows_job_object_v1"
                if sys.platform == "win32"
                else "posix_process_group_v1"
            ),
            "assignment": "before_candidate_import",
            "assignment_timeout_sec": _PROCESS_TREE_ASSIGNMENT_TIMEOUT_SEC,
            "kill_on_owner_close": sys.platform == "win32",
            "active_process_zero_required": True,
            "cleanup_failure": "fatal_run",
        },
    }


class InvalidRouteError(ValueError):
    """Raised when a candidate returns a route outside the entry contract."""


class CandidateProcessCleanupError(RuntimeError):
    """Raised when the evaluator cannot prove that a candidate worker stopped."""


def _create_process_tree_owner() -> WindowsJob | None:
    return WindowsJob.create() if sys.platform == "win32" else None


def _process_tree_probe_worker(candidate_start_allowed) -> None:
    if not candidate_start_allowed.wait(_PROCESS_TREE_ASSIGNMENT_TIMEOUT_SEC):
        return
    if sys.platform != "win32":
        os.setsid()


@lru_cache(maxsize=1)
def probe_candidate_process_tree_capability() -> dict[str, object]:
    backend = (
        "windows_job_object_v1"
        if sys.platform == "win32"
        else "posix_process_group_v1"
    )
    if sys.platform != "win32":
        return {
            "status": "passed",
            "backend": backend,
            "assignment": "before_candidate_import",
        }

    owner = WindowsJob.create()
    candidate_start_allowed = _MP_CTX.Event()
    process = _MP_CTX.Process(
        target=_process_tree_probe_worker,
        args=(candidate_start_allowed,),
    )
    try:
        with _PROCESS_START_LOCK:
            process.start()
        if process.pid is None:
            raise RuntimeError("process-tree probe worker has no process id")
        owner.assign_pid(process.pid)
        candidate_start_allowed.set()
        process.join(_PROCESS_TREE_ASSIGNMENT_TIMEOUT_SEC)
        if process.is_alive():
            raise RuntimeError("process-tree probe worker did not exit")
        if not owner.terminate_and_wait(
            timeout_sec=_PROCESS_CLEANUP_WALL_CAP_SEC
        ):
            raise RuntimeError("process-tree probe Job Object remained active")
        return {
            "status": "passed",
            "backend": backend,
            "assignment": "before_candidate_import",
        }
    except Exception as exc:
        if process.pid is not None and process.is_alive():
            _kill_process_tree(
                process,
                wall_cap_sec=_PROCESS_CLEANUP_WALL_CAP_SEC,
            )
        raise RuntimeError(
            f"candidate process-tree capability probe failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        owner.close()


class _BlockProjectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "agent_ahd" or fullname.startswith("agent_ahd."):
            raise ModuleNotFoundError("candidate cannot import the parent project")
        return None


def execute_candidate_with_timeout(
    *,
    candidate_id: str,
    candidate_path: str | Path,
    instances: list[np.ndarray],
    budget: EvaluationBudget,
    timeout: float,
    entry_contract: EntryContract = DEFAULT_ENTRY_CONTRACT,
    collect_diagnostics: bool = False,
) -> CandidateExecutionResult:
    try:
        process_tree_owner = _create_process_tree_owner()
    except WindowsJobError as exc:
        raise CandidateProcessCleanupError(
            f"candidate process-tree owner could not be created: {exc}"
        ) from exc
    try:
        return _execute_candidate_with_timeout(
            candidate_id=candidate_id,
            candidate_path=candidate_path,
            instances=instances,
            budget=budget,
            timeout=timeout,
            entry_contract=entry_contract,
            collect_diagnostics=collect_diagnostics,
            process_tree_owner=process_tree_owner,
        )
    finally:
        if process_tree_owner is not None:
            try:
                process_tree_owner.close()
            except WindowsJobError as exc:
                raise CandidateProcessCleanupError(
                    f"candidate process-tree owner could not be closed: {exc}"
                ) from exc


def _execute_candidate_with_timeout(
    *,
    candidate_id: str,
    candidate_path: str | Path,
    instances: list[np.ndarray],
    budget: EvaluationBudget,
    timeout: float,
    entry_contract: EntryContract,
    collect_diagnostics: bool,
    process_tree_owner: WindowsJob | None,
) -> CandidateExecutionResult:
    result_queue = _MP_CTX.Queue(maxsize=1)
    candidate_start_allowed = _MP_CTX.Event()
    candidate_loaded = _MP_CTX.Event()
    solve_ready = _MP_CTX.Event()
    start_solver_call = _MP_CTX.Event()
    solve_started = _MP_CTX.Event()
    solve_finished = _MP_CTX.Event()
    finish_ack = _MP_CTX.Event()
    solver_wall_deadline = _MP_CTX.Value("d", 0.0)
    solver_monotonic_deadline = _MP_CTX.Value("d", 0.0)
    process = _MP_CTX.Process(
        target=_candidate_worker,
        args=(
            result_queue,
            candidate_start_allowed,
            candidate_loaded,
            solve_ready,
            start_solver_call,
            solve_started,
            solve_finished,
            finish_ack,
            solver_wall_deadline,
            solver_monotonic_deadline,
            candidate_id,
            Path(candidate_path),
            instances,
            budget,
            entry_contract,
            collect_diagnostics,
        ),
    )
    startup_start = time.monotonic()
    with _PROCESS_START_LOCK:
        process.start()
    try:
        if process_tree_owner is not None:
            if process.pid is None:
                raise RuntimeError("candidate worker has no process id")
            process_tree_owner.assign_pid(process.pid)
        candidate_start_allowed.set()
    except Exception as exc:
        reaped = _kill_process_tree(process)
        raise CandidateProcessCleanupError(
            f"candidate process-tree assignment failed before candidate import; "
            f"candidate_id={candidate_id}; pid={process.pid}; "
            f"worker_reaped={str(reaped and not process.is_alive()).lower()}; "
            f"cause={type(exc).__name__}: {exc}"
        ) from exc
    startup_deadline = startup_start + timeout
    startup_elapsed = 0.0
    solver_start: float | None = None
    solver_runtime = 0.0
    non_solver_deadline = startup_deadline
    candidate_load_observed = False

    def timeout_result(*, detected_at: float, phase: str) -> CandidateExecutionResult:
        cleanup_elapsed, worker_reaped = _cleanup_timed_out_process(
            process,
            timeout_detected_at=detected_at,
            process_tree_owner=process_tree_owner,
        )
        if not worker_reaped:
            raise CandidateProcessCleanupError(
                f"candidate worker could not be reaped; candidate_id={candidate_id}; "
                f"phase={phase}; pid={process.pid}; "
                f"cleanup_elapsed_sec={cleanup_elapsed:.3f}; "
                f"cleanup_cap_sec={_PROCESS_CLEANUP_WALL_CAP_SEC:.3f}"
            )
        if phase == "startup":
            elapsed = max(detected_at - startup_start, 0.0)
            return CandidateExecutionResult(
                candidate_id=candidate_id,
                valid=False,
                routes=None,
                runtime_sec=0.0,
                seeds=list(budget.resolved_seeds),
                error_type="startup_timeout",
                error_message=(
                    f"candidate startup exceeded {timeout:.3f}s; "
                    f"startup_elapsed_sec={elapsed:.3f}; "
                    f"cleanup_elapsed_sec={cleanup_elapsed:.3f}; "
                    "worker_reaped=true"
                ),
                timeout_timing=_timeout_timing(
                    phase="startup",
                    timeout_limit_sec=timeout,
                    elapsed_at_detection_sec=elapsed,
                    cleanup_elapsed_sec=cleanup_elapsed,
                    startup_elapsed_sec=elapsed,
                    worker_reaped=True,
                ),
            )
        if phase == "non_solver":
            overhead_elapsed = max(
                detected_at - (non_solver_deadline - timeout),
                0.0,
            )
            return CandidateExecutionResult(
                candidate_id=candidate_id,
                valid=False,
                routes=None,
                runtime_sec=solver_runtime,
                seeds=list(budget.resolved_seeds),
                error_type="candidate_overhead_timeout",
                error_message=(
                    f"candidate non-solver processing exceeded {timeout:.3f}s; "
                    f"solver_runtime_sec={solver_runtime:.3f}; "
                    f"startup_elapsed_sec={startup_elapsed:.3f}; "
                    f"cleanup_elapsed_sec={cleanup_elapsed:.3f}; "
                    "worker_reaped=true"
                ),
                timeout_timing=_timeout_timing(
                    phase="non_solver",
                    timeout_limit_sec=timeout,
                    elapsed_at_detection_sec=overhead_elapsed,
                    cleanup_elapsed_sec=cleanup_elapsed,
                    startup_elapsed_sec=startup_elapsed,
                    worker_reaped=True,
                ),
            )
        if solver_start is None:
            raise RuntimeError("solver timeout detected without a solver start")
        solver_elapsed_at_detection = max(detected_at - solver_start, 0.0)
        charged_solver_runtime = min(
            solver_elapsed_at_detection,
            budget.time_limit,
        )
        return CandidateExecutionResult(
            candidate_id=candidate_id,
            valid=False,
            routes=None,
            runtime_sec=charged_solver_runtime,
            seeds=list(budget.resolved_seeds),
            error_type="timeout",
            error_message=(
                "candidate solver exceeded its budget; "
                f"solver_budget_sec={budget.time_limit:.3f}; "
                f"solver_elapsed_at_detection_sec={solver_elapsed_at_detection:.3f}; "
                f"startup_elapsed_sec={startup_elapsed:.3f}; "
                f"cleanup_elapsed_sec={cleanup_elapsed:.3f}; "
                "worker_reaped=true"
            ),
            timeout_timing=_timeout_timing(
                phase="solver",
                timeout_limit_sec=budget.time_limit,
                elapsed_at_detection_sec=solver_elapsed_at_detection,
                cleanup_elapsed_sec=cleanup_elapsed,
                startup_elapsed_sec=startup_elapsed,
                worker_reaped=True,
            ),
        )

    deadline = startup_deadline
    while process.is_alive():
        now = time.monotonic()
        if candidate_loaded.is_set() and not candidate_load_observed:
            loaded_observed_at = time.monotonic()
            if loaded_observed_at >= startup_deadline:
                return timeout_result(
                    detected_at=loaded_observed_at,
                    phase="startup",
                )
            startup_elapsed = loaded_observed_at - startup_start
            non_solver_deadline = loaded_observed_at + timeout
            candidate_load_observed = True
            now = loaded_observed_at
        if candidate_load_observed and solve_ready.is_set() and solver_start is None:
            ready_observed_at = time.monotonic()
            if ready_observed_at >= non_solver_deadline:
                return timeout_result(
                    detected_at=ready_observed_at,
                    phase="non_solver",
                )
            solver_start = ready_observed_at
            solver_wall_deadline.value = time.time() + budget.time_limit
            solver_monotonic_deadline.value = solver_start + budget.time_limit
            start_solver_call.set()
            solve_ready.clear()
            now = solver_start
        deadline = (
            float(solver_monotonic_deadline.value)
            if solver_start is not None
            else non_solver_deadline
        )
        remaining = deadline - now
        if remaining <= 0:
            phase = (
                "startup"
                if not candidate_load_observed
                else "non_solver" if solver_start is None else "solver"
            )
            return timeout_result(detected_at=now, phase=phase)
        if solve_finished.is_set() and solver_start is not None:
            finish_observed_at = time.monotonic()
            if finish_observed_at >= deadline:
                return timeout_result(detected_at=finish_observed_at, phase="solver")
            solver_runtime += finish_observed_at - solver_start
            solver_start = None
            solver_wall_deadline.value = 0.0
            solver_monotonic_deadline.value = 0.0
            non_solver_deadline = finish_observed_at + timeout
            deadline = non_solver_deadline
            remaining = deadline - finish_observed_at
            solve_started.clear()
            solve_finished.clear()
            finish_ack.set()
        try:
            payload = result_queue.get(timeout=min(0.01, remaining))
        except queue_module.Empty:
            continue
        payload_observed_at = time.monotonic()
        if payload_observed_at >= deadline:
            phase = (
                "startup"
                if not candidate_load_observed
                else "non_solver" if solver_start is None else "solver"
            )
            return timeout_result(detected_at=payload_observed_at, phase=phase)
        process.join(_SOLVER_KILL_GRACE_SEC)
        worker_reaped = _reap_process_tree(
            process,
            process_tree_owner=process_tree_owner,
            wall_cap_sec=_PROCESS_CLEANUP_WALL_CAP_SEC,
        )
        if not worker_reaped:
            raise CandidateProcessCleanupError(
                f"candidate worker could not be reaped after payload; "
                f"candidate_id={candidate_id}; pid={process.pid}; "
                f"cleanup_cap_sec={_PROCESS_CLEANUP_WALL_CAP_SEC:.3f}"
            )
        return CandidateExecutionResult(**payload)

    process.join()
    exit_observed_at = time.monotonic()
    if exit_observed_at >= deadline:
        phase = (
            "startup"
            if not candidate_load_observed
            else "non_solver" if solver_start is None else "solver"
        )
        return timeout_result(detected_at=exit_observed_at, phase=phase)
    if not _reap_process_tree(
        process,
        process_tree_owner=process_tree_owner,
        wall_cap_sec=_PROCESS_CLEANUP_WALL_CAP_SEC,
    ):
        raise CandidateProcessCleanupError(
            f"candidate process tree remained active after worker exit; "
            f"candidate_id={candidate_id}; pid={process.pid}; "
            f"cleanup_cap_sec={_PROCESS_CLEANUP_WALL_CAP_SEC:.3f}"
        )
    try:
        payload = result_queue.get(timeout=1.0)
    except queue_module.Empty:
        return CandidateExecutionResult(
            candidate_id=candidate_id,
            valid=False,
            routes=None,
            runtime_sec=(
                0.0
                if solver_start is None
                else solver_runtime + exit_observed_at - solver_start
            ),
            seeds=list(budget.resolved_seeds),
            error_type="worker_crash",
            error_message=f"exitcode={process.exitcode}",
        )
    return CandidateExecutionResult(**payload)


def _candidate_worker(
    result_queue,
    candidate_start_allowed,
    candidate_loaded,
    solve_ready,
    start_solver_call,
    solve_started,
    solve_finished,
    finish_ack,
    solver_wall_deadline,
    solver_monotonic_deadline,
    candidate_id: str,
    candidate_path: Path,
    instances: list[np.ndarray],
    budget: EvaluationBudget,
    entry_contract: EntryContract,
    collect_diagnostics: bool,
) -> None:
    if not candidate_start_allowed.wait(_PROCESS_TREE_ASSIGNMENT_TIMEOUT_SEC):
        return
    if sys.platform != "win32":
        os.setsid()

    startup_start = time.monotonic()
    _isolate_candidate_environment(candidate_path.parent)
    load_result = load_candidate_strategy(
        candidate_id=candidate_id,
        candidate_path=candidate_path,
        candidate_root=candidate_path.parent,
        entry_contract=entry_contract,
        source_transform=(
            _instrument_diagnostic_tree if collect_diagnostics else None
        ),
    )
    if not load_result.valid or load_result.strategy_cls is None:
        result_queue.put({
            "candidate_id": candidate_id,
            "valid": False,
            "routes": None,
            "runtime_sec": 0.0,
            "error_type": load_result.error_type,
            "error_message": (
                f"{load_result.error_message}; "
                f"startup_elapsed_sec={time.monotonic() - startup_start:.3f}"
            ),
        })
        return

    candidate_loaded.set()
    try:
        routes_by_seed, runtime_matrix, termination_matrix, diagnostics_matrix = _run_loaded_strategy(
            load_result.strategy_cls,
            instances,
            budget,
            entry_contract=entry_contract,
            collect_diagnostics=collect_diagnostics,
            solve_ready=solve_ready,
            start_solver_call=start_solver_call,
            solve_started=solve_started,
            solve_finished=solve_finished,
            finish_ack=finish_ack,
            solver_wall_deadline=solver_wall_deadline,
            solver_monotonic_deadline=solver_monotonic_deadline,
        )
        solver_runtime = sum(sum(row) for row in runtime_matrix)
        result = {
            "candidate_id": candidate_id,
            "valid": True,
            "routes": routes_by_seed[0],
            "routes_by_seed": routes_by_seed,
            "runtime_matrix": runtime_matrix,
            "termination_matrix": termination_matrix,
            "diagnostics_matrix": diagnostics_matrix,
            "seeds": list(budget.resolved_seeds),
            "runtime_sec": solver_runtime,
            "error_type": None,
            "error_message": None,
        }
    except InvalidRouteError as exc:
        result = {
            "candidate_id": candidate_id,
            "valid": False,
            "routes": None,
            "runtime_sec": 0.0,
            "error_type": "invalid_route",
            "error_message": str(exc),
        }
    except Exception as exc:
        result = {
            "candidate_id": candidate_id,
            "valid": False,
            "routes": None,
            "runtime_sec": 0.0,
            "error_type": "candidate_error",
            "error_message": f"{type(exc).__name__}: {exc}",
        }
    result_queue.put(result)


def _run_loaded_strategy(
    strategy_cls: type,
    instances: list[np.ndarray],
    budget: EvaluationBudget,
    *,
    entry_contract: EntryContract = DEFAULT_ENTRY_CONTRACT,
    collect_diagnostics: bool,
    solve_ready,
    start_solver_call,
    solve_started,
    solve_finished,
    finish_ack,
    solver_wall_deadline,
    solver_monotonic_deadline,
) -> tuple[
    list[list[object]],
    list[list[float]],
    list[list[str]],
    list[list[dict[str, object]]] | None,
]:
    routes_by_seed: list[list[object]] = []
    runtime_matrix: list[list[float]] = []
    termination_matrix: list[list[str]] = []
    diagnostics_matrix: list[list[dict[str, object]]] = []
    for seed in budget.resolved_seeds:
        seed_routes = []
        seed_runtimes = []
        seed_terminations = []
        seed_diagnostics = []
        for instance_reference in instances:
            # Load one immutable instance outside solver time, inside the
            # parent's non-solver watchdog. No full dataset is pickled per worker.
            instance = (instance_reference.materialize()
                        if isinstance(instance_reference, LazyInstance) else instance_reference)
            solve_ready.set()
            start_solver_call.wait()
            start_solver_call.clear()
            solve_ready.clear()
            deadline = float(solver_wall_deadline.value)
            if deadline <= 0.0 or float(solver_monotonic_deadline.value) <= 0.0:
                raise RuntimeError("parent did not freeze the solver deadline")
            solve_started.set()
            solve_start = time.monotonic()
            try:
                if entry_contract.python_random_seed_policy == "reset_per_instance":
                    random.seed(int(seed))
                strategy = strategy_cls()
                if entry_contract.input_kind == "constructive_tsp":
                    route = _construct_tsp(strategy, entry_contract.method_name, instance, seed)
                else:
                    payload = (copy.deepcopy(instance) if entry_contract.input_kind == "mapping"
                               else np.array(instance, copy=True))
                    route = getattr(strategy, entry_contract.method_name)(
                        payload, deadline, budget.ite_max, seed,
                    )
                if entry_contract.output_kind == "routes":
                    route = _normalize_routes(route, node_count=len(instance["distance_matrix"]))
                elif entry_contract.output_kind == "structured":
                    route = normalize_structured_solution(route)
                else:
                    route = _normalize_route(route, node_count=len(instance))
            finally:
                solve_elapsed = time.monotonic() - solve_start
                solve_finished.set()
                finish_ack.wait()
                finish_ack.clear()
                solve_finished.clear()
                solve_started.clear()
            counters = None
            if collect_diagnostics:
                counters = _finish_diagnostics(
                    getattr(strategy, "_ahd_diagnostics", None),
                    deadline=deadline,
                    effective_iteration_limit=getattr(
                        strategy,
                        "_ahd_effective_iteration_limit",
                        None,
                    ),
                )
            seed_terminations.append(
                counters["termination_reason"]
                if counters is not None
                else (
                    "deadline_reached"
                    if time.time() >= deadline
                    else "solver_returned_before_deadline"
                )
            )
            seed_runtimes.append(solve_elapsed)
            seed_routes.append(route)
            if counters is not None:
                seed_diagnostics.append(counters)
        routes_by_seed.append(seed_routes)
        runtime_matrix.append(seed_runtimes)
        termination_matrix.append(seed_terminations)
        if collect_diagnostics:
            diagnostics_matrix.append(seed_diagnostics)
    return (
        routes_by_seed,
        runtime_matrix,
        termination_matrix,
        diagnostics_matrix if collect_diagnostics else None,
    )


def _construct_tsp(strategy, method_name: str, instance, seed: int) -> list[int]:
    """Frozen greedy scaffold: only the next-city policy is candidate code."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    distances = np.array(instance, copy=True)
    distances.flags.writeable = False
    node_count = len(distances)
    remaining = np.ones(node_count, dtype=bool)
    remaining[0] = False
    tour = [0]
    select = getattr(strategy, method_name)
    for _ in range(node_count - 1):
        unvisited = np.flatnonzero(remaining)
        node = select(tour[-1], 0, unvisited, distances)
        if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)):
            raise InvalidRouteError("next-city policy must return an integer")
        node = int(node)
        if node < 0 or node >= node_count or not remaining[node]:
            raise InvalidRouteError("next-city policy returned an unavailable city")
        tour.append(node)
        remaining[node] = False
    return tour


def _normalize_routes(routes: object, *, node_count: int) -> list[list[int]]:
    if type(routes) not in {list, tuple} or not 0 < len(routes) < node_count:
        raise InvalidRouteError("CVRP output must be a nonempty list of customer routes")
    result, visited = [], set()
    for route in routes:
        if type(route) not in {list, tuple, np.ndarray}:
            raise InvalidRouteError("CVRP route must be an integer sequence")
        if type(route) is np.ndarray and route.ndim != 1:
            raise InvalidRouteError("CVRP route must be one-dimensional")
        if not len(route) or len(route) >= node_count:
            raise InvalidRouteError("CVRP route has invalid length")
        row = []
        for node in route:
            if isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer)):
                raise InvalidRouteError("CVRP customers must be integers")
            node = int(node)
            if node <= 0 or node >= node_count or node in visited:
                raise InvalidRouteError("CVRP customer omitted, duplicated, or out of range")
            visited.add(node)
            row.append(node)
        result.append(row)
    if visited != set(range(1, node_count)):
        raise InvalidRouteError("CVRP routes must cover every customer exactly once")
    return result


def _normalize_route(route: object, *, node_count: int) -> list[int]:
    route_type = type(route)
    if route_type is np.ndarray:
        if route.ndim != 1:
            raise InvalidRouteError("route ndarray must be one-dimensional")
        normalized = route.tolist()
    elif route_type is list:
        normalized = list(route)
    elif route_type is tuple:
        normalized = list(route)
    else:
        raise InvalidRouteError(
            "route must be an exact built-in list, tuple, or numpy.ndarray"
        )
    if len(normalized) != node_count:
        raise InvalidRouteError(
            f"route length {len(normalized)} does not match node count {node_count}"
        )
    for value in normalized:
        value_type = type(value)
        if value_type is not int and not issubclass(value_type, np.integer):
            raise InvalidRouteError(
                "route elements must be built-in or numpy integers"
            )
    normalized = [int(value) for value in normalized]
    if any(value < 0 or value >= node_count for value in normalized):
        raise InvalidRouteError("route contains an out-of-range node")
    if len(set(normalized)) != node_count:
        raise InvalidRouteError("route must contain every node exactly once")
    return normalized


def _isolate_candidate_environment(candidate_root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    candidate_root = candidate_root.resolve()
    os.chdir(candidate_root)

    retained_paths = []
    for value in sys.path:
        if not value:
            continue
        try:
            resolved = Path(value).resolve()
        except OSError:
            retained_paths.append(value)
            continue
        if resolved == repository_root or resolved.is_relative_to(repository_root):
            continue
        retained_paths.append(value)
    sys.path[:] = [str(candidate_root), *retained_paths]

    for module_name in list(sys.modules):
        if module_name == "agent_ahd" or module_name.startswith("agent_ahd."):
            sys.modules.pop(module_name, None)
    sys.meta_path.insert(0, _BlockProjectImports())


def _timeout_timing(
    *,
    phase: str,
    timeout_limit_sec: float,
    elapsed_at_detection_sec: float,
    cleanup_elapsed_sec: float,
    startup_elapsed_sec: float,
    worker_reaped: bool,
) -> dict[str, object]:
    return {
        "version": _TIMEOUT_TIMING_VERSION,
        "contract": _CANDIDATE_EXECUTION_MODE,
        "phase": phase,
        "timeout_limit_sec": float(timeout_limit_sec),
        "elapsed_at_detection_sec": float(elapsed_at_detection_sec),
        "cleanup_elapsed_sec": float(cleanup_elapsed_sec),
        "startup_elapsed_sec": float(startup_elapsed_sec),
        "kill_grace_sec": _SOLVER_KILL_GRACE_SEC,
        "cleanup_cap_sec": _PROCESS_CLEANUP_WALL_CAP_SEC,
        "worker_reaped": bool(worker_reaped),
    }


def _cleanup_timed_out_process(
    process: multiprocessing.Process,
    *,
    timeout_detected_at: float,
    process_tree_owner: WindowsJob | None = None,
) -> tuple[float, bool]:
    if process.is_alive() and _SOLVER_KILL_GRACE_SEC > 0.0:
        process.join(_SOLVER_KILL_GRACE_SEC)
    elapsed = max(time.monotonic() - timeout_detected_at, 0.0)
    remaining_cap = max(_PROCESS_CLEANUP_WALL_CAP_SEC - elapsed, 0.0)
    worker_reaped = _reap_process_tree(
        process,
        process_tree_owner=process_tree_owner,
        wall_cap_sec=remaining_cap,
    )
    cleanup_elapsed = max(time.monotonic() - timeout_detected_at, 0.0)
    return cleanup_elapsed, worker_reaped


def _reap_process_tree(
    process: multiprocessing.Process,
    *,
    process_tree_owner: WindowsJob | None,
    wall_cap_sec: float,
) -> bool:
    cleanup_start = time.monotonic()

    def remaining() -> float:
        return max(float(wall_cap_sec) - (time.monotonic() - cleanup_start), 0.0)

    owner_reaped = True
    if process_tree_owner is not None:
        try:
            owner_reaped = process_tree_owner.terminate_and_wait(
                timeout_sec=remaining()
            )
        except WindowsJobError:
            owner_reaped = False
    if process.is_alive() and remaining() > 0.0:
        root_reaped = _kill_process_tree(process, wall_cap_sec=remaining())
    else:
        process.join(remaining())
        root_reaped = not process.is_alive()
    return owner_reaped and root_reaped and not process.is_alive()


def _kill_process_tree(
    process: multiprocessing.Process,
    *,
    wall_cap_sec: float = _PROCESS_CLEANUP_WALL_CAP_SEC,
) -> bool:
    cleanup_deadline = time.monotonic() + max(float(wall_cap_sec), 0.0)

    def remaining() -> float:
        return max(cleanup_deadline - time.monotonic(), 0.0)

    pid = process.pid
    if sys.platform == "win32" and pid is not None and remaining() > 0.0:
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=max(remaining(), 0.001),
            )
        except subprocess.TimeoutExpired:
            pass
        process.join(remaining())
        if not process.is_alive():
            return True
    if sys.platform != "win32" and pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass
    process.join(remaining())
    if process.is_alive():
        if sys.platform != "win32" and pid is not None:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
        process.join(remaining())
    return not process.is_alive()
