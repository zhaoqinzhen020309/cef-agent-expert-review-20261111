"""Offline semantic acceptance; no benchmark instances or model calls."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import time

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "problems/fssp/neh_gls_v1/solver.py"
UPSTREAM = Path(__file__).parent / "fixtures/eoh_fssp_gls_numba.py"
UPSTREAM_SHA256 = "2303b8de5881902870c743a941d43fb41f58733cf8dd18a906bf1deb2592b286"


def test_complete_initial_solver_returns_a_schedule():
    assert SOURCE.is_file(), "the complete NEH+GLS initial solver is missing"
    spec = importlib.util.spec_from_file_location("fssp_gls_initial", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = np.arange(1, 19, dtype=float).reshape(6, 3)
    result = module.SearchStrategy().solve(matrix, time.time() + 30, 1, 122)
    assert sorted(result) == list(range(6))


@pytest.fixture(scope="module")
def solver():
    assert SOURCE.is_file(), "the complete NEH+GLS initial solver is missing"
    spec = importlib.util.spec_from_file_location("fssp_gls_acceptance", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream():
    assert hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() == UPSTREAM_SHA256
    tree = ast.parse(UPSTREAM.read_text(encoding="utf-8"))
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            functions.append(node)
    namespace = {"np": np, "random": random, "time": SimpleNamespace(time=lambda: 0.0),
                 "INVALID": 1e10}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(UPSTREAM), "exec"), namespace)
    return SimpleNamespace(**namespace)


def objective(order, processing_times):
    """Independent FSSP recurrence using a full completion-time table."""
    table = np.zeros((len(order) + 1, processing_times.shape[1] + 1))
    for position, job in enumerate(order, 1):
        for machine in range(1, table.shape[1]):
            table[position, machine] = (
                max(table[position - 1, machine], table[position, machine - 1])
                + processing_times[job, machine - 1]
            )
    return table[-1, -1]


def paper_guide(sequence, matrix, machines, jobs):
    """Figure 10, arXiv:2401.02051v3; independent reference transcription."""
    subset = np.random.choice(machines, max(1, int(0.3 * machines)), replace=False)
    averages = np.average(matrix[:, subset], axis=1, weights=np.random.rand(len(subset)))
    chosen = np.argsort(averages)[-int(0.3 * jobs):]
    modified = matrix.copy()
    factors = np.random.uniform(0.8, 1.2, size=(len(chosen), len(subset)))
    modified[chosen[:, np.newaxis], subset] *= factors
    return modified, chosen


def test_frozen_upstream_reference():
    assert hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() == UPSTREAM_SHA256


def test_neh_and_neighborhoods_preserve_upstream_order(solver, upstream):
    rng = np.random.default_rng(1234)
    for jobs, machines in ((7, 3), (20, 5)):
        matrix = rng.integers(1, 100, (jobs, machines)).astype(float)
        expected, value = upstream.neh(matrix, machines, jobs)
        actual, actual_value = solver.neh(matrix, machines, jobs)
        assert actual == expected
        assert actual_value == value == objective(actual, matrix)
        for threshold in (value, value + 1000, value - 1):
            assert solver.local_search(actual, threshold, matrix, machines) == (
                upstream.local_search(expected, threshold, matrix, machines)
            )
            selected = [0, 2, jobs - 1]
            assert solver.local_search_perturb(actual, threshold, matrix, machines, selected) == (
                upstream.local_search_perturb(expected, threshold, matrix, machines, selected)
            )


def test_guide_matches_published_strategy(solver):
    matrix = np.random.default_rng(123).integers(1, 100, (20, 5)).astype(float)
    np.random.seed(124)
    expected_matrix, expected_jobs = paper_guide(list(range(20)), matrix, 5, 20)
    np.random.seed(124)
    actual_matrix, actual_jobs = solver.get_matrix_and_jobs(list(range(20)), matrix, 5, 20)
    np.testing.assert_array_equal(actual_matrix, expected_matrix)
    np.testing.assert_array_equal(actual_jobs, expected_jobs)


@pytest.mark.parametrize("seed", [125, 126])
@pytest.mark.parametrize("jobs,machines,iterations", [(20, 5, 3), (7, 3, 51)])
def test_fixed_iteration_trajectory_matches_upstream(solver, upstream, seed,
                                                    jobs, machines, iterations):
    matrix = np.random.default_rng(127).integers(1, 100, (jobs, machines)).astype(float)
    original = matrix.copy()
    expected_inputs = []
    actual_inputs = []

    def expected_guide(sequence, times, machines, jobs):
        expected_inputs.append(list(sequence))
        return paper_guide(sequence, times, machines, jobs)

    real_guide = solver.get_matrix_and_jobs

    def actual_guide(sequence, times, machines, jobs):
        actual_inputs.append(list(sequence))
        return real_guide(sequence, times, machines, jobs)

    np.random.seed(seed)
    expected_cost = upstream.gls(jobs, matrix, machines, 1.0, iterations, expected_guide)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(solver, "get_matrix_and_jobs", actual_guide)
        result = solver.SearchStrategy().solve(matrix, time.time() + 30, iterations, seed)
    assert actual_inputs == expected_inputs
    assert objective(result, matrix) == expected_cost
    np.testing.assert_array_equal(matrix, original)


@pytest.mark.parametrize("matrix", [np.zeros((4, 2)), np.full((4, 2), 0.1),
                                   np.ones((1, 1)), np.ones((2, 2))])
def test_nonnegative_float_and_small_inputs_remain_permutations(solver, matrix):
    order = solver.SearchStrategy().solve(matrix, time.time() + 30, 1, 128)
    assert sorted(order) == list(range(len(matrix)))
    assert np.isfinite(objective(order, matrix))


def test_same_seed_is_repeatable_and_best_incumbent_is_preserved(solver):
    matrix = np.random.default_rng(129).integers(1, 100, (12, 3)).astype(float)
    strategy = solver.SearchStrategy()
    initial = strategy.solve(matrix, time.time() + 30, 0, 130)
    first = strategy.solve(matrix, time.time() + 30, 4, 130)
    second = strategy.solve(matrix, time.time() + 30, 4, 130)
    assert first == second
    assert objective(first, matrix) <= objective(initial, matrix)


def test_expired_deadline_fails_instead_of_substituting_a_solver(solver):
    with pytest.raises(TimeoutError):
        solver.SearchStrategy().solve(np.ones((4, 2)), time.time() - 1, 3, 131)


def test_deadline_is_checked_inside_local_search(solver, monkeypatch):
    class Clock:
        def __init__(self):
            self.calls = 0

        def time(self):
            self.calls += 1
            return 0.0 if self.calls < 3 else 2.0

    clock = Clock()
    matrix = np.random.default_rng(132).integers(1, 100, (12, 3)).astype(float)
    rows = []
    original_row = solver._swap_row

    def observed_row(sequence, threshold, times, machines, index):
        rows.append(index)
        return original_row(sequence, threshold, times, machines, index)

    def unexpected_insert(*args):
        pytest.fail("insert kernel was executed after the deadline")

    monkeypatch.setattr(solver, "time", clock)
    monkeypatch.setattr(solver, "_swap_row", observed_row)
    monkeypatch.setattr(solver, "_insert_row", unexpected_insert)
    order = solver.local_search(list(range(12)), objective(list(range(12)), matrix),
                                matrix, 3, deadline=1.0)
    assert clock.calls == 3
    assert rows == [0, 1]
    assert sorted(order) == list(range(12))


def test_guide_failures_are_not_hidden(solver, monkeypatch):
    def broken(*args):
        raise ValueError("injected guide failure")

    monkeypatch.setattr(solver, "get_matrix_and_jobs", broken)
    with pytest.raises(ValueError, match="injected guide failure"):
        solver.SearchStrategy().solve(np.ones((6, 3)), time.time() + 30, 1, 133)


def test_perturbation_stops_running_rows_after_deadline(solver, monkeypatch):
    moments = iter((0.0, 0.0, 2.0))
    rows = []
    original_row = solver._swap_row

    def observed_row(sequence, threshold, times, machines, index):
        rows.append(index)
        return original_row(sequence, threshold, times, machines, index)

    def unexpected_insert(*args):
        pytest.fail("perturbation continued after the deadline")

    monkeypatch.setattr(solver, "time", SimpleNamespace(time=lambda: next(moments)))
    monkeypatch.setattr(solver, "_swap_row", observed_row)
    monkeypatch.setattr(solver, "_insert_row", unexpected_insert)
    matrix = np.arange(1, 19, dtype=float).reshape(6, 3)
    result = solver.local_search_perturb(list(range(6)), objective(list(range(6)), matrix),
                                          matrix, 3, [0, 3, 5], deadline=1.0)
    assert rows == [0, 3]
    assert sorted(result) == list(range(6))


def test_incomplete_neh_is_rejected_on_deadline(solver, monkeypatch):
    moments = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(solver, "time", SimpleNamespace(time=lambda: next(moments)))
    with pytest.raises(TimeoutError, match="during NEH"):
        solver.neh(np.ones((4, 2)), 2, 4, deadline=1.0)


def test_overrunning_kernel_cannot_return_a_late_success(solver, monkeypatch):
    clock = [0.0]

    def overrunning_cost(order, matrix, machines):
        clock[0] = 2.0
        return objective(order, matrix)

    monkeypatch.setattr(solver, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(solver, "makespan", overrunning_cost)
    with pytest.raises(TimeoutError, match="exhausted"):
        solver.SearchStrategy().solve(np.ones((1, 1)), 1.0, 1, 134)


def test_candidate_is_accepted_by_existing_source_contract(solver):
    from agent_ahd.core.code_checks import check_candidate_source
    from agent_ahd.problems.base import EntryContract

    contract = EntryContract("SearchStrategy", "solve", (
        "self", "processing_times", "deadline", "iteration_limit", "seed"))
    result = check_candidate_source(SOURCE, SOURCE.parent, contract)
    assert result.valid, result.error_message
