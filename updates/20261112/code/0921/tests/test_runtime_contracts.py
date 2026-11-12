from dataclasses import asdict
from itertools import permutations
from pathlib import Path
import importlib.util
import json
import math
import time

import numpy as np
import pytest

from agent_ahd.core.archive import _cluster_labeled_rows
from agent_ahd.core.eoh_timeout import execute_candidate_with_timeout, _construct_tsp
from agent_ahd.core.staged_evaluation import EvaluationCache, EvaluationFingerprint
from agent_ahd.evaluators.manifest_evaluator import objective, sha256_file
from agent_ahd.problems.base import EntryContract, EvaluationBudget
from agent_ahd.problems.fssp.problem import SearchStrategy as NEH
from agent_ahd.problems.tsp_gls.problem import SearchStrategy as Original
from agent_ahd.problems.tsp_gls.optimized.solver import SearchStrategy as Optimized


def test_cache_store_idempotent(tmp_path):
    fp = EvaluationFingerprint("a", "b", "c", "train", ("i",), (1,), 1, 1, "gap", "percent", "d", "dev")
    cache = EvaluationCache(tmp_path)
    cache.store(fp, {"valid": True})
    cache.store(fp, {"valid": True})
    assert cache.load(fp) == {"valid": True}
    with pytest.raises(RuntimeError):
        cache.store(fp, {"valid": False})


@pytest.mark.parametrize("size", [3, 8, 30])
@pytest.mark.parametrize("rounded", [False, True])
def test_gls_fixed_work_equivalence(size, rounded):
    rng = np.random.default_rng(size)
    xy = rng.random((size, 2))
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    if rounded:
        d = np.round(d * 100)
    a, b = Original(), Optimized()
    initial = a.nearest_neighbor_2end(d, 0).astype(int)
    np.testing.assert_array_equal(initial, b.nearest_neighbor_2end(d, 0))
    nn = np.argsort(d, axis=1)[:, 1:101]
    cost = a.tour_cost_2end(d, initial)
    for first in (False, True):
        ra, ca, ia = a.guided_local_search(d, nn, initial.copy(), cost, time.time() + 120, 5, first_improvement=first)
        rb, cb, ib = b.guided_local_search(d, nn, initial.copy(), cost, time.time() + 120, 5, first_improvement=first)
        np.testing.assert_array_equal(ra, rb)
        assert ca == cb and ia == ib == 5


def test_neh_accelerated_insertions_equal_full_recomputation():
    rng = np.random.default_rng(33)
    solver = NEH()
    for n, m in [(3, 2), (8, 5), (20, 10)]:
        p = rng.integers(1, 100, size=(n, m))
        sequence = rng.permutation(n - 1).tolist()
        actual = solver.insertion_costs(sequence, n - 1, p)
        expected = [objective("fssp", p, sequence[:i] + [n - 1] + sequence[i:]) for i in range(n)]
        np.testing.assert_array_equal(actual, expected)


def test_feasibility_cvrp_and_fssp():
    item = {"distance_matrix": np.ones((4, 4)) - np.eye(4), "demands": np.array([0, 2, 2, 2]), "capacity": 4}
    assert objective("cvrp", item, [[1, 2], [3]]) == 5
    for routes in ([[1, 2, 3]], [[1, 1], [2, 3]], [[0, 1], [2, 3]], [[1, 2]], [[True, 2], [3]]):
        with pytest.raises(ValueError):
            objective("cvrp", item, routes)
    assert objective("fssp", np.array([[2, 3], [4, 1]]), [0, 1]) == 7


def reference_cluster(matrix, labels, max_clusters):
    n = len(labels)
    k = max(1, min(max_clusters, round(math.sqrt(n))))
    if k == 1:
        return [sorted(labels)]
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    x = (matrix - center) / np.maximum(np.maximum(mad, matrix.std(axis=0)), 1e-9)
    indices = [int(np.argmax(np.linalg.norm(x, axis=1)))]
    while len(indices) < k:
        ds = [min(float(np.linalg.norm(x[i] - x[j])) for j in indices) if i not in indices else -1 for i in range(n)]
        indices.append(int(np.argmax(ds)))
    centroids = x[indices].copy()
    assignments = np.zeros(n, dtype=int)
    for iteration in range(25):
        updated = np.linalg.norm(x[:, None] - centroids[None], axis=2).argmin(axis=1)
        if np.array_equal(updated, assignments) and iteration > 0:
            break
        assignments = updated
        for c in range(k):
            members = x[assignments == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    groups = [sorted(labels[i] for i in range(n) if assignments[i] == c) for c in range(k)]
    return sorted((g for g in groups if g), key=lambda g: g[0])


@pytest.mark.parametrize("seed", range(6))
def test_streaming_clusters_equal_reference(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, size=(100, 64)).astype(float) if seed % 2 else rng.normal(size=(100, 64))
    if seed == 0:
        x[:] = 1
    labels = [f"c{i:04d}" for i in range(len(x))]
    assert _cluster_labeled_rows(x, labels, max_clusters=10) == reference_cluster(x, labels, 10)


@pytest.mark.parametrize("kind", ["fssp", "cvrp", "tsp_constructive"])
def test_real_isolated_process_for_each_interface(tmp_path, kind):
    root = Path(__file__).resolve().parents[1]
    source = root / "problems" / kind / "problem.py"
    target = tmp_path / "solver.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    if kind == "fssp":
        instance = np.array([[2, 3], [4, 1]])
        contract = EntryContract("SearchStrategy", "solve", ("self", "processing_times", "deadline", "iteration_limit", "seed"))
    elif kind == "cvrp":
        instance = {"distance_matrix": np.ones((4, 4)) - np.eye(4), "demands": np.array([0, 2, 2, 2]), "capacity": 4}
        contract = EntryContract("SearchStrategy", "solve", ("self", "instance", "deadline", "iteration_limit", "seed"), "mapping", "routes")
    else:
        instance = np.ones((4, 4)) - np.eye(4)
        contract = EntryContract("SearchStrategy", "select_next_node", ("self", "current_node", "destination_node", "unvisited_nodes", "distance_matrix"), "constructive_tsp")
    result = execute_candidate_with_timeout(candidate_id=kind, candidate_path=target, instances=[instance],
             budget=EvaluationBudget(n_instances=1, time_limit=2, ite_max=5, seeds=(1, 2)), timeout=30, entry_contract=contract)
    assert result.valid, result.error_message
    assert len(result.routes_by_seed) == 2
    assert objective(kind, instance, result.routes[0]) > 0


def test_parent_deadline_stops_infinite_candidate(tmp_path):
    target = tmp_path / "solver.py"
    target.write_text("class SearchStrategy:\n    def solve(self, distance_matrix, deadline, iteration_limit, seed):\n        while True:\n            pass\n")
    started = time.monotonic()
    result = execute_candidate_with_timeout(candidate_id="hang", candidate_path=target, instances=[np.eye(3)],
                 budget=EvaluationBudget(n_instances=1, time_limit=0.1), timeout=20)
    assert not result.valid and "timeout" in result.error_type
    assert time.monotonic() - started < 15


def test_constructor_time_counts_against_solver_budget(tmp_path):
    target = tmp_path / "solver.py"
    target.write_text("class SearchStrategy:\n    def __init__(self):\n        while True:\n            pass\n    def solve(self, distance_matrix, deadline, iteration_limit, seed):\n        return list(range(len(distance_matrix)))\n")
    started = time.monotonic()
    result = execute_candidate_with_timeout(candidate_id="constructor_hang", candidate_path=target,
                 instances=[np.eye(3)], budget=EvaluationBudget(n_instances=1, time_limit=0.1), timeout=20)
    assert not result.valid and "timeout" in result.error_type
    assert time.monotonic() - started < 15
