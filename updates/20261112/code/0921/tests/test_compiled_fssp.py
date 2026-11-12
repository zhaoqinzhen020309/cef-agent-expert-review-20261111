import time
import numpy as np

from agent_ahd.problems.fssp.engineering_controls.compiled_ig import insertion_costs, SearchStrategy
from agent_ahd.core.code_checks import _check_tree_safety
from pathlib import Path
import ast


def makespan(order, p):
    times = [0] * p.shape[1]
    for job in order:
        last = 0
        for k in range(p.shape[1]):
            last = times[k] = max(last, times[k]) + int(p[job, k])
    return times[-1]


def test_compiled_insertion_matches_independent_schedules():
    rng = np.random.default_rng(71234)
    for n, m in ((1, 1), (2, 5), (20, 10), (100, 20)):
        p = rng.integers(1, 100, (n, m)).astype(float)
        order = rng.permutation(n)
        job = order[-1]
        forward, backward, costs = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1)), np.zeros(n)
        insertion_costs(order, n - 1, job, p, forward, backward, costs)
        remaining = order[:-1].tolist()
        expected = [makespan(remaining[:i] + [job] + remaining[i:], p) for i in range(n)]
        np.testing.assert_array_equal(costs, expected)


def test_compiled_ig_returns_permutation_and_feasible_schedule():
    p = np.random.default_rng(71235).integers(1, 100, (20, 5)).astype(float)
    result = SearchStrategy().solve(p, time.time() + 30, 20, 71236)
    assert sorted(result) == list(range(20))
    assert makespan(result, p) <= makespan(list(range(20)), p)


def test_only_cpu_njit_import_is_allowed():
    assert _check_tree_safety(ast.parse("from numba import njit"), Path("solver.py"), set()).valid
    for source in ("import numba", "from numba import cuda", "from numba import njit, objmode", "from numba import *"):
        assert not _check_tree_safety(ast.parse(source), Path("solver.py"), set()).valid
