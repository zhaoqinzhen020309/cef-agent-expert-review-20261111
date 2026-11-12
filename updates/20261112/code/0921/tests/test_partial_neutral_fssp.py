import numpy as np
from agent_ahd.problems.fssp.engineering_controls.partial_neutral_ig import (
    insertion_costs, initialize, local_search, step,
)
from agent_ahd.tests.test_compiled_fssp import makespan


def test_partial_insertion_costs_and_neutral_moves_preserve_objective():
    rng = np.random.default_rng(91428)
    for n, m in ((2, 1), (8, 5), (20, 10)):
        p = rng.integers(1, 20, (n, m)).astype(float)
        forward, backward = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1))
        costs = np.empty(n + 1)
        sequence = rng.permutation(n)
        for count in (0, n // 2, n - 1):
            insertion_costs(sequence, count, sequence[count], p, forward, backward, costs)
            remaining = sequence[:count].tolist()
            expected = [makespan(remaining[:i] + [int(sequence[count])] + remaining[i:], p)
                        for i in range(count + 1)]
            np.testing.assert_array_equal(costs[:count + 1], expected)
        for neutral in (False, True):
            candidate = sequence.copy()
            before = makespan(candidate, p)
            value = local_search(candidate, n, before, p, forward, backward, costs, neutral)
            assert sorted(candidate) == list(range(n))
            assert value == makespan(candidate, p) <= before


def test_partial_reconstruction_keeps_complete_permutation_and_exact_cost():
    p = np.random.default_rng(91429).integers(1, 100, (30, 8)).astype(float)
    n, m = p.shape
    forward, backward, costs = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1)), np.empty(n + 1)
    for neutral in (False, True):
        sequence, value = initialize(p, 731, forward, backward, costs, neutral)
        for _ in range(20):
            sequence, value = step(sequence, p, 4, forward, backward, costs, True, neutral)
            assert sorted(sequence) == list(range(n))
            assert value == makespan(sequence, p)
