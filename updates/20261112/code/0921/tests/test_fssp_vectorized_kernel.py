import numpy as np
from random import Random
from agent_ahd.problems.fssp.problem import SearchStrategy as Scalar
from agent_ahd.problems.fssp.engineering_controls.vectorized_ils import SearchStrategy as Vector
from agent_ahd.problems.fssp.engineering_controls.batched_ig import SearchStrategy as Batch


def scalar_makespan(order, p):
    completion = [0] * p.shape[1]
    for job in order:
        previous = 0
        for machine in range(p.shape[1]):
            previous = max(previous, completion[machine]) + int(p[job, machine])
            completion[machine] = previous
    return completion[-1]


def test_prefix_scans_equal_independent_makespans_at_every_position():
    rng = np.random.RandomState(771)
    solver = Vector()
    for n, m in ((1, 1), (2, 5), (20, 10), (50, 20), (200, 20)):
        p = rng.randint(1, 100, (n, m)).astype(float)
        order = rng.permutation(n).tolist()
        job = order.pop()
        costs = solver.insertion_costs(order, job, p)
        expected = [scalar_makespan(order[:i] + [job] + order[i:], p) for i in range(n)]
        np.testing.assert_array_equal(costs, expected)
        np.testing.assert_array_equal(costs, Scalar().insertion_costs(order, job, p))
        assert solver.makespan(order + [job], p) == scalar_makespan(order + [job], p)


def test_batched_removal_costs_equal_independent_complete_schedules():
    rng = np.random.RandomState(782)
    solver = Batch()
    for n, m in ((1, 1), (2, 5), (20, 10), (50, 20)):
        p = rng.randint(1, 100, (n, m)).astype(float)
        order = rng.permutation(n).tolist()
        positions = rng.permutation(n)[:24].tolist()
        costs = solver._batch_removed_costs(order, positions, p)
        expected = []
        for position in positions:
            remaining, job = order[:], order[position]
            remaining.pop(position)
            expected.append([scalar_makespan(remaining[:i] + [job] + remaining[i:], p) for i in range(n)])
        np.testing.assert_array_equal(costs, expected)


def test_batched_local_search_keeps_move_order_and_rng_state():
    rng = np.random.RandomState(783)
    for n, m in ((20, 5), (50, 10)):
        p = rng.randint(1, 100, (n, m)).astype(float)
        order = rng.permutation(n).tolist()
        initial = scalar_makespan(order, p)
        scalar_rng, batch_rng = Random(32), Random(32)
        expected = Vector()._local_search(order, initial, p, None, scalar_rng)
        actual = Batch()._local_search(order, initial, p, None, batch_rng)
        assert actual == expected
        assert scalar_rng.getstate() == batch_rng.getstate()
