import numpy as np
from numba import njit
from scripts.prepare_minimal_fssp_tiebreak import TIE_KERNEL
from agent_ahd.problems.fssp.engineering_controls.partial_neutral_ig import insertion_costs
from agent_ahd.tests.test_compiled_fssp import makespan


def completion_vector(order, p):
    finish = np.zeros(p.shape[1])
    for job in order:
        left = 0.
        for k in range(p.shape[1]):
            left = finish[k] = max(left, finish[k]) + p[job, k]
    return finish


def test_symmetric_ties_preserve_makespan_and_match_independent_completions():
    namespace = {"np": np, "njit": njit}
    exec(TIE_KERNEL, namespace)
    guided_position = namespace["guided_position"]
    rng = np.random.default_rng(91521)
    for n, m in ((2, 1), (8, 4), (20, 10)):
        p = rng.integers(1, 8, (n, m)).astype(float)
        sequence = rng.permutation(n)
        for count in range(1, n):
            job = sequence[count]
            forward = np.zeros((n + 1, m + 1))
            backward = np.zeros_like(forward)
            costs = np.empty(n + 1)
            insertion_costs(sequence, count, job, p, forward, backward, costs)
            position, value = guided_position(costs, count + 1, -1, job, p, forward, backward)
            order = sequence[:count].tolist()
            values = [makespan(order[:i] + [job] + order[i:], p) for i in range(count + 1)]
            secondary = [completion_vector(order[:i] + [job], p).sum()
                         + completion_vector(list(reversed(order[i:])) + [job], p[:, ::-1]).sum()
                         for i in range(count + 1)]
            expected = min(zip(values, secondary))
            assert (value, secondary[position]) == expected
            assert makespan(order[:position] + [job] + order[position:], p) == value
            # Reversing both axes preserves the entire schedule objective.
            assert makespan(list(reversed(order[:position] + [job] + order[position:])), p[:, ::-1]) == value
