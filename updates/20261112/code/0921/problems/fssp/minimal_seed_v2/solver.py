"""Human engineering IG control, using O(n*m) reusable insertion workspaces.

No reference values, instance IDs or files are read. Lazy JIT compilation occurs
inside solve and is included in the first instance's deadline. Not Agent output.
"""
import time
import numpy as np
from numba import njit


@njit
def insertion_costs(sequence, count, job, p, forward, backward, costs):
    machines = p.shape[1]
    forward[0, :] = 0
    forward[:, 0] = 0
    backward[count, :] = 0
    backward[:, machines] = 0
    for i in range(count):
        for k in range(machines):
            forward[i + 1, k + 1] = max(forward[i, k + 1], forward[i + 1, k]) + p[sequence[i], k]
    for i in range(count - 1, -1, -1):
        for k in range(machines - 1, -1, -1):
            backward[i, k] = max(backward[i + 1, k], backward[i, k + 1]) + p[sequence[i], k]
    for i in range(count + 1):
        finish = 0.0
        cost = 0.0
        for k in range(machines):
            finish = max(finish, forward[i, k + 1]) + p[job, k]
            cost = max(cost, finish + backward[i, k])
        costs[i] = cost


@njit
def insert(sequence, count, job, position):
    for i in range(count, position, -1):
        sequence[i] = sequence[i - 1]
    sequence[position] = job


@njit
def local_search(sequence, value, p, forward, backward, costs, passes):
    count = len(sequence)
    positions = np.arange(count)
    # Full first-improvement insertion neighborhoods; workspaces reused.
    for sweep in range(passes):
        np.random.shuffle(positions)
        improved = False
        for position in positions:
            job = sequence[position]
            for i in range(position, count - 1):
                sequence[i] = sequence[i + 1]
            insertion_costs(sequence, count - 1, job, p, forward, backward, costs)
            best = np.argmin(costs[:count])
            if costs[best] < value:
                insert(sequence, count - 1, job, best)
                value = costs[best]
                improved = True
                break
            insert(sequence, count - 1, job, position)
        if not improved:
            break
    return value


@njit
def initialize(p, seed, forward, backward, costs):
    np.random.seed(seed)
    n = len(p)
    sequence = np.empty(n, dtype=np.int64)
    order = np.argsort(-p.sum(axis=1))
    value = 0.0
    for count in range(n):
        job = order[count]
        insertion_costs(sequence, count, job, p, forward, backward, costs)
        position = np.argmin(costs[:count + 1])
        value = costs[position]
        insert(sequence, count, job, position)
    return sequence, local_search(sequence, value, p, forward, backward, costs, n * 2)


@njit
def step(current, p, destroy, forward, backward, costs):
    candidate = current.copy()
    count = len(candidate)
    removed = np.empty(min(destroy, count), dtype=np.int64)
    for r in range(len(removed)):
        position = np.random.randint(count)
        removed[r] = candidate[position]
        for i in range(position, count - 1):
            candidate[i] = candidate[i + 1]
        count -= 1
    value = 0.0
    for job in removed:
        insertion_costs(candidate, count, job, p, forward, backward, costs)
        minimum = np.min(costs[:count + 1])
        position, ties = 0, 0
        for i in range(count + 1):
            if costs[i] == minimum:
                ties += 1
                if np.random.randint(ties) == 0:
                    position = i
        insert(candidate, count, job, position)
        count += 1
        value = minimum
    value = local_search(candidate, value, p, forward, backward, costs, len(p) * 2)
    return candidate, value


class SearchStrategy:
    DESTROY_SIZE = 4
    TEMPERATURE_FACTOR = 0.2

    def solve(self, processing_times, deadline, iteration_limit, seed):
        p = np.ascontiguousarray(processing_times, dtype=np.float64)
        n, m = p.shape
        if n == 0 or m == 0:
            return list(range(n))
        forward, backward = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1))
        costs = np.empty(n + 1)
        current, value = initialize(p, seed % (2 ** 32), forward, backward, costs)
        best, best_value = current.copy(), value
        rng = np.random.default_rng(seed)
        temperature = max(1e-9, self.TEMPERATURE_FACTOR * p.mean() / 10)
        prefix = np.cumsum(p, axis=1) - p
        suffix = np.cumsum(p[:, ::-1], axis=1)[:, ::-1] - p
        lower_bound = max(p.sum(axis=1).max(), (p.sum(axis=0) + prefix.min(axis=0) + suffix.min(axis=0)).max())
        for iteration in range(iteration_limit):
            if time.time() >= deadline - 0.15 or best_value <= lower_bound:
                break
            candidate, candidate_value = step(current, p, self.DESTROY_SIZE, forward, backward, costs)
            if candidate_value < best_value:
                best, best_value = candidate.copy(), candidate_value
            delta = candidate_value - value
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                current, value = candidate, candidate_value
        return best.tolist()
