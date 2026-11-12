"""Training-only engineering design: partial-sequence IG with bounded plateaus.

Uses only processing times, seed and budget. No benchmark names or references.
Compilation is lazy and charged to solve. Workspaces require O(n*m) memory.
"""
import time
import numpy as np
from numba import njit


@njit
def makespan(sequence, count, p):
    finish = np.zeros(p.shape[1])
    for i in range(count):
        left = 0.0
        for k in range(p.shape[1]):
            left = max(left, finish[k]) + p[sequence[i], k]
            finish[k] = left
    return finish[-1]


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
        finish, value = 0.0, 0.0
        for k in range(machines):
            finish = max(finish, forward[i, k + 1]) + p[job, k]
            value = max(value, finish + backward[i, k])
        costs[i] = value


@njit
def insert(sequence, count, job, position):
    for i in range(count, position, -1):
        sequence[i] = sequence[i - 1]
    sequence[position] = job


@njit
def best_position(costs, count, excluded):
    value = np.inf
    position, ties = 0, 0
    for i in range(count):
        if costs[i] < value:
            position, value, ties = i, costs[i], 1
        elif costs[i] == value and i != excluded:
            ties += 1
            if position == excluded or np.random.randint(ties) == 0:
                position = i
    return position, value


@njit
def local_search(sequence, count, value, p, forward, backward, costs, neutral):
    # Each original job is visited once per sweep, even after earlier moves.
    jobs = sequence[:count].copy()
    positions = np.empty(len(p), dtype=np.int64)
    for i in range(count):
        positions[sequence[i]] = i
    for sweep in range(count * 2):
        np.random.shuffle(jobs)
        improved = False
        for job in jobs:
            old = positions[job]
            for i in range(old, count - 1):
                sequence[i] = sequence[i + 1]
            insertion_costs(sequence, count - 1, job, p, forward, backward, costs)
            pos, candidate = best_position(costs, count, old)
            accept = candidate < value or (neutral and candidate == value and pos != old)
            if accept:
                improved = improved or candidate < value
                insert(sequence, count - 1, job, pos)
                value = candidate
                for i in range(min(old, pos), max(old, pos) + 1):
                    positions[sequence[i]] = i
            else:
                insert(sequence, count - 1, job, old)
        # A plateau sweep is bounded: neutral moves alone do not cause cycling.
        if not improved:
            break
    return value


@njit
def initialize(p, seed, forward, backward, costs, neutral):
    np.random.seed(seed)
    sequence = np.empty(len(p), dtype=np.int64)
    order = np.argsort(-p.sum(axis=1))
    value = 0.0
    for count in range(len(p)):
        job = order[count]
        insertion_costs(sequence, count, job, p, forward, backward, costs)
        position, value = best_position(costs, count + 1, -1)
        insert(sequence, count, job, position)
    value = local_search(sequence, len(p), value, p, forward, backward, costs, neutral)
    return sequence, value


@njit
def step(current, p, destroy, forward, backward, costs, partial, neutral):
    candidate = current.copy()
    count = len(candidate)
    removed = np.empty(min(destroy, count), dtype=np.int64)
    for r in range(len(removed)):
        position = np.random.randint(count)
        removed[r] = candidate[position]
        for i in range(position, count - 1):
            candidate[i] = candidate[i + 1]
        count -= 1
    if partial and count > 1:
        local_search(candidate, count, makespan(candidate, count, p), p,
                     forward, backward, costs, neutral)
    value = 0.0
    for job in removed:
        insertion_costs(candidate, count, job, p, forward, backward, costs)
        position, value = best_position(costs, count + 1, -1)
        insert(candidate, count, job, position)
        count += 1
    value = local_search(candidate, count, value, p, forward, backward, costs, neutral)
    return candidate, value


class SearchStrategy:
    PARTIAL_SEARCH = True
    NEUTRAL_MOVES = True
    DESTROY_SIZE = 4
    TEMPERATURE_FACTOR = 0.4

    def solve(self, processing_times, deadline, iteration_limit, seed):
        start = time.time()
        p = np.ascontiguousarray(processing_times, dtype=np.float64)
        n, m = p.shape
        if n == 0 or m == 0:
            return list(range(n))
        forward = np.zeros((n + 1, m + 1))
        backward = np.zeros((n + 1, m + 1))
        costs = np.empty(n + 1)
        current, value = initialize(p, seed % (2 ** 32), forward, backward, costs, self.NEUTRAL_MOVES)
        best, best_value = current.copy(), value
        rng = np.random.default_rng(seed)
        temperature = max(1e-9, self.TEMPERATURE_FACTOR * p.mean() / 10)
        prefix = np.cumsum(p, axis=1) - p
        suffix = np.cumsum(p[:, ::-1], axis=1)[:, ::-1] - p
        lower_bound = max(p.sum(axis=1).max(), (p.sum(axis=0) + prefix.min(axis=0) + suffix.min(axis=0)).max())
        initialized = time.time() - start
        iterations, improvements = 0, 0
        for _ in range(iteration_limit):
            if time.time() >= deadline - 0.2 or best_value <= lower_bound:
                break
            candidate, candidate_value = step(current, p, self.DESTROY_SIZE, forward,
                backward, costs, self.PARTIAL_SEARCH, self.NEUTRAL_MOVES)
            iterations += 1
            if candidate_value < best_value:
                best, best_value = candidate.copy(), candidate_value
                improvements += 1
            delta = candidate_value - value
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                current, value = candidate, candidate_value
        self.statistics = {"iterations": iterations, "improvements": improvements,
                           "initialization_seconds": initialized, "best_objective": best_value}
        return best.tolist()
