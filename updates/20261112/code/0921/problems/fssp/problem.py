import numpy as np


class SearchStrategy:
    """NEH with Taillard's insertion acceleration: O(n^2*m), O(n*m)."""
    def insertion_costs(self, sequence, job, processing_times):
        p = processing_times
        k, machines = len(sequence), p.shape[1]
        forward = np.zeros((k + 1, machines + 1))
        backward = np.zeros((k + 1, machines + 1))
        for i in range(k):
            for m in range(machines):
                forward[i + 1, m + 1] = max(forward[i, m + 1], forward[i + 1, m]) + p[sequence[i], m]
        for i in range(k - 1, -1, -1):
            for m in range(machines - 1, -1, -1):
                backward[i, m] = max(backward[i + 1, m], backward[i, m + 1]) + p[sequence[i], m]
        costs = np.zeros(k + 1)
        for position in range(k + 1):
            finish = 0.0
            for m in range(machines):
                finish = max(finish, forward[position, m + 1]) + p[job, m]
                costs[position] = max(costs[position], finish + backward[position, m])
        return costs

    def solve(self, processing_times, deadline, iteration_limit, seed):
        order = np.argsort(-processing_times.sum(axis=1), kind="stable")
        sequence = []
        for job in order:
            costs = self.insertion_costs(sequence, int(job), processing_times)
            sequence.insert(int(np.argmin(costs)), int(job))
        return sequence
