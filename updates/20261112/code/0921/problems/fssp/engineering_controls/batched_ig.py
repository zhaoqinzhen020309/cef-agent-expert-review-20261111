# Human kernel optimization of a previous Agent solver; not a new Agent search result.
import time
import numpy as np
from random import Random


class SearchStrategy:
    TIME_MARGIN = 0.15
    CANDIDATE_BATCH = 24
    DESTROY_SIZE = 8
    TEMPERATURE_FACTOR = 0.4

    def _time_ok(self, deadline):
        if deadline is None:
            return True
        return time.time() < deadline - self.TIME_MARGIN


    @staticmethod
    def _completion_table(processing):
        count, machines = processing.shape
        table = np.zeros((count + 1, machines))
        previous = np.zeros(count)
        for machine in range(machines):
            cumulative = np.cumsum(processing[:, machine])
            offsets = np.empty(count)
            if count:
                offsets[0] = 0
                offsets[1:] = cumulative[:-1]
            previous = cumulative + np.maximum.accumulate(previous - offsets)
            table[1:, machine] = previous
        return table

    def insertion_costs(self, sequence, job, processing_times):
        processing = processing_times[np.asarray(sequence, dtype=np.int64)]
        count, machines = processing.shape
        forward = self._completion_table(processing)
        reverse = self._completion_table(processing[::-1, ::-1])
        backward = reverse[::-1, ::-1]
        finish = np.zeros(count + 1)
        costs = np.zeros(count + 1)
        for machine in range(machines):
            finish = np.maximum(finish, forward[:, machine]) + processing_times[job, machine]
            costs = np.maximum(costs, finish + backward[:, machine])
        return costs

    def makespan(self, sequence, processing_times):
        if len(sequence) == 0 or processing_times.shape[1] == 0:
            return 0.0
        return float(self._completion_table(processing_times[np.asarray(sequence, dtype=np.int64)])[-1, -1])




    @staticmethod
    def _batch_completion(processing):
        batch, count, machines = processing.shape
        table = np.zeros((batch, count + 1, machines))
        previous = np.zeros((batch, count))
        for machine in range(machines):
            cumulative = np.cumsum(processing[:, :, machine], axis=1)
            offsets = np.zeros_like(cumulative)
            offsets[:, 1:] = cumulative[:, :-1]
            previous = cumulative + np.maximum.accumulate(previous - offsets, axis=1)
            table[:, 1:, machine] = previous
        return table

    def _batch_removed_costs(self, sequence, positions, p):
        seq = np.asarray(sequence, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        count, batch = len(seq), len(positions)
        keep = np.broadcast_to(np.arange(count), (batch, count))
        keep = keep[keep != positions[:, None]].reshape(batch, count - 1)
        remaining = seq[keep]
        processing = p[remaining]
        forward = self._batch_completion(processing)
        backward = self._batch_completion(processing[:, ::-1, ::-1])[:, ::-1, ::-1]
        finish = np.zeros((batch, count))
        costs = np.zeros_like(finish)
        jobs = seq[positions]
        for machine in range(p.shape[1]):
            finish = np.maximum(finish, forward[:, :, machine]) + p[jobs, machine, None]
            costs = np.maximum(costs, finish + backward[:, :, machine])
        return costs

    def _local_search(self, sequence, initial_ms, processing_times, deadline, rng):
        seq, best_ms = list(sequence), initial_ms
        improved = True
        while improved and self._time_ok(deadline):
            improved = False
            indices = list(range(len(seq)))
            rng.shuffle(indices)
            for start in range(0, len(seq), self.CANDIDATE_BATCH):
                if not self._time_ok(deadline):
                    return seq, best_ms
                positions = indices[start:start + self.CANDIDATE_BATCH]
                costs = self._batch_removed_costs(seq, positions, processing_times)
                minima = np.min(costs, axis=1)
                eligible = np.flatnonzero(minima + 1e-12 < best_ms)
                if len(eligible):
                    # Same first improving job and first best insertion as scalar LS.
                    row = int(eligible[0])
                    job = seq.pop(positions[row])
                    seq.insert(int(np.argmin(costs[row])), job)
                    best_ms = float(minima[row])
                    improved = True
                    break
        return seq, best_ms


    def _perturb(self, sequence, rng, strength):
        seq = list(sequence)
        for _ in range(strength):
            if len(seq) < 2:
                break
            i = rng.randrange(len(seq))
            job = seq.pop(i)
            j = rng.randrange(len(seq) + 1)
            seq.insert(j, job)
        return seq


    def solve(self, processing_times, deadline, iteration_limit, seed):
        import math
        p = np.asarray(processing_times, dtype=float)
        n, machines = p.shape
        if n == 0 or machines == 0:
            return list(range(n))
        rng = Random(seed)
        order = np.argsort(-p.sum(axis=1), kind="stable").tolist()
        sequence = []
        for index, job in enumerate(order):
            if not self._time_ok(deadline):
                return sequence + order[index:]
            costs = self.insertion_costs(sequence, job, p)
            sequence.insert(int(np.argmin(costs)), job)
        current, value = self._local_search(sequence, self.makespan(sequence, p), p, deadline, rng)
        best, best_value = current[:], value
        prefix = np.cumsum(p, axis=1) - p
        suffix = np.cumsum(p[:, ::-1], axis=1)[:, ::-1] - p
        lower_bound = max(float(p.sum(axis=1).max()), float((p.sum(axis=0) + prefix.min(axis=0) + suffix.min(axis=0)).max()))
        temperature = max(1e-9, self.TEMPERATURE_FACTOR * float(p.mean()) / 10.0)
        for _ in range(iteration_limit):
            if not self._time_ok(deadline) or best_value <= lower_bound:
                break
            removed = rng.sample(current, min(self.DESTROY_SIZE, n))
            chosen = set(removed)
            candidate = [job for job in current if job not in chosen]
            for job in removed:
                if not self._time_ok(deadline):
                    return best
                costs = self.insertion_costs(candidate, job, p)
                positions = np.flatnonzero(costs == costs.min()).tolist()
                candidate.insert(rng.choice(positions), job)
            candidate, candidate_value = self._local_search(candidate, self.makespan(candidate, p), p, deadline, rng)
            if candidate_value < best_value:
                best, best_value = candidate[:], candidate_value
            delta = candidate_value - value
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current, value = candidate, candidate_value
        return best

