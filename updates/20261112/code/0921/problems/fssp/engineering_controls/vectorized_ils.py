# Human kernel optimization of a previous Agent solver; not a new Agent search result.
import time
import numpy as np
from random import Random


class SearchStrategy:
    TIME_MARGIN = 0.05

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
        backward = np.zeros_like(forward)
        backward[:count] = reverse[1:][::-1, ::-1]
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



    def _local_search(self, sequence, initial_ms, processing_times, deadline, rng):
        seq = list(sequence)
        best_ms = initial_ms
        improved = True

        while improved and self._time_ok(deadline):
            improved = False
            indices = list(range(len(seq)))
            rng.shuffle(indices)

            for idx in indices:
                if not self._time_ok(deadline):
                    return seq, best_ms

                job = seq[idx]
                sub = seq[:idx] + seq[idx + 1:]
                costs = self.insertion_costs(sub, job, processing_times)
                best_pos = int(np.argmin(costs))
                new_ms = float(costs[best_pos])

                if new_ms + 1e-12 < best_ms:
                    sub.insert(best_pos, job)
                    seq = sub
                    best_ms = new_ms
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
        p = np.asarray(processing_times, dtype=float)
        n_jobs = p.shape[0]
        if n_jobs == 0:
            return []
        if p.shape[1] == 0:
            return list(range(n_jobs))

        if deadline is None and iteration_limit is None:
            iteration_limit = 10

        rng = Random(seed)

        order = np.argsort(-p.sum(axis=1), kind='stable')
        sequence = []
        for idx, job in enumerate(order):
            job = int(job)
            if not self._time_ok(deadline):
                sequence.extend([int(x) for x in order[idx:]])
                break
            costs = self.insertion_costs(sequence, job, p)
            sequence.insert(int(np.argmin(costs)), job)

        best_seq = sequence[:]
        best_ms = self.makespan(best_seq, p)

        if not self._time_ok(deadline):
            return best_seq

        current_seq = best_seq[:]
        current_ms = best_ms
        restart = 0

        while self._time_ok(deadline) and (iteration_limit is None or restart < iteration_limit):
            current_seq, current_ms = self._local_search(current_seq, current_ms, p, deadline, rng)

            if current_ms + 1e-12 < best_ms:
                best_seq = current_seq[:]
                best_ms = current_ms

            if not self._time_ok(deadline):
                break

            strength = min(len(current_seq), max(1, len(current_seq) // 10))
            current_seq = self._perturb(best_seq, rng, strength)
            current_ms = self.makespan(current_seq, p)
            restart += 1

        return best_seq
