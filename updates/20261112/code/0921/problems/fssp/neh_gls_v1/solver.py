"""Complete NEH+GLS seed; every function below is editable candidate source.

Search operators and their traversal order are adapted from Fei Liu's EoH:
https://github.com/FeiLiu36/EoH/blob/main/examples/fssp_gls_numba/prob.py
Frozen upstream SHA256: 2303b8de5881902870c743a941d43fb41f58733cf8dd18a906bf1deb2592b286
The guide is Figure 10 of https://arxiv.org/pdf/2401.02051v3, not an empty hook.

Interface adaptations: return the stored best permutation; honor the caller's
absolute deadline and seed; allow nonnegative float processing times. The NEH
selection sentinel is -1 rather than 1 so zero/small totals are not selected
twice. Inputs with fewer than seven jobs select at least two perturbation jobs;
a one-job instance ends after construction. These boundaries are outside the
paper's 20--200-job test sizes.

CPU njit is required, with no missing-dependency or compilation fallback. Its
cold compilation happens inside solve and counts against the solver deadline.
The two original neighborhood sweeps are scheduled one outer row at a time so
Python can check the deadline between rows. Move order, acceptance threshold,
guide calls, and the 50-iteration incumbent restart are retained. In particular,
the upstream search's incoming threshold and incumbent-update cadence are kept;
this is an interface adaptation, not a claim to repair its search policy.

MIT License -- upstream portions Copyright (c) 2026 Fei Liu.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import random
import time

import numpy as np
from numba import njit


@njit
def makespan(order, tasks, machines_val):
    times = [0.0] * machines_val
    for job in order:
        times[0] += tasks[job][0]
        for machine in range(1, machines_val):
            if times[machine] < times[machine - 1]:
                times[machine] = times[machine - 1]
            times[machine] += tasks[job][machine]
    return max(times)


@njit
def _swap_row(sequence, threshold, tasks, machines_val, index):
    new_seq = sequence[:]
    for position in range(index + 1, len(new_seq)):
        candidate = new_seq[:]
        candidate[index], candidate[position] = candidate[position], candidate[index]
        value = makespan(candidate, tasks, machines_val)
        if value < threshold:
            new_seq = candidate[:]
            threshold = value
    return new_seq, threshold


@njit
def _insert_row(sequence, threshold, tasks, machines_val, job):
    new_seq = sequence[:]
    for position in range(1, len(new_seq)):
        candidate = new_seq[:]
        candidate.remove(job)
        candidate.insert(position, job)
        value = makespan(candidate, tasks, machines_val)
        if value < threshold:
            new_seq = candidate[:]
            threshold = value
    return new_seq, threshold


def local_search(sequence, cmax_old, tasks, machines_val, deadline=float("inf")):
    new_seq = sequence[:]
    for index in range(len(new_seq)):
        if time.time() >= deadline:
            return new_seq
        new_seq, cmax_old = _swap_row(new_seq, cmax_old, tasks, machines_val, index)
    for job in range(1, len(new_seq)):
        if time.time() >= deadline:
            return new_seq
        new_seq, cmax_old = _insert_row(new_seq, cmax_old, tasks, machines_val, job)
    return new_seq


def local_search_perturb(sequence, cmax_old, tasks, machines_val, jobs,
                         deadline=float("inf")):
    new_seq = sequence[:]
    for index in jobs:
        if time.time() >= deadline:
            return new_seq
        new_seq, cmax_old = _swap_row(new_seq, cmax_old, tasks, machines_val, int(index))
    for job in jobs:
        if time.time() >= deadline:
            return new_seq
        new_seq, cmax_old = _insert_row(new_seq, cmax_old, tasks, machines_val, int(job))
    return new_seq


def sum_and_order(tasks_val, machines_val, tasks):
    totals = [0.0] * tasks_val
    order = [0] * tasks_val
    for job in range(tasks_val):
        for machine in range(machines_val):
            totals[job] += tasks[job][machine]
    for position in range(tasks_val):
        maximum = -1.0
        selected = 0
        for job in range(tasks_val):
            if maximum < totals[job]:
                maximum = totals[job]
                selected = job
        totals[selected] = -1.0
        order[position] = selected
    return order


def neh(tasks, machines_val, tasks_val, deadline=float("inf")):
    if time.time() >= deadline:
        raise TimeoutError("deadline reached before NEH construction")
    order = sum_and_order(tasks_val, machines_val, tasks)
    current_seq = [order[0]]
    for index in range(1, tasks_val):
        best_seq = None
        minimum = float("inf")
        for position in range(index + 1):
            if time.time() >= deadline:
                raise TimeoutError("deadline reached during NEH construction")
            candidate = current_seq[:]
            candidate.insert(position, order[index])
            value = makespan(candidate, tasks, machines_val)
            if value < minimum:
                best_seq = candidate
                minimum = value
        current_seq = best_seq
    return current_seq, makespan(current_seq, tasks, machines_val)


def get_matrix_and_jobs(current_sequence, time_matrix, m, n):
    machine_subset = np.random.choice(m, max(1, int(0.3 * m)), replace=False)
    weighted_avg = np.average(time_matrix[:, machine_subset], axis=1,
                              weights=np.random.rand(len(machine_subset)))
    count = min(n, max(2, int(0.3 * n)))
    perturb_jobs = np.argsort(weighted_avg)[-count:]
    new_matrix = time_matrix.copy()
    factors = np.random.uniform(0.8, 1.2, size=(len(perturb_jobs), len(machine_subset)))
    new_matrix[perturb_jobs[:, np.newaxis], machine_subset] *= factors
    return new_matrix, perturb_jobs


class SearchStrategy:
    def solve(self, processing_times, deadline, iteration_limit, seed):
        deadline = float(deadline)
        start = time.time()
        if not np.isfinite(deadline) or start >= deadline:
            raise TimeoutError("solver has no remaining time budget")
        if int(iteration_limit) != iteration_limit or iteration_limit < 0:
            raise ValueError("iteration_limit must be a nonnegative integer")
        tasks = np.ascontiguousarray(processing_times, dtype=np.float64)
        if (tasks.ndim != 2 or min(tasks.shape) < 1 or not np.all(np.isfinite(tasks))
                or np.any(tasks < 0)):
            raise ValueError("processing_times must be a finite nonnegative jobs x machines matrix")
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        # Reserve up to 50 ms (at most 5% of the supplied budget) for returning
        # and external normalization; this does not add to the supplied budget.
        stop_at = deadline - min(0.05, (deadline - start) * 0.05)
        jobs, machines = tasks.shape
        sequence, value = neh(tasks, machines, jobs, deadline=stop_at)
        best_sequence, best_value = sequence[:], value
        iteration = 0
        while jobs > 1 and iteration < int(iteration_limit) and time.time() < stop_at:
            sequence = local_search(sequence, value, tasks, machines, deadline=stop_at)
            value = makespan(sequence, tasks, machines)
            if value < best_value:
                best_sequence, best_value = sequence[:], value
            if time.time() >= stop_at:
                break
            modified, selected = get_matrix_and_jobs(sequence, tasks.copy(), machines, jobs)
            modified = np.ascontiguousarray(modified, dtype=np.float64)
            selected = list(selected)
            if (modified.shape != tasks.shape or not np.all(np.isfinite(modified))
                    or np.any(modified < 0) or len(selected) < 2
                    or any(int(job) != job or not 0 <= int(job) < jobs for job in selected)
                    or len(set(selected)) != len(selected)):
                raise ValueError("guide returned an invalid matrix or perturbation job list")
            selected = [int(job) for job in selected[:5]]
            value = makespan(sequence, modified, machines)
            sequence = local_search_perturb(sequence, value, modified, machines, selected,
                                            deadline=stop_at)
            iteration += 1
            if iteration % 50 == 0:
                sequence, value = best_sequence[:], best_value
        if time.time() >= deadline:
            raise TimeoutError("solver exhausted its deadline, including cold compilation")
        return best_sequence
