# Engineering control derived from historical scope_parent; not Agent-generated.
import random
import time

import numpy as np


class SearchStrategy:
    """Multi-start iterated search with 2-opt and vertex relocation."""

    @staticmethod
    def _cost(route, distances):
        indices = np.asarray(route, dtype=np.intp)
        return float(np.sum(distances[indices, np.roll(indices, -1)]))

    @staticmethod
    def _greedy_route(distances, start, rng, randomized=False):
        n = len(distances)
        unused = np.ones(n, dtype=bool)
        unused[start] = False
        route = [start]
        for _ in range(n - 1):
            current = route[-1]
            available = np.flatnonzero(unused)
            ordered = available[np.argsort(distances[current, available])]
            if randomized and len(ordered) > 1:
                width = min(3, len(ordered))
                weights = (6, 3, 1)[:width]
                nxt = int(rng.choices(ordered[:width].tolist(), weights=weights)[0])
            else:
                nxt = int(ordered[0])
            route.append(nxt)
            unused[nxt] = False
        return route

    @staticmethod
    def _two_opt(route, distances, neighbours, deadline):
        n = len(route)
        position = np.empty(n, dtype=np.intp)
        position[route] = np.arange(n)
        active = np.ones(n, dtype=bool)
        while time.time() < deadline:
            improved = False
            tour = np.asarray(route, dtype=np.intp)
            for a in route:
                if not active[a]:
                    continue
                i = int(position[a])
                b = route[(i + 1) % n]
                candidates = neighbours[a]
                js = position[candidates]
                ds = tour[(js + 1) % n]
                delta = (distances[a, candidates] + distances[b, ds]
                         - distances[a, b] - distances[candidates, ds])
                hits = np.flatnonzero((ds != a) & (candidates != b) & (ds != b) & (delta < -1e-10))
                if hits.size:
                    h = int(hits[0])
                    c, d, j = int(candidates[h]), int(ds[h]), int(js[h])
                    if i > j:
                        i, j = j, i
                    route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                    position[route[i + 1:j + 1]] = np.arange(i + 1, j + 1)
                    active[[a, b, c, d]] = True
                    improved = True
                    break
                active[a] = False
                if (i & 31) == 0 and time.time() >= deadline:
                    return route
            if not improved:
                return route
        return route

    @staticmethod
    def _relocate(route, distances, neighbours, deadline):
        n = len(route)
        while time.time() < deadline:
            position = np.empty(n, dtype=np.intp)
            position[route] = np.arange(n)
            tour = np.asarray(route, dtype=np.intp)
            moved = False
            for i, node in enumerate(route):
                before, after = route[i - 1], route[(i + 1) % n]
                removal = distances[before, after] - distances[before, node] - distances[node, after]
                anchors = neighbours[node]
                js = position[anchors]
                successors = tour[(js + 1) % n]
                delta = (removal + distances[anchors, node] + distances[node, successors]
                         - distances[anchors, successors])
                hits = np.flatnonzero((anchors != node) & (successors != node) & (delta < -1e-10))
                if hits.size:
                    j = int(js[int(hits[0])])
                    route.pop(i)
                    route.insert(j - int(i < j) + 1, node)
                    moved = True
                    break
                if (i & 31) == 0 and time.time() >= deadline:
                    return route
            if not moved:
                return route
        return route

    @staticmethod
    def _or_opt_two(route, distances, neighbours, deadline):
        n = len(route)
        while time.time() < deadline:
            position = np.empty(n, dtype=np.intp)
            position[route] = np.arange(n)
            tour = np.asarray(route, dtype=np.intp)
            moved = False
            for i in range(n):
                a, b = route[i - 1], route[i]
                c, d = route[(i + 1) % n], route[(i + 2) % n]
                removal = distances[a, d] - distances[a, b] - distances[c, d]
                anchors = neighbours[b]
                successors = tour[(position[anchors] + 1) % n]
                delta = (removal + distances[anchors, b] + distances[c, successors]
                         - distances[anchors, successors])
                hits = np.flatnonzero((anchors != b) & (anchors != c) & (successors != b)
                                     & (successors != c) & (delta < -1e-10))
                if hits.size:
                    anchor = int(anchors[int(hits[0])])
                    remaining = [node for node in route if node not in (b, c)]
                    j = remaining.index(anchor)
                    route[:] = remaining[:j + 1] + [b, c] + remaining[j + 1:]
                    moved = True
                    break
                if (i & 31) == 0 and time.time() >= deadline:
                    return route
            if not moved:
                return route
        return route

    @classmethod
    def _local_search(cls, route, distances, neighbours, deadline):
        """Alternate neighborhoods until neither can unlock the other."""
        previous = float("inf")
        while time.time() < deadline:
            route = cls._two_opt(route, distances, neighbours, deadline)
            route = cls._relocate(route, distances, neighbours, deadline)
            route = cls._or_opt_two(route, distances, neighbours, deadline)
            cost = cls._cost(route, distances)
            if cost >= previous - 1e-9:
                return route
            previous = cost
        return route

    @staticmethod
    def _kick(route, neighbours, rng):
        """Break four edges, usually reconnecting through a promising edge."""
        n = len(route)
        if n < 8:
            result = route[:]
            i, j = sorted(rng.sample(range(n), 2))
            result[i:j + 1] = reversed(result[i:j + 1])
            return result

        position = [0] * n
        for i, node in enumerate(route):
            position[node] = i
        a = rng.randrange(n)
        near = [int(x) for x in neighbours[route[a]]
                if 2 < (position[int(x)] - a) % n < n - 2]
        b = position[rng.choice(near)] if near else rng.randrange(n)
        cuts = {a, b}
        while len(cuts) < 4:
            cuts.add(rng.randrange(1, n))
        p, q, r, s = sorted(cuts)
        return (route[:p] + route[r:s] + route[q:r]
                + route[p:q] + route[s:])

    def solve(self, distance_matrix, deadline, iteration_limit, seed):
        distances = np.asarray(distance_matrix, dtype=float)
        n = len(distances)
        if n <= 1:
            return list(range(n))

        rng = random.Random(seed)
        stop_time = max(time.time(), deadline - 0.25)
        width = min(32, n - 1)
        neighbours = np.argsort(distances, axis=1)[:, 1:width + 1]

        best_route = self._greedy_route(distances, 0, rng)
        best_route = self._local_search(best_route, distances, neighbours, stop_time)
        best_cost = self._cost(best_route, distances)

        starts = [int(np.argmax(distances[0]))]
        starts.extend(rng.sample(range(n), min(7, n)))
        for number, start in enumerate(dict.fromkeys(starts)):
            if time.time() >= stop_time:
                break
            route = self._greedy_route(distances, start, rng, randomized=number >= 2)
            route = self._local_search(route, distances, neighbours, stop_time)
            cost = self._cost(route, distances)
            if cost < best_cost:
                best_route, best_cost = route[:], cost

        current = best_route[:]
        current_cost = best_cost
        stagnation = 0
        for _ in range(max(0, iteration_limit)):
            if time.time() >= stop_time:
                break
            trial = self._kick(current if stagnation < 12 else best_route,
                               neighbours, rng)
            trial = self._local_search(trial, distances, neighbours, stop_time)
            trial_cost = self._cost(trial, distances)
            if trial_cost < best_cost - 1e-9:
                best_route, best_cost = trial[:], trial_cost
                current, current_cost = trial, trial_cost
                stagnation = 0
            elif trial_cost <= current_cost * (1.001 + 0.0001 * min(stagnation, 10)):
                current, current_cost = trial, trial_cost
                stagnation += 1
            else:
                stagnation += 1

        return [int(node) for node in best_route]
