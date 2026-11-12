import numpy as np
import time


class SearchStrategy:
    '''Clarke-Wright savings construction, then time-bounded 2-opt, relocate,
    tail-exchange, and perturbation (ruin-and-recreate) restarts.'''

    def _savings_once(self, instance, deadline, iteration_limit, seed, distance_weight):
        dist = np.asarray(instance['distance_matrix'], dtype=np.float64)
        demands = np.asarray(instance['demands'], dtype=np.float64)
        capacity = float(instance['capacity'])
        n = len(dist)
        eps = 1e-9

        remaining = deadline - time.time()
        customers = [int(i) for i in range(1, n) if demands[i] <= capacity + eps]
        if not customers:
            return []
        if remaining <= 0.05:
            return [[c] for c in customers]

        safety = 0.1 if remaining > 0.25 else max(0.02, remaining * 0.05)
        hard_deadline = deadline - safety

        routes = [[c] for c in customers]
        caps = [float(demands[c]) for c in customers]
        route_of = np.full(n, -1, dtype=np.int32)
        for rid, c in enumerate(customers):
            route_of[c] = rid
        first = customers[:]
        last = customers[:]

        if n > 2 and time.time() < hard_deadline:
            d0 = dist[0]
            ii, jj = np.triu_indices(n - 1, k=1)
            ii = ii + 1
            jj = jj + 1
            valid = (demands[ii] <= capacity + eps) & (demands[jj] <= capacity + eps)
            if np.any(valid):
                d0i = d0[ii]
                d0j = d0[jj]
                dij = dist[ii, jj]
                savings = d0i + d0j - distance_weight * dij
                valid &= savings > eps
                si = ii[valid]
                sj = jj[valid]
                ss = savings[valid]
                if len(ss) > 0:
                    order = np.argsort(-ss, kind='stable')
                    for p in order:
                        if time.time() > hard_deadline:
                            break
                        i = int(si[p])
                        j = int(sj[p])
                        r1 = int(route_of[i])
                        r2 = int(route_of[j])
                        if r1 < 0 or r2 < 0 or r1 == r2:
                            continue
                        if caps[r1] + caps[r2] > capacity + eps:
                            continue
                        f1 = first[r1]
                        l1 = last[r1]
                        f2 = first[r2]
                        l2 = last[r2]
                        if not ((i == f1 or i == l1) and (j == f2 or j == l2)):
                            continue
                        r1_seg = routes[r1] if i == l1 else routes[r1][::-1]
                        r2_seg = routes[r2] if j == f2 else routes[r2][::-1]
                        new_route = r1_seg + r2_seg
                        for c in routes[r2]:
                            route_of[c] = r1
                        routes[r1] = new_route
                        routes[r2] = []
                        caps[r1] += caps[r2]
                        caps[r2] = 0.0
                        first[r1] = new_route[0]
                        last[r1] = new_route[-1]
                        first[r2] = last[r2] = -1

        return [r[:] for r in routes if r]

    def _structural_signal(self, demands, capacity):
        '''Cheaply estimate instance structure from observable inputs.

        Returns (n, cv, cap_ratio) where cv is the demand coefficient of
        variation and cap_ratio is capacity divided by mean demand. These are
        computed once per instance and reused for the neighbor width and for
        the construction schedule.'''
        cust = demands[1:]
        n = int(len(cust))
        if n <= 0:
            return 0, 0.0, 0.0
        mean_d = float(np.mean(cust))
        if mean_d <= 1e-12:
            return n, 0.0, float('inf')
        std_d = float(np.std(cust))
        cv = std_d / mean_d
        cap_ratio = float(capacity) / mean_d
        return n, cv, cap_ratio

    def _select_width(self, n, cv, cap_ratio):
        '''Map observable structure to neighbor-list width k.

        Large, low-dispersion instances get a wider list; small or
        high-dispersion instances get a narrower list. Boundary values are
        deliberately conservative so feasibility logic is untouched.'''
        base = 24
        if n <= 0:
            return base
        if n < 250:
            k = 14
        elif n < 500:
            k = 20
        else:
            k = 30
        if cv >= 1.2:
            k -= 6
        elif cv <= 0.6:
            k += 6
        if cap_ratio <= 3.0:
            k -= 4
        elif cap_ratio >= 12.0:
            k += 4
        if k < 12:
            k = 12
        if k > 40:
            k = 40
        return int(k)

    def _savings_construct(self, instance, deadline, iteration_limit, seed, weights):
        best, best_cost = None, float('inf')
        dist = np.asarray(instance['distance_matrix'], dtype=float)
        for weight in weights:
            if best is not None and time.time() >= deadline - .12:
                break
            candidate = self._savings_once(instance, deadline, iteration_limit, seed, weight)
            cost = self._total_cost(candidate, dist)
            if cost < best_cost:
                best, best_cost = candidate, cost
        return best

    def _route_cost(self, route, dist):
        if not route:
            return 0.0
        cost = float(dist[0, route[0]])
        for idx in range(len(route) - 1):
            cost += float(dist[route[idx], route[idx + 1]])
        cost += float(dist[route[-1], 0])
        return cost

    def _total_cost(self, routes, dist):
        return sum(self._route_cost(route, dist) for route in routes if route)

    def _two_opt_route(self, route, dist, hard_deadline):
        improved = True
        while improved and time.time() < hard_deadline:
            improved = False
            k = len(route)
            for i in range(k - 1):
                if time.time() > hard_deadline:
                    break
                a = route[i - 1] if i > 0 else 0
                b = route[i]
                for j in range(i + 1, k):
                    c = route[j]
                    d = route[j + 1] if j + 1 < k else 0
                    delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                    if delta < -1e-10:
                        route[i:j + 1] = route[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        return self._route_cost(route, dist)

    def _relocate_first_improvement(self, routes, caps, dist, demands, capacity, hard_deadline):
        eps = 1e-9
        for src_idx, src_route in enumerate(routes):
            if not src_route:
                continue
            if time.time() > hard_deadline:
                return False
            for pos, c in enumerate(src_route):
                if time.time() > hard_deadline:
                    return False
                demand_c = float(demands[c])
                k = len(src_route)
                prev = src_route[pos - 1] if pos > 0 else 0
                nxt = src_route[pos + 1] if pos + 1 < k else 0
                remove_delta = dist[prev, nxt] - dist[prev, c] - dist[c, nxt]
                for dst_idx, dst_route in enumerate(routes):
                    if dst_idx == src_idx or not dst_route:
                        continue
                    if time.time() > hard_deadline:
                        return False
                    if caps[dst_idx] + demand_c > capacity + eps:
                        continue
                    m = len(dst_route)
                    best_insert_delta = float('inf')
                    best_p = 0
                    for p in range(m + 1):
                        a = dst_route[p - 1] if p > 0 else 0
                        b = dst_route[p] if p < m else 0
                        ins_delta = dist[a, c] + dist[c, b] - dist[a, b]
                        if ins_delta < best_insert_delta:
                            best_insert_delta = ins_delta
                            best_p = p
                    delta = remove_delta + best_insert_delta
                    if delta < -1e-10:
                        del src_route[pos]
                        caps[src_idx] -= demand_c
                        dst_route.insert(best_p, c)
                        caps[dst_idx] += demand_c
                        return True
        return False

    SAVINGS_WEIGHTS = (1.0, 1.428571, 0.833333)
    NEIGHBORS = 24
    TAIL_EXCHANGE = True
    RUIN_MAX_ROUTES = 4
    RUIN_SEG_MIN = 1
    RUIN_SEG_MAX = 6
    RESTART_CAP = 60
    RESTART_SLICE = 0.6

    def _greedy_reinsert(self, removed, routes, caps, dist, demands, capacity, eps=1e-9):
        for c in removed:
            demand_c = float(demands[c])
            best_delta = float('inf')
            best_rid = -1
            best_pos = 0
            for ridx, route in enumerate(routes):
                if not route:
                    continue
                if caps[ridx] + demand_c > capacity + eps:
                    continue
                m = len(route)
                for p in range(m + 1):
                    a = route[p - 1] if p > 0 else 0
                    b = route[p] if p < m else 0
                    ins = dist[a, c] + dist[c, b] - dist[a, b]
                    if ins < best_delta:
                        best_delta = ins
                        best_rid = ridx
                        best_pos = p
            if best_rid < 0:
                routes.append([c])
                caps.append(demand_c)
            else:
                routes[best_rid].insert(best_pos, c)
                caps[best_rid] += demand_c
        return routes, caps

    def _perturb(self, routes, caps, dist, demands, capacity, rng, stop):
        eps = 1e-9
        nonempty = [i for i, r in enumerate(routes) if r]
        if len(nonempty) < 2:
            return routes, caps, False
        num = min(self.RUIN_MAX_ROUTES, len(nonempty))
        if num < 1:
            return routes, caps, False
        chosen = rng.choice(len(nonempty), size=num, replace=False)
        removed = []
        for idx in chosen:
            ridx = nonempty[int(idx)]
            route = routes[ridx]
            if not route:
                continue
            if time.time() >= stop:
                break
            max_seg = min(self.RUIN_SEG_MAX, len(route))
            seg_len = int(rng.integers(self.RUIN_SEG_MIN, max_seg + 1)) if max_seg >= self.RUIN_SEG_MIN else 0
            if seg_len <= 0:
                continue
            start = int(rng.integers(0, len(route) - seg_len + 1))
            for c in route[start:start + seg_len]:
                removed.append(c)
                caps[ridx] -= float(demands[c])
            routes[ridx] = route[:start] + route[start + seg_len:]
        if not removed:
            return routes, caps, False
        rng.shuffle(removed)
        routes, caps = self._greedy_reinsert(removed, routes, caps, dist, demands, capacity, eps)
        return routes, caps, True

    def solve(self, instance, deadline, iteration_limit, seed):
        start = time.time()
        remaining = deadline - start
        dist = np.asarray(instance['distance_matrix'], dtype=float)
        demands = np.asarray(instance['demands'], dtype=float)
        capacity = float(instance['capacity'])
        n = len(dist)

        # Cheap structural signal computed once per instance.
        n_sig, cv, cap_ratio = self._structural_signal(demands, capacity)
        width = self._select_width(n_sig, cv, cap_ratio)

        # Derive the savings weight schedule from the same signal.
        if cv >= 1.2:
            weights = (1.0, 1.2, 0.9)
        elif cv <= 0.6:
            weights = (1.0, 1.5, 0.8, 1.25)
        else:
            weights = self.SAVINGS_WEIGHTS

        # Derive restart slice from the signal: tighter instances get a
        # slightly larger perturbation budget.
        restart_slice = self.RESTART_SLICE
        if cv >= 1.2:
            restart_slice = 0.5
        elif cv <= 0.6:
            restart_slice = 0.7

        construction_deadline = min(deadline, start + max(.25, remaining * .3))
        routes = self._savings_construct(instance, construction_deadline, iteration_limit, seed, weights)

        stop = deadline - .12
        if n < 3 or time.time() >= stop:
            return routes

        k = min(width + 1, n - 1)
        if k < 2:
            k = min(2, n - 1)
        neighbors = np.empty((n, k), dtype=np.int64)
        for row in range(n):
            neighbors[row] = np.argpartition(dist[row, 1:], k - 1)[:k] + 1
        rid = np.full(n, -1, dtype=np.int64)
        position = np.zeros(n, dtype=np.int64)
        previous = np.zeros(n, dtype=np.int64)
        following = np.zeros(n, dtype=np.int64)
        prefix = np.zeros(n)
        loads = np.zeros(len(routes))
        rng = np.random.default_rng(seed)

        def refresh():
            if len(loads) < len(routes):
                pass
            loads_ = np.zeros(max(len(routes), 1))
            for r, route in enumerate(routes):
                total = 0.0
                for i, customer in enumerate(route):
                    total += demands[customer]
                    rid[customer], position[customer] = r, i
                    previous[customer] = route[i - 1] if i else 0
                    following[customer] = route[i + 1] if i + 1 < len(route) else 0
                    prefix[customer] = total
                loads_[r] = total
            return loads_

        loads = refresh()
        incumbent = [r[:] for r in routes if r]
        incumbent_cost = self._total_cost(incumbent, dist)

        restarts = 0
        while time.time() < stop and restarts < self.RESTART_CAP:
            improved = False
            for route in routes:
                if time.time() >= stop:
                    break
                if len(route) > 2:
                    old_cost = self._route_cost(route, dist)
                    if self._two_opt_route(route, dist, stop) < old_cost - 1e-9:
                        improved = True
            loads = refresh()
            for customer in rng.permutation(n - 1) + 1:
                if time.time() >= stop:
                    break
                a = int(customer)
                ra = int(rid[a])
                targets = neighbors[a]
                targets = targets[(targets != a) & (rid[targets] != ra)]
                if not len(targets):
                    continue
                rb = rid[targets]
                pa, na = previous[a], following[a]
                pb, nb = previous[targets], following[targets]
                remove = dist[pa, na] - dist[pa, a] - dist[a, na]
                deltas = np.full((4, len(targets)), np.inf)
                can_relocate = loads[rb] + demands[a] <= capacity + 1e-9
                deltas[0] = np.where(can_relocate, remove + dist[pb, a] + dist[a, targets] - dist[pb, targets], np.inf)
                deltas[1] = np.where(can_relocate, remove + dist[targets, a] + dist[a, nb] - dist[targets, nb], np.inf)
                can_swap = ((loads[ra] - demands[a] + demands[targets] <= capacity + 1e-9)
                            & (loads[rb] - demands[targets] + demands[a] <= capacity + 1e-9))
                deltas[2] = np.where(can_swap,
                    dist[pa, targets] + dist[targets, na] - dist[pa, a] - dist[a, na]
                    + dist[pb, a] + dist[a, nb] - dist[pb, targets] - dist[targets, nb], np.inf)
                if self.TAIL_EXCHANGE:
                    can_exchange = ((prefix[a] + loads[rb] - prefix[targets] <= capacity + 1e-9)
                                    & (prefix[targets] + loads[ra] - prefix[a] <= capacity + 1e-9))
                    deltas[3] = np.where(can_exchange, dist[a, nb] + dist[targets, na] - dist[a, na] - dist[targets, nb], np.inf)
                index = int(np.argmin(deltas))
                move, column = divmod(index, len(targets))
                if deltas[move, column] >= -1e-9:
                    best_star = None
                    for b in targets[:12]:
                        if time.time() >= stop:
                            break
                        b = int(b)
                        r = int(rid[b])
                        if loads[ra] - demands[a] + demands[b] > capacity + 1e-9 or loads[r] - demands[b] + demands[a] > capacity + 1e-9:
                            continue
                        remaining_a = [v for v in routes[ra] if v != a]
                        remaining_b = [v for v in routes[r] if v != b]
                        edges_a = np.asarray([0] + remaining_a + [0], dtype=int)
                        edges_b = np.asarray([0] + remaining_b + [0], dtype=int)
                        add_b = dist[edges_a[:-1], b] + dist[b, edges_a[1:]] - dist[edges_a[:-1], edges_a[1:]]
                        add_a = dist[edges_b[:-1], a] + dist[a, edges_b[1:]] - dist[edges_b[:-1], edges_b[1:]]
                        insert_b, insert_a = int(np.argmin(add_b)), int(np.argmin(add_a))
                        remove_b = dist[previous[b], following[b]] - dist[previous[b], b] - dist[b, following[b]]
                        delta = remove + remove_b + add_b[insert_b] + add_a[insert_a]
                        if delta < -1e-9 and (best_star is None or delta < best_star[0]):
                            remaining_a.insert(insert_b, b)
                            remaining_b.insert(insert_a, a)
                            best_star = (delta, r, remaining_a, remaining_b)
                    if best_star is not None:
                        routes[ra], routes[best_star[1]] = best_star[2], best_star[3]
                        loads = refresh()
                        improved = True
                    continue
                b = int(targets[column])
                r = int(rid[b])
                ia, ib = int(position[a]), int(position[b])
                if move < 2:
                    del routes[ra][ia]
                    routes[r].insert(ib + move, a)
                elif move == 2:
                    routes[ra][ia], routes[r][ib] = b, a
                else:
                    tail_a, tail_b = routes[ra][ia + 1:], routes[r][ib + 1:]
                    routes[ra] = routes[ra][:ia + 1] + tail_b
                    routes[r] = routes[r][:ib + 1] + tail_a
                loads = refresh()
                improved = True

            cur = [r[:] for r in routes if r]
            cur_cost = self._total_cost(cur, dist)
            if cur_cost < incumbent_cost - 1e-9:
                incumbent, incumbent_cost = cur, cur_cost

            if time.time() >= stop:
                break
            if not improved and restarts >= self.RESTART_CAP:
                break

            if not improved:
                restarts += 1
                slice_stop = min(stop, time.time() + restart_slice)
                trial = [r[:] for r in routes if r]
                trial_caps = []
                for r in trial:
                    trial_caps.append(float(sum(demands[c] for c in r)))
                trial, trial_caps, perturbed = self._perturb(
                    trial, trial_caps, dist, demands, capacity, rng, slice_stop)
                if perturbed:
                    routes = [r for r in trial if r]
                    loads = refresh()
                else:
                    routes = [r[:] for r in incumbent]
                    loads = refresh()
                    if restarts >= 4:
                        break
            if time.time() >= stop:
                break

        return [r for r in incumbent if r]
