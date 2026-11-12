import numpy as np
import time


class SearchStrategy:
    '''Clarke-Wright savings construction, then time-bounded 2-opt and relocate.'''

    def solve(self, instance, deadline, iteration_limit, seed):
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
                savings = d0i + d0j - dij
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

        active_indices = [i for i, r in enumerate(routes) if r]
        best_routes = [routes[i][:] for i in active_indices]
        best_cost = self._total_cost(best_routes, dist)
        current_routes = [routes[i][:] for i in active_indices]
        current_caps = [caps[i] for i in active_indices]
        current_cost = best_cost

        improvement = True
        while improvement and time.time() < hard_deadline:
            improvement = False

            non_empty = [i for i, r in enumerate(current_routes) if r]
            if len(non_empty) != len(current_routes):
                current_routes = [current_routes[i] for i in non_empty]
                current_caps = [current_caps[i] for i in non_empty]
            current_cost = self._total_cost(current_routes, dist)

            for ri, route in enumerate(current_routes):
                if time.time() > hard_deadline:
                    break
                if len(route) < 2:
                    continue
                before = self._route_cost(route, dist)
                after = self._two_opt_route(route, dist, hard_deadline)
                if after < before - 1e-10:
                    current_cost += after - before
                    improvement = True
                    if current_cost < best_cost - 1e-10:
                        best_cost = current_cost
                        best_routes = [r[:] for r in current_routes]

            if time.time() > hard_deadline:
                break

            moved = self._relocate_first_improvement(
                current_routes, current_caps, dist, demands, capacity, hard_deadline
            )
            if moved:
                current_cost = self._total_cost(current_routes, dist)
                improvement = True
                if current_cost < best_cost - 1e-10:
                    best_cost = current_cost
                    best_routes = [r[:] for r in current_routes]

        return [r[:] for r in best_routes if r]

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
