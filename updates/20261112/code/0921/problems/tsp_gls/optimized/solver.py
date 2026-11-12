# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang,
#           Zhichao Lu, Qingfu Zhang, Evolution of Heuristics: Towards Efficient
#           Automatic Algorithm Design Using Large Language Model, ICML 2024.

import random
import time

import numpy as np


class SearchStrategy:
    """Complete TSP solver with the classic GLS guiding mechanism.

    The fixed evaluator lives outside this module. This class owns initialization,
    local moves, local search, guided penalties, and the search loop.
    """

    DEADLINE_RETURN_MARGIN_SEC = 0.25

    @staticmethod
    def _deadline_reached(deadline):
        return deadline is not None and time.time() >= deadline

    def nearest_neighbor_2end(self, distance_matrix, depot):
        node_count = len(distance_matrix)
        tour = [int(depot)]
        visited = np.zeros(node_count, dtype=bool)
        visited[depot] = True
        while len(tour) < node_count:
            available = np.flatnonzero(~visited)
            next_node = int(available[np.argmin(distance_matrix[tour[-1], available])])
            tour.append(next_node)
            visited[next_node] = True
        tour.append(depot)

        route = np.zeros((node_count, 2))
        route[0, 0] = tour[-2]
        route[0, 1] = tour[1]
        for index in range(1, node_count):
            route[tour[index], 0] = tour[index - 1]
            route[tour[index], 1] = tour[index + 1]
        return route

    def tour_cost_2end(self, dis_m, tour2end):
        cost = 0
        start = 0
        end = tour2end[0, 1]
        for _ in range(tour2end.shape[0]):
            cost += dis_m[start, end]
            start = end
            end = tour2end[start, 1]
        return cost

    def two_opt(self, tour, i, j):
        if i == j:
            return tour
        a = tour[i, 0]
        b = tour[j, 0]
        tour[i, 0] = tour[i, 1]
        tour[i, 1] = j
        tour[j, 0] = i
        tour[a, 1] = b
        tour[b, 1] = tour[b, 0]
        tour[b, 0] = a
        c = tour[b, 1]
        while tour[c, 1] != j:
            d = tour[c, 0]
            tour[c, 0] = tour[c, 1]
            tour[c, 1] = d
            c = d
        return tour

    def two_opt_cost(self, tour, distances, i, j):
        if i == j:
            return 0
        a = tour[i, 0]
        b = tour[j, 0]
        return distances[a, b] + distances[i, j] - distances[a, i] - distances[b, j]

    def two_opt_a2a(self, tour, distances, nearest_indices,
                    first_improvement=False, set_delta=0, deadline=None):
        best_move, best_delta = None, set_delta
        # One O(k) row workspace; retain original scan order and tie handling.
        for i in range(len(tour) - 1):
            if self._deadline_reached(deadline):
                break
            j = nearest_indices[i]
            a, b = tour[i, 0], tour[j, 0]
            delta = distances[a, b] + distances[i, j] - distances[a, i] - distances[b, j]
            valid = ((tour[j, 0] != i) & (tour[j, 1] != i) &
                     (j != tour[i, 0]) & (j != tour[i, 1]) &
                     (delta < best_delta) & ~np.isclose(0, delta))
            positions = np.flatnonzero(valid)
            if len(positions):
                pos = positions[0] if first_improvement else positions[np.argmin(delta[positions])]
                best_move, best_delta = (i, int(j[pos])), float(delta[pos])
        if best_move is not None:
            return best_delta, self.two_opt(tour, *best_move)
        return 0, tour

    def relocate(self, tour, i, j):
        a = tour[i, 0]
        b = tour[i, 1]
        tour[a, 1] = b
        tour[b, 0] = a

        d = tour[j, 1]
        tour[d, 0] = i
        tour[i, 0] = j
        tour[i, 1] = d
        tour[j, 1] = i
        return tour

    def relocate_cost(self, tour, distances, i, j):
        if i == j:
            return 0
        a = tour[i, 0]
        b = i
        c = tour[i, 1]
        d = j
        e = tour[j, 1]
        return (
            -distances[a, b] - distances[b, c] + distances[a, c]
            - distances[d, e] + distances[d, b] + distances[b, e]
        )

    def relocate_a2a(self, tour, distances, nearest_indices,
                     first_improvement=False, set_delta=0, deadline=None):
        best_move, best_delta = None, set_delta
        for i in range(len(tour) - 1):
            if self._deadline_reached(deadline):
                break
            j = nearest_indices[i]
            a, c, e = tour[i, 0], tour[i, 1], tour[j, 1]
            delta = (-distances[a, i] - distances[i, c] + distances[a, c]
                     - distances[j, e] + distances[j, i] + distances[i, e])
            # relocate_cost(i, i) is explicitly zero in the reference.
            valid = (e != i) & (j != i) & (delta < best_delta) & ~np.isclose(0, delta)
            positions = np.flatnonzero(valid)
            if len(positions):
                pos = positions[0] if first_improvement else positions[np.argmin(delta[positions])]
                best_move, best_delta = (i, int(j[pos])), float(delta[pos])
        if best_move is not None:
            return best_delta, self.relocate(tour, *best_move)
        return 0, tour

    def route_to_tour(self, route):
        start = 0
        tour = []
        for _ in range(len(route)):
            tour.append(route[start, 1])
            start = route[start, 1]
        return tour

    def local_search(self, init_tour, init_cost, distances, nearest_indices,
                     first_improvement=False, deadline=None):
        cur_route, cur_cost = init_tour, init_cost
        improved = True
        while improved:
            if self._deadline_reached(deadline):
                break
            improved = False

            delta, new_tour = self.two_opt_a2a(
                cur_route,
                distances,
                nearest_indices,
                first_improvement,
                deadline=deadline,
            )
            if delta < 0:
                improved = True
                cur_cost += delta
                cur_route = new_tour

            if self._deadline_reached(deadline):
                break
            delta, new_tour = self.relocate_a2a(
                cur_route,
                distances,
                nearest_indices,
                first_improvement,
                deadline=deadline,
            )
            if delta < 0:
                improved = True
                cur_cost += delta
                cur_route = new_tour

        return cur_route, cur_cost

    def update_edge_penalties(self, distances, local_opt_route, penalties):
        edge_utilities = []
        for node in range(len(local_opt_route)):
            next_node = int(local_opt_route[node, 1])
            utility = distances[node, next_node] / (
                1.0 + penalties[node, next_node]
            )
            edge_utilities.append((utility, node, next_node))

        max_utility = max(utility for utility, _node, _next in edge_utilities)
        updated = penalties.copy()
        for utility, node, next_node in edge_utilities:
            if utility == max_utility:
                updated[node, next_node] += 1.0
                updated[next_node, node] += 1.0
        return updated

    def build_guided_distances(self, distances, penalties, penalty_lambda):
        return distances + penalty_lambda * penalties

    def guided_local_search(self, edge_weight, nearest_indices, init_tour, init_cost,
                            time_limit, ite_max, penalty_alpha=0.3,
                            first_improvement=False):
        cur_route, cur_cost = self.local_search(
            init_tour,
            init_cost,
            edge_weight,
            nearest_indices,
            first_improvement,
            deadline=time_limit,
        )
        cur_cost = self.tour_cost_2end(edge_weight, cur_route)
        best_route, best_cost = cur_route.copy(), cur_cost

        length = len(edge_weight[0])
        iter_i = 0
        edge_penalty = {}
        guided_distances = edge_weight.copy()
        penalty_lambda = penalty_alpha * cur_cost / length

        while iter_i < ite_max and time.time() < time_limit:
            # Only maximally useful edges change; no dense O(n^2) copy per iteration.
            edges = [(node, int(cur_route[node, 1])) for node in range(length)]
            utilities = [edge_weight[a, b] / (1.0 + edge_penalty.get((a, b), 0.0)) for a, b in edges]
            maximum = max(utilities)
            for utility, (a, b) in zip(utilities, edges):
                if utility == maximum:
                    for edge in ((a, b), (b, a)):
                        edge_penalty[edge] = edge_penalty.get(edge, 0.0) + 1.0
                        guided_distances[edge] = edge_weight[edge] + penalty_lambda * edge_penalty[edge]
            guided_cost = self.tour_cost_2end(guided_distances, cur_route)
            cur_route, _guided_cost = self.local_search(
                cur_route,
                guided_cost,
                guided_distances,
                nearest_indices,
                first_improvement,
                deadline=time_limit,
            )
            cur_cost = self.tour_cost_2end(edge_weight, cur_route)

            if cur_cost < best_cost:
                best_route, best_cost = cur_route.copy(), cur_cost
            iter_i += 1

        return best_route, best_cost, iter_i

    def solve(self, distance_matrix, deadline, iteration_limit, seed):
        random.seed(seed)
        search_deadline = deadline - self.DEADLINE_RETURN_MARGIN_SEC
        initial_route = self.nearest_neighbor_2end(distance_matrix, 0).astype(int)
        initial_cost = self.tour_cost_2end(distance_matrix, initial_route)
        nearest_indices = np.argsort(distance_matrix, axis=1)[:, 1:101].astype(int)
        best_route, _best_cost, _iterations = self.guided_local_search(
            distance_matrix,
            nearest_indices,
            initial_route,
            initial_cost,
            search_deadline,
            iteration_limit,
            penalty_alpha=0.3,
            first_improvement=False,
        )
        return self.route_to_tour(best_route)
