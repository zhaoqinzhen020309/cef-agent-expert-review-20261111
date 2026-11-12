import numpy as np
import time


class SearchStrategy:
    """Capacity-feasible nearest-neighbor baseline, unbounded vehicle count.

    Routes contain customers only; the evaluator adds depot 0 at both ends.
    """
    def solve(self, instance, deadline, iteration_limit, seed):
        distance = instance["distance_matrix"]
        demands = instance["demands"]
        capacity = instance["capacity"]
        pending = np.ones(len(distance), dtype=bool)
        pending[0] = False
        routes = []
        while np.any(pending):
            route, node, remaining = [], 0, capacity
            while True:
                feasible = np.flatnonzero(pending & (demands <= remaining + 1e-9))
                if len(feasible) == 0:
                    break
                node = int(feasible[np.argmin(distance[node, feasible])])
                route.append(node)
                pending[node] = False
                remaining -= demands[node]
            routes.append(route)
        return routes
