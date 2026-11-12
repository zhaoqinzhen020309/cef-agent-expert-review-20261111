"""Generate a standalone optimized solver while preserving the historical baseline."""
from pathlib import Path


def build():
    root = Path(__file__).resolve().parents[1]
    source = (root / "problems/tsp_gls/problem.py").read_text(encoding="utf-8")
    start = source.index("        tour = [depot]")
    end = source.index("        tour.append(depot)", start)
    source = source[:start] + '''        node_count = len(distance_matrix)
        tour = [int(depot)]
        visited = np.zeros(node_count, dtype=bool)
        visited[depot] = True
        while len(tour) < node_count:
            available = np.flatnonzero(~visited)
            next_node = int(available[np.argmin(distance_matrix[tour[-1], available])])
            tour.append(next_node)
            visited[next_node] = True
''' + source[end:]
    start = source.index("        best_move = None", source.index("    def two_opt_a2a"))
    end = source.index("    def relocate(", start)
    source = source[:start] + '''        best_move, best_delta = None, set_delta
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

''' + source[end:]
    start = source.index("        best_move = None", source.index("    def relocate_a2a"))
    end = source.index("    def route_to_tour", start)
    source = source[:start] + '''        best_move, best_delta = None, set_delta
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

''' + source[end:]
    source = source.replace("edge_penalty = np.zeros((length, length))", "edge_penalty = {}\n        guided_distances = edge_weight.copy()")
    start = source.index("            edge_penalty = self.update_edge_penalties(")
    end = source.index("            guided_cost =", start)
    source = source[:start] + '''            # Only maximally useful edges change; no dense O(n^2) copy per iteration.
            edges = [(node, int(cur_route[node, 1])) for node in range(length)]
            utilities = [edge_weight[a, b] / (1.0 + edge_penalty.get((a, b), 0.0)) for a, b in edges]
            maximum = max(utilities)
            for utility, (a, b) in zip(utilities, edges):
                if utility == maximum:
                    for edge in ((a, b), (b, a)):
                        edge_penalty[edge] = edge_penalty.get(edge, 0.0) + 1.0
                        guided_distances[edge] = edge_weight[edge] + penalty_lambda * edge_penalty[edge]
''' + source[end:]
    output = root / "problems/tsp_gls/optimized/solver.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    build()
