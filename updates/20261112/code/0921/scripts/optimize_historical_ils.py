"""Derive an optional optimized ILS control, preserving the historical source.

This is a human engineering control, never an Agent-generated candidate or a
change to a registered campaign. Candidate order and arithmetic are preserved.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/p3_0812a/runs/pilot_pair_001/S1_fixed_single/nodes/scope_parent/solver.py"
TARGET = ROOT / "problems/tsp_gls/engineering_controls/historical_ils_optimized.py"

REPLACEMENTS = {
"_two_opt": '''    @staticmethod
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
''',
"_relocate": '''    @staticmethod
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
''',
"_or_opt_two": '''    @staticmethod
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
'''
}


def build():
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    methods = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.FunctionDef) and node.name in REPLACEMENTS]
    for node in sorted(methods, key=lambda n: n.lineno, reverse=True):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        lines[first:node.end_lineno] = [REPLACEMENTS[node.name]]
    result = "# Engineering control derived from historical scope_parent; not Agent-generated.\n" + "".join(lines)
    ast.parse(result)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(result, encoding="utf-8")
    return TARGET


if __name__ == "__main__":
    print(build())
