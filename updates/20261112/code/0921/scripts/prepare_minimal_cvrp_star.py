"""Training-only SWAP* work reuse and ordered-candidate ablations."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

STAR = '''    @staticmethod
    def _swap_star(a, targets, routes, rid, previous, following, loads, dist, demands, capacity, stop):
        ra = int(rid[a])
        targets = targets[:12]
        feasible = ((loads[ra] - demands[a] + demands[targets] <= capacity + 1e-9)
                    & (loads[rid[targets]] - demands[targets] + demands[a] <= capacity + 1e-9))
        targets = targets[feasible]
        if not len(targets):
            return None
        remaining_a = [v for v in routes[ra] if v != a]
        edges = np.asarray([0] + remaining_a + [0], dtype=int)
        adds = (dist[edges[:-1, None], targets] + dist[targets, edges[1:, None]]
                - dist[edges[:-1], edges[1:]][:, None])
        insert_b = np.argmin(adds, axis=0)
        removal_a = dist[previous[a], following[a]] - dist[previous[a], a] - dist[a, following[a]]
        route_cache = {}
        best = None
        for column, b in enumerate(targets):
            if time.time() >= stop:
                break
            b, r = int(b), int(rid[b])
            if r not in route_cache:
                route = routes[r]
                edges_r = np.asarray([0] + route + [0], dtype=int)
                insertion = dist[edges_r[:-1], a] + dist[a, edges_r[1:]] - dist[edges_r[:-1], edges_r[1:]]
                # Removing b destroys only its two incident edges. Three best
                # original edges plus the new joining edge therefore suffice.
                order = np.argsort(insertion, kind='stable')[:3]
                route_cache[r] = (insertion, order, {v: i for i, v in enumerate(route)})
            insertion, order, positions = route_cache[r]
            ib = positions[b]
            add_a = dist[previous[b], a] + dist[a, following[b]] - dist[previous[b], following[b]]
            pos_a = ib
            for edge in order:
                if edge == ib or edge == ib + 1:
                    continue
                position = int(edge - (edge > ib))
                cost = insertion[edge]
                if cost < add_a or (cost == add_a and position < pos_a):
                    add_a, pos_a = cost, position
            removal_b = dist[previous[b], following[b]] - dist[previous[b], b] - dist[b, following[b]]
            pos_b = int(insert_b[column])
            delta = removal_a + removal_b + adds[pos_b, column] + add_a
            if delta < -1e-9 and (best is None or delta < best[0]):
                new_a = remaining_a.copy()
                new_b = [v for v in routes[r] if v != b]
                new_a.insert(pos_b, b)
                new_b.insert(pos_a, a)
                best = (delta, r, new_a, new_b)
        return best

'''


def cached_star(source):
    start = source.index('                    best_star = None')
    end = source.index('                    if best_star is not None:', start)
    code = source[:start] + '''                    best_star = self._swap_star(a, targets, routes, rid, previous, following,
                                                loads, dist, demands, capacity, stop)
''' + source[end:]
    return code.replace('    def solve(', STAR + '    def solve(', 1)


def main():
    root = ROOT / 'runs/minimal_cvrp_star_v1'
    if root.exists():
        raise FileExistsError(root)
    parent = ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/selected_artifact/selected/solver.py'
    original = parent.read_text()
    ordered = original.replace('            neighbors[row] = np.argpartition(dist[row, 1:], k - 1)[:k] + 1',
        '''            chosen = np.argpartition(dist[row, 1:], k - 1)[:k] + 1
            neighbors[row] = chosen[np.argsort(dist[row, chosen], kind='stable')]''')
    protocol = ROOT / 'data/bundles/minimal_v1/cvrp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = {'problem': 'cvrp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'evidence_type': 'human_engineering_single_mechanism_training_ablation',
        'budget': {'n_instances': 5, 'time_limit': 2, 'ite_max': 1000, 'seeds': [71400, 71401]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
        'hypothesis': 'Most profiled cost occurs in solve neighborhood scanning. Reuse SWAP* insertion costs across candidates, or order the existing nearest list; compare separately.',
        'diagnostic_receipt': str(ROOT / 'runs/cvrp_training_diagnostics_v1/receipt.json')}
    for name, code in {'parent': original, 'cached_star': cached_star(original), 'ordered_neighbors': ordered}.items():
        path = root / 'sources' / name / 'solver.py'
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding='utf-8')
        spec['sources'][name] = {'path': str(path), 'sha256': sha(path)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
