"""Register one state-maintenance ablation, preserving the selected algorithm."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

REFRESH = '''    @staticmethod
    def _refresh_routes(routes, demands, rid, position, previous, following, prefix,
                        loads, affected=None):
        if affected is None:
            loads = np.zeros(max(len(routes), 1))
            affected = range(len(routes))
        for r in affected:
            route = routes[r]
            total = 0.0
            for i, customer in enumerate(route):
                total += demands[customer]
                rid[customer], position[customer] = r, i
                previous[customer] = route[i - 1] if i else 0
                following[customer] = route[i + 1] if i + 1 < len(route) else 0
                prefix[customer] = total
            loads[r] = total
        return loads

'''


def build(source):
    begin = source.index('        def refresh():')
    end = source.index('        loads = refresh()', begin)
    replacement = '''        def refresh(affected=None):
            return self._refresh_routes(routes, demands, rid, position, previous,
                                        following, prefix, loads, affected)

'''
    code = source[:begin] + replacement + source[end:]
    code = code.replace('    def solve(', REFRESH + '    def solve(', 1)
    code = code.replace('routes[ra], routes[best_star[1]] = best_star[2], best_star[3]\n                        loads = refresh()',
                        'routes[ra], routes[best_star[1]] = best_star[2], best_star[3]\n                        loads = refresh((ra, best_star[1]))')
    code = code.replace('                loads = refresh()\n                improved = True',
                        '                loads = refresh((ra, r))\n                improved = True')
    if code.count('refresh((ra,') != 2:
        raise ValueError('expected two accepted move sites')
    return code


def main():
    root = ROOT / 'runs/minimal_cvrp_incremental_v1'
    if root.exists():
        raise FileExistsError(root)
    parent = ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/selected_artifact/selected/solver.py'
    source = parent.read_text()
    protocol = ROOT / 'data/bundles/minimal_v1/cvrp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = {'problem': 'cvrp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
            'evidence_type': 'human_engineering_single_mechanism_training_ablation',
            'budget': {'n_instances': 5, 'time_limit': 2, 'ite_max': 1000, 'seeds': [71400, 71401]},
            'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
            'parent_source_sha256': sha(parent),
            'hypothesis': 'Refresh only the two affected routes after accepted inter-route moves; preserve move checks, neighborhoods and random choices.'}
    for name, code in {'full_refresh': source, 'incremental_refresh': build(source)}.items():
        path = root / 'sources' / name / 'solver.py'
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding='utf-8')
        spec['sources'][name] = {'path': str(path), 'sha256': sha(path)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
