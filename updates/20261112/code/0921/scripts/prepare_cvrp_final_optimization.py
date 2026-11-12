"""Preserve the confirmed Agent mechanism while reducing state/allocation work."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_minimal_cvrp_incremental import build


def optimize(source):
    # Global relocate compacts the route list, so that site still needs a full
    # refresh. The parent transform targets the two granular move sites only.
    source = source.replace('            if moved:\n                routes = [r for r in routes if r]\n                loads = refresh()',
                            '            if moved:\n                routes = [r for r in routes if r]\n                loads = refresh(affected=None)')
    code = build(source)
    code = code.replace('                di = demands[ii]\n                dj = demands[jj]\n', '')
    code = code.replace('            loads = refresh()\n            routes = [r for r in routes if r]\n            loads = refresh()',
                        '            routes = [r for r in routes if r]\n            loads = refresh()')
    code = code.replace('            caps = [float(np.sum(demands[r])) if len(r) else 0.0 for r in routes]',
                        '            caps = loads')
    return code


def main():
    root = ROOT / 'runs/minimal_cvrp_final_optimization_v1'
    if root.exists():
        raise FileExistsError(root)
    source = ROOT / 'runs/minimal_cvrp_confirmation_v1/selected_artifact/solver.py'
    protocol = ROOT / 'data/bundles/minimal_v1/cvrp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = {'problem': 'cvrp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'evidence_type': 'semantics_preserving_engineering_optimization_of_confirmed_agent_candidate',
        'budget': {'n_instances': 5, 'time_limit': 2, 'ite_max': 1000, 'seeds': [71400, 71401, 71402]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {}, 'parent_sha256': sha(source),
        'hypothesis': 'Refresh only affected routes; reuse exact loads; remove duplicate refresh and two unused quadratic demand-index arrays. Preserve every move and random choice.'}
    for name, text in {'confirmed_agent': source.read_text(), 'optimized_agent': optimize(source.read_text())}.items():
        target = root / 'sources' / name / 'solver.py'
        target.parent.mkdir(parents=True)
        target.write_text(text, encoding='utf-8')
        spec['sources'][name] = {'path': str(target), 'sha256': sha(target)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
