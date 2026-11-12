"""Repair the observed Numba sort-kind failure in a new immutable candidate."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / 'runs/minimal_fssp_compile_repair_v1'
    if root.exists():
        raise FileExistsError(root)
    run = ROOT / 'runs/minimal_diagnostic_search_v1/jobs/fssp_diagnostic'
    state = json.loads((run / 'search_state.json').read_text())
    node = next(n for n in state['nodes'] if n['node_id'] == 'node_0003')
    original = run / node['solver_relpath']
    source = original.read_text()
    if source.count('np.argsort(secondary, kind="stable")') != 1:
        raise ValueError('unexpected failed candidate')
    fixed = source.replace('np.argsort(secondary, kind="stable")', 'np.argsort(secondary, kind="mergesort")')
    parent = ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py'
    protocol = ROOT / 'data/bundles/minimal_v1/fssp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = {'problem': 'fssp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'evidence_type': 'human_compile_repair_of_agent_diagnostic_candidate',
        'budget': {'n_instances': 3, 'time_limit': 30, 'ite_max': 10000000, 'seeds': [71400, 71401]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
        'original_failed_source_sha256': sha(original),
        'hypothesis': 'Keep the bounded tie-ranking algorithm, replacing unsupported stable with Numba-supported stable mergesort. No uninitialized sorting keys.'}
    for name, code in {'parent': parent.read_text(), 'compatible_tie_sort': fixed}.items():
        path = root / 'sources' / name / 'solver.py'
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding='utf-8')
        spec['sources'][name] = {'path': str(path), 'sha256': sha(path)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
