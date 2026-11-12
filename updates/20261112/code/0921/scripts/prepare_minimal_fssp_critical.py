"""Single-mechanism critical-path/removal-bound insertion controls."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

KERNEL = '''
@njit
def removal_relief(sequence, count, p, forward, backward):
    machines = p.shape[1]
    forward[0, :] = 0
    forward[:, 0] = 0
    backward[count, :] = 0
    backward[:, machines] = 0
    for i in range(count):
        for k in range(machines):
            forward[i + 1, k + 1] = max(forward[i, k + 1], forward[i + 1, k]) + p[sequence[i], k]
    for i in range(count - 1, -1, -1):
        for k in range(machines - 1, -1, -1):
            backward[i, k] = max(backward[i + 1, k], backward[i, k + 1]) + p[sequence[i], k]
    value = forward[count, machines]
    relief = np.zeros(len(p))
    for i in range(count):
        removed_cost = 0.0
        for k in range(machines):
            removed_cost = max(removed_cost, forward[i, k + 1] + backward[i + 1, k])
        relief[sequence[i]] = value - removed_cost
    return relief

'''


def variants(source):
    base = source.replace('@njit\ndef local_search', KERNEL + '@njit\ndef local_search')
    ordered = base.replace('        np.random.shuffle(jobs)\n        improved', '''        np.random.shuffle(jobs)
        relief = removal_relief(sequence, count, p, forward, backward)
        jobs = jobs[np.argsort(-relief[jobs], kind='mergesort')]
        improved''')
    # A zero deletion relief certifies that reinsertion cannot strictly improve.
    # Recompute after ANY accepted move, including neutral moves, to avoid stale bounds.
    filtered = base.replace('        improved = False\n        for job in jobs:', '''        improved = False
        dirty = True
        for job in jobs:
            if dirty:
                relief = removal_relief(sequence, count, p, forward, backward)
                dirty = False
            if relief[job] <= 0.0:
                continue''')
    filtered = filtered.replace('            if accept:\n', '            if accept:\n                dirty = True\n')
    return {'symmetric_parent': source, 'critical_order': ordered, 'critical_filter': filtered}


def main():
    root = ROOT / 'runs/minimal_fssp_critical_v1'
    if root.exists():
        raise FileExistsError(root)
    parent = ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py'
    protocol = ROOT / 'data/bundles/minimal_v1/fssp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    diagnostic = ROOT / 'runs/fssp_training_diagnostics_v2/receipt.json'
    spec = {'problem': 'fssp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'evidence_type': 'human_engineering_single_mechanism_training_ablation',
        'budget': {'n_instances': 3, 'time_limit': 30, 'ite_max': 10000000, 'seeds': [71400, 71401]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
        'diagnostic_receipt': str(diagnostic), 'diagnostic_sha256': sha(diagnostic),
        'hypothesis': 'Low strict-move rate motivates critical-path-aware job ordering OR exact deletion-bound filtering. Compare separately, without larger ruin or restarts.',
        'filter_semantics': 'O(n*m) recomputation after every accepted move; skips only jobs with no deletion relief. Deliberately removes their neutral relocations, so quality must be measured.'}
    for name, code in variants(parent.read_text()).items():
        path = root / 'sources' / name / 'solver.py'
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding='utf-8')
        spec['sources'][name] = {'path': str(path), 'sha256': sha(path)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
