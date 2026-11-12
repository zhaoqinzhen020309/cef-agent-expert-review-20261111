"""Compare bounded pair relocation with/without critical-block boundary guidance."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

BOUNDARIES = '''
@njit
def critical_boundaries(sequence, p, forward):
    n, m = len(sequence), p.shape[1]
    forward[0, :] = 0
    forward[:, 0] = 0
    for i in range(n):
        for k in range(m):
            forward[i + 1, k + 1] = max(forward[i, k + 1], forward[i + 1, k]) + p[sequence[i], k]
    chosen = np.zeros(max(0, n - 1), dtype=np.bool_)
    i, k = n, m
    while i > 0 and k > 0:
        if i > 1 and (k == 1 or forward[i - 1, k] >= forward[i, k - 1]):
            i -= 1
        else:
            # At each machine change, try the adjacent pair on either side.
            if i >= 2:
                chosen[i - 2] = True
            if i < n:
                chosen[i - 1] = True
            k -= 1
    return np.nonzero(chosen)[0]

'''


def variants(parent):
    old = (ROOT / 'problems/fssp/minimal_refinement_seed_v1/solver.py').read_text()
    tree = ast.parse(old)
    block = '\n\n'.join('@njit\n' + ast.get_source_segment(old, n) for n in tree.body
                         if isinstance(n, ast.FunctionDef) and n.name in ('block_costs', 'block_search'))
    base = parent.replace('class SearchStrategy:', block + '\n\nclass SearchStrategy:')
    base = base.replace('for _ in range(iteration_limit):', 'for iteration in range(iteration_limit):')
    base = base.replace('            iterations += 1', '''            if iteration % 32 == 0:
                candidate_value = block_search(candidate, candidate_value, p, forward, backward, costs, 2)
                candidate_value = local_search(candidate, n, candidate_value, p, forward, backward, costs, self.NEUTRAL_MOVES)
            iterations += 1''')
    guided = base.replace('@njit\ndef block_search', BOUNDARIES + '@njit\ndef block_search')
    guided = guided.replace('        np.random.shuffle(positions)\n        improved',
                            '        positions = critical_boundaries(sequence, p, forward)\n        np.random.shuffle(positions)\n        improved')
    return {'parent': parent, 'uniform_blocks': base, 'critical_boundary_blocks': guided}


def main():
    root = ROOT / 'runs/minimal_fssp_critical_blocks_v2'
    if root.exists():
        raise FileExistsError(root)
    protocol = ROOT / 'data/bundles/minimal_v1/fssp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = {'problem': 'fssp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'evidence_type': 'human_engineering_paired_neighborhood_ablation',
        'budget': {'n_instances': 3, 'time_limit': 30, 'ite_max': 10000000, 'seeds': [71400, 71401]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
        'hypothesis': 'Insertion stagnation motivates pair relocation every 32 steps, at most 2 improving passes. Critical variant visits only pairs near critical-path machine changes; uniform variant isolates boundary guidance.',
        'prior_negative': 'critical_order regressed, deletion-zero filter had no effect; no extra restarts or destruction size changes'}
    parent = (ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py').read_text()
    for name, code in variants(parent).items():
        path = root / 'sources' / name / 'solver.py'
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding='utf-8')
        spec['sources'][name] = {'path': str(path), 'sha256': sha(path)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
