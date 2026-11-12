"""Keep ruin size four; test reconstruction decisions independently."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

REGRET = '''    while len(removed) > 0:
        chosen, position, value = 0, 0, 0.0
        best_regret, best_load = -np.inf, -np.inf
        for r in range(len(removed)):
            job = removed[r]
            insertion_costs(candidate, count, job, p, forward, backward, costs)
            pos, cost = guided_position(costs, count + 1, -1, job, p, forward, backward)
            second = np.inf
            for i in range(count + 1):
                if i != pos:
                    second = min(second, costs[i])
            regret = second - cost if count > 0 else 0.0
            load = p[job].sum()
            if regret > best_regret or (regret == best_regret and load > best_load):
                chosen, position, value = r, pos, cost
                best_regret, best_load = regret, load
        insert(candidate, count, removed[chosen], position)
        count += 1
        removed = np.concatenate((removed[:chosen], removed[chosen + 1:]))
'''

BEAM = '''    width = 3
    beams = np.empty((width, len(candidate)), dtype=np.int64)
    beams[0] = candidate
    active = 1
    beam_values = np.empty(width)
    for job in removed:
        scores = np.empty((active, count + 1))
        for b in range(active):
            insertion_costs(beams[b], count, job, p, forward, backward, costs)
            scores[b] = costs[:count + 1]
        next_beams = np.empty_like(beams)
        selected = 0
        # Random tie selection without adding floating jitter to the objective.
        while selected < width and np.isfinite(scores).any():
            best_value = np.min(scores)
            indexes = np.flatnonzero(scores.ravel() == best_value)
            flat = indexes[np.random.randint(len(indexes))]
            b, pos = flat // (count + 1), flat % (count + 1)
            scores[b, pos] = np.inf
            proposed = beams[b].copy()
            insert(proposed, count, job, pos)
            duplicate = False
            for prior in range(selected):
                if np.array_equal(proposed[:count + 1], next_beams[prior, :count + 1]):
                    duplicate = True
                    break
            if not duplicate:
                next_beams[selected] = proposed
                beam_values[selected] = best_value
                selected += 1
        beams, active = next_beams, selected
        count += 1
    # Explore the surviving reconstructions with the unchanged insertion kernel.
    value = np.inf
    for b in range(active):
        candidate_value = local_search(beams[b], count, beam_values[b], p, forward, backward, costs, neutral)
        if candidate_value < value:
            candidate, value = beams[b].copy(), candidate_value
    return candidate, value
'''


def variants(source):
    start = source.index('    value = 0.0\n    for job in removed:', source.index('def step('))
    end = source.index('    value = local_search(candidate, count, value', start)
    tail = source.index('\n\n\nclass SearchStrategy:', end)
    regret = source[:start] + REGRET + source[end:]
    beam = source[:start] + BEAM + source[tail:]
    return {'parent': source, 'regret_reconstruction': regret, 'beam_three': beam}


def main():
    root = ROOT / 'runs/minimal_fssp_reconstruction_v1'
    if root.exists():
        raise FileExistsError(root)
    protocol = ROOT / 'data/bundles/minimal_v1/fssp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = {'problem': 'fssp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'evidence_type': 'human_engineering_single_mechanism_training_ablation',
        'budget': {'n_instances': 3, 'time_limit': 30, 'ite_max': 10000000, 'seeds': [71400, 71401]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
        'hypothesis': 'Low accepted-step rate and long incumbent stagnation motivate choosing reconstructions by regret or retaining three insertion branches. Preserve ruin size, temperature and restart policy.',
        'complexity': 'Regret O(d^2*n*m), beam O(B*d*n*m + B*LS); d=4, B=3. Auxiliary memory O(n*m+B*n).'}
    source = (ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py').read_text()
    for name, code in variants(source).items():
        path = root / 'sources' / name / 'solver.py'
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding='utf-8')
        spec['sources'][name] = {'path': str(path), 'sha256': sha(path)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
