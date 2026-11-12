"""Choose final CVRP source on a fresh paired TRAIN receipt, before reading repeat validation."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / 'runs/minimal_cvrp_confirmation_v1'
    if root.exists():
        raise FileExistsError(root)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sources = {'prior_agent': ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/selected_artifact/selected/solver.py'}
    registry = ROOT / 'runs/minimal_cvrp_star_v1/registry.json'
    spec = json.loads(registry.read_text())
    scores = [(json.loads((registry.parent / (name + '.json')).read_text())['score'], name)
              for name in spec['sources'] if json.loads((registry.parent / (name + '.json')).read_text())['valid']]
    name = min(scores)[1]
    sources['engineering_' + name] = Path(spec['sources'][name]['path'])
    for repetition in (1, 2):
        run = ROOT / 'runs/minimal_diagnostic_search_v1/jobs' / f'cvrp_repeat_{repetition}'
        state = json.loads((run / 'search_state.json').read_text())
        if state['current_round'] != 1 or state.get('stopped_reason'):
            raise ValueError('independent search is not complete')
        rows = [r for r in state['evaluations'] if r['valid'] and r['split'] == 'train' and r['seeds'] == [71400, 71401]]
        best = min(rows, key=lambda r: (np.mean(r['gap_matrix']), r['candidate_id']))
        if best['candidate_id'] == '__baseline_control__':
            source = run / 'controls/baseline/solver.py'
        else:
            node = next(n for n in state['nodes'] if n['node_id'] == best['candidate_id'])
            source = run / node['solver_relpath']
        sources[f'independent_search_{repetition}'] = source
    protocol = ROOT / 'data/bundles/minimal_v1/cvrp/protocol.json'
    registration = {'problem': 'cvrp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'budget': {'n_instances': 5, 'time_limit': 2, 'ite_max': 1000, 'seeds': [71400, 71401, 71402]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
        'evidence_type': 'training_only_confirmation_of_frozen_engineering_and_independent_search_finalists',
        'candidate_membership': {}, 'validation_used_for_selection': False,
        'finalize_rule': 'Minimum paired training mean; evaluate that source once on fixed 5x10 validation. No change based on any validation score.'}
    seen = {}
    for name, original in sources.items():
        digest = sha(original)
        if digest in seen:
            registration['candidate_membership'][name] = seen[digest]
            continue
        seen[digest] = name
        target = root / 'sources' / name / 'solver.py'
        target.parent.mkdir(parents=True)
        target.write_bytes(original.read_bytes())
        registration['sources'][name] = {'path': str(target), 'sha256': sha(target), 'origin': str(original)}
        registration['candidate_membership'][name] = name
    (root / 'registry.json').write_text(json.dumps(registration, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
