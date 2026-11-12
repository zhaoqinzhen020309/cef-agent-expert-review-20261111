"""Preserve failed artifacts and register new copies with the two identified fixes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def repaired(source, node):
    if node == 'node_0001':
        source = source.replace('(src_idx, pos, seg_len, dst_idx, best_p)',
                                '(src_idx, pos, seg_len, dst_idx, best_p, seg_demand)')
        source = source.replace('s_idx, pos, seg_len, d_idx, p = best_move',
                                's_idx, pos, seg_len, d_idx, p, seg_demand = best_move')
        source = source.replace('                    dst_route[insert_p:insert_p] = seg',
            '                    dst_route[insert_p:insert_p] = seg\n                    for customer in seg:\n                        rid[customer] = d_idx')
    elif node == 'node_0002':
        source = source.replace('sum(float(demands[c]) for c in routes[ra][ia:])',
                                'sum(float(demands[c]) for c in routes[ra][ia + 1:])')
        source = source.replace('sum(float(demands[c]) for c in route[ib:])',
                                'sum(float(demands[c]) for c in route[ib + 1:])')
    else:
        raise ValueError(node)
    return source


def main():
    root = ROOT / 'runs/minimal_cvrp_capacity_repairs_v1'
    if root.exists():
        raise FileExistsError(root)
    prior = ROOT / 'runs/minimal_campaign_v3/jobs/cvrp'
    state = json.loads((prior / 'search_state.json').read_text())
    protocol = ROOT / 'data/bundles/minimal_v1/cvrp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = {'problem': 'cvrp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'evidence_type': 'human_repair_of_failed_agent_candidates_training_only',
        'budget': {'n_instances': 5, 'time_limit': 2, 'ite_max': 1000, 'seeds': [71400, 71401]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
        'repairs': {'node_0001': 'selected segment demand must travel with move; update moved customer route ids',
                    'node_0002': 'keep cut customer in both post-tail-exchange capacity calculations'}}
    for node in state['nodes']:
        if node['node_id'] not in spec['repairs']:
            continue
        original = prior / node['solver_relpath']
        name = node['node_id'] + '_repaired'
        path = root / 'sources' / name / 'solver.py'
        path.parent.mkdir(parents=True)
        path.write_text(repaired(original.read_text(), node['node_id']), encoding='utf-8')
        spec['sources'][name] = {'path': str(path), 'sha256': sha(path), 'parent_path': str(original), 'parent_sha256': sha(original)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
