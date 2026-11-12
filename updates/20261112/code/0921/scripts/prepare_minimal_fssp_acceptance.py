"""Isolate acceptance from neighborhoods after observing a 1.46% acceptance rate."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / 'runs/minimal_fssp_acceptance_v1'
    if root.exists():
        raise FileExistsError(root)
    parent = (ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py').read_text()
    adaptive = parent.replace('        initialized = time.time() - start', '''        base_temperature = temperature
        window_accepted = 0
        initialized = time.time() - start''')
    adaptive = adaptive.replace('                current, value = candidate, candidate_value', '''                current, value = candidate, candidate_value
                window_accepted += 1
            if iterations % 64 == 0:
                # Feedback uses accepted steps only, never reference values.
                observed = window_accepted / 64.0
                temperature *= np.exp(0.5 * (0.15 - observed))
                temperature = min(base_temperature * 16.0, max(base_temperature * 0.25, temperature))
                window_accepted = 0''')
    beam = (ROOT / 'runs/minimal_fssp_reconstruction_v1/sources/beam_three/solver.py').read_text()
    protocol = ROOT / 'data/bundles/minimal_v1/fssp/protocol.json'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    diagnostic = ROOT / 'runs/fssp_training_diagnostics_v2/receipt.json'
    spec = {'problem': 'fssp', 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
        'evidence_type': 'human_engineering_acceptance_only_training_ablation',
        'budget': {'n_instances': 3, 'time_limit': 30, 'ite_max': 10000000, 'seeds': [71400, 71401]},
        'selection_rule': 'minimum_training_mean_then_name', 'sources': {},
        'diagnostic_sha256': sha(diagnostic),
        'hypothesis': 'The 50x10 training probe accepts only 1.456% of steps. Compare fixed temperature x4 with bounded feedback targeting 15% acceptance; keep ruin size, local search and restart policy unchanged. Include current beam candidate as paired control.',
        'promotion_rule': 'Only a new acceptance variant winning training is newly validated; parent/beam receipts already exist and are retained without repeat validation.'}
    sources = {'parent': parent, 'warmer_fixed': parent.replace('TEMPERATURE_FACTOR = 0.4', 'TEMPERATURE_FACTOR = 1.6'),
               'acceptance_feedback': adaptive, 'beam_control': beam}
    for name, code in sources.items():
        path = root / 'sources' / name / 'solver.py'
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding='utf-8')
        spec['sources'][name] = {'path': str(path), 'sha256': sha(path)}
    (root / 'registry.json').write_text(json.dumps(spec, indent=2))
    print(root / 'registry.json')


if __name__ == '__main__':
    main()
