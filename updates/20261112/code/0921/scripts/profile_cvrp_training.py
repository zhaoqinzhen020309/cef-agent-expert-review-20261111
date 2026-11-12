"""Separate cProfile training probe, never a scored performance measurement."""
from pathlib import Path
import cProfile
import hashlib
import importlib.util
import json
import pstats
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/selected_artifact/selected/solver.py'
    out = ROOT / 'runs/cvrp_training_diagnostics_v1'
    out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('cvrp_probe', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    manifest_path = ROOT / 'data/bundles/minimal_v1/cvrp/train.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['split'] == 'train'
    rows = []
    for row in manifest['instances']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
        with np.load(row['path'], allow_pickle=False) as raw:
            data = {k: raw[k].copy() for k in ('distance_matrix', 'demands', 'capacity')}
        profile = cProfile.Profile()
        tick = time.time()
        routes = profile.runcall(mod.SearchStrategy().solve, data, tick + 2, 1000, 71400)
        stats = pstats.Stats(profile).stats
        functions = sorted([{'function': key[2], 'calls': value[1], 'self_seconds': value[2],
                              'cumulative_seconds': value[3]} for key, value in stats.items()
                             if key[0] == str(source)], key=lambda x: -x['cumulative_seconds'])
        record = {'instance_id': row['id'], 'seed': 71400, 'input_sha256': row['sha256'],
            'route_load_max': max(sum(data['demands'][c] for c in r) for r in routes),
            'capacity': float(data['capacity']), 'functions': functions}
        rows.append(record)
        print(json.dumps(record), flush=True)
    (out / 'receipt.json').write_text(json.dumps({'split': 'train', 'rows': rows,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'measurement': 'cProfile includes substantial overhead; never substitute for scored timing'}, indent=2))


if __name__ == '__main__':
    main()
