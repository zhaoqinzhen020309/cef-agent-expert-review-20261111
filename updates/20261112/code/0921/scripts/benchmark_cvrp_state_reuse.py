"""Fixed-work equivalence and isolated microbenchmarks, not solver gap evidence."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import tracemalloc
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parent = ROOT / 'runs/minimal_cvrp_final_optimization_v1/sources'
    modules = {name: load(name, parent / name / 'solver.py') for name in ('confirmed_agent', 'optimized_agent')}
    manifest = json.loads((ROOT / 'data/bundles/minimal_v1/cvrp/train.json').read_text())
    row = manifest['instances'][-1]
    assert manifest['split'] == 'train'
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
    with np.load(row['path'], allow_pickle=False) as data:
        instance = {k: data[k].copy() for k in ('distance_matrix', 'demands', 'capacity')}
    memories, routes = {}, {}
    for name, module in modules.items():
        tracemalloc.start()
        route = module.SearchStrategy()._savings_once(instance, time.time() + 120, 1000, 17, 1.)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memories[name] = peak
        routes[name] = route
    assert routes['confirmed_agent'] == routes['optimized_agent']
    refresh = modules['optimized_agent'].SearchStrategy._refresh_routes
    route = routes['optimized_agent']
    n = len(instance['demands'])
    state = [np.zeros(n, dtype=int) for _ in range(4)] + [np.zeros(n), np.zeros(len(route))]
    state[-1] = refresh(route, instance['demands'], *state)
    timings = {}
    repetitions = 3000
    for name, affected in (('full', None), ('two_routes', (0, 1))):
        tick = time.perf_counter()
        for _ in range(repetitions):
            state[-1] = refresh(route, instance['demands'], *state, affected=affected)
        timings[name] = (time.perf_counter() - tick) / repetitions
    result = {'evidence_type': 'isolated_fixed_work_engineering_microbenchmark_not_end_to_end_speedup',
        'instance_id': row['id'], 'customers': n - 1, 'savings_output_equal': True,
        'savings_peak_allocated_bytes': memories, 'refresh_seconds_per_call': timings,
        'affected_route_customers': len(route[0]) + len(route[1]),
        'complexity': 'Accepted inter-route state refresh O(n) becomes O(|route_a|+|route_b|), still O(n) worst case; full solver retains quadratic distance and savings arrays.',
        'source_sha256': {name: hashlib.sha256((parent / name / 'solver.py').read_bytes()).hexdigest() for name in modules}}
    out = ROOT / 'analysis_outputs/current_audit/cvrp_state_reuse_benchmark.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
