"""Host-owned instrumentation of the registered IG kernels on TRAIN inputs only.

Numba objmode timers belong to this diagnostic harness, not candidate code or
the scored evaluator. Thus these timings are diagnostic, not speed benchmarks.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def instrument(source):
    tree = ast.parse(source)
    kernels = {'local_search', 'initialize', 'step'}
    class Counters(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in kernels:
                node.args.append(ast.Name(id='_probe_metrics', ctx=ast.Load()))
            return node

        def visit_FunctionDef(self, node):
            if node.name not in kernels:
                return node
            node.args.args.append(ast.arg(arg='_probe_metrics'))
            self.generic_visit(node)
            if node.name != 'local_search':
                return node
            entry = ast.parse('''
with objmode(_probe_start='float64'):
    _probe_start = time.perf_counter()
_probe_phase = 1 if count == len(p) else 0
_probe_metrics[_probe_phase, 1] += 1
''').body
            class Moves(ast.NodeTransformer):
                def visit_Expr(self, expr):
                    call = expr.value
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'insertion_costs':
                        return ast.parse('_probe_metrics[_probe_phase, 2] += 1\n_probe_metrics[_probe_phase, 5] += count').body + [expr]
                    return expr

                def visit_If(self, item):
                    self.generic_visit(item)
                    if isinstance(item.test, ast.Name) and item.test.id == 'accept':
                        item.body = ast.parse('''
if candidate < value:
    _probe_metrics[_probe_phase, 3] += 1
elif candidate == value and pos != old:
    _probe_metrics[_probe_phase, 4] += 1
''').body + item.body
                    return item
            node = Moves().visit(node)
            if not isinstance(node.body[-1], ast.Return):
                raise ValueError('unsupported local_search return structure')
            finish = ast.parse('''
with objmode(_probe_end='float64'):
    _probe_end = time.perf_counter()
_probe_metrics[_probe_phase, 0] += _probe_end - _probe_start
''').body
            node.body = entry + node.body[:-1] + finish + node.body[-1:]
            return node
    # Only recognized pure kernels; the solver's self-reported statistics are unused.
    tree.body = [n for n in tree.body if not isinstance(n, ast.ClassDef)]
    tree = Counters().visit(tree)
    tree.body.insert(0, ast.ImportFrom(module='numba', names=[ast.alias(name='objmode')], level=0))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def measure(module, p, seed, seconds):
    n, m = p.shape
    forward, backward = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1))
    costs, metrics = np.empty(n + 1), np.zeros((2, 6))
    start = time.perf_counter()
    current, value = module['initialize'](p, seed, forward, backward, costs, True, metrics)
    init_seconds = time.perf_counter() - start
    initial_metrics = metrics.copy()
    metrics[:] = 0
    best, best_value = current.copy(), value
    rng = np.random.default_rng(seed)
    temperature = max(1e-9, .4 * p.mean() / 10)
    iterations = improvements = accepted = stagnation = max_stagnation = 0
    completed_stagnations = []
    traces = []
    loop_start = time.perf_counter()
    first_step_seconds = None
    while time.perf_counter() - loop_start < seconds:
        tick = time.perf_counter()
        candidate, cv = module['step'](current, p, 4, forward, backward, costs, True, True, metrics)
        if first_step_seconds is None:
            first_step_seconds = time.perf_counter() - tick
        iterations += 1
        stagnation += 1
        if cv < best_value:
            best, best_value = candidate.copy(), cv
            improvements += 1
            completed_stagnations.append(stagnation)
            max_stagnation = max(max_stagnation, stagnation)
            stagnation = 0
            traces.append({'iteration': iterations, 'seconds': time.perf_counter() - loop_start, 'objective': cv})
        delta = cv - value
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            current, value = candidate, cv
            accepted += 1
    actual = time.perf_counter() - loop_start
    # Independent host objective; this file never loads benchmark references.
    finish = np.zeros(m)
    for job in best:
        for k in range(m):
            finish[k] = max(finish[k], finish[k - 1] if k else 0) + p[job, k]
    if not np.isclose(finish[-1], best_value) or sorted(best.tolist()) != list(range(n)):
        raise ValueError('diagnostic kernel returned inconsistent result')
    neighborhoods = {}
    for i, name in enumerate(('partial_insertion', 'full_insertion')):
        elapsed, calls, attempts, strict, neutral, positions = metrics[i]
        neighborhoods[name] = {'seconds_with_timer_overhead': elapsed, 'calls': int(calls),
            'job_attempts': int(attempts), 'strict_improving_moves': int(strict),
            'neutral_moves': int(neutral), 'positions_scored': int(positions),
            'strict_move_rate': strict / attempts if attempts else None}
    return {'seed': seed, 'shape': [n, m], 'initialization_seconds_including_jit': init_seconds,
        'initialization_counters': initial_metrics.tolist(), 'loop_seconds_including_first_step_jit': actual,
        'first_step_seconds_including_jit': first_step_seconds, 'iterations': iterations,
        'incumbent_improvements': improvements, 'accepted_steps': accepted, 'best_objective': best_value,
        'longest_stagnation_iterations': max(max_stagnation, stagnation),
        'right_censored_final_stagnation_iterations': stagnation,
        'completed_stagnations': completed_stagnations, 'improvement_trace': traces,
        'neighborhoods': neighborhoods}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/fssp_training_diagnostics_v1')
    parser.add_argument('--seconds', type=float, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    generated = args.output / 'instrumented_kernels.py'
    generated.write_text(instrument(args.source.read_text()), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('fssp_host_probe', generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    namespace = vars(module)
    manifest_path = ROOT / 'data/bundles/minimal_v1/fssp/train.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['split'] != 'train':
        raise ValueError('diagnostics must be training-only')
    records = []
    for row in manifest['instances']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('input changed')
        with np.load(row['path'], allow_pickle=False) as data:
            p = np.ascontiguousarray(data['processing_times'], dtype=np.float64)
        record = {'instance_id': row['id'], 'input_sha256': row['sha256'],
                  **measure(namespace, p, 71400, args.seconds)}
        records.append(record)
        (args.output / (row['id'] + '.json')).write_text(json.dumps(record, indent=2))
        print(json.dumps({'instance': row['id'], 'iterations': record['iterations'],
                          'neighborhoods': record['neighborhoods']}), flush=True)
    receipt = {'schema': 'host_training_diagnostics_v1', 'split': 'train',
        'source_sha256': sha(args.source), 'instrumentation_sha256': sha(__file__),
        'instrumented_source_sha256': sha(generated), 'manifest_sha256': sha(manifest_path),
        'instance_ids': [r['id'] for r in manifest['instances']], 'seeds': [71400],
        'measurement': 'host-inserted counters and objmode timers; includes timer overhead; separate diagnostic budget; no validation or reference access; not scored performance',
        'rows': records}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
