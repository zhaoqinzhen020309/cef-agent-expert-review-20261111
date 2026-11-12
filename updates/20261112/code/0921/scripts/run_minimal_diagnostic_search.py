"""Three bounded fresh searches; never resumes the stopped full campaigns."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_campaign import write, summarize_run, digest
from scripts.snapshot_runtime import snapshot
from scripts.prepare_minimal_protocols import write_once
from scripts.run_research_campaign import balance

CAMPAIGN = ROOT / 'runs/minimal_diagnostic_search_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    if CAMPAIGN.exists():
        raise FileExistsError(CAMPAIGN)
    origins = {
        'fssp': ROOT / 'problems/fssp/minimal_repair_seed_v1/solver.py',
        'cvrp': ROOT / 'runs/minimal_campaign_v3/jobs/cvrp/selected_artifact/selected/solver.py'}
    for kind, source in origins.items():
        seed = ROOT / 'problems' / kind / 'minimal_diagnostic_seed_v1/solver.py'
        seed.parent.mkdir(parents=True, exist_ok=True)
        if seed.exists() and seed.read_bytes() != source.read_bytes():
            raise ValueError('seed changed')
        seed.write_bytes(source.read_bytes())
        protocol = json.loads((ROOT / 'data/bundles/minimal_v1' / kind / 'protocol.json').read_text())
        directory = ROOT / 'data/bundles/minimal_diagnostic_v1' / kind
        protocol.update(protocol_id=kind + '_minimal_diagnostic_v1', data_root=str(directory))
        protocol['task']['baseline_path'] = str(seed)
        protocol['metadata'].update(baseline_source_sha256=sha(source), validation_used_for_selection=False)
        if kind == 'fssp':
            diagnostic = ROOT / 'runs/fssp_training_diagnostics_v2/receipt.json'
            protocol['metadata']['training_diagnostics'] = {'path': str(diagnostic), 'sha256': sha(diagnostic)}
        for split in ('train', 'validation'):
            manifest = json.loads(Path(protocol[split]['source_path']).read_text())
            manifest['contract_id'] = protocol['protocol_id']
            target = directory / (split + '.json')
            write_once(target, manifest)
            protocol[split]['source_path'] = str(target)
        write_once(directory / 'protocol.json', protocol)
    runtime = ROOT / 'runs/runtime_minimal_diagnostic_v1'
    snapshot(runtime)
    jobs = []
    for name, kind in (('fssp_diagnostic', 'fssp'), ('cvrp_repeat_1', 'cvrp'), ('cvrp_repeat_2', 'cvrp')):
        protocol = runtime / 'protocols/minimal_diagnostic_v1' / kind / 'protocol.json'
        jobs.append({'id': name, 'kind': kind, 'protocol': str(protocol), 'protocol_sha256': sha(protocol),
            'instances': 3 if kind == 'fssp' else 5, 'seconds': 30 if kind == 'fssp' else 2,
            'iterations': 10000000 if kind == 'fssp' else 1000, 'validation_seeds': 3 if kind == 'fssp' else 10,
            'candidates': 4, 'model_token_cap': 500000})
    registration = {'schema': 'minimal_diagnostic_search_v1', 'runtime': str(runtime),
        'snapshot_sha256': sha(runtime / 'snapshot.json'), 'runner_sha256': sha(__file__), 'jobs': jobs,
        'model': 'deepseek-flash', 'thinking': 'disabled', 'max_tokens': 16384,
        'train_seeds': [71400, 71401], 'full_validation_scheduled': False,
        'claim_boundary': 'Fresh independent model calls, isolated run memory; common solver seeds. Two CVRP searches are small-budget stability probes, not a matched multi-method study.',
        'finalize_rule': 'Only a new training winner beating its seed by >1e-6 pp is validated; otherwise retain prior seed receipt.'}
    registration['sha256'] = digest(registration)
    write_once(CAMPAIGN / 'registry.json', registration)
    return registration


def load():
    spec = json.loads((CAMPAIGN / 'registry.json').read_text())
    if spec['sha256'] != digest({k: v for k, v in spec.items() if k != 'sha256'}) or sha(__file__) != spec['runner_sha256']:
        raise ValueError('registration/runner changed')
    runtime = Path(spec['runtime'])
    if sha(runtime / 'snapshot.json') != spec['snapshot_sha256']:
        raise ValueError('snapshot changed')
    for name, expected in json.loads((runtime / 'snapshot.json').read_text())['source_sha256'].items():
        if sha(runtime / 'agent_ahd' / name) != expected:
            raise ValueError('runtime source changed')
    for job in spec['jobs']:
        if sha(job['protocol']) != job['protocol_sha256']:
            raise ValueError('protocol changed')
    return spec


def stage(spec, job, phase, extra):
    prefix = CAMPAIGN / 'logs' / (job['id'] + '_' + phase)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'agent_ahd.experiments.run_task_search', phase,
               '--run-dir', str(CAMPAIGN / 'jobs' / job['id'])] + extra
    with prefix.with_suffix('.stdout.log').open('a') as out, prefix.with_suffix('.stderr.log').open('a') as err:
        proc = subprocess.Popen(command, cwd=spec['runtime'], stdout=out, stderr=err,
            env={**os.environ, 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'NUMBA_NUM_THREADS': '1'}, creationflags=subprocess.CREATE_NO_WINDOW)
        write(prefix.with_suffix('.process.json'), {'pid': proc.pid, 'started_at': time.time()})
        code = proc.wait()
    write(prefix.with_suffix('.exit.json'), {'exit_code': code, 'finished_at': time.time()})
    if code:
        raise RuntimeError(f'{phase} failed: exit {code}')


def run_job(spec, job):
    root = CAMPAIGN / 'jobs' / job['id']
    receipt = CAMPAIGN / 'receipts' / (job['id'] + '.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    try:
        if not (root / 'planner_config.json').exists():
            stage(spec, job, 'init', ['--protocol', job['protocol'], '--problem', job['kind'], '--phase', 'benchmark',
                '--rounds', '2', '--batch-size', '2', '--instances', str(job['instances']), '--seconds', str(job['seconds']),
                '--iterations', str(job['iterations']), '--workers', '2', '--seed', '71400', '--solver-seeds', '2',
                '--final-seeds', str(job['validation_seeds']), '--token-budget', str(job['model_token_cap']),
                '--max-tokens', '16384', '--wall-seconds', '14400', '--baseline-parent', '--deterministic-ablation'])
        if summarize_run(root)['status'] != 'COMPLETED_SEARCH':
            stage(spec, job, 'run', ['--env-file', str(ROOT / '.env'), '--model', spec['model'], '--reasoning-effort', 'none'])
        progress = summarize_run(root)
        if progress['status'] != 'COMPLETED_SEARCH':
            raise RuntimeError('proposal budget incomplete')
        if progress['best_train_gap'] < progress['baseline_train_gap'] - 1e-6:
            stage(spec, job, 'finalize', [])
            outcome = 'VALIDATED_TRAINING_WINNER'
        else:
            outcome = 'NO_IMPROVING_CANDIDATE'
        result = {'status': outcome, 'search': summarize_run(root), 'finished_at': time.time()}
    except Exception as exc:
        result = {'status': 'FAILED_EXECUTION', 'error': str(exc), 'finished_at': time.time()}
    write(receipt, result)
    return result


def run(spec):
    from agent_ahd.core.engine import _PlannerRunFileLock
    with _PlannerRunFileLock(CAMPAIGN):
        if (CAMPAIGN / 'exit.json').exists():
            return
        if balance() < 5 + 8 * sum(j['model_token_cap'] for j in spec['jobs']) / 1e6:
            raise RuntimeError('resource admission failed')
        results = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(run_job, spec, j): j for j in spec['jobs']}
            for future in as_completed(futures):
                name = futures[future]['id']
                results[name] = future.result()
                print(json.dumps({'job': name, **results[name]}), flush=True)
        write(CAMPAIGN / 'exit.json', {'status': 'FAILED_JOBS' if any(r['status'] == 'FAILED_EXECUTION' for r in results.values()) else 'RETURNED',
                                      'jobs': results, 'finished_at': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('register', 'run', 'status'))
    args = parser.parse_args()
    if args.command == 'register':
        register()
        print(CAMPAIGN / 'registry.json')
    elif args.command == 'run':
        run(load())
    else:
        spec = load()
        print(json.dumps({j['id']: summarize_run(CAMPAIGN / 'jobs' / j['id']) for j in spec['jobs']}, indent=2))
