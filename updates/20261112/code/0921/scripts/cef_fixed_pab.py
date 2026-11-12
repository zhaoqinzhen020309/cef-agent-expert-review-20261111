"""Repeat the exact historical PAB solver on TSPLIB29; no algorithm search."""
import argparse
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

from cef.config import file_hash
from cef.data import audit_test,TSPLIB_METRIC,TSPLIB_DISTANCE_RULE,TSPLIB_REFERENCE_RULE
from cef.parallel import completed_tasks
from cef.patch import digest
from cef.runner import Runner
from cef.tasks import TaskAdapter

ROOT=Path(__file__).resolve().parents[1]
PAB_SOURCE=ROOT/'algorithms/tsp_pab.py'
PAB_SHA256='5dc825d1fa50a6c0250f967875dff54eb875ae1360f1c6353ae17079a35c62c7'
HISTORICAL=ROOT/'experiments/cef_tsp_effect_20261027_004940'
SOLVER_SECONDS=60
MEMORY_MB=1024


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def load_fixed():
    if file_hash(PAB_SOURCE)!=PAB_SHA256: raise ValueError('fixed PAB source hash mismatch')
    source=PAB_SOURCE.read_bytes().decode('utf-8')
    freeze=json.loads((HISTORICAL/'best_test/freeze.json').read_text(encoding='utf-8'))
    if freeze['source_hash']!=PAB_SHA256 or freeze['search_best']['variant']!='PAB':
        raise ValueError('historical PAB provenance mismatch')
    current=json.loads((ROOT/'configs/cef_v1_3/tsp.template.json').read_text(encoding='utf-8'))['data_manifest']['test']
    if {r['id']:r['reference'] for r in current}!={r['id']:r['reference'] for r in freeze['benchmark']}:
        raise ValueError('original frozen TSPLIB references differ from the current metric')
    return source,current


def aggregate(records):
    valid=bool(records) and all(row['valid'] for row in records)
    return dict(valid=valid,count=len(records),valid_count=sum(row['valid'] for row in records),
                mean_gap_pct=statistics.fmean(row['gap_pct'] for row in records) if valid else None)


def main(settings,argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='检查固定算法与测试数据，不运行求解器')
    args=parser.parse_args(argv)
    if set(settings)!={'seeds','workers','output_dir'}: raise ValueError('only seeds, workers and output_dir are used for fixed PAB evaluation')
    seeds=settings['seeds']; workers=settings['workers']
    if not isinstance(seeds,list) or not seeds or any(type(s) is not int or not 0<=s<2**32 for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('seeds must be a nonempty list of distinct integers in 0..2**32-1')
    if type(workers) is not int or workers<1: raise ValueError('workers must be a positive integer')
    source,rows=load_fixed()
    audit_test('tsp',rows)
    # Check again at the entry boundary: no changed source/data may start a run.
    if digest(source)!=PAB_SHA256: raise ValueError('fixed PAB source hash mismatch')
    print(f'PAB | TSPLIB29 | seeds={seeds} | workers={workers} | limit={SOLVER_SECONDS}s',flush=True)
    if args.check:
        print('Configuration OK | fixed_solver=verified | instances=29',flush=True)
        return 0
    folder=Path(settings['output_dir']).resolve()/('pab_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'solver.py').write_bytes(source.encode('utf-8'))
    runtime_paths=[ROOT/'runCEF.py',Path(__file__).resolve()]+[ROOT/'cef'/name for name in ('runner.py','worker.py','tasks.py','patch.py','parallel.py','data.py')]+[ROOT/'core/windows_job.py',ROOT/'agent_ahd/__init__.py']
    manifest=dict(experiment='fixed_historical_PAB_new_metric_evaluation',source_hash=PAB_SHA256,source_origin=str(PAB_SOURCE),
        historical_freeze=str(HISTORICAL/'best_test/freeze.json'),historical_gap_pct=0.019939775685579607,
        historical_gap_note='historical integer-matrix result; not directly comparable to this unrounded evaluation',
        evaluation_metric=TSPLIB_METRIC,distance_rule=TSPLIB_DISTANCE_RULE,reference_rule=TSPLIB_REFERENCE_RULE,
        benchmark=rows,seeds=seeds,evaluation_workers=workers,solver_time_limit_s=SOLVER_SECONDS,
        memory_mb_per_worker=MEMORY_MB,library_threads_per_worker=1,model_requests=0,
        environment=dict(python=sys.version,numpy=version('numpy'),platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count()),
        evaluator_sources=[dict(path=str(path),sha256=file_hash(path)) for path in runtime_paths])
    write_json(folder/'manifest.json',manifest)
    print(f'Run: {folder}',flush=True)
    adapter=TaskAdapter('tsp'); runner=Runner(SOLVER_SECONDS,MEMORY_MB)
    records=[]; started=time.monotonic()
    def jobs():
        for seed in seeds:
            for row in rows: yield seed,row,adapter.load(row)
    def work(item):
        seed,row,instance=item
        return runner.case(source,adapter,instance,seed)
    try:
        for item,result in completed_tasks(jobs(),work,workers):
            seed,row,instance=item
            result.update(seed=seed,instance_id=row['id'],reference=row['reference'],source_hash=PAB_SHA256,
                          evaluation_metric=TSPLIB_METRIC,
                          gap_pct=100*(result['objective']/row['reference']-1) if result['valid'] else None)
            result['unrounded_reference_gap_pct']=result['gap_pct']
            with (folder/'results.jsonl').open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            records.append(result)
            gap=f"{result['gap_pct']:.6f}%" if result['valid'] else 'FAILED'
            print(f"{len(records)}/{len(seeds)*len(rows)} | seed={seed} | {row['id']} | gap={gap}",flush=True)
        # Preserve manifest order for numerical aggregation, independent of completion order.
        rank={row['id']:i for i,row in enumerate(rows)}
        records.sort(key=lambda row:(seeds.index(row['seed']),rank[row['instance_id']]))
        per_seed=[dict(seed=seed,**aggregate([r for r in records if r['seed']==seed])) for seed in seeds]
        summary=dict(**aggregate(records),source_hash=PAB_SHA256,model_requests=0,per_seed=per_seed,
                     evaluation_metric=TSPLIB_METRIC,distance_rule=TSPLIB_DISTANCE_RULE,reference_rule=TSPLIB_REFERENCE_RULE,
                     evaluation_workers=workers,solver_time_limit_s=SOLVER_SECONDS,wallclock_s=time.monotonic()-started)
        summary['mean_unrounded_reference_gap_pct']=summary['mean_gap_pct']
        summary['seed_mean_gap_stdev_pct']=statistics.stdev(r['mean_gap_pct'] for r in per_seed) if summary['valid'] and len(seeds)>1 else None
        write_json(folder/'summary.json',summary)
        for result in per_seed:
            gap=f"{result['mean_gap_pct']:.6f}%" if result['valid'] else 'unavailable'
            print(f"Seed {result['seed']} | valid={result['valid_count']}/29 | mean_gap={gap}",flush=True)
        print(f'Results: {folder / "summary.json"}',flush=True)
        return 0 if summary['valid'] else 1
    except BaseException as error:
        write_json(folder/'interrupted.json',dict(status='interrupted',error=type(error).__name__+': '+str(error),completed_cases=len(records),required_cases=len(seeds)*len(rows),model_requests=0))
        raise


def run(settings):
    try: return main(settings)
    except (ValueError,RuntimeError,OSError) as error:
        print(f'Error: {error}',file=sys.stderr,flush=True)
        return 1
