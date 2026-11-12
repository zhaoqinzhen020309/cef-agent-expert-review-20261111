"""External per-round monitoring. Never contributes to Agent inputs or selection."""
import csv,json,statistics
from pathlib import Path
from .data import audit_test
from .patch import digest
from .store import canonical
from .runner import Runner
from .tasks import TaskAdapter
from .parallel import completed_tasks
from .task_tools import _atomic

def evaluate_snapshot(source,config,test,output):
    output=Path(output)
    audit_test('tsp',test['instances'])
    identity=dict(source_hash=digest(source),benchmark=test['instances'],seeds=test['evaluation_seeds'],
        repeats=test['test_repeats'],seconds=config['solver_time_limit_s'],workers=config['evaluation_workers'],
        memory_mb=config.get('memory_mb',1024),purpose='external monitoring only; not Agent feedback or training selection')
    output.mkdir(parents=True,exist_ok=True)
    freeze=output/'freeze.json'
    if freeze.exists():
        if json.loads(freeze.read_text(encoding='utf-8'))!=identity:raise ValueError('round benchmark identity changed')
    else:_atomic(freeze,canonical(identity));_atomic(output/'solver.py',source)
    summary=output/'summary.json'
    if summary.exists():
        result=json.loads(summary.read_text(encoding='utf-8'))
        if (digest((output/'solver.py').read_text(encoding='utf-8'))!=identity['source_hash'] or
            result.get('source_hash')!=identity['source_hash'] or
            result.get('count')!=len(test['instances'])*identity['repeats'] or
            result.get('evaluation_metric')!=test['instances'][0].get('evaluation_metric')):
            raise ValueError('cached benchmark result identity changed')
        return result
    adapter=TaskAdapter('tsp');runner=Runner(identity['seconds'],identity['memory_mb'])
    jobs=[(row,repeat) for repeat in range(identity['repeats']) for row in test['instances']]
    records=[];pending=[]
    for row,repeat in jobs:
        key=f'{row["id"]}_{repeat}';path=output/(key+'.json')
        if path.exists():records.append(json.loads(path.read_text(encoding='utf-8')))
        elif (output/(key+'.started')).exists():raise RuntimeError('benchmark case outcome unknown: '+key)
        else:pending.append((row,repeat))
    def work(item):
        row,repeat=item;key=f'{row["id"]}_{repeat}'
        (output/(key+'.started')).write_text('started',encoding='utf-8')
        result=runner.case(source,adapter,adapter.load(row),identity['seeds'][row['id']+':'+str(repeat)])
        result.update(instance_id=row['id'],repeat=repeat,reference=row['reference'])
        result['gap_pct']=100*(result['objective']/row['reference']-1) if result['valid'] else None
        _atomic(output/(key+'.json'),canonical(result))
        return result
    for item,result in completed_tasks(pending,work,identity['workers']):
        records.append(result)
        print(f'Round TSPLIB | {len(records)}/{len(jobs)} | {result["instance_id"]} | valid={result["valid"]}',flush=True)
    records.sort(key=lambda r:(r['repeat'],r['instance_id']))
    valid=len(records)==len(jobs) and all(r['valid'] for r in records)
    result=dict(valid=valid,count=len(records),mean_gap_pct=statistics.fmean(r['gap_pct'] for r in records) if valid else None,
        source_hash=identity['source_hash'],evaluation_metric=test['instances'][0].get('evaluation_metric'),
        scope='external monitoring; not used for search decisions',cases=records)
    _atomic(summary,canonical(result))
    return result

def monitor_round(engine,config,test,folder):
    count=len(engine.archive.of('round_complete'))
    if not count or engine.training_epoch!=count or engine.completed_iterations()!=count:return None
    requests=engine.archive.of('manual_step_requested')
    if requests and requests[-1]['target_round']>count:return None
    best=engine.best();source=engine.archive.read_source(best['source_hash'])
    result=evaluate_snapshot(source,config,test,Path(folder)/'round_benchmarks'/f'{count:03d}')
    gap=result['mean_gap_pct']
    print(f'Round {count} | TSPLIB_gap={gap:.6f}%' if gap is not None else f'Round {count} | TSPLIB invalid; see saved case errors',flush=True)
    return result
