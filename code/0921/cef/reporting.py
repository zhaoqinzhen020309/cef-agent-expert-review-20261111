"""Frozen-output tests and external result ingress never update search memory."""
import csv,json,statistics
from pathlib import Path
from .store import Archive,canonical
from .patch import digest
from .tasks import TaskAdapter
from .runner import Runner
from .data import (audit_test,TSPLIB_METRIC,TSPLIB_DISTANCE_RULE,TSPLIB_REFERENCE_RULE,
                   UNROUNDED_TSPLIB_METRIC,UNROUNDED_DISTANCE_RULE,UNROUNDED_REFERENCE_RULE)
from .parallel import completed_tasks

def verify_export(run_dir):
    root=Path(run_dir); a=Archive(root); result=json.loads((root/'result.json').read_text(encoding='utf-8'))
    b=result['best']
    with (root/'best_solver.py').open(encoding='utf-8',newline='') as f: exported=f.read()
    if not b or digest(exported)!=b['source_hash']: raise ValueError('frozen output hash mismatch')
    source=a.read_source(b['source_hash'])
    from .engine import Engine
    config=a.of('manifest')[0]
    if config.get('alternating_agents'):
        from .alternating import AlternatingEngine
        actual=AlternatingEngine(config,None).best()
    else:actual=Engine(config,None).best()
    if not actual or (actual['candidate_id'],actual['source_hash'],actual['fitness'])!=(b['candidate_id'],b['source_hash'],b['fitness']): raise ValueError('export is not archive-ranked best candidate')
    return a,result,source

def final_test(run_dir,output,rows,repeats,seed_map):
    a,result,source=verify_export(run_dir); c=a.of('manifest')[0]
    if result['status'] not in ('finished','budget_terminated'): raise ValueError('freeze a successfully terminated search before final testing')
    if repeats!=c['test_repeats'] or type(repeats) is not int or repeats<=0: raise ValueError('test repeats mismatch')
    if canonical(rows)!=canonical(c['data_manifest']['test']): raise ValueError('test manifest differs from frozen search configuration')
    if any(f'{r["id"]}:{i}' not in seed_map for r in rows for i in range(repeats)): raise ValueError('complete test seed map required before launch')
    audit_test(c['task'],rows)
    metric={}
    if c['task']=='tsp':
        identifiers={r.get('evaluation_metric') for r in rows}
        if identifiers=={TSPLIB_METRIC}:
            metric=dict(evaluation_metric=TSPLIB_METRIC,distance_rule=TSPLIB_DISTANCE_RULE,reference_rule=TSPLIB_REFERENCE_RULE)
        elif identifiers=={UNROUNDED_TSPLIB_METRIC}:
            metric=dict(evaluation_metric=UNROUNDED_TSPLIB_METRIC,distance_rule=UNROUNDED_DISTANCE_RULE,reference_rule=UNROUNDED_REFERENCE_RULE)
        elif any(identifiers):raise ValueError('mixed TSPLIB evaluation metrics')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    limit=c['solver_time_limit_s'] if c.get('diagnostic_run') else 60
    workers=c.get('evaluation_workers',1)
    adapter=TaskAdapter(c['task']); runner=Runner(limit,c.get('memory_mb',1024))
    freeze=dict(search_best=result['best'],source_hash=digest(source),benchmark=rows,test_repeats=repeats,seeds=seed_map,search_journal_hash=digest((a.path).read_text(encoding='utf-8')),diagnostic_run=c.get('diagnostic_run',False),solver_time_limit_s=limit,evaluation_workers=workers)
    freeze.update(metric)
    (output/'freeze.json').write_text(canonical(freeze),encoding='utf-8')
    scores=[]
    def jobs():
        for repeat in range(repeats):
            for row in rows:
                yield repeat,row,adapter.load(row),seed_map[f'{row["id"]}:{repeat}']
    def work(item):
        repeat,row,instance,seed=item
        return runner.case(source,adapter,instance,seed)
    total=len(rows)*repeats
    for item,r in completed_tasks(jobs(),work,workers):
        repeat,row,instance,seed=item
        r.update(instance_id=row['id'],repeat=repeat,source_hash=digest(source),reference=row['reference'])
        r['gap_pct']=100*(r['objective']/row['reference']-1) if r['valid'] else None
        r.update(metric)
        if metric:r['original_distance_gap_pct' if metric['evaluation_metric']==TSPLIB_METRIC else 'unrounded_reference_gap_pct']=r['gap_pct']
        with (output/'results.jsonl').open('a',encoding='utf-8',newline='\n') as f: f.write(canonical(r)+'\n')
        scores.append(r)
        if c.get('console_progress') and (len(scores)%8==0 or len(scores)==total):
            print(f'Test | {len(scores)}/{total} | workers={workers}',flush=True)
    # Completion order is nondeterministic; aggregate in frozen benchmark order.
    rank={row['id']:i for i,row in enumerate(rows)}
    scores.sort(key=lambda r:(r['repeat'],rank[r['instance_id']]))
    summary=dict(valid=all(r['valid'] for r in scores),count=len(scores),mean_gap_pct=statistics.fmean(r['gap_pct'] for r in scores) if all(r['valid'] for r in scores) else None,source_hash=digest(source),search_unchanged=digest(a.path.read_text(encoding='utf-8'))==freeze['search_journal_hash'])
    summary.update(diagnostic_run=c.get('diagnostic_run',False),solver_time_limit_s=limit,evaluation_workers=workers)
    summary.update(metric)
    if metric:summary['mean_original_distance_gap_pct' if metric['evaluation_metric']==TSPLIB_METRIC else 'mean_unrounded_reference_gap_pct']=summary['mean_gap_pct']
    (output/'summary.json').write_text(canonical(summary),encoding='utf-8')
    return summary

def import_external(manifest_path,results_path,output):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    required=('method','source_version','algorithm_boundary','model_snapshot','design_budget','environment_manifest','solver_limit_s','internal_limits','data_hash','reference_hash','search_seed')
    if any(k not in manifest for k in required): raise ValueError('external baseline manifest incomplete')
    records=json.loads(Path(results_path).read_text(encoding='utf-8-sig'))
    if not isinstance(records,list) or not records: raise ValueError('raw external records required')
    for row in records:
        if any(k not in row for k in ('instance_id','objective','reference','valid','elapsed','repeat')): raise ValueError('external row incomplete')
        if row['valid'] and (row['reference']<=0 or row['objective']<0): raise ValueError('invalid external objective')
    out=dict(manifest=manifest,records=records,manifest_hash=digest(canonical(manifest)),results_hash=digest(canonical(records)),valid=all(r['valid'] for r in records),mean_gap_pct=statistics.fmean(100*(r['objective']/r['reference']-1) for r in records) if all(r['valid'] for r in records) else None)
    with Path(output).open('x',encoding='utf-8') as f: f.write(canonical(out))
    return out
