"""Read-only audit of actual saved CEF decisions, sources and executions."""
import argparse,json,sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cef.store import Archive,canonical
from cef.engine import Engine
from cef.patch import compose,digest
from cef.tasks import TaskAdapter
from cef.feedback import contrast
from cef.config import load_json,file_hash

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    a=Archive(args.run_dir); before=a.path.read_bytes(); c=a.of('manifest')[0]; adapter=TaskAdapter(c['task'])
    rows={r['id']:r for r in c['data_manifest']['train']}; candidates={r['candidate_id']:r for r in a.of('candidate')}; executions={r['execution_id']:r for r in a.of('execution')}
    checks=[]
    for cand in candidates.values(): a.read_source(cand['source_hash'])
    for r in executions.values():
        if r['valid']:
            objective,fitness=adapter.score(adapter.load(rows[r['instance_id']]),r['solution'])
            if abs(objective-r['objective'])>1e-9 or abs(fitness-r['fitness'])>1e-9: raise AssertionError('saved score mismatch')
    for m in a.of('measurement'):
        actual=[executions[eid] for eid in m['execution_ids']]
        if len(actual)!=64 or len({r['instance_id'] for r in actual})!=64: raise AssertionError('incomplete formal measurement')
    for event in a.of('contrast'):
        if contrast(event['scores'])!=event['contrasts']: raise AssertionError('contrast arithmetic mismatch')
        saved=next(x for x in a.of('package') if x['package']['package_id']==event['package_id'])
        package=saved['package']; parent=a.read_source(event['parent_hash']); variants=compose(parent,package)
        for label,text in variants.items():
            if digest(text)!=candidates[event['variant_ids'][label]]['source_hash']: raise AssertionError('variant hash mismatch')
        checks.append(dict(event_id=event['event_id'],variants=list(variants),scores=event['scores'],contrasts=event['contrasts'],selected=event['selected_candidate']))
    events=a.of('contrast'); starts=a.of('model_start'); responses=a.of('model_response')
    first=events[0]['event_id'] if events else None
    planners=[r for r in starts if r['role']=='planner' and first and first in [e['event_id'] for e in r['payload'].get('recent_events',[])]]
    editors=[r for r in starts if r['role']=='editor' and first and first in [e['event_id'] for e in r['payload'].get('contrasts',[])]]
    second=[r for r in responses if r['role']=='editor' and any(s['call_id']==r['call_id'] for s in editors)]
    if not (len(events)>=2 and planners and editors and second and a.of('card_revision')): raise AssertionError('real two-package feedback/revision chain incomplete')
    revision_ids={r['card_id'] for r in a.of('card_revision')}
    for r in second:
        if not r['response']['edits']: raise AssertionError('second actual edit missing')
    e=Engine(c,None); result=load_json(Path(args.run_dir)/'result.json')
    if e.best()['source_hash']!=result['best']['source_hash']: raise AssertionError('resume best differs')
    if e.ledger.used('case_runs')!=len(a.of('execution_start')): raise AssertionError('case cost mismatch')
    for source in c['implementation_source_manifest']:
        relative=Path(source['path']).relative_to(ROOT)
        snapshot=Path(args.run_dir)/'runtime_snapshot'/relative
        if file_hash(snapshot)!=source['sha256']: raise AssertionError('frozen runtime implementation corrupted')
    if a.path.read_bytes()!=before: raise AssertionError('audit changed search memory')
    report=dict(level='C',passed=True,scope='bounded real model implementation acceptance; NOT algorithm performance',model=c['model_snapshot'],model_revision=c.get('model_revision'),provider=c['provider'],training_instances=64,complete_measurements=len(a.of('measurement')),events=checks,actual_model_requests=e.ledger.used('llm_requests'),actual_case_runs=e.ledger.used('case_runs'),saved_executions=len(executions),all_scores_independently_recomputed=True,first_event_reaches_later_planner=[r['call_id'] for r in planners],first_event_reaches_later_editor=[r['call_id'] for r in editors],actual_second_generation=[r['call_id'] for r in second],revised_cards=sorted(revision_ids),resume_best_matches=True,search_unchanged=True,costs=result['resource_usage'],test_executions=0)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
