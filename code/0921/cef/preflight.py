"""A complete parsed code response gate, not a ping/models/short chat test."""
import copy,json,time
from pathlib import Path
from .store import Archive,canonical
from .ledger import BudgetLedger
from .model import Model
from .patch import compose,check_source,digest
from .tasks import TaskAdapter
from .errors import exception_chain
from .config import freeze_research_contract
from .runner import Runner

def identity(c):
    value=dict({k:c.get(k) for k in ('provider','model_snapshot','endpoint','api_key_env','route','wire_api','model_sampling','max_output_tokens','request_timeout_s','environment_manifest')},transport='openai_sdk',stream=c.get('stream',True))
    if 'transport_retry_limit' in c: value['transport_retry_limit']=c['transport_retry_limit']
    if c.get('common_start_probe'):
        value.update(research_contract_version=c.get('research_contract_version'),
                     research_prompt_version=c.get('research_prompt_version'),
                     trace_policy=copy.deepcopy(c.get('trace_policy')),
                     research_task=c.get('task'),research_runtime_limit_s=c.get('probe_seconds'))
    return value

def receipt_matches(c,receipt):
    """Solver process count does not change the sequential model request route."""
    if receipt.get('passed') is not True: return False
    if c.get('common_start_probe'):
        checks=receipt.get('runtime_checks')
        if receipt.get('research_interface_verified') is not True or not isinstance(checks,list): return False
        if [row.get('kind') for row in checks]!=['normal','trace','common_start']: return False
        if any(not isinstance(row.get('result'),dict) or row['result'].get('valid') is not True for row in checks): return False
    expected=copy.deepcopy(identity(c)); actual=copy.deepcopy(receipt.get('identity'))
    for item in (expected,actual):
        if not isinstance(item,dict): return False
        resources=(item.get('environment_manifest') or {}).get('worker_resources',{})
        resources.pop('concurrency',None)
    return actual==expected

def preflight(c):
    # Check the opt-in interface before constructing Model or creating evidence.
    if c.get('common_start_probe') or 'common_start_probe' in c: c=freeze_research_contract(c)
    if c.get('common_start_probe'):
        rows=c.get('data_manifest',{}).get('train',[])
        if not isinstance(rows,list) or not rows or rows[0].get('split')!='train':
            raise ValueError('research preflight requires a frozen training instance')
    root=Path(c['preflight_dir'])
    if root.exists(): raise FileExistsError('preflight evidence directory already exists')
    a=Archive(root)
    attempts=1+c.get('transport_retry_limit',0)
    caps=dict(max_llm_requests=attempts,max_llm_tokens=c['max_llm_tokens'],max_case_runs=3 if c.get('common_start_probe') else 1,max_analysis_seconds=1,max_wallclock_s=c['request_timeout_s']*attempts+80)
    ledger=BudgetLedger(a,caps,enforce_limits=c.get('stop_rule')!='iterations')
    # This is one complete code-response check, not an algorithm-search episode.
    # Keep its one-edit payload consistent even when the later search is joint-only.
    model=Model({k:v for k,v in c.items() if k!='evaluation_policy'},a,ledger)
    adapter=TaskAdapter(c['task']); source=adapter.seed_source()
    instruction='Produce one complete alternative solver, including all helper definitions, as a single whole-source replacement. This is a transport/parser check; no performance is claimed.'
    interface=adapter.description
    if c.get('common_start_probe'):
        research=' Implement solve(instance, time_limit_s, seed, *, initial_solution=None, observer=None). For a supplied feasible initial_solution, copy it into the actual search state, skip construction, and use the same subsequent search implementation as ordinary solving. Call observer(solution) on the initial feasible state and periodically in the search loop; observer returns no search decision. Keep a best-so-far solution. Both optional keyword-only parameters must default to None, and ordinary three-argument calls must work.'
        instruction+=research
        interface+=research
    payload=dict(parent_id='preflight_seed',parent_hash=digest(source),parent_code=source,task_interface=interface,edit_count=1,decision=dict(proposed_intents=[instruction]),contrasts=[],experiences=[])
    report=dict(identity=identity(c),passed=False,concurrency=1,retry_limit=c.get('transport_retry_limit',0))
    try:
        package=model.call('editor',payload)
        variants=compose(source,package); emitted=variants['PA']
        a.source(emitted)
        report.update(source_hash=digest(emitted),response_bytes=len(canonical(package).encode()),source_bytes=len(emitted.encode()))
        check_source(emitted,require_research=True) if c.get('common_start_probe') else check_source(emitted)
        if c.get('common_start_probe'):
            report['runtime_checks']=[]
            row=rows[0]
            instance=adapter.load(row); instance_id=row['id']
            fixture_scope='first frozen training instance; interface check only'
            key='distance_matrix' if c['task']=='tsp' else 'processing_times'
            initial=list(reversed(range(len(instance[key]))))
            seed=next(iter(c['evaluation_seed_manifest'].values()))
            limit=c['probe_seconds']; runner=Runner(limit,c.get('memory_mb',1024))
            report['runtime_contract']=dict(description=fixture_scope,instance_id=instance_id,
                instance_hash=digest(canonical(adapter.solver_input(instance))),seed=seed,local_limit_s=limit,
                initial_solution=initial,initial_objective=adapter.objective(instance,initial),trace_policy=c['trace_policy'])
            for kind in ('normal','trace','common_start'):
                request=None if kind=='normal' else dict(kind=kind,local_limit_s=limit,trace_policy=c['trace_policy'])
                if kind=='common_start': request['initial_solution']=initial
                def started(kind=kind):
                    a.add('preflight_execution_start',dict(execution_id='preflight:'+kind,purpose='preflight_interface',kind=kind,source_hash=report['source_hash']))
                    ledger.charge('case_runs',1,purpose='preflight_interface',execution_id='preflight:'+kind)
                result=runner.case(emitted,adapter,instance,seed,on_started=started,research_request=request)
                evidence=dict(kind=kind,result=result,execution_id='preflight:'+kind,purpose='preflight_interface')
                report['runtime_checks'].append(evidence); a.add('execution',evidence)
                if not result['valid']: raise ValueError('preflight '+kind+' execution failed: '+str(result.get('error')))
            report['research_interface_verified']=True
        report['passed']=True
    except Exception as e:
        report['error']=type(e).__name__+': '+str(e)
        report['error_chain']=exception_chain(e,getattr(model,'key',None))
    report['cost']=[r for r in a.of('charge')]; report['time']=time.time()
    (root/'receipt.json').write_text(canonical(report),encoding='utf-8')
    return report
