"""Training-only common-start experiments. Formal ranking remains in Engine."""
import json
from pathlib import Path

from .feedback import changed_definitions
from .parallel import completed_tasks
from .patch import digest,research_capable
from .replay import RESEARCH_VERSION,TRACE_POLICY,agent_report,cache_key,select_states,summarize
from .store import canonical


class ResearchTools:
    def __init__(self,engine):
        self.engine=engine
        self.rows=[]
        for row in engine.rows:
            if row.get('split')!='train': raise ValueError('research requires training split')
            instance=engine.adapter.load(row)
            self.rows.append(dict(row,instance_hash=digest(canonical(engine.adapter.solver_input(instance)))))
        # Canonical order is stable across JSON round-trips and resume.
        self.seed=int(digest(canonical(engine.config['evaluation_seed_manifest']))[:8],16)
        if engine.config.get('state_selection_seed',self.seed)!=self.seed:
            raise ValueError('frozen state selection seed mismatch')

    def load_instance(self,row):
        instance=self.engine.adapter.load(row)
        if digest(canonical(self.engine.adapter.solver_input(instance)))!=row['instance_hash']:
            raise ValueError('training instance hash changed before diagnostic execution')
        return instance

    def capture(self,cid,row,record):
        if not record.get('valid'): return
        e=self.engine; candidate=e.candidates()[cid]
        bound=next(r for r in self.rows if r['id']==row['id'])
        instance=e.adapter.load(row); best=None; last_improved=0.
        for index,point in enumerate(record.get('trace',[])):
            objective=e.adapter.objective(instance,point['solution'])
            elapsed=point['elapsed_s']
            if best is None or objective<best-1e-12:
                best=objective; last_improved=elapsed
            stalled=elapsed-last_improved
            phase='early' if index==0 else 'late'
            if phase=='late':
                if stalled<max(1e-6,record['elapsed']*0.1): continue
            payload=dict(schema_version=RESEARCH_VERSION,source_candidate_id=cid,
                source_hash=candidate['source_hash'],instance_id=row['id'],split='train',
                instance_hash=bound['instance_hash'],source_seed=record['seed'],
                elapsed_s=elapsed,solution=list(point['solution']),
                solution_hash=digest(canonical(point['solution'])),objective=objective,
                phase=phase,stagnation_s=stalled,trace_policy_hash=digest(canonical(TRACE_POLICY)))
            if e.config.get('alternating_agents'):payload['training_epoch']=e.config['training_epoch']
            snapshot_id='snapshot:'+digest(canonical(payload))
            if any(r['snapshot_id']==snapshot_id for r in e.archive.of('snapshot')): continue
            value=dict(payload,snapshot_id=snapshot_id)
            artifact=e.archive.artifact('snapshots',value)
            e.archive.add('snapshot',dict(value,artifact=artifact))

    def states(self,reference_id):
        e=self.engine; ref=e.candidates()[reference_id]
        snapshots=e.archive.of('snapshot')
        if e.config.get('alternating_agents'):
            snapshots=[s for s in snapshots if s.get('training_epoch')==e.config['training_epoch']]
        return select_states(snapshots,self.rows,ref['source_hash'],
                             self.seed,e.config.get('probe_states',8))

    def search_closed(self):
        e=self.engine
        if e.config.get('manual_rounds'):return bool(e.archive.of('manual_finished'))
        limit=e.config.get('max_iterations')
        return limit is not None and e.completed_iterations()>=limit

    def pairs(self):
        e=self.engine
        completed={r['step'] for r in e.archive.of('action_complete')}
        pending=[r for r in e.archive.of('probe_start') if r['step'] not in completed]
        if pending:
            r=pending[-1]
            return [dict(reference_candidate_id=r['reference_candidate_id'],candidate_id=r['candidate_id'],evidence_id=None)]
        if self.search_closed(): return []
        if any(r['iteration']==e.completed_iterations() for r in e.archive.of('probe_start')): return []
        events=e.archive.of('contrast')
        if e.config.get('alternating_agents'):
            events=[r for r in events if r.get('tested_conditions',{}).get('config_hash')==e.eval_hash]
        if not events: return []
        latest=events[-1]; child=latest['variant_ids'].get('PAB'); parent=latest['parent_id']
        candidates=e.candidates()
        if child==parent or not all(candidates.get(cid,{}).get('eligibility') for cid in (parent,child)): return []
        if not all(research_capable(e.archive.read_source(candidates[cid]['source_hash'])) for cid in (parent,child)): return []
        stalled=len(events)>=2 and all(not r.get('global_best_updated') for r in events[-2:])
        if candidates[child]['fitness']<candidates[parent]['fitness']-1e-9 and not stalled: return []
        if any(r['reference_candidate_id']==parent and r['candidate_id']==child for r in e.archive.of('probe_start')): return []
        if not self.states(parent): return []
        return [dict(reference_candidate_id=parent,candidate_id=child,evidence_id=latest['event_id'])]

    @staticmethod
    def brief(result):
        if 'report' in result and 'direction' in result:
            return {k:v for k,v in result.items() if k not in ('run_results','records','status','question')}
        view=agent_report(result,result.get('reference_candidate_id'),result.get('candidate_id'),
                          result.get('question',''),result.get('experiment'))
        for key in ('event_id','reference_hash','candidate_hash','manifest_artifact','result_artifact','iteration','training_epoch'):
            if key in result: view[key]=result[key]
        view['state_results']=[dict(snapshot_id=s.get('snapshot_id'),valid=bool(s.get('valid')),
            delta_gain_ratio=s.get('delta'),delta_percent_of_start=None if s.get('delta') is None else 100*s['delta'])
            for s in result.get('state_results',[])]
        return view

    @staticmethod
    def integration_option(result, used=False):
        wins_required=5
        mean=result.get('mean_candidate_minus_reference_gain_ratio')
        completed=result.get('execution',{}).get('completed',False)
        wins=result.get('candidate_better_states',0)
        available=(completed and not used and wins>=wins_required and mean is not None
                   and mean>1e-12)
        return dict(event_id=result.get('event_id'),candidate_id=result.get('candidate_id'),
            available=available,
            requirements=dict(minimum_candidate_wins=wins_required,mean_gain_difference_above=1e-12),
            reason=('At least five starting states favor the candidate and the mean gain difference is positive.'
                if available else 'The measured result does not meet the integration rule, is incomplete, or was already used.'))

    def write_views(self,manifest,result):
        key=result['event_id'].split(':',1)[1]
        folder=self.engine.archive.root/'probes'/key; folder.mkdir(parents=True,exist_ok=True)
        (folder/'manifest.json').write_text(canonical(manifest),encoding='utf-8',newline='\n')
        (folder/'summary.json').write_text(canonical(self.brief(result)),encoding='utf-8',newline='\n')
        with (folder/'results.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
            for record in result['run_results']: stream.write(canonical(record)+'\n')

    def context(self):
        e=self.engine; used={r['probe_id'] for r in e.archive.of('integration_attempt')}
        results=[self.brief(r) for r in e.archive.of('probe_result')[-2:]]
        options=[self.integration_option(r,r.get('event_id') in used) for r in results]
        if e.config.get('alternating_agents'):
            for result,option in zip(results,options):
                if result.get('training_epoch')!=e.config['training_epoch']:
                    option.update(available=False,reason='Diagnostic was measured on a different training epoch.')
        return dict(contract_version=RESEARCH_VERSION,available_comparisons=self.pairs(),
            latest_results=results,integration_options=options,
            tool=dict(name='compare_from_states',description='Continue two complete solvers from identical frozen training solutions under the same restart seeds and time limits. The question is recorded only as the research purpose; it does not change the measured comparison.',arguments=dict(question='research purpose: why this diagnostic was requested; the operation always compares the listed candidate and reference solver continuations.')),
            execution_conditions=dict(time_limit_s=e.config['probe_seconds'],restart_seeds=list(e.config['probe_restart_seeds'])),
            rule='This is a diagnostic comparison of complete solver continuations. It does not isolate edit effects or promote a solver; promotion still requires complete formal evaluation.')

    def run(self,decision,step):
        e=self.engine; ref_id=decision['reference_candidate_id']; cid=decision['candidate_id']
        candidates=e.candidates()
        saved=next((r for r in e.archive.of('probe_start') if r['step']==step),None)
        if saved:
            if (saved['reference_candidate_id'],saved['candidate_id'],saved['question'])!=(ref_id,cid,decision['question']):
                raise ValueError('resumed diagnostic request differs from frozen manifest')
            completed=next((r for r in e.archive.of('probe_result') if r['event_id']==saved['probe_id']),None)
            if completed is not None:
                self.write_views({k:v for k,v in saved.items() if k not in ('step','manifest_artifact')},completed)
                return self.brief(completed)
            manifest={k:v for k,v in saved.items() if k not in ('step','manifest_artifact')}
            artifact=saved['manifest_artifact']
            states=manifest['states']; seeds=manifest['restart_seeds']; seconds=manifest['seconds']
            probe_id=manifest['probe_id']; key=probe_id.split(':',1)[1]
        else:
            if not any(p['reference_candidate_id']==ref_id and p['candidate_id']==cid for p in self.pairs()):
                raise ValueError('comparison unavailable or already used in this iteration')
            states=self.states(ref_id)
            seeds=e.config['probe_restart_seeds']; seconds=e.config['probe_seconds']
            implementation={name:digest(Path(__file__).with_name(name).read_text(encoding='utf-8'))
                            for name in ('research.py','replay.py','runner.py','worker.py','tasks.py','patch.py')}
            key=cache_key(candidates[ref_id]['source_hash'],candidates[cid]['source_hash'],states,
                          seeds,seconds,implementation,e.workers,e.config.get('environment_manifest'))
            probe_id='probe:'+key
            manifest=dict(probe_id=probe_id,reference_candidate_id=ref_id,candidate_id=cid,
                reference_hash=candidates[ref_id]['source_hash'],candidate_hash=candidates[cid]['source_hash'],
                states=states,restart_seeds=seeds,seconds=seconds,implementation=implementation,
                question=decision['question'],iteration=e.completed_iterations())
            if e.config.get('alternating_agents'):manifest['training_epoch']=e.config['training_epoch']
            artifact=e.archive.artifact('probe_manifests',manifest)
            e.archive.add('probe_start',dict(manifest,step=step,manifest_artifact=artifact))
        for state in states:
            row=next(r for r in self.rows if r['id']==state['instance_id'])
            if state['instance_hash']!=row['instance_hash'] or e.adapter.objective(self.load_instance(row),state['solution'])!=state['objective']:
                raise ValueError('snapshot hash/objective differs from training instance')
        folder=e.archive.root/'probes'/key; folder.mkdir(parents=True,exist_ok=True)
        (folder/'manifest.json').write_text(canonical(manifest),encoding='utf-8',newline='\n')
        total=2*len(states)*len(seeds); reservation=step+':probe'
        if reservation not in e.ledger.reservations(): e.ledger.reserve(reservation,total)
        if e.config.get('console_progress'):
            print(f'Probe | candidate={cid} | reference={ref_id} | states={len(states)} | runs=0/{total}',flush=True)
        sources={role:e.archive.read_source(candidates[target]['source_hash'])
                 for role,target in (('reference',ref_id),('candidate',cid))}
        def jobs():
            index=0
            for state in states:
                for seed in seeds:
                    order=['reference','candidate'] if index%2==0 else ['candidate','reference']
                    yield state,seed,order
                    index+=1
        def work(job):
            state,seed,order=job; output=[]
            row=next(r for r in self.rows if r['id']==state['instance_id'])
            for role in order:
                target=ref_id if role=='reference' else cid
                execution_id=f'{probe_id}:{state["snapshot_id"]}:{seed}:{role}'
                previous=next((r for r in e.archive.of('probe_execution') if r['execution_id']==execution_id),None)
                if previous is not None:
                    output.append(previous); continue
                if any(r['execution_id']==execution_id for r in e.archive.of('execution_start')):
                    record=dict(valid=False,objective=None,elapsed=None,wallclock_s=None,cpu_seconds=None,
                                started=True,error='infrastructure: interrupted diagnostic; outcome unknown; not rerun')
                else:
                    record=e.runner.case(sources[role],e.adapter,self.load_instance(row),seed,
                        lambda eid=execution_id,target=target:e.ledger.start_case(eid,reservation,target),
                        research_request=dict(kind='common_start',local_limit_s=seconds,
                                              initial_solution=state['solution'],trace_policy=TRACE_POLICY))
                record.update(execution_id=execution_id,probe_id=probe_id,candidate_id=target,
                    snapshot_id=state['snapshot_id'],instance_id=row['id'],restart_seed=seed,role=role,
                    start_objective=state['objective'],purpose='common_start',order=order)
                e.archive.add('probe_execution',record); output.append(record)
            return output
        records=[]
        try:
            for job,output in completed_tasks(jobs(),work,e.workers):
                records.extend(output)
                if e.config.get('console_progress'):
                    print(f'Probe | runs={len(records)}/{total}',flush=True)
        finally:
            e.ledger.release(reservation)
        records.sort(key=lambda r:(r['snapshot_id'],r['restart_seed'],r['role']))
        result=dict(summarize(records,states,seeds),event_id=probe_id,reference_candidate_id=ref_id,
            candidate_id=cid,reference_hash=manifest['reference_hash'],candidate_hash=manifest['candidate_hash'],
            manifest_artifact=artifact,question=decision['question'],iteration=e.completed_iterations(),
            unit='ratio',experiment=dict(method='paired_common_start',comparison_unit='complete_solver_continuation',
                time_limit_s=seconds,states=len(states),restart_seeds=list(seeds)))
        if e.config.get('alternating_agents'):result['training_epoch']=e.config['training_epoch']
        evidence=e.archive.artifact('probe_results',result); result['result_artifact']=evidence
        e.archive.add('probe_result',result)
        self.write_views(manifest,result)
        if e.config.get('console_progress'):
            print(f'Probe | status={result["status"]} | positive_states={result["positive_states"]}/{len(states)} | mean_delta={result["mean_delta"]}',flush=True)
        return self.brief(result)

    def integration(self,decision):
        e=self.engine; pid=decision.get('integration_probe_id')
        if decision.get('revision_goal')!='integrate_probe': return None
        result=next((r for r in e.archive.of('probe_result') if r['event_id']==pid),None)
        if e.config.get('alternating_agents') and result and result.get('training_epoch')!=e.config['training_epoch']:
            raise ValueError('integration diagnostic belongs to a different training epoch')
        if not result or not self.integration_option(self.brief(result),used=any(r['probe_id']==pid for r in e.archive.of('integration_attempt')) )['available']:
            raise ValueError('integration requires at least five candidate-favored states and a positive mean gain difference')
        if decision['parent_or_target_id']!=result['candidate_id']: raise ValueError('integration must explicitly target the measured candidate')
        if any(r['probe_id']==pid for r in e.archive.of('integration_attempt')):
            raise ValueError('diagnostic already used for an integration attempt')
        if pid not in decision.get('evidence_ids',[]): raise ValueError('integration must cite its diagnostic')
        return dict(probe=self.brief(result),reference_code=e.archive.read_source(result['reference_hash']),
            requirement='Explicitly identify the continuation to retain and changes to its integration. Do not claim that a nonempty citation proves semantic preservation.')

    def record_integration(self,decision,step,ids):
        existing=next((r for r in self.engine.archive.of('integration_attempt') if r['step']==step),None)
        if existing:
            child=self.engine.candidates().get(ids.get('PAB'))
            if not child or existing['probe_id']!=decision.get('integration_probe_id') or existing['generated_hash']!=child['source_hash']:
                raise ValueError('resumed integration differs from durable attempt')
            return
        context=self.integration(decision)
        if context is None: return
        e=self.engine; pid=decision['integration_probe_id']; target=ids.get('PAB')
        if target is None: raise ValueError('integration missing complete candidate')
        parent=e.candidates()[decision['parent_or_target_id']]; child=e.candidates()[target]
        change=changed_definitions(e.archive.read_source(parent['source_hash']),e.archive.read_source(child['source_hash']))
        e.archive.add('integration_attempt',dict(probe_id=pid,step=step,source_candidate_id=parent['candidate_id'],
            generated_candidate_id=target,source_hash=parent['source_hash'],generated_hash=child['source_hash'],
            changed_definitions=change,formal_fitness=child['fitness'],formal_valid=child['eligibility'],
            semantic_preservation='Requires source/behavior review; not inferred from function names or citations.'))

    def export(self):
        e=self.engine; results=e.archive.of('probe_result'); attempts=e.archive.of('integration_attempt')
        payload=dict(probes=[self.brief(r) for r in results],integration_attempts=attempts,
                     unused_integration_eligible_probes=[dict(probe_id=r['event_id'],status=('not_attempted_search_finished' if e.config.get('manual_rounds') else 'not_attempted_no_iteration') if self.search_closed() else 'awaiting_agent_decision')
                        for r in results if self.integration_option(self.brief(r),used=any(a['probe_id']==r['event_id'] for a in attempts))['available']],
                     final_best=e.best(),algorithm_superiority='not established by implementation acceptance')
        (e.archive.root/'integration_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
        return dict(probe_batches=len(results),probe_executions=len(e.archive.of('probe_execution')),
                    integration_attempts=len(attempts))
