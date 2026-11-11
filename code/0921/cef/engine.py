"""Generic CEF controller: task adapters never choose or generate an edit."""
import ast,json,statistics,time
from pathlib import Path
from .patch import compose,digest,check_source
from .feedback import choose,contrast,changed_definitions
from .store import Archive,ExperienceStore,canonical
from .ledger import BudgetLedger,BudgetError
from .runner import Runner
from .tasks import TaskAdapter
from .protocol import validate_planner,RUNTIME_CONTRACT
from .parallel import completed_tasks

class Engine:
    def __init__(self,config,model):
        if config.get('alternating_agents') and type(self) is Engine:
            raise ValueError('alternating configuration requires AlternatingEngine; use runCEFAuto.py')
        self.config=config; self.model=model; self.archive=Archive(config['run_dir'])
        old=self.archive.of('manifest')
        if old and canonical(old[0])!=canonical(config): raise ValueError('resume configuration differs from frozen manifest')
        if not old: self.archive.add('manifest',config)
        limits={k:config[k] for k in ('max_case_runs','max_wallclock_s','max_llm_requests','max_llm_tokens','max_analysis_seconds')}
        self.iteration_mode=config.get('stop_rule')=='iterations'
        self.joint_only=config.get('evaluation_policy')=='joint_only'
        self.ledger=BudgetLedger(self.archive,limits,enforce_limits=not self.iteration_mode)
        self.experiences=ExperienceStore(self.archive)
        self.adapter=TaskAdapter(config['task'],config['data_manifest'].get('training_protocol'))
        self.runner=Runner(config['solver_time_limit_s'],config.get('memory_mb',1024))
        self.rows=config['data_manifest']['train']
        self.augmented=bool(config['data_manifest'].get('training_protocol'))
        self.workers=config.get('evaluation_workers',1)
        if type(self.workers) is not int or self.workers<1: raise ValueError('evaluation_workers must be a positive integer')
        if config.get('diagnostic_run'):
            count=config['diagnostic_train_limit']
            if self.augmented:
                if count<2: raise ValueError('augmented diagnostic requires at least two cases')
                self.rows=([r for r in self.rows if r.get('training_group')!='supplement'][:(count+1)//2]+
                           [r for r in self.rows if r.get('training_group')=='supplement'][:count//2])
            else: self.rows=self.rows[:count]
        conditions=dict(rows=self.rows,seeds=config['evaluation_seed_manifest'],time=config['solver_time_limit_s'],environment=config.get('environment_manifest'),runner=digest(Path(__file__).with_name('runner.py').read_text(encoding='utf-8')))
        # Preserve historical serial hashes; new configurations freeze concurrency.
        if 'evaluation_workers' in config: conditions['evaluation_workers']=self.workers
        if self.joint_only: conditions['evaluation_policy']='joint_only'
        if self.augmented: conditions['training_protocol']=config['data_manifest']['training_protocol']
        self.eval_hash=digest(canonical(conditions))
        self.research=None
        if config.get('common_start_probe'):
            from .research import ResearchTools
            self.research=ResearchTools(self)
        self.shared_measurements=None
        if config.get('integrated_feedback_v1'):
            from .measurements import SharedMeasurements
            self.shared_measurements=SharedMeasurements(self.archive.root.parent/'measurements')

    def candidates(self):
        candidates={c['candidate_id']:dict(c) for c in self.archive.of('candidate')}
        for c in candidates.values():
            measurements=[r for r in self.archive.of('measurement') if r['candidate_id']==c['candidate_id']]
            c['eligibility']=bool(measurements) and all(m['valid'] for m in measurements)
            c['fitness']=statistics.fmean(m['fitness'] for m in measurements) if c['eligibility'] else None
            if self.augmented and measurements:
                c['training_group_scores']={g:statistics.fmean(m['training_group_scores'][g] for m in measurements)
                    for g in ('original','supplement')} if c['eligibility'] else None
        return candidates

    def best(self):
        c=self.candidates(); key=choose({k:v['fitness'] for k,v in c.items() if v['eligibility']})
        return c.get(key)

    def select_eligible(self,ids):
        candidates=self.candidates()
        return choose({label:candidates[cid]['fitness'] for label,cid in ids.items() if candidates[cid]['eligibility']})

    def visible_event(self,event):
        view=dict(event) if self.config['mode'] in ('CEF','B4') else {k:v for k,v in event.items() if k not in ('contrasts','per_instance_contrasts','groups','prediction_observations')}
        view.update(score_unit=self.adapter.fitness_metric['unit'],
                    scores_display={k:self.adapter.format_fitness(v) for k,v in view.get('scores',{}).items()})
        return view

    def execution_failures(self,cid,records=None):
        records=records if records is not None else [row for row in self.archive.of('execution') if row['candidate_id']==cid]
        failed=[row for row in records if not row['valid']]
        examples=[]; seen=set()
        for row in failed:
            error=row.get('error') or 'unknown execution failure'
            if error in seen: continue
            seen.add(error)
            examples.append(dict(execution_id=row.get('execution_id'),instance_id=row.get('instance_id'),error=error,error_location=row.get('error_location',[])))
            if len(examples)==3: break
        return dict(total=len(records),failed=len(failed),examples=examples)

    def action_context(self):
        candidates=self.candidates()
        if not candidates: return None
        allowed=['revise'] if self.config['mode']=='B1' else ['inspect','profile','analyze','revise']
        if self.joint_only: allowed=['inspect','analyze','revise']
        probe_pairs=self.research.pairs() if self.research else []
        research_context=self.research.context() if self.research else {}
        integration_options=research_context.get('integration_options',[])
        if probe_pairs: allowed.append('compare_from_states')
        if not self.iteration_mode: allowed.append('finish')
        return dict(allowed_actions=allowed,candidate_eligibility={key:value['eligibility'] for key,value in candidates.items()},
            bootstrap_parent_ids=['seed'] if self.joint_only else [],
            events=[] if self.config['mode']=='B1' else self.index()['events'],cards=[] if self.config['mode']=='B1' else self.index()['cards'],
            probes=self.index().get('probes',[]),probe_pairs=probe_pairs,
            integration_options=integration_options,
            integration_opportunities=[option for option in integration_options if option['available']],
            max_profile_instances=min(self.config['max_profile_instances'],len(self.rows)),
            profile_rule='Disabled in joint_only; only complete PAB evaluation executes solvers.' if self.joint_only else 'Any known candidate, including failed candidates. Profiling is diagnostic and does not make a candidate eligible.',
            revise_rule='Eligible measured parents or explicit unmeasured bootstrap seed. The bootstrap has no measured score and is not a ranked result. Every revision has two intents, evaluated only jointly.' if self.joint_only else 'Only eligible parents. Never silently replace the requested parent.',
            **({'tool_evidence':self.index().get('tool_evidence',[])} if self.config.get('agent_task_runtime_v2') else {}))

    def visible_action_context(self):
        context=self.action_context()
        if context is not None and self.config['mode']=='B1':
            return {key:context[key] for key in ('allowed_actions','revise_rule')}
        return context

    def execute_case(self,cid,row,eid,reservation,purpose='evaluation',order=None):
        completed=[r for r in self.archive.of('execution') if r['execution_id']==eid]
        if completed:
            if self.research: self.research.capture(cid,row,completed[0])
            return completed[0]
        self.ledger.check('wallclock_s')
        if self.ledger.remaining('wallclock_s')<self.config['solver_time_limit_s']+20: raise BudgetError('insufficient remaining wallclock for one bounded worker')
        if any(r['execution_id']==eid for r in self.archive.of('execution_start')):
            # Reconcile old two-record journals idempotently; new starts are atomic.
            starts=[r for r in self.archive.of('execution_start') if r['execution_id']==eid]
            if not starts[0].get('resource') and not any(r.get('execution_id')==eid for r in self.archive.of('charge')):
                self.ledger.charge('case_runs',1,reservation=reservation,execution_id=eid,reconciled_interrupted_start=True)
            rec=dict(valid=False,objective=None,fitness=None,error='infrastructure: interrupted execution; result unknown; not silently rerun',elapsed=0,started=True,wallclock_s=0)
        else:
            instance=self.adapter.load(row)
            extra={'research_request':dict(kind='trace',trace_policy=self.config.get('trace_policy',{}))} if self.research else {}
            rec=self.runner.case(self.archive.read_source(self.candidates()[cid]['source_hash']),self.adapter,instance,self.config['evaluation_seed_manifest'][row['id']],lambda:self.ledger.start_case(eid,reservation,cid),**extra)
        rec.update(execution_id=eid,candidate_id=cid,instance_id=row['id'],seed=self.config['evaluation_seed_manifest'][row['id']],config_hash=self.eval_hash,group=self.adapter.group(self.adapter.load(row)),purpose=purpose,order=order)
        if self.augmented: rec['training_group']=row.get('training_group','original')
        self.archive.add('execution',rec)
        if self.research: self.research.capture(cid,row,rec)
        return rec

    def register(self,id,source,parent_id=None,**extra):
        if self.research and id!='seed': check_source(source,require_research=True)
        existing=self.candidates().get(id)
        h=self.archive.source(source)
        if existing:
            if existing['source_hash']!=h: raise ValueError('candidate id/source collision')
        else: self.archive.add('candidate',dict(candidate_id=id,parent_id=parent_id,source_hash=h,source_path=str(self.archive.root/'sources'/f'{h}.py'),evaluation_config_hash=self.eval_hash,**extra))

    def execute_batch(self,jobs,reservation,purpose='evaluation'):
        """Workers execute only Runner.case; controller owns preparation and results."""
        if self.shared_measurements is not None:
            yield from self.shared_measurements.execute_batch(self,jobs,reservation,purpose)
            return
        if self.workers==1:
            for cid,row,eid,order in jobs:
                yield self.execute_case(cid,row,eid,reservation,purpose,order)
            return
        existing={r['execution_id'] for r in self.archive.of('execution')+self.archive.of('execution_start')}
        sources={}
        def prepared():
            for cid,row,eid,order in jobs:
                if eid in existing:
                    yield (cid,row,eid,order,None,None,self.execute_case(cid,row,eid,reservation,purpose,order))
                    continue
                self.ledger.check('wallclock_s')
                if self.ledger.remaining('wallclock_s')<self.config['solver_time_limit_s']+20:
                    raise BudgetError('insufficient remaining wallclock for one bounded worker')
                if cid not in sources: sources[cid]=self.archive.read_source(self.candidates()[cid]['source_hash'])
                yield (cid,row,eid,order,self.adapter.load(row),sources[cid],None)
        def work(item):
            cid,row,eid,order,instance,source,previous=item
            if previous is not None: return previous
            extra={'research_request':dict(kind='trace',trace_policy=self.config.get('trace_policy',{}))} if self.research else {}
            return self.runner.case(source,self.adapter,instance,self.config['evaluation_seed_manifest'][row['id']],lambda:self.ledger.start_case(eid,reservation,cid),**extra)
        for item,rec in completed_tasks(prepared(),work,self.workers):
            cid,row,eid,order,instance,source,previous=item
            if previous is None:
                rec.update(execution_id=eid,candidate_id=cid,instance_id=row['id'],seed=self.config['evaluation_seed_manifest'][row['id']],config_hash=self.eval_hash,group=self.adapter.group(instance),purpose=purpose,order=order)
                if self.augmented: rec['training_group']=row.get('training_group','original')
                self.archive.add('execution',rec)
                if self.research: self.research.capture(cid,row,rec)
            yield rec

    def evaluate(self,variants,reservation,measure_id):
        """Rotate version submission; aggregate only after all paired cases finish."""
        completed={r['execution_id']:r for r in self.archive.of('execution')}
        def jobs():
            for index,row in enumerate(self.rows):
                order=list(variants)
                order=order[index%len(order):]+order[:index%len(order)]
                for label in order:
                    cid=variants[label]; eid=f'{measure_id}:{label}:{row["id"]}'
                    if eid not in completed: yield cid,row,eid,order
        total=len(variants)*len(self.rows)
        done=sum(f'{measure_id}:{label}:{r["id"]}' in completed for label in variants for r in self.rows)
        if self.config.get('console_progress'): print(f'Evaluate {measure_id} | {done}/{total} | workers={self.workers}',flush=True)
        for rec in self.execute_batch(jobs(),reservation):
            completed[rec['execution_id']]=rec; done+=1
            if self.config.get('console_progress') and (done%8==0 or done==total):
                print(f'Evaluate {measure_id} | {done}/{total} | workers={self.workers}',flush=True)
        scores={}; records={}
        for label,cid in variants.items():
            rows=[completed[f'{measure_id}:{label}:{r["id"]}'] for r in self.rows]
            if self.shared_measurements is not None and self.research:
                for row,record in zip(self.rows,rows):self.research.capture(cid,row,record)
            valid=all(r['valid'] for r in rows)
            fitness=statistics.fmean(r['fitness'] for r in rows) if valid else None
            mid=f'{measure_id}:{label}'
            group_scores={}
            if self.augmented:
                group_scores={g:statistics.fmean(r['fitness'] for r in rows if r['training_group']==g) if valid else None
                              for g in ('original','supplement')}
            if not any(m['measurement_id']==mid for m in self.archive.of('measurement')):
                extra={'training_group_scores':group_scores} if self.augmented else {}
                self.archive.add('measurement',dict(measurement_id=mid,candidate_id=cid,valid=valid,fitness=fitness,execution_ids=[r['execution_id'] for r in rows],config_hash=self.eval_hash,**extra))
            scores[label]=fitness; records[label]=rows
        return scores,records

    def state(self,parent_id=None):
        c=self.candidates(); b=self.best()
        if parent_id is None:
            completed=self.archive.of('action_complete')
            parent_id=completed[-1].get('active_parent') if completed else None
        parent=c.get(parent_id) or b or (c.get('seed') if self.joint_only else None)
        ancestors=[]; node=parent
        for _ in range(2):
            node=c.get(node.get('parent_id')) if node else None
            if node: ancestors.append(node['source_hash'])
        events=self.archive.of('contrast'); recent=events[-2:]
        functions=events[-1].get('affected_functions',[]) if events else []
        cards=self.experiences.retrieve(parent['source_hash'],ancestors,functions) if parent else []
        mode=self.config['mode']
        if mode in ('B0','B2','B4','B1'): cards=[]
        state=dict(active_parent=parent,global_best=b,candidate_summaries=[v for v in c.values() if v['eligibility'] and v!=parent and v!=b][:3],recent_events=[self.visible_event(e) for e in recent],retrieved_cards=cards,archive_index=self.index(),budget_balance=self.ledger.balance(),task_interface=self.adapter.description,executions_since_improvement=self.executions_since_improvement(),run_purpose=self.config.get('acceptance_note','algorithm design'))
        if mode=='B1':
            # Explicit no-performance-history generation mode, not hidden raw access.
            state={k:state[k] for k in ('task_interface','budget_balance')}
            state['active_parent']={k:parent[k] for k in ('candidate_id','source_hash')} if parent else None
            if parent: state['current_program']=self.archive.read_source(parent['source_hash'])
        if self.iteration_mode:
            state['search_control']=dict(stop_rule='iterations',max_iterations=self.config['max_iterations'],completed_iterations=self.completed_iterations(),allowed_actions=self.action_context()['allowed_actions'],resource_usage_is_accounting_only=True,finish_allowed=False)
        state['action_context']=self.visible_action_context()
        state['runtime_contract']=RUNTIME_CONTRACT
        if self.joint_only:
            state['evaluation_policy']='joint_only'
            state['current_program']=self.archive.read_source(parent['source_hash']) if parent else None
            state['measurement_rule']='Generate two intents A+B; evaluate only their complete joint PAB. The seed is an unmeasured bootstrap parent, never a ranked result. Individual edits and interaction effects are not measured. No profile executions.'
        if mode!='B1':
            state['candidate_diagnostics']=[dict(candidate_id=cid,execution_failures=self.execution_failures(cid)) for cid in list(c)[-3:] if not c[cid]['eligibility']]
        if self.config.get('diagnostic_run'):
            state['measurement_protocol']=dict(diagnostic_run=True,training_instances=len(self.rows),solver_time_limit_s=self.config['solver_time_limit_s'],scope='runtime validation only, not formal algorithm performance')
        if self.research:
            state['common_start']=self.research.context()
            state['runtime_contract']=dict(RUNTIME_CONTRACT,research_interface='solve(instance,time_limit_s,seed,*,initial_solution=None,observer=None); same continuation in both modes; observer(solution) is mandatory in research calls.')
            state['measurement_rule']+=' Common-start diagnostics run complete versions from frozen training solutions; they do not measure edit components or alter full-score ranking.'
        state['fitness_metric']=self.adapter.fitness_metric
        if mode!='B1':
            for name in ('active_parent','global_best'):
                if state.get(name): state[name]=dict(state[name],fitness_display=self.adapter.format_fitness(state[name]['fitness']))
            state['candidate_summaries']=[dict(v,fitness_display=self.adapter.format_fitness(v['fitness'])) for v in state['candidate_summaries']]
        return state

    def completed_iterations(self):
        return sum(r['action']=='revise' for r in self.archive.of('action_complete'))

    def executions_since_improvement(self):
        last=0
        for r in self.archive.records:
            if r['kind']=='contrast' and r['data'].get('global_best_updated'): last=r['seq']
        return sum(r['kind']=='execution' and r['seq']>last for r in self.archive.records)

    def index(self):
        value=dict(candidates=list(self.candidates()),events=[e['event_id'] for e in self.archive.of('contrast')],cards=[c['card_id'] for c in self.experiences.cards()])
        if self.config.get('agent_task_runtime_v2'):
            value['tool_evidence']=[r['evidence_id'] for r in self.archive.of('agent_tool_evidence')]
        if self.research: value['probes']=[r['event_id'] for r in self.archive.of('probe_result')]
        return value

    def known_evidence_ids(self):
        index=self.index()
        return set(index['events']+index['cards']+index.get('probes',[])+index.get('tool_evidence',[]))

    def request(self,role,payload,id,*,allow_schema_repair=True):
        done=[r for r in self.archive.of('model_response') if r['call_id']==id]
        if done: return done[0]['response']
        raw=[r for r in self.archive.of('model_raw_response') if r['call_id']==id]
        started=any(r['call_id']==id for r in self.archive.of('model_start'))
        native=not raw and self.config.get('agent_tool_loop_v1') and role in ('planner','editor')
        if started and not raw and not native:
            raise RuntimeError('infrastructure: interrupted model request; usage/outcome unknown, no automatic duplicate request')
        if started and not raw and native:
            # Resume against the original task input, not a newly assembled
            # state containing changed accounting or provider-event counts.
            payload=next(r['payload'] for r in self.archive.of('model_start') if r['call_id']==id)
        if raw: response=raw[0]['response']
        else:
            if self.config.get('console_progress'): print(f'{role.title()} | {id}',flush=True)
            if not started:self.archive.add('model_start',dict(call_id=id,role=role,payload=payload))
            if native:
                from .tool_agents import run_algorithm_task
                response=run_algorithm_task(self,self.model,role,payload,id)
            else:response=self.model.call(role,payload)
            self.archive.add('model_raw_response',dict(call_id=id,role=role,response=response))
        if role=='planner':
            try: validate_planner(response,self.config,self.action_context())
            except ValueError as error:
                if not any(r['call_id']==id for r in self.archive.of('model_validation_failure')):
                    self.archive.add('model_validation_failure',dict(call_id=id,response=response,error=str(error)))
                if not allow_schema_repair: raise
                repair=dict(payload,protocol_error=str(error),previous_response=response,action_context=self.visible_action_context(),
                    repair_instruction='Return the complete corrected Planner JSON object. Explicitly supply every action-specific field, especially integer edit_count for revise. Do not invent evaluations or evidence. This is a recorded protocol repair, not a new experiment.')
                response=self.request(role,repair,id+':schema_repair',allow_schema_repair=False)
        self.archive.add('model_response',dict(call_id=id,role=role,response=response))
        return response

    def update_experience(self,updates):
        allowed=self.known_evidence_ids()
        for u in updates:
            if not set(u.get('evidence_ids',[]))<=allowed: raise ValueError('experience cites nonexistent evidence')
            self.experiences.revise(u['card_id'],u['interpretation'],u['status'],u.get('evidence_ids',[]))

    def tool(self,decision,step):
        if any(r['step']==step for r in self.archive.of('tool_result')): return
        action=decision['action']; target=decision.get('parent_or_target_id')
        if self.config['mode']=='B1': raise ValueError('B1 cannot inspect performance history')
        if action=='compare_from_states':
            if not self.research: raise ValueError('common-start tool disabled')
            result=self.research.run(decision,step)
            self.archive.add('tool_result',dict(step=step,action=action,result=result))
            return
        if action=='inspect':
            if target in self.candidates(): result=dict(source=self.archive.read_source(self.candidates()[target]['source_hash']),candidate=self.candidates()[target],execution_failures=self.execution_failures(target))
            elif self.config.get('agent_task_runtime_v2') and target in self.index().get('tool_evidence',[]):
                from .config import file_hash
                entry=next(r for r in self.archive.of('agent_tool_evidence') if r['evidence_id']==target)
                if file_hash(entry['path'])!=entry['sha256']:raise RuntimeError('tool evidence changed')
                result=json.loads(Path(entry['path']).read_text(encoding='utf-8'))
            else:
                matches=[x for x in self.archive.of('contrast')+self.experiences.cards()+self.archive.of('probe_result') if x.get('event_id',x.get('card_id'))==target]
                if not matches: raise ValueError('inspect unknown ID')
                result=matches[0]
            if self.config['mode'] in ('B0','B2','B3') and isinstance(result,dict):
                result={k:v for k,v in result.items() if k not in ('contrasts','per_instance_contrasts','groups')}
            self.archive.add('tool_result',dict(step=step,action=action,result=result)); return
        if action=='profile':
            n=decision.get('resource_request',{}).get('instances')
            if type(n) is not int or not 1<=n<=self.config['max_profile_instances']: raise ValueError('invalid profile instance count')
            c=self.candidates().get(target)
            if not c: raise ValueError('unknown profile candidate')
            if step not in self.ledger.reservations(): self.ledger.reserve(step,n)
            try:
                output=[]
                jobs=((target,row,f'{step}:profile:{row["id"]}',None) for row in self.rows[:n])
                output=list(self.execute_batch(jobs,step,purpose='profile'))
                output.sort(key=lambda rec:next(i for i,row in enumerate(self.rows[:n]) if row['id']==rec['instance_id']))
                self.archive.add('tool_result',dict(step=step,action=action,result=output))
            finally: self.ledger.release(step)
            return
        if action=='analyze':
            from .analysis import analyze
            since=self.archive.of('action_complete')
            count=0
            for r in reversed(since):
                if r.get('action')=='revise': break
                count+=r.get('action')=='analyze'
            if not self.iteration_mode and count>=self.config['max_analysis_per_round']:
                raise BudgetError('analysis actions per round exhausted')
            seconds=min(self.config['analysis_timeout_s'],self.ledger.remaining('analysis_seconds'),self.ledger.remaining('wallclock_s'))
            if seconds<=0: raise BudgetError('analysis budget exhausted')
            events=[self.visible_event(e) for e in self.archive.of('contrast')]
            if self.research: events.extend(self.research.brief(r) for r in self.archive.of('probe_result'))
            result=analyze(decision['analysis_code'],events,seconds)
            self.ledger.charge('analysis_seconds',result['elapsed'],memory_peak_bytes=result.get('memory_peak_bytes'))
            self.archive.add('tool_result',dict(step=step,action=action,result=result)); return
        raise ValueError('unknown action')

    def editor_context(self,cid,decision,count):
        parent=self.candidates()[cid]; state=self.state(cid)
        related=list(state.get('recent_events',[])); cards=list(state.get('retrieved_cards',[]))
        selected=set(decision.get('evidence_ids',[]))
        for card in self.experiences.cards():
            if card['card_id'] in selected and not any(c['card_id']==card['card_id'] for c in cards): cards.append(card)
        event_ids=selected|{id for c in cards for id in c['source_event_ids']}
        for ev in self.archive.of('contrast'):
            if ev['event_id'] in event_ids and not any(x['event_id']==ev['event_id'] for x in related): related.append(self.visible_event(ev))
        value=dict(task_interface=self.adapter.description,fitness_metric=self.adapter.fitness_metric,parent_id=cid,parent_hash=parent['source_hash'],parent_code=self.archive.read_source(parent['source_hash']),decision=decision,contrasts=related,experiences=cards,entry='solve(instance, time_limit_s, seed)',edit_count=count,runtime_contract=state['runtime_contract'])
        if self.augmented: value['training_group_scores']=parent.get('training_group_scores')
        if self.research:
            value['probe_evidence']=[self.research.brief(r) for r in self.archive.of('probe_result') if r['event_id'] in event_ids]
            value['integration_context']=self.research.integration(decision)
        return value

    def package(self,decision,step):
        cid=decision['parent_or_target_id']; parent=self.candidates().get(cid)
        if not parent or not (parent['eligibility'] or (self.joint_only and cid=='seed')): raise ValueError('parent must be a legal historical candidate or explicit unmeasured bootstrap')
        count=decision.get('edit_count')
        if count not in (1,2) or (self.config['mode']=='B0' and count!=1): raise ValueError('invalid edit_count for mode')
        if self.joint_only and count!=2: raise ValueError('joint_only requires edit_count=2')
        cases=(1 if self.joint_only else 2 if count==1 else 4)*len(self.rows)
        if step not in self.ledger.reservations(): self.ledger.reserve(step,cases)
        before=self.best()
        try:
            saved=[r for r in self.archive.of('package') if r['step']==step]
            if saved: package=saved[0]['package']
            else:
                payload=self.editor_context(cid,decision,count)
                if self.config['mode']=='B1': payload={k:v for k,v in payload.items() if k not in ('contrasts','experiences','decision')}
                package=self.request('editor',payload,step+':editor')
                for attempt in range(2):
                    try:
                        if package['parent_id']!=cid or len(package['edits'])!=count: raise ValueError('editor changed parent/count')
                        variants=compose(payload['parent_code'],package)
                        if not package.get('issue') or not isinstance(package.get('evidence_ids'),list): raise ValueError('package metadata missing')
                        if not set(package['evidence_ids'])<=self.known_evidence_ids(): raise ValueError('editor cites unknown evidence')
                        if self.research: check_source(variants['PAB'],require_research=True)
                        break
                    except (ValueError,KeyError,TypeError) as e:
                        self.archive.add('package_failure',dict(step=step,attempt=attempt,package=package,error=str(e)))
                        if attempt: raise
                        package=self.request('editor',dict(payload,repair_error=str(e),previous_package=package,require_new_package_id=True),step+':repair')
                self.archive.add('package',dict(step=step,package=package))
            variants=compose(self.archive.read_source(parent['source_hash']),package)
            ids={} if self.joint_only else {'P':cid}
            for label,text in variants.items():
                if label=='P': continue
                if self.joint_only and label!='PAB': continue
                id=step+':'+label; self.register(id,text,cid,package_id=package['package_id'],variant=label); ids[label]=id
            if self.config.get('agent_task_runtime_v2') and (self.archive.root/'pause.request').exists():
                from .tool_session import TaskPaused
                raise TaskPaused('pause requested after Editor submission, before full candidate evaluation')
            scores,records=self.evaluate(ids,step,step)
            selected=self.select_eligible(ids); event_id=step+':contrast'
            source=self.archive.read_source(parent['source_hash'])
            functions=[]
            for n in ast.walk(ast.parse(source)):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    text=ast.get_source_segment(source,n) or ''
                    if any(r['old'] in text for e in package['edits'] for r in e['exact_replacements']): functions.append(n.name)
            measured_contrasts={} if self.joint_only else contrast(scores)
            per_instance={} if self.joint_only else {row['id']:contrast({label:records[label][i]['fitness'] for label in ids}) for i,row in enumerate(self.rows)}
            groups={}
            for label,rows in records.items():
                for group in sorted({r['group'] for r in rows}):
                    values=[r['fitness'] for r in rows if r['group']==group]
                    groups.setdefault(group,{})[label]=statistics.fmean(values) if all(x is not None for x in values) else None
            after=self.best()
            event=dict(event_id=event_id,package_id=package['package_id'],parent_id=cid,parent_hash=parent['source_hash'],variant_ids=ids,source_hashes={k:self.candidates()[v]['source_hash'] for k,v in ids.items()},scores=scores,contrasts=measured_contrasts,per_instance_contrasts=per_instance,groups=groups,missing_fields=['component_contrasts_not_measured'] if self.joint_only else [k for k,v in measured_contrasts.items() if v is None],selected_candidate=ids.get(selected),execution_ids=[r['execution_id'] for rows in records.values() for r in rows],tested_conditions=dict(config_hash=self.eval_hash,instances=[r['id'] for r in self.rows],seeds=self.config['evaluation_seed_manifest'],time_limit_s=self.config['solver_time_limit_s']),affected_functions=functions,cost=dict(case_runs=sum(r['started'] and not r.get('reused_observation',False) for rows in records.values() for r in rows),solve_seconds=sum(r['elapsed'] for rows in records.values() for r in rows if not r.get('reused_observation',False)),wallclock_s=sum(r['wallclock_s'] for rows in records.values() for r in rows)),prediction_observations=[dict(edit_id=e['edit_id'],prediction=e['prediction'],measured=scores.get('P'+e['edit_id'])) for e in package['edits']],global_best_updated=bool(after and (not before or after['source_hash']!=before['source_hash'])))
            if self.joint_only: event.update(evaluation_policy='joint_only',component_contrasts_measured=False)
            if self.augmented:
                event['training_group_scores']={label:self.candidates()[id].get('training_group_scores') for label,id in ids.items()}
            event['execution_failures']={label:self.execution_failures(ids[label],rows) for label,rows in records.items()}
            if not any(e['event_id']==event_id for e in self.archive.of('contrast')): self.archive.add('contrast',event)
            if self.config['mode'] in ('CEF','B3'):
                card_id=step+':card'
                if not any(c['card_id']==card_id for c in self.experiences.cards()):
                    selected_text=variants.get(selected,source)
                    changes=changed_definitions(source,selected_text)
                    observation=event['contrasts'] if self.config['mode']=='CEF' else dict(selected_candidate=event['selected_candidate'],changed_functions=changes['functions'],changed_constants=changes['constants'],total_fitness_change=None if selected is None or scores['P'] is None else scores['P']-scores[selected])
                    if self.joint_only: observation=dict(scores=scores,execution_failures=event['execution_failures'],component_contrasts_measured=False)
                    self.experiences.create(card_id,dict(source_event_ids=[event_id],parent_hash=parent['source_hash'],edit_ids=[e['edit_id'] for e in package['edits']],affected_functions=functions,measured_observation=observation,tested_conditions=event['tested_conditions'],cross_version_unknowns='unconfirmed outside the measured parent and test conditions'))
                if len(package['edits'])==1 and any(len(r['old'])>len(source)*0.8 for r in package['edits'][0]['exact_replacements']):
                    self.archive.add('structural_rewrite',dict(event_id=event_id,cross_version_applicability='unconfirmed'))
            if self.research: self.research.record_integration(decision,step,ids)
            return ids.get(selected) or cid
        finally: self.ledger.release(step)

    def run(self):
        try:
            if not self.candidates(): self.register('seed',self.adapter.seed_source())
            if not self.joint_only and not self.archive.of('measurement'):
                if 'initial' not in self.ledger.reservations(): self.ledger.reserve('initial',len(self.rows))
                try: self.evaluate({'P':'seed'},'initial','initial')
                finally: self.ledger.release('initial')
            if not self.joint_only and not self.best(): raise RuntimeError('no legal initial candidate')
            while True:
                if self.iteration_mode:
                    if self.completed_iterations()>=self.config['max_iterations']:
                        if self.joint_only and not self.best(): return self.export('failed','configured iterations completed; no valid generated PAB; no baseline fallback')
                        return self.export('finished','configured iterations completed')
                else:
                    self.ledger.check('wallclock_s')
                    if any(self.ledger.remaining(k)<=0 for k in ('case_runs','llm_requests','llm_tokens','analysis_seconds')): raise BudgetError('a run resource hard cap reached')
                step='step_'+str(len(self.archive.of('action_complete')))
                state=self.state()
                state['tool_observations']=self.archive.of('tool_result')[-2:]
                decision=self.request('planner',state,step+':planner')
                action=decision['action']
                if action not in ('inspect','analyze','profile','revise','finish','compare_from_states'): raise ValueError('invalid planner action')
                if not set(decision.get('evidence_ids',[]))<=self.known_evidence_ids(): raise ValueError('nonexistent evidence reference')
                self.update_experience(decision.get('previous_event_update',[]))
                if action=='finish':
                    if self.iteration_mode: raise ValueError('finish is not allowed before configured iterations complete')
                    return self.export('finished','planner finish')
                parent=self.state().get('active_parent',{})
                active=parent.get('candidate_id') if parent else None
                if action=='revise': active=self.package(decision,step)
                else: self.tool(decision,step)
                self.archive.add('action_complete',dict(step=step,action=action,active_parent=active))
                if self.config.get('console_progress') and action=='revise':
                    scores=self.archive.of('contrast')[-1]['scores']; offspring=[value for key,value in scores.items() if key!='P']
                    best=self.best(); gap=f"{best['fitness']:.6f}%" if best else 'unavailable'
                    print(f"Iteration {self.completed_iterations()}/{self.config['max_iterations']} | best_gap={gap} | valid_offspring={sum(value is not None for value in offspring)}/{len(offspring)}",flush=True)
        except BudgetError as e: return self.export('failed' if self.iteration_mode else 'budget_terminated',str(e))
        except Exception as e:
            self.archive.add('failure',dict(error=type(e).__name__+': '+str(e)))
            return self.export('failed',type(e).__name__+': '+str(e))

    def export(self,status,reason):
        best=self.best()
        packages=self.archive.of('package'); events=self.archive.of('contrast'); charges=self.archive.of('charge')
        two=sum(len(p['package']['edits'])==2 for p in packages); single=len(packages)-two
        stats=dict(packages=len(packages),two_edit=two,single_edit=single,full_rewrites=len(self.archive.of('structural_rewrite')),complete_contrasts=sum(not e['missing_fields'] for e in events),two_edit_fraction=two/len(packages) if packages else None)
        costs=dict(model_requests=self.ledger.used('llm_requests'),model_tokens=self.ledger.used('llm_tokens'),input_tokens=sum(r.get('input_tokens',0) for r in charges),cached_input_tokens=sum(r.get('cached_input_tokens',0) for r in charges),output_tokens=sum(r.get('output_tokens',0) for r in charges),model_fee=sum(r.get('fee') or 0 for r in charges),unknown_usage_attempts=sum(r.get('usage_known') is False for r in charges),case_runs=self.ledger.used('case_runs'),solve_seconds=sum(r['elapsed'] for r in self.archive.of('execution') if not r.get('reused_observation',False)),analysis_seconds=self.ledger.used('analysis_seconds'),wallclock_s=self.ledger.used('wallclock_s'))
        if self.research:
            probes=self.archive.of('probe_execution')
            starts=self.archive.of('execution_start')
            probe_starts=[r for r in starts if r['execution_id'].startswith('probe:')]
            unknown=sum(r.get('elapsed') is None for r in probes)+len({r['execution_id'] for r in probe_starts}-{r['execution_id'] for r in probes})
            known_seconds=sum(r['elapsed'] for r in probes if r.get('elapsed') is not None)
            costs.update(formal_case_runs=len(starts)-len(probe_starts),probe_case_runs=len(probe_starts),
                         probe_execution_records=len(probes),probe_unknown_duration_runs=unknown,
                         known_probe_solve_seconds=known_seconds,probe_solve_seconds=None if unknown else known_seconds)
            costs['solve_seconds']=None if unknown else costs['solve_seconds']+known_seconds
        if self.config.get('integrated_feedback_v1'):
            starts=self.archive.of('execution_start')
            costs['solve_seconds']=sum(r['record']['elapsed'] for r in self.shared_measurements.archive.of('observation'))
            if self.research:
                costs['solve_seconds']=None if unknown else costs['solve_seconds']+known_seconds
            costs['data_calibration_case_runs']=sum(r['execution_id'].startswith('data_round_') for r in starts)
            costs['formal_case_runs']=sum(not r['execution_id'].startswith(('data_round_','probe:')) for r in starts)
            costs['reused_observation_views']=sum(bool(r.get('reused_observation')) for r in self.archive.of('execution'))
        if self.config.get('agent_task_runtime_v2'):
            issued={(r['experiment_id'],r['id']) for r in self.archive.of('experimental_case_start')}
            finished={(r['experiment_id'],r['id']) for r in self.archive.of('experimental_execution')}
            generated={(r['experiment_id'],r['seed']) for r in self.archive.of('experimental_generation_start')}
            generation_done={(r['experiment_id'],r['seed']) for r in self.archive.of('experimental_generation')}
            costs.update(experimental_case_runs=len(issued),experimental_unknown_outcomes=len(issued-finished),
                experimental_generator_runs=len(generated),experimental_unknown_generation_outcomes=len(generated-generation_done))
            tool_started={r['execution_id'] for r in self.archive.of('agent_tool_execution_start')}
            tool_finished={r['execution_id'] for r in self.archive.of('agent_tool_execution_complete')}
            costs['agent_tool_unknown_outcomes']=len(tool_started-tool_finished)
            validations={r['validation_id'] for r in self.archive.of('data_validation_start')}
            checked={r['validation_id'] for r in self.archive.of('data_validation_result')}
            costs.update(data_validation_seed_pairs=len(validations),data_validation_unknown_outcomes=len(validations-checked))
        result=dict(status=status,reason=reason,best=best,budget=self.ledger.balance(),packet_statistics=stats,resource_usage=costs,algorithm_performance='not established by implementation acceptance')
        if self.joint_only: result.update(evaluation_policy='joint_only',experiment_variant='PAB-only generation, no component contrasts')
        if self.research: result['experiment_variant']='PAB generation with common-start diagnostics; no component contrasts'
        if self.config.get('diagnostic_run'):
            result['measurement_protocol']=dict(diagnostic_run=True,training_instances=len(self.rows),solver_time_limit_s=self.config['solver_time_limit_s'])
        if self.iteration_mode:
            costs['known_model_tokens']=sum(r['amount'] for r in charges if r['resource']=='llm_tokens' and r.get('usage_known'))
            costs['model_tokens']=None if costs['unknown_usage_attempts'] else costs['known_model_tokens']
            costs['model_fee']=None
            costs['fee_source']='provider amount unavailable in generation response; see billing_summary.json'
            result.update(completed_iterations=self.completed_iterations(),max_iterations=self.config['max_iterations'],stop_rule='iterations')
        if self.config.get('manual_rounds'):
            result.update(manual_rounds=True,completed_rounds=len(self.archive.of('round_complete')),
                awaiting_user=status=='paused')
        if self.research: result['common_start']=self.research.export()
        if best:
            text=self.archive.read_source(best['source_hash']); (self.archive.root/'best_solver.py').write_text(text,encoding='utf-8',newline='')
        self.archive.add('termination',result)
        (self.archive.root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result
