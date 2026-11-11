"""Opt-in alternating search. Every fitness belongs to one immutable data epoch."""
import copy
import json
from pathlib import Path
import statistics

from .config import file_hash, load_json, freeze_research_contract
from .data import audit_training, receipt
from .engine import Engine
from .instance_design import DataDesigner, write_json
from .reference_jobs import ReferencePending
from .patch import digest, check_source
from .store import canonical
from .tasks import TaskAdapter
from .tool_session import TaskPaused,TaskBlocked


class AlternatingEngine(Engine):
    def __init__(self, config, model, data_provider=None):
        self.frozen_config=copy.deepcopy(config)
        self.training_epoch=0
        self.data_provider=data_provider
        super().__init__(copy.deepcopy(config),model)
        if not config.get('alternating_agents'): raise ValueError('explicit alternating_agents configuration required')
        epochs=self.archive.of('training_epoch')
        if epochs: self.activate(epochs[-1])

    def activate(self, epoch):
        artifact=epoch['artifact']
        if file_hash(artifact['path'])!=artifact['sha256']: raise ValueError('training epoch artifact changed')
        saved=load_json(artifact['path'])
        if saved['round']!=epoch['round']: raise ValueError('training epoch number mismatch')
        data=saved['data_manifest']
        c=copy.deepcopy(self.frozen_config)
        if not c.get('diagnostic_run'):
            audit_training('tsp',data['train'],data.get('training_protocol'))
            binding=data['supplement_manifest']
            if file_hash(binding['path'])!=binding['sha256']: raise ValueError('epoch supplement manifest changed')
            supplement=load_json(binding['path'])
            if (supplement.get('diagnostic') is not False or supplement.get('protocol')!=data.get('training_protocol') or
                    supplement['train']!=data['train'][64:]):
                raise ValueError('epoch must bind formal supplemental training data')
        c.update(data_manifest=data,training_epoch=epoch['round'],
            evaluation_seed_manifest={r['id']:saved['solver_seed'] for r in data['train']})
        for key in ('state_selection_seed','probe_restart_seeds','research_contract_version',
                    'research_prompt_version','trace_policy','probe_seed_derivation'):
            c.pop(key,None)
        c=freeze_research_contract(c)
        self.config=c;self.training_epoch=epoch['round'];self.rows=data['train']
        self.augmented=bool(data.get('training_protocol'))
        self.adapter=TaskAdapter(c['task'],data.get('training_protocol'))
        self.eval_hash=digest(canonical(dict(epoch=epoch['round'],data=data,
            seeds=c['evaluation_seed_manifest'],time=c['solver_time_limit_s'],workers=self.workers,
            environment=c.get('environment_manifest'),runner=file_hash(Path(__file__).with_name('runner.py')))))
        if epoch.get('evaluation_hash') and epoch['evaluation_hash']!=self.eval_hash:
            raise ValueError('restored epoch evaluation conditions differ')
        if c.get('common_start_probe'):
            from .research import ResearchTools
            self.research=ResearchTools(self)

    def candidates(self):
        candidates={c['candidate_id']:dict(c) for c in self.archive.of('candidate')}
        for c in candidates.values():
            history=[m for m in self.archive.of('measurement') if m['candidate_id']==c['candidate_id']]
            measured=[m for m in history if m.get('config_hash')==self.eval_hash]
            c['eligibility']=bool(measured) and all(m['valid'] for m in measured)
            c['fitness']=statistics.fmean(m['fitness'] for m in measured) if c['eligibility'] else None
            c['evaluation_epoch']=self.training_epoch
            c['fitness_evaluation_hash']=self.eval_hash if measured else None
            past=[m for m in history if m.get('config_hash')!=self.eval_hash]
            if past:
                c['historical_fitness']=past[-1]['fitness']
                c['historical_evaluation_hash']=past[-1].get('config_hash')
            if self.augmented and measured:
                c['training_group_scores']={g:statistics.fmean(m['training_group_scores'][g] for m in measured)
                    for g in ('original','supplement')} if c['eligibility'] else None
        return candidates

    def execution_failures(self,cid,records=None):
        if records is None:
            records=[r for r in self.archive.of('execution') if r['candidate_id']==cid and r['config_hash']==self.eval_hash]
        return super().execution_failures(cid,records)

    def phase(self):
        return 'round_complete' if self.completed_iterations()>len(self.archive.of('round_complete')) else 'algorithm'

    def action_context(self):
        context=super().action_context()
        if context is None: return None
        actions=context['allowed_actions']
        if self.phase()=='round_complete':
            actions=[a for a in actions if a!='revise']+['advance_round']
            context['integration_options']=[];context['integration_opportunities']=[]
        remeasurable=[cid for cid,c in self.candidates().items() if c['fitness_evaluation_hash'] is None]
        if remeasurable: actions.append('remeasure_candidate')
        context.update(allowed_actions=actions,bootstrap_parent_ids=[],
            revise_rule='Only candidates measured under the current evaluation_hash can be revised. Use remeasure_candidate on an archived candidate before selecting it on this dataset.',
            evaluation_epoch=self.training_epoch,evaluation_hash=self.eval_hash,
            remeasurable_candidate_ids=remeasurable)
        return context

    def visible_event(self,event):
        view=super().visible_event(event)
        measured_hash=event.get('tested_conditions',{}).get('config_hash')
        view.update(evaluation_hash=measured_hash,current_dataset=measured_hash==self.eval_hash)
        return view

    def state(self,parent_id=None):
        best=self.best()
        value=super().state(parent_id or (best['candidate_id'] if best else None))
        value['alternating']=dict(round=self.training_epoch,total_rounds=self.config['max_iterations'],
            phase=self.phase(),evaluation_hash=self.eval_hash,
            completed_rounds=len(self.archive.of('round_complete')),
            historical_scores='Scores from other evaluation hashes are history, not current candidate fitness.')
        feedback=self.archive.of('round_feedback')
        if feedback:
            value['previous_round_feedback']=copy.deepcopy(feedback[-1])
        if self.phase()=='round_complete' and 'common_start' in value:
            value['common_start']['integration_options']=[]
        if self.config.get('integrated_feedback_v1'):
            value['complete_program_comparisons']=self.comparison_report()
            measured=value.get('active_parent')
            if measured and measured.get('eligibility'):
                from .integrated_data import data_report,render_data_report
                measurements=[m for m in self.archive.of('measurement') if m['candidate_id']==measured['candidate_id'] and m['config_hash']==self.eval_hash]
                if measurements:
                    ids=set(measurements[-1]['execution_ids'])
                    records=[r for r in self.archive.of('execution') if r['execution_id'] in ids]
                    supplemental=[r for r in self.rows if r.get('training_group')=='supplement']
                    original_ids={r['id'] for r in self.rows if r.get('training_group')!='supplement'}
                    value['current_training_report']=dict(candidate_id=measured['candidate_id'],
                        evaluation_hash=self.eval_hash,mean_gap_pct=measured['fitness'],
                        original_per_instance=[dict(id=r['instance_id'],gap_pct=r['fitness']) for r in records if r['instance_id'] in original_ids],
                        supplement=render_data_report(data_report(supplemental,records,measured['source_hash'],self.runner.limit,
                            next(iter(self.config['evaluation_seed_manifest'].values())))))
            value['evaluation_usage']=dict(actual_solver_runs=len(self.archive.of('execution_start')),
                reused_views=sum(bool(r.get('reused_observation')) for r in self.archive.of('execution')))
        return value

    def comparison_report(self):
        result=[]
        executions={r['execution_id']:r for r in self.archive.of('execution')}
        measurements=self.archive.of('measurement')
        for event in self.archive.of('contrast'):
            epoch=event['tested_conditions']['config_hash']
            parent=next((m for m in measurements if m['candidate_id']==event['parent_id'] and m['config_hash']==epoch),None)
            for child in event['variant_ids'].values():
                measured=next((m for m in measurements if m['candidate_id']==child and m['config_hash']==epoch),None)
                if not parent or not measured:continue
                groups={}
                for group in ('original','supplement'):
                    a=parent.get('training_group_scores',{}).get(group)
                    b=measured.get('training_group_scores',{}).get(group)
                    if a is not None and b is not None:
                        groups[group]=dict(parent_gap_pct=a,child_gap_pct=b,
                            result='better' if b<a-1e-9 else 'worse' if b>a+1e-9 else 'tied')
                result.append(dict(event_id=event['event_id'],comparison='complete programs on the same frozen training set',
                    parent=event['parent_id'],child=child,evaluation_hash=epoch,
                    parent_gap_pct=parent['fitness'],child_gap_pct=measured['fitness'],groups=groups))
        return result

    def measure_rows(self,cid,rows,stage):
        """Data-side access to the exact same Runner, traces and observation store."""
        if self.config.get('agent_task_runtime_v2') and (self.archive.root/'pause.request').exists():
            raise TaskPaused('pause requested before evaluation stage '+stage)
        if self.shared_measurements is None:raise ValueError('shared evaluation requires integrated_feedback_v1')
        completed={r['execution_id']:r for r in self.archive.of('execution')}
        jobs=[(cid,row,f'{stage}:{row["id"]}',None) for row in rows if f'{stage}:{row["id"]}' not in completed]
        if self.config.get('exploration_efficiency_v1'):
            # Reconstruct the full batch after a partial commit. SharedMeasurements
            # reuses completed observations under the full batch contract, rather
            # than silently measuring the remaining cases under fewer slots.
            jobs=[(cid,row,f'{stage}:{row["id"]}',None) for row in rows]
        if stage not in self.ledger.reservations():self.ledger.reserve(stage,len(jobs))
        try:
            if self.config.get('console_progress'):print(f'Evaluate {stage} | {len(rows)-len(jobs)}/{len(rows)} | workers={self.workers}',flush=True)
            done=len(rows)-len(jobs)
            for record in self.execute_batch(jobs,stage,purpose='data_calibration'):
                completed[record['execution_id']]=record;done+=1
                if self.config.get('console_progress') and (done%8==0 or done==len(rows)):
                    print(f'Evaluate {stage} | {done}/{len(rows)} | workers={self.workers}',flush=True)
        finally:self.ledger.release(stage)
        return [completed[f'{stage}:{row["id"]}'] for row in rows]

    def measure_current(self,cid,measurement_id):
        reservation=measurement_id
        if reservation not in self.ledger.reservations(): self.ledger.reserve(reservation,len(self.rows))
        try:self.evaluate({'P':cid},reservation,measurement_id)
        finally:self.ledger.release(reservation)

    def editor_context(self,cid,decision,count):
        value=super().editor_context(cid,decision,count)
        value['alternating']=dict(round=self.training_epoch,evaluation_hash=self.eval_hash,
            phase=self.phase(),total_rounds=self.config['max_iterations'])
        return value

    def tool(self,decision,step):
        if decision['action']!='remeasure_candidate': return super().tool(decision,step)
        if any(r['step']==step for r in self.archive.of('tool_result')):return
        target=decision['parent_or_target_id']
        self.measure_current(target,f'epoch_{self.training_epoch}:remeasure:{step}')
        candidate=self.candidates()[target]
        self.archive.add('historical_remeasurement',dict(round=self.training_epoch,candidate_id=target,
            evaluation_hash=self.eval_hash,fitness=candidate['fitness']))
        self.archive.add('tool_result',dict(step=step,action='remeasure_candidate',result=candidate))

    def prepare_round(self):
        round_number=len(self.archive.of('round_complete'))+1
        existing=next((e for e in self.archive.of('training_epoch') if e['round']==round_number),None)
        if existing:
            self.activate(existing)
            incumbent=existing['incumbent_id']
        else:
            finished=self.archive.of('round_complete')
            incumbent=finished[-1]['winner']['candidate_id'] if finished else 'warm_start'
            parent=self.candidates()[incumbent]
            if self.data_provider is None: raise RuntimeError('data round provider is required to advance search')
            if self.config.get('console_progress'):
                print(f'Round {round_number}'+(' | manual' if self.config.get('manual_rounds') else f'/{self.config["max_iterations"]}')+' | data design',flush=True)
            data=self.data_provider(self,round_number,parent)
            seeds=set(self.frozen_config['evaluation_seed_manifest'].values())
            if len(seeds)!=1:raise ValueError('alternating run requires one explicit solver seed')
            seed=next(iter(seeds))
            saved=dict(round=round_number,incumbent_id=incumbent,solver_seed=seed,data_manifest=data)
            folder=self.archive.root/'training_epochs';folder.mkdir(exist_ok=True)
            path=folder/f'{round_number:03d}.json'
            if path.exists():
                if canonical(load_json(path))!=canonical(saved):raise ValueError('pending epoch artifact differs')
            else:write_json(path,saved)
            existing=dict(round=round_number,incumbent_id=incumbent,artifact=receipt(path))
            self.activate(existing)
            existing['evaluation_hash']=self.eval_hash
            self.archive.add('training_epoch',copy.deepcopy(existing))
        self.measure_current(incumbent,f'epoch_{round_number}:incumbent')
        if not self.candidates()[incumbent]['eligibility']:
            raise RuntimeError('incumbent invalid on current dataset; no unmeasured fallback')

    def close_round(self):
        if any(r['round']==self.training_epoch for r in self.archive.of('round_complete')):return
        winner=self.best()
        if not winner:raise RuntimeError('no measured winner in current epoch')
        if self.config.get('measured_research_loop_v1'):
            self.save_round_feedback(winner)
        self.archive.add('round_complete',dict(round=self.training_epoch,evaluation_hash=self.eval_hash,
            winner=copy.deepcopy(winner),completed_iterations=self.completed_iterations()))
        if self.config.get('manual_rounds'):self.save_manual_round()
        if self.config.get('console_progress'):
            label=str(self.training_epoch) if self.config.get('manual_rounds') else f'{self.training_epoch}/{self.config["max_iterations"]}'
            print(f'Round {label} completed | current_gap={winner["fitness"]:.6f}%',flush=True)

    def save_manual_round(self):
        from .integrated_data import save_json
        from .task_tools import _atomic
        records=self.archive.of('round_complete')
        if not records:return
        record=records[-1];winner=record['winner']
        root=self.archive.root.parent/'rounds'/f'{record["round"]:03d}'
        summary=dict(round=record['round'],winner=winner,evaluation_hash=record['evaluation_hash'],
            completed_iterations=record['completed_iterations'],unit='percent',scope='training only; no TSPLIB evaluation')
        path=root/'round_summary.json'
        if path.exists() and load_json(path)!=summary:raise ValueError('saved manual round summary changed')
        if not path.exists():save_json(path,summary)
        source=self.archive.read_source(winner['source_hash']);path=root/'best_solver.py'
        if path.exists() and digest(path.read_text(encoding='utf-8'))!=winner['source_hash']:
            raise ValueError('saved manual round solver changed')
        if not path.exists():_atomic(path,source)

    def finish_manual(self):
        if not self.config.get('manual_rounds'):raise ValueError('explicit finish is for manual-round runs only')
        count=len(self.archive.of('round_complete'))
        requests=self.archive.of('manual_step_requested')
        if (not count or self.training_epoch!=count or self.completed_iterations()!=count or
                requests and requests[-1]['target_round']>count):
            raise ValueError('finish requires a completed round boundary; resume the pending round first')
        winner=self.best()
        if not winner:raise ValueError('no evaluated winner to freeze')
        identity=dict(rounds=count,source_hash=winner['source_hash'],evaluation_hash=self.eval_hash)
        prior=self.archive.of('manual_finished')
        if prior and prior[-1]!=identity:raise ValueError('frozen manual search identity changed')
        result_path=self.archive.root/'result.json'
        if prior and result_path.is_file():
            saved=load_json(result_path)
            if saved.get('status')=='finished':
                if canonical(saved.get('best'))!=canonical(winner) or digest((self.archive.root/'best_solver.py').read_text(encoding='utf-8'))!=winner['source_hash']:
                    raise ValueError('frozen manual output changed')
                return saved
        if not prior:self.archive.add('manual_finished',identity)
        self.save_manual_round()
        return self.export('finished','user finished manual search at completed round boundary')

    def save_round_feedback(self,winner):
        from .integrated_data import save_json
        if any(r['round']==self.training_epoch for r in self.archive.of('round_feedback')):return
        measurements=[m for m in self.archive.of('measurement') if m['config_hash']==self.eval_hash]
        ids={eid for m in measurements for eid in m['execution_ids']}
        records=[r for r in self.archive.of('execution') if r['execution_id'] in ids]
        comparisons=[r for r in self.comparison_report() if r['evaluation_hash']==self.eval_hash]
        header=dict(round=self.training_epoch,evaluation_hash=self.eval_hash,winner=copy.deepcopy(winner),
            comparisons=comparisons,measurement_count=len(measurements),
            failed_executions=sum(not r['valid'] for r in records),unit='percent')
        path=self.archive.root/'round_feedback'/f'{self.training_epoch:03d}.json'
        save_json(path,dict(header,measurements=measurements,executions=records,
            conditions=dict(solver_time_limit_s=self.runner.limit,workers=self.workers,
                evaluation_seed_manifest=self.config['evaluation_seed_manifest'])))
        self.archive.add('round_feedback',dict(header,artifact=receipt(path),
            artifact_id=f'round_feedback_{self.training_epoch:03d}.json'))

    def run(self,manual_next=False):
        manual=bool(self.config.get('manual_rounds'))
        if manual_next and not manual:raise ValueError('next round requires manual-round mode')
        if manual and self.archive.of('manual_finished'):raise ValueError('manual search is frozen for final testing; cannot continue')
        try:
            target=None
            if manual:
                count=len(self.archive.of('round_complete'));requests=self.archive.of('manual_step_requested')
                target=requests[-1]['target_round'] if requests else count
                if target<=count and (manual_next or not requests):
                    target=count+1
                    self.archive.add('manual_step_requested',dict(target_round=target))
            if not self.candidates():
                source=Path(self.frozen_config['warm_start_source']).read_text(encoding='utf-8-sig')
                check_source(source,require_research=self.config.get('common_start_probe',False))
                self.register('warm_start',source,initialization='explicit warm start')
            while len(self.archive.of('round_complete'))<(target if manual else self.config['max_iterations']):
                if self.config.get('agent_task_runtime_v2') and (self.archive.root/'pause.request').exists():
                    raise TaskPaused('pause requested before starting the next search action')
                round_number=len(self.archive.of('round_complete'))+1
                if self.config.get('measured_research_loop_v1') and self.phase()=='round_complete':
                    self.close_round();continue
                if self.training_epoch!=round_number or not self.best(): self.prepare_round()
                elif self.phase()=='algorithm': self.prepare_round()
                pending_close=any(a.get('round')==round_number and a['action']=='advance_round'
                                  for a in self.archive.of('action_complete'))
                if pending_close:self.close_round();continue
                step='step_'+str(len(self.archive.of('action_complete')))
                payload=self.state();payload['tool_observations']=self.archive.of('tool_result')[-2:]
                decision=self.request('planner',payload,step+':planner')
                self.update_experience(decision.get('previous_event_update',[]))
                action=decision['action']
                if action=='advance_round':
                    self.archive.add('action_complete',dict(step=step,action=action,round=round_number,
                        active_parent=self.best()['candidate_id']))
                    self.close_round();continue
                if action=='revise':self.package(decision,step)
                else:self.tool(decision,step)
                self.archive.add('action_complete',dict(step=step,action=action,round=round_number,
                    active_parent=self.best()['candidate_id'] if self.best() else None))
            if manual:
                self.save_manual_round()
                return self.export('paused','requested round completed; awaiting user next or finish command')
            return self.export('finished','configured alternating rounds completed')
        except (TaskPaused,TaskBlocked) as error:
            status='paused' if isinstance(error,TaskPaused) else 'blocked'
            self.archive.add(status,dict(reason=str(error),round=self.training_epoch))
            return self.export(status,str(error))
        except ReferencePending as error:
            self.archive.add('reference_waiting',dict(error=str(error),pending_round=len(self.archive.of('round_complete'))+1))
            return self.export('waiting_reference',str(error))
        except Exception as error:
            self.archive.add('failure',dict(error=type(error).__name__+': '+str(error),round=self.training_epoch))
            return self.export('failed',type(error).__name__+': '+str(error))


class DataRoundProvider:
    """Each stage is frozen separately; completed stages are recovered, never resent."""
    def __init__(self,settings,model):
        self.settings=copy.deepcopy(settings);self.model=model

    def __call__(self,engine,round_number,parent):
        if getattr(engine,'config',{}).get('integrated_feedback_v1'):
            from .integrated_data import IntegratedDataProvider
            return IntegratedDataProvider(self.settings,self.model)(engine,round_number,parent)
        from .data import bind_supplement
        starts=engine.archive.of('data_round_start')
        saved=next((r for r in starts if r['round']==round_number),None)
        path=engine.archive.root.parent/'rounds'/f'{round_number:03d}'/'data'
        if saved and saved['parent_hash']!=parent['source_hash']:raise ValueError('data-round incumbent changed')
        initial=engine.frozen_config['data_manifest']
        prior=[r for r in engine.archive.of('data_round_complete') if r['round']<round_number]
        previous=prior[-1]['manifest'] if prior else initial['supplement_manifest']
        if file_hash(previous['path'])!=previous['sha256']:raise ValueError('previous generator manifest changed')
        manifest=load_json(previous['path']);row=manifest['train'][0]
        for artifact in row['source'].values():
            if file_hash(artifact['path'])!=artifact['sha256']:raise ValueError('previous generator/parameters changed')
        generator=load_json(row['source']['parameters']['path'])
        source=Path(row['source']['generator']['path']).read_text(encoding='utf-8')
        previous_evaluation=manifest.get('evaluation_record')
        if previous_evaluation:
            if file_hash(previous_evaluation['path'])!=previous_evaluation['sha256']:raise ValueError('previous evaluation changed')
            previous_evaluation=Path(previous_evaluation['path'])
        else:previous_evaluation=Path(row['source']['generator']['path']).with_name('evaluation.json')
        observations=[load_json(previous_evaluation)] if previous_evaluation.is_file() else []
        evaluation=receipt(previous_evaluation) if previous_evaluation.is_file() else None
        expected=prior[-1].get('evaluation') if prior else engine.frozen_config.get('alternating_initial_evaluation')
        if expected and evaluation!=expected:raise ValueError('previous data evaluation changed')
        settings=copy.deepcopy(self.settings)
        seeds=set(engine.frozen_config['evaluation_seed_manifest'].values())
        if len(seeds)!=1:raise ValueError('one explicit calibration/search seed required')
        settings.update(output_dir=str(path),iterations=1,solver_source=parent['source_path'],
            solver_seed=next(iter(seeds)),
            calibration_research_trace=bool(engine.frozen_config.get('common_start_probe',False)),
            calibration_trace_policy=copy.deepcopy(engine.frozen_config.get('trace_policy',{})),
            initial_generator_source=source,initial_parameters=generator,prior_results=observations,
            global_round=round_number,global_total_rounds=engine.frozen_config.get('max_iterations'))
        if not saved:
            engine.archive.add('data_round_start',dict(round=round_number,parent_hash=parent['source_hash'],
                path=str(path),previous_manifest=previous,previous_evaluation=evaluation))
        if path.exists():
            if not (path/'result.json').is_file():
                # A complete saved proposal is replayed without another model call.
                # Unknown model responses and invalid generators remain errors.
                DataDesigner.recover(path)
            result=load_json(path/'result.json')
            if result.get('status')!='finished':raise ValueError('data stage did not finish')
            measured=load_json(path/'supplement_manifest.json')
            if measured['calibration_solver']['sha256']!=parent['source_hash']:
                raise ValueError('data stage used a different incumbent')
        else:
            designer=DataDesigner(settings,self.model);result=designer.run()
        binding=receipt(result['supplement_manifest'])
        base=copy.deepcopy(initial)
        base['train']=base['train'][:64]
        for key in ('training_protocol','supplement_manifest','audit'):base.pop(key,None)
        combined=bind_supplement(base,binding['path'])
        if not any(r['round']==round_number for r in engine.archive.of('data_round_complete')):
            output_manifest=load_json(binding['path'])
            measured_evaluation=output_manifest.get('evaluation_record')
            result_file=Path(measured_evaluation['path']) if measured_evaluation else Path(output_manifest['train'][0]['source']['generator']['path']).with_name('evaluation.json')
            if measured_evaluation and receipt(result_file)!=measured_evaluation:raise ValueError('completed data evaluation changed')
            engine.archive.add('data_round_complete',dict(round=round_number,parent_hash=parent['source_hash'],
                manifest=binding,evaluation=receipt(result_file) if result_file.is_file() else None))
        return combined
