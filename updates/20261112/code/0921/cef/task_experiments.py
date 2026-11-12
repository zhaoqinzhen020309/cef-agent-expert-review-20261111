"""Agent-selected training experiments; formal selection remains untouched."""
import copy,json,math,time,uuid
from pathlib import Path

import numpy as np

from .instance_design import generate_coordinates
from .parallel import completed_tasks
from .runner import Runner
from .reference import ReferenceBackendFailure
from .store import canonical
from .task_tools import _atomic


class ProbeAdapter:
    def __init__(self,adapter):self.adapter=adapter
    def solver_input(self,instance):return self.adapter.solver_input(instance)
    def objective(self,instance,solution):return self.adapter.objective(instance,solution)
    def score(self,instance,solution):
        objective=self.objective(instance,solution)
        if 'reference' not in instance:return objective,None
        return self.adapter.score(instance,solution)


class TaskExperiments:
    def __init__(self,engine,tools,data_context=None):
        self.engine,self.tools=engine,tools
        self.data_context=data_context
        tools.source_loader=engine.archive.read_source

    def measure_data_candidate(self,candidate_path,case_indices):
        """Measure selected members of exactly the eventual formal data plan."""
        from .integrated_data import IntegratedDataProvider,candidate_preparation,data_report
        from .reference_jobs import ReferencePending
        if self.data_context is None:raise ValueError('candidate data measurement is available only in a data-design task')
        context=self.data_context;payload=context['payload'];settings=copy.deepcopy(context['settings'])
        seeds=payload['instance_seeds']
        if (not isinstance(case_indices,list) or not case_indices or
                any(type(i) is not int or not 0<=i<len(seeds) for i in case_indices) or
                len(set(case_indices))!=len(case_indices)):
            raise ValueError(f'case_indices must be unique integer indices from 0 through {len(seeds)-1}')
        proposal=json.loads(self.tools._path(candidate_path).read_text(encoding='utf-8'))
        if self.engine.config.get('exploration_efficiency_v1'):
            proposal=self.tools.resolve_data_proposal(proposal)
        if not isinstance(proposal,dict):raise ValueError('candidate proposal must be a JSON object containing rules')
        rules,plan,root=candidate_preparation(context['root'],proposal,seeds,settings)
        versions={r['version']:r for r in rules}
        for rule in rules:self.engine.archive.source(rule['source'])
        engine=self.engine;source_hash=payload['calibration_source_hash']
        parent=engine.candidates().get(context['parent_id'])
        if not parent or parent['source_hash']!=source_hash:raise ValueError('frozen calibration program changed')
        # One existing reference work quantum per call. A pending answer remains
        # resumable; it does not terminate the Agent or substitute an estimate.
        settings['reference_work_slices']=1
        provider=IntegratedDataProvider(settings,None)
        started=time.monotonic();rows=[];pending=[];errors=[]
        indices=sorted(case_indices)
        def prepare(index):
            item=plan[index];case_started=time.monotonic()
            try:return dict(row=provider.prepare_case(root,item,versions[item['rule_version']],settings))
            except ReferenceBackendFailure as error:return dict(error=dict(id=item['id'],index=index,phase='reference_backend',error_type=error.error_type,error=str(error)))
            except ReferencePending as error:return dict(pending=dict(id=item['id'],index=index,error=str(error)))
            except (ValueError,TypeError,KeyError,SyntaxError) as error:
                return dict(error=dict(id=item['id'],index=index,error=type(error).__name__+': '+str(error)))
            finally:engine.ledger.charge('analysis_seconds',time.monotonic()-case_started,phase='data_candidate_reference',instance_id=item['id'])
        for index,value in completed_tasks(indices,prepare,settings.get('reference_workers',engine.workers)):
            if 'row' in value:rows.append(value['row'])
            elif 'pending' in value:pending.append(value['pending'])
            else:errors.append(value['error'])
        rows.sort(key=lambda r:r['id']);pending.sort(key=lambda r:r['index']);errors.sort(key=lambda r:r['index'])
        # Finish every reference computation before running timed solver work.
        from .patch import digest
        batch_identity=dict(indices=indices,ready_ids=[r['id'] for r in rows],source_hash=source_hash,
            contract=engine.shared_measurements.contract(engine),batch_parallel_slots=min(engine.workers,len(rows)))
        stage='data_probe:'+root.name+':'+digest(canonical(batch_identity))
        records=engine.measure_rows(context['parent_id'],rows,stage) if rows else []
        errors.extend(dict(id=r['instance_id'],phase='calibration_solver',error=r.get('error','invalid solver execution'))
                      for r in records if not r['valid'])
        report=data_report(rows,records,source_hash,payload['solver_seconds'],payload['solver_seed']) if rows else None
        observations={r['key']:r for r in engine.shared_measurements.archive.of('observation')}
        if report:
            for case in report['per_instance']:
                case['observation_contract']=observations[case['observation_id']]['identity']['contract']
        from .data_coverage import accumulate
        expected=payload.get('calibration_conditions')
        prior=[]
        for event in engine.archive.of('data_candidate_measurement'):
            if event.get('candidate_identity')==root.name:
                prior.append(json.loads(Path(event['artifact_path']).read_text(encoding='utf-8')))
        current=dict(candidate_identity=root.name,calibration_source_hash=source_hash,plan=plan,rules=rules,
            prepared_rows=rows,executions=records,cases=report['per_instance'] if report else [])
        coverage=accumulate(prior+[current],candidate_identity=root.name,plan=plan,rules=rules,
            calibration_source_hash=source_hash,contract=expected)
        complete_report=data_report(coverage['rows'],coverage['executions'],source_hash,payload['solver_seconds'],payload['solver_seed']) if coverage['complete'] else None
        if engine.config.get('exploration_efficiency_v1') and not pending and not errors and coverage['complete']:
            try:
                from .data import audit_training,audit_instance_diversity,SUPPLEMENT_PROTOCOL
                original=[r for r in engine.frozen_config['data_manifest']['train'] if r.get('training_group')!='supplement']
                if engine.config.get('diagnostic_run'):
                    audit_instance_diversity([engine.adapter.load(r)['distance_matrix'] for r in original+coverage['rows']])
                else:audit_training('tsp',original+coverage['rows'],SUPPLEMENT_PROTOCOL)
            except (ValueError,TypeError,KeyError) as error:
                errors.append(dict(phase='formal_collection_validation',error=type(error).__name__+': '+str(error)))
        whole_plan=indices==list(range(len(plan)))
        result=dict(ok=not pending and not errors and bool(report and report['valid']),
            scope='complete fixed formal instance plan' if whole_plan else 'selected subset of the fixed formal instance plan',requested_case_indices=indices,
            formal_case_count=len(plan),measured_case_count=len(rows),calibration_source_hash=source_hash,
            conditions=dict(solver_seconds=payload['solver_seconds'],solver_seed=payload['solver_seed'],
                workers=engine.workers,reference_time_slice_s=settings['reference_seconds'],
                active_parallel_cases=min(engine.workers,len(rows)),
                reference_gap_tolerance_pct=settings['reference_gap_pct']),
            mean_reference_gap_pct=report['mean_reference_gap_pct'] if report else None,
            cases=report['per_instance'] if report else [],pending_cases=pending,failed_cases=errors,
            candidate_identity=root.name,unit='percent')
        path=self.tools.results/(uuid.uuid4().hex+'.data_measurement.json')
        evidence=dict(result,prepared_rows=rows,report=report,executions=records,plan=plan,rules=rules,
            candidate_path=candidate_path,cumulative_report=complete_report,wallclock_s=time.monotonic()-started)
        _atomic(path,canonical(evidence))
        engine.archive.add('data_candidate_measurement',dict(artifact_path=str(path),candidate_identity=root.name,
            requested_case_indices=indices,measured_case_count=len(rows),pending_cases=pending,failed_cases=errors))
        result['artifact_id']='execution:'+path.name
        result['cumulative_coverage']=dict(complete=coverage['complete'],measured_case_count=coverage['covered'],formal_case_count=coverage['total'],missing_ids=coverage['missing_ids'],mean_reference_gap_pct=complete_report['mean_reference_gap_pct'] if complete_report else None)
        result['reference_failures_are_not_difficulty']=True
        result['detail']=('The complete fixed plan was requested; completion and failure counts describe actual coverage. ' if whole_plan else
            'Subset measurements only; final submission still evaluates the complete fixed plan. ')+ 'Pending references have no gap. Full rules, plan, prepared rows and report are in artifact_id.'
        old=(payload.get('previous_results') or [{}])[0]
        result['comparison_to_current_data']=dict(current_complete_mean_gap_pct=old.get('mean_reference_gap_pct'),candidate_measured_mean_gap_pct=result['mean_reference_gap_pct'],complete_candidate=whole_plan and result['ok'],note='Partial coverage cannot establish complete-set improvement; failed references leave difficulty unknown.')
        result['execution_health']=dict(reference_backend_failures=sum(e.get('phase')=='reference_backend' for e in errors),unfinished_references=len(pending),other_failures=sum(e.get('phase')!='reference_backend' for e in errors))
        expected=payload.get('calibration_conditions')
        if (engine.config.get('exploration_efficiency_v1') and result['ok'] and coverage['complete'] and complete_report and
                old.get('valid') and isinstance(old.get('mean_reference_gap_pct'),(int,float)) and expected and
                all(canonical(case['observation_contract'])==canonical(expected) for case in coverage['cases'])):
            from .recipes import prefer_new
            if prefer_new(old['mean_reference_gap_pct'],complete_report['mean_reference_gap_pct']):
                result['_goal_proposal']=proposal
        return result

    def run_batch(self,question,jobs,handoff=None):
        import inspect
        if not isinstance(question,str) or not question.strip():raise ValueError('question must describe the planned experiment')
        if not isinstance(jobs,list) or not jobs:raise ValueError('jobs must be a nonempty experiment list')
        if handoff is not None and (not isinstance(handoff,str) or not handoff.strip()):raise ValueError('handoff must be nonempty progress text')
        methods={'run_solver_experiment':self.run}
        if self.data_context is not None:methods['measure_data_candidate']=self.measure_data_candidate
        for job in jobs:
            if not isinstance(job,dict) or set(job)!={'tool','arguments'} or job['tool'] not in methods or not isinstance(job['arguments'],dict):
                raise ValueError('each job must name an available experiment tool and arguments object')
            inspect.signature(methods[job['tool']]).bind(**job['arguments'])
        records=[];goal=None
        for index,job in enumerate(jobs):
            try:result=methods[job['tool']](**job['arguments'])
            except (ValueError,TypeError,KeyError,SyntaxError) as error:
                result=dict(ok=False,error=type(error).__name__+': '+str(error))
            goal=result.pop('_goal_proposal',None)
            records.append(dict(index=index,tool=job['tool'],result=result))
            if goal is not None:break
        path=self.tools.results/(uuid.uuid4().hex+'.experiment_batch.json')
        evidence=dict(question=question,jobs=jobs,experiments=records,remaining_jobs=len(jobs)-len(records),
            scheduling='Jobs serial; each uses the same shared worker allowance. No nested parallel candidate pools.')
        _atomic(path,canonical(evidence))
        summaries=[]
        for record in records:
            value=record['result']
            preview={key:value[key] for key in ('ok','error','artifact_id','experiment_id','candidate_identity',
                'scope','conditions','requested_case_indices','formal_case_count','measured_case_count','mean_reference_gap_pct','unit') if key in value}
            preview['cases']=[{key:case[key] for key in ('id','valid','objective','reference_gap_pct','error') if key in case}
                              for case in value.get('cases',[])]
            preview['pending_cases']=value.get('pending_cases',[])
            preview['failed_cases']=value.get('failed_cases',[])
            summaries.append(dict(index=record['index'],tool=record['tool'],result=preview))
        result=dict(ok=all(r['result'].get('ok',False) for r in records),experiments=summaries,
            remaining_jobs=evidence['remaining_jobs'],artifact_id='execution:'+path.name)
        if goal is not None:result['_goal_proposal']=goal
        elif handoff:
            compact=[dict(index=r['index'],tool=r['tool'],ok=r['result'].get('ok'),
                mean_reference_gap_pct=r['result'].get('mean_reference_gap_pct'),artifact_id=r['result'].get('artifact_id'),
                error=r['result'].get('error')) for r in records]
            checkpoint=self.tools._checkpoint(canonical(dict(research_progress=handoff,question=question,
                actual_results=compact,full_evidence=result['artifact_id'])))
            result['context_checkpoint']=checkpoint['context_checkpoint']
        return result

    def run(self,source_id,instance_ids=None,generator_id=None,seeds=None,parameters=None,n_cities=None,time_limit_s=None,trace=None,solver_seed=None,initial_solutions=None):
        if self.data_context is not None and self.engine.config.get('exploration_efficiency_v1') and generator_id is not None:
            raise ValueError('new candidate data uses measure_data_candidate; this diagnostic accepts existing training instance_ids only')
        if (instance_ids is None)==(generator_id is None):raise ValueError('choose training instance_ids OR generator_id with seeds')
        if not isinstance(source_id,str):raise ValueError('source_id must name actual source')
        kind,source=self.tools._material(source_id)
        if kind!='text':raise ValueError('source_id must be Python text')
        engine=self.engine
        experiment_id=uuid.uuid4().hex
        source_hash=engine.archive.source(source)
        source_receipt=dict(source_hash=source_hash,artifact_id='source:'+source_hash,path=str(engine.archive.root/'sources'/(source_hash+'.py')))
        generator_receipt=None
        formal=engine.runner.limit
        seconds=formal if time_limit_s is None else time_limit_s
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<=formal:
            raise ValueError('time_limit_s must be positive and at most the frozen formal per-case limit')
        if parameters is not None and not isinstance(parameters,dict):raise ValueError('parameters must be an object')
        research=bool(engine.research) if trace is None else trace
        if type(research) is not bool:raise ValueError('trace must be boolean')
        conditions=dict(solver_time_limit_s=seconds,formal_solver_time_limit_s=formal,trace=research,
            formal_trace_enabled=bool(engine.research),
            memory_mb=engine.runner.memory_mb,workers=engine.workers,
            matched_formal_execution=seconds==formal and research==bool(engine.research),
            comparison_scope='Per-case time/memory/trace contract only; does not assert equal machine load or formal training distribution.',
            differences_from_formal=[name for name,different in (('solver_time_limit_s',seconds!=formal),('trace',research!=bool(engine.research))) if different],
            purpose='exploratory execution; not candidate ranking or component causal attribution',
            reference='known training reference when available; generated probes have no reference')
        path=self.tools.results/(experiment_id+'.experiment.json')
        start=time.monotonic()
        evidence=dict(experiment_id=experiment_id,source_id=source_id,source_hash=source_hash,source_receipt=source_receipt,
            generator_receipt=None,conditions=conditions,input_spec=dict(instance_ids=instance_ids,generator_id=generator_id,
            seeds=seeds,parameters=parameters,n_cities=n_cities,solver_seed=solver_seed,initial_solutions=initial_solutions),input_cases=[],cases=[],status='preparing')
        def admit():
            _atomic(path,canonical(evidence))
            engine.archive.add('experimental_start',dict(experiment_id=experiment_id,artifact_path=str(path)))
        if solver_seed is not None and (type(solver_seed) is not int or not 0<=solver_seed<2**32):raise ValueError('solver_seed must be uint32')
        if initial_solutions is not None and (not isinstance(initial_solutions,dict) or instance_ids is None or set(initial_solutions)!=set(instance_ids)):
            raise ValueError('initial_solutions must cover exactly the requested training IDs')
        if initial_solutions is not None and not research:raise ValueError('common-start requires trace=True')
        cases=[]
        known={r['id']:r for r in self.tools.artifacts['training_cases.json']['value']}
        if instance_ids is not None:
            if not isinstance(instance_ids,list) or not instance_ids or len(set(instance_ids))!=len(instance_ids):
                raise ValueError('nonempty unique training instance_ids required')
            if seeds is not None or n_cities is not None or parameters is not None:raise ValueError('generator parameters apply only to generated probes')
            for key in instance_ids:
                if key not in known:raise ValueError('unknown training instance ID: '+str(key))
                row=known[key];instance=dict(row['instance'])
                if row['reference'] is not None:instance['reference']=row['reference']
                cases.append(dict(id=key,instance=instance,seed=row['seed'] if solver_seed is None else solver_seed))
            admit()
        else:
            if engine.config['task']!='tsp':raise ValueError('coordinate generation is supported for TSP only')
            if not isinstance(seeds,list) or not seeds or any(type(s) is not int or not 0<=s<2**32 for s in seeds):
                raise ValueError('explicit uint32 generator seeds required')
            if len(set(seeds))!=len(seeds):raise ValueError('unique generator seeds required')
            kind,generator=self.tools._material(generator_id)
            if kind!='text':raise ValueError('generator_id must be Python text')
            first=next(iter(known.values()))['instance']['distance_matrix']
            size=len(first) if n_cities is None else n_cities
            training_sizes={len(row['instance']['distance_matrix']) for row in known.values()}
            if type(size) is not int or size not in training_sizes:
                raise ValueError('generated probe size must match declared training sizes: '+str(sorted(training_sizes)))
            generator_hash=engine.archive.source(generator)
            generator_receipt=dict(source_id=generator_id,source_hash=generator_hash,
                artifact_id='source:'+generator_hash,path=str(engine.archive.root/'sources'/(generator_hash+'.py')),parameters=parameters or {},n_cities=size,seeds=seeds)
            engine.archive.add('experimental_generator',dict(experiment_id=experiment_id,**generator_receipt))
            evidence['generator_receipt']=generator_receipt;admit()
            for seed in seeds:
                started=time.monotonic()
                engine.archive.add('experimental_generation_start',dict(experiment_id=experiment_id,seed=seed))
                error=None
                try:
                    coords=generate_coordinates(generator,parameters or {},size,seed,
                        engine.config.get('alternating_data_config',{}).get('generator_seconds',5))
                except BaseException as exc:
                    error=type(exc).__name__+': '+str(exc)
                    evidence.update(status='generation_failed',error=error,wallclock_s=time.monotonic()-start)
                    _atomic(path,canonical(evidence))
                    engine.archive.add('experimental_complete',dict(experiment_id=experiment_id,artifact_path=str(path),status='generation_failed'))
                    if not isinstance(exc,Exception):raise
                    return dict(ok=False,error=error,artifact_id='execution:'+path.name,conditions=conditions,cases=[])
                finally:
                    elapsed=time.monotonic()-started
                    engine.ledger.charge('analysis_seconds',elapsed,experiment_id=experiment_id,phase='probe_generation')
                    engine.archive.add('experimental_generation',dict(experiment_id=experiment_id,seed=seed,error=error,wallclock_s=elapsed))
                xy=np.asarray(coords,dtype=float);matrix=np.linalg.norm(xy[:,None]-xy[None,:],axis=2).tolist()
                cases.append(dict(id='generated_'+str(seed),instance=dict(distance_matrix=matrix),
                    seed=next(iter(engine.config['evaluation_seed_manifest'].values())),generator_seed=seed,coordinates=coords))
        research_request=dict(kind='trace',trace_policy=engine.config.get('trace_policy',{})) if research else None
        conditions.update(evaluation_hash=engine.eval_hash,solver_seed_override=solver_seed,common_start=initial_solutions is not None,requested_cases=len(cases),active_parallel_cases=min(engine.workers,len(cases)))
        if solver_seed is not None or initial_solutions is not None:
            conditions['matched_formal_execution']=False
            conditions['differences_from_formal']+=(["solver_seed_override"] if solver_seed is not None else [])+(["same_path_restart"] if initial_solutions is not None else [])
        runner=Runner(seconds,engine.runner.memory_mb)
        evidence.update(generator_receipt=generator_receipt,input_cases=cases,status='started')
        _atomic(path,canonical(evidence))
        def execute(case):
            def launched():engine.archive.add('experimental_case_start',dict(experiment_id=experiment_id,id=case['id'],seed=case['seed']))
            result=runner.case(source,ProbeAdapter(engine.adapter),case['instance'],case['seed'],research_request=(dict(research_request,kind='common_start',initial_solution=initial_solutions[case['id']]) if initial_solutions is not None else research_request),on_started=launched)
            record=dict(result,id=case['id'],generator_seed=case.get('generator_seed'),
                reference_gap_pct=result['fitness'] if 'reference' in case['instance'] and result['valid'] else None)
            engine.archive.add('experimental_execution',dict(experiment_id=experiment_id,source_hash=source_hash,conditions=conditions,**record))
            engine.ledger.charge('analysis_seconds',result.get('wallclock_s',0),experiment_id=experiment_id)
            return record
        for case,result in completed_tasks(cases,execute,engine.workers):
            evidence['cases'].append(result)
            evidence['status']='running';_atomic(path,canonical(evidence))
            if engine.config.get('console_progress'):
                print(f'Experiment | {len(evidence["cases"])}/{len(cases)} | seconds={seconds:g}',flush=True)
        evidence['status']='completed';evidence['wallclock_s']=time.monotonic()-start
        _atomic(path,canonical(evidence))
        engine.archive.add('experimental_complete',dict(experiment_id=experiment_id,artifact_path=str(path)))
        previews=[{key:value for key,value in case.items() if key not in ('solution','trace','isolation','error_location')}
                  for case in evidence['cases']]
        return dict(ok=True,experiment_id=experiment_id,conditions=conditions,cases=previews,
            artifact_id='execution:'+path.name,
            detail='Full inputs, solutions and traces are in artifact_id; run_python can analyze that JSON via input_artifact_ids.')

    def compare(self,reference_source_id,candidate_source_id,instance_ids,question,solver_seed=None,time_limit_s=None,initial_solutions=None):
        from .research_progress import paired_view
        if not isinstance(question,str) or not question.strip():raise ValueError('question required')
        ids=[]
        for identifier in (reference_source_id,candidate_source_id):
            kind,text=self.tools._material(identifier)
            if kind!='text':raise ValueError('comparison source must be text')
            ids.append('source:'+self.engine.archive.source(text))
        results=[]
        for source in ids:
            results.append(self.run(source,instance_ids=instance_ids,solver_seed=solver_seed,
                time_limit_s=time_limit_s,trace=True,initial_solutions=initial_solutions))
        report=paired_view(*results)
        from .research_progress import progress_view
        report['observed_repeats']=[c for c in progress_view(self.engine.archive)['candidates'] if c['source_id'] in ids]
        report.update(conditions=results[0]['conditions'],question=question,evidence_ids=[r['artifact_id'] for r in results],
            mode='same-path restart' if initial_solutions is not None else 'complete run',
            reference_source=ids[0],candidate_source=ids[1])
        path=self.tools.results/(uuid.uuid4().hex+'.comparison.json')
        _atomic(path,canonical(report))
        self.engine.archive.add('research_comparison',dict(report,artifact_id='execution:'+path.name))
        report['artifact_id']='execution:'+path.name
        return report

    def update_research(self,question,candidate_source_id,findings,open_questions,next_action,evidence_ids):
        from .research_progress import progress_view
        if next_action not in ('modify','measure','change_direction','submit','defer'):raise ValueError('unknown research action')
        for value in (question,findings):
            if not isinstance(value,str) or not value.strip():raise ValueError('nonempty question/findings required')
        if not isinstance(open_questions,list) or any(not isinstance(v,str) for v in open_questions):raise ValueError('open_questions must be strings')
        if not isinstance(evidence_ids,list) or not set(evidence_ids)<=self.engine.known_evidence_ids():raise ValueError('unknown research evidence')
        kind,source=self.tools._material(candidate_source_id)
        if kind!='text':raise ValueError('candidate must be source text')
        source_hash=self.engine.archive.source(source)
        self.engine.archive.add('research_update',dict(question=question,source_hash=source_hash,
            findings=findings,open_questions=open_questions,next_action=next_action,evidence_ids=evidence_ids))
        report=progress_view(self.engine.archive)
        if self.data_context is not None:
            from .data_coverage import candidate_state
            report['data_candidates']=candidate_state(self.engine.archive,self.data_context['payload']['calibration_source_hash'],self.data_context['payload'].get('calibration_conditions'))
        report['working_source_id']='source:'+source_hash
        path=self.tools.results/(uuid.uuid4().hex+'.research_progress.json');_atomic(path,canonical(report))
        return dict(ok=True,artifact_id='execution:'+path.name,progress=report,
            context_checkpoint=self.tools._checkpoint(canonical(dict(question=question,source_id='source:'+source_hash,
                findings=findings,open_questions=open_questions,next_action=next_action,evidence_ids=evidence_ids,measured_candidates=report.get('data_candidates'),working_source_id=report['working_source_id'])))['context_checkpoint'])

    def continue_comparison(self,comparison_id,instance_ids,question):
        kind,raw=self.tools._material(comparison_id)
        prior=json.loads(raw)
        if prior.get('conditions',{}).get('evaluation_hash')!=self.engine.eval_hash:raise ValueError('continue requires the same dataset and execution identity; start a new comparison')
        if prior.get('mode')!='complete run':raise ValueError('same-path experiments require explicit starts; use compare_sources')
        previous={r['id'] for r in prior['cases']}
        if not instance_ids or len(instance_ids)!=len(set(instance_ids)) or previous.intersection(instance_ids):
            raise ValueError('provide new unique case IDs; use compare_sources for intentional repeats')
        result=self.compare(prior['reference_source'],prior['candidate_source'],instance_ids,question,
            solver_seed=prior['conditions'].get('solver_seed_override'),time_limit_s=prior['conditions']['solver_time_limit_s'])
        result['cases']=prior['cases']+result['cases']
        result['paired_cases']=len(result['cases'])
        result['valid_pairs']=sum(r['valid'] for r in result['cases'])
        result['candidate_better']=sum(r['valid'] and r['candidate_minus_reference_objective'] < -1e-12 for r in result['cases'])
        result['candidate_worse']=sum(r['valid'] and r['candidate_minus_reference_objective'] > 1e-12 for r in result['cases'])
        result['evidence_ids']=prior['evidence_ids']+result['evidence_ids']
        result['continued_from']=comparison_id
        path=self.tools.results/(uuid.uuid4().hex+'.comparison.json');_atomic(path,canonical(result))
        result['artifact_id']='execution:'+path.name
        return result

    def reproduce_training_case(self,instance_id):
        if self.data_context is None:raise ValueError('reproduction requires the data-design task')
        row=next((r for r in self.engine.rows if r['id']==instance_id),None)
        if row is None or 'generator_seed' not in row:raise ValueError('instance has no generator provenance')
        rules=self.data_context['payload'].get('current_rules',[])
        rule=next((r for r in rules if r.get('version')==row.get('rule_version')),None)
        if rule is None:raise ValueError('matching original rule version is unavailable')
        material=next(r for r in self.tools.artifacts['training_cases.json']['value'] if r['id']==instance_id)
        points=generate_coordinates(rule['source'],rule['parameters'],len(material['coordinates']),row['generator_seed'],self.data_context['settings']['generator_seconds'])
        result=dict(instance_id=instance_id,generator_seed=row['generator_seed'],solver_seed=material['seed'],
            coordinates=points,identical_to_saved=points==material['coordinates'],rule_version=row['rule_version'])
        path=self.tools.results/(uuid.uuid4().hex+'.reproduction.json');_atomic(path,canonical(result))
        return dict(ok=True,artifact_id='execution:'+path.name,**result)
