"""Agent-selected training experiments; formal selection remains untouched."""
import copy,json,math,time,uuid

import numpy as np

from .instance_design import generate_coordinates
from .parallel import completed_tasks
from .runner import Runner
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
        if engine.config.get('exploration_efficiency_v1') and not pending and not errors and len(rows)==len(plan):
            try:
                from .data import audit_training,audit_instance_diversity,SUPPLEMENT_PROTOCOL
                original=[r for r in engine.frozen_config['data_manifest']['train'] if r.get('training_group')!='supplement']
                if engine.config.get('diagnostic_run'):
                    audit_instance_diversity([engine.adapter.load(r)['distance_matrix'] for r in original+rows])
                else:audit_training('tsp',original+rows,SUPPLEMENT_PROTOCOL)
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
            candidate_path=candidate_path,wallclock_s=time.monotonic()-started)
        _atomic(path,canonical(evidence))
        engine.archive.add('data_candidate_measurement',dict(artifact_path=str(path),candidate_identity=root.name,
            requested_case_indices=indices,measured_case_count=len(rows),pending_cases=pending,failed_cases=errors))
        result['artifact_id']='execution:'+path.name
        result['detail']=('The complete fixed plan was requested; completion and failure counts describe actual coverage. ' if whole_plan else
            'Subset measurements only; final submission still evaluates the complete fixed plan. ')+ 'Pending references have no gap. Full rules, plan, prepared rows and report are in artifact_id.'
        old=(payload.get('previous_results') or [{}])[0]
        expected=payload.get('calibration_conditions')
        if (engine.config.get('exploration_efficiency_v1') and result['ok'] and whole_plan and
                old.get('valid') and isinstance(old.get('mean_reference_gap_pct'),(int,float)) and expected and
                all(canonical(case['observation_contract'])==canonical(expected) for case in result['cases'])):
            from .recipes import prefer_new
            if prefer_new(old['mean_reference_gap_pct'],result['mean_reference_gap_pct']):
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

    def run(self,source_id,instance_ids=None,generator_id=None,seeds=None,parameters=None,n_cities=None,time_limit_s=None,trace=None):
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
            seeds=seeds,parameters=parameters,n_cities=n_cities),input_cases=[],cases=[],status='preparing')
        def admit():
            _atomic(path,canonical(evidence))
            engine.archive.add('experimental_start',dict(experiment_id=experiment_id,artifact_path=str(path)))
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
                cases.append(dict(id=key,instance=instance,seed=row['seed']))
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
        conditions.update(requested_cases=len(cases),active_parallel_cases=min(engine.workers,len(cases)))
        runner=Runner(seconds,engine.runner.memory_mb)
        evidence.update(generator_receipt=generator_receipt,input_cases=cases,status='started')
        _atomic(path,canonical(evidence))
        def execute(case):
            def launched():engine.archive.add('experimental_case_start',dict(experiment_id=experiment_id,id=case['id'],seed=case['seed']))
            result=runner.case(source,ProbeAdapter(engine.adapter),case['instance'],case['seed'],research_request=research_request,on_started=launched)
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
