"""Native investigation tasks attached to the existing CEF role boundaries."""
import copy
import hashlib
import json
import uuid
import time
from pathlib import Path

import numpy as np

from .instance_design import generate_coordinates
from .patch import check_source, compose
from .protocol import validate_planner
from .recipes import normalize_rules, dispatch
from .store import canonical
from .task_tools import TaskTools
from .tool_session import ToolSession



def binding(value):
    return dict(type='text' if isinstance(value,str) else 'json',value=value)


def material_shape(value):
    if isinstance(value,list):return dict(type='array',length=len(value),item_shape=material_shape(value[0]) if value else None)
    if isinstance(value,dict):return dict(type='object',fields={str(k):type(v).__name__ for k,v in value.items()})
    return dict(type=type(value).__name__)


def coordinate_receipt(root,item,n_cities):
    identity=dict(plan=item,n_cities=n_cities)
    key=hashlib.sha256(canonical(identity).encode()).hexdigest()[:24]
    return Path(root)/'generated'/(key+'.json'),identity


def _training_materials(engine):
    cases=[]
    for row in engine.rows:
        if row.get('split')!='train':raise ValueError('task materials must be training only')
        instance=engine.adapter.load(row)
        if row.get('path'):
            with np.load(row['path'],allow_pickle=False) as z:
                coordinates=z['coordinates'].tolist() if 'coordinates' in z else None
        else:coordinates=None
        item=dict(id=row['id'],solver_seed=engine.config['evaluation_seed_manifest'].get(row['id']),seed=engine.config['evaluation_seed_manifest'].get(row['id']),
                  training_group=row.get('training_group','original'),reference=row.get('reference',instance.get('reference')))
        if engine.config.get('agent_task_runtime_v2'):
            item.update(instance=engine.adapter.solver_input(instance),coordinates=coordinates)
        elif coordinates is not None:item['coordinates']=coordinates
        else:item['instance']=instance
        if row.get('generator_seed') is not None:item['generator_seed']=row['generator_seed']
        cases.append(item)
    from .research_progress import progress_view
    progress=progress_view(engine.archive)
    records=[{k:v for k,v in row.items() if k not in ('isolation',)}
             for row in engine.archive.of('execution') if row.get('config_hash')==engine.eval_hash]
    # One actual observation may have several reuse views. Keep distinct per-case
    # records by observation ID/source candidate, rather than inflating evidence.
    unique={str(r.get('observation_id',r['execution_id']))+':'+r['candidate_id']:r for r in records}
    materials={'training_cases.json':binding(cases),'training_records.json':binding(list(unique.values())),
            'evaluation_contract.json':binding(dict(
                fitness_metric=engine.adapter.fitness_metric,current_evaluation_hash=engine.eval_hash,
                field_definitions=dict(config_hash='Evaluation conditions and data epoch identity; not program identity.',
                    candidate_id='Program candidate identity; source_hash identifies its code.',
                    observation_id='One recorded execution observation; reused views are not independent samples.',
                    training_group='Original or supplemental instance group.'),
                comparison='Formal selection compares eligible complete programs measured in the same current data epoch.'))}
    for feedback in engine.archive.of('round_feedback')[-2:]:
        artifact=feedback['artifact']
        if hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest()!=artifact['sha256']:
            raise RuntimeError('round feedback evidence changed')
        materials[feedback['artifact_id']]=binding(json.loads(Path(artifact['path']).read_text(encoding='utf-8')))
    if engine.config.get('agent_task_runtime_v2'):
        for evidence in engine.archive.of('agent_tool_evidence'):
            path=Path(evidence['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=evidence['sha256']:
                raise RuntimeError('immutable tool evidence changed: '+evidence['evidence_id'])
            materials[evidence['evidence_id']]=binding(json.loads(path.read_text(encoding='utf-8')))
    from .research_progress import data_observations
    materials['data_observations.json']=binding(data_observations(cases,list(unique.values())))
    materials['research_progress.json']=binding(progress)
    return materials


def _run(engine,model,role,payload,root,artifacts,submit,summary,settings=None,data_context=None):
    settings=settings or engine.config
    v2=bool(engine.config.get('agent_task_runtime_v2'))
    efficient=bool(engine.config.get('exploration_efficiency_v1'))
    frozen=copy.deepcopy(artifacts)
    task_payload=copy.deepcopy(payload)
    if v2 and role=='editor':
        task_payload.pop('parent_code',None)
        task_payload['parent_source_artifact']='parent.py'
    if role!='editor':
        for key in ('calibration_solver','current_program'):task_payload.pop(key,None)
        if 'current_rules' in task_payload:
            task_payload['current_rules']=[dict({k:v for k,v in rule.items() if k!='source'},source_artifact=f'generator_{i}.py')
                                          for i,rule in enumerate(task_payload['current_rules'])]
    frozen['task.json']=binding(task_payload)
    if efficient:
        from .task_worker import capability_manifest
        frozen['analysis_contract.json']=binding(capability_manifest())
    if v2:
        from .task_tools import _atomic
        bound=Path(root)/'bound_materials.json'
        if bound.exists():
            saved=json.loads(bound.read_text(encoding='utf-8'))
            if saved['task.json']!=frozen['task.json']:raise RuntimeError('original task payload changed')
            frozen=saved
        else:_atomic(bound,canonical(frozen))
    index=[dict(id=k,type=v['type'],characters=len(v['value'] if isinstance(v['value'],str) else canonical(v['value'])),
                sha256=hashlib.sha256((v['value'] if v['type']=='text' else canonical(v['value'])).encode()).hexdigest())
           for k,v in sorted(frozen.items())]
    if efficient:
        for item in index:item['value_shape']=material_shape(frozen[item['id']]['value'])
    task=TaskTools(root,frozen,lambda path:submit(path,task) if efficient and data_context is not None else submit(path),
        timeout_s=settings.get('tool_timeout_s',10),runtime_v2=v2)
    if v2:
        from .task_experiments import TaskExperiments
        experiments=TaskExperiments(engine,task,data_context=data_context)
        task.experiment=experiments.run
        task.enable_research(experiments)
        if data_context is not None:
            task.enable_data_measurement(experiments.measure_data_candidate)
        if efficient:
            task.enable_efficiency(role)
            task.experiment_batch=experiments.run_batch
    maximum=settings.get('tool_max_turns') if v2 else settings.get('tool_max_turns',32)
    initial=dict(role=role,task=summary,artifacts=index,max_model_turns=maximum)
    if data_context is not None:
        initial['measured_starting_point']=dict(current_reports=payload.get('previous_results',[]),coordinate_source='training_cases.json',joined_measurements='data_observations.json',generator_sources=[k for k in frozen if k.startswith('generator_')])
    from .task_prompts import research_system
    system=research_system(role)
    if v2:
        initial['capabilities']=dict(lightweight_script_seconds=task.timeout_s,
            solver_experiment_seconds=engine.runner.limit,model_sampling=model.config.get('model_sampling'),
            termination='Validated deliverable or explicit external blocker')
    if engine.config.get('console_progress'):
        print(f'{role.title()} task | runtime={"v2" if v2 else "v1"}',flush=True)
    def execute_tool(name,args):
        tracked=v2 and name in ('run_python','run_solver_experiment','measure_data_candidate','run_experiment_batch','compare_sources','continue_comparison','update_research','reproduce_training_case')
        execution_id=uuid.uuid4().hex if tracked else None
        if tracked:engine.archive.add('agent_tool_execution_start',dict(execution_id=execution_id,task=str(root),tool=name))
        result=task.dispatch(name,args)
        if tracked:
            if name=='run_python' and 'wallclock_s' in result:
                engine.ledger.charge('analysis_seconds',result['wallclock_s'],task=str(root),tool=name,execution_id=execution_id)
            engine.archive.add('agent_tool_execution_complete',dict(execution_id=execution_id,task=str(root),tool=name))
        if v2 and name in ('run_python','run_solver_experiment','measure_data_candidate','run_experiment_batch','compare_sources','continue_comparison','update_research','reproduce_training_case'):
            evidence_id=result.get('execution_artifact_id') or result.get('artifact_id')
            if evidence_id:
                path=task.results/evidence_id.split(':',1)[1]
                entry=dict(evidence_id=evidence_id,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    task=str(root),tool=name,scope='Actual tool execution; no independent causal-component claim')
                prior=[r for r in engine.archive.of('agent_tool_evidence') if r['evidence_id']==evidence_id]
                if prior and prior[0]!=entry:raise RuntimeError('tool evidence identity changed')
                if not prior:engine.archive.add('agent_tool_evidence',entry)
                result=dict(result,evidence_id=evidence_id)
        proposal=result.pop('_goal_proposal',None)
        if efficient and data_context is not None and proposal is not None:
            completed=task._submit(content=canonical(proposal))
            if completed.get('accepted'):result['task_completion']=completed
            else:result['submission_error']=completed.get('error')
        return result
    session=ToolSession(root,model,task.schemas,execute_tool,role=role,system=system,payload=initial,
                        max_turns=maximum,
                        finalize_turns=0 if v2 else 2,runtime_v2=v2,
                        pause_check=(lambda:(engine.archive.root/'pause.request').exists()) if v2 else None,
                        context_window_tokens=engine.config.get('tool_context_window_tokens') if engine.config.get('measured_research_loop_v1') else None,
                        context_storage=task.store_context_result if engine.config.get('measured_research_loop_v1') else None,
                        completion_tools=['measure_data_candidate','run_experiment_batch'] if efficient and data_context is not None else None,
                        checkpoint_tools=['checkpoint_context','run_experiment_batch','update_research'] if efficient else None)
    restored=task.restore_completed_writes(session.archive.records)
    if restored:session.archive.add('workspace_restored',dict(files=restored))
    result=session.run()
    stats=dict(task=str(root),role=role,model_turns=len(session.archive.of('model_completed')),
               tool_calls=len(session.archive.of('tool_completed')),
               python_checks=sum(isinstance(r['call'].get('function'),dict) and r['call']['function'].get('name')=='run_python'
                                 for r in session.archive.of('tool_completed')))
    if not any(r.get('task')==stats['task'] for r in engine.archive.of('agent_task_complete')):
        engine.archive.add('agent_task_complete',stats)
    return result



def run_algorithm_task(engine,model,role,payload,call_id):
    if role not in ('planner','editor'):raise ValueError('unknown algorithm task role')
    artifacts=_training_materials(engine)
    source=payload.get('parent_code',payload.get('current_program'))
    if source is not None:artifacts['parent.py']=binding(source)
    handoff=payload.get('decision',{}).get('candidate_source_id')
    if handoff:
        if not isinstance(handoff,str) or not handoff.startswith('source:'):raise ValueError('candidate_source_id must be immutable source ID')
        artifacts['proposed_candidate.py']=binding(engine.archive.read_source(handoff.split(':',1)[1]))
    # Material content is accessible through tools, not repeated in every prompt.
    summary={k:payload[k] for k in ('alternating','active_parent','global_best','parent_id','parent_hash',
        'edit_count','decision','training_group_scores','action_context','fitness_metric','previous_round_feedback') if k in payload}
    if handoff:summary['candidate_handoff']=dict(source_artifact='proposed_candidate.py',source_id=handoff,remaining_questions=payload.get('decision',{}).get('open_questions',[]))
    summary['goal']='Select next algorithm-design action' if role=='planner' else 'Implement the submitted two-intent complete PAB modification'
    root=engine.archive.root/'agent_tasks'/call_id.replace(':','_')
    def submit(path):
        value=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value,dict):raise ValueError('submission must be a JSON object')
        if role=='planner':
            if value.get('prepared_package') is not None:
                parent=engine.candidates().get(value.get('parent_or_target_id'))
                package=value['prepared_package']
                if not parent or package.get('parent_id')!=value['parent_or_target_id'] or package.get('parent_hash')!=parent['source_hash']:raise ValueError('prepared package parent mismatch')
                if len(package.get('edits',[]))!=value.get('edit_count'):raise ValueError('prepared package edit count mismatch')
                if not package.get('issue') or not isinstance(package.get('evidence_ids'),list) or not set(package['evidence_ids'])<=engine.known_evidence_ids():raise ValueError('prepared package evidence invalid')
                check_source(compose(engine.archive.read_source(parent['source_hash']),package)['PAB'],require_research=bool(engine.research))
            if value.get('candidate_source_id'):
                check_source(engine.archive.read_source(value['candidate_source_id'].split(':',1)[1]),require_research=bool(engine.research))
            validate_planner(value,engine.config,engine.action_context())
            if not set(value.get('evidence_ids',[]))<=engine.known_evidence_ids():raise ValueError('unknown evidence IDs')
            for update in value.get('previous_event_update',[]):
                if update.get('card_id') not in {r['card_id'] for r in engine.experiences.cards()}:
                    raise ValueError('unknown experience card')
                if not set(update.get('evidence_ids',[]))<=engine.known_evidence_ids():raise ValueError('unknown card evidence')
        else:
            if not isinstance(value.get('package_id'),str) or not value['package_id'].strip():
                raise ValueError('package_id must be a nonempty string')
            if value.get('parent_id')!=payload['parent_id'] or value.get('parent_hash')!=payload['parent_hash']:
                raise ValueError('editor changed fixed parent')
            edits=value.get('edits')
            if not isinstance(edits,list) or len(edits)!=payload['edit_count'] or not all(isinstance(e,dict) for e in edits):
                raise ValueError('editor changed edit count or edit object format')
            if [e.get('edit_id') for e in edits]!=['A','B']:raise ValueError('complete PAB requires edits A and B')
            for edit in edits:
                replacements=edit.get('exact_replacements')
                if not isinstance(replacements,list) or not replacements or not all(isinstance(r,dict) and
                    isinstance(r.get('old'),str) and isinstance(r.get('new'),str) for r in replacements):
                    raise ValueError('each edit requires string old/new replacement objects')
            if not value.get('issue') or not isinstance(value.get('evidence_ids'),list):raise ValueError('package metadata missing')
            if not set(value['evidence_ids'])<=engine.known_evidence_ids():raise ValueError('unknown package evidence')
            variants=compose(payload['parent_code'],value)
            check_source(variants['PAB'],require_research=bool(engine.research))
        return dict(accepted=True,value=value)
    return _run(engine,model,role,payload,root,artifacts,submit,summary)


def run_data_task(engine,model,payload,root,settings):
    from .integrated_data import save_json
    root=Path(root)
    artifacts=_training_materials(engine)
    artifacts['parent.py']=binding(payload['calibration_solver'])
    for i,rule in enumerate(payload.get('current_rules',[])):
        artifacts[f'generator_{i}.py']=binding(rule['source'])
    summary={k:payload[k] for k in ('n_cities','instances','instance_seeds','global_round',
        'global_total_rounds','previous_results','recent_data_selections','objective','solver_seconds','solver_seed','previous_round_feedback') if k in payload}
    summary['current_rules']=[{k:v for k,v in r.items() if k!='source'} for r in payload.get('current_rules',[])]
    summary['goal']=payload.get('task','Deliver a complete reproducible supplemental-generator proposal; investigate actual measured weaknesses as useful.')
    def submit(path,tools=None):
        value=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value,dict):raise ValueError('proposal must be a JSON object')
        if engine.config.get('exploration_efficiency_v1'):
            if value.get('action')=='keep_current':
                from .integrated_data import validate_keep_current
                validate_keep_current(value,engine)
                return dict(accepted=True,value=value)
            value=tools.resolve_data_proposal(value)
        rules=normalize_rules(value)
        plan=dispatch(rules,payload['instance_seeds'])
        versions={r['version']:r for r in rules}
        prepared_root=None
        if engine.config.get('measured_research_loop_v1'):
            from .integrated_data import candidate_preparation
            _,_,prepared_root=candidate_preparation(root,value,payload['instance_seeds'],settings)
        def check_measured_points(item,points):
            prepared=prepared_root/'cases'/item['id']/'prepared.json' if prepared_root else None
            if prepared is not None and prepared.is_file():
                row=json.loads(prepared.read_text(encoding='utf-8'))['row']
                if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('previously measured instance changed')
                with np.load(row['path'],allow_pickle=False) as z:
                    if not np.array_equal(z['coordinates'],np.asarray(points)):
                        raise ValueError('generator differs from previously measured coordinates for '+item['id'])
        matrices=[engine.adapter.load(row)['distance_matrix'] for row in engine.frozen_config['data_manifest']['train']
                  if row.get('training_group')!='supplement'] if engine.config.get('exploration_efficiency_v1') else []
        for index,item in enumerate(plan):
            rule=versions[item['rule_version']]
            generated,identity=coordinate_receipt(root,item,settings['n_cities'])
            generated.parent.mkdir(parents=True,exist_ok=True)
            if generated.exists():
                old=json.loads(generated.read_text(encoding='utf-8'))
                if old['identity']==identity:
                    check_measured_points(item,old['coordinates'])
                    xy=np.asarray(old['coordinates']);matrices.append(np.linalg.norm(xy[:,None]-xy[None,:],axis=2))
                    continue
            started=time.monotonic();validation_id=uuid.uuid4().hex
            v2=engine.config.get('agent_task_runtime_v2')
            if v2:engine.archive.add('data_validation_start',dict(validation_id=validation_id,seed=item['instance_seed'],rule_version=item['rule_version']))
            error=None
            try:
                points=generate_coordinates(rule['source'],rule['parameters'],settings['n_cities'],
                                            item['instance_seed'],settings['generator_seconds'])
                replay=generate_coordinates(rule['source'],rule['parameters'],settings['n_cities'],
                                            item['instance_seed'],settings['generator_seconds'])
                if points!=replay:raise ValueError('generator not reproducible for '+item['id'])
            except BaseException as exc:
                error=type(exc).__name__+': '+str(exc)
                raise
            finally:
                if v2:
                    elapsed=time.monotonic()-started
                    engine.ledger.charge('analysis_seconds',elapsed,phase='generator_submission_validation',validation_id=validation_id)
                    engine.archive.add('data_validation_result',dict(validation_id=validation_id,error=error,wallclock_s=elapsed))
            check_measured_points(item,points)
            xy=np.asarray(points);matrices.append(np.linalg.norm(xy[:,None]-xy[None,:],axis=2))
            save_json(generated,dict(identity=identity,coordinates=points))
            if engine.config.get('console_progress') and ((index+1)%8==0 or index+1==len(plan)):
                print(f'Validate generator | {index+1}/{len(plan)}',flush=True)
        if engine.config.get('exploration_efficiency_v1'):
            from .data import audit_instance_diversity
            audit_instance_diversity(matrices)
        return dict(accepted=True,value=value,validated_cases=len(plan))
    return _run(engine,model,'data_designer',payload,root/'agent_task',artifacts,submit,summary,
                dict(settings,tool_max_turns=engine.config.get('tool_max_turns',None if engine.config.get('agent_task_runtime_v2') else 32),
                     tool_timeout_s=engine.config.get('tool_timeout_s',10)),
                data_context=dict(root=root,settings=settings,payload=payload,
                    parent_id=next(cid for cid,c in engine.candidates().items() if c['source_hash']==payload['calibration_source_hash']))
                    if engine.config.get('measured_research_loop_v1') else None)
