"""Measured old/new data competition, with executable provenance and recovery."""
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
from concurrent.futures import CancelledError
import tempfile

import numpy as np

from .config import load_json, file_hash
from .data import receipt, SUPPLEMENT_PROTOCOL, bind_supplement, audit_training
from .instance_design import design_seeds, generate_coordinates
from .parallel import completed_tasks
from .patch import digest
from .recipes import normalize_rules, dispatch, summarize_rules, prefer_new
from .reference_jobs import independent_reference, ReferencePending, validate_reference
from .store import Archive, canonical


def save_json(path, value):
    """Publish once, atomically; recovery never overwrites an existing decision."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if canonical(load_json(path))!=canonical(value):raise ValueError('immutable stage artifact differs: '+str(path))
        return
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',dir=path.parent,delete=False) as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno());temporary=stream.name
    os.replace(temporary,path)


def save_source(path, source):
    path=Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8')!=source:raise ValueError('prepared generator changed')
        return
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',dir=path.parent,delete=False) as stream:
        stream.write(source);stream.flush();os.fsync(stream.fileno());temporary=stream.name
    os.replace(temporary,path)


def validate_rows(rows):
    for row in rows:
        for artifact in (row,row['reference_record'],*row['source'].values()):
            if file_hash(artifact['path'])!=artifact['sha256']:raise ValueError('prepared data artifact changed')


def rules_from_manifest(manifest):
    if manifest.get('rule_bundle'):
        entry=manifest['rule_bundle']
        if file_hash(entry['path'])!=entry['sha256']:raise ValueError('selected rule bundle changed')
        return normalize_rules(load_json(entry['path']))
    row=manifest['train'][0]
    validate_rows(manifest['train'])
    return normalize_rules(dict(rules=[dict(name='legacy mixed generator',
        source=Path(row['source']['generator']['path']).read_text(encoding='utf-8'),
        parameters=load_json(row['source']['parameters']['path']),weight=1)]))


def candidate_preparation(root,proposal,seeds,settings=None):
    """One frozen rule/weight/seed plan for exploration and full preparation."""
    rules=normalize_rules(proposal)
    plan=dispatch(rules,seeds)
    identity=dict(rules=rules,instance_seeds=seeds,dispatch=plan)
    if settings is not None:
        identity['instance_contract']=dict(n_cities=settings['n_cities'],
            reference_gap_pct=settings['reference_gap_pct'],distance_rule='original generated Euclidean matrix')
    folder=Path(root)/'candidate_measurements'/digest(canonical(identity))
    save_json(folder/'identity.json',identity)
    return rules,plan,folder


def validate_keep_current(value,engine):
    if not engine.config.get('exploration_efficiency_v1'):raise ValueError('keep_current requires the explicit efficiency workflow')
    if not isinstance(value.get('reason'),str) or not value['reason'].strip():raise ValueError('keep_current needs a reason')
    ids=value.get('evidence_ids')
    if not isinstance(ids,list) or any(not isinstance(v,str) for v in ids):raise ValueError('evidence_ids must be a string list')
    if not set(ids)<=engine.known_evidence_ids():raise ValueError('unknown keep_current evidence')
    if set(value)-{'action','reason','evidence_ids'}:raise ValueError('keep_current cannot also propose replacement rules')


def data_report(rows, records, source_hash, seconds, seed):
    by_id={r['instance_id']:r for r in records}
    cases=[]
    for row in rows:
        record=by_id[row['id']]
        reference=load_json(row['reference_record']['path'])
        cases.append(dict(id=row['id'],valid=record['valid'],
            execution_error=record.get('error'),
            rule_version=row.get('rule_version','legacy:'+row['source']['generator']['sha256']+':'+row['source']['parameters']['sha256']),
            rule_name=row.get('rule_name','legacy mixed generator'),
            instance_seed=row['generator_seed'],reference_gap_pct=record['fitness'],
            objective=record['objective'],solver_seconds=record['elapsed'],
            reference=row['reference'],reference_kind=row['reference_kind'],
            lower_bound=reference['lower_bound'],reference_bound_gap_pct=reference['bound_gap_pct'],
            reference_gap_tolerance_pct=row['reference_gap_tolerance_pct'],
            gap_to_numerical_lower_bound_pct=100*(record['objective']/reference['lower_bound']-1) if record['valid'] else None,
            observation_id=record['observation_id']))
    valid=all(r['valid'] for r in cases)
    return dict(valid=valid,calibration_source_hash=source_hash,solver_seed=seed,solver_time_limit_s=seconds,
        unit='percent',comparison='current complete solver against frozen per-instance unrounded references',
        direction='larger mean reference gap means harder for this solver',
        mean_reference_gap_pct=statistics.fmean(r['reference_gap_pct'] for r in cases) if valid else None,
        rule_results=summarize_rules(cases),per_instance=cases)


def render_data_report(report, example_limit=8):
    """Decision-focused projection of the canonical report, with full data on disk."""
    rules=sorted(report['rule_results'],key=lambda r:(-(r['mean_reference_gap_pct'] or 0.),r['rule_version']))
    positive=[r for r in rules if r['positive_count']>0]
    cases=sorted(report['per_instance'],key=lambda r:(-(r['reference_gap_pct'] or 0.),r['id']))
    return dict(calibration_source_hash=report['calibration_source_hash'],
        solver_seed=report['solver_seed'],solver_time_limit_s=report['solver_time_limit_s'],
        comparison=report['comparison'],unit=report['unit'],direction=report['direction'],
        valid=report['valid'],mean_reference_gap_pct=report['mean_reference_gap_pct'],
        measured_case_count=len(cases),executed_rule_count=len(rules),
        rules_with_positive_gap_count=len(positive),
        rules_with_positive_gap=[dict(rule_name=r['rule_name'],rule_version=r['rule_version'],
            positive_count=r['positive_count'],count=r['count'],mean_reference_gap_pct=r['mean_reference_gap_pct']) for r in positive],
        rule_results=rules,hardest_examples=cases[:example_limit],
        reference_accuracy=dict(max_bound_gap_pct=max((r.get('reference_bound_gap_pct',0.) for r in cases),default=0.),
            kind='numerically bounded reference tours; per-case bounds retained in the canonical report'))


class IntegratedDataProvider:
    def __init__(self, settings, model):
        self.settings=copy.deepcopy(settings);self.model=model

    def prepare_case(self, root, item, rule, settings):
        folder=root/'cases'/item['id'];folder.mkdir(parents=True,exist_ok=True)
        finished=folder/'prepared.json'
        if finished.is_file():
            saved=load_json(finished)
            if saved['plan']!=item:raise ValueError('prepared case dispatch changed')
            validate_rows([saved['row']])
            return saved['row']
        source=rule['source'];parameters=rule['parameters']
        generator=folder/'generator.py'
        save_source(generator,source)
        save_json(folder/'parameters.json',parameters)
        generated=None
        if settings.get('agent_tool_loop_v1'):
            from .tool_agents import coordinate_receipt
            generated,identity=coordinate_receipt(root,item,settings['n_cities'])
        if generated is not None and generated.is_file():
            checked=load_json(generated)
            if checked['identity']!=identity:raise ValueError('validated generator input changed')
            points=checked['coordinates']
            from .instance_design import validate_coordinates
            validate_coordinates(points,settings['n_cities'])
        else:
            points=generate_coordinates(source,parameters,settings['n_cities'],item['instance_seed'],settings['generator_seconds'])
            if points!=generate_coordinates(source,parameters,settings['n_cities'],item['instance_seed'],settings['generator_seconds']):
                raise ValueError('rule is not reproducible with its fixed seed')
        coordinates=np.asarray(points);matrix=np.linalg.norm(coordinates[:,None]-coordinates[None,:],axis=2)
        refpath=folder/'reference.json'
        if refpath.is_file():
            reference=validate_reference(matrix,load_json(refpath),settings['reference_gap_pct'])
        else:
            try:
                reference=independent_reference(matrix,settings['reference_seconds'],settings['reference_gap_pct'],
                    folder/'reference_work.json',settings.get('reference_work_slices'))
            except (ReferencePending,CancelledError):raise
            except (RuntimeError,OSError,subprocess.SubprocessError) as exc:
                if settings.get('agent_tool_loop_v1'):
                    raise ReferencePending('reference backend unavailable for '+item['id']+': '+str(exc)) from exc
                raise
            save_json(refpath,reference)
        path=folder/'instance.npz'
        if path.is_file():
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['coordinates'],coordinates):raise ValueError('prepared coordinates changed')
        else:
            with tempfile.NamedTemporaryFile(mode='wb',dir=folder,delete=False) as stream:
                np.savez(stream,coordinates=coordinates,distance_matrix=matrix)
                stream.flush();os.fsync(stream.fileno());temporary=stream.name
            os.replace(temporary,path)
        row=dict(id=item['id'],split='train',training_group='supplement',**receipt(path),
            reference=reference['objective'],reference_kind=reference['reference_kind'],reference_record=receipt(refpath),
            reference_gap_tolerance_pct=settings['reference_gap_pct'],generator_seed=item['instance_seed'],
            rule_version=item['rule_version'],rule_name=item['rule_name'],
            source=dict(generator=receipt(generator),parameters=receipt(folder/'parameters.json')))
        save_json(finished,dict(plan=item,row=row))
        return row

    def __call__(self, engine, round_number, parent):
        root=engine.archive.root.parent/'rounds'/f'{round_number:03d}'/'data'
        root.mkdir(parents=True,exist_ok=True);archive=Archive(root)
        initial=engine.frozen_config['data_manifest']
        previous=engine.archive.of('data_round_complete')
        previous=[r for r in previous if r['round']<round_number]
        binding=previous[-1]['manifest'] if previous else initial['supplement_manifest']
        if file_hash(binding['path'])!=binding['sha256']:raise ValueError('previous supplement manifest changed')
        old=load_json(binding['path']);old_rows=old['train']
        settings=copy.deepcopy(self.settings)
        seed=next(iter(engine.frozen_config['evaluation_seed_manifest'].values()))
        header=dict(round=round_number,parent_hash=parent['source_hash'],previous_manifest=binding,
                    settings=settings,protocol='integrated_feedback_v1')
        save_json(root/'stage.json',header)
        if not any(r['round']==round_number for r in engine.archive.of('data_round_start')):
            engine.archive.add('data_round_start',dict(round=round_number,parent_hash=parent['source_hash'],path=str(root),previous_manifest=binding))
        if (root/'result.json').is_file():
            result=load_json(root/'result.json')
            selected_binding=receipt(result['supplement_manifest'])
            if selected_binding!=result['manifest']:raise ValueError('completed selected manifest changed')
            selected=load_json(selected_binding['path']);validate_rows(selected['train'])
        else:
            # Includes original64 once. Later rounds normally reuse all old observations.
            original=initial['train'][:64] if not engine.config.get('diagnostic_run') else [r for r in initial['train'] if r.get('training_group')!='supplement']
            records=engine.measure_rows(parent['candidate_id'],original+old_rows,f'data_round_{round_number}:old')
            old_records=[r for r in records if r['instance_id'] in {v['id'] for v in old_rows}]
            old_report=data_report(old_rows,old_records,parent['source_hash'],engine.runner.limit,seed)
            if not old_report['valid']:raise RuntimeError('current champion invalid on retained data')
            save_json(root/'old_evaluation.json',old_report)
            rules=rules_from_manifest(old)
            payload=dict(task='Design supplemental Euclidean TSP generators for the current complete solver',
                interface_version='integrated_feedback_v1',global_round=round_number,
                global_total_rounds=engine.config['max_iterations'],
                n_cities=settings['n_cities'],instances=settings['instances'],
                instance_seeds=design_seeds(settings['data_seed'],settings['instances']),
                calibration_solver=engine.archive.read_source(parent['source_hash']),
                calibration_source_hash=parent['source_hash'],solver_seconds=engine.runner.limit,solver_seed=seed,
                current_rules=rules,previous_results=[render_data_report(old_report)],
                recent_data_selections=[r['selection'] for r in previous[-2:] if r.get('selection')],
                objective='Increase measured mean reference-gap percentage against this current solver.',
                generator_interface='rules: [{name, source, parameters, weight}]; source defines generate(n_cities, seed, parameters) -> coordinates',
                dispatch='The host chooses a rule using a separate fixed seeded stream and records its executed version for each case.')
            if engine.config.get('measured_research_loop_v1'):
                payload['previous_round_feedback']=engine.archive.of('round_feedback')[-1] if engine.archive.of('round_feedback') else None
            if engine.config.get('exploration_efficiency_v1'):
                observations={r['key']:r for r in engine.shared_measurements.archive.of('observation')}
                contracts=[observations[r['observation_id']]['identity']['contract'] for r in old_records]
                uniform=bool(contracts) and all(canonical(c)==canonical(contracts[0]) for c in contracts)
                payload['calibration_conditions']=contracts[0] if uniform else None
                payload['calibration_conditions_uniform']=uniform
            save_json(root/'model_input.json',payload)
            proposal_path=root/'proposal.json'
            if proposal_path.is_file():proposal=load_json(proposal_path)
            else:
                saved_responses=archive.of('data_model_response')
                if saved_responses:proposal=saved_responses[-1]['response']
                elif settings.get('agent_tool_loop_v1'):
                    from .tool_agents import run_data_task
                    if not archive.of('data_model_start'):
                        archive.add('data_model_start',dict(payload=payload,iteration=0))
                    else:payload=archive.of('data_model_start')[0]['payload']
                    proposal=run_data_task(engine,self.model,payload,root,settings)
                    archive.add('data_model_response',dict(iteration=0,response=proposal,tool_session=True))
                else:
                    starts=archive.of('data_model_start')
                    if starts:
                        responses=[]
                        model_archive=getattr(self.model,'archive',None)
                        for response in model_archive.of('provider_response') if model_archive else []:
                            if response.get('role')!='data_designer' or not response.get('parsed'):continue
                            submitted=json.loads(response['user'])
                            if all(submitted.get(k)==v for k,v in payload.items()):responses.append(response)
                        if not responses:
                            raise RuntimeError('unfinished Data model request; result unknown; no automatic redispatch')
                        proposal=responses[-1]['parsed']
                        archive.add('data_model_response',dict(iteration=0,response=proposal,recovered_provider_response=True))
                    else:
                        archive.add('data_model_start',dict(payload=payload,iteration=0))
                        if engine.config.get('console_progress'):print(f'Data Agent round {round_number} | generating rules',flush=True)
                        proposal=self.model.call('data_designer',payload)
                        archive.add('data_model_response',dict(iteration=0,response=proposal))
                save_json(proposal_path,proposal)
            new_report=None;new_rows=[];new_rules=[];error=None
            if engine.config.get('agent_task_runtime_v2') and (engine.archive.root/'pause.request').exists():
                from .tool_session import TaskPaused
                raise TaskPaused('pause requested after Data task submission, before reference evaluation')
            keep_current=proposal.get('action')=='keep_current'
            if keep_current:
                validate_keep_current(proposal,engine)
            else:
                try:
                    new_rules=normalize_rules(proposal)
                    plan=dispatch(new_rules,payload['instance_seeds']);save_json(root/'dispatch.json',plan)
                    preparation_root=root
                    if engine.config.get('measured_research_loop_v1'):
                        new_rules,plan,preparation_root=candidate_preparation(root,proposal,payload['instance_seeds'],settings)
                    by_version={r['version']:r for r in new_rules}
                    errors=[]
                    def prepare(item):
                        try:return self.prepare_case(preparation_root,item,by_version[item['rule_version']],settings)
                        except (ReferencePending,CancelledError):raise
                        except Exception as exc:return dict(error=type(exc).__name__+': '+str(exc),id=item['id'])
                    if engine.config.get('console_progress'):print(f'Prepare data | 0/{len(plan)} | workers={engine.workers}',flush=True)
                    preparation_workers=settings.get('reference_workers',engine.workers)
                    for item,row in completed_tasks(plan,prepare,preparation_workers):
                        if 'error' in row:errors.append(row)
                        else:new_rows.append(row)
                        done=len(errors)+len(new_rows)
                        if engine.config.get('console_progress') and (done%8==0 or done==len(plan)):
                            print(f'Prepare data | {done}/{len(plan)} | valid={len(new_rows)} | failed={len(errors)}',flush=True)
                    if errors:raise ValueError(canonical(errors))
                    new_rows.sort(key=lambda r:r['id'])
                    if not engine.config.get('diagnostic_run'):audit_training('tsp',original+new_rows,SUPPLEMENT_PROTOCOL)
                    new_records=engine.measure_rows(parent['candidate_id'],new_rows,f'data_round_{round_number}:new')
                    new_report=data_report(new_rows,new_records,parent['source_hash'],engine.runner.limit,seed)
                    if not new_report['valid']:error='current champion failed on proposed data; proposal not adopted'
                except ReferencePending:raise
                except (ValueError,SyntaxError,TypeError,KeyError) as exc:
                    error=type(exc).__name__+': '+str(exc)
            save_json(root/'new_evaluation.json',dict(report=new_report,error=error))
            adopted=bool(new_report and new_report['valid'] and prefer_new(old_report['mean_reference_gap_pct'],new_report['mean_reference_gap_pct']))
            selected_rows=new_rows if adopted else old_rows
            selected_rules=new_rules if adopted else rules
            selected_report=new_report if adopted else old_report
            save_json(root/'selected_rules.json',dict(rules=selected_rules))
            selection=dict(round=round_number,calibration_source_hash=parent['source_hash'],
                selected='new' if adopted else 'old',old_mean_gap_pct=old_report['mean_reference_gap_pct'],
                new_mean_gap_pct=new_report['mean_reference_gap_pct'] if new_report else None,error=error,
                rule='higher measured mean gap under same current solver; ties retain old data',
                old_evaluation=receipt(root/'old_evaluation.json'),new_evaluation=receipt(root/'new_evaluation.json'))
            if keep_current:selection.update(decision='keep_current',reason=proposal['reason'],evidence_ids=proposal['evidence_ids'])
            save_json(root/'selection.json',selection)
            save_json(root/'selected_evaluation.json',selected_report)
            selected=dict(protocol=SUPPLEMENT_PROTOCOL,diagnostic=bool(engine.config.get('diagnostic_run')),
                train=selected_rows,rule_bundle=receipt(root/'selected_rules.json'),
                evaluation_record=receipt(root/'selected_evaluation.json'),selection_record=receipt(root/'selection.json'),
                calibration_solver=dict(path=parent['source_path'],sha256=parent['source_hash']))
            save_json(root/'supplement_manifest.json',selected)
            selected_binding=receipt(root/'supplement_manifest.json')
            save_json(root/'result.json',dict(status='finished',supplement_manifest=selected_binding['path'],manifest=selected_binding))
            archive.add('data_selection',selection)
            if engine.config.get('console_progress'):
                new_display=f'{selection["new_mean_gap_pct"]:.8f}%' if selection['new_mean_gap_pct'] is not None else 'not proposed' if keep_current else 'invalid'
                print(f'Data selection | {selection["selected"]} | old_gap={selection["old_mean_gap_pct"]:.8f}% | new_gap={new_display}',flush=True)
        if not any(r['round']==round_number for r in engine.archive.of('data_round_complete')):
            engine.archive.add('data_round_complete',dict(round=round_number,parent_hash=parent['source_hash'],
                manifest=selected_binding,evaluation=selected['evaluation_record'],selection=load_json(root/'selection.json')))
        base=copy.deepcopy(initial)
        if engine.config.get('diagnostic_run'):
            base['train']=[r for r in initial['train'] if r.get('training_group')!='supplement']+selected['train']
            base.update(supplement_manifest=selected_binding,training_protocol=SUPPLEMENT_PROTOCOL)
            return base
        base['train']=base['train'][:64]
        for key in ('training_protocol','supplement_manifest','audit'):base.pop(key,None)
        return bind_supplement(base,selected_binding['path'])
