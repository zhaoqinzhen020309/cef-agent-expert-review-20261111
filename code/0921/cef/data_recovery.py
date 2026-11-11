"""Explicit offline fork of a known reference-failed alternating run.

The source run is immutable. Finished search measurements retain their original
data and runner identities; only the incomplete data stage uses the repaired
reference backend. A changed implementation is recorded, never relabelled as
the original frozen runtime.
"""
import copy
from datetime import datetime
from pathlib import Path
import shutil

from .config import file_hash,load_json,validate
from .data import receipt
from .instance_design import DataDesigner,write_json
from .store import Archive,canonical
from .reference_jobs import ReferencePending


REPAIRABLE={'cef/reference.py','cef/instance_design.py','cef/alternating.py',
            'scripts/cef_alternating_launcher.py','cef/data_recovery.py','cef/reference_jobs.py','cef/parallel.py',
            'cef/reference_native.py','cef/config.py','cef/data.py',
            'cef/reference_scip_solver.py','cef/model.py'}


def recover_run(source, destination=None):
    source=Path(source).resolve();project=Path(__file__).resolve().parents[1]
    if not (source/'search/facts.jsonl').is_file():raise ValueError('actual alternating run required')
    if (source/'search/owner.lock').exists():raise ValueError('stop the owning experiment before data recovery')
    if (source/'test').exists():raise ValueError('data recovery cannot alter a frozen final benchmark')
    original=load_json(source/'search_config.json');records=_records(source/'search/facts.jsonl')
    manifests=[e['data'] for e in records if e['kind']=='manifest']
    if len(manifests)!=1 or canonical(manifests[0])!=canonical(original):raise ValueError('search configuration differs from journal')
    if not original.get('alternating_agents') or Path(original['run_dir']).resolve()!=source/'search':
        raise ValueError('matching alternating run directory required')
    result=load_json(source/'search/result.json')
    paused_audit=result.get('status')=='paused'
    transport_repair=(result.get('status')=='failed' and original.get('agent_task_runtime_v2') and
        any(result.get('reason','').startswith(name+':') for name in
            ('APIConnectionError','APITimeoutError','RemoteProtocolError','ConnectError','ReadTimeout')))
    if paused_audit:
        if (not original.get('agent_task_runtime_v2') or not (source/'search/pause.request').is_file()
                or load_json(source/'status_paused.json').get('status')!='paused'):
            raise ValueError('audit repair requires a saved safe-boundary pause')
    if not (paused_audit or transport_repair) and result.get('status')!='waiting_reference' and (result.get('status')!='failed' or 'no valid measured generator' not in result.get('reason','')):
        raise ValueError('recovery requires a recorded failed data generator evaluation')
    completed=[e['data'] for e in records if e['kind']=='round_complete']
    number=len(completed)+1
    pending=[e['data'] for e in records if e['kind']=='data_round_start' and e['data']['round']==number]
    activated=[e for e in records if e['kind']=='training_epoch' and e['data']['round']==number]
    if len(pending)!=1 or (len(activated)!=1 if paused_audit or transport_repair else bool(activated)):
        raise ValueError('one unactivated pending data round required')
    data_path=source/'rounds'/f'{number:03d}'/'data'
    integrated=bool(original.get('integrated_feedback_v1'))
    if integrated:
        proposal=load_json(data_path/'proposal.json')
        from .recipes import normalize_rules
        normalize_rules(proposal)
        if original.get('agent_tool_loop_v1'):
            submitted=[r['data'] for r in _records(data_path/'agent_task/session/facts.jsonl') if r['kind']=='submitted']
            if len(submitted)!=1 or canonical(submitted[0]['value'])!=canonical(proposal):
                raise ValueError('integrated recovery requires the original accepted Data submission')
    previous_repair=load_json(source/'recovery_provenance.json') if (source/'recovery_provenance.json').is_file() else None
    if previous_repair:
        for key in ('source_config','source_search_journal'):
            if file_hash(previous_repair[key]['path'])!=previous_repair[key]['sha256']:
                raise ValueError('previous repair provenance changed')
        prior_round=previous_repair['pending_round']
        if prior_round!=number:
            finished={e['round'] for e in completed}
            prior_data=[e['data'] for e in records if e['kind']=='data_round_complete' and e['data']['round']==prior_round]
            if prior_round>=number or prior_round not in finished or not prior_data:
                raise ValueError('previous repair is not a completed ancestor of this data round')
            artifact=prior_data[-1]['manifest']
            if file_hash(artifact['path'])!=artifact['sha256']:raise ValueError('ancestor data manifest changed')
    pending_path=Path(pending[0]['path']).resolve()
    legacy_path=Path(previous_repair['source_run'])/'rounds'/f'{number:03d}'/'data' if previous_repair and previous_repair['pending_round']==number else None
    if pending_path!=data_path and pending_path!=legacy_path:
        raise ValueError('pending data directory differs')
    if (data_path/'result.json').exists() and not previous_repair and not (paused_audit or transport_repair):
        raise ValueError('only a documented repair fork may reuse a recovered data result')
    for name in ('data_agent','search'):
        events=_records(source/name/'facts.jsonl')
        if sum(e['kind']=='provider_request' for e in events)!=sum(e['kind']=='provider_response' for e in events):
            raise ValueError('unknown model response retained; recovery cannot resend it')
    updated=[];changed=[]
    for old in original['implementation_source_manifest']:
        path=Path(old['path']).resolve();relative=path.relative_to(project).as_posix()
        snapshot=source/'search/runtime_snapshot'/relative
        if not snapshot.is_file() or file_hash(snapshot)!=old['sha256']:
            raise ValueError('original frozen implementation is missing or changed: '+relative)
        now=receipt(path)
        if now['sha256']!=old['sha256']:
            if transport_repair and relative not in {'cef/model.py','cef/data_recovery.py'}:
                raise ValueError('transport repair cannot change research implementation: '+relative)
            if paused_audit and relative not in {'cef/data.py','cef/data_recovery.py'}:
                raise ValueError('paused audit repair cannot change research implementation: '+relative)
            if relative not in REPAIRABLE:raise ValueError('unrelated implementation changed: '+relative)
            changed.append(dict(path=relative,before_sha256=old['sha256'],after_sha256=now['sha256']))
        updated.append(now)
    for name in ('data_recovery.py','reference_jobs.py'):
        helper=receipt(Path(__file__).with_name(name))
        if not any(x['path']==helper['path'] for x in updated):updated.append(helper)
    destination=Path(destination).resolve() if destination else source.with_name(source.name.split('_recovered_')[0]+'_recovered_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if destination==source or destination.parent!=source.parent or destination.exists():
        raise ValueError('recovery requires a new sibling directory')
    # Validate all immutable bindings before copying or computing any cases.
    config=copy.deepcopy(original);config['run_dir']=str(destination/'search');config['implementation_source_manifest']=updated
    if integrated and not (paused_audit or transport_repair):
        from .reference_windows import backend_manifest
        config['reference_backend_manifest']=backend_manifest()
        for dependency in config['reference_backend_manifest']['files']:
            if Path(dependency['path']).resolve().is_relative_to(project) and not any(x['path']==dependency['path'] for x in updated):
                updated.append(dependency)
        config['implementation_source_manifest']=updated
    config=validate(config,formal=not config.get('diagnostic_run',False))
    if any(p.is_symlink() for p in source.rglob('*')):raise ValueError('run recovery does not follow symlinks')
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','owner.lock'))
    provenance=dict(source_run=str(source),source_search_journal=receipt(source/'search/facts.jsonl'),
        source_config=receipt(source/'search_config.json'),pending_round=number,completed_rounds=len(completed),
        changed_implementation=changed,semantics='explicit runtime repair fork; original scores and failed evidence preserved',
        model_requests_during_recovery=0)
    if (destination/'recovery_provenance.json').exists():
        (destination/'recovery_provenance.json').rename(destination/('recovery_provenance_previous_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'))
    write_json(destination/'recovery_provenance.json',provenance)
    # Original versions remain in the source run and the copied pre-repair snapshot.
    snapshot=destination/'search/runtime_snapshot_before_repair';index=0
    while snapshot.exists():
        index+=1;snapshot=destination/'search'/f'runtime_snapshot_before_repair_{index:03d}'
    (destination/'search/runtime_snapshot').rename(snapshot)
    for event in records:
        if event['kind']=='manifest':event['data']=config
        if event['kind']=='data_round_start' and event['data']['round']==number:
            event['data']=dict(event['data'],path=str(destination/'rounds'/f'{number:03d}'/'data'),
                               copied_from_path=event['data']['path'])
    (destination/'search/facts.jsonl').write_text(''.join(canonical(e)+'\n' for e in records),encoding='utf-8',newline='\n')
    (destination/'search_config.json').write_text(canonical(config)+'\n',encoding='utf-8',newline='\n')
    from .cli import freeze_runtime
    freeze_runtime(config['run_dir'],config)
    Archive(destination/'search').add('runtime_repair_fork',provenance)
    print(f'Recover data round {number} | source run preserved | no model request',flush=True)
    if transport_repair:
        ready=dict(status='ready_to_resume',run_dir=str(destination),current_round=number,
                   model_requests_during_recovery=0,
                   policy='Native stream transport repair only; active epoch, tasks and completed tools retained; explicit failed-request retry required.')
        Archive(destination/'search').add('transport_recovery_ready',ready)
        write_json(destination/'transport_recovery_result.json',ready)
        return destination
    if paused_audit:
        ready=dict(status='paused',run_dir=str(destination),current_round=number,
                   model_requests_during_recovery=0,
                   policy='Audit implementation repair only; historical data selection, active epoch and completed native-tool effects retained; manual resume required.')
        Archive(destination/'search').add('paused_audit_recovery_ready',ready)
        write_json(destination/'audit_recovery_result.json',ready)
        return destination
    if integrated:
        # A completed native Data task must not be sent to the legacy standalone
        # designer. Resume the normal integrated pipeline with its accepted
        # proposal and coordinate receipts; no generation or timed evaluation here.
        ready=dict(status='ready_to_resume',run_dir=str(destination),pending_round=number,
                   accepted_proposal=receipt(destination/'rounds'/f'{number:03d}'/'data/proposal.json'),
                   model_requests_during_recovery=0)
        Archive(destination/'search').add('integrated_data_recovery_ready',ready)
        if (destination/'recovery_result.json').exists():
            (destination/'recovery_result.json').rename(destination/('recovery_result_previous_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'))
        write_json(destination/'recovery_result.json',ready)
        return destination
    try:result=DataDesigner.recover(destination/'rounds'/f'{number:03d}'/'data')
    except ReferencePending as error:
        Archive(destination/'search').add('reference_waiting',dict(pending_round=number,error=str(error)))
        write_json(destination/('reference_pending_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'),
            dict(status='waiting_reference',run_dir=str(destination),pending_round=number,reason=str(error)))
        return destination
    Archive(destination/'search').add('data_stage_recovered',dict(round=number,result=result,
        manifest=receipt(result['supplement_manifest'])))
    if (destination/'recovery_result.json').exists():
        (destination/'recovery_result.json').rename(destination/('recovery_result_previous_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'))
    write_json(destination/'recovery_result.json',dict(status='finished',run_dir=str(destination),
        recovered_round=number,completed_rounds=len(completed),data_result=result))
    return destination


def _records(path):
    import json
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
