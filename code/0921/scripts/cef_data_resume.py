"""Explicit recovery of a recorded truncated Data proposal; frozen search unchanged."""
import argparse
import copy
import hashlib
from datetime import datetime
import json
from pathlib import Path
import shutil

from cef.alternating import AlternatingEngine, DataRoundProvider
from cef.billing import balance_snapshot, balance_changes, summarize_usage
from cef.cli import claim_owner, freeze_runtime, load_env
from cef.config import load_json, validate, file_hash
from cef.ledger import BudgetLedger
from cef.model import Model
from cef.store import Archive, canonical
from scripts.cef_alternating_launcher import finish_search
from scripts.cef_launcher import ENV_FILE, matching_receipt


def matching_payload(value, payload):
    if not isinstance(value,dict):return False
    return all(value.get(k)==v for k,v in payload.items())


def failed_data_request(search, provider, local, round_number):
    completed=search.of('round_complete')
    if round_number!=len(completed)+1:raise ValueError('selected round is not the pending round')
    if local.of('data_model_response'):return None
    starts=local.of('data_model_start')
    if len(starts)!=1:raise ValueError('one saved Data request start is required')
    payload=starts[0]['payload']
    if payload.get('global_round')!=round_number:raise ValueError('saved Data round differs')
    requests=[];responses=[]
    for record in provider.of('provider_request'):
        if record.get('role')!='data_designer':continue
        submitted=json.loads(record['messages'][-1]['content'])
        if matching_payload(submitted,payload):requests.append(record)
    for record in provider.of('provider_response'):
        if record.get('role')!='data_designer':continue
        if matching_payload(json.loads(record['user']),payload):responses.append(record)
    if any(r.get('parsed') is not None for r in responses):return None
    if not requests or len(requests)!=len(responses):
        raise ValueError('all selected provider attempts must be fully recorded; unknown request is not redispatched')
    if responses[-1].get('finish_reason')!='length':
        raise ValueError('a recorded length-truncated response is required')
    last=requests[-1]['request_parameters']
    old_cap=last.get('max_tokens',last.get('max_output_tokens'))
    if type(old_cap) is not int:raise ValueError('saved output limit missing')
    return dict(payload=payload,old_cap=old_cap,previous_attempts=len(requests))


def retry_data_request(search, provider, local, model, round_number, max_output_tokens):
    selected=failed_data_request(search,provider,local,round_number)
    if selected is None:return dict(status='already_available',round=round_number)
    if type(max_output_tokens) is not int or not selected['old_cap']<max_output_tokens<=65536:
        raise ValueError('explicit output limit must exceed the last truncated limit and be at most 65536')
    entry=provider.add('manual_data_recovery_start',dict(round=round_number,
        old_max_output_tokens=selected['old_cap'],new_max_output_tokens=max_output_tokens,
        original_payload_hash=hashlib.sha256(canonical(selected['payload']).encode()).hexdigest(),
        helper_sha256=file_hash(Path(__file__)),
        policy='Explicit new request for the same saved task with a larger output allowance; preserves truncated response and usage. Not continuation of partial code.'))
    try:
        response=model.call('data_designer',selected['payload'])
        local.add('data_model_response',dict(iteration=0,response=response,manual_recovery_seq=entry['seq']))
        provider.add('manual_data_recovery_complete',dict(round=round_number,recovery_seq=entry['seq']))
        return dict(status='recovered',round=round_number)
    except Exception as error:
        provider.add('manual_data_recovery_failure',dict(round=round_number,recovery_seq=entry['seq'],error_type=type(error).__name__))
        raise


def check_saved_requests(search, provider):
    """A cold resume must consume saved replies, never repeat unknown requests."""
    starts={r['call_id'] for r in search.of('model_start')}
    complete={r['call_id'] for kind in ('model_raw_response','model_response') for r in search.of(kind)}
    if starts-complete:raise ValueError('unresolved algorithm request requires explicit request recovery')
    for archive in (search,provider):
        if len(archive.of('provider_request'))!=len(archive.of('provider_response')):
            raise ValueError('unfinished provider request; no automatic redispatch')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',required=True,type=Path)
    parser.add_argument('--round',type=int,dest='round_number')
    parser.add_argument('--resume-saved',action='store_true',help='resume a fully saved checkpoint without retrying a failed Data request')
    parser.add_argument('--max-output-tokens',type=int,default=32768)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    if args.resume_saved and args.round_number is not None:parser.error('--resume-saved does not take --round')
    if not args.resume_saved and args.round_number is None:parser.error('--round or --resume-saved is required')
    folder=args.resume.resolve();c=validate(load_json(folder/'search_config.json'))
    if c.get('agent_tool_loop_v1'):
        raise ValueError('native tool sessions require runCEFAuto.py --resume or explicit resumeCEFTools.py network recovery')
    test=load_json(folder/'test_config.json')
    if not c.get('integrated_feedback_v1') or Path(c['run_dir']).resolve()!=folder/'search':
        raise ValueError('explicit integrated run folder required')
    for source in c['implementation_source_manifest']:
        if file_hash(source['path'])!=source['sha256']:raise ValueError('frozen source changed: '+source['path'])
    search=Archive(folder/'search');provider=Archive(folder/'data_agent')
    if args.resume_saved:
        check_saved_requests(search,provider)
        args.round_number=len(search.of('round_complete'))+1
        if c['max_iterations'] is not None and args.round_number>c['max_iterations']:raise ValueError('search rounds already complete')
        if not 1<=args.max_output_tokens<=65536:raise ValueError('invalid Data output allowance')
    stage=folder/'rounds'/f'{args.round_number:03d}'/'data'
    if not args.resume_saved and (not stage.is_dir() or (stage/'result.json').exists()):raise ValueError('pending data stage required')
    local=Archive(stage) if stage.is_dir() else None
    selected=None if args.resume_saved else failed_data_request(search,provider,local,args.round_number)
    if selected and not selected['old_cap']<args.max_output_tokens<=65536:
        raise ValueError('new output allowance must exceed the failed allowance and be at most 65536')
    observations=Archive(folder/'measurements')
    if {r['key'] for r in observations.of('observation_start')}-{r['key'] for r in observations.of('observation')}:
        raise ValueError('unfinished solver observations require separate recovery')
    if args.check:
        label='Saved checkpoint check OK' if args.resume_saved else 'Data recovery check OK'
        print(f'{label} | round={args.round_number} | completed_rounds={len(search.of("round_complete"))} | output_limit={args.max_output_tokens} | no API request',flush=True)
        return 0
    load_env(ENV_FILE if ENV_FILE.is_file() else None)
    if not matching_receipt(c):raise ValueError('matching complete-code preflight required')
    dc=copy.deepcopy(provider.of('provider_manifest')[0]['config'])
    old_cap=dc['max_output_tokens'];dc.update(max_output_tokens=args.max_output_tokens,transport_retry_limit=0)
    lease=claim_owner(c['run_dir'])
    before=None
    try:
        freeze_runtime(c['run_dir'],c)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup=folder/f'data_recovery_{stamp}';backup.mkdir()
        for path in [folder/'search_config.json',folder/'search/result.json',folder/'data_agent/facts.jsonl',
                     stage/'facts.jsonl',stage/'model_input.json',stage/'old_evaluation.json']:
            if path.is_file():
                destination=backup/path.relative_to(folder);destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,destination)
        override=dict(scope='this explicit recovery invocation and remaining Data requests',
            original_data_output_limit=old_cap,data_output_limit=args.max_output_tokens,transport_retry_limit=0,
            unchanged='search config, algorithm model controls, solver/evaluation conditions, completed work',
            helper_sha256=file_hash(Path(__file__)))
        (backup/'override.json').write_text(canonical(override),encoding='utf-8',newline='\n')
        provider.add('manual_data_recovery_configuration',override)
        before=balance_snapshot(c)
        data_model=Model(dc,provider,BudgetLedger(provider,{},enforce_limits=False))
        print(f'Recover Data round {args.round_number} | output_limit={args.max_output_tokens} | previous work retained',flush=True)
        if not args.resume_saved:
            retry_data_request(search,provider,local,data_model,args.round_number,args.max_output_tokens)
        engine=AlternatingEngine(c,None)
        engine.model=Model(c,engine.archive,engine.ledger)
        engine.data_provider=DataRoundProvider(c['alternating_data_config'],data_model)
        result=engine.run()
        tested=finish_search(engine,c,test,folder,result)
        summary=dict(search=result,test=tested,data_override=override)
        (backup/'result.json').write_text(canonical(summary),encoding='utf-8',newline='\n')
        if tested is None:
            print('Reference work pending | saved stages retained',flush=True)
        elif not tested.get('valid'):
            raise RuntimeError('final TSPLIB evaluation invalid; recorded results retained')
        else:
            print(f'Completed | rounds={len(engine.archive.of("round_complete"))} | TSPLIB_gap={tested["mean_gap_pct"]:.6f}%',flush=True)
        return 0
    finally:
        if before is not None:
            after=balance_snapshot(c)
            charges=engine.archive.of('charge') if 'engine' in locals() else search.of('charge')
            billing=dict(usage=summarize_usage(charges+provider.of('charge')),
                account_balance_changes=balance_changes(before,after),authoritative_per_request_debit=None,
                scope='Cumulative search/Data usage; balance change covers this recovery invocation.')
            (backup/'billing.json').write_text(canonical(billing),encoding='utf-8',newline='\n')
        lease.unlink()


if __name__=='__main__':raise SystemExit(main())
