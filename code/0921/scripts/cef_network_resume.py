"""Explicit recovery of a fully recorded network failure; frozen search code stays intact."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from cef.cli import claim_owner,freeze_runtime,load_env
from cef.config import validate,load_json,file_hash
from cef.store import Archive
from cef.ledger import BudgetLedger
from cef.model import Model
from cef.billing import balance_snapshot,balance_changes,summarize_usage
from scripts.cef_launcher import ENV_FILE,matching_receipt

NETWORK_ERRORS=('APIConnectionError:', 'APITimeoutError:', 'RemoteProtocolError:', 'ConnectError:', 'ReadTimeout:')


def failed_request(archive,call_id):
    if any(r.get('call_id')==call_id for kind in ('model_raw_response','model_response') for r in archive.of(kind)):
        return None
    starts=[r for r in archive.records if r['kind']=='model_start' and r['data']['call_id']==call_id]
    if len(starts)!=1:raise ValueError('one original model request is required')
    start=starts[0]
    if start['data']['role'] not in ('planner','editor'):raise ValueError('unsupported recovery role')
    known={r['call_id'] for kind in ('model_raw_response','model_response') for r in archive.of(kind)}
    pending={r['call_id'] for r in archive.of('model_start')}-known
    if pending!={call_id}:raise ValueError('only the explicitly selected failed request may be pending')
    failures=archive.of('failure')
    if not failures or not failures[-1]['error'].startswith(NETWORK_ERRORS):
        raise ValueError('a recorded terminal network failure is required')
    following=archive.records[start['seq']+1:]
    requests=[r['data'] for r in following if r['kind']=='provider_request']
    responses=[r['data'] for r in following if r['kind']=='provider_response']
    if not requests or len(requests)!=len(responses):
        raise ValueError('all failed provider attempts must be fully recorded')
    if any(r['role']!=start['data']['role'] for r in requests+responses):
        raise ValueError('unexpected provider role after failed request')
    if any(r.get('parsed') is not None or not r.get('error_chain') for r in responses):
        raise ValueError('provider response requires individual inspection; no complete response is discarded')
    issued={r['execution_id'] for r in archive.of('execution_start')}
    finished={r['execution_id'] for kind in ('execution','probe_execution') for r in archive.of(kind)}
    if issued-finished:raise ValueError('unfinished executions require separate recovery')
    return start


def retry_failed_request(archive,model,call_id):
    start=failed_request(archive,call_id)
    if start is None:return dict(status='already_available',call_id=call_id)
    entry=archive.add('manual_model_recovery_start',dict(call_id=call_id,original_start_seq=start['seq'],
        role=start['data']['role'],helper_sha256=file_hash(Path(__file__)),
        policy='Explicit user redispatch of saved payload; may incur additional provider usage; all failed attempts and unknown charges remain.'))
    try:
        response=model.call(start['data']['role'],start['data']['payload'])
        archive.add('model_raw_response',dict(call_id=call_id,role=start['data']['role'],response=response))
        archive.add('manual_model_recovery_complete',dict(call_id=call_id,recovery_seq=entry['seq']))
        return dict(status='recovered',call_id=call_id)
    except Exception as error:
        archive.add('manual_model_recovery_failure',dict(call_id=call_id,recovery_seq=entry['seq'],error_type=type(error).__name__))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path,required=True)
    parser.add_argument('--retry-failed-request',required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    folder=args.resume.resolve()
    c=validate(load_json(folder/'search_config.json'));test=load_json(folder/'test_config.json')
    if c.get('agent_tool_loop_v1'):
        raise ValueError('native tool sessions require resumeCEFTools.py, not legacy request recovery')
    if not c.get('alternating_agents') or Path(c['run_dir']).resolve()!=(folder/'search').resolve():
        raise ValueError('explicit alternating run folder required')
    for row in c['implementation_source_manifest']:
        if file_hash(row['path'])!=row['sha256']:raise ValueError('frozen search source changed: '+row['path'])
    archive=Archive(c['run_dir'])
    selected=failed_request(archive,args.retry_failed_request)
    if args.check:
        print('Network recovery check OK | '+args.retry_failed_request+' | saved evaluations retained',flush=True)
        return 0
    if selected is not None:
        load_env(ENV_FILE if ENV_FILE.is_file() else None)
        if not os.environ.get(c['api_key_env']):raise ValueError('missing API authentication: '+c['api_key_env'])
        if not matching_receipt(c):raise ValueError('matching complete-code preflight required')
        from cef.reference_windows import ensure_backend_ready
        ensure_backend_ready()
        lease=claim_owner(c['run_dir'])
        try:
            freeze_runtime(c['run_dir'],c)
            before=balance_snapshot(c)
            try:
                ledger=BudgetLedger(archive,{},enforce_limits=False)
                print('Retry failed request | '+args.retry_failed_request,flush=True)
                retry_failed_request(archive,Model(c,archive,ledger),args.retry_failed_request)
            finally:
                after=balance_snapshot(c)
                billing=dict(usage=summarize_usage(archive.of('charge')),
                    account_balance_changes=balance_changes(before,after),
                    scope='Cumulative search usage; balance change covers only the explicit recovery invocation.')
                path=folder/('billing_network_recovery_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
                path.write_text(json.dumps(billing,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        finally:
            lease.unlink()
    print('Resume saved search',flush=True)
    from scripts.cef_alternating_launcher import run_folder
    run_folder(c,test,folder,True)
    return 0
