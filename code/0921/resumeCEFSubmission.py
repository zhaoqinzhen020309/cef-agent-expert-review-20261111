"""Explicit submission-phase recovery using a run's verified frozen runtime.

Only the task-session controller is overlaid. Original configurations, source
snapshots, evaluations and failed model responses are retained unchanged.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--prepare-only',action='store_true',help='finish the pending Data submission only; do not start reference or solver evaluation')
    args=parser.parse_args();folder=args.resume.resolve()
    config=read(folder/'search_config.json');snapshot=folder/'search/runtime_snapshot'
    if not config.get('agent_tool_loop_v1') or Path(config['run_dir']).resolve()!=folder/'search':
        raise ValueError('native tool-loop run required')
    for item in config['implementation_source_manifest']:
        frozen=snapshot/Path(item['path']).resolve().relative_to(ROOT)
        if sha(frozen)!=item['sha256']:raise ValueError('frozen runtime changed: '+str(frozen))
    allowed={ROOT/'cef/tool_session.py',ROOT/'cef/tool_agents.py'}
    changed=[]
    for item in config['implementation_source_manifest']:
        path=Path(item['path']).resolve()
        actual=sha(path)
        if actual!=item['sha256']:
            if path not in allowed:raise ValueError('unrelated implementation changed: '+str(path))
            changed.append(dict(path=str(path),old_sha256=item['sha256'],new_sha256=actual))
    sys.path.insert(0,str(ROOT))
    from cef.cli import claim_owner,load_env
    from cef.config import validate
    from cef.store import Archive,canonical
    from cef.billing import summarize_usage
    from cef.alternating import AlternatingEngine
    import cef.tool_agents as agents
    # Validate data/protocol against the frozen runtime. Source integrity was
    # checked against its actual snapshot above, not against edited live files.
    validated=validate(dict(config,implementation_source_manifest=[]))
    validated['implementation_source_manifest']=config['implementation_source_manifest']
    if canonical(validated)!=canonical(config):raise ValueError('configuration normalization differs from original')
    overlay_path=ROOT/'cef/tool_session.py'
    spec=importlib.util.spec_from_file_location('cef._submission_recovery',overlay_path)
    overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
    overlay_hash=sha(overlay_path)
    lease=claim_owner(config['run_dir'])
    try:
        targets=[]
        for path in list((folder/'rounds').glob('*/data/agent_task/session/facts.jsonl'))+list((folder/'search/agent_tasks').glob('*/session/facts.jsonl')):
            journal=Archive(path.parent)
            if journal.of('submitted'):continue
            errors=journal.of('incomplete')
            if errors and errors[-1]['reason']=='task turn limit reached without accepted submission':targets.append(path.parent.parent)
        receipt_path=folder/'submission_recovery.json'
        if receipt_path.exists():
            receipt=read(receipt_path)
            if receipt['overlay_sha256']!=overlay_hash or receipt['source_changes']!=changed:raise ValueError('recovery controller changed after authorization')
            target=Path(receipt['task'])
        else:
            if len(targets)!=1:raise ValueError('exactly one completed research-limit task required')
            target=targets[0]
            receipt=dict(task=str(target),overlay_path=str(overlay_path),overlay_sha256=overlay_hash,source_changes=changed,
                policy='Original frozen runtime plus explicit submission-phase controller; future tasks retain 32 research responses and 2 submission responses. No evaluation/scoring change.')
            if not args.check:receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8',newline='\n')
        if args.check:
            print('Submission recovery check OK | frozen runtime verified | no API request',flush=True)
            return 0
        def factory(root,*pos,**kw):
            root=Path(root)
            identity=Archive(root/'session').of('identity') if (root/'session/facts.jsonl').exists() else []
            kw['finalize_turns']=identity[0].get('finalize_turns',0) if identity else 2
            session=overlay.ToolSession(root,*pos,**kw)
            if root.resolve()==target.resolve() and not session.archive.of('finalization_authorized') and not session.archive.of('submitted'):
                session.authorize_finalization()
            return session
        agents.ToolSession=factory
        if args.prepare_only:
            if target.parent.name!='data':raise ValueError('--prepare-only requires a Data task')
            from cef.model import Model
            from cef.ledger import BudgetLedger
            from scripts.cef_launcher import ENV_FILE
            load_env(ENV_FILE if ENV_FILE.is_file() else None)
            engine=AlternatingEngine(config,None)
            provider=Archive(folder/'data_agent');dc=provider.of('provider_manifest')[0]['config']
            local=Archive(target.parent)
            if not local.of('data_model_response'):
                proposal=agents.run_data_task(engine,Model(dc,provider,BudgetLedger(provider,{},enforce_limits=False)),
                    local.of('data_model_start')[0]['payload'],target.parent,config['alternating_data_config'])
                local.add('data_model_response',dict(iteration=0,response=proposal,tool_session=True,submission_recovery=receipt))
            usage=summarize_usage(provider.of('charge'))
            (folder/'submission_recovery_usage.json').write_text(json.dumps(usage,indent=2),encoding='utf-8',newline='\n')
            print('Data submission recovered | existing 128 evaluations retained | formal search not resumed',flush=True)
            return 0
    finally:lease.unlink()
    import scripts.cef_alternating_launcher as launcher
    def frozen_runtime(root,c):
        for item in c['implementation_source_manifest']:
            if sha(snapshot/Path(item['path']).resolve().relative_to(ROOT))!=item['sha256']:
                raise ValueError('frozen runtime changed')
        if sha(overlay_path)!=overlay_hash:raise ValueError('recovery controller changed')
        for item in changed:
            if sha(Path(item['path']))!=item['new_sha256']:raise ValueError('recovery source changed')
    launcher.freeze_runtime=frozen_runtime
    launcher.run_folder(config,read(folder/'test_config.json'),folder,True)
    return 0


if __name__=='__main__':raise SystemExit(main())
