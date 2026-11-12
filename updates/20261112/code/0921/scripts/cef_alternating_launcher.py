"""One manual entry: paired data/algorithm rounds, recovery, then one final test."""
import argparse
import copy
from datetime import datetime,timedelta,timezone
import json
import os
import uuid
from pathlib import Path

from cef.alternating import AlternatingEngine,DataRoundProvider
from cef.billing import balance_snapshot,summarize_usage,balance_changes
from cef.cli import claim_owner,freeze_runtime,load_env
from cef.config import validate,load_json,file_hash
from cef.data import receipt,audit_test
from cef.instance_design import write_json
from cef.ledger import BudgetLedger
from cef.model import Model
from cef.patch import check_source
from cef.reporting import final_test,verify_export
from cef.store import Archive,canonical
from cef.task_tools import _atomic
from scripts.cef_launcher import build_config,matching_receipt,ENV_FILE,API_KEY_ENV,ROOT
from scripts.prepare_cef_training import build_data_model_config


def build_auto_config(settings,folder,seed):
    rounds=settings['rounds']
    manual=rounds is None
    if not manual and (type(rounds) is not int or rounds<1):raise ValueError('rounds must be None (manual) or a positive integer')
    c,test=build_config(seed,folder,dict(settings,iterations=1 if manual else rounds,evaluation_policy='joint_only',
        supplement_manifest=settings['initial_supplement']))
    source=Path(settings['initial_solver']).resolve()
    check_source(source.read_text(encoding='utf-8-sig'),require_research=c['common_start_probe'])
    c.update(alternating_agents=True,warm_start_source=str(source),warm_start_receipt=receipt(source))
    if manual:c.update(manual_rounds=True,max_iterations=None)
    if settings.get('round_benchmark'):c['round_benchmark']=True
    if 'evaluate_each_round' in settings:c['evaluate_each_round']=settings['evaluate_each_round']
    if settings.get('baseline_benchmark'):
        c['baseline_benchmark']=receipt(settings['baseline_benchmark'])
    if settings.get('integrated_feedback_v1'):
        c['integrated_feedback_v1']=True
    if settings.get('measured_research_loop_v1'):
        if not settings.get('agent_task_runtime_v2'):raise ValueError('measured research requires runtime v2')
        c['measured_research_loop_v1']=True
        c['tool_context_window_tokens']=settings.get('tool_context_window_tokens',1000000)
    if settings.get('exploration_efficiency_v1'):
        if not settings.get('measured_research_loop_v1'):raise ValueError('efficient exploration requires the measured research loop')
        c['exploration_efficiency_v1']=True
    if settings.get('agent_tool_loop_v1') and c.get('integrated_feedback_v1'):
        c.update(agent_tool_loop_v1=True,tool_max_turns=settings.get('tool_max_turns',None if settings.get('agent_task_runtime_v2') else 32),
            tool_timeout_s=settings.get('tool_timeout_s',10),tool_max_output_tokens=settings.get('tool_max_output_tokens',32768))
        if settings.get('agent_task_runtime_v2'):c['agent_task_runtime_v2']=True
    c['baseline_source_manifest'].update(source_version=receipt(source),algorithm_boundary='explicit warm start complete solver; not original seed cold start')
    from runDataAgent import CONFIG as DATA
    data={k:str(v) if isinstance(v,Path) else v for k,v in DATA.items()}
    data.update(model=settings['model'],base_url=settings['base_url'],iterations=1,
        solver_seed=seed,data_seed=settings['data_seed'],workers=settings.get('workers',1),
        thinking=settings.get('data_thinking',False),max_output_tokens=settings.get('data_max_output_tokens',8192),
        solver_source=str(source),output_dir=str(Path(folder)/'rounds'))
    if c.get('integrated_feedback_v1'):data['integrated_feedback_v1']=True
    if c.get('agent_tool_loop_v1'):
        data.update(agent_tool_loop_v1=True,reference_workers=settings.get('reference_workers',2),
            tool_max_turns=c['tool_max_turns'],tool_timeout_s=c['tool_timeout_s'],
            tool_max_output_tokens=c['tool_max_output_tokens'])
    c['alternating_data_config']=data
    from cef.reference_windows import backend_manifest
    c['reference_backend_manifest']=backend_manifest()
    for entry in c['reference_backend_manifest']['files']:
        if not Path(entry['path']).resolve().is_relative_to(ROOT.resolve()):continue
        if not any(r['path']==entry['path'] for r in c['implementation_source_manifest']):
            c['implementation_source_manifest'].append(entry)
    from cef.instance_design import validate_design_config
    validate_design_config(data)
    build_data_model_config(data,folder)
    initial=load_json(settings['initial_supplement'])
    evaluation=Path(initial['evaluation_record']['path']) if initial.get('evaluation_record') else Path(initial['train'][0]['source']['generator']['path']).with_name('evaluation.json')
    if evaluation.is_file():c['alternating_initial_evaluation']=receipt(evaluation)
    for path in (Path(__file__),ROOT/'runCEFAuto.py',ROOT/'scripts/prepare_cef_training.py'):
        if not any(r['path']==str(path.resolve()) for r in c['implementation_source_manifest']):
            c['implementation_source_manifest'].append(receipt(path))
    return validate(c),test


def finish_search(engine,config,test,folder,result):
    if result.get('status') in ('waiting_reference','paused','blocked'):return None
    if result.get('status')!='finished':
        raise RuntimeError('alternating search incomplete; no final benchmark: '+str(result.get('reason',result.get('status'))))
    rounds=len(engine.archive.of('round_complete')) if config.get('manual_rounds') else config['max_iterations']
    if config.get('manual_rounds') and not engine.archive.of('manual_finished'):
        raise RuntimeError('manual search must be explicitly finished before final testing')
    if (result.get('status')!='finished' or result.get('completed_iterations')!=rounds or
            len(engine.archive.of('round_complete'))!=rounds or not result.get('best')):
        raise RuntimeError('alternating search incomplete; no final benchmark: '+str(result.get('reason','')))
    if config.get('round_benchmark'):
        from cef.round_benchmark import monitor_round
        return monitor_round(engine,config,test,folder)
    folder=Path(folder)
    if (folder/'test/summary.json').is_file():
        # An explicit resume can reuse an already finished final evaluation.
        verify_export(config['run_dir'])
        freeze=load_json(folder/'test/freeze.json')
        if freeze['source_hash']!=result['best']['source_hash']:raise ValueError('final test belongs to another solver')
        if (freeze['benchmark']!=test['instances'] or freeze['test_repeats']!=test['test_repeats'] or
                freeze['seeds']!=test['evaluation_seeds']):raise ValueError('final test conditions differ from frozen configuration')
        return load_json(folder/'test/summary.json')
    if (folder/'test').exists():raise RuntimeError('incomplete final benchmark retained; no silent rerun')
    return final_test(config['run_dir'],folder/'test',test['instances'],test['test_repeats'],test['evaluation_seeds'])


def run_folder(config,test,folder,resume=False,*,operation='resume'):
    if operation not in ('resume','next','finish','evaluate'):raise ValueError('invalid manual operation')
    if operation!='resume' and not (resume and config.get('manual_rounds')):
        raise ValueError('next/finish require an existing manual-round run')
    folder=Path(folder)
    if Path(config['run_dir']).resolve()!=(folder/'search').resolve():
        raise ValueError('config.run_dir must be this run folder/search')
    expected_seed=next(iter(config['evaluation_seed_manifest'].values()))
    if (test['instances']!=config['data_manifest']['test'] or test['test_repeats']!=config['test_repeats'] or
            test['evaluation_seeds']!={f'{r["id"]}:{repeat}':expected_seed for r in test['instances'] for repeat in range(test['test_repeats'])}):
        raise ValueError('test conditions differ from frozen search configuration')
    if not resume:
        folder.mkdir(parents=True,exist_ok=False)
        if not matching_receipt(config):config['connectivity_receipt']=str(folder/'preflight/receipt.json')
        write_json(folder/'search_config.json',config);write_json(folder/'test_config.json',test)
    elif not (folder/'search/facts.jsonl').exists():raise ValueError('resume requires a real search journal')
    load_env(ENV_FILE if ENV_FILE.is_file() else None)
    research=operation not in ('finish','evaluate')
    if research and not os.environ.get(API_KEY_ENV):raise ValueError('missing API authentication: '+API_KEY_ENV)
    before=balance_snapshot(config) if research else {}
    engine=None;data_archive=None;lease=None;summary=None
    try:
        if research and not matching_receipt(config):
            if resume:raise RuntimeError('matching complete-code preflight required for resume')
            from cef.preflight import preflight
            print('Preflight...',flush=True)
            gate=preflight(config)
            if not gate.get('passed'):raise RuntimeError('preflight failed; records retained')
        Path(config['run_dir']).mkdir(parents=True,exist_ok=True)
        lease=claim_owner(config['run_dir'])
        if resume and config.get('agent_task_runtime_v2'):
            (Path(config['run_dir'])/'pause.request').unlink(missing_ok=True)
        freeze_runtime(config['run_dir'],config)
        engine=AlternatingEngine(config,None)
        if operation=='evaluate':
            from cef.round_benchmark import monitor_round
            measured=monitor_round(engine,config,test,folder)
            if measured is None:raise ValueError('complete the current round before evaluation')
            return measured
        if research:
            engine.model=Model(config,engine.archive,engine.ledger)
            data_archive=Archive(folder/'data_agent')
            dc=build_data_model_config(config['alternating_data_config'],folder)
            data_model=Model(dc,data_archive,BudgetLedger(data_archive,{},enforce_limits=False))
            if not data_archive.of('provider_manifest'):
                data_archive.add('provider_manifest',dict(config=dc))
            engine.data_provider=DataRoundProvider(config['alternating_data_config'],data_model)
        label='manual' if config.get('manual_rounds') else str(config['max_iterations'])
        print(f'Alternating CEF | model={config["model"]} | rounds={label} | workers={config["evaluation_workers"]}',flush=True)
        print('Run: '+str(folder),flush=True)
        if config.get('baseline_benchmark'):
            binding=config['baseline_benchmark']
            if file_hash(binding['path'])!=binding['sha256']:raise ValueError('baseline benchmark record changed')
            baseline=load_json(binding['path']);conditions=load_json(Path(binding['path']).with_name('freeze.json'))
            if (baseline.get('valid') and baseline.get('source_hash')==config['warm_start_receipt']['sha256'] and
                baseline.get('evaluation_metric')==test['instances'][0].get('evaluation_metric') and
                conditions.get('solver_time_limit_s')==config['solver_time_limit_s'] and
                conditions.get('workers')==config['evaluation_workers'] and conditions.get('seeds')==test['evaluation_seeds'] and
                [(r['id'],r['sha256'],r['reference'],r.get('evaluation_metric')) for r in conditions.get('benchmark',[])]==
                [(r['id'],r['sha256'],r['reference'],r.get('evaluation_metric')) for r in test['instances']]):
                print(f'Starting solver | TSPLIB_gap={baseline["mean_gap_pct"]:.6f}% | recorded evaluation',flush=True)
        existing=load_json(folder/'search/result.json') if (folder/'search/result.json').is_file() else None
        if operation=='finish':result=engine.finish_manual()
        elif config.get('manual_rounds'):result=engine.run(manual_next=operation=='next')
        elif (resume and existing and existing.get('status')=='finished' and
                len(engine.archive.of('round_complete'))==config['max_iterations']):result=existing
        else:result=engine.run()
        if config.get('round_benchmark') and config.get('evaluate_each_round',True) and result.get('status')=='paused':
            from cef.round_benchmark import monitor_round
            monitor_round(engine,config,test,folder)
        tested=finish_search(engine,config,test,folder,result)
        if tested is None:
            if result.get('status') in ('paused','blocked'):
                summary=dict(status=result['status'],rounds=len(engine.archive.of('round_complete')),reason=result['reason'])
                _atomic(folder/('status_'+result['status']+'.json'),json.dumps(summary,ensure_ascii=False,indent=2))
                print(result['status'].title()+': '+result['reason'],flush=True)
                if result['status']=='paused':
                    if config.get('manual_rounds'):
                        count=len(engine.archive.of('round_complete'))
                        if count:print('Round results: '+str(folder/'rounds'/f'{count:03d}'/'round_summary.json'),flush=True)
                        print('Evaluate: python .\\runCEFAuto.py --evaluate "'+str(folder)+'"',flush=True)
                        print('Next: python .\\runCEFAuto.py --next "'+str(folder)+'"',flush=True)
                        print('Finish: python .\\runCEFAuto.py --finish "'+str(folder)+'"',flush=True)
                    print('Resume: python .\\runCEFAuto.py --resume "'+str(folder)+'"',flush=True)
                return summary
            summary=dict(status='waiting_reference',rounds=len(engine.archive.of('round_complete')),
                pending_round=len(engine.archive.of('round_complete'))+1,reason=result['reason'],
                protocol='alternating_data_algorithm_v1')
            name='reference_pending_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'
            write_json(folder/name,summary)
            print(f'Reference work pending | round={summary["pending_round"]} | completed rounds retained; use --resume to continue',flush=True)
            return summary
        summary=dict(status='finished' if tested.get('valid') else 'invalid_test',rounds=len(engine.archive.of('round_complete')),
            best=result['best'],test=tested,protocol='alternating_data_algorithm_v1',
            warm_start=config['warm_start_receipt'])
        destination=folder/('resume_summary_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]+'.json' if resume else 'launch_summary.json')
        write_json(destination,summary)
        if not tested.get('valid'):raise RuntimeError('final TSPLIB evaluation invalid; records retained')
        print(f'Completed | rounds={summary["rounds"]} | TSPLIB_gap={tested["mean_gap_pct"]:.6f}%',flush=True)
        return summary
    finally:
        if lease is not None:lease.unlink()
        after=balance_snapshot(config) if research else {}
        charges=[];unresolved=0
        for stage,path in (('search',folder/'search'),('data',folder/'data_agent'),('preflight',folder/'preflight')):
            journal=path/'facts.jsonl'
            if not journal.is_file():continue
            a=Archive(path);charges.extend(a.of('charge'))
            unresolved+=max(0,len(a.of('provider_request'))-len(a.of('provider_response')))
        charges.extend(dict(resource='llm_tokens',amount=0,usage_known=False) for _ in range(unresolved))
        billing=dict(usage=summarize_usage(charges),account_balance_changes=balance_changes(before,after),
            authoritative_per_request_debit=None,scope='Cumulative provider usage; account change covers this invocation and may include other activity.')
        name='billing_summary.json' if not resume else 'billing_resume_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]+'.json'
        write_json(folder/name,billing)


def run_until(config,test,folder,target,check=False):
    """Continue the same manual run to an absolute completed-round target."""
    if not config.get('manual_rounds'):raise ValueError('--auto requires a manual-round run')
    if type(target) is not int or target<1:raise ValueError('--until must be positive')
    archive=Archive(folder/'search')
    if archive.of('manual_finished'):raise ValueError('search is already finished')
    count=len(archive.of('round_complete'))
    if target<count:raise ValueError('target precedes completed rounds')
    if check:
        print(f'Automatic continuation OK | completed={count} | target={target}')
        return
    while count<target:
        run_folder(config,test,folder,True,operation='next')
        after=len(Archive(folder/'search').of('round_complete'))
        if after<=count:
            print('Automatic continuation paused; pending work retained. Repeat the same --auto command to continue.',flush=True)
            return
        count=after
    run_folder(config,test,folder,True,operation='evaluate')


def main(settings,argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--auto',type=Path,help='continue this existing manual run automatically')
    p.add_argument('--until',type=int,help='absolute total rounds for --auto, e.g. 10')
    p.add_argument('--resume',type=Path,help='resume an explicit existing run folder')
    p.add_argument('--next',type=Path,help='run one next round in an existing manual run; pending round is resumed first')
    p.add_argument('--finish',type=Path,help='finish a manual run at a complete round boundary and evaluate TSPLIB29')
    p.add_argument('--evaluate',type=Path,help='evaluate the completed round without continuing or finishing search')
    p.add_argument('--pause',type=Path,help='request a pause after the current evaluation/tool/model call finishes')
    p.add_argument('--recover-data',type=Path,help='offline recovery fork for a known reference-failed data stage; no API calls')
    args=p.parse_args(argv)
    if sum(value is not None for value in (args.resume,args.next,args.finish,args.evaluate,args.pause,args.recover_data,args.auto))>1:
        p.error('choose one of --resume, --next, --finish, --pause or --recover-data')
    if (args.auto is None)!=(args.until is None):p.error('--auto and --until must be supplied together')
    if args.auto:
        folder=args.auto.resolve();c=validate(load_json(folder/'search_config.json'));test=load_json(folder/'test_config.json')
        run_until(c,test,folder,args.until,args.check)
        return 0
    if args.pause:
        if args.resume or args.check or args.recover_data:raise ValueError('--pause is a separate command')
        folder=args.pause.resolve()
        saved=load_json(folder/'search_config.json')
        if not saved.get('agent_task_runtime_v2') or Path(saved['run_dir']).resolve()!=folder/'search':
            raise ValueError('safe pause requires a runtime-v2 run folder')
        if not (folder/'search/facts.jsonl').is_file():raise ValueError('active search journal required')
        _atomic(folder/'search/pause.request',json.dumps(dict(reason='user requested safe-boundary pause')))
        print('Pause requested; current work will finish before stopping.',flush=True)
        return 0
    if args.recover_data:
        if args.resume or args.check:raise ValueError('--recover-data is a separate offline command')
        from cef.data_recovery import recover_run
        folder=recover_run(args.recover_data)
        label='Recovered'
        if (folder/'recovery_provenance.json').is_file():
            number=load_json(folder/'recovery_provenance.json')['pending_round']
            if not (folder/'rounds'/f'{number:03d}'/'data/result.json').is_file():label='Reference work pending'
        print(label+': '+str(folder),flush=True)
        print('Resume: python .\\runCEFAuto.py --resume "'+str(folder)+'"',flush=True)
        return 0
    if args.resume or args.next or args.finish or args.evaluate:
        folder=(args.resume or args.next or args.finish or args.evaluate).resolve();c=validate(load_json(folder/'search_config.json'));test=load_json(folder/'test_config.json')
        if not c.get('alternating_agents'):raise ValueError('resume target is not an alternating run')
        if (args.next or args.finish or args.evaluate) and not c.get('manual_rounds'):raise ValueError('--next/--finish/--evaluate require a manual-round run')
        if args.check:print('Alternating resume configuration OK');return 0
        if args.next or args.finish or args.evaluate:run_folder(c,test,folder,True,operation='evaluate' if args.evaluate else 'next' if args.next else 'finish')
        else:run_folder(c,test,folder,True)
        return 0
    seeds=settings['seeds']
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or not 0<=s<2**32 for s in seeds):
        raise ValueError('distinct explicit uint32 solver seeds required')
    for seed in seeds:
        from cef.reference_windows import ensure_backend_ready
        ensure_backend_ready()
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        folder=Path(settings['output_dir']).resolve()/f'tsp_s{seed}_{stamp}'
        c,test=build_auto_config(settings,folder,seed)
        if args.check:
            label='manual (one round per command)' if c.get('manual_rounds') else str(c['max_iterations'])
            print(f'Alternating configuration OK | rounds={label} | seed={seed} | original=64 | supplement=64 | test=29 | workers={c["evaluation_workers"]} | metric={c["data_manifest"]["evaluation_metric"]} | measured_research={c.get("measured_research_loop_v1",False)}')
        else:run_folder(c,test,folder)
    return 0
