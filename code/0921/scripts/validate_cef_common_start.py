"""Small, explicitly diagnostic live acceptance; never a formal benchmark run."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true',help='Make recorded real model requests for a small acceptance run')
    args=parser.parse_args()
    if not args.live:
        parser.error('Explicit --live is required; offline coverage is in test_cef_common_start*.py')
    from scripts.cef_launcher import build_config
    from cef.cli import load_env
    from cef.config import validate
    from cef.preflight import preflight
    from cef.engine import Engine
    from cef.model import Model
    from cef.billing import balance_snapshot,write_billing
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    folder=ROOT/'acceptance'/('cef_common_start_'+stamp)
    folder.mkdir(parents=True,exist_ok=False)
    settings=dict(model='deepseek-flash',base_url='https://api.deepseek.com',iterations=3,
                  workers=2,evaluation_policy='joint_only',common_start_probe=True,probe_states=2,probe_seconds=3.)
    config,_=build_config(2025,folder,settings)
    config.update(preflight_dir=str(folder/'preflight'),connectivity_receipt=str(folder/'preflight/receipt.json'))
    config=validate(config)
    load_env(ROOT/'.env')
    before=balance_snapshot(config)
    report=dict(scope='Small runtime/model acceptance, not an algorithm performance result',folder=str(folder),
                formal_experiment_started=False,preflight_passed=False,search_passed=False)
    try:
        print('Acceptance | complete-code preflight and three real interface paths',flush=True)
        gate=preflight(config)
        report['preflight_passed']=gate['passed']
        if not gate['passed']: raise RuntimeError('Live preflight failed: '+str(gate.get('error')))
        small=copy.deepcopy(config)
        small.update(diagnostic_run=True,diagnostic_train_limit=2,solver_time_limit_s=1.,probe_seconds=.2,
            acceptance_note='Runtime acceptance on only two original training instances, 1 second per case. Not a formal performance experiment. Use actual evidence; do not invent measurements. Exercise compare_from_states when available and a continuation question warrants it, otherwise revise normally.')
        small=validate(small,formal=False)
        with (folder/'diagnostic_config.json').open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(small,stream,ensure_ascii=False,indent=2)
        print('Acceptance | three iterations, two training instances, no TSPLIB run',flush=True)
        engine=Engine(small,None); engine.model=Model(small,engine.archive,engine.ledger)
        result=engine.run()
        report.update(search_passed=result['status']=='finished' and result['completed_iterations']==3,
            search_result=result,formal_executions=len(engine.archive.of('execution')),
            probe_executions=len(engine.archive.of('probe_execution')),
            all_formal_executions_valid=all(r['valid'] for r in engine.archive.of('execution')))
        if not report['search_passed']: raise RuntimeError('Live mini-loop failed: '+str(result['reason']))
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error))
    finally:
        after=balance_snapshot(config)
        report['billing']=write_billing(folder,before,after)
        with (folder/'validation_summary.json').open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('search_result','billing')},ensure_ascii=False),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
