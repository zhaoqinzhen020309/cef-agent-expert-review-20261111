"""Manual entry for the independent data Agent; --check never dispatches a model."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from cef.billing import balance_snapshot, balance_changes, summarize_usage
from cef.cli import load_env
from cef.instance_design import DataDesigner, write_json, validate_design_config
from cef.ledger import BudgetLedger
from cef.model import Model
from scripts.cef_launcher import build_config, ENV_FILE, API_KEY_ENV


def build_data_model_config(settings, folder):
    """Data-code generation has explicit sampling, independent of solver search."""
    thinking=settings.get('thinking',False)
    tokens=settings.get('max_output_tokens',8192)
    if type(thinking) is not bool: raise ValueError('thinking must be boolean')
    if type(tokens) is not int or tokens<1: raise ValueError('positive max_output_tokens required')
    sampling=dict(thinking=dict(type='enabled' if thinking else 'disabled'))
    if thinking:
        effort=settings.get('reasoning_effort','low')
        if effort not in ('low','high','max'): raise ValueError('reasoning_effort must be low/high/max')
        sampling['reasoning_effort']=effort
    c,_=build_config(settings['solver_seed'],Path(folder)/'unused_search',
        dict(model=settings['model'],base_url=settings['base_url'],iterations=settings['iterations'],workers=settings['workers']))
    for key in ('data_manifest','reference_manifest','evaluation_seed_manifest'):
        c.pop(key,None)
    c.update(model_sampling=sampling,max_output_tokens=tokens,model_progress_interval_s=30.)
    if settings.get('integrated_feedback_v1'):c['integrated_feedback_v1']=True
    if settings.get('agent_tool_loop_v1'):
        c.update(agent_tool_loop_v1=True,tool_max_output_tokens=settings.get('tool_max_output_tokens',32768))
    return c


def main(settings, argv=None):
    parser=argparse.ArgumentParser(description='CEF supplemental TSP data design')
    parser.add_argument('--check',action='store_true',help='validate configuration without creating output or API calls')
    args=parser.parse_args(argv)
    c=dict(settings)
    for name in ('output_dir','solver_source'): c[name]=str(Path(c[name]).resolve())
    if not Path(c['solver_source']).is_file(): raise FileNotFoundError(c['solver_source'])
    if Path(c['output_dir']).exists(): raise FileExistsError('data output already exists: '+c['output_dir'])
    validate_design_config(c)
    import scipy
    from scipy.optimize import milp
    model_config=build_data_model_config(c,Path(c['output_dir']))
    if args.check:
        print(f"Data configuration OK | instances={c['instances']} | iterations={c['iterations']} | workers={c['workers']} | thinking={model_config['model_sampling']['thinking']['type']} | max_tokens={model_config['max_output_tokens']} | scipy={scipy.__version__}")
        return 0
    load_env(ENV_FILE if ENV_FILE.is_file() else None)
    if not os.environ.get(API_KEY_ENV): raise ValueError('missing API authentication: '+API_KEY_ENV)
    designer=DataDesigner(c,None)
    designer.archive.add('provider_manifest',dict(identity={k:model_config[k] for k in
        ('provider','model_snapshot','endpoint','wire_api','route','model_sampling','model_seed_support','environment_manifest')},
        scipy_version=scipy.__version__))
    ledger=BudgetLedger(designer.archive,{},enforce_limits=False)
    designer.model=Model(model_config,designer.archive,ledger)
    before=balance_snapshot(model_config)
    print(f"Data design | model={c['model']} | iterations={c['iterations']} | instances={c['instances']} | workers={c['workers']} | thinking={model_config['model_sampling']['thinking']['type']}",flush=True)
    print('Run: '+str(designer.root),flush=True)
    failed=None
    try:
        result=designer.run()
        if not c.get('diagnostic',False):
            from cef.config import load_json
            from cef.data import bind_supplement
            root=Path(__file__).resolve().parents[1]
            base=load_json(root/'configs/cef_v1_3/tsp.template.json')['data_manifest']
            combined=bind_supplement(base,result['supplement_manifest'])
            write_json(designer.root/'training_manifest.json',combined)
        print('Supplement: '+result['supplement_manifest'],flush=True)
    except Exception as error:
        failed=error
        print('Data design failed: '+type(error).__name__+': '+str(error),flush=True)
    finally:
        after=balance_snapshot(model_config)
        usage=summarize_usage(designer.archive.of('charge'))
        result=dict(usage=usage,account_balance_changes=balance_changes(before,after),
                    authoritative_per_request_debit=None,
                    attribution_note='Account-wide changes may include other activity; provider console bills govern attribution.')
        for name,value in [('billing_summary.json',result),('balance_before.json',before),('balance_after.json',after)]:
            write_json(designer.root/name,value)
        print(f"Usage | known_tokens={usage['known_total_tokens']} | unknown_requests={usage['unknown_usage_requests']}",flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    from runDataAgent import CONFIG
    raise SystemExit(main(CONFIG))
