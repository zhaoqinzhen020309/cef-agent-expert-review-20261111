"""Construct honest templates from verified local artifacts; unknowns stay null."""
import json,os,platform,sys
from pathlib import Path
import numpy as np
from .config import file_hash,load_json
from .data import receipt,bind_original_tsplib,TSPLIB_METRIC,TSPLIB_DISTANCE_RULE,TSPLIB_REFERENCE_RULE

def templates(root,output):
    root=Path(root).resolve(); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    env=dict(python=sys.version,dependencies={'numpy':np.__version__},hardware=dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),logical_cpus=os.cpu_count()),worker_resources=dict(concurrency=1,threads=1,memory_mb=1024,isolation='restricted fresh Windows Job worker, not hostile-code sandbox'))
    for task,path in [('tsp',root/'data/cef_v1_3_20261026/tsp_manifest.json'),('fssp',root/'data/cef_fssp_U01_v1_3_20261026/manifest.json')]:
        data=load_json(path)
        if task=='tsp':
            data=bind_original_tsplib(data)
            data.update(evaluation_metric=TSPLIB_METRIC,distance_rule='continuous Euclidean author TSP100; TSPLIB '+TSPLIB_DISTANCE_RULE,reference_rule=TSPLIB_REFERENCE_RULE)
        seed=root/f'cef/seeds/{task}.py'
        refs=[receipt(root/'problems/tsp_gls/TrainingData/TSPAEL64.pkl'),receipt(root/'evaluators/tsplib_protocol.py')] if task=='tsp' else data['reference_files']
        c=dict(task=task,mode='CEF',run_dir=str(root/f'runs/cef_{task}_acceptance'),provider=None,model=None,model_snapshot=None,endpoint=None,wire_api=None,api_key_env=None,route=None,model_sampling=None,model_seed_support=None,rates=None,max_case_runs=320,max_wallclock_s=3600,max_llm_requests=8,max_llm_tokens=100000,max_output_tokens=8192,request_timeout_s=180,max_analysis_seconds=120,analysis_timeout_s=15,max_analysis_per_round=2,max_profile_instances=4,solver_time_limit_s=60,test_repeats=1,environment_manifest=env,baseline_source_manifest=dict(source_version=receipt(seed),algorithm_boundary='shared complete NN+2opt seed' if task=='tsp' else 'shared complete NEH seed',run_parameters={'time_limit_s':60}),data_manifest=data,reference_manifest=dict(training='author supplied Concorde reference costs' if task=='tsp' else 'no normalized training reference; mean raw makespan',test='canonical TSPLIB references' if task=='tsp' else 'original Taillard header feasible upper bounds, negative gaps retained',distance_rule=data.get('distance_rule','permutation flowshop independent DP makespan'),reference_files=refs),evaluation_seed_manifest={r['id']:2024 for r in data['train']},seed_sources=dict(data_seed=data.get('data_seed','author published fixed TSPAEL64'),solver_seed='2024 reused from previous explicit evaluation policy; fixed per fresh instance for both random and numpy',model_seed='unconfigured'),preflight_dir=str(root/f'acceptance/cef_{task}_connectivity'),connectivity_receipt=str(root/f'acceptance/cef_{task}_connectivity/receipt.json'))
        if task=='tsp':
            c['reference_manifest'].update(evaluation_metric=TSPLIB_METRIC,test=TSPLIB_REFERENCE_RULE)
            c['reference_manifest']['reference_files'].extend(r['source']['coordinates'] for r in data['test'])
        (output/f'{task}.template.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(status='configuration_required',observed_codex_model='gpt-6.1-sol',observed_codex_auth='ChatGPT account tokens; no OPENAI_API_KEY in auth.json or process env',project_env='0921/.env contains DeepSeek key; no provider/model snapshot/rate values',missing=['CEF provider + explicit model snapshot','API-compatible auth for selected provider','endpoint/wire route','sampling/seed capability','frozen billing rates/source'],not_attempted='No model call, no long-code preflight, no C-level acceptance; never use ChatGPT account tokens as an API key.')
    (output/'live_acceptance_blocker.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
