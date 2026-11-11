"""Validate before creating a formal run. No guessed model, seeds or rates."""
import copy,hashlib,json,math
from pathlib import Path

MODES=('CEF','B0','B1','B2','B3','B4')
REQUIRED=('model_snapshot','environment_manifest','baseline_source_manifest','data_manifest','reference_manifest','max_case_runs','max_wallclock_s','test_repeats','evaluation_seed_manifest','provider','model','endpoint','api_key_env','rates','task','mode','run_dir','max_output_tokens','request_timeout_s','model_sampling','model_seed_support','route','wire_api')

def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def freeze_research_contract(c):
    """Validate optional research settings before model or solver execution."""
    c={**dict(common_start_probe=False,probe_states=8,probe_seconds=3.0),**c}
    if type(c['common_start_probe']) is not bool: raise ValueError('common_start_probe must be a boolean')
    if type(c['probe_states']) is not int or not 1<=c['probe_states']<=8: raise ValueError('probe_states must be an integer in 1..8')
    seconds=c['probe_seconds']
    limit=c.get('solver_time_limit_s',60)
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('probe_seconds must be positive and finite')
    if not c['common_start_probe']: return c
    if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit<=0 or seconds>limit:
        raise ValueError('probe_seconds must not exceed solver_time_limit_s')
    if c.get('mode')!='CEF' or c.get('evaluation_policy')!='joint_only' or c.get('stop_rule')!='iterations':
        raise ValueError('common_start_probe requires CEF joint_only iteration mode')
    seeds=c.get('evaluation_seed_manifest')
    if not isinstance(seeds,dict) or not seeds or any(type(seed) is not int or not 0<=seed<2**32 for seed in seeds.values()):
        raise ValueError('common_start_probe requires explicit valid solver seeds')
    # Import only for the opt-in mode so historical runs retain their dependency path.
    from .replay import TRACE_POLICY
    basis=dict(contract='common_start_v1',solver_seeds=sorted(set(seeds.values())))
    restart_seeds=[]
    for index in range(2):
        encoded=json.dumps(dict(basis,restart_index=index),sort_keys=True,separators=(',',':')).encode()
        restart_seeds.append(int.from_bytes(hashlib.sha256(encoded).digest()[:4],'big'))
    if len(set(restart_seeds))!=2: raise ValueError('derived probe restart seeds collide')
    frozen=dict(research_contract_version='common_start_v1',research_prompt_version='common_start_v1',
                trace_policy=copy.deepcopy(TRACE_POLICY),probe_restart_seeds=restart_seeds,
                state_selection_seed=int(hashlib.sha256(json.dumps(seeds,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()[:8],16),
                probe_seed_derivation='SHA256 canonical JSON(contract, sorted unique solver_seeds, restart_index=0,1); first 4 bytes big-endian uint32')
    for key,value in frozen.items():
        if key in c and c[key]!=value: raise ValueError('frozen research contract mismatch: '+key)
        c[key]=value
    return c

def validate(c,*,formal=True):
    iteration_mode=c.get('stop_rule')=='iterations'
    unlimited={'max_case_runs','max_wallclock_s','max_llm_requests','max_llm_tokens','max_analysis_seconds'} if iteration_mode else set()
    missing=[k for k in REQUIRED if k not in unlimited|({'rates'} if iteration_mode else set()) and (k not in c or c[k] is None)]
    if missing: raise ValueError('missing required configuration: '+', '.join(missing))
    for k in ('model_snapshot','model','provider','endpoint','api_key_env','run_dir'):
        if not isinstance(c[k],str) or not c[k].strip(): raise ValueError('nonempty identity required: '+k)
    if c['mode'] not in MODES or c['task'] not in ('tsp','fssp'): raise ValueError('invalid mode/task')
    if c.get('evaluation_policy','full_contrasts') not in ('full_contrasts','joint_only'): raise ValueError('invalid evaluation_policy')
    if c.get('evaluation_policy')=='joint_only' and (c['mode']!='CEF' or not iteration_mode): raise ValueError('joint_only requires CEF iteration mode and is labeled as a separate experimental variant')
    if type(c.get('transport_retry_limit',0)) is not int or not 0<=c.get('transport_retry_limit',0)<=2: raise ValueError('transport_retry_limit must be an integer in 0..2')
    if 'manual_rounds' in c and type(c['manual_rounds']) is not bool:raise ValueError('manual_rounds must be boolean')
    if c.get('manual_rounds'):
        if not (iteration_mode and c.get('alternating_agents') and c.get('measured_research_loop_v1') and c.get('max_iterations') is None):
            raise ValueError('manual rounds require measured alternating iteration mode with max_iterations=None')
    elif iteration_mode and (type(c.get('max_iterations')) is not int or c['max_iterations']<=0): raise ValueError('positive integer max_iterations required')
    defaults=dict(max_llm_requests=120,max_llm_tokens=250000,max_analysis_seconds=120,analysis_timeout_s=15,max_analysis_per_round=2,max_profile_instances=4,solver_time_limit_s=60)
    c={**defaults,**c}
    c=freeze_research_contract(c)
    if type(c.get('evaluation_workers',1)) is not int or c.get('evaluation_workers',1)<1: raise ValueError('evaluation_workers must be a positive integer')
    if 'diagnostic_run' in c and type(c['diagnostic_run']) is not bool: raise ValueError('diagnostic_run must be a boolean')
    if c.get('diagnostic_run'):
        if formal: raise ValueError('diagnostic configuration cannot be validated as a formal experiment')
        if type(c.get('diagnostic_train_limit')) is not int or not 1<=c['diagnostic_train_limit']<=64:
            raise ValueError('diagnostic_train_limit must be an integer in 1..64')
    if iteration_mode:
        for k in unlimited: c[k]=None
        c['rates']=None
    for k in ('max_case_runs','test_repeats','max_llm_requests','max_llm_tokens','max_output_tokens','max_analysis_per_round','max_profile_instances'):
        if k in unlimited: continue
        if type(c[k]) is not int or c[k]<=0: raise ValueError('positive integer required: '+k)
    for k in ('max_wallclock_s','max_analysis_seconds','analysis_timeout_s','solver_time_limit_s','request_timeout_s'):
        if k in unlimited: continue
        if isinstance(c[k],bool) or not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<=0: raise ValueError('positive finite configuration: '+k)
    if formal and c['solver_time_limit_s']!=60: raise ValueError('formal CEF requires 60s external limit')
    if c['max_profile_instances']>4 or c['analysis_timeout_s']>15: raise ValueError('tool cap exceeds specification')
    for k in ('environment_manifest','baseline_source_manifest','reference_manifest','evaluation_seed_manifest','data_manifest'):
        if not isinstance(c[k],dict) or not c[k]: raise ValueError('nonempty manifest required: '+k)
    if not iteration_mode:
        for k in ('input_per_million','cached_input_per_million','output_per_million','currency','source'):
            if k not in c['rates']: raise ValueError('missing frozen price: '+k)
        for k in ('input_per_million','cached_input_per_million','output_per_million'):
            if isinstance(c['rates'][k],bool) or not isinstance(c['rates'][k],(int,float)) or not math.isfinite(c['rates'][k]) or c['rates'][k]<0: raise ValueError('invalid rate')
    if c['model']!=c['model_snapshot']: raise ValueError('model must equal explicit frozen snapshot')
    if c.get('agent_tool_loop_v1'):
        if not c.get('alternating_agents') or not c.get('integrated_feedback_v1'):
            raise ValueError('tool loop requires integrated alternating workflow')
        keys=('tool_max_output_tokens',) if c.get('agent_task_runtime_v2') and c.get('tool_max_turns') is None else ('tool_max_turns','tool_max_output_tokens')
        for key in keys:
            if type(c.get(key)) is not int or c[key]<1:raise ValueError('positive integer required: '+key)
        if isinstance(c.get('tool_timeout_s'),bool) or not isinstance(c.get('tool_timeout_s'),(int,float)) or not math.isfinite(c['tool_timeout_s']) or c['tool_timeout_s']<=0:
            raise ValueError('positive tool_timeout_s required')
    if c.get('measured_research_loop_v1'):
        if not (c.get('alternating_agents') and c.get('integrated_feedback_v1') and c.get('agent_task_runtime_v2')):
            raise ValueError('measured research requires integrated alternating runtime v2')
        capacity=c.get('tool_context_window_tokens')
        if type(capacity)is not int or capacity<=c.get('tool_max_output_tokens',0):
            raise ValueError('context capacity must exceed the reserved model output')
    if c.get('exploration_efficiency_v1') and not c.get('measured_research_loop_v1'):
        raise ValueError('efficient exploration requires the measured research loop')
    if c.get('alternating_agents'):
        if c['task']!='tsp' or c.get('evaluation_policy')!='joint_only' or not iteration_mode:
            raise ValueError('alternating agents require TSP joint-only iteration mode')
        warm=c.get('warm_start_receipt',{})
        if (not warm or warm.get('path')!=c.get('warm_start_source') or
                file_hash(warm['path'])!=warm['sha256']):raise ValueError('frozen warm-start source mismatch')
        initial=c.get('alternating_initial_evaluation')
        if initial and file_hash(initial['path'])!=initial['sha256']:raise ValueError('initial data feedback changed')
        from .data import SUPPLEMENT_PROTOCOL
        if c['data_manifest'].get('training_protocol')!=SUPPLEMENT_PROTOCOL:
            raise ValueError('alternating run requires an explicit original64+supplement64 initial binding')
    from .model import sanitize_identity
    sanitize_identity(c['endpoint'])
    if c['route'] not in ('direct','environment_proxy') or c['wire_api'] not in ('chat_completions','responses'): raise ValueError('explicit supported transport required')
    if not isinstance(c['model_sampling'],dict) or set(c['model_sampling'])-{'temperature','top_p','seed','reasoning_effort','thinking','reasoning'}: raise ValueError('invalid sampling configuration')
    if c['model_seed_support'] not in ('supported','unsupported'): raise ValueError('model seed support must be explicit')
    if 'stream' in c and type(c['stream']) is not bool: raise ValueError('stream must be a boolean')
    if c['model_seed_support']=='supported' and 'seed' not in c['model_sampling']: raise ValueError('model sampling seed missing')
    if c['model_seed_support']=='unsupported' and 'seed' in c['model_sampling']: raise ValueError('unsupported model seed supplied')
    for id,seed in c['evaluation_seed_manifest'].items():
        if type(seed) is not int or not 0<=seed<2**32: raise ValueError('invalid solver seed: '+id)
    if formal:
        for k in ('python','dependencies','hardware','worker_resources'):
            if not c['environment_manifest'].get(k): raise ValueError('missing actual environment '+k)
        for k in ('source_version','algorithm_boundary','run_parameters'):
            if not c['baseline_source_manifest'].get(k): raise ValueError('missing baseline '+k)
        for k in ('training','test','distance_rule','reference_files'):
            if not c['reference_manifest'].get(k): raise ValueError('missing reference provenance '+k)
        resources=c['environment_manifest']['worker_resources']
        if resources.get('memory_mb')!=c.get('memory_mb',1024) or resources.get('concurrency')!=c.get('evaluation_workers',1) or resources.get('threads')!=1: raise ValueError('worker resource manifest does not match actual runner')
    rows=c['data_manifest'].get('train',[])
    protocol=c['data_manifest'].get('training_protocol')
    if protocol is not None:
        if c.get('diagnostic_run') and c['diagnostic_train_limit']<2:
            raise ValueError('augmented diagnostic requires at least two cases')
        from .data import audit_training
        audit_training(c['task'],rows,protocol)
        binding=c['data_manifest'].get('supplement_manifest',{})
        if not binding or file_hash(binding['path'])!=binding['sha256']:
            raise ValueError('frozen supplement manifest changed')
        supplement=load_json(binding['path'])
        if supplement.get('protocol')!=protocol or supplement.get('diagnostic') is not False:
            raise ValueError('formal frozen supplemental manifest required; diagnostic data rejected')
        if supplement['train']!=rows[64:]: raise ValueError('supplement rows differ from frozen manifest')
    elif formal and len(rows)!=64: raise ValueError('formal train requires all 64 instances')
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate training ids')
    testids={r['id'] for r in c['data_manifest'].get('test',[])}
    if set(ids)&testids: raise ValueError('train/test overlap')
    for row in rows:
        if row.get('split')!='train': raise ValueError('search may only load train rows')
        if 'path' in row and file_hash(row['path'])!=row['sha256']: raise ValueError('instance hash mismatch: '+row['id'])
        if formal and ('instance' in row or not row.get('source')): raise ValueError('formal rows require verified source files')
        if row['id'] not in c['evaluation_seed_manifest']: raise ValueError('missing solver seed: '+row['id'])
    for ref in c['reference_manifest'].get('reference_files',[]):
        if file_hash(ref['path'])!=ref['sha256']: raise ValueError('reference provenance hash mismatch')
    for source in c.get('implementation_source_manifest',[]):
        if file_hash(source['path'])!=source['sha256']: raise ValueError('implementation source changed since freeze: '+source['path'])
    for dependency in c.get('reference_backend_manifest',{}).get('files',[]):
        if file_hash(dependency['path'])!=dependency['sha256']:raise ValueError('reference backend dependency changed: '+dependency['path'])
    return c
