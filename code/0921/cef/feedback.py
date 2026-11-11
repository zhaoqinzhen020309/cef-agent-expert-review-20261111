"""Finite-sample contrasts in the task's unchanged fitness unit."""
import math
import ast

def compact_tool_observation(observation):
    if observation.get('action')!='compare_from_states': return observation
    result=observation['result']
    return dict(observation,result={k:v for k,v in result.items() if k not in ('run_results','records')})

def planner_history_observation(payload):
    """Replay measured observations, without repeating entire old search states."""
    view={k:payload[k] for k in ('fitness_metric','search_control','alternating') if k in payload}
    for name in ('active_parent','global_best'):
        value=payload.get(name)
        if name in payload:
            view[name]=None if value is None else {k:value[k] for k in
                ('candidate_id','parent_id','source_hash','fitness','fitness_display','training_group_scores',
                 'evaluation_epoch','fitness_evaluation_hash','historical_evaluation_hash') if k in value}
    if 'recent_events' in payload:
        keys=('event_id','parent_id','variant_ids','scores','score_unit','scores_display',
              'contrasts','missing_fields','global_best_updated','training_group_scores','evaluation_hash','current_dataset')
        view['recent_events']=[{k:event[k] for k in keys if k in event} for event in payload['recent_events']]
    if 'tool_observations' in payload:
        view['tool_observations']=[compact_tool_observation(x) for x in payload['tool_observations']]
    if 'common_start' in payload:
        view['common_start']=dict(latest_results=[{k:v for k,v in r.items() if k not in ('run_results','records')}
            for r in payload['common_start'].get('latest_results',[])])
    return view

def changed_definitions(parent,child):
    def inventory(source):
        functions={}; constants={}
        for node in ast.parse(source).body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                functions[node.name]=ast.dump(node,include_attributes=False)
            elif isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name,ast.Name): constants[name.id]=ast.dump(node,include_attributes=False)
        return functions,constants
    pf,pc=inventory(parent); cf,cc=inventory(child)
    return dict(functions=sorted(k for k in pf.keys()|cf.keys() if pf.get(k)!=cf.get(k)),constants=sorted(k for k in pc.keys()|cc.keys() if pc.get(k)!=cc.get(k)))


def choose(scores):
    eligible={k:v for k,v in scores.items() if v is not None and math.isfinite(v)}
    if not eligible: return None
    best=next(iter(eligible))
    for k,v in eligible.items():
        if v < eligible[best]-1e-9: best=k
    return best


def contrast(scores):
    def diff(a,b):
        x,y=scores.get(a),scores.get(b)
        return None if x is None or y is None else x-y
    out=dict(gA=diff('P','PA'))
    if 'PB' in scores:
        out.update(gB=diff('P','PB'),gAB=diff('P','PAB'),gA_if_B=diff('PB','PAB'),gB_if_A=diff('PA','PAB'))
        out['I']=None if any(out[k] is None for k in ('gA','gB','gAB')) else out['gAB']-out['gA']-out['gB']
    return out
