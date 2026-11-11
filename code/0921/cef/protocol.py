"""Validate controller replies without guessing missing design decisions."""
RUNTIME_CONTRACT=dict(
    timing_rule='The external case limit is strict. Leave return-time headroom inside time_limit_s, e.g. reserve max(0.02, 0.02*time_limit_s). Check the soft deadline inside expensive inner loops and return a complete feasible solution before the hard limit.',
    allowed_imports=['math','random','time','itertools','functools','collections','heapq','numpy'],
    denied_builtins=['open','eval','exec','compile','input','breakpoint','help','globals','locals','vars','getattr','setattr','delattr'],
    denied_attributes='Any attribute starting with __, including .__getitem__ and .__setitem__. Use normal indexing/assignment instead; e.g. key=lambda j: row[j], not key=row.__getitem__.',
    denied_capabilities=['filesystem','network','process creation','reflection'],
    dependency_rule='Each intended standalone version needs its imports/helpers. If intents are inseparable, propose one edit or explicitly record joint-only validity.')


def validate_planner(reply, config, context=None):
    if not isinstance(reply,dict): raise ValueError('Planner reply must be a JSON object')
    action=reply.get('action')
    permitted=('inspect','analyze','profile','revise','finish','compare_from_states')
    if config.get('alternating_agents'):permitted+=('advance_round','remeasure_candidate')
    if action not in permitted: raise ValueError('invalid Planner action')
    if action=='compare_from_states':
        if not config.get('common_start_probe'): raise ValueError('common-start tool disabled')
        for key in ('reference_candidate_id','candidate_id','question'):
            if not isinstance(reply.get(key),str) or not reply[key].strip(): raise ValueError('comparison requires '+key)
        if reply['reference_candidate_id']==reply['candidate_id']: raise ValueError('comparison requires two different programs')
    if action=='finish' and config.get('stop_rule')=='iterations':
        raise ValueError('finish is not permitted in iteration-controlled search')
    for key in ('evidence_ids','previous_event_update'):
        if key in reply and not isinstance(reply[key],list): raise ValueError(key+' must be a list')
    if not all(isinstance(value,str) for value in reply.get('evidence_ids',[])):
        raise ValueError('evidence_ids must contain strings')
    if 'resource_request' in reply and not isinstance(reply['resource_request'],dict):
        raise ValueError('resource_request must be an object')
    if action in ('inspect','profile','revise','remeasure_candidate'):
        if not isinstance(reply.get('parent_or_target_id'),str) or not reply['parent_or_target_id']:
            raise ValueError('parent_or_target_id must be a nonempty string')
    if action=='revise':
        count=reply.get('edit_count')
        if type(count) is not int or count not in (1,2):
            raise ValueError('revise requires edit_count as integer 1 or 2; the controller cannot infer it from proposed_intents')
        if config['mode']=='B0' and count!=1: raise ValueError('B0 requires edit_count=1')
        if config.get('evaluation_policy')=='joint_only' and count!=2: raise ValueError('joint_only requires edit_count=2')
        if not isinstance(reply.get('proposed_intents'),list) or not reply['proposed_intents']:
            raise ValueError('revise requires a nonempty proposed_intents list')
    if action=='analyze' and not isinstance(reply.get('analysis_code'),str):
        raise ValueError('analyze requires analysis_code')
    if action=='profile':
        count=reply.get('resource_request',{}).get('instances')
        if type(count) is not int or not 1<=count<=config['max_profile_instances']:
            raise ValueError('profile requires integer resource_request.instances within the tool limit')
    for update in reply.get('previous_event_update',[]):
        if not isinstance(update,dict) or not isinstance(update.get('card_id'),str) or not isinstance(update.get('interpretation'),str):
            raise ValueError('experience update requires card_id and interpretation strings')
        if update.get('status') not in ('待复查','本次观察支持','出现不一致'): raise ValueError('invalid experience status')
        refs=update.get('evidence_ids',[])
        if not isinstance(refs,list) or not all(isinstance(value,str) for value in refs): raise ValueError('invalid experience evidence_ids')
    if context is None: return
    if action not in context['allowed_actions']: raise ValueError('action is not allowed in this comparison mode')
    candidates=context['candidate_eligibility']; target=reply.get('parent_or_target_id')
    known_evidence=set(context['events']+context['cards']+context.get('probes',[])+context.get('tool_evidence',[]))
    if not set(reply.get('evidence_ids',[]))<=known_evidence: raise ValueError('unknown evidence ID; use IDs exposed in action_context')
    if action=='inspect' and target not in set(candidates)|known_evidence: raise ValueError('unknown inspect target')
    if action=='remeasure_candidate' and target not in candidates:raise ValueError('unknown historical candidate')
    if action=='remeasure_candidate' and target not in context.get('remeasurable_candidate_ids',[]):
        raise ValueError('candidate already measured under the current training epoch')
    if action=='profile':
        if target not in candidates: raise ValueError('unknown profile candidate; known failed candidates ARE permitted')
        if reply['resource_request']['instances']>context['max_profile_instances']: raise ValueError('profile requests more instances than available')
    if action=='revise' and not candidates.get(target,False) and target not in context.get('bootstrap_parent_ids',[]):
        raise ValueError('revision parent must be an eligible known candidate; inspect/profile failed code for evidence, then choose an eligible parent explicitly')
    if action=='compare_from_states':
        pair=(reply['reference_candidate_id'],reply['candidate_id'])
        if not any((r['reference_candidate_id'],r['candidate_id'])==pair for r in context.get('probe_pairs',[])):
            raise ValueError('comparison pair unavailable; use a listed measured training pair')
    goal=reply.get('revision_goal','ordinary')
    if 'integration_probe_id' in reply and goal!='integrate_probe': raise ValueError('integration_probe_id requires revision_goal=integrate_probe')
    if goal not in ('ordinary','integrate_probe'): raise ValueError('unknown revision_goal')
    if goal=='integrate_probe':
        if action!='revise' or not config.get('common_start_probe'): raise ValueError('integration is a revision in common-start mode')
        pid=reply.get('integration_probe_id')
        if not any(r['event_id']==pid and r['candidate_id']==target for r in context.get('integration_opportunities',[])):
            raise ValueError('integration must target an unused supported diagnostic candidate')
        if pid not in reply.get('evidence_ids',[]): raise ValueError('integration must cite its diagnostic')
    for update in reply.get('previous_event_update',[]):
        if update['card_id'] not in context['cards']: raise ValueError('unknown experience card')
        if not set(update.get('evidence_ids',[]))<=known_evidence: raise ValueError('unknown experience evidence ID')
