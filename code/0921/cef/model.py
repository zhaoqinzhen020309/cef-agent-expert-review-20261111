"""Official SDK transport with metering and recorded, bounded response repair."""
import asyncio,json,os,time,urllib.parse
import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, APIConnectionError, APIStatusError
from .store import canonical
from .ledger import BudgetError
from .errors import exception_chain
from .feedback import planner_history_observation

PLANNER='''You are the sole frozen-LLM CEF research controller. Choose inspect/analyze/profile/revise/finish. You may change all internal solver logic, not evaluator, data or budgets. Edit A and B are arbitrary design intents, NOT fixed slots. Joint validity/benefit may differ from each single. Facts are immutable; explanations are conditional, not causal proof. Use prior contrasts in the next design when relevant; explicitly explain adoption/rejection of cited evidence. New directions without precedent are allowed. Return one JSON object with previous_event_update (list of {card_id,interpretation,status,evidence_ids}), action, parent_or_target_id, issue, evidence_ids, proposed_intents, expected_observations, resource_request. revise additionally requires edit_count 1 or 2. profile resource_request.instances <=4. analyze additionally requires analysis_code defining analyze(events) and analysis_question; no external IO. Legal historical parents are selectable using inspect. Read full parent source before designing precise intents. Finish when an action cannot fit budgets. Do not fabricate measurements. Status values: 待复查/本次观察支持/出现不一致.'''
EDITOR='''Implement only the requested complete Python solver change. Same frozen model as Planner, independent short context. All internal logic may change, including whole algorithm replacement in a single edit. Return JSON package_id,parent_id,parent_hash,issue,evidence_ids,edits,joint_reason (two edits) or single_edit_reason (one). Each edit: edit_id A/B, purpose,prediction,exact_replacements list of {old,new}. All old anchors must match exactly once in the ORIGINAL parent text; all replacement spans disjoint. Any helper belongs to that intent. If inseparable, request single edit rather than pretend independence. Do not modify evaluator/resources. Preserve solve(instance,time_limit_s,seed)->{"solution": permutation}. Available standard imports: math,random,time,itertools,functools,collections,heapq,numpy. No filesystem/network/process/reflection. Return complete replacement strings, not placeholders. Explicitly use supplied relevant contrasts and selected experience in actual code or explain nonuse in purpose.'''

PLANNER+=' For action="revise", edit_count is a REQUIRED TOP-LEVEL INTEGER, not a field inside proposed_intents. Minimal revision shape: {"action":"revise","parent_or_target_id":"seed","edit_count":2,"issue":"...","proposed_intents":["intent A","intent B"],"evidence_ids":[],"previous_event_update":[],"expected_observations":[],"resource_request":{}}. Choose 1 or 2 yourself; never omit edit_count.'
PLANNER+=' Follow action_context and runtime_contract in the input. Profile any known candidate including failed ones; only revise eligible parents. Inspect execution_failures before hypothesizing why a score is null. Evidence IDs must be known event/card IDs, not candidate/execution IDs. resource_request.instances is an integer count, not a list of instance names.'
EDITOR+=' The runtime_contract in the input is authoritative. Dunder attribute access, including .__getitem__ and .__setitem__, is forbidden even for ordinary data operations; use normal indexing and indexed assignment. Include imports/helpers required by each intended standalone change, or explicitly declare inseparability.'

DATA_DESIGNER='''Design an executable supplemental Euclidean TSP instance generator using the supplied fixed solver and measured tool results. Return a JSON object with generator_source (complete Python source), parameters (JSON object), and reason. Implement generate(n_cities, seed, parameters) returning n_cities coordinate pairs in [0,1], with no duplicate cities. Use the supplied seed for all random draws. Available imports: math,random,time,itertools,functools,collections,heapq,numpy. Execution has no filesystem, network, subprocess or reflection capabilities. The controller runs your generator, independently constructs reference tours, executes the fixed complete solver, and returns measured reference-gap percentages or execution errors. Use those measurements to revise your generator. Larger measured mean reference gap indicates harder valid training data for this fixed solver. The starting geometric families are disclosed prior-method input. The output is training data, not test data. Submit the complete generator each iteration.'''

DATA_RULE_DESIGNER='''Design executable supplemental Euclidean TSP instance generators for the current complete solver, using the measured report. Return JSON with rules (a nonempty list) and reason. Each rule has name (short readable name), source (complete Python source), parameters (JSON object), weight (positive finite mixture weight). Each source implements generate(n_cities, seed, parameters), returning n_cities distinct finite coordinate pairs in [0,1]. All randomness uses the supplied seed. Available imports: math,random,time,itertools,functools,collections,heapq,numpy. No filesystem/network/process/reflection. The host selects an executable rule using a separate fixed random stream, records the exact version, constructs independent references and evaluates the current solver. Rule reports summarize these actual executions. Larger mean reference-gap percentage means harder data for this solver. Propose new generators or revise existing ones to improve measured challenge. The host adopts the proposed set only if its measured mean gap exceeds the retained set under the same current solver. These are training data. Submit complete source for every proposed rule.'''

def parse_json_content(text):
    text=text.strip()
    if text.startswith('```'):
        lines=text.splitlines()
        if lines[-1].strip()!='```': raise ValueError('incomplete JSON fence')
        text='\n'.join(lines[1:-1])
    value=json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(value,dict): raise ValueError('model JSON object required')
    return value

def sanitize_identity(endpoint):
    u=urllib.parse.urlsplit(endpoint)
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment: raise ValueError('credential-free HTTPS endpoint required')
    return endpoint.rstrip('/')

def complete_usage(usage):
    return isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens'))

class ModelFormatError(ValueError):
    """A finished, metered response whose text cannot form the required object."""
    def __init__(self,error,content):
        super().__init__(type(error).__name__+': '+str(error))
        self.content=content

class Model:
    # Native task transport recovery is journaled by ToolSession, outside tools.
    automatic_network_recovery = True
    def __init__(self,config,archive,ledger):
        self.config,self.archive,self.ledger=config,archive,ledger
        self.endpoint=sanitize_identity(config['endpoint'])
        self.key=os.environ.get(config['api_key_env'])
        if not self.key: raise ValueError('missing authentication environment: '+config['api_key_env'])
        if config.get('route') not in ('direct','environment_proxy'): raise ValueError('explicit route required')

    async def _receive_native(self,request,timeout,state):
        """Chat native tools retain the complete message, including DeepSeek reasoning."""
        async with asyncio.timeout(timeout):
            async with AsyncOpenAI(
                api_key=self.key,base_url=self.endpoint,max_retries=0,timeout=timeout,
                http_client=DefaultAsyncHttpxClient(trust_env=self.config['route']=='environment_proxy'),
            ) as client:
                response=await client.chat.completions.create(**request)
                if request.get('stream'):
                    message=dict(role='assistant',content=None)
                    calls={};started=time.monotonic();last=started
                    state['request_id']=response.response.headers.get('x-request-id')
                    state['stream_observation']=dict(chunks=0,first_chunk_s=None,last_chunk_s=None)
                    async with response:
                        async for chunk in response:
                            item=chunk.model_dump(exclude_none=True)
                            elapsed=time.monotonic()-started
                            observation=state['stream_observation'];observation['chunks']+=1
                            if observation['first_chunk_s'] is None:observation['first_chunk_s']=elapsed
                            observation['last_chunk_s']=elapsed
                            state['provider_response_id']=item.get('id') or state['provider_response_id']
                            if item.get('usage'):state['usage']=item['usage']
                            for choice in item.get('choices',[]):
                                if choice.get('index')!=0:raise ValueError('one native tool completion required')
                                delta=choice.get('delta',{})
                                if delta.get('role') not in (None,'assistant'):raise ValueError('invalid streamed assistant role')
                                for key in ('content','reasoning_content','refusal'):
                                    if delta.get(key) is not None:
                                        message[key]=(message.get(key) or '')+delta[key]
                                for call in delta.get('tool_calls',[]):
                                    index=call.get('index')
                                    if type(index)is not int or index<0:raise ValueError('invalid streamed tool index')
                                    target=calls.setdefault(index,dict(function=dict(name='',arguments='')))
                                    for key in ('id','type'):
                                        if call.get(key):
                                            if key in target and target[key]!=call[key]:raise ValueError('streamed tool identity changed')
                                            target[key]=call[key]
                                    for key in ('name','arguments'):
                                        target['function'][key]+=call.get('function',{}).get(key) or ''
                                if choice.get('finish_reason'):
                                    state['finish']=choice['finish_reason']
                            if self.config.get('console_progress') and time.monotonic()-last>=30:
                                print(f'Model receiving | {elapsed:.0f}s | chunks={observation["chunks"]}',flush=True)
                                last=time.monotonic()
                    if state['finish'] not in ('stop','tool_calls'):raise ValueError('incomplete native model stream: '+str(state['finish']))
                    if not complete_usage(state['usage']):raise ValueError('stream omitted complete usage')
                    if calls:
                        if sorted(calls)!=list(range(len(calls))):raise ValueError('noncontiguous streamed tool indexes')
                        for call in calls.values():
                            if not call.get('id') or call.get('type')!='function' or not call['function']['name']:
                                raise ValueError('incomplete streamed tool identity')
                        message['tool_calls']=[calls[i] for i in sorted(calls)]
                    if state['finish']=='tool_calls' and not calls:raise ValueError('stream finished without tool calls')
                    # No partial assistant message escapes to the tool executor.
                    state['message']=message
                    return
                item=response.model_dump(exclude_none=True)
                state['request_id']=getattr(response,'_request_id',None)
                state['provider_response_id']=item.get('id')
                state['usage']=item.get('usage')
                if len(item.get('choices',[]))!=1:
                    raise ValueError('one native tool completion required')
                choice=item['choices'][0]
                state['message']=choice['message']
                state['message'].setdefault('content',None)
                state['finish']=choice.get('finish_reason')

    def exchange(self,messages,tools,role):
        """One metered native chat exchange. No retry, JSON repair, or text parsing."""
        if self.config['wire_api']=='responses':
            raise ValueError('native CEF tool loop requires chat completions transport')
        if (self.config.get('rates') or {}).get('valid_until_utc'):
            from datetime import datetime,timezone
            if datetime.now(timezone.utc)>=datetime.fromisoformat(self.config['rates']['valid_until_utc']):
                raise BudgetError('frozen tariff expired; require new manifest/run')
        max_tokens=self.config.get('tool_max_output_tokens',self.config['max_output_tokens'])
        if type(max_tokens) is not int or max_tokens<1:
            raise ValueError('positive tool_max_output_tokens required')
        sampling=dict(self.config['model_sampling'])
        extra_body={k:sampling.pop(k) for k in ('thinking',) if k in sampling}
        request=dict(model=self.config['model_snapshot'],messages=messages,tools=tools,
                     max_tokens=max_tokens,stream=True,stream_options={'include_usage':True},**sampling)
        if extra_body: request['extra_body']=extra_body
        ceiling=len(canonical(dict(messages=messages,tools=tools)).encode())+max_tokens
        self.ledger.check('llm_requests',1); self.ledger.check('llm_tokens',ceiling)
        self.ledger.charge('llm_requests',1,role=role)
        hold_id='provider_'+str(len(self.archive.of('provider_request')))
        self.ledger.hold('llm_tokens',ceiling,hold_id)
        requested=self.archive.add('provider_request',dict(role=role,attempt=1,
            model=self.config['model_snapshot'],provider=self.config['provider'],endpoint=self.endpoint,
            route=self.config['route'],auth_source=self.config['api_key_env'],messages=messages,
            token_ceiling=ceiling,transport='openai_sdk_native_tools',
            request_parameters={k:v for k,v in request.items() if k!='messages'}))
        state=dict(message=None,usage=None,finish=None,request_id=None,provider_response_id=None)
        start=time.monotonic(); errors=None
        try:
            timeout=min(self.config['request_timeout_s'],self.ledger.remaining('wallclock_s'))
            if timeout<=0: raise BudgetError('wallclock exhausted before model dispatch')
            asyncio.run(self._receive_native(request,timeout,state))
            if state['finish'] not in ('stop','tool_calls'):
                raise ValueError('incomplete native model response: '+str(state['finish']))
            if not complete_usage(state['usage']):
                raise ValueError('provider omitted required usage or returned invalid token counts')
            if not isinstance(state['message'],dict) or state['message'].get('role')!='assistant':
                raise ValueError('provider omitted complete assistant message')
            return dict(message=state['message'],finish_reason=state['finish'],usage=state['usage'])
        except BaseException as error:
            errors=exception_chain(error,self.key)
            raise
        finally:
            usage=state['usage']
            if complete_usage(usage):
                input_tokens=usage['prompt_tokens']; output=usage['completion_tokens']
                cached=usage.get('prompt_cache_hit_tokens',usage.get('prompt_tokens_details',{}).get('cached_tokens',0))
                rates=self.config.get('rates')
                fee=((input_tokens-cached)*rates['input_per_million']+cached*rates['cached_input_per_million']+output*rates['output_per_million'])/1e6 if rates else None
                self.ledger.charge('llm_tokens',input_tokens+output,hold_id=hold_id,role=role,
                    input_tokens=input_tokens,cached_input_tokens=cached,output_tokens=output,
                    actual_usage=usage,fee=fee,currency=rates['currency'] if rates else None,usage_known=True)
            else:
                accounting_only=self.config.get('stop_rule')=='iterations'
                self.ledger.charge('llm_tokens',0 if accounting_only else ceiling,hold_id=hold_id,
                    role=role,usage_known=False,conservative_debit=not accounting_only,
                    reservation_upper_bound=ceiling,actual_usage=usage,fee=None)
            self.archive.add('provider_response',dict(role=role,attempt=1,request_seq=requested['seq'],
                request_id=state['request_id'],provider_response_id=state['provider_response_id'],
                message=state['message'],content=(state['message'] or {}).get('content'),parsed=None,
                usage=usage,finish_reason=state['finish'],elapsed=time.monotonic()-start,error_chain=errors,
                stream_observation=state.get('stream_observation')))

    async def _receive_with_progress(self,request,timeout,state):
        """Report elapsed time/counters without exposing generated or reasoning text."""
        task=asyncio.create_task(self._receive(request,timeout,state))
        started=time.monotonic()
        interval=self.config.get('model_progress_interval_s',30.)
        if isinstance(interval,bool) or not isinstance(interval,(float,int)) or interval<=0:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            raise ValueError('positive model progress interval required')
        try:
            while True:
                done,_=await asyncio.wait({task},timeout=interval)
                if done: return task.result()
                print(f'Model waiting | {time.monotonic()-started:.0f}s | output_chars={len(state["content"])} | reasoning_chars={state["reasoning_chars"]}',flush=True)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)

    async def _receive(self,request,timeout,state):
        # Async SDK calls let cancellation enforce a whole-request deadline on
        # Windows as well as socket timeouts, including a stalled/heartbeat stream.
        async with asyncio.timeout(timeout):
            async with AsyncOpenAI(
                api_key=self.key,
                base_url=self.endpoint,
                max_retries=0,
                timeout=timeout,
                http_client=DefaultAsyncHttpxClient(trust_env=self.config['route']=='environment_proxy'),
            ) as client:
                if self.config['wire_api']=='responses':
                    response=await client.responses.create(**request)
                else:
                    response=await client.chat.completions.create(**request)
                if not request['stream']:
                    item=response.model_dump(exclude_none=True)
                    state['usage']=item.get('usage')
                    if self.config['wire_api']=='responses':
                        state['content']=response.output_text
                        state['finish']='stop' if item.get('status')=='completed' else item.get('status')
                        if state['usage']:
                            u=state['usage']
                            state['usage']=dict(prompt_tokens=u['input_tokens'],completion_tokens=u['output_tokens'],prompt_tokens_details=u.get('input_tokens_details',{}),provider_raw=u)
                    else:
                        if len(response.choices)!=1: raise ValueError('one model completion required')
                        state['content']=response.choices[0].message.content or ''
                        state['finish']=response.choices[0].finish_reason
                    return
                async with response:
                    async for chunk in response:
                        item=chunk.model_dump(exclude_none=True)
                        state['stream_chunks']+=1
                        if self.config['wire_api']=='responses':
                            if item.get('type')=='response.output_text.delta': state['content']+=item['delta']
                            elif item.get('type')=='response.completed':
                                r=item['response']; state['finish']='stop' if r.get('status')=='completed' else r.get('status')
                                if r.get('usage'):
                                    u=r['usage']
                                    state['usage']=dict(prompt_tokens=u['input_tokens'],completion_tokens=u['output_tokens'],prompt_tokens_details=u.get('input_tokens_details',{}),provider_raw=u)
                            elif item.get('type') in ('response.failed','error'): raise ValueError('provider response failed')
                        else:
                            if item.get('usage'): state['usage']=item['usage']
                            for choice in item.get('choices',[]):
                                state['reasoning_chars']+=len(choice.get('delta',{}).get('reasoning_content') or '')
                                state['content']+=choice.get('delta',{}).get('content') or ''
                                if choice.get('finish_reason'): state['finish']=choice['finish_reason']

    def call(self,role,payload):
        for correction in range(2):
            try: return self._call_with_transport_retry(role,payload)
            except ModelFormatError as error:
                response_seq=self.archive.records[-1]['seq']
                failure=self.archive.add('model_format_failure',dict(role=role,response_seq=response_seq,
                    correction_attempt=correction,error=str(error)))
                if correction: raise
                payload=dict(payload,previous_response_text=error.content,protocol_error=str(error),
                    repair_instruction='Correct the JSON format of previous_response_text and return the COMPLETE JSON object. Preserve the same decision, evidence, intentions, code and exact replacement strings; do not invent measurements or substitute a new plan. Escape strings correctly and close every array/object. No commentary, placeholders or partial fragments. The original task constraints still apply. This is one recorded format correction, not a new search iteration.')
                self.archive.add('model_format_repair',dict(role=role,failure_seq=failure['seq'],
                    response_seq=response_seq,correction_limit=1,
                    policy='Request a complete corrected object; preserve and meter both responses; normal workflow validation still required.'))
                if self.config.get('console_progress'): print(f'{role.title()} | correcting JSON format',flush=True)

    def _call_with_transport_retry(self,role,payload):
        limit=self.config.get('transport_retry_limit',0)
        if type(limit) is not int or not 0<=limit<=2: raise ValueError('transport_retry_limit must be an integer in 0..2')
        for attempt in range(limit+1):
            try: return self._call_once(role,payload,attempt)
            except Exception as error:
                transient=isinstance(error,(APIConnectionError,httpx.TransportError,TimeoutError)) or (isinstance(error,APIStatusError) and (error.status_code in (408,409,429) or error.status_code>=500))
                if not transient or attempt==limit: raise
                delay=min(2**attempt,4)
                self.archive.add('transport_retry',dict(role=role,failed_attempt=attempt+1,next_attempt=attempt+2,
                    error_chain=exception_chain(error,self.key),delay_s=delay,
                    policy='Redispatch identical parameters, not resume partial generation; may generate different output and incur another charge. All failed usage records retained.'))
                if self.config.get('console_progress'): print(f'{role.title()} transport interrupted | retry {attempt+1}/{limit}',flush=True)
                time.sleep(delay)

    def _call_once(self,role,payload,attempt):
        if (self.config.get('rates') or {}).get('valid_until_utc'):
            from datetime import datetime,timezone
            if datetime.now(timezone.utc)>=datetime.fromisoformat(self.config['rates']['valid_until_utc']): raise BudgetError('frozen tariff expired; require new manifest/run')
        system=DATA_DESIGNER if role=='data_designer' else PLANNER if role=='planner' else EDITOR
        integrated=self.config.get('integrated_feedback_v1') or payload.get('interface_version')=='integrated_feedback_v1'
        if integrated and role=='data_designer':system=DATA_RULE_DESIGNER
        if self.config.get('alternating_agents') and role=='planner':
            system=system.replace('Finish when an action cannot fit budgets.','')
            system=system.replace('Choose inspect/analyze/profile/revise/finish.','Choose an action from action_context.allowed_actions.')
            system+=' Training changes at each explicit alternating round. Current candidate fitness is defined only for the evaluation_hash in alternating. To select an archived unmeasured program, request action=remeasure_candidate with parent_or_target_id first. In phase round_complete, choose existing diagnostic tools or action=advance_round to commit the round; another revise is unavailable. In phase algorithm, produce one full PAB revision using measured current-dataset feedback. Old results retain their original evaluation conditions.'
        if self.config.get('evaluation_policy')=='joint_only' and role in ('planner','editor'):
            if role=='planner':
                system=system.replace('only revise eligible parents.','revise measured eligible parents or the explicit unmeasured bootstrap parent.')
                system=system.replace('revise additionally requires edit_count 1 or 2.','revise requires edit_count=2.')
                system=system.replace('Choose 1 or 2 yourself; never omit edit_count.','Use edit_count=2; never omit it.')
                if self.config.get('alternating_agents'):
                    system+=' This experiment generates complete joint PAB candidates. Every revise has edit_count=2. Allowed actions follow the current phase in action_context; profile is unavailable. The explicit warm_start and archived programs become rankable only after measurement on the current dataset. Read current_program. Only joint scores/errors are observed; no individual edit gain or interaction is measured.'
                else:
                    system+=' This experiment generates ONLY complete joint PAB candidates. Every revise MUST have edit_count=2. Inspect/analyze/revise only; no profile. The bootstrap seed is source only, has no measured score and is not selectable as a final result. Read current_program. Only joint scores/errors are observed: never infer individual edit gains or interactions from them.'
            else:
                system=system.replace('If inseparable, request single edit rather than pretend independence.','Keep the requested two design intents; dependent edits are allowed because only their joint program will be evaluated. State dependency in joint_reason; do not claim standalone validity.')
                system+=' Produce exactly edits A and B, evaluated together as PAB only. No standalone effect or interaction is measured in this experiment.'
        if role=='planner' and self.config.get('stop_rule')=='iterations' and not self.config.get('alternating_agents'):
            system=system.replace('Finish when an action cannot fit budgets.','')
            system=system.replace('Choose inspect/analyze/profile/revise/finish.','Choose inspect/analyze/profile/revise.')
            system+=' Search is controlled ONLY by completed edit-evaluation iterations. Usage and money are observations, not limits. Do not choose finish; choose inspect/analyze/profile/revise until the external controller completes the requested iterations.'
        if role=='planner' and self.config.get('evaluation_policy')=='joint_only':
            system=system.replace('inspect/analyze/profile/revise','inspect/analyze/revise')
            system=system.replace('Profile any known candidate including failed ones; ','')
            system=system.replace('profile resource_request.instances <=4. ','')
        if self.config.get('common_start_probe') and role in ('planner','editor'):
            interface=' Preserve solve(instance,time_limit_s,seed,*,initial_solution=None,observer=None). On normal calls construct a solution; on provided initial_solution copy and use that solution directly, skipping construction. Both paths must share the SAME continuation implementation. When observer is provided, call observer(solution) on the first feasible starting solution and periodically in the outer search loop, including stagnation. It returns no search advice. No separate diagnostic-only optimizer, fake starting observation, evaluator/reference access or training-ID lookup.'
            if role=='planner':
                system=system.replace('Inspect/analyze/revise only; no profile.','Inspect/analyze/revise and, when action_context permits, compare_from_states; no profile.')
                system=system.replace('Only joint scores/errors are observed:','Formal feedback contains joint scores/errors, with separately labeled common-start diagnostic gains:')
                system+=' Common-start experiments are available ONLY for action_context.probe_pairs. To request one return action=compare_from_states, reference_candidate_id, candidate_id, question, evidence_ids and previous_event_update. The question records only why you want the measurement; it does not change what the tool executes. Read the returned comparison, metric definition, units, state counts, conditions and limits as the measured result. The tool compares complete solver continuations from identical frozen training solutions; it does not isolate edit effects or promote candidates. Only use a probe for integration when its exact ID appears in action_context.integration_opportunities; that list is created from the measured rule and already-used probes are excluded. An integration consumes a normal iteration; otherwise choose ordinary revise. No repeated probes of the same pair or tool calls after the configured iterations.'
            system+=interface
        messages=[dict(role='system',content=system)]
        # Planner persists the conversation, Editor gets an independent short context.
        if role=='planner' and not integrated:
            for r in self.archive.of('provider_response'):
                if r['role']=='planner' and r.get('parsed'):
                    observation=planner_history_observation(json.loads(r['user']))
                    messages.extend([dict(role='user',content=canonical(observation)),dict(role='assistant',content=r['content'])])
        elif role=='data_designer' and not integrated:
            for r in self.archive.of('provider_response'):
                if r['role']=='data_designer' and r.get('parsed'):
                    old=json.loads(r['user'])
                    observation={k:old[k] for k in ('iteration','previous_results') if k in old}
                    messages.extend([dict(role='user',content=canonical(observation)),dict(role='assistant',content=r['content'])])
        user=canonical(payload); messages.append(dict(role='user',content=user))
        responses=self.config['wire_api']=='responses'
        sampling=dict(self.config['model_sampling'])
        extra_body={k:sampling.pop(k) for k in ('thinking',) if k in sampling}
        stream=self.config.get('stream',True)
        request=(dict(model=self.config['model_snapshot'],input=messages,max_output_tokens=self.config['max_output_tokens'],stream=stream,**sampling) if responses else dict(model=self.config['model_snapshot'],messages=messages,max_tokens=self.config['max_output_tokens'],stream=stream,**sampling))
        if extra_body: request['extra_body']=extra_body
        if stream and not responses: request['stream_options']={'include_usage':True}
        # UTF-8 bytes upper-bound ordinary tokenization. Reserve worst-case output
        # and settle provider usage; never silently approximate successful usage.
        ceiling=len(canonical(messages).encode())+self.config['max_output_tokens']
        self.ledger.check('llm_requests',1); self.ledger.check('llm_tokens',ceiling)
        self.ledger.charge('llm_requests',1,role=role)
        hold_id='provider_'+str(len(self.archive.of('provider_request')))
        self.ledger.hold('llm_tokens',ceiling,hold_id)
        self.archive.add('provider_request',dict(role=role,attempt=attempt+1,model=self.config['model_snapshot'],provider=self.config['provider'],endpoint=self.endpoint,route=self.config['route'],auth_source=self.config['api_key_env'],messages=messages,token_ceiling=ceiling,transport='openai_sdk',request_parameters={k:v for k,v in request.items() if k not in ('messages','input')}))
        start=time.monotonic(); state=dict(content='',usage=None,finish=None,stream_chunks=0,reasoning_chars=0); parsed=None; errors=None
        try:
            timeout=min(self.config['request_timeout_s'],self.ledger.remaining('wallclock_s'))
            if timeout<=0: raise BudgetError('wallclock exhausted before model dispatch')
            receive=self._receive_with_progress if role=='data_designer' and self.config.get('console_progress') else self._receive
            asyncio.run(receive(request,timeout,state))
            if state['finish']!='stop':
                usage=state['usage'] or {}
                raise ValueError('incomplete model response: '+str(state['finish'])+f" (completion_tokens={usage.get('completion_tokens')}, reasoning_tokens={usage.get('completion_tokens_details',{}).get('reasoning_tokens')}, max_output_tokens={self.config['max_output_tokens']})")
            if not complete_usage(state['usage']): raise ValueError('provider omitted required usage or returned invalid token counts')
            try: parsed=parse_json_content(state['content'])
            except ValueError as error: raise ModelFormatError(error,state['content']) from error
            return parsed
        except Exception as error:
            errors=exception_chain(error,self.key)
            raise
        finally:
            usage=state['usage']
            if complete_usage(usage):
                input_tokens=usage.get('prompt_tokens',0); output=usage.get('completion_tokens',0)
                cached=usage.get('prompt_cache_hit_tokens',usage.get('prompt_tokens_details',{}).get('cached_tokens',0))
                amount=input_tokens+output
                rates=self.config.get('rates')
                fee=((input_tokens-cached)*rates['input_per_million']+cached*rates['cached_input_per_million']+output*rates['output_per_million'])/1e6 if rates else None
                self.ledger.charge('llm_tokens',amount,hold_id=hold_id,role=role,input_tokens=input_tokens,cached_input_tokens=cached,output_tokens=output,actual_usage=usage,fee=fee,currency=rates['currency'] if rates else None,usage_known=True)
            else:
                accounting_only=self.config.get('stop_rule')=='iterations'
                self.ledger.charge('llm_tokens',0 if accounting_only else ceiling,hold_id=hold_id,role=role,usage_known=False,conservative_debit=not accounting_only,reservation_upper_bound=ceiling,actual_usage=usage,fee=None)
            self.archive.add('provider_response',dict(role=role,attempt=attempt+1,user=user,content=state['content'],parsed=parsed,usage=usage,finish_reason=state['finish'],elapsed=time.monotonic()-start,error_chain=errors,stream_observation=dict(chunks=state['stream_chunks'],reasoning_chars=state['reasoning_chars'],content_chars=len(state['content']))))
