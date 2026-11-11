"""Durable single-task native tool loop; uncertain effects are never replayed."""
import copy
import json
import time
from pathlib import Path

from .store import Archive, canonical


class ToolSessionError(RuntimeError):
    pass


class TaskPaused(RuntimeError):
    pass


class TaskBlocked(RuntimeError):
    pass


class ToolSession:
    def __init__(self, root, model, tools, dispatch, *, role, system, payload, max_turns=None, finalize_turns=0,runtime_v2=False,pause_check=None,context_window_tokens=None,context_storage=None,completion_tools=None,checkpoint_tools=None):
        if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
            raise ValueError('max_turns must be None or a positive integer')
        if max_turns is None and finalize_turns:raise ValueError('unlimited tasks have no forced finalization')
        self.runtime_v2=runtime_v2
        if context_window_tokens is not None and (type(context_window_tokens)is not int or context_window_tokens<=0):
            raise ValueError('context_window_tokens must be a positive model capacity')
        self.context_window_tokens=context_window_tokens
        self.context_storage=context_storage
        self.completion_tools=set(completion_tools or [])
        self.checkpoint_tools=set(checkpoint_tools or ['checkpoint_context'])
        self.pause_check=pause_check
        self.model, self.dispatch = model, dispatch
        if type(finalize_turns) is not int or not 0 <= finalize_turns <= 8:
            raise ValueError('finalize_turns must be an integer in 0..8')
        self.finalize_turns=finalize_turns
        self.tools, self.role, self.max_turns = copy.deepcopy(tools), role, max_turns
        self.archive = Archive(Path(root) / 'session')
        config = getattr(model, 'config', {})
        model_identity = {k: config.get(k) for k in (
            'model_snapshot', 'endpoint', 'provider', 'route', 'wire_api',
            'model_sampling', 'max_output_tokens', 'tool_max_output_tokens')}
        identity = dict(role=role, system=system, payload=payload, tools=tools,
                        max_turns=max_turns, model=model_identity)
        if finalize_turns:identity['finalize_turns']=finalize_turns
        if runtime_v2:identity['runtime_version']=2
        if context_window_tokens is not None:identity['context_window_tokens']=context_window_tokens
        if completion_tools:identity['completion_tools']=sorted(self.completion_tools)
        if checkpoint_tools:identity['checkpoint_tools']=sorted(self.checkpoint_tools)
        if self.archive.records:
            if canonical(self.archive.of('identity')[0]) != canonical(identity):
                raise ToolSessionError('task identity mismatch')
        else:
            self.archive.add('identity', identity)
        if not self.archive.of('messages'):
            if len(self.archive.records) != 1:
                raise ToolSessionError('session has execution records but no initial messages')
            self.archive.add('messages', [dict(role='system', content=system),
                                          dict(role='user', content=canonical(payload))])
        self.messages = self.archive.of('messages')[-1]

    def _fail(self, reason):
        self.archive.add('incomplete', dict(reason=reason))
        raise ToolSessionError(reason)

    def _tools_for_turn(self,turn):
        if self.max_turns is not None and turn>self.max_turns:
            return [tool for tool in self.tools if tool['function']['name']=='submit_candidate']
        return self.tools

    def authorize_finalization(self):
        """Explicit recovery of a legacy, fully completed research-limit stop."""
        failures=[r for r in self.archive.records if r['kind']=='incomplete']
        if (not failures or failures[-1]['data']['reason']!='task turn limit reached without accepted submission'
                or self.archive.of('submitted') or self.archive.of('finalization_authorized')):
            raise ToolSessionError('only one explicit finalization of a legacy research-limit stop is allowed')
        starts=self.archive.of('model_started')
        if len(starts)!=self.max_turns or len(self.archive.of('model_completed'))!=len(starts):
            raise ToolSessionError('unfinished model exchange cannot enter finalization')
        completed={r['call_id'] for r in self.archive.of('tool_completed')}
        if any(r['call_id'] not in completed for r in self.archive.of('tool_started')):
            raise ToolSessionError('unfinished tool cannot enter finalization')
        self.archive.add('finalization_authorized',dict(failure_seq=failures[-1]['seq'],finalize_turns=2))
        self.finalize_turns=2

    def _enter_finalization(self,turn):
        if self.max_turns is None:return
        if turn!=self.max_turns+1:return
        message=dict(role='user',content=(
            'Research response budget completed. Submission phase: use the existing observations '
            'and work files to submit the current task deliverable with submit_candidate. '
            'Only submission is available in this phase. You may submit inline content or an existing '
            'candidate path. Host validation still applies; a rejected submission returns the actual '
            f'error for correction. This phase allows at most {self.finalize_turns} responses.'))
        saved=self.archive.of('finalization_started')
        if not saved:self.archive.add('finalization_started',dict(turn=turn,message=message))
        elif saved[0]['message']!=message:raise ToolSessionError('finalization message identity mismatch')
        self.messages.append(message)
        self.archive.add('messages',copy.deepcopy(self.messages))

    def _pending_attempt(self, started):
        provider = getattr(self.model, 'archive', None)
        boundary = started.get('provider_start_seq')
        retries = [r for r in self.archive.of('retry_started') if r['turn'] == started['turn']]
        if retries:
            boundary = retries[-1]['provider_start_seq']
        if provider is None or boundary is None:
            raise ToolSessionError('unknown started model outcome; missing provider journal')
        requests = [r for r in provider.records if r['kind'] == 'provider_request'
                    and r['seq'] >= boundary and r['data'].get('role') == self.role
                    and r['data'].get('messages') == self.messages
                    and r['data'].get('request_parameters', {}).get('tools') == self._tools_for_turn(started['turn'])]
        if len(requests) != 1:
            raise ToolSessionError('unknown started model outcome; require exactly one matching request')
        request = requests[0]
        responses = [r for r in provider.records if r['kind'] == 'provider_response'
                     and r['data'].get('request_seq') == request['seq']]
        if len(responses) != 1:
            raise ToolSessionError('unknown started model outcome; missing unique provider response')
        return request, responses[0]

    def authorize_failed_exchange_retry(self):
        """Authorize one redispatch of one fully recorded network failure.

        Calling this method is an explicit host operation. It never sends a
        request, erases failures, or authorizes a missing/uncertain outcome.
        The next run consumes this receipt before dispatch; further failures
        require a new explicit authorization and retain their original usage.
        """
        completed = {r['turn'] for r in self.archive.of('model_completed')}
        pending = [r for r in self.archive.of('model_started') if r['turn'] not in completed]
        if len(pending) != 1 or self.archive.of('submitted'):
            raise ToolSessionError('retry requires exactly one pending model exchange')
        self.messages = copy.deepcopy(self.archive.of('messages')[-1])
        request, response = self._pending_attempt(pending[0])
        result = response['data']
        if result.get('message'):
            raise ToolSessionError('recorded usable message cannot be retried; resume adopts it')
        allowed = ('APIConnectionError:', 'APITimeoutError:', 'RemoteProtocolError:',
                   'ConnectError:', 'ReadTimeout:')
        if not any(isinstance(error, str) and error.startswith(allowed)
                   for error in (result.get('error_chain') or [])):
            raise ToolSessionError('only fully recorded network failures may be explicitly retried')
        used = {r['authorization_seq'] for r in self.archive.of('retry_started')}
        if any(r['kind'] == 'retry_authorized' and r['seq'] not in used
               and r['data']['failed_request_seq'] == request['seq'] for r in self.archive.records):
            raise ToolSessionError('this failed exchange already has an unused retry authorization')
        receipt = dict(turn=pending[0]['turn'], failed_request_seq=request['seq'],
                       failed_response_seq=response['seq'], messages=copy.deepcopy(self.messages))
        self.archive.add('retry_authorized', receipt)
        return receipt

    def _recover_model(self, started):
        """Adopt an already fsync'd provider response after a host interruption."""
        try:
            request, recorded = self._pending_attempt(started)
        except ToolSessionError as error:
            self._fail(str(error))
        result = recorded['data']
        if not result.get('error_chain') and result.get('message'):
            reply = dict(message=result['message'], finish_reason=result['finish_reason'])
            if result.get('usage'):reply['usage']=result['usage']
        else:
            used = {r['authorization_seq'] for r in self.archive.of('retry_started')}
            matches = [r for r in self.archive.records if r['kind'] == 'retry_authorized'
                       and r['seq'] not in used and r['data']['turn'] == started['turn']
                       and r['data']['failed_request_seq'] == request['seq']
                       and r['data']['failed_response_seq'] == recorded['seq']
                       and r['data']['messages'] == self.messages]
            if not matches and self._can_reconnect(result):
                self._prepare_reconnect(started)
                return self._recover_model(started)
            if len(matches) != 1:
                self._fail('recorded model exchange failed; explicit retry authorization required')
            self.archive.add('retry_started', dict(turn=started['turn'], authorization_seq=matches[0]['seq'],
                provider_start_seq=len(self.model.archive.records)))
            reply = self._exchange(started)
        self.archive.add('model_completed', dict(turn=started['turn'], response=reply))
        return reply

    def _can_reconnect(self, result):
        # Classify the recorded top-level failure, not arbitrary nested text.
        errors = result.get('error_chain') or []
        return (getattr(self.model, 'automatic_network_recovery', False)
                and not result.get('message') and bool(errors)
                and errors[0].startswith(('APIConnectionError:', 'APITimeoutError:',
                    'RemoteProtocolError:', 'ConnectError:', 'ReadTimeout:')))

    def _prepare_reconnect(self, started):
        receipt = self.authorize_failed_exchange_retry()
        retries = sum(r['turn'] == started['turn'] for r in self.archive.of('retry_started'))
        delay = min(60, 2 ** min(retries + 1, 6))
        self.archive.add('network_reconnect_wait', dict(turn=started['turn'],
            failed_request_seq=receipt['failed_request_seq'], seconds=delay,
            policy='automatic transport retry; unknown usage retained; no tool replay'))
        print(f'{self.role} | connection interrupted; reconnect in {delay}s | response={started["turn"]}', flush=True)
        for _ in range(delay):
            self._check_pause()
            time.sleep(1)
        self._check_pause()

    def _exchange(self, started):
        while True:
            self._check_pause()
            try:
                return self.model.exchange(copy.deepcopy(self.messages),
                    self._tools_for_turn(started['turn']), self.role)
            except Exception as original:
                if not getattr(self.model, 'automatic_network_recovery', False):
                    raise
                try:
                    request, recorded = self._pending_attempt(started)
                except ToolSessionError:
                    raise original
                if not self._can_reconnect(recorded['data']):
                    raise
                self._prepare_reconnect(started)
                # Pause checks may append records; locate the durable receipt.
                authorization = next(r for r in reversed(self.archive.records)
                    if r['kind']=='retry_authorized' and r['data']['failed_request_seq']==request['seq'])
                self.archive.add('retry_started', dict(turn=started['turn'],
                    authorization_seq=authorization['seq'], provider_start_seq=len(self.model.archive.records)))

    def run(self):
        submitted = self.archive.of('submitted')
        if submitted:
            return submitted[-1]['value']
        if self.archive.of('blocked'):raise TaskBlocked(self.archive.of('blocked')[-1]['reason'])
        failures = [r for r in self.archive.records if r['kind'] == 'incomplete']
        authorizations = [r for r in self.archive.records if r['kind'] == 'retry_authorized']
        final=self.archive.of('finalization_authorized')
        final_recovery=bool(final and failures and final[-1]['failure_seq']==failures[-1]['seq'])
        if final_recovery:self.finalize_turns=final[-1]['finalize_turns']
        if failures and not final_recovery and (not authorizations or failures[-1]['seq'] > authorizations[-1]['seq']):
            raise ToolSessionError(failures[-1]['data']['reason'])
        starts = self.archive.of('tool_started')
        completed = {r['call_id']: r for r in self.archive.of('tool_completed')}
        if any(r['call_id'] not in completed for r in starts):
            self._fail('unknown started tool execution; refusing to repeat effects')
        # Rebuild messages from completed events, including crashes between a
        # completed tool/response fsync and its messages snapshot.
        initial = self.archive.of('messages')[0]
        self.messages = copy.deepcopy(initial)
        responses = {r['turn']: r['response'] for r in self.archive.of('model_completed')}
        for started in self.archive.of('model_started'):
            turn = started['turn']
            self._enter_finalization(turn)
            reply = responses.get(turn)
            if reply is None:
                reply = self._recover_model(started)
            value, accepted = self._consume(reply, turn, completed)
            if accepted:
                return value
        turns = len(self.archive.of('model_started'))
        while self.max_turns is None or turns < self.max_turns+self.finalize_turns:
            self._check_pause()
            turns += 1
            if self.runtime_v2 and getattr(self.model,'config',{}).get('console_progress'):
                print(f'{self.role} | response={turns}',flush=True)
            self._enter_finalization(turns)
            provider = getattr(self.model, 'archive', None)
            started = dict(turn=turns,
                provider_start_seq=len(provider.records) if provider is not None else None)
            self.archive.add('model_started', started)
            reply = self._exchange(started)
            self.archive.add('model_completed', dict(turn=turns, response=reply))
            value, accepted = self._consume(reply, turns, completed)
            if accepted:
                return value
        self._fail('task submission phase exhausted without accepted submission' if self.finalize_turns else 'task turn limit reached without accepted submission')

    def _check_pause(self):
        if self.pause_check and self.pause_check():
            self.archive.add('paused',dict(reason='pause requested; no new tool/model execution started'))
            raise TaskPaused('pause requested at a task boundary')

    def _consume(self, reply, turn, completed):
        previous_bytes=len(canonical(self.messages).encode('utf-8'))
        message = reply.get('message')
        finish = reply.get('finish_reason')
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            self._fail('invalid assistant message')
        if finish not in ('stop', 'tool_calls'):
            self._fail('incomplete model response: ' + str(finish))
        calls = message.get('tool_calls') or []
        if finish != 'tool_calls' or not calls:
            if self.runtime_v2:
                if message.get('refusal'):
                    self.archive.add('blocked',dict(reason=message['refusal']))
                    raise TaskBlocked(message['refusal'])
                # A normal text ending is not proof that execution is blocked.
                # Check the actual missing delivery, without judging the prose.
                self.messages.append(copy.deepcopy(message))
                self.messages.append(dict(role='user',content=(
                    'Task delivery check: a deliverable was not submitted to host validation. '
                    'The task is still active and the same tools are available. Continue investigation '
                    'or submit the deliverable with submit_candidate; use report_blocker if an actual '
                    'external condition prevents proceeding.')))
                if not any(r['turn']==turn for r in self.archive.of('delivery_check_failed')):
                    self.archive.add('delivery_check_failed',dict(turn=turn,submitted=False))
                self.archive.add('messages',copy.deepcopy(self.messages))
                self._manage_context(reply,turn,previous_bytes,set())
                return None,False
            self._fail('model stopped without submit_candidate')
        if not isinstance(calls, list):
            self._fail('invalid tool_calls')
        ids = [c.get('id') for c in calls if isinstance(c, dict)]
        previous_ids = {m['tool_call_id'] for m in self.messages if m.get('role') == 'tool'}
        if len(ids) != len(calls) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids) or previous_ids.intersection(ids):
            self._fail('invalid or reused tool call IDs')
        for index, call in enumerate(calls):
            function = call.get('function')
            if isinstance(function, dict) and function.get('name') == 'submit_candidate' and index != len(calls)-1:
                self._fail('submit_candidate must be the last call in its assistant response')
        self.messages.append(copy.deepcopy(message))
        self.archive.add('messages', copy.deepcopy(self.messages))
        names = {t['function']['name'] for t in self._tools_for_turn(turn)}
        automatic=None
        for call in calls:
            call_id = call['id']
            raw_function = call.get('function')
            function = raw_function if isinstance(raw_function, dict) else {}
            name = function.get('name')
            existing = completed.get(call_id)
            if existing:
                if existing['call'] != call:
                    self._fail('tool call identity mismatch on resume')
                result = existing['result']
            elif automatic is not None:
                result=dict(ok=False,accepted=False,not_executed=True,
                    reason='The host already verified this task goal; remaining calls were not executed.')
                record=dict(turn=turn,call_id=call_id,call=call,result=result)
                self.archive.add('tool_completed',record);completed[call_id]=record
            else:
                self._check_pause()
                self.archive.add('tool_started', dict(turn=turn, call_id=call_id, call=call))
                try:
                    if not isinstance(raw_function, dict):
                        raise ValueError('tool function must be a JSON object')
                    if call.get('type') != 'function' or name not in names:
                        raise ValueError('unknown or invalid function tool: ' + str(name))
                    args = json.loads(function['arguments'])
                    if not isinstance(args, dict):
                        raise ValueError('tool arguments must be a JSON object')
                    canonical(args)
                    result = self.dispatch(name, args)
                    canonical(result)
                except (ValueError, TypeError, KeyError) as error:
                    result = dict(error=dict(type=type(error).__name__, message=str(error)))
                record = dict(turn=turn, call_id=call_id, call=call, result=result)
                self.archive.add('tool_completed', record)
                completed[call_id] = record
            self.messages.append(dict(role='tool', tool_call_id=call_id, content=canonical(result)))
            self.archive.add('messages', copy.deepcopy(self.messages))
            if name in self.completion_tools and isinstance(result,dict) and result.get('task_completion',{}).get('accepted') is True:
                automatic=dict(call_id=call_id,value=result['task_completion']['value'],completion_kind='verified_goal')
            if name == 'submit_candidate' and isinstance(result, dict) and result.get('accepted') is True:
                if 'value' not in result:
                    self._fail('accepted submission missing host validated value')
                self.archive.add('submitted', dict(call_id=call_id, value=result['value']))
                return result['value'], True
            if self.runtime_v2 and name=='report_blocker' and result.get('blocked'):
                self.archive.add('blocked',result)
                raise TaskBlocked(result['reason'])
        if automatic is not None:
            self.archive.add('submitted',automatic)
            return automatic['value'],True
        checkpoints=[completed[c['id']]['result'] for c in calls
                     if isinstance(c.get('function'),dict) and c['function'].get('name') in self.checkpoint_tools
                     and completed[c['id']]['result'].get('context_checkpoint')]
        if self.runtime_v2 and checkpoints:
            checkpoint=checkpoints[-1]['context_checkpoint']
            recent=[dict(tool=c.get('function'),result=completed[c['id']]['result']) for c in calls]
            self.messages=copy.deepcopy(self.archive.of('messages')[0])+[dict(role='user',content=canonical(dict(
                continuation='Continue the same task from this saved checkpoint. Full prior executions remain readable by artifact ID.',
                checkpoint=checkpoint,recent_tool_results=recent)))]
            key=calls[-1]['id']
            if not any(r['call_id']==key for r in self.archive.of('context_restarted')):
                self.archive.add('context_restarted',dict(call_id=key,checkpoint=checkpoint))
            self.archive.add('messages',copy.deepcopy(self.messages))
        elif self.runtime_v2:
            self._manage_context(reply,turn,previous_bytes,{c['id'] for c in calls})
        return None, False

    def _manage_context(self,reply,turn,previous_bytes,current_call_ids):
        usage=reply.get('usage') or {}
        tokens=usage.get('prompt_tokens')
        if self.context_window_tokens is None or type(tokens)is not int:return
        output=getattr(self.model,'config',{}).get('tool_max_output_tokens',
            getattr(self.model,'config',{}).get('max_output_tokens',0))
        # Provider prompt usage is measured. Newly appended UTF-8 bytes are a
        # conservative growth allowance, not a tokenizer or a billing count.
        growth=max(0,len(canonical(self.messages).encode('utf-8'))-previous_bytes)
        if tokens+growth+output<self.context_window_tokens:return
        pruned=[]
        if self.context_storage:
            for message in self.messages:
                if message.get('role')!='tool' or message.get('tool_call_id') in current_call_ids:continue
                content=message.get('content','')
                pointer=dict(context_pruned=True,artifact_id='',
                    detail='Full original tool result is preserved. Read artifact_id or analyze it with run_python.')
                # Never replace a result by a longer pointer, or prune twice.
                try:old=json.loads(content)
                except (TypeError,ValueError):old=None
                if isinstance(old,dict) and old.get('context_pruned'):continue
                if len(content)<=len(canonical(pointer))+100:continue
                pointer['artifact_id']=self.context_storage(content)
                if isinstance(old,dict):
                    for key in ('ok','error'):
                        if key in old:pointer[key]=old[key]
                message['content']=canonical(pointer)
                pruned.append(dict(tool_call_id=message['tool_call_id'],artifact_id=pointer['artifact_id'],original_characters=len(content)))
        if pruned and not any(r['turn']==turn for r in self.archive.of('context_pruned')):
            self.archive.add('context_pruned',dict(turn=turn,results=pruned,prompt_tokens=tokens,
                context_window_tokens=self.context_window_tokens))
        # Use bytes only as an upper bound for the remaining serialized context;
        # do not subtract removed bytes from measured tokens as if interchangeable.
        if len(canonical(dict(messages=self.messages,tools=self.tools)).encode('utf-8'))+output>=self.context_window_tokens:
            notice=dict(role='user',content=canonical(dict(context_maintenance=dict(
                observed_prompt_tokens=tokens,context_window_tokens=self.context_window_tokens,
                reserved_output_tokens=output,
                next_action='Save current findings, open questions and work artifact IDs with checkpoint_context before continuing a long investigation. The goal and tools remain available; full raw results are retained.'))))
            self.messages.append(notice)
            if not any(r['turn']==turn for r in self.archive.of('context_pressure')):
                self.archive.add('context_pressure',dict(turn=turn,prompt_tokens=tokens,context_window_tokens=self.context_window_tokens))
        self.archive.add('messages',copy.deepcopy(self.messages))
