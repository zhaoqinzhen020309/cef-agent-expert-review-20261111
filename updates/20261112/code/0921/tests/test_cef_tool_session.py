"""Offline protocol tests: no provider requests are made."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from cef.tool_session import ToolSession, ToolSessionError
from cef.model import Model
from cef.store import Archive
from cef.ledger import BudgetLedger


TOOLS = [dict(type='function', function=dict(name='submit_candidate', parameters={'type':'object'})),
         dict(type='function', function=dict(name='check', parameters={'type':'object'}))]


def response(name='submit_candidate', args='{}', call_id='call_1', **extra):
    return dict(message=dict(role='assistant', content=None, reasoning_content='retained reasoning',
                tool_calls=[dict(id=call_id, type='function', function=dict(name=name, arguments=args))], **extra),
                finish_reason='tool_calls')


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []
        self.tool_sets=[]

    def exchange(self, messages, tools, role):
        self.messages.append(copy.deepcopy(messages))
        self.tool_sets.append(copy.deepcopy(tools))
        item = next(self.replies)
        if isinstance(item, BaseException):
            raise item
        return item


class SessionTests(unittest.TestCase):
    def test_legacy_limit_stop_can_explicitly_finalize_without_research_replay(self):
        calls=[]
        model=FakeModel([response('check',call_id='old')])
        dispatch=lambda n,a:calls.append(n) or dict(accepted=n=='submit_candidate',value=9)
        with self.assertRaisesRegex(ToolSessionError,'turn limit'):
            self.session(model,dispatch,max_turns=1).run()
        continued=FakeModel([response(call_id='submitted')])
        resumed=self.session(continued,dispatch,max_turns=1)
        resumed.authorize_finalization()
        self.assertEqual(resumed.run(),9)
        self.assertEqual(calls,['check','submit_candidate'])
        self.assertEqual(self.session(FakeModel([]),dispatch,max_turns=1).run(),9)

    def test_bounded_research_transitions_to_submission_only_then_repairs(self):
        model=FakeModel([response('check',call_id='research'),response(call_id='bad'),response(call_id='fixed')])
        seen=[]
        def dispatch(name,args):
            seen.append(name)
            return dict(accepted=len(seen)==3,value=7,error='missing field')
        session=self.session(model,dispatch,max_turns=1,finalize_turns=2)
        self.assertEqual(session.run(),7)
        self.assertEqual(seen,['check','submit_candidate','submit_candidate'])
        self.assertIn('submission',model.messages[1][-1]['content'].lower())
        self.assertEqual([t['function']['name'] for t in model.tool_sets[1]],['submit_candidate'])
        self.assertEqual(len(session.archive.of('finalization_started')),1)

    def test_final_submission_replay_does_not_repeat_research(self):
        model=FakeModel([response('check',call_id='research'),response(call_id='done')])
        calls=[]
        session=self.session(model,lambda n,a:calls.append(n) or dict(accepted=n=='submit_candidate',value=4),max_turns=1,finalize_turns=2)
        original=session.archive.add
        def crash(kind,data):
            if kind=='model_started' and data['turn']==2:raise KeyboardInterrupt()
            return original(kind,data)
        session.archive.add=crash
        with self.assertRaises(KeyboardInterrupt):session.run()
        resumed=self.session(model,lambda n,a:calls.append(n) or dict(accepted=True,value=4),max_turns=1,finalize_turns=2)
        self.assertEqual(resumed.run(),4)
        self.assertEqual(calls,['check','submit_candidate'])
        self.assertEqual(len(resumed.archive.of('finalization_started')),1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def session(self, model, dispatch, **kwargs):
        return ToolSession(self.root, model, TOOLS, dispatch, role='editor', system='test',
                           payload={'task': 1}, **kwargs)

    def test_error_and_rejected_submission_are_same_task_observations(self):
        model = FakeModel([response('check', 'bad', 'c1'), response(call_id='c2'), response(call_id='c3')])
        count = []
        def dispatch(name, args):
            count.append(name)
            return {'accepted': len(count) == 2, 'value': {'candidate': 7}, 'error':'invalid candidate'}
        self.assertEqual(self.session(model, dispatch).run(), {'candidate': 7})
        self.assertEqual(count, ['submit_candidate', 'submit_candidate'])
        replay = model.messages[-1]
        self.assertEqual(replay[2]['reasoning_content'], 'retained reasoning')
        self.assertEqual(replay[3]['tool_call_id'], 'c1')
        self.assertIn('error', replay[3]['content'])
        self.assertEqual(replay[5]['tool_call_id'], 'c2')

    def test_resume_identity_committed_before_initial_messages(self):
        session=self.session(FakeModel([]),lambda n,a:{})
        identity=session.archive.of('identity')[0]
        session.archive.path.write_text(json.dumps(dict(seq=0,kind='identity',time=0,data=identity))+'\n',encoding='utf-8')
        resumed=self.session(FakeModel([response()]),lambda n,a:{'accepted':True,'value':4})
        self.assertEqual(resumed.run(),4)

    def test_malformed_function_becomes_tool_error_and_can_be_corrected(self):
        for function in (None,[],4,'submit_candidate'):
            with self.subTest(function=function),tempfile.TemporaryDirectory() as root:
                malformed=response('check',call_id='bad')
                malformed['message']['tool_calls'][0]['function']=function
                model=FakeModel([malformed,response(call_id='good')])
                session=ToolSession(root,model,TOOLS,lambda n,a:{'accepted':True,'value':7},
                    role='editor',system='test',payload={})
                self.assertEqual(session.run(),7)
                self.assertEqual(model.messages[-1][-1]['tool_call_id'],'bad')
                self.assertIn('error',model.messages[-1][-1]['content'])

    def test_terminal_resume_never_repeats_calls(self):
        model = FakeModel([response()])
        calls = []
        dispatch = lambda n, a: calls.append(n) or {'accepted':True, 'value':3}
        self.assertEqual(self.session(model, dispatch).run(), 3)
        self.assertEqual(self.session(FakeModel([]), dispatch).run(), 3)
        self.assertEqual(calls, ['submit_candidate'])

    def test_resume_completed_tool_and_response_after_host_interruption(self):
        model = FakeModel([response('check', call_id='c1'), response(call_id='c2')])
        calls=[]
        dispatch=lambda n,a: calls.append(n) or {'accepted':True,'value':3}
        session=self.session(model,dispatch,max_turns=2)
        original=session.archive.add
        def crash(kind,data):
            if kind == 'messages' and data[-1].get('role') == 'tool':
                raise KeyboardInterrupt()
            return original(kind,data)
        session.archive.add=crash
        with self.assertRaises(KeyboardInterrupt):
            session.run()
        self.assertEqual(self.session(model,dispatch,max_turns=2).run(),3)
        self.assertEqual(calls,['check','submit_candidate'])
        self.assertEqual(len(model.messages[-1]),4)

    def test_unknown_model_outcome_does_not_retry_on_resume(self):
        with self.assertRaises(TimeoutError):
            self.session(FakeModel([TimeoutError('unknown')]), lambda n,a: {}).run()
        with self.assertRaisesRegex(ToolSessionError, 'unknown'):
            self.session(FakeModel([]), lambda n,a: {}).run()

    def test_unknown_tool_outcome_does_not_repeat(self):
        def dispatch(n, a):
            raise RuntimeError('infrastructure unavailable')
        with self.assertRaises(RuntimeError):
            self.session(FakeModel([response()]), dispatch).run()
        with self.assertRaisesRegex(ToolSessionError, 'unknown'):
            self.session(FakeModel([]), dispatch).run()

    def test_no_submission_and_truncation_fail_closed(self):
        for finish in ('stop', 'length'):
            with self.subTest(finish=finish), tempfile.TemporaryDirectory() as root:
                model = FakeModel([dict(message=dict(role='assistant',content='done'),finish_reason=finish)])
                with self.assertRaises(ToolSessionError):
                    ToolSession(root,model,TOOLS,lambda n,a: {},role='editor',system='test',payload={}).run()

    def test_identity_and_turn_limit_are_durable(self):
        model = FakeModel([response('check')])
        with self.assertRaisesRegex(ToolSessionError, 'turn'):
            self.session(model, lambda n,a: {'accepted':True,'value':5}, max_turns=1).run()
        with self.assertRaisesRegex(ToolSessionError, 'identity'):
            self.session(FakeModel([]), lambda n,a:{}, max_turns=2).run()

    def test_model_config_identity_cannot_change_on_resume(self):
        model=FakeModel([response()])
        model.config={'model_snapshot':'model-a','route':'direct','endpoint':'https://example.invalid'}
        self.session(model,lambda n,a:{'accepted':True,'value':3}).run()
        changed=FakeModel([])
        changed.config=dict(model.config,model_snapshot='model-b')
        with self.assertRaisesRegex(ToolSessionError,'identity'):
            self.session(changed,lambda n,a:{}).run()

    def test_submission_cannot_precede_more_mutations_in_same_response(self):
        item = response()
        item['message']['tool_calls'].append(response('check', call_id='c2')['message']['tool_calls'][0])
        calls=[]
        with self.assertRaises(ToolSessionError):
            self.session(FakeModel([item]), lambda n,a: calls.append(n) or {'accepted':True,'value':1}).run()
        self.assertEqual(calls, [])


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive=Archive(self.temp.name)
        self.ledger=BudgetLedger(self.archive,{},enforce_limits=False)
        self.config=dict(endpoint='https://example.invalid/v1',api_key_env='CEF_TEST_KEY',route='direct',
            wire_api='chat_completions',model_snapshot='test',provider='test',model_sampling={},
            max_output_tokens=100,request_timeout_s=2,stop_rule='iterations',stream=True,
            rates=dict(input_per_million=1,cached_input_per_million=.1,output_per_million=2,currency='USD'))
        with patch.dict('os.environ',CEF_TEST_KEY='secret-test-key'):
            self.model=Model(self.config,self.archive,self.ledger)

    def test_native_preserves_message_and_meters_actual_usage(self):
        requests=[]
        async def receive(request,timeout,state):
            requests.append(request)
            state.update(message=response()['message'],finish='tool_calls',request_id='req-1',
                         usage=dict(prompt_tokens=20,completion_tokens=10,prompt_cache_hit_tokens=5))
        self.model._receive_native=receive
        result=self.model.exchange([dict(role='user',content='test')],TOOLS,'editor')
        self.assertEqual(result['message'],response()['message'])
        self.assertTrue(requests[0]['stream'])
        self.assertEqual(requests[0]['stream_options'],{'include_usage':True})
        self.assertEqual(requests[0]['tools'],TOOLS)
        self.assertEqual(self.ledger.used('llm_requests'),1)
        self.assertEqual(self.ledger.used('llm_tokens'),30)
        charge=self.archive.of('charge')[-1]
        self.assertAlmostEqual(charge['fee'],35.5/1e6)
        self.assertEqual(self.archive.of('provider_response')[-1]['request_id'],'req-1')
        self.assertNotIn('secret-test-key',str(self.archive.records))

    def test_native_truncation_is_charged_and_never_retried(self):
        calls=[]
        async def receive(request,timeout,state):
            calls.append(1)
            state.update(message=dict(role='assistant',content='partial'),finish='length',
                         usage=dict(prompt_tokens=5,completion_tokens=100))
        self.model._receive_native=receive
        with self.assertRaisesRegex(ValueError,'incomplete'):
            self.model.exchange([],TOOLS,'editor')
        self.assertEqual(calls,[1])
        self.assertEqual(self.ledger.used('llm_tokens'),105)
        self.assertEqual(self.archive.of('provider_response')[-1]['finish_reason'],'length')

    def test_actual_sdk_boundary_keeps_reasoning_and_no_automatic_retry(self):
        captured={}
        item=dict(id='completion-id',choices=[dict(message=response()['message'],finish_reason='tool_calls')],
                  usage=dict(prompt_tokens=5,completion_tokens=10))
        class ProviderResponse:
            _request_id='header-request-id'
            response=type('HTTPResponse',(),{'headers':{'x-request-id':'header-request-id'}})()
            def model_dump(self,**kwargs):return copy.deepcopy(item)
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            def __aiter__(self):
                async def chunks():
                    delta=copy.deepcopy(item['choices'][0]['message'])
                    for index,call in enumerate(delta.get('tool_calls',[])):call['index']=index
                    value=dict(id=item['id'],choices=[dict(index=0,delta=delta,finish_reason='tool_calls')],usage=item['usage'])
                    yield type('Chunk',(),{'model_dump':lambda self,**kw:value})()
                return chunks()
        class Client:
            def __init__(self,**kwargs):
                captured.update(kwargs)
                self.chat=self;self.completions=self
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def create(self,**kwargs):
                captured['request']=kwargs
                return ProviderResponse()
        with patch('cef.model.AsyncOpenAI',Client),patch('cef.model.DefaultAsyncHttpxClient',lambda **kwargs:kwargs):
            result=self.model.exchange([],TOOLS,'editor')
        self.assertEqual(result['message']['reasoning_content'],'retained reasoning')
        self.assertEqual(captured['max_retries'],0)
        self.assertEqual(captured['http_client'],{'trust_env':False})
        self.assertEqual(self.archive.of('provider_response')[-1]['request_id'],'header-request-id')

    def test_unknown_usage_is_not_invented(self):
        async def receive(request,timeout,state):
            raise TimeoutError('unknown provider outcome')
        self.model._receive_native=receive
        with self.assertRaises(TimeoutError):self.model.exchange([],TOOLS,'editor')
        charge=self.archive.of('charge')[-1]
        self.assertFalse(charge['usage_known'])
        self.assertIsNone(charge['fee'])
        self.assertEqual(len(self.archive.of('provider_request')),1)

    def test_resume_adopts_durable_provider_response_without_redispatch(self):
        async def receive(request,timeout,state):
            state.update(message=response()['message'],finish='tool_calls',
                         usage=dict(prompt_tokens=5,completion_tokens=10))
        self.model._receive_native=receive
        root=Path(self.temp.name)/'task'
        calls=[]
        dispatch=lambda n,a:calls.append(n) or {'accepted':True,'value':9}
        session=ToolSession(root,self.model,TOOLS,dispatch,role='editor',system='test',payload={})
        original=session.archive.add
        def crash(kind,data):
            if kind=='model_completed': raise KeyboardInterrupt()
            return original(kind,data)
        session.archive.add=crash
        with self.assertRaises(KeyboardInterrupt):session.run()
        self.assertEqual(ToolSession(root,self.model,TOOLS,dispatch,role='editor',system='test',payload={}).run(),9)
        self.assertEqual(self.ledger.used('llm_requests'),1)
        self.assertEqual(calls,['submit_candidate'])

    def retry_session(self, receive):
        # Exercise the retained opt-out/manual recovery policy explicitly.
        self.model.automatic_network_recovery=False
        self.model._receive_native=receive
        return ToolSession(Path(self.temp.name)/'retry_task',self.model,TOOLS,
            lambda n,a:{'accepted':True,'value':17},role='editor',system='test',payload={},max_turns=1)

    def test_recorded_disconnect_requires_explicit_single_retry(self):
        calls=[]
        disconnected=type('RemoteProtocolError',(Exception,),{})
        async def receive(request,timeout,state):
            calls.append(copy.deepcopy(request['messages']))
            if len(calls)==1:raise disconnected('peer closed connection')
            state.update(message=response()['message'],finish='tool_calls',
                         usage=dict(prompt_tokens=5,completion_tokens=10))
        session=self.retry_session(receive)
        with self.assertRaises(disconnected):session.run()
        with self.assertRaises(ToolSessionError):self.retry_session(receive).run()
        retry=self.retry_session(receive)
        receipt=retry.authorize_failed_exchange_retry()
        self.assertEqual(receipt['turn'],1)
        self.assertEqual(retry.run(),17)
        self.assertEqual(calls[0],calls[1])
        self.assertEqual(self.ledger.used('llm_requests'),2)
        self.assertEqual(len(retry.archive.of('model_started')),1)
        self.assertEqual(len(retry.archive.of('retry_started')),1)

    def test_repeated_disconnect_needs_new_authorization(self):
        disconnected=type('ReadTimeout',(Exception,),{})
        calls=[]
        async def receive(request,timeout,state):
            calls.append(1)
            raise disconnected('read failed')
        session=self.retry_session(receive)
        with self.assertRaises(disconnected):session.run()
        session.authorize_failed_exchange_retry()
        with self.assertRaises(disconnected):session.run()
        with self.assertRaises(ToolSessionError):self.retry_session(receive).run()
        self.assertEqual(len(calls),2)
        resumed=self.retry_session(receive)
        resumed.authorize_failed_exchange_retry()
        with self.assertRaises(disconnected):resumed.run()
        self.assertEqual(len(calls),3)

    def test_missing_provider_outcome_cannot_be_authorized(self):
        session=self.retry_session(None)
        session.archive.add('model_started',dict(turn=1,provider_start_seq=len(self.archive.records)))
        with self.assertRaisesRegex(ToolSessionError,'unknown'):
            session.authorize_failed_exchange_retry()

    def test_complete_provider_message_cannot_be_authorized(self):
        async def receive(request,timeout,state):
            state.update(message=response()['message'],finish='tool_calls',
                         usage=dict(prompt_tokens=5,completion_tokens=10))
        session=self.retry_session(receive)
        original=session.archive.add
        def crash(kind,data):
            if kind=='model_completed':raise KeyboardInterrupt()
            return original(kind,data)
        session.archive.add=crash
        with self.assertRaises(KeyboardInterrupt):session.run()
        resumed=self.retry_session(receive)
        with self.assertRaisesRegex(ToolSessionError,'usable'):
            resumed.authorize_failed_exchange_retry()
        self.assertEqual(resumed.run(),17)
        self.assertEqual(self.ledger.used('llm_requests'),1)

    def test_retry_authorization_is_consumed_before_dispatch(self):
        disconnected=type('ConnectError',(Exception,),{})
        async def receive(request,timeout,state):raise disconnected('connect failed')
        session=self.retry_session(receive)
        with self.assertRaises(disconnected):session.run()
        session.authorize_failed_exchange_retry()
        original=self.model.exchange
        self.model.exchange=lambda *args:(_ for _ in ()).throw(KeyboardInterrupt())
        try:
            with self.assertRaises(KeyboardInterrupt):session.run()
        finally:
            self.model.exchange=original
        resumed=self.retry_session(receive)
        with self.assertRaisesRegex(ToolSessionError,'unknown'):resumed.run()
        with self.assertRaisesRegex(ToolSessionError,'unknown'):resumed.authorize_failed_exchange_retry()
        self.assertEqual(self.ledger.used('llm_requests'),1)

    def test_non_network_failure_cannot_be_authorized(self):
        async def receive(request,timeout,state):raise ValueError('bad provider payload')
        session=self.retry_session(receive)
        with self.assertRaises(ValueError):session.run()
        with self.assertRaisesRegex(ToolSessionError,'network'):session.authorize_failed_exchange_retry()


if __name__ == '__main__':
    unittest.main()
