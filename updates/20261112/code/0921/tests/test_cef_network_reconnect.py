"""Offline durable reconnection tests; no API or paid solver runs."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cef.store import Archive
from cef.tool_session import ToolSession, ToolSessionError, TaskPaused
from cef.model import Model
from cef.ledger import BudgetLedger


class RecordedModel:
    automatic_network_recovery = True
    config = {}
    def __init__(self, root, outcomes):
        self.archive = Archive(Path(root)/'provider')
        self.outcomes = iter(outcomes)
        self.calls = 0

    def exchange(self, messages, tools, role):
        self.calls += 1
        request = self.archive.add('provider_request', dict(role=role,
            messages=copy.deepcopy(messages), request_parameters=dict(tools=tools)))
        outcome = next(self.outcomes)
        if isinstance(outcome, str):
            self.archive.add('provider_response', dict(request_seq=request['seq'],
                message=None, usage=None, error_chain=[outcome]))
            raise ConnectionError(outcome)
        self.archive.add('provider_response', dict(request_seq=request['seq'],
            message=outcome['message'], finish_reason='tool_calls', error_chain=[]))
        return outcome


def reply():
    return dict(message=dict(role='assistant', content=None, tool_calls=[dict(
        id='submit',type='function',function=dict(name='submit_candidate',arguments='{}'))]),
        finish_reason='tool_calls')


class ReconnectTests(unittest.TestCase):
    def session(self, root, model, dispatch=None, **kwargs):
        tools=[dict(type='function',function=dict(name='submit_candidate',parameters={'type':'object'}))]
        return ToolSession(Path(root)/'task', model, tools, dispatch or (
            lambda n,a:dict(accepted=True,value=7)), role='editor',system='test',payload={},**kwargs)

    @patch('time.sleep')
    def test_repeated_disconnect_retries_same_turn_and_dispatches_once(self, sleep):
        with tempfile.TemporaryDirectory() as root:
            model=RecordedModel(root,['APIConnectionError: SSL EOF',
                'RemoteProtocolError: incomplete chunk',reply()])
            calls=[]
            session=self.session(root,model,lambda n,a:calls.append(n) or dict(accepted=True,value=7))
            self.assertEqual(session.run(),7)
            self.assertEqual(calls,['submit_candidate'])
            self.assertEqual(len(session.archive.of('model_started')),1)
            self.assertEqual(len(session.archive.of('retry_started')),2)
            self.assertEqual(len(model.archive.of('provider_response')),3)

    @patch('time.sleep')
    def test_saved_failure_resumes_without_manual_authorization(self, sleep):
        with tempfile.TemporaryDirectory() as root:
            model=RecordedModel(root,['APIConnectionError: EOF',reply()])
            model.automatic_network_recovery=False
            with self.assertRaises(ConnectionError):self.session(root,model).run()
            model.automatic_network_recovery=True
            self.assertEqual(self.session(root,model).run(),7)
            self.assertEqual(model.calls,2)

    @patch('time.sleep')
    def test_non_transport_failures_are_not_retried(self, sleep):
        for error in ['APIStatusError: 402 Insufficient Balance',
                      'PermissionDeniedError: 403', 'ValueError: invalid JSON']:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as root:
                model=RecordedModel(root,[error,reply()])
                with self.assertRaises(ConnectionError):self.session(root,model).run()
                self.assertEqual(model.calls,1)
                sleep.assert_not_called()

    @patch('time.sleep')
    def test_pause_during_outage_retains_pending_exchange(self, sleep):
        with tempfile.TemporaryDirectory() as root:
            model=RecordedModel(root,['APIConnectionError: EOF',reply()])
            with self.assertRaises(TaskPaused):
                self.session(root,model,pause_check=lambda:model.calls>0).run()
            self.assertEqual(model.calls,1)
            self.assertEqual(self.session(root,model).run(),7)

    def test_unknown_outcome_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            model=RecordedModel(root,[reply()])
            model.exchange=lambda *args: (_ for _ in ()).throw(ConnectionError('unrecorded'))
            with self.assertRaises(ConnectionError):self.session(root,model).run()
            with self.assertRaises(ToolSessionError):self.session(root,model).run()

    @patch('time.sleep')
    def test_real_metered_model_reconnects_and_keeps_unknown_usage(self, sleep):
        with tempfile.TemporaryDirectory() as root:
            archive=Archive(Path(root)/'provider')
            ledger=BudgetLedger(archive,{},enforce_limits=False)
            config=dict(endpoint='https://example.invalid',api_key_env='CEF_TEST_KEY',
                route='direct',model_snapshot='test',wire_api='chat_completions',
                provider='test',model_sampling={},max_output_tokens=100,
                request_timeout_s=2,stop_rule='iterations',stream=True,
                rates=dict(input_per_million=1,cached_input_per_million=.1,
                           output_per_million=2,currency='USD'))
            with patch.dict('os.environ',CEF_TEST_KEY='offline-test'):
                model=Model(config,archive,ledger)
            attempts=[]
            disconnected=type('RemoteProtocolError',(Exception,),{})
            async def receive(request,timeout,state):
                attempts.append(request)
                if len(attempts)==1:raise disconnected('peer closed')
                state.update(message=reply()['message'],finish='tool_calls',
                    usage=dict(prompt_tokens=5,completion_tokens=10))
            model._receive_native=receive
            self.assertEqual(self.session(root,model).run(),7)
            self.assertEqual(ledger.used('llm_requests'),2)
            responses=archive.of('provider_response')
            self.assertIsNone(responses[0]['usage'])
            self.assertEqual(responses[1]['usage']['completion_tokens'],10)

