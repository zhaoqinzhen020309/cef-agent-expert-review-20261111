"""Deterministic regressions from the user's three real failed PAB runs."""
import httpx
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_cef_sdk import sdk_transport,completion
import test_cef_sdk as sdk


class LiveFailureRegressions(unittest.TestCase):
    def test_recorded_sub_nanosecond_deadline_difference_is_not_real_overtime(self):
        from cef.runner import deadline_exceeded
        self.assertFalse(deadline_exceeded(60.00000000005821,60))
        self.assertFalse(deadline_exceeded(60,60))
        self.assertTrue(deadline_exceeded(60.00001,60))
        self.assertTrue(deadline_exceeded(60.01,60))

    def test_transient_stream_failure_retries_once_and_retains_unknown_usage(self):
        class BrokenBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"id":"fixture","object":"chat.completion.chunk","created":0,"model":"offline","choices":[{"index":0,"delta":{"reasoning_content":"still thinking"},"finish_reason":null}]}\n\n'
                raise httpx.RemoteProtocolError('incomplete chunked read')
        requests=[]
        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests)==1:
                return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=BrokenBody())
            body=b'data: {"id":"fixture","object":"chat.completion.chunk","created":0,"model":"offline","choices":[{"index":0,"delta":{"content":"{\\\"action\\\":\\\"finish\\\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\ndata: [DONE]\n\n'
            return httpx.Response(200,headers={'content-type':'text/event-stream'},content=body)
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}),sdk_transport(handler),patch('cef.model.time.sleep'):
            model,archive,ledger=sdk.SDKTests().fixture(folder,stream=True,transport_retry_limit=1,stop_rule='iterations',rates=None)
            result=model.call('editor',{})
            self.assertEqual(result,{'action':'finish'})
            self.assertEqual(requests[0],requests[1])
            self.assertEqual(ledger.used('llm_requests'),2)
            charges=[r for r in archive.of('charge') if r['resource']=='llm_tokens']
            self.assertEqual([r['usage_known'] for r in charges],[False,True])
            self.assertIsNone(charges[0]['fee'])
            self.assertEqual(len(archive.of('transport_retry')),1)
            responses=archive.of('provider_response')
            self.assertGreater(responses[0]['stream_observation']['reasoning_chars'],0)

    def test_transport_retry_is_bounded(self):
        requests=[]
        def handler(request):
            requests.append(request)
            raise httpx.ConnectError('fixture connection down')
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}),sdk_transport(handler),patch('cef.model.time.sleep'):
            model,archive,ledger=sdk.SDKTests().fixture(folder,transport_retry_limit=1,stop_rule='iterations',rates=None)
            with self.assertRaises(Exception): model.call('editor',{})
            self.assertEqual(len(requests),2)
            self.assertEqual(len(archive.of('transport_retry')),1)
            self.assertEqual(ledger.used('llm_requests'),2)

    def test_truncated_reasoning_response_is_not_silently_retried_or_parsed(self):
        requests=[]
        def handler(request):
            requests.append(request)
            data=completion('{"partial":',finish='length')
            data['usage'].update(completion_tokens=32767,total_tokens=32777,completion_tokens_details={'reasoning_tokens':30341})
            return httpx.Response(200,json=data)
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}),sdk_transport(handler):
            model,archive,ledger=sdk.SDKTests().fixture(folder,transport_retry_limit=1,stop_rule='iterations',rates=None)
            with self.assertRaisesRegex(ValueError,'reasoning_tokens=30341'): model.call('editor',{})
            self.assertEqual(len(requests),1)
            self.assertEqual(archive.of('charge')[-1]['amount'],32777)
            self.assertIsNone(archive.of('provider_response')[-1]['parsed'])

    def test_new_config_carries_full_generation_capacity_and_audited_recovery(self):
        from scripts.cef_launcher import build_config
        from cef.preflight import identity
        with tempfile.TemporaryDirectory() as folder:
            c,_=build_config(2025,Path(folder)/'run',dict(model='deepseek-flash',base_url='https://api.deepseek.com',iterations=1,workers=4,evaluation_policy='joint_only'))
            self.assertEqual(c['max_output_tokens'],65536)
            self.assertEqual(c['request_timeout_s'],600)
            self.assertEqual(c['transport_retry_limit'],1)
            self.assertEqual(identity(c)['transport_retry_limit'],1)

    def test_partial_usage_remains_unknown_instead_of_filling_missing_tokens_with_zero(self):
        def handler(request):
            data=completion(); data['usage']={'prompt_tokens':10}
            return httpx.Response(200,json=data)
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}),sdk_transport(handler):
            model,archive,ledger=sdk.SDKTests().fixture(folder,transport_retry_limit=1,stop_rule='iterations',rates=None)
            with self.assertRaisesRegex(ValueError,'required usage'): model.call('editor',{})
            charge=[r for r in archive.of('charge') if r['resource']=='llm_tokens'][-1]
            self.assertFalse(charge['usage_known'])
            self.assertEqual(charge['actual_usage']['prompt_tokens'],10)
            self.assertNotIn('output_tokens',charge)
            self.assertFalse(archive.of('transport_retry'))


if __name__=='__main__': unittest.main()
