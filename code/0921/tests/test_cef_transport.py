import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.model import Model
from cef.store import Archive
from cef.ledger import BudgetLedger

def sse(items):
    raw=''.join('data: '+json.dumps(item)+'\n\n' for item in items)+'data: [DONE]\n\n'
    return httpx.Response(200,headers={'content-type':'text/event-stream'},content=raw.encode())

def chat_chunk(**kwargs):
    return dict(id='offline',object='chat.completion.chunk',created=0,model='offline',**kwargs)

class TransportTests(unittest.TestCase):
    def test_stream_and_failed_usage_cost(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'CEF_TEST_KEY':'offline-not-secret'}):
            a=Archive(d); b=BudgetLedger(a,dict(max_llm_requests=3,max_llm_tokens=50000,max_case_runs=1,max_analysis_seconds=1,max_wallclock_s=60))
            c=dict(endpoint='https://example.com/v1',api_key_env='CEF_TEST_KEY',route='direct',model_snapshot='offline-fixture',model_sampling={},max_output_tokens=100,provider='offline',request_timeout_s=1,wire_api='chat_completions',rates=dict(input_per_million=1,cached_input_per_million=.1,output_per_million=2,currency='fixture'))
            calls=[]
            replies=[[
                chat_chunk(choices=[dict(index=0,delta={'content':'{"action":"finish"}'},finish_reason='stop')]),
                chat_chunk(usage=dict(prompt_tokens=10,completion_tokens=2,total_tokens=12,prompt_cache_hit_tokens=3),choices=[]),
            ],[chat_chunk(choices=[dict(index=0,delta={'content':'{'},finish_reason='length')])]]
            def handler(request):
                calls.append(request)
                return sse(replies[len(calls)-1])
            def client(**kwargs): return httpx.AsyncClient(transport=httpx.MockTransport(handler),**kwargs)
            with patch('cef.model.DefaultAsyncHttpxClient',side_effect=client):
                m=Model(c,a,b)
                self.assertEqual(m.call('planner',{})['action'],'finish')
                self.assertEqual(b.used('llm_requests'),1); self.assertEqual(b.used('llm_tokens'),12)
                self.assertAlmostEqual(a.of('charge')[-1]['fee'],.0000113)
                with self.assertRaises(ValueError): m.call('editor',{})
                self.assertEqual(len(calls),2)
                self.assertTrue(a.of('charge')[-1]['conservative_debit'])
                self.assertIsNone(a.of('charge')[-1]['fee'])

    def test_responses(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}):
            a=Archive(d); b=BudgetLedger(a,dict(max_llm_requests=1,max_llm_tokens=50000,max_case_runs=1,max_analysis_seconds=1,max_wallclock_s=60))
            c=dict(endpoint='https://example.com/v1',api_key_env='CEF_TEST_KEY',route='direct',model_snapshot='offline',model_sampling={},max_output_tokens=100,provider='offline',request_timeout_s=1,wire_api='responses',rates=dict(input_per_million=0,cached_input_per_million=0,output_per_million=0,currency='fixture'))
            items=[dict(type='response.output_text.delta',delta='{"x":1}'),dict(type='response.completed',response=dict(status='completed',usage=dict(input_tokens=4,output_tokens=3,input_tokens_details={'cached_tokens':1})))]
            def client(**kwargs): return httpx.AsyncClient(transport=httpx.MockTransport(lambda request:sse(items)),**kwargs)
            with patch('cef.model.DefaultAsyncHttpxClient',side_effect=client):
                m=Model(c,a,b)
                self.assertEqual(m.call('editor',{}),{'x':1}); self.assertEqual(b.used('llm_tokens'),7)

if __name__=='__main__': unittest.main()
