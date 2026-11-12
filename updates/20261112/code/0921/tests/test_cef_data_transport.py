"""Regression: a data generator must not inherit the solver's 64K high-thinking call."""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from cef.model import Model
from cef.store import Archive
from cef.ledger import BudgetLedger
from scripts.prepare_cef_training import build_data_model_config
from scripts.cef_launcher import build_config
from runDataAgent import CONFIG
from test_cef_sdk import sdk_transport, completion, config


class DataTransportTests(unittest.TestCase):
    def test_default_data_call_disables_explicit_thinking_without_changing_search(self):
        with tempfile.TemporaryDirectory() as td:
            data=build_data_model_config(CONFIG,Path(td))
            original,_=build_config(2025,Path(td)/'search',dict(model=CONFIG['model'],
                base_url=CONFIG['base_url'],iterations=2,workers=1))
            self.assertEqual(data['model_sampling'],{'thinking':{'type':'disabled'}})
            self.assertEqual(data['max_output_tokens'],8192)
            self.assertEqual(original['model_sampling']['reasoning_effort'],'high')
            self.assertEqual(original['max_output_tokens'],65536)
            self.assertNotIn('data_manifest',data)
            requests=[]
            def handler(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200,json=completion('{"generator_source":"def generate(n_cities,seed,parameters): return []","parameters":{}}'))
            data.update(api_key_env='CEF_TEST_KEY',stream=False)
            with patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}),sdk_transport(handler):
                archive=Archive(td);ledger=BudgetLedger(archive,{},enforce_limits=False)
                Model(data,archive,ledger).call('data_designer',{})
            self.assertEqual(requests[0]['thinking'],{'type':'disabled'})
            self.assertNotIn('reasoning_effort',requests[0])
            self.assertEqual(requests[0]['max_tokens'],8192)

    def test_explicit_thinking_can_be_enabled_with_its_own_effort(self):
        with tempfile.TemporaryDirectory() as td:
            c=build_data_model_config(dict(CONFIG,thinking=True,reasoning_effort='low'),Path(td))
            self.assertEqual(c['model_sampling'],dict(thinking=dict(type='enabled'),reasoning_effort='low'))

    def test_length_response_preserves_real_usage_and_is_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}):
            archive=Archive(td);ledger=BudgetLedger(archive,{},enforce_limits=False)
            model=Model(config(stop_rule='iterations'),archive,ledger)
            def handler(request):
                response=completion('',finish='length')
                response['usage']=dict(prompt_tokens=8568,completion_tokens=65536,total_tokens=74104,
                    completion_tokens_details=dict(reasoning_tokens=65536))
                return httpx.Response(200,json=response)
            with sdk_transport(handler),self.assertRaisesRegex(ValueError,'incomplete model response: length'):
                model.call('data_designer',{})
            self.assertEqual(len(archive.of('provider_request')),1)
            self.assertEqual(ledger.used('llm_tokens'),74104)
            self.assertEqual(archive.of('provider_response')[0]['content'],'')

    def test_waiting_progress_is_visible_without_printing_model_text(self):
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}):
            archive=Archive(td);ledger=BudgetLedger(archive,{},enforce_limits=False)
            model=Model(config(console_progress=True,model_progress_interval_s=.01),archive,ledger)
            async def receive(request,timeout,state):
                await asyncio.sleep(.04)
                state.update(content='{"parameters":{}}',finish='stop',
                             usage=dict(prompt_tokens=10,completion_tokens=2,total_tokens=12))
            out=io.StringIO()
            with patch.object(model,'_receive',side_effect=receive),contextlib.redirect_stdout(out):
                result=model._call_once('data_designer',{},0)
            self.assertEqual(result,{'parameters':{}})
            self.assertIn('Model waiting',out.getvalue())
            self.assertNotIn('{"parameters"',out.getvalue())


if __name__=='__main__':unittest.main()
