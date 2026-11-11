"""Explicit recovery only for a fully recorded truncated Data response."""
import tempfile
import unittest
from pathlib import Path

from cef.store import Archive,canonical
from scripts.cef_data_resume import failed_data_request,retry_data_request,check_saved_requests


class DataResumeTests(unittest.TestCase):
    def test_saved_checkpoint_accepts_persisted_editor_but_rejects_unknown_request(self):
        with tempfile.TemporaryDirectory() as td:
            search=Archive(Path(td)/'search');provider=Archive(Path(td)/'data_agent')
            search.add('model_start',dict(call_id='step_17:editor',role='editor'))
            with self.assertRaisesRegex(ValueError,'unresolved'):check_saved_requests(search,provider)
            search.add('model_raw_response',dict(call_id='step_17:editor',response={'saved':True}))
            check_saved_requests(search,provider)
            provider.add('provider_request',dict(role='data_designer'))
            with self.assertRaisesRegex(ValueError,'unfinished'):check_saved_requests(search,provider)
    def fixture(self,folder):
        search=Archive(Path(folder)/'search');provider=Archive(Path(folder)/'data_agent');local=Archive(Path(folder)/'rounds/008/data')
        payload=dict(global_round=8,interface_version='integrated_feedback_v1',calibration_source_hash='champion')
        for i in range(1,8):search.add('round_complete',dict(round=i,winner={'source_hash':'champion'}))
        local.add('data_model_start',dict(iteration=0,payload=payload))
        provider.add('provider_request',dict(role='data_designer',messages=[dict(role='user',content=canonical(payload))],request_parameters={'max_tokens':8192}))
        provider.add('provider_response',dict(role='data_designer',user=canonical(payload),parsed=None,finish_reason='length',content='incomplete',error_chain=['ValueError: incomplete model response: length']))
        return search,provider,local,payload

    def test_known_truncation_reuses_exact_payload_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as td:
            search,provider,local,payload=self.fixture(td)
            before=provider.path.read_bytes();seen=[]
            class Model:
                def call(self,role,value):seen.append((role,value));return {'rules':[],'reason':'parsed fixture'}
            result=retry_data_request(search,provider,local,Model(),8,32768)
            self.assertEqual(seen,[('data_designer',payload)])
            self.assertEqual(result['status'],'recovered')
            self.assertTrue(provider.path.read_bytes().startswith(before))
            self.assertEqual(local.of('data_model_response')[0]['response']['rules'],[])
            self.assertEqual(len(provider.of('manual_data_recovery_start')),1)
            self.assertEqual(retry_data_request(search,provider,local,None,8,32768)['status'],'already_available')

    def test_unknown_request_or_wrong_round_is_not_redispatched(self):
        with tempfile.TemporaryDirectory() as td:
            search,provider,local,payload=self.fixture(td)
            with self.assertRaisesRegex(ValueError,'pending round'):failed_data_request(search,provider,local,7)
            provider.add('provider_request',dict(role='data_designer',messages=[dict(role='user',content=canonical(payload))],request_parameters={'max_tokens':8192}))
            with self.assertRaisesRegex(ValueError,'fully recorded'):failed_data_request(search,provider,local,8)

    def test_network_or_parse_failure_is_not_misclassified_as_length(self):
        with tempfile.TemporaryDirectory() as td:
            search,provider,local,payload=self.fixture(td)
            provider.records[-1]['data']['finish_reason']='stop'
            with self.assertRaisesRegex(ValueError,'length'):failed_data_request(search,provider,local,8)

    def test_failed_retry_never_injects_a_synthetic_success(self):
        with tempfile.TemporaryDirectory() as td:
            search,provider,local,payload=self.fixture(td)
            class Model:
                def call(self,*args):raise RuntimeError('length again')
            with self.assertRaises(RuntimeError):retry_data_request(search,provider,local,Model(),8,32768)
            self.assertFalse(local.of('data_model_response'))
            self.assertEqual(len(provider.of('manual_data_recovery_failure')),1)


if __name__=='__main__':unittest.main()
