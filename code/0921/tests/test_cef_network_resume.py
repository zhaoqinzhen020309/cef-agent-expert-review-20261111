import importlib
from pathlib import Path
import tempfile
import unittest
from cef.store import Archive
from cef.engine import Engine


class NetworkResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.archive=Archive(Path(self.temp.name))

    def helper(self):
        try:return importlib.import_module('scripts.cef_network_resume')
        except ModuleNotFoundError:self.fail('Explicit failed-request recovery helper is missing')

    def seed_failure(self):
        a=self.archive
        a.add('model_start',dict(call_id='step_11:editor',role='editor',payload={'current_program':'original','seed':2025}))
        a.add('provider_request',dict(role='editor',attempt=1))
        a.add('charge',dict(resource='llm_tokens',amount=0,usage_known=False))
        a.add('provider_response',dict(role='editor',parsed=None,error_chain=['APIConnectionError: Connection error.']))
        a.add('failure',dict(error='APIConnectionError: Connection error.',round=6))

    def test_success_reuses_original_payload_and_normal_request_reads_saved_response(self):
        h=self.helper();self.seed_failure();seen=[]
        class FakeModel:
            def call(self,role,payload):seen.append((role,payload));return {'generated':'complete real response fixture'}
        before=self.archive.path.read_bytes()
        result=h.retry_failed_request(self.archive,FakeModel(),'step_11:editor')
        self.assertEqual(result['status'],'recovered')
        self.assertEqual(seen,[('editor',{'current_program':'original','seed':2025})])
        self.assertTrue(self.archive.path.read_bytes().startswith(before))
        self.assertEqual(len(self.archive.of('charge')),1)
        engine=Engine.__new__(Engine);engine.archive=self.archive
        engine.model=None
        self.assertEqual(engine.request('editor',{},'step_11:editor'),{'generated':'complete real response fixture'})

    def test_unknown_unfinished_provider_request_is_not_redispatched(self):
        h=self.helper();self.seed_failure()
        self.archive.add('provider_request',dict(role='editor',attempt=2))
        with self.assertRaisesRegex(ValueError,'fully recorded'):
            h.retry_failed_request(self.archive,None,'step_11:editor')

    def test_repeat_network_failure_does_not_fabricate_response(self):
        h=self.helper();self.seed_failure()
        class FakeModel:
            def call(self,role,payload):raise ConnectionError('still offline')
        with self.assertRaises(ConnectionError):
            h.retry_failed_request(self.archive,FakeModel(),'step_11:editor')
        self.assertEqual(self.archive.of('model_raw_response'),[])
        self.assertEqual(len(self.archive.of('manual_model_recovery_failure')),1)

    def test_already_recovered_request_does_not_call_model_twice(self):
        h=self.helper();self.seed_failure()
        self.archive.add('model_raw_response',dict(call_id='step_11:editor',role='editor',response={'saved':True}))
        self.assertEqual(h.retry_failed_request(self.archive,None,'step_11:editor')['status'],'already_available')

    def test_non_network_failure_is_not_retried(self):
        h=self.helper();self.seed_failure()
        self.archive.add('failure',dict(error='ValueError: invalid edit_count for mode',round=6))
        with self.assertRaisesRegex(ValueError,'network'):
            h.retry_failed_request(self.archive,None,'step_11:editor')


if __name__=='__main__':unittest.main()
