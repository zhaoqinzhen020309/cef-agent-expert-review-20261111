"""One-entry orchestration checks without a provider or benchmark launch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runCEFAuto import CONFIG
from scripts.cef_alternating_launcher import build_auto_config, main, finish_search


class AlternatingLauncherTests(unittest.TestCase):
    def test_failed_manual_search_reports_original_failure_before_finish_gate(self):
        from types import SimpleNamespace
        engine=SimpleNamespace(archive=SimpleNamespace(of=lambda k:[]))
        with self.assertRaisesRegex(RuntimeError,'APIConnectionError: Connection error'):
            finish_search(engine,dict(manual_rounds=True,max_iterations=None),{},None,
                dict(status='failed',reason='APIConnectionError: Connection error.'))

    def test_pending_references_do_not_trigger_final_test_or_search_failure(self):
        with patch('scripts.cef_alternating_launcher.final_test',side_effect=AssertionError('No final test')):
            self.assertIsNone(finish_search(None,{},None,None,dict(status='waiting_reference')))

    def test_sdk_round_close_prompt_matches_the_available_actions(self):
        import os,httpx
        from cef.model import Model
        from cef.ledger import BudgetLedger
        from cef.store import Archive
        from test_cef_sdk import sdk_transport,completion,config
        requests=[]
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200,json=completion('{"action":"advance_round"}'))
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}):
            archive=Archive(td);ledger=BudgetLedger(archive,{},enforce_limits=False)
            c=config(alternating_agents=True,evaluation_policy='joint_only',stop_rule='iterations')
            with sdk_transport(handler):
                self.assertEqual(Model(c,archive,ledger).call('planner',{'alternating':{'phase':'round_complete'}})['action'],'advance_round')
        system=requests[0]['messages'][0]['content']
        self.assertIn('advance_round',system)
        self.assertNotIn('Inspect/analyze/revise only',system)
        self.assertNotIn('not selectable as a final result',system)

    def test_resume_cannot_redirect_to_another_search_folder(self):
        from scripts.cef_alternating_launcher import run_folder
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'search').mkdir();(root/'search/facts.jsonl').write_text('')
            with patch('scripts.cef_alternating_launcher.balance_snapshot',side_effect=AssertionError('No network')):
                with self.assertRaisesRegex(ValueError,'run_dir'):
                    run_folder({'run_dir':str(root/'elsewhere')},{},root,resume=True)

    def test_completed_test_reuse_verifies_export_and_exact_conditions(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'test').mkdir()
            (root/'test/summary.json').write_text('{"valid": true}')
            (root/'test/freeze.json').write_text(json.dumps(dict(source_hash='h',benchmark=[],test_repeats=1,seeds={'wrong':1})))
            engine=SimpleNamespace(archive=SimpleNamespace(of=lambda k:[1]),config={})
            c=dict(run_dir=str(root/'search'),max_iterations=1)
            t=dict(instances=[],test_repeats=1,evaluation_seeds={})
            result=dict(status='finished',completed_iterations=1,best=dict(source_hash='h'))
            with patch('scripts.cef_alternating_launcher.verify_export',return_value=(None,result,'')) as verify:
                with self.assertRaisesRegex(ValueError,'conditions'):finish_search(engine,c,t,root,result)
                verify.assert_called_once()

    def test_check_is_read_only_and_freezes_warm_start_and_two_model_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            settings=dict(CONFIG,rounds=10,output_dir=Path(td)/'not_created')
            with patch('scripts.cef_alternating_launcher.balance_snapshot',side_effect=AssertionError('No network')):
                self.assertEqual(main(settings,['--check']),0)
            self.assertFalse(Path(settings['output_dir']).exists())
            c,test=build_auto_config(settings,Path(td)/'example',2025)
            self.assertTrue(c['alternating_agents'])
            self.assertEqual(c['max_iterations'],10)
            self.assertEqual(len(c['data_manifest']['train']),128)
            self.assertEqual(len(test['instances']),29)
            self.assertEqual(c['baseline_source_manifest']['source_version']['path'],str(Path(settings['initial_solver']).resolve()))
            self.assertEqual(c['model_sampling']['reasoning_effort'],'high')
            self.assertFalse(c['alternating_data_config']['thinking'])

    def test_final_test_only_after_all_rounds_and_only_once(self):
        from types import SimpleNamespace
        class Archive:
            def of(self,kind):return list(range(10)) if kind=='round_complete' else []
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            engine=SimpleNamespace(archive=Archive(),config={'max_iterations':10,'run_dir':str(root/'search')},
                completed_iterations=lambda:10)
            c=dict(max_iterations=10,run_dir=str(root/'search'))
            test=dict(instances=[],test_repeats=1,evaluation_seeds={})
            result=dict(status='finished',completed_iterations=10,best={'candidate_id':'winner'})
            with patch('scripts.cef_alternating_launcher.final_test',return_value={'valid':True}) as evaluate:
                finish_search(engine,c,test,root,result)
                self.assertEqual(evaluate.call_count,1)
            with patch('scripts.cef_alternating_launcher.final_test',side_effect=AssertionError('No premature test')):
                with self.assertRaises(RuntimeError):finish_search(engine,c,test,root,dict(result,status='failed'))


if __name__=='__main__':unittest.main()
