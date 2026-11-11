"""PAB-only generation repeats, with no historical winning source or ablations."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cef.engine import Engine
from test_cef_iterations_billing import iteration_config


class JointFake:
    def __init__(self): self.inputs=[]
    def call(self,role,payload):
        self.inputs.append((role,payload))
        if role=='planner':
            return dict(action='revise',edit_count=2,parent_or_target_id='seed',issue='offline joint fixture',proposed_intents=['ordering','import marker'],expected_observations='measure joint',evidence_ids=[],previous_event_update=[])
        return dict(package_id=f'joint_{len(self.inputs)}',parent_id=payload['parent_id'],parent_hash=payload['parent_hash'],issue='fixture',evidence_ids=[],joint_reason='only joint measured',edits=[
            dict(edit_id='A',purpose='fixture order',prediction='unmeasured alone',exact_replacements=[dict(old='key=lambda j:-sum(p[j])',new='key=lambda j:sum(p[j])')]),
            dict(edit_id='B',purpose='fixture marker',prediction='unmeasured alone',exact_replacements=[dict(old='import time',new='import time\n# independent offline fixture')])])


class PABGenerationTests(unittest.TestCase):
    def test_64_cases_evaluate_only_new_joint_no_initial_or_singletons(self):
        with tempfile.TemporaryDirectory() as d:
            c=iteration_config(d,1); c.update(evaluation_policy='joint_only',evaluation_workers=4)
            c['data_manifest']['train']=[dict(id=f'x{i}',split='train',instance={'processing_times':[[1,2],[3,4]]}) for i in range(64)]
            c['evaluation_seed_manifest']={f'x{i}':2025 for i in range(64)}
            model=JointFake(); e=Engine(c,model)
            def case(source,adapter,instance,seed,on_started):
                on_started()
                return dict(valid=True,objective=6,fitness=6,started=True,elapsed=.01,wallclock_s=.02)
            with patch.object(e.runner,'case',side_effect=case) as worker:
                result=e.run()
            self.assertEqual(result['status'],'finished',result['reason'])
            self.assertEqual(worker.call_count,64)
            self.assertEqual({r['candidate_id'] for r in e.archive.of('execution')},{'step_0:PAB'})
            self.assertEqual(set(e.candidates()),{'seed','step_0:PAB'})
            self.assertFalse(e.candidates()['seed']['eligibility'])
            self.assertEqual(result['best']['variant'],'PAB')
            self.assertIn('current_program',model.inputs[0][1])
            self.assertIn('bootstrap',model.inputs[0][1]['action_context']['revise_rule'])
            self.assertEqual(model.inputs[0][1]['current_program'],e.adapter.seed_source())
            event=e.archive.of('contrast')[0]
            self.assertFalse(event['component_contrasts_measured'])
            self.assertEqual(event['contrasts'],{})
            self.assertEqual(event['scores'],{'PAB':6})
            self.assertEqual(result['packet_statistics']['complete_contrasts'],0)

    def test_two_real_worker_rounds_feed_joint_results_to_next_decision(self):
        with tempfile.TemporaryDirectory() as d:
            c=iteration_config(d,2); c['evaluation_policy']='joint_only'
            model=JointFake(); e=Engine(c,model); result=e.run()
            self.assertEqual(result['status'],'finished',result['reason'])
            self.assertEqual(len(e.archive.of('execution')),2)
            second=[payload for role,payload in model.inputs if role=='planner'][1]
            self.assertEqual(second['recent_events'][0]['scores'].keys(),{'PAB'})
            self.assertEqual(second['retrieved_cards'][0]['measured_observation']['scores'].keys(),{'PAB'})
            self.assertEqual(result['evaluation_policy'],'joint_only')

    def test_sdk_planner_editor_and_real_workers_are_fresh_in_each_repeat(self):
        import os,httpx
        from cef.model import Model
        from test_cef_sdk import sdk_transport,completion,config as sdk_config
        from cef.reporting import final_test
        from cef.patch import digest
        sources=[]
        with tempfile.TemporaryDirectory() as root:
            for seed in (2025,2026):
                d=str(Path(root)/str(seed)); c=iteration_config(d,1)
                c.update(sdk_config(),evaluation_policy='joint_only',rates=None,diagnostic_run=True,diagnostic_train_limit=1,test_repeats=1)
                c['evaluation_seed_manifest']['x']=seed
                c['data_manifest']['test']=[dict(id='offline_test',split='test',reference=6,instance={'processing_times':[[1,2],[3,4]]})]
                engine=Engine(c,None); fixture=JointFake(); requests=[]
                def handler(request):
                    body=json.loads(request.content); requests.append(body)
                    payload=json.loads(body['messages'][-1]['content'])
                    role='editor' if 'parent_code' in payload else 'planner'
                    return httpx.Response(200,json=completion(json.dumps(fixture.call(role,payload))))
                with patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}),sdk_transport(handler):
                    engine.model=Model(c,engine.archive,engine.ledger)
                    result=engine.run()
                self.assertEqual(result['status'],'finished',result['reason'])
                self.assertEqual(len(requests),2)
                self.assertIn('Only joint scores/errors',requests[0]['messages'][0]['content'])
                self.assertNotIn('inspect/analyze/profile/revise',requests[0]['messages'][0]['content'])
                self.assertNotIn('Profile any known candidate',requests[0]['messages'][0]['content'])
                self.assertIn('evaluated together as PAB only',requests[1]['messages'][0]['content'])
                self.assertEqual([m['role'] for m in requests[0]['messages']],['system','user'])
                sources.append(json.loads(requests[0]['messages'][-1]['content'])['current_program'])
                self.assertEqual(engine.ledger.used('llm_requests'),2)
                self.assertEqual(len(engine.archive.of('execution')),1)
                before=digest(engine.archive.path.read_text(encoding='utf-8'))
                with patch('cef.reporting.audit_test'):
                    tested=final_test(d,Path(root)/f'test_{seed}',c['data_manifest']['test'],1,{'offline_test:0':seed})
                self.assertTrue(tested['valid']); self.assertTrue(tested['search_unchanged'])
                self.assertEqual(before,digest(engine.archive.path.read_text(encoding='utf-8')))
            self.assertEqual(sources,[engine.adapter.seed_source()]*2)

    def test_all_invalid_joint_has_no_fallback_best(self):
        with tempfile.TemporaryDirectory() as d:
            c=iteration_config(d,1); c['evaluation_policy']='joint_only'
            e=Engine(c,JointFake())
            with patch.object(e.runner,'case',return_value=dict(valid=False,objective=None,fitness=None,started=False,elapsed=0,wallclock_s=0,error='offline failure')):
                result=e.run()
            self.assertEqual(result['status'],'failed'); self.assertIsNone(result['best'])
            self.assertEqual(result['completed_iterations'],1)
            self.assertFalse((Path(d)/'best_solver.py').exists())

    def test_entry_keeps_joint_generation_with_user_configurable_repeats(self):
        spec=importlib.util.spec_from_file_location('pab_entry',ROOT/'runCEF.py')
        entry=importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
        self.assertEqual(entry.run.__name__,'run_pab')
        self.assertIs(type(entry.CONFIG['iterations']),int)
        self.assertGreater(entry.CONFIG['iterations'],0)
        self.assertTrue(entry.CONFIG['seeds'])
        self.assertTrue(all(type(seed) is int for seed in entry.CONFIG['seeds']))

    def test_config_is_joint_only_and_never_loads_historical_winner(self):
        from scripts.cef_launcher import build_config
        from cef.config import validate
        with tempfile.TemporaryDirectory() as d:
            settings=dict(model='deepseek-flash',base_url='https://api.deepseek.com',iterations=1,workers=4,evaluation_policy='joint_only')
            c,_=build_config(2025,Path(d)/'fresh',settings)
            self.assertEqual(validate(c)['evaluation_policy'],'joint_only')
            self.assertNotIn('cef_tsp_effect',json.dumps(c))
            self.assertEqual(c['experiment_variant'],'PAB-only generation, no component contrasts')

    def test_new_preflight_keeps_its_single_code_response_contract(self):
        from cef.preflight import preflight
        from cef.patch import digest
        captured=[]
        class OfflinePreflight:
            def __init__(self,c,archive,ledger): captured.append(c)
            def call(self,role,payload):
                self.payload=payload
                self.assert_count=payload['edit_count']
                return dict(parent_hash=digest(payload['parent_code']),single_edit_reason='transport fixture',edits=[dict(edit_id='A',purpose='complete fixture',prediction='none',exact_replacements=[dict(old=payload['parent_code'],new=payload['parent_code']+'\n# offline complete response\n')])])
        with tempfile.TemporaryDirectory() as d:
            c=dict(preflight_dir=str(Path(d)/'preflight'),task='fssp',max_llm_tokens=None,request_timeout_s=1,stop_rule='iterations',evaluation_policy='joint_only')
            with patch('cef.preflight.Model',OfflinePreflight): result=preflight(c)
            self.assertTrue(result['passed'],result.get('error'))
            self.assertNotIn('evaluation_policy',captured[0])
            self.assertEqual(c['evaluation_policy'],'joint_only')

    def test_seed_repeats_preserve_failure_and_continue_without_retry(self):
        import scripts.cef_launcher as launcher
        with tempfile.TemporaryDirectory() as d:
            settings=dict(model='deepseek-flash',base_url='https://api.deepseek.com',seeds=[2025,2026],iterations=1,workers=4,evaluation_policy='joint_only',output_dir=Path(d)/'runs')
            seen=[]
            def run_one(c,test,folder):
                folder.mkdir(parents=True); seen.append(next(iter(c['evaluation_seed_manifest'].values())))
                if seen[-1]==2025: raise ValueError('offline invalid generation')
                launcher.write_json(folder/'launch_summary.json',dict(search={'best':{'fitness':.2}},test={'valid':True,'mean_gap_pct':.3}))
            with patch.object(launcher,'run_one',side_effect=run_one),patch.object(launcher,'balance_snapshot',return_value={}),patch.object(launcher,'write_billing',return_value={'usage':{'known_total_tokens':0,'unknown_usage_requests':0},'account_balance_changes':None}),patch.object(launcher,'load_env'):
                self.assertEqual(launcher.main(settings,[]),1)
            self.assertEqual(seen,[2025,2026])
            series=json.loads(next(settings['output_dir'].glob('series_*.json')).read_text(encoding='utf-8'))
            self.assertEqual(series['attempts'],2); self.assertEqual(series['successful_repeats'],1)
            self.assertIsNone(series['mean_test_gap_pct'])
            self.assertEqual(series['results'][0]['status'],'failed')


if __name__=='__main__': unittest.main()
