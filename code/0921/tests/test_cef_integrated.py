"""Behavioral tests for the approved integrated feedback/evaluation contract."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import itertools
import numpy as np

from cef.recipes import normalize_rules, dispatch, summarize_rules, prefer_new
from cef.measurements import SharedMeasurements, observation_key


GEN = '''import numpy as np
def generate(n_cities, seed, parameters):
    return np.random.default_rng(seed).random((n_cities, 2)).tolist()
'''


class RecipeTests(unittest.TestCase):
    def test_model_report_exposes_all_positive_rules_and_is_order_independent(self):
        from cef.integrated_data import render_data_report
        cases=[dict(id='a',valid=True,rule_version='v1',rule_name='grid',reference_gap_pct=.1),
               dict(id='b',valid=True,rule_version='v2',rule_name='hex',reference_gap_pct=.2),
               dict(id='c',valid=True,rule_version='v3',rule_name='strip',reference_gap_pct=0.)]
        report=dict(calibration_source_hash='solver',solver_seed=1,solver_time_limit_s=60,
            comparison='fixed solver against references',unit='percent',direction='maximize',valid=True,
            mean_reference_gap_pct=.1,rule_results=summarize_rules(cases),per_instance=cases)
        rendered=render_data_report(report)
        self.assertEqual(rendered['rules_with_positive_gap_count'],2)
        self.assertEqual([r['rule_name'] for r in rendered['rules_with_positive_gap']],['hex','grid'])
        report['per_instance'].reverse();report['rule_results'].reverse()
        self.assertEqual(render_data_report(report),rendered)

    def test_actual_dispatch_is_seeded_and_preserves_executable_version(self):
        rules = normalize_rules(dict(rules=[dict(name='grid', source=GEN, parameters={}, weight=1),
            dict(name='strip', source=GEN, parameters={'x': 1}, weight=2)]))
        a = dispatch(rules, [11, 12, 13, 14])
        self.assertEqual(a, dispatch(rules, [11, 12, 13, 14]))
        self.assertTrue(all(x['rule_version'] in {r['version'] for r in rules} for x in a))
        self.assertEqual(len({x['id'] for x in a}), 4)
        self.assertNotEqual(rules[0]['version'], rules[1]['version'])

    def test_summary_uses_actual_rule_identity_not_model_names_or_row_order(self):
        rows = [dict(id='a', valid=True, rule_version='grid-v1', rule_name='grid', reference_gap_pct=.1),
                dict(id='b', valid=True, rule_version='strip-v1', rule_name='strip', reference_gap_pct=0),
                dict(id='c', valid=True, rule_version='grid-v1', rule_name='grid', reference_gap_pct=.2)]
        report = summarize_rules(rows)
        grid = next(r for r in report if r['rule_version']=='grid-v1')
        self.assertEqual(grid['count'], 2)
        self.assertAlmostEqual(grid['mean_reference_gap_pct'], .15)
        self.assertFalse(prefer_new(.013680637, .002925211))
        self.assertFalse(prefer_new(.1, .1))
        self.assertTrue(prefer_new(.01, .02))

    def test_invalid_weights_or_duplicate_rule_versions_are_rejected(self):
        for weights in ([0], [-1], [float('nan')]):
            with self.assertRaises(ValueError):
                normalize_rules(dict(rules=[dict(name='x',source=GEN,parameters={},weight=weights[0])]))
        with self.assertRaises(ValueError):
            normalize_rules(dict(rules=[dict(name='a',source=GEN,parameters={},weight=1)]*2))


class MeasurementTests(unittest.TestCase):
    def test_identity_binds_actual_instance_reference_code_seed_and_contract(self):
        base = dict(source='solver', instance={'distance_matrix': [[0, 1], [1, 0]], 'reference': 2},
                    seed=2025, contract={'seconds': 60, 'workers': 4, 'trace': False})
        key = observation_key(**base)
        changes=[{'source':'other'}, {'seed':2026}, {'contract':dict(base['contract'],workers=8)},
                 {'contract':dict(base['contract'],trace=True)},
                 {'instance':dict(base['instance'],reference=3)},
                 {'instance':{'distance_matrix':[[0,2],[2,0]],'reference':2}}]
        for delta in changes:
            self.assertNotEqual(key, observation_key(**dict(base,**delta)))

    def test_reuse_is_one_observation_and_does_not_execute_again_after_resume(self):
        with tempfile.TemporaryDirectory() as td:
            calls=[]
            def execute():
                calls.append(1)
                return dict(valid=True,objective=3.,fitness=0.,solution=[0,1,2],elapsed=.1,started=True)
            store=SharedMeasurements(td)
            first=store.obtain('key',{'source_hash':'x'},execute)
            second=SharedMeasurements(td).obtain('key',{'source_hash':'x'},execute)
            self.assertFalse(first[1]);self.assertTrue(second[1]);self.assertEqual(len(calls),1)
            self.assertEqual(first[0]['solution'],second[0]['solution'])

    def test_unknown_started_work_is_not_silently_reexecuted(self):
        with tempfile.TemporaryDirectory() as td:
            store=SharedMeasurements(td)
            def crash():raise RuntimeError('disconnected worker')
            with self.assertRaises(RuntimeError):store.obtain('key',{},crash)
            with self.assertRaisesRegex(RuntimeError,'unfinished'):
                SharedMeasurements(td).obtain('key',{},lambda:self.fail('must not rerun'))


def small_fixture(folder, rounds=2):
    from test_cef_alternating import config, SOURCE
    from cef.data import receipt, SUPPLEMENT_PROTOCOL
    from cef.integrated_data import save_json
    root=Path(folder);c=config(folder,rounds)
    c.update(integrated_feedback_v1=True,diagnostic_train_limit=4,solver_time_limit_s=.15)
    source=root/'generator.py';source.write_text(GEN)
    params=root/'parameters.json';save_json(params,{})
    rows=[]
    for i,seed in enumerate([71,72,73,74]):
        xy=np.random.default_rng(seed).random((7,2));matrix=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
        route=min(((0,)+tail for tail in itertools.permutations(range(1,7))),key=lambda t:sum(matrix[t[j],t[(j+1)%7]] for j in range(7)))
        objective=float(sum(matrix[route[j],route[(j+1)%7]] for j in range(7)))
        p=root/f'instance_{i}.npz';np.savez(p,coordinates=xy,distance_matrix=matrix)
        ref=root/f'reference_{i}.json';save_json(ref,dict(objective=objective,lower_bound=objective,
            bound_gap_pct=0,solution=route,reference_kind='HiGHS_numerically_bounded_tour'))
        rows.append(dict(id=f'original_{i}' if i<2 else f'supplement_{i-2:04d}',split='train',
            training_group='original' if i<2 else 'supplement',**receipt(p),reference=objective,
            reference_kind='HiGHS_numerically_bounded_tour',reference_record=receipt(ref),
            generator_seed=seed,reference_gap_tolerance_pct=.001,
            source=dict(generator=receipt(source),parameters=receipt(params))))
    manifest=root/'initial_supplement.json';save_json(manifest,dict(train=rows[2:],protocol=SUPPLEMENT_PROTOCOL,diagnostic=True))
    c['data_manifest']=dict(train=rows,training_protocol=SUPPLEMENT_PROTOCOL,supplement_manifest=receipt(manifest))
    c['evaluation_seed_manifest']={r['id']:2025 for r in rows}
    settings=dict(n_cities=7,instances=2,data_seed=42,workers=1,generator_seconds=3,
        reference_seconds=10,reference_gap_pct=.001,reference_work_slices=None)
    return c,settings


class IntegratedLoopTests(unittest.TestCase):
    def test_cold_resume_after_saved_editor_reply_does_not_request_it_again(self):
        from cef.alternating import AlternatingEngine,DataRoundProvider
        from test_cef_alternating import AlgorithmFake
        class Data:
            def __init__(self):self.calls=0
            def call(self,role,payload):
                self.calls+=1
                return dict(rules=[dict(name='uniform',source=GEN,parameters={},weight=1)],reason='fixture')
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td,rounds=2);data=Data();algorithm=AlgorithmFake()
            engine=AlternatingEngine(c,algorithm,DataRoundProvider(settings,data))
            add=engine.archive.add
            def forced_close(kind,value):
                record=add(kind,value)
                if kind=='model_response' and value.get('role')=='editor':raise RuntimeError('forced close after complete reply')
                return record
            with patch.object(engine.archive,'add',side_effect=forced_close):first=engine.run()
            self.assertEqual(first['status'],'failed')
            self.assertEqual(algorithm.edits,1)
            restored=AlternatingEngine(c,algorithm,DataRoundProvider(settings,data))
            output=restored.run()
            self.assertEqual(output['status'],'finished',output['reason'])
            self.assertEqual(algorithm.edits,2)
            self.assertEqual(data.calls,2)
            self.assertEqual(len([r for r in restored.archive.of('model_start') if r['call_id']=='step_0:editor']),1)

    def test_formal_trace_is_reconstructed_after_execution_view_commit_gap(self):
        from cef.alternating import AlternatingEngine
        from cef.research import ResearchTools
        with tempfile.TemporaryDirectory() as td:
            c,_=small_fixture(td,rounds=1)
            c.update(common_start_probe=True,probe_seconds=.05)
            source='''import time
def solve(instance, time_limit_s, seed, *, initial_solution=None, observer=None):
    solution=list(range(len(instance['distance_matrix'])))
    if observer is not None: observer(solution)
    return {"solution":solution}
'''
            Path(c['warm_start_source']).write_text(source)
            provider=lambda engine,number,parent:c['data_manifest']
            engine=AlternatingEngine(c,None,provider);engine.register('warm_start',source)
            with patch.object(ResearchTools,'capture',side_effect=RuntimeError('interrupted before snapshot')):
                with self.assertRaises(RuntimeError):engine.prepare_round()
            first=engine.archive.of('execution')[0]
            self.assertTrue(first['trace'])
            restored=AlternatingEngine(c,None,provider);restored.prepare_round()
            snapshots=restored.archive.of('snapshot')
            self.assertIn(first['instance_id'],{r['instance_id'] for r in snapshots})
            self.assertEqual(len(restored.archive.of('execution_start')),4)

    def test_physical_elapsed_survives_observation_to_view_commit_gap(self):
        from cef.alternating import AlternatingEngine,DataRoundProvider
        from test_cef_alternating import AlgorithmFake
        class Data:
            def call(self,role,payload):
                return dict(rules=[dict(name='new',source=GEN,parameters={},weight=1)],reason='fixture')
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td,rounds=1)
            engine=AlternatingEngine(c,AlgorithmFake(),DataRoundProvider(settings,Data()))
            engine.register('warm_start',Path(c['warm_start_source']).read_text())
            add=engine.archive.add
            def interrupted(kind,data):
                if kind=='execution':raise RuntimeError('interrupted before execution view')
                return add(kind,data)
            def worker(source,adapter,instance,seed,on_started,**kw):
                on_started()
                return dict(valid=True,fitness=0.,objective=instance['reference'],elapsed=1.25,started=True,solution=list(range(7)),wallclock_s=1.25)
            with patch.object(engine.archive,'add',side_effect=interrupted),patch.object(engine.runner,'case',side_effect=worker):
                with self.assertRaises(RuntimeError):engine.measure_rows('warm_start',engine.rows[:1],'gap')
            restored=AlternatingEngine(c,None)
            with patch.object(restored.runner,'case',side_effect=AssertionError('must reuse')):
                restored.measure_rows('warm_start',restored.rows[:1],'gap')
            exported=restored.export('failed','fixture')
            self.assertEqual(exported['resource_usage']['solve_seconds'],1.25)
            self.assertEqual(exported['resource_usage']['case_runs'],1)

    def test_measured_old_new_choice_and_no_automatic_adoption_of_invalid_rules(self):
        from cef.alternating import AlternatingEngine,DataRoundProvider
        class Data:
            def __init__(self,invalid):self.invalid=invalid
            def call(self,role,payload):
                return dict(rules=[dict(name='new',source=GEN,parameters={},weight=0 if self.invalid else 1)],reason='fixture')
        for new_score,invalid,expected in [(2.,False,'new'),(0.,False,'old'),(1.,False,'old'),(2.,True,'old')]:
            with self.subTest(new_score=new_score,invalid=invalid),tempfile.TemporaryDirectory() as td:
                c,settings=small_fixture(td,rounds=1)
                engine=AlternatingEngine(c,None,DataRoundProvider(settings,Data(invalid)))
                engine.register('warm_start',Path(c['warm_start_source']).read_text())
                calls=[]
                def case(source,adapter,instance,seed,on_started,**kwargs):
                    on_started();calls.append(1);score=1. if len(calls)<=4 else new_score
                    return dict(valid=True,fitness=score,objective=instance['reference']*(1+score/100),
                                solution=list(range(7)),started=True,elapsed=.01,wallclock_s=.01)
                with patch.object(engine.runner,'case',side_effect=case):engine.prepare_round()
                selected=json.loads((Path(td)/'rounds/001/data/selection.json').read_text())
                self.assertEqual(selected['selected'],expected)
                self.assertEqual(len(calls),4 if invalid else 6)
                self.assertEqual(len(engine.archive.of('measurement')),1)
                self.assertAlmostEqual(engine.ledger.used('case_runs'),len(calls))

    def test_completed_data_response_is_recovered_if_proposal_publish_was_interrupted(self):
        from cef.alternating import AlternatingEngine,DataRoundProvider
        from cef.integrated_data import save_json
        from test_cef_alternating import AlgorithmFake
        class Data:
            def __init__(self):self.calls=0
            def call(self,role,payload):
                self.calls+=1
                return dict(rules=[dict(name='new',source=GEN,parameters={},weight=1)],reason='fixture')
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td,rounds=1);data=Data();algorithm=AlgorithmFake()
            engine=AlternatingEngine(c,algorithm,DataRoundProvider(settings,data))
            def interrupted(path,value):
                if Path(path).name=='proposal.json':raise RuntimeError('interrupted before publish')
                return save_json(path,value)
            with patch('cef.integrated_data.save_json',side_effect=interrupted):
                first=engine.run()
            self.assertEqual(first['status'],'failed')
            restored=AlternatingEngine(c,algorithm,DataRoundProvider(settings,data))
            self.assertEqual(restored.run()['status'],'finished')
            self.assertEqual(data.calls,1)

    def test_sdk_committed_data_response_is_recovered_without_duplicate_request(self):
        from cef.alternating import AlternatingEngine,DataRoundProvider
        from cef.store import Archive,canonical
        from test_cef_alternating import AlgorithmFake
        class Data:
            def __init__(self,folder):self.archive=Archive(folder);self.calls=0
            def call(self,role,payload):
                self.calls+=1
                response=dict(rules=[dict(name='new',source=GEN,parameters={},weight=1)],reason='fixture')
                self.archive.add('provider_response',dict(role=role,user=canonical(payload),parsed=response))
                raise RuntimeError('interrupted after SDK response commit')
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td,rounds=1);data=Data(Path(td)/'data_agent');algorithm=AlgorithmFake()
            first=AlternatingEngine(c,algorithm,DataRoundProvider(settings,data)).run()
            self.assertEqual(first['status'],'failed')
            restored=AlternatingEngine(c,algorithm,DataRoundProvider(settings,data))
            self.assertEqual(restored.run()['status'],'finished')
            self.assertEqual(data.calls,1)

    def test_default_trace_enabled_data_calibration_waits_for_matching_epoch(self):
        from cef.alternating import AlternatingEngine,DataRoundProvider
        from test_cef_alternating import AlgorithmFake
        class Algorithm(AlgorithmFake):
            def call(self,role,payload):
                value=super().call(role,payload)
                if role=='editor':
                    value['edits'][1]['exact_replacements']=[dict(old='    return {"solution":solution}',new='    # complete program\n    return {"solution":solution}')]
                return value
        class Data:
            def call(self,role,payload):
                return dict(rules=[dict(name='uniform',source=GEN,parameters={},weight=1)],reason='fixture')
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td,rounds=1)
            c['common_start_probe']=True
            c['probe_seconds']=.05
            Path(c['warm_start_source']).write_text('''import time
def solve(instance, time_limit_s, seed, *, initial_solution=None, observer=None):
    solution=list(range(len(instance['distance_matrix']))) if initial_solution is None else list(initial_solution)
    if observer is not None: observer(solution)
    return {"solution":solution}
''')
            engine=AlternatingEngine(c,Algorithm(),DataRoundProvider(settings,Data()))
            output=engine.run()
            self.assertEqual(output['status'],'finished',output['reason'])
            self.assertTrue(engine.archive.of('snapshot'))
            self.assertTrue(all(r.get('training_epoch')==1 for r in engine.archive.of('snapshot')))
            self.assertTrue(engine.state()['current_training_report']['supplement']['rule_results'])

    def test_real_small_two_round_loop_reuses_parent_and_resumes_without_calls(self):
        from cef.alternating import AlternatingEngine,DataRoundProvider
        from test_cef_alternating import AlgorithmFake
        from cef.reporting import verify_export
        class Data:
            def __init__(self):self.inputs=[]
            def call(self,role,payload):
                self.inputs.append(copy.deepcopy(payload))
                source=GEN if len(self.inputs)==1 else GEN.replace('rng', 'rng')+'\n# second proposal\n'
                return dict(rules=[dict(name='random',source=source,parameters={},weight=1)],reason='fixture')
        with tempfile.TemporaryDirectory() as td:
            c,settings=small_fixture(td);data=Data();algorithm=AlgorithmFake()
            engine=AlternatingEngine(c,algorithm,DataRoundProvider(settings,data))
            output=engine.run()
            self.assertEqual(output['status'],'finished',output['reason'])
            self.assertEqual(len(data.inputs),2);self.assertEqual(algorithm.edits,2)
            self.assertLessEqual(len(engine.archive.of('execution_start')),4+2*(2+4))
            self.assertGreater(sum(r.get('reused_observation',False) for r in engine.archive.of('execution')),0)
            self.assertEqual(engine.ledger.used('case_runs'),len(engine.archive.of('execution_start')))
            self.assertTrue(all(r['valid'] for r in engine.archive.of('execution')))
            self.assertEqual(data.inputs[0]['previous_results'][0]['measured_case_count'],2)
            self.assertTrue(data.inputs[0]['previous_results'][0]['rule_results'])
            before=(len(data.inputs),len(algorithm.inputs),len(engine.archive.of('execution_start')))
            restored=AlternatingEngine(c,algorithm,DataRoundProvider(settings,data))
            self.assertEqual(restored.run()['status'],'finished')
            self.assertEqual(before,(len(data.inputs),len(algorithm.inputs),len(restored.archive.of('execution_start'))))
            self.assertEqual(verify_export(c['run_dir'])[1]['best']['source_hash'],output['best']['source_hash'])

    def test_data_model_uses_current_context_without_accumulated_old_assertions(self):
        import os,httpx
        from cef.model import Model
        from cef.store import Archive
        from cef.ledger import BudgetLedger
        from test_cef_sdk import config,sdk_transport,completion
        requests=[]
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200,json=completion(json.dumps(dict(rules=[dict(name='x',source=GEN,parameters={},weight=1)],reason='fixture'))))
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}):
            archive=Archive(td);model=Model(config(integrated_feedback_v1=True,stop_rule='iterations'),archive,BudgetLedger(archive,{},enforce_limits=False))
            with sdk_transport(handler):
                model.call('data_designer',dict(previous_results=[]))
                model.call('data_designer',dict(previous_results=[dict(mean_reference_gap_pct=.1)]))
        self.assertEqual([r['role'] for r in requests[1]['messages']],['system','user'])
        self.assertIn('rules',requests[1]['messages'][0]['content'])


if __name__=='__main__':unittest.main()
