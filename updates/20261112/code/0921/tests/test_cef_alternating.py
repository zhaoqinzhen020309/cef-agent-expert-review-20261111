"""Semantic acceptance for dataset-scoped alternating data/algorithm search."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cef.alternating import AlternatingEngine
from cef.tasks import TaskAdapter
from cef.patch import digest
from cef.reporting import verify_export
from test_cef_iterations_billing import iteration_config

SOURCE='''import time
def solve(instance, time_limit_s, seed):
    return {"solution": list(range(len(instance["distance_matrix"])))}
'''


def config(folder, rounds=2):
    c=iteration_config(str(Path(folder)/'search'),rounds)
    c.update(task='tsp',alternating_agents=True,common_start_probe=False,console_progress=False,
        diagnostic_run=True,diagnostic_train_limit=2,evaluation_policy='joint_only',
        evaluation_workers=1,warm_start_source=str(Path(folder)/'warm.py'))
    Path(c['warm_start_source']).write_text(SOURCE)
    c['data_manifest']={'train':[]}
    c['evaluation_seed_manifest']={'original':2025,'supplement':2025}
    return c


def epoch_rows(round_number):
    matrix=[[0,1,1],[1,0,1],[1,1,0]]
    return dict(training_protocol='cef_tsp_original64_supplement64_v1',train=[
        dict(id='original',split='train',instance=dict(distance_matrix=matrix,reference=3)),
        dict(id='supplement',split='train',training_group='supplement',
             instance=dict(distance_matrix=matrix,reference=3+round_number))])


class AlgorithmFake:
    def __init__(self, recall=False):self.inputs=[];self.edits=0;self.recall=recall;self.did_recall=False
    def call(self,role,payload):
        self.inputs.append((role,copy.deepcopy(payload)))
        if role=='planner':
            if payload['alternating']['phase']=='round_complete':
                return dict(action='advance_round',evidence_ids=[],previous_event_update=[])
            if self.recall and payload['alternating']['round']==2 and not self.did_recall:
                self.did_recall=True
                return dict(action='remeasure_candidate',parent_or_target_id='warm_start',evidence_ids=[],previous_event_update=[])
            return dict(action='revise',edit_count=2,parent_or_target_id=payload['global_best']['candidate_id'],
                issue='offline alternating experiment',proposed_intents=['marker A','marker B'],
                evidence_ids=[],previous_event_update=[],expected_observations=[],resource_request={})
        self.edits+=1
        return dict(package_id=f'new_{self.edits}',parent_id=payload['parent_id'],parent_hash=payload['parent_hash'],
            issue='offline',evidence_ids=[],joint_reason='only joint execution',edits=[
                dict(edit_id='A',purpose='fixture',prediction='measure',exact_replacements=[
                    dict(old='import time',new=f'import time\n# offspring_{self.edits}')]),
                dict(edit_id='B',purpose='fixture',prediction='measure',exact_replacements=[
                    dict(old='def solve(instance, time_limit_s, seed):',new='def solve(instance, time_limit_s, seed):\n    # marker')])])


class AlternatingTests(unittest.TestCase):
    def test_pending_reference_work_can_resume_without_repeating_closed_round(self):
        from cef.reference_jobs import ReferencePending
        with tempfile.TemporaryDirectory() as td:
            c,e,model,calls,case=self.fixture(td)
            provider=e.data_provider;waited=[]
            def pending(engine,number,parent):
                if number==2 and not waited:
                    waited.append(True);raise ReferencePending('saved reference work')
                return provider(engine,number,parent)
            e.data_provider=pending
            with patch.object(e.runner,'case',side_effect=case):
                first=e.run()
                self.assertEqual(first['status'],'waiting_reference')
                self.assertEqual(first['completed_iterations'],1)
                self.assertEqual(len(e.archive.of('round_complete')),1)
                second=e.run()
            self.assertEqual(second['status'],'finished',second['reason'])
            self.assertEqual(model.edits,2)
            self.assertEqual([r[0] for r in calls],[1,2])

    def test_fixed_data_engine_cannot_silently_execute_alternating_config(self):
        from cef.engine import Engine
        with tempfile.TemporaryDirectory() as td:
            c=config(td)
            with self.assertRaisesRegex(ValueError,'AlternatingEngine'):Engine(c,None)
            self.assertFalse(Path(c['run_dir']).exists())

    def test_formal128_two_rounds_execute_only_parent_and_complete_pab(self):
        from scripts.cef_alternating_launcher import build_auto_config
        from runCEFAuto import CONFIG
        from cef.data import bind_supplement
        from test_cef_supplement_binding import supplement_fixture
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);initial=folder/'initial';initial.mkdir();new=folder/'next';new.mkdir()
            first=supplement_fixture(initial);second=supplement_fixture(new)
            warm=folder/'warm.py';warm.write_text(SOURCE)
            settings=dict(CONFIG,integrated_feedback_v1=False,measured_research_loop_v1=False,exploration_efficiency_v1=False,
                rounds=2,workers=2,common_start_probe=False,initial_solver=warm,initial_supplement=first)
            c,test=build_auto_config(settings,folder/'run',2026);c['console_progress']=False
            original=copy.deepcopy(c['data_manifest']);original['train']=original['train'][:64]
            for key in ('training_protocol','supplement_manifest','audit'):original.pop(key,None)
            provider_calls=[]
            def provider(engine,number,parent):
                provider_calls.append((number,parent['candidate_id']))
                return bind_supplement(original,first if number==1 else second)
            e=AlternatingEngine(c,AlgorithmFake(),provider)
            def case(source,adapter,instance,seed,on_started,**kw):
                self.assertEqual(seed,2026);on_started()
                score=0. if '# offspring_1' in source else 1.
                return dict(valid=True,fitness=score,objective=instance['reference']*(1+score/100),started=True,elapsed=.01,wallclock_s=.02)
            with patch.object(e.runner,'case',side_effect=case) as run:out=e.run()
            self.assertEqual(out['status'],'finished',out['reason'])
            self.assertEqual(run.call_count,512)
            self.assertEqual(len(e.archive.of('package')),2)
            self.assertEqual([r[0] for r in provider_calls],[1,2])
            self.assertEqual(e.config['data_manifest']['train'][:64],original['train'])
            self.assertEqual(e.config['data_manifest']['test'],test['instances'])
            self.assertEqual(len(e.rows),128)
            self.assertTrue(all(r.get('variant')=='PAB' for r in e.archive.of('candidate') if r['candidate_id']!='warm_start'))

    def test_diagnostics_and_history_keep_their_dataset_epoch(self):
        from cef.research import ResearchTools
        from cef.feedback import planner_history_observation
        from cef.data import receipt
        with tempfile.TemporaryDirectory() as td:
            c,e,model,calls,case=self.fixture(td)
            e.register('warm_start',SOURCE)
            with patch.object(e.runner,'case',side_effect=case):e.prepare_round()
            e.research=ResearchTools(e)
            record=dict(valid=True,trace=[dict(solution=[0,1,2],elapsed_s=.01),dict(solution=[0,1,2],elapsed_s=.8)],elapsed=1.,seed=2025)
            e.research.capture('warm_start',e.rows[0],record)
            self.assertTrue(e.research.states('warm_start'))
            path=Path(td)/'epoch2.json'
            path.write_text(json.dumps(dict(round=2,solver_seed=2025,data_manifest=epoch_rows(2))))
            e.activate(dict(round=2,artifact=receipt(path)))
            e.research=ResearchTools(e)
            self.assertEqual(e.research.states('warm_start'),[])
            e.archive.add('probe_result',dict(event_id='old_probe',training_epoch=1))
            with self.assertRaisesRegex(ValueError,'different training epoch'):
                e.research.integration(dict(revision_goal='integrate_probe',integration_probe_id='old_probe'))
            payload=dict(alternating=dict(round=1,evaluation_hash='epoch1'),recent_events=[
                dict(event_id='x',evaluation_hash='epoch1',current_dataset=True)],
                global_best=dict(candidate_id='x',fitness=1.,evaluation_epoch=1,fitness_evaluation_hash='epoch1'))
            compact=planner_history_observation(payload)
            self.assertEqual(compact['recent_events'][0]['evaluation_hash'],'epoch1')
            self.assertEqual(compact['global_best']['fitness_evaluation_hash'],'epoch1')
            self.assertEqual(compact['alternating']['round'],1)

    def test_real_provider_replaces_supplement_and_preserves_original_and_test(self):
        from types import SimpleNamespace
        from cef.alternating import DataRoundProvider
        from cef.data import bind_supplement
        from cef.store import Archive
        from test_cef_supplement_binding import supplement_fixture,ROOT
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);one=root/'initial';one.mkdir();two=root/'next';two.mkdir()
            initial=supplement_fixture(one);new=supplement_fixture(two)
            original=json.loads((ROOT/'configs/cef_v1_3/tsp.template.json').read_text())['data_manifest']
            merged=bind_supplement(original,initial)
            archive=Archive(root/'search');engine=SimpleNamespace(archive=archive,frozen_config={'data_manifest':merged,'evaluation_seed_manifest':{'x':2026}})
            solver=root/'solver.py';solver.write_text(SOURCE);parent=dict(source_hash=digest(SOURCE),source_path=str(solver))
            class Designer:
                def __init__(self,settings,model):self.settings=settings
                def run(self):return dict(status='finished',supplement_manifest=str(new))
            provider=DataRoundProvider({},None)
            with patch('cef.alternating.DataDesigner',Designer):out=provider(engine,1,parent)
            self.assertEqual(len(out['train']),128)
            self.assertEqual(out['train'][:64],original['train'])
            self.assertEqual(out['test'],original['test'])
            self.assertEqual(out['train'][64:],json.loads(new.read_text())['train'])

    def fixture(self, folder, recall=False, rounds=2):
        c=config(folder,rounds);model=AlgorithmFake(recall);calls=[]
        def provider(engine,round_number,parent):
            calls.append((round_number,parent['candidate_id'],parent['source_hash']))
            return epoch_rows(round_number)
        engine=AlternatingEngine(c,model,provider)
        def case(source,adapter,instance,seed,on_started,**kw):
            on_started()
            if engine.training_epoch==1:
                score=.5 if '# offspring_1' in source else 5.
            else:
                score=20. if '# offspring_2' in source else 10. if '# offspring_1' in source else 30.
            return dict(valid=True,fitness=score,objective=instance['reference']*(1+score/100),
                started=True,elapsed=.01,wallclock_s=.02)
        return c,engine,model,calls,case

    def test_ten_automatic_rounds_make_exactly_ten_new_complete_programs(self):
        with tempfile.TemporaryDirectory() as td:
            c,e,model,calls,case=self.fixture(td,rounds=10)
            with patch.object(e.runner,'case',side_effect=case) as worker:out=e.run()
            self.assertEqual(out['status'],'finished',out['reason'])
            self.assertEqual([r[0] for r in calls],list(range(1,11)))
            self.assertEqual(out['completed_iterations'],10)
            self.assertEqual(len(e.archive.of('round_complete')),10)
            self.assertEqual(len(e.archive.of('package')),10)
            self.assertEqual(worker.call_count,40)
            self.assertEqual(len(e.rows),2) # deliberately tiny offline fixture, never a formal result

    def test_new_dataset_excludes_old_scores_and_feeds_updated_solver_to_data(self):
        with tempfile.TemporaryDirectory() as td:
            c,e,model,calls,case=self.fixture(td)
            with patch.object(e.runner,'case',side_effect=case) as worker:out=e.run()
            self.assertEqual(out['status'],'finished',out['reason'])
            self.assertEqual(out['completed_iterations'],2)
            self.assertEqual(len(e.archive.of('round_complete')),2)
            self.assertEqual(worker.call_count,8) # parent+new, two cases, two rounds
            winner=e.archive.of('round_complete')[0]['winner']
            self.assertEqual(calls[1][1],winner['candidate_id'])
            self.assertEqual(calls[1][2],winner['source_hash'])
            self.assertEqual(out['best']['fitness'],10.) # not mean(.5,10)
            self.assertFalse(e.candidates()['warm_start']['eligibility'])
            self.assertEqual(e.candidates()['warm_start']['historical_fitness'],5.)
            self.assertEqual(len(e.archive.of('package')),2)
            self.assertTrue(all(set(r['variant_ids'])=={'PAB'} for r in e.archive.of('contrast')))
            self.assertEqual(e.training_epoch,2)
            planners=[p for role,p in model.inputs if role=='planner' and p['alternating']['phase']=='algorithm']
            self.assertEqual(planners[1]['global_best']['fitness'],10.)
            self.assertEqual(planners[1]['alternating']['round'],2)
            self.assertEqual(verify_export(c['run_dir'])[1]['best']['fitness'],10.)

    def test_historical_candidate_is_remeasured_before_becoming_eligible(self):
        with tempfile.TemporaryDirectory() as td:
            c,e,model,calls,case=self.fixture(td,recall=True)
            with patch.object(e.runner,'case',side_effect=case) as worker:out=e.run()
            self.assertEqual(out['status'],'finished',out['reason'])
            self.assertEqual(worker.call_count,10)
            self.assertEqual(e.candidates()['warm_start']['fitness'],30.)
            self.assertEqual(e.candidates()['warm_start']['historical_fitness'],5.)
            self.assertTrue(e.candidates()['warm_start']['eligibility'])
            self.assertEqual(len(e.archive.of('historical_remeasurement')),1)

    def test_completed_round_resume_does_not_repeat_data_or_algorithm_calls(self):
        with tempfile.TemporaryDirectory() as td:
            c,e,model,calls,case=self.fixture(td)
            original=e.archive.add
            def stop(kind,data):
                item=original(kind,data)
                if kind=='round_complete' and data['round']==1:raise RuntimeError('simulated post-checkpoint interruption')
                return item
            with patch.object(e.runner,'case',side_effect=case),patch.object(e.archive,'add',side_effect=stop):
                out=e.run()
            self.assertEqual(out['status'],'failed')
            count=len(model.inputs)
            restored=AlternatingEngine(c,model,e.data_provider)
            def real_fake(*args,**kwargs):
                on_started=args[4] if len(args)>4 else kwargs['on_started'];on_started()
                return dict(valid=True,fitness=1.,objective=3.,started=True,elapsed=.01,wallclock_s=.02)
            with patch.object(restored.runner,'case',side_effect=real_fake):out=restored.run()
            self.assertEqual(out['status'],'finished',out['reason'])
            self.assertEqual([r[0] for r in calls],[1,2])
            self.assertEqual(len(restored.archive.of('package')),2)
            self.assertEqual(len(model.inputs)-count,3) # planner, editor, advance


if __name__=='__main__':unittest.main()
