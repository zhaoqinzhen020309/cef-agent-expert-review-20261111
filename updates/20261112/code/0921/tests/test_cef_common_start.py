"""Behavioral controller acceptance, real small solver workers, offline model."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from cef.patch import digest
from cef.store import canonical
from test_cef_iterations_billing import iteration_config


def source(mode):
    return '''import time
MODE = %d
def search(route, limit, observer):
    route = list(route)
    if (MODE == 0 and limit > 0.1) or (MODE == 1 and limit < 0.1) or MODE == 2:
        route = [0, 1, 2, 3]
    if observer is not None:
        observer(route)
    return route
def solve(instance, time_limit_s, seed, *, initial_solution=None, observer=None):
    route = [0, 2, 1, 3] if initial_solution is None else list(initial_solution)
    if observer is not None:
        observer(route)
    return {"solution": search(route, time_limit_s, observer)}
''' % mode


def settings(folder):
    config = iteration_config(folder, 3)
    config.update(task='tsp', evaluation_policy='joint_only', common_start_probe=True,
                  probe_states=1, probe_seconds=0.04, probe_restart_seeds=[101, 102],
                  solver_time_limit_s=0.5, evaluation_workers=2)
    matrix = [[0, 1, 1.41421356237, 1], [1, 0, 1, 1.41421356237],
              [1.41421356237, 1, 0, 1], [1, 1.41421356237, 1, 0]]
    config['data_manifest'] = {'train': [dict(id='tiny', split='train',
        instance={'distance_matrix': matrix, 'reference': 4})]}
    config['evaluation_seed_manifest'] = {'tiny': 2025}
    return config


class ExperimentModel:
    def __init__(self,negative=False):
        self.inputs = []
        self.generated = 0
        self.probed = False
        self.negative=negative

    def call(self, role, payload):
        self.inputs.append((role, payload))
        if role == 'planner':
            if self.generated == 2 and not self.probed:
                self.probed = True
                return dict(action='compare_from_states', reference_candidate_id='step_0:PAB',
                            candidate_id='step_1:PAB', question='Compare continuation from the same solution.',
                            evidence_ids=['step_1:contrast'], previous_event_update=[])
            decision = dict(action='revise', edit_count=2, issue='offline controlled fixture',
                proposed_intents=['retain search', 'adjust integration'], expected_observations=[],
                evidence_ids=[], previous_event_update=[],
                parent_or_target_id='seed' if self.generated == 0 else payload['active_parent']['candidate_id'])
            if self.probed:
                result = payload['common_start']['latest_results'][-1]
                if any(r['event_id']==result['event_id'] for r in payload['action_context']['integration_opportunities']):
                    decision.update(revision_goal='integrate_probe', integration_probe_id=result['event_id'],
                                    parent_or_target_id=result['candidate_id'], evidence_ids=[result['event_id']])
            return decision
        first, _, tail = payload['parent_code'].partition('\n')
        mode=self.generated
        if mode==1 and self.negative: mode=3
        if mode==2:
            # The emitted algorithm genuinely depends on the observed experiment.
            mode=2 if payload.get('integration_context',{} ) else 4
        new = source(mode)
        self.generated += 1
        return dict(package_id='offline_' + str(self.generated), parent_id=payload['parent_id'],
            parent_hash=payload['parent_hash'], issue='measured fixture', evidence_ids=payload['decision']['evidence_ids'],
            joint_reason='Only the complete generated program is measured.', edits=[
                dict(edit_id='A', purpose='implementation', prediction='measure full solver',
                     exact_replacements=[dict(old=tail, new=new.partition('\n')[2])]),
                dict(edit_id='B', purpose='version marker', prediction='joint only',
                     exact_replacements=[dict(old=first+'\n', new=first+'\n# fixture '+str(self.generated)+'\n')])])


class CommonStartControllerTests(unittest.TestCase):
    def test_real_offline_experiment_changes_next_generation_without_ranking_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            model = ExperimentModel()
            engine = Engine(settings(folder), model)
            result = engine.run()
            self.assertEqual(result['status'], 'finished', result['reason'])
            self.assertEqual(result['completed_iterations'], 3)
            self.assertEqual(len(engine.archive.of('measurement')), 3)
            probes = engine.archive.of('probe_result')
            self.assertEqual(len(probes), 1, 'Agent must actually execute a comparison')
            self.assertEqual(probes[0]['status'], 'supported', probes[0])
            self.assertEqual(len(engine.archive.of('probe_execution')), 4)
            self.assertEqual(engine.ledger.used('case_runs'), 7)
            self.assertEqual(len(engine.archive.of('integration_attempt')), 1)
            integrated = [p for role,p in model.inputs if role == 'editor' and p['decision'].get('revision_goal') == 'integrate_probe']
            self.assertEqual(len(integrated), 1)
            self.assertEqual(integrated[0]['integration_context']['probe']['event_id'], probes[0]['event_id'])
            self.assertIn('MODE = 1', integrated[0]['parent_code'])
            final_source = engine.archive.read_source(engine.candidates()['step_3:PAB']['source_hash'])
            self.assertIn('MODE = 2', final_source)
            self.assertEqual(result['best']['fitness'], 0)
            self.assertTrue((Path(folder)/'integration_summary.json').exists())
            requests = len(model.inputs)
            resumed = Engine(settings(folder), model).run()
            self.assertEqual(resumed['status'], 'finished')
            self.assertEqual(len(model.inputs), requests)

    def test_bootstrap_has_no_probe_action_and_final_round_does_not_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            config = settings(folder); config['max_iterations'] = 1
            model = ExperimentModel(); engine = Engine(config, model)
            result = engine.run()
            self.assertEqual(result['status'], 'finished', result['reason'])
            self.assertNotIn('compare_from_states', model.inputs[0][1]['action_context']['allowed_actions'])
            self.assertEqual(engine.archive.of('probe_execution'), [])
            self.assertNotIn('compare_from_states', engine.action_context()['allowed_actions'])

    def test_probe_facts_never_make_unmeasured_program_rankable(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Engine(settings(folder), None)
            engine.register('unmeasured', source(1))
            engine.archive.add('probe_result', dict(event_id='probe:test', candidate_id='unmeasured',
                status='supported', mean_delta=1))
            self.assertIsNone(engine.best())

    def test_negative_probe_changes_the_next_program_not_only_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=Engine(settings(folder),ExperimentModel(negative=True))
            result=engine.run()
            self.assertEqual(result['status'],'finished',result['reason'])
            self.assertEqual(engine.archive.of('probe_result')[0]['status'],'not_supported')
            self.assertEqual(engine.archive.of('integration_attempt'),[])
            self.assertIn('MODE = 4',engine.archive.read_source(engine.candidates()['step_3:PAB']['source_hash']))

    def test_resume_reuses_completed_probe_and_integration_without_new_starts(self):
        for stop_kind in ('probe_result','integration_attempt'):
            with self.subTest(stop_kind=stop_kind),tempfile.TemporaryDirectory() as folder:
                model=ExperimentModel(); engine=Engine(settings(folder),model)
                original=engine.archive.add; injected=[False]
                def add(kind,data):
                    out=original(kind,data)
                    if kind==stop_kind and not injected[0]:
                        injected[0]=True
                        raise RuntimeError('simulated crash after durable '+kind)
                    return out
                with patch.object(engine.archive,'add',side_effect=add):
                    first=engine.run()
                self.assertEqual(first['status'],'failed')
                starts=len(engine.archive.of('execution_start'))
                restored=Engine(settings(folder),model); result=restored.run()
                self.assertEqual(result['status'],'finished',result['reason'])
                expected_extra=1 if stop_kind=='probe_result' else 0
                self.assertEqual(len(restored.archive.of('execution_start')),starts+expected_extra)
                self.assertEqual(len(restored.archive.of('probe_result')),1)
                self.assertEqual(len(restored.archive.of('integration_attempt')),1)
                probe=restored.archive.of('probe_result')[0]
                folder_path=Path(folder)/'probes'/probe['event_id'].split(':',1)[1]
                self.assertTrue((folder_path/'summary.json').is_file())
                self.assertTrue((folder_path/'results.jsonl').is_file())

    def test_state_selection_seed_is_independent_of_dictionary_order(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            left=settings(a); right=settings(b)
            left['evaluation_seed_manifest']={'x':2025,'y':2026}
            right['evaluation_seed_manifest']={'y':2026,'x':2025}
            self.assertEqual(Engine(left,None).research.seed,Engine(right,None).research.seed)

    def test_resume_never_repeats_interrupted_probe_and_reports_unknown_time(self):
        with tempfile.TemporaryDirectory() as folder:
            model=ExperimentModel(); engine=Engine(settings(folder),model)
            self.assertEqual(engine.run()['status'],'finished')
            journal=engine.archive.records
            cut=next(r['seq'] for r in journal if r['kind']=='execution_start' and r['data']['execution_id'].startswith('probe:'))
            engine.archive.path.write_text(''.join(canonical(r)+'\n' for r in journal[:cut+1]),encoding='utf-8',newline='\n')
            model.generated=2; model.probed=True
            unknown=[journal[cut]['data']]
            self.assertTrue(unknown)
            restored=Engine(settings(folder),model); result=restored.run()
            self.assertEqual(result['status'],'finished',result['reason'])
            for started in unknown:
                self.assertEqual(sum(r['execution_id']==started['execution_id'] for r in restored.archive.of('execution_start')),1)
            probe=restored.archive.of('probe_result')[0]
            self.assertEqual(probe['status'],'inconclusive')
            self.assertIsNone(probe['usage']['elapsed'])
            self.assertEqual(result['resource_usage']['case_runs'],
                             result['resource_usage']['formal_case_runs']+result['resource_usage']['probe_case_runs'])

    def test_matrix_content_change_is_rejected_even_when_tour_cost_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            config=settings(folder)
            instance=config['data_manifest']['train'][0]['instance']
            path=Path(folder)/'training.npz'
            matrix=np.asarray(instance['distance_matrix'])
            np.savez(path,distance_matrix=matrix)
            config['data_manifest']['train']=[dict(id='tiny',split='train',path=str(path),reference=4)]
            engine=Engine(config,None)
            for cid,mode in (('a',0),('b',1)):
                engine.register(cid,source(mode))
                engine.archive.add('measurement',dict(candidate_id=cid,valid=True,fitness=mode))
            row=engine.rows[0]; start=[0,2,1,3]
            record=dict(valid=True,seed=2025,elapsed=.5,
                        trace=[dict(solution=start,elapsed_s=0.)])
            engine.research.capture('a',row,record)
            engine.archive.add('contrast',dict(event_id='test:contrast',variant_ids={'PAB':'b'},parent_id='a',global_best_updated=False))
            matrix[0,1]=matrix[1,0]=9
            np.savez(path,distance_matrix=matrix)
            request=dict(reference_candidate_id='a',candidate_id='b',question='test')
            with self.assertRaisesRegex(ValueError,'hash'):
                engine.research.run(request,'test')


if __name__ == '__main__':
    unittest.main()
