"""Current Agent input contracts; real SDK mock transport and small workers only."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from cef.ledger import BudgetLedger
from cef.model import Model
from cef.store import Archive, canonical
from cef.tasks import TaskAdapter
from test_cef_engine import config
from test_cef_sdk import completion, config as sdk_config, sdk_transport


class AgentInputTests(unittest.TestCase):
    def test_metric_units_are_native_and_candidate_displays_do_not_change_scores(self):
        for task, value, unit, display in [('tsp', 0.000185, 'percent', '1.850000000e-04%'),
                                           ('fssp', 31.5, 'processing_time', '31.5')]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as folder:
                c = dict(config(folder), task=task, evaluation_policy='joint_only')
                e = Engine(c, None)
                e.register('seed', e.adapter.seed_source())
                e.archive.add('measurement', dict(measurement_id='fixture', candidate_id='seed',
                                                  valid=True, fitness=value))
                original = canonical(e.archive.records)
                state = e.state()
                self.assertEqual(state['fitness_metric']['unit'], unit)
                self.assertEqual(state['global_best']['fitness'], value)
                self.assertEqual(state['global_best']['fitness_display'], display)
                self.assertEqual(state['active_parent']['fitness_display'], display)
                context = e.editor_context('seed', {'evidence_ids': []}, 2)
                self.assertEqual(context['fitness_metric'], state['fitness_metric'])
                self.assertEqual(canonical(e.archive.records), original)
                e.config['mode'] = 'B1'
                self.assertNotIn('fitness', e.state()['active_parent'])
                self.assertNotIn('fitness_display', e.state()['active_parent'])

    def test_event_score_display_keeps_negative_and_missing_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            e = Engine(dict(config(folder), task='tsp'), None)
            event = {'event_id': 'x', 'scores': {'PAB': -0.01, 'PA': None}}
            before = copy.deepcopy(event)
            view = e.visible_event(event)
            self.assertEqual(view['scores'], before['scores'])
            self.assertEqual(view['score_unit'], 'percent')
            self.assertEqual(view['scores_display'], {'PAB': '-1.000000000e-02%', 'PA': None})
            self.assertEqual(event, before)

    def test_current_tool_observation_is_compact_and_keeps_raw_archive(self):
        from test_cef_common_start import ExperimentModel, settings
        with tempfile.TemporaryDirectory() as folder:
            fixture = ExperimentModel()
            engine = Engine(settings(folder), fixture)
            result = engine.run()
            self.assertEqual(result['status'], 'finished', result['reason'])
            self.assertEqual(result['completed_iterations'], 3)
            probe = engine.archive.of('probe_result')[0]
            contract = probe['experiment']
            self.assertEqual(contract['method'], 'paired_common_start')
            self.assertEqual(contract['comparison_unit'], 'complete_solver_continuation')
            self.assertEqual(contract['time_limit_s'], engine.config['probe_seconds'])
            self.assertEqual(contract['states'], 1)
            self.assertEqual(contract['restart_seeds'], [101, 102])
            self.assertEqual(probe['unit'], 'ratio')
            self.assertTrue(probe['run_results'])
            after_probe = [p for role, p in fixture.inputs if role == 'planner' and p.get('tool_observations')][-1]
            view = after_probe['tool_observations'][0]['result']
            self.assertNotIn('run_results', view)
            self.assertEqual(view['event_id'], probe['event_id'])
            self.assertEqual(view['experiment'], contract)
            self.assertEqual(view['result_artifact'], probe['result_artifact'])
            self.assertEqual(len(engine.archive.of('execution')), 3)
            self.assertEqual(engine.ledger.used('case_runs'), 7)

    def test_sdk_history_replays_small_observations_and_original_decisions(self):
        from test_cef_tool_interface import measured_result
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'CEF_TEST_KEY': 'offline'}):
            archive = Archive(folder)
            ledger = BudgetLedger(archive, dict(max_llm_requests=4, max_llm_tokens=500000,
                max_case_runs=1, max_analysis_seconds=1, max_wallclock_s=60))
            model = Model(sdk_config(), archive, ledger)
            old = dict(current_program='OLD_CODE_' * 5000,
                       active_parent=dict(candidate_id='old', source_hash='hash', fitness=0.006522),
                       recent_events=[dict(event_id='ev', parent_id='seed', scores={'PAB': 0.006522},
                                           execution_ids=['large'] * 5000)],
                       common_start=dict(latest_results=[measured_result()]))
            decision = dict(action='revise', issue='original model reasoning',
                            parent_or_target_id='old', edit_count=2, proposed_intents=['A', 'B'])
            requests = []
            def handler(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json=completion(json.dumps(decision)))
            with sdk_transport(handler):
                model.call('planner', old)
                saved = archive.path.read_bytes()
                current = dict(current_program='LATEST COMPLETE CODE', extra='current input stays intact')
                model.call('planner', current)
            messages = requests[1]['messages']
            historical = json.loads(messages[1]['content'])
            self.assertFalse('current_program' in historical)
            self.assertEqual(historical['active_parent']['candidate_id'], 'old')
            self.assertEqual(historical['recent_events'][0]['scores'], {'PAB': 0.006522})
            self.assertNotIn('execution_ids', historical['recent_events'][0])
            self.assertEqual(historical['common_start']['latest_results'][0]['candidate_better_states'], 4)
            self.assertNotIn('run_results', historical['common_start']['latest_results'][0])
            self.assertEqual(json.loads(messages[2]['content']), decision)
            self.assertEqual(json.loads(messages[-1]['content']), current)
            self.assertLess(len(messages[1]['content']), len(canonical(old)) / 10)
            self.assertEqual(json.loads(archive.of('provider_response')[0]['user']), old)
            self.assertTrue(archive.path.read_bytes().startswith(saved))


if __name__ == '__main__':
    unittest.main()
