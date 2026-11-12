"""Tool-contract behavior, not a claim about a model's research judgment."""
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
from cef.model import Model
from cef.ledger import BudgetLedger
from cef.replay import summarize
from cef.research import ResearchTools
from cef.store import Archive, canonical
from test_cef_common_start import settings, source
from test_cef_sdk import completion, config as sdk_config, sdk_transport


def measured_result(candidate_costs=None, incomplete=False):
    states = [dict(snapshot_id=f'state:{i}', objective=100.) for i in range(8)]
    seeds = [101, 102]
    costs = candidate_costs or [80., 80., 80., 80., 91., 90., 90., 90.]
    records = [dict(snapshot_id=s['snapshot_id'], restart_seed=seed, role=role,
                    valid=True, start_objective=100., objective=cost,
                    elapsed=3., wallclock_s=3., cpu_seconds=3.)
               for i, s in enumerate(states) for seed in seeds
               for role, cost in [('reference', 90.), ('candidate', costs[i])]]
    if incomplete:
        records.pop()
    return dict(summarize(records, states, seeds), event_id='probe:contract',
        reference_candidate_id='reference:PAB', candidate_id='candidate:PAB',
        reference_hash='reference-hash', candidate_hash='candidate-hash',
        question='Investigate the full-score regression and construction.',
        unit='ratio', iteration=2, result_artifact={'path': 'raw.json', 'sha256': 'raw-hash'},
        experiment=dict(method='paired_common_start', comparison_unit='complete_solver_continuation',
                        states=8, restart_seeds=seeds, time_limit_s=3.))


def engine_with_result(folder, result):
    e = Engine(settings(folder), None)
    for cid, mode, score in [('reference:PAB', 0, 0.), ('candidate:PAB', 1, 1.)]:
        e.register(cid, source(mode))
        e.archive.add('measurement', dict(candidate_id=cid, valid=True, fitness=score))
    e.archive.add('probe_result', result)
    return e


class ObservationContractTests(unittest.TestCase):
    def test_report_describes_measurements_without_hypothesis_status(self):
        raw = measured_result()
        before = copy.deepcopy(raw)
        view = ResearchTools.brief(raw)
        self.assertNotIn('status', view)
        self.assertNotIn('question', view)
        self.assertNotIn('run_results', view)
        self.assertEqual(view['research_purpose'], raw['question'])
        self.assertEqual(view['candidate_better_states'], 4)
        self.assertEqual(view['reference_better_states'], 1)
        self.assertEqual(view['tied_states'], 3)
        self.assertEqual(view['execution'], dict(completed=True, valid_runs=32, total_runs=32, expected_runs=32, errors=[]))
        self.assertEqual(view['experiment'], raw['experiment'])
        self.assertEqual(view['result_artifact'], raw['result_artifact'])
        self.assertIn('candidate minus reference', view['metric_definition'])
        self.assertIn('4', view['report'])
        self.assertNotIn('not_supported', canonical(view))
        self.assertEqual(raw, before)
        self.assertEqual(ResearchTools.brief(view), view)

    def test_research_purpose_does_not_change_the_executed_comparison(self):
        raw = measured_result()
        other = dict(raw, question='Is the reference worse? Also investigate the constructor.')
        a, b = ResearchTools.brief(raw), ResearchTools.brief(other)
        a.pop('research_purpose'); b.pop('research_purpose')
        self.assertEqual(a, b)

    def test_swapping_programs_reverses_named_measurements(self):
        raw = measured_result()
        swapped = copy.deepcopy(raw)
        states = [dict(snapshot_id=f'state:{i}', objective=100.) for i in range(8)]
        records = copy.deepcopy(raw['run_results'])
        for row in records:
            row['role'] = 'reference' if row['role'] == 'candidate' else 'candidate'
        swapped.update(summarize(records, states, [101, 102]),
                       reference_candidate_id=raw['candidate_id'], candidate_id=raw['reference_candidate_id'])
        a, b = ResearchTools.brief(raw), ResearchTools.brief(swapped)
        self.assertEqual((b['candidate_better_states'], b['reference_better_states']), (1, 4))
        self.assertAlmostEqual(a['mean_candidate_minus_reference_gain_ratio'], -b['mean_candidate_minus_reference_gain_ratio'])
        self.assertEqual(b['candidate_id'], raw['reference_candidate_id'])

    def test_incomplete_execution_is_not_presented_as_complete_or_a_zero_mean(self):
        raw = measured_result(incomplete=True)
        view = ResearchTools.brief(raw)
        self.assertFalse(view['execution']['completed'])
        self.assertEqual(view['execution']['valid_runs'], 31)
        self.assertEqual(view['execution']['expected_runs'], 32)
        self.assertTrue(view['execution']['errors'])
        self.assertIsNone(view['mean_candidate_minus_reference_gain_ratio'])
        self.assertEqual(sum(s['valid'] for s in view['state_results']), 7)

    def test_action_context_explains_both_available_and_unavailable_integration(self):
        for costs, available in [(None, False), ([90.] * 8, False), ([80.] * 8, True),
                                  ([80.] * 5 + [300.] * 3, False)]:
            with self.subTest(costs=costs), tempfile.TemporaryDirectory() as folder:
                e = engine_with_result(folder, measured_result(costs))
                context = e.action_context()
                option = context['integration_options'][0]
                self.assertEqual(option['available'], available)
                self.assertEqual(option['requirements']['minimum_candidate_wins'], 5)
                self.assertEqual(option['requirements']['mean_gain_difference_above'], 1e-12)
                self.assertEqual(len(context['integration_opportunities']), int(available))
                self.assertIn('revise', context['allowed_actions'])
                self.assertNotIn('status', e.research.context()['latest_results'][0])
                self.assertNotIn('integration_opportunities', e.research.context())
                if available:
                    e.archive.add('integration_attempt', dict(probe_id='probe:contract'))
                    self.assertFalse(e.action_context()['integration_options'][0]['available'])
                    self.assertEqual(e.action_context()['integration_opportunities'], [])

    def test_raw_inspection_editor_and_analysis_preserve_the_same_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = measured_result()
            e = engine_with_result(folder, raw)
            expected = ResearchTools.brief(raw)
            e.tool(dict(action='inspect', parent_or_target_id=raw['event_id']), 'inspect-step')
            # Inspection is the full stored artifact; editor/analysis use its
            # readable report. Compare meaning, retaining the raw trace as well.
            inspected=e.archive.of('tool_result')[-1]['result']
            self.assertEqual(inspected,raw)
            self.assertEqual(ResearchTools.brief(inspected), expected)
            editor = e.editor_context('reference:PAB', {'evidence_ids': [raw['event_id']]}, 2)
            self.assertEqual(editor['probe_evidence'], [expected])
            captured = []
            def analysis(code, events, seconds):
                captured.extend(events)
                return {'elapsed': 0., 'result': 'measured'}
            with patch('cef.analysis.analyze', side_effect=analysis):
                e.tool(dict(action='analyze', analysis_code='unused'), 'analysis-step')
            self.assertEqual(captured, [expected])
            self.assertEqual(e.archive.of('probe_result'), [raw])

    def test_specialized_revision_keeps_the_existing_two_result_window(self):
        with tempfile.TemporaryDirectory() as folder:
            raw=measured_result([80.]*8)
            e=engine_with_result(folder,raw)
            for name in ('probe:second','probe:third'):
                e.archive.add('probe_result',dict(raw,event_id=name))
            self.assertEqual([r['event_id'] for r in e.action_context()['integration_opportunities']],
                             ['probe:second','probe:third'])

    def test_tool_definition_states_the_operation_argument_meanings_and_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            e = engine_with_result(folder, measured_result())
            context = e.research.context()
            tool = context['tool']
            self.assertEqual(tool['name'], 'compare_from_states')
            self.assertIn('research purpose', tool['arguments']['question'])
            self.assertIn('two complete solvers', tool['description'])
            self.assertIn('identical frozen training solutions', tool['description'])
            self.assertIn('same restart seeds and time limits', tool['description'])
            self.assertEqual(context['execution_conditions']['time_limit_s'], e.config['probe_seconds'])
            self.assertEqual(context['execution_conditions']['restart_seeds'], e.config['probe_restart_seeds'])

    def test_sdk_history_and_tool_inspection_replay_use_the_same_report(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'CEF_TEST_KEY': 'offline'}):
            archive = Archive(folder)
            ledger = BudgetLedger(archive, dict(max_llm_requests=3, max_llm_tokens=500000,
                max_case_runs=1, max_analysis_seconds=1, max_wallclock_s=60))
            model = Model(dict(sdk_config(), common_start_probe=True), archive, ledger)
            raw = measured_result()
            old = dict(common_start=dict(latest_results=[raw]),
                tool_observations=[dict(action='inspect', result=raw)])
            decision = dict(action='revise', issue='offline transport fixture')
            requests = []
            def handler(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json=completion(json.dumps(decision)))
            with sdk_transport(handler):
                model._call_once('planner', old, 0)
                stored = archive.path.read_bytes()
                model._call_once('planner', {'current_program': 'current source'}, 0)
            replayed = json.loads(requests[1]['messages'][1]['content'])
            # Historical context is compacted, while explicit inspect retains
            # the original artifact. Both must retain all measured comparisons.
            history=replayed['common_start']['latest_results']
            self.assertEqual(len(history),1)
            self.assertNotIn('run_results',history[0])
            for key in ('event_id','reference_candidate_id','candidate_id','state_results','experiment','unit'):
                self.assertEqual(history[0][key],raw[key])
            self.assertEqual(replayed['tool_observations'][0]['result'], raw)
            self.assertEqual(json.loads(requests[1]['messages'][2]['content']), decision)
            self.assertEqual(json.loads(archive.of('provider_response')[0]['user']), old)
            self.assertTrue(archive.path.read_bytes().startswith(stored))
            self.assertNotIn('If supported', requests[1]['messages'][0]['content'])


if __name__ == '__main__':
    unittest.main()
