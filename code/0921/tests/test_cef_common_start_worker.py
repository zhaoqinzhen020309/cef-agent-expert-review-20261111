"""Real isolated worker interface checks; no models or benchmark claims."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cef import patch as source_patch
from cef.runner import Runner
from cef.tasks import TaskAdapter


INSTANCE = {'distance_matrix': [[0, 1, 5, 1], [1, 0, 1, 5],
                                [5, 1, 0, 1], [1, 5, 1, 0]], 'reference': 4}
SOLVER = '''import time
def solve(instance, time_limit_s, seed, *, initial_solution=None, observer=None):
    route = list(initial_solution) if initial_solution is not None else [0, 2, 1, 3]
    if observer is not None:
        observer(route)
    # This same continuation runs after either initialization path.
    route[:] = [0, 1, 2, 3]
    if observer is not None:
        observer(route)
    return {"solution": route}
'''


class CommonStartWorkerTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TaskAdapter('tsp')
        self.runner = Runner(1, 256)

    def research(self, source=SOLVER, initial=None, **extra):
        request = dict(kind='common_start', local_limit_s=0.3,
                       initial_solution=[0, 2, 1, 3] if initial is None else initial)
        request.update(extra)
        return self.runner.case(source, self.adapter, INSTANCE, 7,
                                research_request=request)

    def test_signature_requires_explicit_optional_keyword_parameters(self):
        helper = getattr(source_patch, 'research_capable', None)
        self.assertTrue(callable(helper), 'research_capable helper missing')
        self.assertTrue(helper(SOLVER))
        old = 'def solve(instance, time_limit_s, seed):\n    return {"solution": [0,1,2,3]}\n'
        source_patch.check_source(old)
        self.assertFalse(helper(old))
        bad_headers = [
            'initial_solution=None, observer=None',
            '*, initial_solution, observer=None',
            '*, initial_solution=[], observer=None',
            '**kwargs',
        ]
        for header in bad_headers:
            text = 'def solve(instance, time_limit_s, seed, '+header+'):\n    pass\n'
            with self.subTest(header=header), self.assertRaises(ValueError):
                source_patch.check_source(text, require_research=True)

    def test_two_given_starts_observed_and_same_continuation_used(self):
        for initial in ([0, 2, 1, 3], [3, 2, 1, 0]):
            with self.subTest(initial=initial):
                before = list(initial)
                out = self.research(initial=initial)
                self.assertTrue(out['valid'], out)
                self.assertEqual(out['trace'][0]['solution'], before)
                self.assertTrue(out['initial_solution_verified'])
                self.assertEqual(out['solution'], [0, 1, 2, 3])
                self.assertEqual(initial, before)
                self.assertEqual(out['research_kind'], 'common_start')
                self.assertGreaterEqual(out['cpu_seconds'], 0)
        trace = self.runner.case(SOLVER, self.adapter, INSTANCE, 7,
                                 research_request={'kind': 'trace'})
        self.assertTrue(trace['valid'], trace)
        self.assertEqual(trace['trace'][0]['solution'], [0, 2, 1, 3])
        self.assertEqual(trace['trace'][0]['objective'], 12)
        self.assertEqual(trace['trace'][-1]['objective'], 4)
        self.assertGreaterEqual(trace['first_retained_improvement_s'], 0)
        self.assertNotIn('first_improvement_s', trace)
        ordinary = self.runner.case(SOLVER, self.adapter, INSTANCE, 7)
        self.assertTrue(ordinary['valid'], ordinary)
        self.assertNotIn('trace', ordinary)

    def test_ignored_or_unobserved_start_is_protocol_failure(self):
        ignored = SOLVER.replace('list(initial_solution) if initial_solution is not None else ', '')
        silent = SOLVER.replace('observer(route)', 'pass')
        for source in (ignored, silent):
            out = self.research(source, initial=[3, 2, 1, 0])
            self.assertFalse(out['valid'], out)
            self.assertFalse(out['initial_solution_verified'])
            self.assertIn('initial', out['error'])

    def test_trace_requires_solver_observation_before_trusted_final(self):
        silent = SOLVER.replace('observer(route)', 'pass')
        out = self.runner.case(silent, self.adapter, INSTANCE, 7,
                               research_request={'kind': 'trace'})
        self.assertFalse(out['valid'], out)
        self.assertIn('observation', out['error'])
        self.assertEqual(out['trace'], [])

    def test_snapshot_survives_candidate_in_place_mutation(self):
        out = self.research()
        self.assertTrue(out['valid'], out)
        self.assertEqual(out['trace'][0]['solution'], [0, 2, 1, 3])
        self.assertEqual(out['trace'][-1]['solution'], [0, 1, 2, 3])

    def test_caught_invalid_observation_still_fails_closed(self):
        source = SOLVER.replace('    # This same continuation', '''    try:
        observer([0, 0, 1, 2])
    except Exception:
        pass
    # This same continuation''')
        out = self.research(source)
        self.assertFalse(out['valid'], out)
        self.assertIn('observation', out['error'])

    def test_candidate_cannot_force_trusted_final_capture(self):
        source = SOLVER.replace('        observer(route)', '        observer(route, final=True)')
        out = self.research(source)
        self.assertFalse(out['valid'], out)
        self.assertIn('TypeError', out['error'])

    def test_trace_is_bounded_and_retains_final(self):
        source = SOLVER.replace('    # This same continuation', '''    until = time.monotonic() + time_limit_s * 0.8
    while time.monotonic() < until:
        observer(route)
    # This same continuation''')
        out = self.research(source)
        self.assertTrue(out['valid'], out)
        self.assertLessEqual(len(out['trace']), 6)
        self.assertGreaterEqual(len(out['trace']), 3)
        self.assertEqual(out['trace'][-1]['solution'], out['solution'])
        self.assertEqual(sorted(x['elapsed_s'] for x in out['trace']),
                         [x['elapsed_s'] for x in out['trace']])

    def test_external_objective_and_score_units_unchanged(self):
        self.assertEqual(self.adapter.objective(self.adapter.solver_input(INSTANCE), [0, 1, 2, 3]), 4)
        self.assertEqual(self.adapter.objective(INSTANCE, [0, 2, 1, 3]), 12)
        self.assertEqual(self.adapter.score(INSTANCE, [0, 2, 1, 3]), (12., 200.))
        flow = TaskAdapter('fssp')
        data = {'processing_times': [[1, 5], [4, 1]]}
        self.assertEqual(flow.objective(data, [0, 1]), 7.)
        self.assertEqual(flow.objective(data, [1, 0]), 10.)
        self.assertEqual(flow.score(data, [1, 0]), (10., 10.))
        for route in ([0, 0, 1, 2], [False, 1, 2, 3], [0, 1, 2, float('nan')]):
            with self.assertRaises(ValueError):
                self.adapter.objective(INSTANCE, route)

    def test_request_rejected_before_start_for_bad_start_limit_or_kind(self):
        for update in ({'initial_solution': [0, 0, 1, 2]},
                       {'local_limit_s': 0}, {'local_limit_s': 2},
                       {'local_limit_s': float('nan')}, {'kind': 'unknown'}):
            out = self.research(**update)
            self.assertFalse(out['valid'], out)
            self.assertFalse(out['started'], out)

    def test_concurrent_short_limit_does_not_change_normal_limit(self):
        runner = Runner(0.6, 256)
        source = '''import time
def solve(instance, time_limit_s, seed, *, initial_solution=None, observer=None):
    route = list(initial_solution) if initial_solution is not None else [0, 2, 1, 3]
    if observer is not None:
        observer(route)
    time.sleep(0.14)
    return {"solution": route}
'''
        barrier = threading.Barrier(2, timeout=10)
        def run(request):
            return runner.case(source, self.adapter, INSTANCE, 7,
                               on_started=barrier.wait, research_request=request)
        with ThreadPoolExecutor(max_workers=2) as pool:
            short = pool.submit(run, {'kind': 'common_start', 'local_limit_s': 0.08,
                                     'initial_solution': [0, 1, 2, 3]})
            normal = pool.submit(run, None)
            short, normal = short.result(), normal.result()
        self.assertFalse(short['valid'], short)
        self.assertIn('deadline', short['error'])
        self.assertTrue(normal['valid'], normal)
        self.assertEqual(normal['solution'], [0, 2, 1, 3])
        self.assertEqual(runner.limit, 0.6)
        self.assertTrue(runner.case(source, self.adapter, INSTANCE, 7)['valid'])

    def test_old_three_argument_solver_remains_usable(self):
        source = 'def solve(instance, time_limit_s, seed):\n    return {"solution": [0,1,2,3]}\n'
        out = self.runner.case(source, self.adapter, INSTANCE, 7)
        self.assertTrue(out['valid'], out)
        failed = self.runner.case(source, self.adapter, INSTANCE, 7,
                                  research_request={'kind': 'trace'})
        self.assertFalse(failed['valid'])
        self.assertFalse(failed['started'])


if __name__ == '__main__':
    unittest.main()
