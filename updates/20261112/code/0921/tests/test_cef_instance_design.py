"""Behavioral acceptance for the separate, feedback-driven data designer."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cef.reference import solve_tsp_reference
from cef.instance_design import DataDesigner, generate_coordinates, design_seeds


GENERATOR = '''import numpy as np
def generate(n_cities, seed, parameters):
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2)).tolist()
'''
SOLVER = '''def solve(instance, time_limit_s, seed):
    return {"solution": list(range(len(instance["distance_matrix"])))}
'''


class DesignerFake:
    def __init__(self): self.inputs = []
    def call(self, role, payload):
        self.inputs.append((role, payload))
        source=GENERATOR
        if payload['previous_results'] and payload['previous_results'][-1]['mean_reference_gap_pct']>0:
            source=GENERATOR.replace('rng.random((n_cities, 2))','(rng.random((n_cities, 2)) ** 2)')
        return dict(generator_source=source, parameters={}, reason='measured data design')


class DataDesignTests(unittest.TestCase):
    def test_calibration_can_match_algorithm_observer_conditions(self):
        source='''def solve(instance, time_limit_s, seed, *, initial_solution=None, observer=None):
    if observer is None: raise ValueError("observer was not passed")
    solution=list(range(len(instance['distance_matrix'])))
    observer(solution)
    return {'solution':solution}
'''
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'solver.py';p.write_text(source)
            c=dict(output_dir=str(root/'design'),solver_source=str(p),iterations=1,data_seed=42,
                solver_seed=2025,instances=1,n_cities=7,workers=1,solver_seconds=1,
                generator_seconds=3,reference_seconds=15,reference_gap_pct=.001,diagnostic=True,
                calibration_research_trace=True)
            self.assertEqual(DataDesigner(c,DesignerFake()).run()['status'],'finished')

    def test_check_rejects_bad_seed_or_time_without_creating_output(self):
        from scripts.prepare_cef_training import main
        from runDataAgent import CONFIG
        with tempfile.TemporaryDirectory() as td:
            for changes in ({'solver_seed':-1},{'reference_seconds':-1},{'reference_gap_pct':float('nan')}):
                c=dict(CONFIG,output_dir=Path(td)/'not_created',**changes)
                with self.assertRaises(ValueError):main(c,['--check'])
                self.assertFalse(Path(c['output_dir']).exists())

    def test_sdk_has_separate_data_role_and_replays_its_measurements(self):
        import os
        import httpx
        from unittest.mock import patch
        from cef.model import Model
        from cef.store import Archive
        from cef.ledger import BudgetLedger
        from test_cef_sdk import sdk_transport, completion, config
        requests = []
        def handler(request):
            body=json.loads(request.content); requests.append(body)
            return httpx.Response(200,json=completion(json.dumps(dict(generator_source=GENERATOR,parameters={},reason='fixture'))))
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'CEF_TEST_KEY':'offline'}):
            archive=Archive(td)
            ledger=BudgetLedger(archive,{},enforce_limits=False)
            model=Model(config(stop_rule='iterations'),archive,ledger)
            with sdk_transport(handler):
                model.call('data_designer',dict(iteration=0,previous_results=[]))
                model.call('data_designer',dict(iteration=1,previous_results=[dict(mean_reference_gap_pct=.12)]))
        self.assertIn('instance generator',requests[0]['messages'][0]['content'])
        self.assertNotIn('exact_replacements',requests[0]['messages'][0]['content'])
        self.assertEqual([r['role'] for r in requests[1]['messages']],['system','user','assistant','user'])
        self.assertEqual(json.loads(requests[1]['messages'][-1]['content'])['previous_results'][0]['mean_reference_gap_pct'],.12)

    def test_reference_matches_exhaustive_continuous_objective(self):
        p = np.random.default_rng(9).random((7, 2))
        d = np.linalg.norm(p[:, None] - p[None, :], axis=2)
        expected = min(sum(d[t[i], t[(i+1) % 7]] for i in range(7))
                       for tail in itertools.permutations(range(1, 7)) for t in [(0,)+tail])
        r = solve_tsp_reference(d, seconds=15)
        self.assertAlmostEqual(r['objective'], expected, places=8)
        self.assertLessEqual(r['lower_bound'], expected+1e-7)
        self.assertEqual(sorted(r['solution']), list(range(7)))
        self.assertNotIn('Concorde', r['reference_kind'])

    def test_generation_is_seeded_and_uses_restricted_worker(self):
        a = generate_coordinates(GENERATOR, {}, 6, 7, seconds=3)
        b = generate_coordinates(GENERATOR, {}, 6, 7, seconds=3)
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            generate_coordinates('import os\n'+GENERATOR, {}, 6, 7, seconds=3)
        with self.assertRaises(ValueError):
            generate_coordinates(GENERATOR.replace('rng.random((n_cities, 2))',
                                                  'np.zeros((n_cities, 2))'), {}, 6, 7, seconds=3)

    def test_data_agent_receives_actual_previous_measurements(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); solver = root/'solver.py'; solver.write_text(SOLVER)
            c = dict(output_dir=str(root/'design'), solver_source=str(solver),
                     iterations=2, data_seed=42, solver_seed=2025, instances=3,
                     n_cities=7, workers=1, solver_seconds=1, generator_seconds=3,
                     reference_seconds=15, reference_gap_pct=0.001, diagnostic=True)
            model = DesignerFake(); designer = DataDesigner(c, model)
            result = designer.run()
            self.assertEqual(result['status'], 'finished')
            self.assertEqual(len(model.inputs), 2)
            self.assertEqual(model.inputs[1][0], 'data_designer')
            observation = model.inputs[1][1]['previous_results'][-1]
            self.assertTrue(observation['valid'])
            self.assertEqual(len(observation['per_instance']), 3)
            self.assertIn('mean_reference_gap_pct', observation)
            self.assertEqual(observation['solver_seed'], 2025)
            self.assertEqual(result['instance_seeds'], design_seeds(42, 3))
            manifest = json.loads(Path(result['supplement_manifest']).read_text())
            self.assertEqual(len(manifest['train']), 3)
            self.assertTrue(manifest['diagnostic'])
            self.assertNotIn('test', manifest)
            first=np.load(root/'design/iteration_000/supplement_0000.npz')['coordinates']
            second=np.load(root/'design/iteration_001/supplement_0000.npz')['coordinates']
            self.assertFalse(np.array_equal(first,second))
            self.assertNotEqual((root/'design/iteration_000/generator.py').read_text(),
                                (root/'design/iteration_001/generator.py').read_text())
            with self.assertRaises(FileExistsError): DataDesigner(c, DesignerFake())

    def test_invalid_generator_is_recorded_and_never_exported(self):
        class Invalid:
            def call(self,role,payload):
                return dict(generator_source='def generate(n_cities,seed,parameters):\n    return [[0,0]] * n_cities',parameters={})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'solver.py';source.write_text(SOLVER)
            c=dict(output_dir=str(root/'design'),solver_source=str(source),iterations=1,
                   data_seed=42,solver_seed=2025,instances=1,n_cities=7,workers=1,
                   solver_seconds=1,generator_seconds=2,reference_seconds=5,reference_gap_pct=.001,diagnostic=True)
            designer=DataDesigner(c,Invalid())
            with self.assertRaisesRegex(RuntimeError,'no valid measured generator'): designer.run()
            self.assertFalse((root/'design/supplement_manifest.json').exists())
            self.assertFalse(designer.archive.of('data_case')[0]['valid'])


if __name__ == '__main__': unittest.main()
