"""Real geometry that exhausted the original reference backend."""
import json
from pathlib import Path
import unittest
import tempfile
import numpy as np
from cef.reference import solve_tsp_reference


class ReferenceRegressionTests(unittest.TestCase):
    def test_reference_deadline_checkpoints_and_continues_same_instance(self):
        p=np.random.default_rng(31).random((7,2));d=np.linalg.norm(p[:,None]-p[None,:],axis=2)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'checkpoint.json'
            with self.assertRaises(TimeoutError):solve_tsp_reference(d,seconds=.000001,checkpoint_path=path)
            saved=json.loads(path.read_text())
            self.assertTrue(saved['cuts'])
            result=solve_tsp_reference(d,seconds=15,checkpoint_path=path)
            self.assertLessEqual(result['bound_gap_pct'],.00001+1e-8)
            self.assertGreaterEqual(result['work_slices'],2)
            altered=d.copy();altered[0,1]+=1;altered[1,0]+=1
            with self.assertRaisesRegex(ValueError,'matrix'):
                solve_tsp_reference(altered,seconds=1,checkpoint_path=path)

    def test_logged_lattice_case_keeps_strict_reference_tolerance(self):
        from cef.reference import ReferenceIncomplete
        fixture=json.loads((Path(__file__).parent/'fixtures/reference_lattice_20261102.json').read_text())
        p=np.asarray(fixture['coordinates']);d=np.linalg.norm(p[:,None]-p[None,:],axis=2)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'work.json'
            try:result=solve_tsp_reference(d,seconds=.25,gap_pct=.00001,checkpoint_path=path)
            except ReferenceIncomplete:
                saved=json.loads(path.read_text())
                self.assertEqual(saved['gap_pct'],.00001)
                self.assertEqual(saved['history'][-1]['phase'],'pending')
                self.assertNotEqual(saved.get('status'),'finished')
            else:
                self.assertEqual(sorted(result['solution']),list(range(100)))
                self.assertLessEqual(result['bound_gap_pct'],.00001+1e-8)

    def test_logged_banded_case_keeps_strict_reference_tolerance(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/reference_banded_20261102.json').read_text())['cases'][0]
        p=np.asarray(fixture['coordinates']);d=np.linalg.norm(p[:,None]-p[None,:],axis=2)
        result=solve_tsp_reference(d,seconds=15,gap_pct=.00001)
        tour=result['solution']
        self.assertEqual(sorted(tour),list(range(100)))
        length=float(sum(d[tour[i],tour[(i+1)%100]] for i in range(100)))
        self.assertAlmostEqual(result['objective'],length,places=10)
        self.assertLessEqual(result['bound_gap_pct'],.00001+1e-8)
        self.assertLessEqual(result['lower_bound'],length+1e-7)
