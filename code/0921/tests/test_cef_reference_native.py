"""Native TSP references must certify the original floating objective."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class NativeReferenceTests(unittest.TestCase):
    def test_cancellation_and_interrupt_stop_the_native_worker(self):
        from cef.reference_native import solve_tsp_reference
        from concurrent.futures import CancelledError
        from unittest.mock import Mock
        import subprocess
        for cancelled in (True,False):
            process=Mock(returncode=0)
            process.wait.side_effect=[0] if cancelled else [KeyboardInterrupt(),0]
            d=np.ones((10,10))-np.eye(10)
            def cancel(p,request,work):p.kill();p.wait()
            with tempfile.TemporaryDirectory() as td,patch('cef.reference_native.cancel_native',side_effect=cancel),patch('cef.reference_native.subprocess.Popen',return_value=process),patch(
                    'cef.parallel.cancellation_requested',return_value=cancelled):
                with self.assertRaises(CancelledError if cancelled else KeyboardInterrupt):
                    solve_tsp_reference(d,checkpoint_path=Path(td)/'cancel.json',console=False)
            process.kill.assert_called_once()

    def test_diagnostic_deadline_preserves_unfinished_work_without_score(self):
        from cef.reference_native import solve_tsp_reference
        from cef.reference import ReferenceIncomplete
        from unittest.mock import Mock
        process=Mock(returncode=0)
        def cancel(p,request,work):p.kill();p.wait()
        with tempfile.TemporaryDirectory() as td,patch('cef.reference_native.time.monotonic',side_effect=[0.,0.,1.,1.]),patch('cef.reference_native.cancel_native',side_effect=cancel),patch('cef.reference_native.subprocess.Popen',return_value=process):
            with self.assertRaises(ReferenceIncomplete):
                solve_tsp_reference(np.ones((10,10))-np.eye(10),seconds=1e-9,checkpoint_path=Path(td)/'pending.json',console=False)
        process.kill.assert_called_once()

    def test_backend_available_and_frozen(self):
        from cef.reference_native import backend_manifest
        manifest=backend_manifest()
        self.assertEqual(manifest['backend'],'Concorde Linux 64-bit')
        self.assertTrue(manifest['files'])
        self.assertTrue(all(Path(row['path']).is_file() for row in manifest['files']))

    def test_precision_bound_contains_every_original_tour(self):
        from cef.reference_native import prepare_problem
        xy=np.random.default_rng(29).random((7,2));d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
        p=prepare_problem(d);c=np.asarray(p['integer_matrix']);scale=p['scale'];epsilon=p['edge_error_bound']
        for perm in itertools.permutations(range(1,7)):
            t=(0,)+perm
            original=sum(d[t[i],t[(i+1)%7]] for i in range(7))
            represented=sum(int(c[t[i],t[(i+1)%7]]) for i in range(7))/scale
            self.assertLessEqual(abs(original-represented),7*epsilon+1e-13)
        self.assertLessEqual(c.max(),2**29)

    def test_nearly_symmetric_input_has_one_symmetric_integer_representation(self):
        from cef.reference_native import prepare_problem
        d=np.ones((4,4))-np.eye(4)
        d[0,1]=.100000005-1e-13;d[1,0]=.100000005+1e-13
        p=prepare_problem(d);c=np.asarray(p['integer_matrix'])
        np.testing.assert_array_equal(c,c.T)
        self.assertLessEqual(float(np.max(abs(d-c/p['scale']))),p['edge_error_bound'])

    def test_false_or_incomplete_native_success_cannot_be_reference(self):
        from cef.reference_native import certify
        d=np.ones((4,4))-np.eye(4)
        for record,log in ((dict(returncode=0,solution=[0,1,2,3]),''),
                (dict(returncode=1,solution=[0,1,2,3]),'Optimal Solution: 4.00')):
            with self.assertRaises(ValueError):certify(d,record,log,.00001)

    def test_real_small_euclidean_and_nonmetric_match_exhaustive_optimum(self):
        from cef.reference_native import solve_tsp_reference
        from cef.reference_jobs import validate_reference
        from functools import lru_cache
        for seed in range(2):
            rng=np.random.default_rng(seed)
            if seed%2:
                d=rng.uniform(.5,1.,(10,10));d=(d+d.T)/2;np.fill_diagonal(d,0)
            else:
                p=rng.random((10,2));d=np.linalg.norm(p[:,None]-p[None,:],axis=2)
            @lru_cache(None)
            def exact(last,remaining):
                if not remaining:return float(d[last,0])
                return min(float(d[last,j])+exact(j,remaining^(1<<j)) for j in range(1,10) if remaining&(1<<j))
            optimum=exact(0,(1<<10)-2)
            with tempfile.TemporaryDirectory() as td:
                r=solve_tsp_reference(d,seconds=20,checkpoint_path=Path(td)/'case.json',console=False)
                validate_reference(d,r,.00001)
                self.assertAlmostEqual(r['objective'],optimum,places=7)
                self.assertLessEqual(r['lower_bound'],optimum+1e-10)
                # Completed reference reuse must not launch a new native solve.
                with patch('cef.reference_native.subprocess.Popen',side_effect=AssertionError('no new worker')):
                    self.assertEqual(solve_tsp_reference(d,checkpoint_path=Path(td)/'case.json')['objective'],r['objective'])

    def test_zero_and_invalid_input(self):
        from cef.reference_native import solve_tsp_reference,prepare_problem
        self.assertEqual(solve_tsp_reference(np.zeros((4,4)))['objective'],0.)
        with self.assertRaises(ValueError):prepare_problem(np.ones((4,4)))

    def test_small_positive_coordinates_refine_precision_and_remain_json_serializable(self):
        from cef.reference_native import solve_tsp_reference
        from cef.reference_jobs import validate_reference
        xy=np.random.default_rng(12).random((10,2))*1e-10
        d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
        with tempfile.TemporaryDirectory() as td:
            r=solve_tsp_reference(d,seconds=20,checkpoint_path=Path(td)/'tiny.json',console=False)
            validate_reference(d,r,.00001)
            json.dumps(r,allow_nan=False)

