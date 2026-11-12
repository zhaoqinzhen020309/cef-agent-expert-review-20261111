"""Reference work is retriable computation, never an unverified training score."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class ReferenceJobTests(unittest.TestCase):
    def test_zero_cost_nonnegative_problem_finishes_with_verified_zero_bound(self):
        from cef.reference_jobs import independent_reference
        from cef.reference import solve_tsp_reference
        self.assertEqual(solve_tsp_reference(np.zeros((4,4)),seconds=.01)['objective'],0.)
        result=independent_reference(np.zeros((4,4)),1,.00001)
        self.assertEqual(result['objective'],0.)
        self.assertEqual(result['lower_bound'],0.)
        self.assertEqual(result['bound_gap_pct'],0.)

    def test_default_reference_is_one_unlimited_native_search(self):
        from cef.reference_jobs import independent_reference
        with tempfile.TemporaryDirectory() as td,patch('cef.reference_jobs.run_reference_slice',return_value=dict(valid=True,
                result=dict(solution=[0,1,2],objective=3.,lower_bound=3.,bound_gap_pct=0.))) as work:
            independent_reference(np.ones((3,3))-np.eye(3),120,.00001,Path(td)/'case.json')
            self.assertEqual(work.call_count,1)
            self.assertIsNone(work.call_args.args[1])

    def test_old_restart_budgets_do_not_limit_new_native_search(self):
        from cef.reference_jobs import independent_reference
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'case.json';journal=p.with_suffix('.attempts.jsonl')
            journal.write_text('{"kind":"start","slice_seconds":120}\n{"kind":"pending"}\n')
            with patch('cef.reference_jobs.run_reference_slice',return_value=dict(valid=True,result=dict(
                    solution=[0,1,2],objective=3.,lower_bound=3.,bound_gap_pct=0.))) as work:
                independent_reference(np.ones((3,3))-np.eye(3),120,.00001,p)
            self.assertIsNone(work.call_args.args[1])

    def test_interrupted_attempt_log_preserves_tail_and_resumes_committed_budget(self):
        from cef.reference_jobs import independent_reference
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'case.json';journal=p.with_suffix('.attempts.jsonl')
            journal.write_text('{"kind":"start","slice_seconds":120}\n{"kind":"pend')
            raw=journal.read_bytes()
            with patch('cef.reference_jobs.run_reference_slice',return_value=dict(valid=True,result=dict(
                    solution=[0,1,2],objective=3.,lower_bound=3.,bound_gap_pct=0.))) as work:
                independent_reference(np.ones((3,3))-np.eye(3),120,.00001,p)
            self.assertIsNone(work.call_args.args[1])
            for line in journal.read_text().splitlines():json.loads(line)
            import hashlib
            frozen=Path(__file__).resolve().parents[1]/'experiments/reference_snapshots'/(hashlib.sha256(raw).hexdigest()+'.jsonl')
            self.assertEqual(frozen.read_bytes(),raw)

    def test_reference_evidence_names_work_in_long_windows_run_directories(self):
        from cef.reference_jobs import independent_reference
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);folder=root/('x'*(185-len(str(root))-1));folder.mkdir()
            with patch('cef.reference_jobs.run_reference_slice',return_value=dict(valid=True,result=dict(
                    solution=[0,1,2],objective=3.,lower_bound=3.,bound_gap_pct=0.))):
                result=independent_reference(np.ones((3,3))-np.eye(3),1,.00001,folder/'case.json')
            artifact=Path(result['reference_work']['attempts']['path'])
            self.assertLess(len(str(artifact)),260)
            self.assertTrue(artifact.is_file())

    def test_pending_slice_continues_same_checkpoint_and_tolerance(self):
        from cef.reference_jobs import independent_reference
        matrix=np.ones((3,3))-np.eye(3);seen=[]
        def slice(matrix,seconds,gap,path):
            seen.append((seconds,gap,str(path)))
            if len(seen)==1:
                Path(path).write_text('{"saved_cuts": [[0,1]]}')
                return dict(valid=False,pending=True,error='bounded slice finished')
            self.assertTrue(Path(path).exists())
            return dict(valid=True,result=dict(solution=[0,1,2],objective=3.,lower_bound=3.,bound_gap_pct=0.))
        with tempfile.TemporaryDirectory() as td,patch('cef.reference_jobs.run_reference_slice',side_effect=slice):
            result=independent_reference(matrix,1,.00001,Path(td)/'case.json',max_slices=2)
        self.assertEqual(result['objective'],3.)
        self.assertEqual(len(seen),2)
        self.assertEqual([r[0] for r in seen],[1,1])
        self.assertEqual(seen[0][1:],seen[1][1:])

    def test_exhausted_slices_stay_pending_without_fabricating_reference(self):
        from cef.reference_jobs import independent_reference,ReferencePending
        with tempfile.TemporaryDirectory() as td,patch('cef.reference_jobs.run_reference_slice',return_value=dict(
                valid=False,pending=True,error='not yet bounded')) as work:
            with self.assertRaises(ReferencePending):independent_reference(np.ones((3,3))-np.eye(3),1,.00001,Path(td)/'case.json',max_slices=2)
            self.assertEqual(work.call_count,2)
            journal=Path(td)/'case.attempts.jsonl'
            self.assertTrue(journal.exists())
            self.assertIn('pending',journal.read_text())

    def test_backend_error_is_not_automatically_retried(self):
        from cef.reference_jobs import independent_reference
        with tempfile.TemporaryDirectory() as td,patch('cef.reference_jobs.run_reference_slice',return_value=dict(
                valid=False,pending=False,error='invalid degree constraints')) as work:
            with self.assertRaisesRegex(RuntimeError,'invalid degree'):independent_reference(np.ones((3,3))-np.eye(3),1,.00001,Path(td)/'case.json')
            self.assertEqual(work.call_count,1)

    def test_uncertified_tour_cannot_be_promoted_to_reference(self):
        from cef.reference_jobs import independent_reference
        with tempfile.TemporaryDirectory() as td,patch('cef.reference_jobs.run_reference_slice',return_value=dict(
                valid=True,result=dict(solution=[0,1,2],objective=3.,lower_bound=2.,bound_gap_pct=50.))):
            with self.assertRaisesRegex(ValueError,'bound'):independent_reference(np.ones((3,3))-np.eye(3),1,.00001,Path(td)/'case.json')
