"""Fixed historical solver replication; no real provider or benchmark runs."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class FixedPABTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('scripts.cef_fixed_pab')

    def test_fixed_evaluation_remains_a_separate_helper(self):
        self.assertEqual(self.module().run.__module__,'scripts.cef_fixed_pab')

    def test_check_validates_historical_hash_without_running_or_writing(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'not-created'
            settings=dict(seeds=[2025],workers=4,output_dir=output)
            with patch.object(module.Runner,'case',side_effect=AssertionError('no solver in check')), patch('cef.model.Model.call',side_effect=AssertionError('no model')):
                self.assertEqual(module.main(settings,['--check']),0)
            self.assertFalse(output.exists())
            source,rows=module.load_fixed()
            from cef.patch import digest
            self.assertEqual(digest(source),module.PAB_SHA256)
            self.assertEqual(len(rows),29)

    def test_two_seeds_run_only_58_fixed_cases_and_keep_percent_units(self):
        module=self.module(); calls=[]; lock=threading.Lock()
        source,rows=module.load_fixed()
        def case(source_arg,adapter,instance,seed):
            self.assertEqual(source_arg,source)
            with lock: calls.append(seed)
            return dict(valid=True,objective=instance['reference']*1.01,fitness=1.0,elapsed=0.01,started=True,wallclock_s=0.02,seed=seed)
        with tempfile.TemporaryDirectory() as d:
            settings=dict(seeds=[2025,2026],workers=4,output_dir=Path(d)/'out')
            with patch.object(module.Runner,'case',side_effect=case), patch('cef.model.Model.call',side_effect=AssertionError('no model')):
                self.assertEqual(module.main(settings,[]),0)
            folders=list(settings['output_dir'].iterdir()); self.assertEqual(len(folders),1)
            folder=folders[0]
            summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            records=[json.loads(line) for line in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(calls),58); self.assertEqual(calls.count(2025),29)
            self.assertEqual(len(records),58); self.assertEqual(len({(r['seed'],r['instance_id']) for r in records}),58)
            self.assertEqual(summary['source_hash'],module.PAB_SHA256)
            self.assertEqual(summary['count'],58); self.assertEqual(summary['model_requests'],0)
            self.assertAlmostEqual(summary['mean_gap_pct'],1.0)
            self.assertEqual([r['seed'] for r in summary['per_seed']],[2025,2026])
            self.assertEqual((folder/'solver.py').read_bytes(),module.PAB_SOURCE.read_bytes())

    def test_changed_solver_or_data_is_rejected_before_output(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as d:
            changed=Path(d)/'changed.py'; changed.write_text('changed',encoding='utf-8')
            settings=dict(seeds=[2025],workers=4,output_dir=Path(d)/'out')
            with patch.object(module,'PAB_SOURCE',changed):
                with self.assertRaisesRegex(ValueError,'hash'): module.main(settings,[])
            self.assertFalse(settings['output_dir'].exists())
            source,rows=module.load_fixed(); rows[0]=dict(rows[0],sha256='bad')
            with patch.object(module,'load_fixed',return_value=(source,rows)):
                with self.assertRaises(ValueError): module.main(settings,[])
            self.assertFalse(settings['output_dir'].exists())

    def test_failure_is_recorded_without_retry_or_success_mean(self):
        module=self.module(); source,rows=module.load_fixed()
        def case(source,adapter,instance,seed):
            return dict(valid=False,objective=None,fitness=None,error='fixture failure',elapsed=0,started=False,seed=seed)
        with tempfile.TemporaryDirectory() as d:
            settings=dict(seeds=[2025],workers=4,output_dir=Path(d)/'out')
            with patch.object(module.Runner,'case',side_effect=case) as execute:
                self.assertEqual(module.main(settings,[]),1)
            self.assertEqual(execute.call_count,29)
            summary=json.loads(next(settings['output_dir'].glob('*/summary.json')).read_text(encoding='utf-8'))
            self.assertFalse(summary['valid']); self.assertIsNone(summary['mean_gap_pct'])

    def test_fixed_solver_real_worker_on_tiny_synthetic_instance(self):
        module=self.module(); source,_=module.load_fixed()
        from cef.tasks import TaskAdapter
        instance=dict(distance_matrix=[[0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,1,0]],reference=4)
        result=module.Runner(1,1024).case(source,TaskAdapter('tsp'),instance,2025)
        self.assertTrue(result['valid'],result.get('error'))
        self.assertEqual(result['objective'],4)


if __name__=='__main__': unittest.main()
