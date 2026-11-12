"""Approved TSPLIB metric: real data path, fixed references, explicit reports."""
import json
import math
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cef import data
from cef.tasks import TaskAdapter

METRIC = 'tsplib_unrounded_original_reference_v1'


class UnroundedTSPLIBTests(unittest.TestCase):
    def fixture(self, root, scale=1.51):
        raw = root / 'data/raw/tsplib/triangle.tsp'
        raw.parent.mkdir(parents=True)
        raw.write_text('NAME: triangle\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n'
                       f'1 0 0\n2 {scale} 0\n3 0 {scale}\nEOF\n', encoding='utf-8', newline='\n')
        coords = np.array([[0., 0.], [scale, 0.], [0., scale]])
        distances = np.linalg.norm(coords[:, None, :]-coords[None, :, :], axis=2)
        source = root / 'integer.npz'
        np.savez(source, distance_matrix=np.floor(distances+.5))
        return dict(id='triangle', split='test', **data.receipt(source), reference=6.,
                    reference_kind='published optimal tour', source={'kind':'offline fixture'}), raw

    def test_solver_and_scorer_use_fractional_matrix_and_frozen_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, raw = self.fixture(root)
            original = json.loads(json.dumps(row))
            rows = data.prepare_tsplib_unrounded(root, [row], root/'unrounded')
            self.assertEqual(row, original)
            self.assertEqual(rows[0]['reference'], 6.)
            self.assertEqual(rows[0]['evaluation_metric'], METRIC)
            self.assertEqual(rows[0]['source']['coordinates']['sha256'], data.receipt(raw)['sha256'])
            instance = TaskAdapter('tsp').load(rows[0])
            self.assertAlmostEqual(instance['distance_matrix'][0][1], 1.51)
            self.assertEqual(TaskAdapter('tsp').solver_input(instance)['distance_matrix'], instance['distance_matrix'])
            objective, gap = TaskAdapter('tsp').score(instance, [0, 1, 2])
            expected = 1.51*(2+math.sqrt(2))
            self.assertAlmostEqual(objective, expected)
            self.assertAlmostEqual(gap, 100*(expected/6-1))
            self.assertLess(gap, 0)
            self.assertEqual(data.prepare_tsplib_unrounded(root, [row], root/'unrounded'), rows)

    def test_missing_coordinates_fails_without_integer_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, raw = self.fixture(root)
            raw.unlink()
            with self.assertRaises(FileNotFoundError):
                data.prepare_tsplib_unrounded(root, [row], root/'unrounded')

    def test_mismatched_coordinates_and_modified_cache_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, raw = self.fixture(root)
            converted = data.prepare_tsplib_unrounded(root, [row], root/'unrounded')
            np.savez(converted[0]['path'], distance_matrix=np.zeros((3, 3)))
            with self.assertRaisesRegex(ValueError, 'unrounded'):
                data.prepare_tsplib_unrounded(root, [row], root/'unrounded')
            raw.write_text(raw.read_text().replace('2 1.51 0', '2 10 0'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'coordinate'):
                data.prepare_tsplib_unrounded(root, [row], root/'other')

    def test_runcef_default_freezes_new_metric_for_all_29_tests(self):
        from scripts.cef_launcher import build_config
        settings = dict(model='deepseek-flash', base_url='https://api.deepseek.com',
                        seeds=[2025], iterations=1, workers=4)
        with tempfile.TemporaryDirectory() as folder:
            config, test = build_config(2025, Path(folder)/'not_started', settings)
            self.assertFalse((Path(folder)/'not_started').exists())
        self.assertEqual(len(test['instances']), 29)
        self.assertEqual({r.get('evaluation_metric') for r in test['instances']}, {data.TSPLIB_METRIC})
        self.assertEqual(config['reference_manifest']['evaluation_metric'], data.TSPLIB_METRIC)
        self.assertEqual(len(config['data_manifest']['train']), 64)
        eil = next(r for r in test['instances'] if r['id']=='eil51')
        self.assertEqual(eil['reference'], 426.)
        matrix = TaskAdapter('tsp').load(eil)['distance_matrix']
        self.assertTrue(all(x == int(x) for row in matrix for x in row))

    def test_future_formal_test_rejects_legacy_integer_manifest(self):
        legacy=json.loads((ROOT/'data/cef_v1_3_20261026/tsp_manifest.json').read_text())
        with self.assertRaisesRegex(ValueError, 'unlabelled or mixed TSPLIB'):
            data.audit_test('tsp', legacy['test'])

    def test_entry_rejects_legacy_before_loading_frozen_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); search=root/'search'; snapshot=search/'runtime_snapshot/cef'
            snapshot.mkdir(parents=True)
            (snapshot/'__init__.py').write_text('',encoding='utf-8')
            (snapshot/'cli.py').write_text('def main():\n    print("LEGACY_SNAPSHOT_EXECUTED")\n    return 0\n',encoding='utf-8')
            config=root/'test.json'
            row=dict(id='eil51',split='test')
            config.write_text(json.dumps({'instances':[row]}),encoding='utf-8')
            args=[sys.executable,str(ROOT/'scripts/run_cef.py'),'--quiet','test','--run-dir',str(search),
                  '--test-config',str(config),'--output',str(root/'test')]
            for test_args in (['--test-config',str(config)],['--test-config='+str(config)],['--test-c',str(config)]):
                invocation=args[:6]+test_args+args[8:]
                result=subprocess.run(invocation,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0,test_args)
                self.assertIn('unlabelled or mixed TSPLIB',result.stderr)
                self.assertNotIn('LEGACY_SNAPSHOT_EXECUTED',result.stdout)
            row['evaluation_metric']=METRIC
            config.write_text(json.dumps({'instances':[row]}),encoding='utf-8')
            (search/'facts.jsonl').write_text(json.dumps(dict(kind='manifest',data={'task':'tsp','data_manifest':{'test':[dict(id='eil51')]}}))+'\n',encoding='utf-8')
            result=subprocess.run(args,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('unlabelled or mixed TSPLIB',result.stderr)
            self.assertNotIn('LEGACY_SNAPSHOT_EXECUTED',result.stdout)
            (search/'facts.jsonl').write_text(json.dumps(dict(kind='manifest',data={'task':'tsp','data_manifest':{'test':[row]}}))+'\n',encoding='utf-8')
            result=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_final_report_records_metric_and_preserves_negative_gap(self):
        from cef.reporting import final_test
        from cef.store import Archive
        from cef.patch import digest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row, _ = self.fixture(root)
            rows = data.prepare_tsplib_unrounded(root, [row], root/'unrounded')
            archive = Archive(root/'search')
            config = dict(task='tsp', test_repeats=1, data_manifest={'test':rows},
                          diagnostic_run=True, solver_time_limit_s=.5, evaluation_workers=1)
            archive.add('manifest', config)
            source = 'def solve(instance, time_limit_s, seed):\n    return {"solution": [0, 1, 2]}\n'
            result = dict(status='finished', best={'source_hash':digest(source)})
            with patch('cef.reporting.verify_export', return_value=(archive, result, source)), patch('cef.reporting.audit_test'):
                summary = final_test(root/'search', root/'test', rows, 1, {'triangle:0':2025})
            self.assertTrue(summary['valid'])
            self.assertEqual(summary['evaluation_metric'], METRIC)
            self.assertLess(summary['mean_gap_pct'], 0)
            self.assertEqual(summary['mean_unrounded_reference_gap_pct'], summary['mean_gap_pct'])
            record = json.loads((root/'test/results.jsonl').read_text())
            self.assertEqual(record['evaluation_metric'], METRIC)
            self.assertEqual(record['unrounded_reference_gap_pct'], record['gap_pct'])
            self.assertNotIn('reference', TaskAdapter('tsp').solver_input(TaskAdapter('tsp').load(rows[0])))


if __name__ == '__main__':
    unittest.main()
