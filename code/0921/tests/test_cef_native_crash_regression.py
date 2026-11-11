"""Real failing generated data and the high-precision backend contract."""
from pathlib import Path
import json
import unittest
import numpy as np
from cef.instance_design import generate_coordinates


class NativeCrashRegressionTests(unittest.TestCase):
    def test_backend_uses_wide_cost_and_proof_arithmetic(self):
        from cef.reference_native import backend_manifest
        manifest=backend_manifest()
        self.assertEqual(manifest['gain_bits'],64)
        self.assertEqual(manifest['proof_bits'],128)
        self.assertEqual(manifest['architecture'],'linux-x86_64')

    def test_actual_access_violation_and_proof_overflow_instances(self):
        from cef.reference_native import solve_tsp_reference
        from cef.reference_jobs import validate_reference
        import tempfile
        folder=Path(__file__).parent/'fixtures/crash_20261104'
        source=(folder/'generator.py').read_text();settings=json.loads((folder/'seeds.json').read_text());params=json.loads((folder/'parameters.json').read_text())
        for index in (1,18):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as td:
                xy=np.asarray(generate_coordinates(source,params,100,settings['seeds'][index]));d=np.linalg.norm(xy[:,None]-xy[None,:],axis=2)
                result=solve_tsp_reference(d,seconds=60,checkpoint_path=Path(td)/'replay.json',console=False)
                validate_reference(d,result,.00001)
                self.assertLessEqual(result['bound_gap_pct'],.00001+1e-8)
