"""Supported NumPy operations must work inside the candidate worker."""
from pathlib import Path
import json
import unittest
from cef.instance_design import generate_coordinates


class NumpyImportTests(unittest.TestCase):
    def test_ndarray_reductions_work_without_opening_candidate_imports(self):
        source='''import numpy as np
def generate(n_cities, seed, parameters):
    a=np.arange(n_cities*2,dtype=float).reshape(n_cities,2)
    assert float(a.max())==n_cities*2-1
    assert float(a.mean())==(n_cities*2-1)/2
    assert float(a.std())>0 and float(a.var())>0
    return (a/(a.max()+1)).tolist()
'''
        self.assertEqual(len(generate_coordinates(source,{},10,2025)),10)

    def test_all_actual_logged_generator_seeds_run(self):
        folder=Path(__file__).parent/'fixtures/crash_20261104'
        source=(folder/'generator.py').read_text();settings=json.loads((folder/'seeds.json').read_text())
        params=json.loads((folder/'parameters.json').read_text())
        for index in (3,9,22,47,50,53,54,59):
            with self.subTest(index=index):
                self.assertEqual(len(generate_coordinates(source,params,100,settings['seeds'][index])),100)

    def test_direct_private_and_unapproved_imports_remain_rejected(self):
        for name in ('os','numpy.core._methods'):
            with self.subTest(name=name),self.assertRaises(ValueError):
                generate_coordinates('import '+name+'\ndef generate(n_cities,seed,parameters):\n return []',{},10,1)
