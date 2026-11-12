import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cef.patch import compose, digest
from cef.feedback import contrast, choose


class ContractTests(unittest.TestCase):
    def test_same_parent(self):
        p = 'def solve(instance, time_limit_s, seed):\n    a = 1\n    b = 2\n    return {"solution": [a,b]}\n'
        package = dict(parent_hash=digest(p), edits=[
            dict(edit_id='A', purpose='a', prediction='x', exact_replacements=[dict(old='a = 1', new='a = 3')]),
            dict(edit_id='B', purpose='b', prediction='y', exact_replacements=[dict(old='b = 2', new='b = 4')])], joint_reason='independent')
        v = compose(p, package)
        self.assertEqual(set(v), {'P','PA','PB','PAB'})
        self.assertIn('a = 1', v['PB'])
        self.assertIn('a = 3', v['PAB'])
        package['parent_hash'] = 'bad'
        with self.assertRaises(ValueError): compose(p, package)

    def test_joint_and_missing(self):
        for scores, winner in [({'P':100,'PA':94,'PB':104,'PAB':102},'PA'),
                               ({'P':100,'PA':102,'PB':101,'PAB':93},'PAB'),
                               ({'P':100,'PA':None,'PB':None,'PAB':93},'PAB'),
                               ({'P':100,'PA':102,'PB':101,'PAB':103},'P')]:
            self.assertEqual(choose(scores), winner)
        c = contrast({'P':100,'PA':None,'PB':None,'PAB':93})
        self.assertEqual(c['gAB'], 7)
        self.assertIsNone(c['I'])
        self.assertEqual(contrast({'P':100,'PA':102,'PB':101,'PAB':93})['I'], 10)

    def test_overlap(self):
        p='def solve(instance,time_limit_s,seed): return {"solution": []}\n'
        e=dict(edit_id='A',purpose='x',prediction='x',exact_replacements=[dict(old='return',new='return')])
        with self.assertRaises(ValueError): compose(p,dict(parent_hash=digest(p),edits=[e,e],joint_reason='x'))

if __name__ == '__main__': unittest.main()
