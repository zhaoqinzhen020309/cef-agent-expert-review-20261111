import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.runner import Runner
from cef.tasks import TaskAdapter

class RunnerTests(unittest.TestCase):
    def test_real_process_and_denial(self):
        r=Runner(1,128)
        a=TaskAdapter('fssp'); instance={'processing_times':[[1,2],[3,4]]}
        out=r.case(a.seed_source(),a,instance,7)
        self.assertTrue(out['valid'],out)
        self.assertEqual(out['objective'],8)
        bad='def solve(instance, time_limit_s, seed):\n    open("secret", "w").write("x")\n    return {"solution": [0,1]}\n'
        self.assertFalse(r.case(bad,a,instance,7)['valid'])

if __name__=='__main__': unittest.main()
