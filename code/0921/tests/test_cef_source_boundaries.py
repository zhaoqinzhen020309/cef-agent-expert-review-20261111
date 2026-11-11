import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.store import Archive
from cef.feedback import changed_definitions
class SourceBoundaryTests(unittest.TestCase):
    def test_source_crlf_hash_is_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            a=Archive(d); s='def solve(instance, time_limit_s, seed):\r\n    return {"solution": [0]}\r\n'
            h=a.source(s)
            self.assertEqual(a.read_source(h),s)
            self.assertEqual(a.source(s),h)
    def test_b3_function_and_global_constant_diff(self):
        old='LIMIT=1\ndef solve(instance,time_limit_s,seed):\n    return {"solution": [0]}\n'
        new=old.replace('LIMIT=1','LIMIT=2')
        self.assertEqual(changed_definitions(old,new),dict(functions=[],constants=['LIMIT']))
        rewritten=new.replace('[0]','[1]')
        self.assertEqual(changed_definitions(old,rewritten),dict(functions=['solve'],constants=['LIMIT']))
if __name__=='__main__': unittest.main()
