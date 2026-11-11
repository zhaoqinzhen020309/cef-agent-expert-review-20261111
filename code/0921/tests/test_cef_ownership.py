import os,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.cli import claim_owner
class OwnershipTests(unittest.TestCase):
    def test_active_owner_not_stolen(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'owner.lock').write_text(str(os.getpid()))
            with self.assertRaises(FileExistsError): claim_owner(p)
    def test_dead_owner_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'owner.lock').write_text('99999999')
            lock=claim_owner(p)
            self.assertEqual(lock.read_text(),str(os.getpid()))
            self.assertEqual(len(list(p.glob('owner.abandoned.*.lock'))),1)
if __name__=='__main__': unittest.main()
