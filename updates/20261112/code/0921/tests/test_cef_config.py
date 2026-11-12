import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.config import validate
from cef.reporting import verify_export
class ConfigTests(unittest.TestCase):
    def test_missing(self):
        with self.assertRaisesRegex(ValueError,'model_snapshot'): validate({})
if __name__=='__main__': unittest.main()
