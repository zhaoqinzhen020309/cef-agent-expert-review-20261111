import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.model import parse_json_content, sanitize_identity
class ModelTests(unittest.TestCase):
    def test_json(self):
        self.assertEqual(parse_json_content('```json\n{"x":1}\n```'),{'x':1})
        with self.assertRaises(ValueError): parse_json_content('{"x":NaN}')
    def test_identity(self):
        with self.assertRaises(ValueError): sanitize_identity('https://user:secret@example.com/v1')
if __name__=='__main__': unittest.main()
