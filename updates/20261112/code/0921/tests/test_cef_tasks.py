import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.tasks import TaskAdapter
class TaskTests(unittest.TestCase):
    def test_objectives(self):
        f=TaskAdapter('fssp')
        self.assertEqual(f.score({'processing_times':[[1,2],[3,4]]},[0,1])[0],8)
        t=TaskAdapter('tsp')
        raw,fitness=t.score({'distance_matrix':[[0,1,1],[1,0,1],[1,1,0]],'reference':3},[0,1,2])
        self.assertEqual((raw,fitness),(3,0))
        with self.assertRaises(ValueError): f.score({'processing_times':[[1],[2]]},[0,0])
if __name__=='__main__': unittest.main()
