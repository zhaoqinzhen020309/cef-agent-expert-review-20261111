import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.store import Archive, ExperienceStore
from cef.ledger import BudgetLedger, BudgetError

class StateTests(unittest.TestCase):
    def test_reserve_failure_resume(self):
        with tempfile.TemporaryDirectory() as d:
            a=Archive(d); b=BudgetLedger(a,dict(max_case_runs=3,max_llm_requests=2,max_llm_tokens=100,max_analysis_seconds=2,max_wallclock_s=60))
            b.reserve('p',3)
            with self.assertRaises(BudgetError): b.reserve('q',1)
            b.charge('case_runs',1, reservation='p'); b.release('p')
            c=BudgetLedger(Archive(d),b.limits)
            self.assertEqual(c.used('case_runs'),1)
            self.assertEqual(c.remaining('case_runs'),2)
    def test_fact_revision(self):
        with tempfile.TemporaryDirectory() as d:
            a=Archive(d); e=ExperienceStore(a)
            e.create('c',dict(source_event_ids=['ev'],parent_hash='h',edit_ids=['A'],affected_functions=['f'],measured_observation={'gA':1},tested_conditions={'x':1}))
            e.revise('c','explain','本次观察支持',['ev'])
            e.revise('c','counterexample','出现不一致',['ev'])
            card=e.cards()[0]
            self.assertEqual(card['measured_observation'],{'gA':1})
            self.assertEqual(len(card['revision_history']),2)
            self.assertEqual(e.retrieve('h',[],[])[0]['card_id'],'c')

if __name__=='__main__': unittest.main()
