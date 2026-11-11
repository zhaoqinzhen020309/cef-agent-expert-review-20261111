import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from cef.patch import digest,compose
from cef.store import Archive
from cef.ledger import BudgetLedger,BudgetError
from test_cef_engine import config,Fake

class BoundaryTests(unittest.TestCase):
    def test_actual_dependent_validity(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); p='def solve(instance, time_limit_s, seed):\n    marker=0\n    return {"solution": [0,1]}\n'
            pkg=dict(parent_hash=digest(p),edits=[dict(edit_id='A',purpose='use',prediction='joint only',exact_replacements=[dict(old='marker=0',new='marker=helper()')]),dict(edit_id='B',purpose='define plus constraint',prediction='joint only',exact_replacements=[dict(old='return {"solution": [0,1]}',new='return {"solution": [0,1] if marker==1 else []}\n\ndef helper():\n    return 1')])],joint_reason='semantic dependency intentionally tested')
            v=compose(p,pkg)
            r={k:e.runner.case(s,e.adapter,{'processing_times':[[1,2],[3,4]]},7) for k,s in v.items()}
            self.assertTrue(r['P']['valid']); self.assertFalse(r['PA']['valid']); self.assertFalse(r['PB']['valid']); self.assertTrue(r['PAB']['valid'])

    def test_insufficient_package_no_editor(self):
        with tempfile.TemporaryDirectory() as d:
            c=config(d); c['max_case_runs']=2; model=Fake(); e=Engine(c,model)
            result=e.run()
            self.assertEqual(result['status'],'budget_terminated')
            self.assertEqual([r for r in model.inputs if r[0]=='editor'],[])
            self.assertEqual(result['best']['candidate_id'],'seed')

    def test_mode_visibility(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); e.run()
            for mode in ('CEF','B0','B1','B2','B3','B4'):
                e.config['mode']=mode; state=e.state()
                if mode=='B1': self.assertNotIn('recent_events',state); continue
                self.assertEqual('contrasts' in state['recent_events'][0],mode in ('CEF','B4'))
                if mode in ('B0','B2','B4'): self.assertEqual(state['retrieved_cards'],[])

    def test_historical_parent_not_previous_only(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); e.run()
            self.assertTrue(e.candidates()['seed']['eligibility'])
            self.assertTrue(e.candidates()['step_0:PA']['eligibility'])
            self.assertEqual(e.state('seed')['active_parent']['candidate_id'],'seed')

    def test_hold_survives_interruption(self):
        with tempfile.TemporaryDirectory() as d:
            a=Archive(d); b=BudgetLedger(a,dict(max_case_runs=3,max_llm_requests=3,max_llm_tokens=100,max_analysis_seconds=2,max_wallclock_s=60))
            b.hold('llm_tokens',80,'x')
            self.assertEqual(BudgetLedger(Archive(d),b.limits).remaining('llm_tokens'),20)
            b.charge('llm_tokens',50,hold_id='x')
            self.assertEqual(b.used('llm_tokens'),50)

    def test_torn_tail_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            a=Archive(d); a.add('x',{'v':1})
            with a.path.open('ab') as f: f.write(b'{"seq":1')
            b=Archive(d); self.assertEqual(b.of('x'),[{'v':1}])
            self.assertTrue(list(Path(d).glob('torn_tail*')))

if __name__=='__main__': unittest.main()
