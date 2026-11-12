import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine

class Fake:
    """Explicitly offline fixture, never a real model acceptance."""
    def __init__(self): self.inputs=[]; self.n=0
    def call(self,role,payload):
        self.inputs.append((role,payload))
        if role=='planner':
            self.n+=1
            if self.n==3: return dict(action='finish',previous_event_update=[],evidence_ids=[])
            return dict(action='revise',parent_or_target_id='seed',edit_count=1,issue='fixture',proposed_intents=['change ordering'],expected_observations='different order',evidence_ids=[e['event_id'] for e in payload.get('recent_events',[])],previous_event_update=[])
        p=payload['parent_code']; old='key=lambda j:-sum(p[j])'; new='key=lambda j:sum(p[j])'
        return dict(package_id='fixture_'+str(self.n),parent_id=payload['parent_id'],parent_hash=payload['parent_hash'],issue='fixture',evidence_ids=payload['decision']['evidence_ids'],single_edit_reason='one full intent',edits=[dict(edit_id='A',purpose='order',prediction='check',exact_replacements=[dict(old=old,new=new)])])

def config(d):
    return dict(task='fssp',mode='CEF',run_dir=d,max_case_runs=20,max_wallclock_s=60,max_llm_requests=10,max_llm_tokens=10000,max_analysis_seconds=2,solver_time_limit_s=1,memory_mb=256,max_profile_instances=4,max_analysis_per_round=2,analysis_timeout_s=1,
                data_manifest={'train':[dict(id='x',split='train',instance={'processing_times':[[1,2],[3,4]]})]},evaluation_seed_manifest={'x':7})

class EngineTests(unittest.TestCase):
    def test_feedback_history_resume(self):
        with tempfile.TemporaryDirectory() as d:
            model=Fake(); e=Engine(config(d),model); out=e.run()
            self.assertEqual(out['status'],'finished')
            self.assertEqual(len(e.archive.of('contrast')),2)
            self.assertTrue(model.inputs[2][1]['recent_events'])
            self.assertTrue(model.inputs[3][1]['contrasts'])
            self.assertTrue(e.experiences.cards())
            self.assertEqual(e.ledger.used('case_runs'),5)
            restored=Engine(config(d),model)
            self.assertEqual(restored.best()['source_hash'],e.best()['source_hash'])
            self.assertEqual(restored.ledger.used('case_runs'),5)

if __name__=='__main__': unittest.main()
