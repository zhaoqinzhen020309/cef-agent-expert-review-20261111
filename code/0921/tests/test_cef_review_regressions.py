import math,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from cef.config import validate,load_json
from cef.model import sanitize_identity
from test_cef_engine import config,Fake

class ReviewRegressions(unittest.TestCase):
    def test_job_failure_still_charges_started_worker(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); e.register('seed',e.adapter.seed_source()); e.ledger.reserve('r',1)
            from agent_ahd.core.windows_job import WindowsJob
            with patch.object(WindowsJob,'create',side_effect=RuntimeError('injected Job failure')):
                r=e.execute_case('seed',e.rows[0],'jobfailure','r')
            self.assertTrue(r['started']); self.assertFalse(r['valid']); self.assertEqual(e.ledger.used('case_runs'),1)

    def test_atomic_case_charge(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); e.run()
            self.assertEqual(e.archive.of('execution_start')[0].get('resource'),'case_runs')
            self.assertEqual(e.ledger.used('case_runs'),5)

    def test_profile_replay_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); e.run()
            decision=dict(action='profile',parent_or_target_id='seed',resource_request={'instances':1})
            before=e.ledger.used('case_runs'); e.tool(decision,'profile_test'); e.tool(decision,'profile_test')
            self.assertEqual(e.ledger.used('case_runs'),before+1)
            self.assertTrue(any(r['execution_id']=='profile_test:profile:x' for r in e.archive.of('execution')))

    def test_analysis_no_derived_leak(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); e.run(); e.config['mode']='B2'; observed=[]
            def analyze(code,events,seconds):
                observed.extend(events); return {'elapsed':0,'result':None}
            with patch('cef.analysis.analyze',analyze): e.tool(dict(action='analyze',analysis_code='def analyze(events): return events'),'analysis_test')
            self.assertTrue(observed)
            self.assertNotIn('contrasts',observed[0])

    def test_aggregate_selection(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); source=e.adapter.seed_source()
            for id in ('seed','child'): e.register(id,source)
            for id,value in [('seed',0),('seed',100),('child',75)]: e.archive.add('measurement',dict(candidate_id=id,valid=True,fitness=value))
            self.assertEqual(e.select_eligible({'P':'seed','PA':'child'}),'P')

    def test_selected_old_card_reaches_editor(self):
        with tempfile.TemporaryDirectory() as d:
            e=Engine(config(d),Fake()); e.run(); parent=e.candidates()['seed']
            ev=e.archive.of('contrast')[0]['event_id']
            for i in range(5):
                e.experiences.create('extra'+str(i),dict(source_event_ids=[ev],parent_hash=parent['source_hash'],edit_ids=['A'],affected_functions=[],measured_observation={'gA':i},tested_conditions={}))
            decision=dict(parent_or_target_id='seed',evidence_ids=['extra0'])
            context=e.editor_context('seed',decision,1)
            self.assertIn('extra0',[c['card_id'] for c in context['experiences']])
            self.assertIn(ev,[r['event_id'] for r in context['contrasts']])

    def test_numeric_identity_rejection(self):
        root=Path(__file__).resolve().parents[1]
        c=load_json(root/'configs/cef_v1_3/fssp.acceptance.json')
        for value in (float('nan'),float('inf'),True):
            c['rates']['input_per_million']=value
            with self.assertRaises(ValueError): validate(c)
        with self.assertRaises(ValueError): sanitize_identity('https://')

if __name__=='__main__': unittest.main()
