"""Failure-path controller tests; fake model only, fresh real workers, no API calls."""
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from test_cef_engine import config
from test_cef_planner_protocol import ReplySequence

BAD='def solve(instance, time_limit_s, seed):\n    items=[0,1]\n    items.__setitem__(0,1)\n    return {"solution":items}\n'
GOOD='def solve(instance, time_limit_s, seed):\n    return {"solution":[0,1]}\n'


def prepared(folder):
    engine=Engine(config(folder),None)
    engine.register('seed',engine.adapter.seed_source())
    engine.archive.add('measurement',dict(candidate_id='seed',valid=True,fitness=8,measurement_id='seed'))
    engine.register('failed',BAD,'seed')
    engine.archive.add('measurement',dict(candidate_id='failed',valid=False,fitness=None,measurement_id='failed'))
    engine.archive.add('execution',dict(execution_id='old_failure',candidate_id='failed',instance_id='x',valid=False,
        error='ValueError: dunder introspection denied',error_location=[dict(lineno=3,source_line='items.__setitem__(0,1)')]))
    return engine


class ActionRecoveryTests(unittest.TestCase):
    def test_failed_candidate_can_be_profiled_without_becoming_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=prepared(folder)
            decision=dict(action='profile',parent_or_target_id='failed',resource_request={'instances':1})
            engine.tool(decision,'profile_failure')
            result=engine.archive.of('tool_result')[-1]['result'][0]
            self.assertFalse(result['valid'])
            self.assertIn('dunder',result['error'])
            self.assertTrue(result['error_location'])
            self.assertFalse(engine.candidates()['failed']['eligibility'])
            before=engine.ledger.used('case_runs')
            engine.tool(decision,'profile_failure')
            self.assertEqual(engine.ledger.used('case_runs'),before)

    def test_inspect_and_next_state_expose_actual_failure_not_only_null_score(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=prepared(folder)
            engine.tool(dict(action='inspect',parent_or_target_id='failed'),'inspect_failure')
            result=engine.archive.of('tool_result')[-1]['result']
            self.assertIn('dunder',str(result['execution_failures']))
            self.assertIn('dunder',str(engine.state()['candidate_diagnostics']))

    def test_invalid_revision_parent_is_returned_to_model_before_editor(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=prepared(folder)
            invalid=dict(action='revise',parent_or_target_id='failed',edit_count=1,proposed_intents=['fix'],evidence_ids=[])
            model=ReplySequence([invalid,dict(invalid,parent_or_target_id='seed')]);engine.model=model
            reply=engine.request('planner',engine.state(),'plan')
            self.assertEqual(reply['parent_or_target_id'],'seed')
            self.assertEqual(len(model.calls),2)
            self.assertIn('eligible',model.calls[1][1]['protocol_error'])

    def test_unknown_target_and_unknown_evidence_are_validated_before_execution(self):
        for reply in (dict(action='inspect',parent_or_target_id='missing',evidence_ids=[]),
                      dict(action='profile',parent_or_target_id='failed',resource_request={'instances':1},evidence_ids=['missing'])):
            with tempfile.TemporaryDirectory() as folder:
                engine=prepared(folder);model=ReplySequence([reply]);engine.model=model
                with self.assertRaisesRegex(ValueError,'unknown'):
                    engine.request('planner',engine.state(),'plan')
                self.assertEqual(len(model.calls),2)

    def test_invalid_experience_revision_is_validated_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=prepared(folder)
            reply=dict(action='inspect',parent_or_target_id='seed',evidence_ids=[],previous_event_update=[
                dict(card_id='missing',interpretation='text',status='待复查',evidence_ids=[])])
            engine.model=ReplySequence([reply])
            with self.assertRaisesRegex(ValueError,'unknown'):
                engine.request('planner',engine.state(),'plan')
            self.assertFalse(engine.archive.of('card_revision'))

    def test_B1_does_not_receive_historical_eligibility_through_tool_context(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=prepared(folder);engine.config['mode']='B1'
            state=engine.state()
            self.assertNotIn('candidate_diagnostics',state)
            self.assertNotIn('candidate_eligibility',state['action_context'])
            self.assertNotIn('events',state['action_context'])
            invalid=dict(action='inspect',parent_or_target_id='failed',evidence_ids=[])
            corrected=dict(action='revise',parent_or_target_id='seed',edit_count=1,proposed_intents=['fixture'],evidence_ids=[])
            model=ReplySequence([invalid,corrected]);engine.model=model
            self.assertEqual(engine.request('planner',state,'b1_plan'),corrected)
            self.assertNotIn('candidate_eligibility',model.calls[1][1]['action_context'])

    def test_twenty_offline_iterations_cover_failure_inspect_profile_analyze_and_revise(self):
        class Scenario:
            def __init__(self):self.round=0;self.phase='revise';self.last_child=None;self.bad_profile_once=False
            def call(self,role,payload):
                if role=='editor':
                    self.round+=1
                    self.phase='inspect'
                    self.last_child='step_'+str(len(engine.archive.of('action_complete')))+':PA'
                    return dict(package_id='fixture_'+str(self.round),parent_id=payload['parent_id'],parent_hash=payload['parent_hash'],
                        issue='offline failure-path fixture',evidence_ids=[],single_edit_reason='single fixture change',edits=[dict(
                        edit_id='A',purpose='exercise failure or success',prediction='observe',exact_replacements=[
                            dict(old=payload['parent_code'],new=BAD if self.round%2 else GOOD)])])
                if 'protocol_error' in payload:
                    return dict(payload['previous_response'],resource_request={'instances':1})
                if self.phase=='inspect':
                    self.phase='profile';return dict(action='inspect',parent_or_target_id=self.last_child,evidence_ids=[])
                if self.phase=='profile':
                    self.phase='analyze'
                    count=1
                    if not self.bad_profile_once:count=['x'];self.bad_profile_once=True
                    return dict(action='profile',parent_or_target_id=self.last_child,evidence_ids=[],resource_request={'instances':count})
                if self.phase=='analyze':
                    self.phase='revise';return dict(action='analyze',evidence_ids=[],analysis_code='def analyze(events):\n    return {"events":len(events)}')
                return dict(action='revise',parent_or_target_id='seed',edit_count=1,proposed_intents=['offline fixture'],evidence_ids=[])
        with tempfile.TemporaryDirectory() as folder:
            c=dict(config(folder),stop_rule='iterations',max_iterations=20)
            model=Scenario();engine=Engine(c,model)
            result=engine.run()
            self.assertEqual(result['status'],'finished',result['reason'])
            self.assertEqual(result['completed_iterations'],20)
            self.assertEqual(model.round,20)
            tools=engine.archive.of('tool_result')
            for action in ('inspect','profile','analyze'):
                self.assertEqual(sum(row['action']==action for row in tools),19)
            self.assertTrue(any(not result['valid'] for row in tools if row['action']=='profile' for result in row['result']))
            self.assertEqual(len(engine.archive.of('model_validation_failure')),1)


if __name__=='__main__':unittest.main()
