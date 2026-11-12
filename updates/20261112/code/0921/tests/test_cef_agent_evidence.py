"""Verify real evidence reaches the existing Agent; no prescribed research strategy."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from cef.feedback import planner_history_observation
from cef.patch import digest
from cef.research import ResearchTools
from test_cef_common_start import settings,source
from test_cef_tool_interface import measured_result


class AgentEvidenceTests(unittest.TestCase):
    def make_engine(self,folder):
        e=Engine(settings(folder),None)
        e.register('older',source(0))
        e.register('newer',source(1),'older')
        for cid,score in [('older',0.),('newer',1.)]:
            e.archive.add('measurement',dict(candidate_id=cid,valid=True,fitness=score))
        event=dict(event_id='change',parent_id='older',variant_ids={'PAB':'newer'},
                   scores={'PAB':1.},global_best_updated=False,affected_functions=['search'])
        e.archive.add('contrast',event)
        return e,event

    def test_normal_feedback_has_actual_version_bound_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            e,event=self.make_engine(folder)
            view=e.visible_event(event)
            diff=view['source_comparisons']['PAB']
            self.assertEqual(diff['parent_id'],'older')
            self.assertEqual(diff['candidate_id'],'newer')
            self.assertEqual(diff['parent_hash'],digest(source(0)))
            self.assertEqual(diff['candidate_hash'],digest(source(1)))
            self.assertIn('-MODE = 0',diff['diff'])
            self.assertIn('+MODE = 1',diff['diff'])
            self.assertTrue(diff['diff_complete'])
            self.assertEqual(e.archive.of('contrast'),[event])
            replay=planner_history_observation({'recent_events':[view]})
            self.assertEqual(replay['recent_events'][0]['source_comparisons'],view['source_comparisons'])

    def test_failed_candidate_is_compared_not_replaced_by_the_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            e,event=self.make_engine(folder)
            e.archive.add('measurement',dict(candidate_id='newer',valid=False,fitness=None))
            self.assertFalse(e.candidates()['newer']['eligibility'])
            view=e.visible_event(dict(event,scores={'PAB':None}))
            self.assertIn('+MODE = 1',view['source_comparisons']['PAB']['diff'])

    def test_old_memory_and_inspect_and_editor_use_the_same_evaluated_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            e,event=self.make_engine(folder)
            e.experiences.create('old-card',dict(parent_hash=digest(source(0)),
                source_event_ids=['change'],affected_functions=['search']))
            for i in range(3):
                e.archive.add('contrast',dict(event,event_id='later:'+str(i)))
            card=next(c for c in e.state('newer')['retrieved_cards'] if c['card_id']=='old-card')
            diff=card['source_comparisons']['change']['PAB']
            self.assertEqual(diff['parent_id'],'older')
            e.tool({'action':'inspect','parent_or_target_id':'old-card'},'read-card')
            self.assertEqual(e.archive.of('tool_result')[-1]['result']['source_comparisons']['change']['PAB'],diff)
            e.tool({'action':'inspect','parent_or_target_id':'newer'},'read-source')
            self.assertEqual(e.archive.of('tool_result')[-1]['result']['source_comparison'],diff)
            editor=e.editor_context('older',{'evidence_ids':['old-card']},2)
            self.assertEqual(editor['parent_code'],source(0))
            self.assertEqual(editor['experiences'][0]['source_comparisons']['change']['PAB'],diff)

    def test_full_rewrite_is_explicitly_bounded_without_claiming_the_diff_is_complete(self):
        from cef import feedback
        before='def alpha():\n    return 1\n'
        after='def beta():\n'+''.join('    x%d = %d\n'%(i,i) for i in range(3000))+'    return 2\n'
        self.assertTrue(hasattr(feedback,'source_comparison'),'source comparison is not implemented')
        view=feedback.source_comparison('a','b',before,after)
        self.assertFalse(view['diff_complete'])
        self.assertLess(len(view['diff']),16000)
        self.assertIn('alpha',view['changed_definitions']['functions'])
        self.assertIn('beta',view['changed_definitions']['functions'])

    def test_tied_improvement_and_zero_and_worsening_are_different_measurements(self):
        raw=measured_result([90.]*8)
        for i,s in enumerate(raw['state_results']):
            g=.15 if i==0 else -.05 if i==1 else 0.
            s.update(reference_gain_ratio=g,candidate_gain_ratio=g,delta=0.)
        view=ResearchTools.brief(raw)
        self.assertEqual(view['tied_outcomes'],dict(both_improved=1,both_unchanged=6,both_worsened=1,mixed_within_tolerance=0))
        self.assertEqual(view['state_results'][0]['reference_gain_display'],'15% of starting objective')
        self.assertEqual(view['state_results'][1]['candidate_gain_display'],'-5% of starting objective')
        self.assertEqual(view['mean_candidate_minus_reference_gain_ratio'],raw['mean_delta'])
        self.assertEqual(ResearchTools.brief(view),view)

    def test_preceding_report_version_remains_readable_without_rewriting_it(self):
        raw=measured_result();old=ResearchTools.brief(raw)
        old['tool_interface_version']='common_start_tool_v2'
        old.pop('tied_outcomes',None)
        before=copy.deepcopy(old)
        view=ResearchTools.brief(old)
        self.assertIn('tied_outcomes',view)
        self.assertEqual(old,before)

    def test_existing_analysis_action_has_an_executable_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            e,event=self.make_engine(folder)
            tools=e.action_context()['tools']
            self.assertIn('analysis_code',tools['analyze']['arguments'])
            self.assertIn('source_comparisons',tools['analyze']['description'])
            reply={'action':'analyze','analysis_code':
                'def analyze(events):\n    return {"target": 0.3*60+0.7*(60-1.2), "stop": 60-1.2}\n'}
            e.tool(reply,'numeric-check')
            result=e.archive.of('tool_result')[-1]['result']
            self.assertIsNone(result['error'])
            self.assertAlmostEqual(result['result']['target'],59.16)
            self.assertAlmostEqual(result['result']['stop'],58.8)


if __name__=='__main__':unittest.main()
