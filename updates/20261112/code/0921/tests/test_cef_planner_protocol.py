"""Strict Planner validation and explicit, accounted format repair."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from test_cef_engine import config


def malformed():
    return dict(action='revise',parent_or_target_id='seed',evidence_ids=[],
                previous_event_update=[],proposed_intents=['A','B'],issue='two changes')


class ReplySequence:
    def __init__(self,replies):self.replies=replies;self.calls=[]
    def call(self,role,payload):
        self.calls.append((role,payload))
        return self.replies[min(len(self.calls)-1,len(self.replies)-1)]


class PlannerProtocolTests(unittest.TestCase):
    def test_missing_edit_count_is_returned_to_model_not_inferred(self):
        with tempfile.TemporaryDirectory() as folder:
            model=ReplySequence([malformed(),dict(malformed(),edit_count=2)])
            engine=Engine(config(folder),model)
            reply=engine.request('planner',{'task_interface':'fixture'},'p')
            self.assertEqual(reply['edit_count'],2)
            self.assertEqual(len(model.calls),2)
            self.assertIn('edit_count',model.calls[1][1]['protocol_error'])
            self.assertNotIn('edit_count',model.calls[1][1]['previous_response'])
            self.assertEqual(len(engine.archive.of('model_validation_failure')),1)
            self.assertEqual(engine.request('planner',{},'p'),reply)
            self.assertEqual(len(model.calls),2)

    def test_repeated_malformed_reply_stops_after_one_explicit_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            model=ReplySequence([malformed()]);engine=Engine(config(folder),model)
            with self.assertRaisesRegex(ValueError,'edit_count'):
                engine.request('planner',{},'p')
            self.assertEqual(len(model.calls),2)
            self.assertFalse(engine.archive.of('model_response'))

    def test_boolean_or_string_edit_count_is_not_coerced(self):
        for count in (True,'2',None,0,3):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as folder:
                model=ReplySequence([dict(malformed(),edit_count=count)])
                with self.assertRaises(ValueError):Engine(config(folder),model).request('planner',{},'p')

    def test_resume_after_repair_does_not_repeat_model_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            model=ReplySequence([malformed(),dict(malformed(),edit_count=1)])
            engine=Engine(config(folder),model);add=engine.archive.add
            def interrupted(kind,data):
                if kind=='model_response' and data.get('call_id')=='p':raise RuntimeError('injected interruption')
                return add(kind,data)
            with patch.object(engine.archive,'add',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    engine.request('planner',{},'p')
            self.assertEqual(len(model.calls),2)
            self.assertEqual(Engine(config(folder),model).request('planner',{},'p')['edit_count'],1)
            self.assertEqual(len(model.calls),2)

    def test_diagnostic_short_limit_cannot_be_accepted_as_formal(self):
        from scripts.cef_launcher import build_config
        from cef.config import validate
        with tempfile.TemporaryDirectory() as folder:
            settings=dict(model='deepseek-flash',base_url='https://api.deepseek.com',
                          seeds=[2025],iterations=20,workers=4,output_dir=Path(folder))
            standard,_=build_config(2025,Path(folder)/'run',settings)
            self.assertEqual(standard['solver_time_limit_s'],60)
            self.assertEqual(len(standard['data_manifest']['train']),64)
            diagnostic=dict(standard,diagnostic_run=True,diagnostic_train_limit=4,solver_time_limit_s=1)
            with self.assertRaisesRegex(ValueError,'diagnostic'):
                validate(diagnostic)
            checked=validate(diagnostic,formal=False)
            self.assertEqual(len(Engine(checked,None).rows),4)


if __name__=='__main__':unittest.main()
