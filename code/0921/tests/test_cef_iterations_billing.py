import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import os
from unittest.mock import patch
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from test_cef_engine import Fake, config


def iteration_config(folder, count=2):
    return dict(config(folder), stop_rule="iterations", max_iterations=count,
                max_case_runs=1, max_llm_requests=1, max_llm_tokens=1,
                max_wallclock_s=.001, max_analysis_seconds=.001)


class IterationTests(unittest.TestCase):
    def test_only_iterations_end_search_despite_tiny_resource_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Engine(iteration_config(folder), Fake())
            result = engine.run()
            self.assertEqual(result["status"], "finished", result["reason"])
            self.assertEqual(len(engine.archive.of("contrast")), 2)
            self.assertEqual(result["completed_iterations"], 2)
            self.assertTrue(all(value is None for value in result["budget"].values()))

    def test_inspect_does_not_count_and_completed_resume_does_not_generate(self):
        class InspectFirst(Fake):
            inspected = False
            def call(self, role, payload):
                if role == "planner" and not self.inspected:
                    self.inspected = True
                    return dict(action="inspect", parent_or_target_id="seed", evidence_ids=[], previous_event_update=[])
                return super().call(role, payload)
        with tempfile.TemporaryDirectory() as folder:
            model = InspectFirst()
            engine = Engine(iteration_config(folder), model)
            result = engine.run()
            self.assertEqual(result["status"], "finished", result["reason"])
            self.assertEqual(len(engine.archive.of("action_complete")), 3)
            calls = len(model.inputs)
            self.assertEqual(Engine(iteration_config(folder), model).run()["status"], "finished")
            self.assertEqual(len(model.inputs), calls)

    def test_early_planner_finish_is_a_protocol_failure_not_success(self):
        class EarlyFinish:
            def call(self, role, payload):
                return dict(action="finish", previous_event_update=[], evidence_ids=[])
        with tempfile.TemporaryDirectory() as folder:
            result = Engine(iteration_config(folder), EarlyFinish()).run()
            self.assertEqual(result["status"], "failed")
            self.assertIn("iteration", result["reason"])

    def test_analysis_usage_is_not_a_cumulative_or_per_round_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=Engine(iteration_config(folder),Fake())
            with patch('cef.analysis.analyze',return_value=dict(elapsed=.3,result=None)):
                for number in range(3):
                    step='analysis_'+str(number)
                    engine.tool(dict(action='analyze',analysis_code='def analyze(events): return None'),step)
                    engine.archive.add('action_complete',dict(step=step,action='analyze',active_parent=None))
            result=engine.run()
            self.assertEqual(result['status'],'finished',result['reason'])
            self.assertEqual(result['completed_iterations'],2)
            self.assertAlmostEqual(result['resource_usage']['analysis_seconds'],.9)


class BillingTests(unittest.TestCase):
    def billing(self):
        return importlib.import_module("cef.billing")

    def test_unknown_reservations_are_not_counted_as_actual_usage(self):
        billing = self.billing()
        records = [dict(resource="llm_requests", amount=1),
                   dict(resource="llm_tokens", amount=35519, usage_known=False, conservative_debit=True),
                   dict(resource="llm_requests", amount=1),
                   dict(resource="llm_tokens", amount=12, usage_known=True,
                        input_tokens=10, cached_input_tokens=3, output_tokens=2,
                        actual_usage=dict(prompt_tokens=10, completion_tokens=2))]
        result = billing.summarize_usage(records)
        self.assertEqual(result["known_total_tokens"], 12)
        self.assertEqual(result["unknown_usage_requests"], 1)
        self.assertIsNone(result["total_tokens"])
        self.assertIsNone(result["provider_reported_cost"])

    def test_balance_change_preserves_currency_and_is_not_per_request_invoice(self):
        billing = self.billing()
        before = dict(balance_infos=[dict(currency="CNY", total_balance="5.00")])
        after = dict(balance_infos=[dict(currency="CNY", total_balance="4.70")])
        result = billing.balance_changes(before, after)
        self.assertEqual(result[0]["currency"], "CNY")
        self.assertEqual(result[0]["balance_decrease"], "0.30")
        self.assertEqual(result[0]["scope"], "account-wide balance change, not per-request invoice")

    def test_failed_balance_lookup_does_not_invent_zero_cost(self):
        result = self.billing().balance_changes(dict(error="unavailable"), dict(error="unavailable"))
        self.assertIsNone(result)

    def test_balance_lookup_uses_real_sdk_without_guessing_amount(self):
        billing=self.billing()
        requests=[]
        def handler(request):
            requests.append(str(request.url))
            return httpx.Response(200,json=dict(is_available=True,balance_infos=[dict(currency="CNY",total_balance="5.00")]))
        def client(**kwargs): return httpx.AsyncClient(transport=httpx.MockTransport(handler),**kwargs)
        with patch.dict(os.environ,{"CEF_TEST_KEY":"offline"}),patch.object(billing,"DefaultAsyncHttpxClient",side_effect=client):
            result=billing.balance_snapshot(dict(api_key_env="CEF_TEST_KEY",endpoint="https://example.com/v1",route="direct"))
        self.assertIn("balance_infos",result)
        self.assertEqual(result["balance_infos"][0]["total_balance"],"5.00")
        self.assertEqual(requests,["https://example.com/user/balance"])


if __name__ == "__main__":
    unittest.main()
