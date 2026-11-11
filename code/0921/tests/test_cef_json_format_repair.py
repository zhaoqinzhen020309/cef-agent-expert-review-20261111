"""Malformed raw completions pass through the real SDK/parser, never a live API."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cef.engine import Engine
from cef.ledger import BudgetLedger
from cef.model import Model
from cef.store import Archive
from test_cef_sdk import completion, config as sdk_config, sdk_transport


BROKEN = '{"proposed_intents":["A","B","expected_observations":[]}'
CORRECTED = {"proposed_intents": ["A", "B"], "expected_observations": []}


def response(content, *, stream=False, finish="stop", usage=True):
    if not stream:
        return httpx.Response(200, json=completion(content, finish, usage))
    items = [dict(id="offline", object="chat.completion.chunk", created=0,
                  model="offline", choices=[dict(index=0, delta={"content": part},
                  finish_reason=None)]) for part in (content[:13], content[13:])]
    items.append(dict(id="offline", object="chat.completion.chunk", created=0,
                      model="offline", choices=[dict(index=0, delta={}, finish_reason=finish)]))
    if usage:
        items.append(dict(id="offline", object="chat.completion.chunk", created=0,
                          model="offline", choices=[], usage=completion()["usage"]))
    body = "".join("data: " + json.dumps(item) + "\n\n" for item in items) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


class JSONFormatRepairTests(unittest.TestCase):
    def fixture(self, folder, **changes):
        archive = Archive(folder)
        ledger = BudgetLedger(archive, dict(max_llm_requests=None, max_llm_tokens=None,
            max_case_runs=None, max_analysis_seconds=None, max_wallclock_s=None), enforce_limits=False)
        return Model(sdk_config(stop_rule="iterations", rates=None, **changes), archive, ledger), archive, ledger

    def test_missing_bracket_is_explicitly_corrected_and_both_calls_are_metered(self):
        for role in ("planner", "editor"):
            for stream in (False, True):
                with self.subTest(role=role, stream=stream), tempfile.TemporaryDirectory() as folder:
                    requests = []
                    def handler(request):
                        requests.append(json.loads(request.content))
                        return response(BROKEN if len(requests) == 1 else json.dumps(CORRECTED), stream=stream)
                    with patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
                        model, archive, ledger = self.fixture(folder, stream=stream)
                        payload = {"task_interface": "unchanged", "evidence_ids": ["verified:1"]}
                        result = model.call(role, payload)
                    self.assertEqual(result, CORRECTED)
                    self.assertEqual(len(requests), 2)
                    repair = json.loads(requests[1]["messages"][-1]["content"])
                    self.assertEqual(repair["previous_response_text"], BROKEN)
                    self.assertIn("delimiter", repair["protocol_error"])
                    self.assertEqual(repair["task_interface"], "unchanged")
                    self.assertEqual(repair["evidence_ids"], ["verified:1"])
                    self.assertNotIn("previous_response_text", payload, "Caller input must not be mutated")
                    self.assertEqual(ledger.used("llm_requests"), 2)
                    self.assertEqual(ledger.used("llm_tokens"), 24)
                    self.assertEqual([r["usage_known"] for r in archive.of("charge") if r["resource"] == "llm_tokens"], [True, True])
                    raw = archive.of("provider_response")
                    self.assertEqual(raw[0]["content"], BROKEN)
                    self.assertIsNone(raw[0]["parsed"])
                    self.assertEqual(raw[1]["parsed"], CORRECTED)
                    self.assertEqual(len(archive.of("model_format_repair")), 1)

    def test_repeated_bad_json_stops_after_one_recorded_correction(self):
        requests = []
        def handler(request):
            requests.append(request)
            return response(BROKEN)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
            model, archive, ledger = self.fixture(folder, transport_retry_limit=2)
            with self.assertRaisesRegex(ValueError, "delimiter"):
                model.call("planner", {})
            self.assertEqual(len(requests), 2, "Format errors must not consume transport retries")
            self.assertEqual(len(archive.of("model_format_repair")), 1)
            self.assertEqual(len(archive.of("model_format_failure")), 2)
            self.assertEqual(ledger.used("llm_tokens"), 24)
            self.assertTrue(all(r["parsed"] is None for r in archive.of("provider_response")))

    def test_missing_usage_or_truncated_response_is_not_a_format_correction(self):
        for finish, usage in (("length", True), ("stop", False)):
            with self.subTest(finish=finish, usage=usage), tempfile.TemporaryDirectory() as folder:
                requests = []
                def handler(request):
                    requests.append(request)
                    return response(BROKEN, finish=finish, usage=usage)
                with patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
                    model, archive, _ = self.fixture(folder)
                    with self.assertRaises(ValueError):
                        model.call("editor", {})
                self.assertEqual(len(requests), 1)
                self.assertFalse(archive.of("model_format_repair"))

    def test_editor_embedded_code_survives_correction_exactly(self):
        source = 'def solve(x):\n    return {"solution": [0], "label": "a]b\\\\c"}\n'
        corrected = {"edits": [{"exact_replacements": [{"old": "original", "new": source}]}]}
        requests = []
        def handler(request):
            requests.append(request)
            return response('{"edits":[' if len(requests) == 1 else json.dumps(corrected))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
            model, _, _ = self.fixture(folder)
            result = model.call("editor", {"parent_code": "original"})
        self.assertEqual(result["edits"][0]["exact_replacements"][0]["new"], source)

    def test_exact_failed_run_response_is_recoverable_without_rewriting_history(self):
        text = (Path(__file__).parent / "fixtures/cef_planner_missing_array_20261029.txt").read_text(encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError) as failure:
            json.loads(text)
        self.assertEqual(failure.exception.pos, 7459)
        # The fixed reply is an explicit offline provider fixture, never a runtime repair heuristic.
        pos = text.index(',"expected_observations"')
        corrected_text = text[:pos] + "]" + text[pos:]
        requests = []
        def handler(request):
            requests.append(request)
            return response(text if len(requests) == 1 else corrected_text, stream=True)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
            model, archive, _ = self.fixture(folder, stream=True)
            result = model.call("planner", {"task_interface": "offline historical-response replay"})
            self.assertEqual(result, json.loads(corrected_text))
            self.assertEqual(result["parent_or_target_id"], "step_1:PAB")
            self.assertEqual(len(result["proposed_intents"]), 2)
            self.assertEqual(archive.of("provider_response")[0]["content"], text)
            self.assertEqual(len(requests), 2)

    def test_corrected_planner_still_requires_workflow_validation(self):
        from test_cef_engine import config as engine_config
        requests = []
        def handler(request):
            requests.append(request)
            return response(BROKEN if len(requests) == 1 else '{"action":"revise","edit_count":0}')
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
            cfg = dict(engine_config(folder), **sdk_config(), stop_rule="iterations")
            engine = Engine(cfg, None)
            engine.model = Model(cfg, engine.archive, engine.ledger)
            with self.assertRaises(ValueError):
                engine.request("planner", {}, "step_0:planner", allow_schema_repair=False)
            self.assertFalse(engine.archive.of("model_response"))
            self.assertFalse(engine.archive.of("package"))
            self.assertEqual(len(requests), 2)

    def test_common_start_loop_survives_bad_planner_and_editor_text_and_tests_export(self):
        from test_cef_common_start import ExperimentModel, settings
        from cef.reporting import final_test
        from cef.patch import digest
        fixture = ExperimentModel()
        requests, failures, pending = [], set(), {}
        def handler(request):
            body = json.loads(request.content)
            payload = json.loads(body["messages"][-1]["content"])
            role = "planner" if "research controller" in body["messages"][0]["content"] else "editor"
            requests.append((role, payload))
            if payload.get('internal_task')=='evidence_review':
                return response(json.dumps(dict(findings=[],edits=[])),stream=True)
            if "previous_response_text" in payload:
                return response(pending.pop(role), stream=True)
            answer = fixture.call(role, payload)
            # Trigger real raw-text failures after an already evaluated candidate.
            if fixture.generated == (1 if role == "planner" else 2) and role not in failures:
                failures.add(role)
                pending[role] = json.dumps(answer)
                return response(BROKEN, stream=True)
            return response(json.dumps(answer), stream=True)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
            cfg = dict(settings(str(Path(folder) / "search")), **sdk_config(stream=True, rates=None),
                       diagnostic_run=True, diagnostic_train_limit=1, test_repeats=1)
            cfg["data_manifest"]["test"] = [dict(cfg["data_manifest"]["train"][0], id="tiny_test", split="test", reference=4)]
            engine = Engine(cfg, None)
            engine.model = Model(cfg, engine.archive, engine.ledger)
            result = engine.run()
            self.assertEqual(result["status"], "finished", result["reason"])
            self.assertEqual(result["completed_iterations"], 3)
            self.assertEqual({r["candidate_id"] for r in engine.archive.of("execution")}, {"step_0:PAB", "step_1:PAB", "step_3:PAB"})
            self.assertEqual(len(engine.archive.of("execution")), 3)
            self.assertEqual(len(engine.archive.of("probe_result")), 1)
            self.assertEqual(len(engine.archive.of("model_format_repair")), 2)
            self.assertEqual(failures, {"planner", "editor"})
            count = len(requests)
            restored = Engine(cfg, None)
            restored.model = Model(cfg, restored.archive, restored.ledger)
            self.assertEqual(restored.run()["completed_iterations"], 3)
            self.assertEqual(len(requests), count, "Completed search must not be regenerated")
            self.assertEqual(len(restored.archive.of("execution")), 3)
            before = digest(restored.archive.path.read_text(encoding="utf-8"))
            with patch("cef.reporting.audit_test"):
                tested = final_test(cfg["run_dir"], Path(folder) / "test", cfg["data_manifest"]["test"], 1, {"tiny_test:0": 2025})
            self.assertTrue(tested["valid"])
            self.assertTrue(tested["search_unchanged"])
            self.assertEqual(before, digest(restored.archive.path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
