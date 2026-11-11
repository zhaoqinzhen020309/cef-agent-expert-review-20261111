"""Exercise the real OpenAI SDK against an in-memory HTTP transport, never an API."""
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cef.model import Model
from cef.store import Archive
from cef.ledger import BudgetLedger
from cef.preflight import identity


def config(**changes):
    value = dict(endpoint="https://example.com", api_key_env="CEF_TEST_KEY", route="direct",
                 model_snapshot="offline", model_sampling={"reasoning_effort": "high",
                 "thinking": {"type": "enabled"}}, max_output_tokens=100, provider="offline",
                 request_timeout_s=1, wire_api="chat_completions", stream=False,
                 rates=dict(input_per_million=1, cached_input_per_million=.1,
                            output_per_million=2, currency="fixture"))
    return dict(value, **changes)


def completion(content='{"action":"finish"}', finish="stop", usage=True):
    result = dict(id="offline", object="chat.completion", created=0, model="offline",
                  choices=[dict(index=0, message=dict(role="assistant", content=content),
                                finish_reason=finish)])
    if usage:
        result["usage"] = dict(prompt_tokens=10, completion_tokens=2, total_tokens=12,
                               prompt_cache_hit_tokens=3)
    return result


@contextmanager
def sdk_transport(handler):
    options = []
    def make_client(**kwargs):
        options.append(kwargs)
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    with patch("cef.model.DefaultAsyncHttpxClient", side_effect=make_client, create=True), patch(
        "urllib.request.OpenerDirector.open", side_effect=AssertionError("SDK required; no legacy HTTP dispatch")
    ):
        yield options


class SDKTests(unittest.TestCase):
    def fixture(self, folder, **changes):
        archive = Archive(folder)
        ledger = BudgetLedger(archive, dict(max_llm_requests=4, max_llm_tokens=50000,
            max_case_runs=1, max_analysis_seconds=1, max_wallclock_s=60))
        return Model(config(**changes), archive, ledger), archive, ledger

    def test_standard_nonstream_body_and_usage(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=completion())
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler) as options:
            model, archive, ledger = self.fixture(folder)
            self.assertEqual(model.call("planner", {}), {"action": "finish"})
        body = json.loads(requests[0].content)
        self.assertEqual(str(requests[0].url), "https://example.com/chat/completions")
        self.assertEqual(body["model"], "offline")
        self.assertFalse(body["stream"])
        self.assertNotIn("stream_options", body)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["max_tokens"], 100)
        self.assertFalse(options[0]["trust_env"])
        self.assertEqual(ledger.used("llm_tokens"), 12)
        self.assertAlmostEqual(archive.of("charge")[-1]["fee"], .0000113)

    def test_provider_errors_are_not_retried(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(429, json={"error": {"message": "rate limited", "type": "rate_limit"}})
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
            model, archive, ledger = self.fixture(folder)
            with self.assertRaises(openai.RateLimitError):
                model.call("editor", {})
            self.assertEqual(len(requests), 1)
            self.assertEqual(ledger.used("llm_requests"), 1)
            self.assertTrue(archive.of("charge")[-1]["conservative_debit"])

    def test_total_deadline_cancels_slow_response(self):
        async def handler(request):
            await asyncio.sleep(.5)
            return httpx.Response(200, json=completion())
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(handler):
            model, archive, _ = self.fixture(folder, request_timeout_s=.05)
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                model.call("editor", {})
            self.assertLess(time.monotonic() - started, .4)
            self.assertTrue(archive.of("charge")[-1]["conservative_debit"])

    def test_truncation_records_known_usage_and_refuses_partial_json(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(
            lambda request: httpx.Response(200, json=completion('{"x":', "length"))
        ):
            model, archive, ledger = self.fixture(folder)
            with self.assertRaisesRegex(ValueError, "incomplete model response"):
                model.call("editor", {})
            self.assertEqual(ledger.used("llm_tokens"), 12)
            self.assertIsNone(archive.of("provider_response")[-1]["parsed"])

    def test_sdk_transport_and_stream_are_part_of_receipt_identity(self):
        first = identity(config())
        self.assertEqual(first.get("transport"), "openai_sdk")
        self.assertNotEqual(first, identity(config(stream=True)))

    def test_missing_usage_fails_without_inventing_token_counts(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(
            lambda request: httpx.Response(200, json=completion(usage=False))
        ):
            model, archive, _ = self.fixture(folder)
            with self.assertRaisesRegex(ValueError, "omitted required usage"):
                model.call("editor", {})
            self.assertTrue(archive.of("charge")[-1]["conservative_debit"])
            self.assertIsNone(archive.of("charge")[-1]["fee"])

    def test_stream_heartbeats_cannot_extend_total_deadline(self):
        class SlowBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    await asyncio.sleep(.01)
                    yield b": heartbeat\n\n"
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(
            lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowBody())
        ):
            model, archive, _ = self.fixture(folder, request_timeout_s=.08, stream=True)
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                model.call("editor", {})
            self.assertLess(time.monotonic() - started, .5)
            self.assertTrue(archive.of("charge")[-1]["conservative_debit"])

    def test_environment_proxy_is_an_explicit_sdk_client_option(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"CEF_TEST_KEY": "offline"}), sdk_transport(
            lambda request: httpx.Response(200, json=completion())
        ) as options:
            model, _, _ = self.fixture(folder, route="environment_proxy")
            model.call("editor", {})
            self.assertTrue(options[0]["trust_env"])


if __name__ == "__main__":
    unittest.main()
