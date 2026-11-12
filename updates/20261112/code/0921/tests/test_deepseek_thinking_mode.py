import io
import json
from types import SimpleNamespace
import pytest

from agent_ahd.agents.agent_runner import OpenAICompatibleAgentRunner


@pytest.mark.parametrize("effort,thinking", [("none", "disabled"), ("low", "enabled"), ("high", "enabled")])
def test_deepseek_effort_controls_wire_thinking_mode(monkeypatch, effort, thinking):
    captured = []
    def open_request(request, **kwargs):
        captured.append(json.loads(request.data))
        chunk = {"choices": [{"index": 0, "delta": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                 "usage": {"total_tokens": 1}}
        return io.BytesIO(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: SimpleNamespace(open=open_request))
    runner = OpenAICompatibleAgentRunner(endpoint="https://example.invalid/chat/completions", api_key="test-only",
        model="deepseek-flash", provider="deepseek", reasoning_effort=effort, max_output_tokens=16384)
    answer = runner.complete_json(system_prompt="Return JSON", user_prompt="Return {ok: true}")
    assert answer.payload == {"ok": True}
    assert captured[0]["thinking"] == {"type": thinking}
    assert captured[0]["reasoning_effort"] == effort
    assert captured[0]["max_tokens"] == 16384
    assert captured[0]["stream"] is True
