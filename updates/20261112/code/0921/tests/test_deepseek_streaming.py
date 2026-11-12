import io
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from agent_ahd.agents.agent_runner import AgentResponseError, OpenAICompatibleAgentRunner, _read_deepseek_stream
from agent_ahd.core.backend_factory import build_agent_runner, backend_descriptor


def event(content=None, *, finish=None, usage=None, reasoning=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    value = {"id": "stream-1", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        value["usage"] = usage
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


USAGE = {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21,
         "completion_tokens_details": {"reasoning_tokens": 4}}
DONE = "data: [DONE]\n\n"


def run_stream(monkeypatch, data, *, factory=None):
    calls = []

    def open_request(request, **kwargs):
        calls.append(json.loads(request.data))
        return io.BytesIO(data.encode("utf-8"))

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: SimpleNamespace(open=open_request))
    runner = factory() if factory else OpenAICompatibleAgentRunner(
        endpoint="https://example.invalid/chat/completions", api_key="test-only",
        model="deepseek-flash", provider="deepseek", reasoning_effort="high", max_output_tokens=32768)
    return runner, calls


def test_stream_assembles_content_and_preserves_provider_usage(monkeypatch):
    data = ": keep-alive\n\n" + event(reasoning="private reasoning")
    data += event('{"message": "排') + event('程", "ok": true}')
    data += event(finish="stop", usage=USAGE) + DONE
    runner, calls = run_stream(monkeypatch, data)
    answer = runner.complete_json(system_prompt="Return JSON", user_prompt="Return a JSON object")
    assert answer.payload == {"message": "排程", "ok": True}
    assert answer.usage == USAGE
    assert "private reasoning" not in answer.raw_content
    assert calls[0]["stream"] is True
    assert calls[0]["stream_options"] == {"include_usage": True}
    assert calls[0]["max_tokens"] == 32768
    assert len(calls) == 1


@pytest.mark.parametrize("tail", ["", DONE])
def test_valid_json_without_finished_stream_is_rejected(monkeypatch, tail):
    runner, calls = run_stream(monkeypatch, event('{"ok":true}') + tail)
    with pytest.raises(AgentResponseError) as failure:
        runner.complete_json(system_prompt="Return JSON", user_prompt="JSON object")
    assert failure.value.stage == "network"
    assert '{"ok":true}' in failure.value.raw_content
    assert len(calls) == 1


def test_stop_without_done_keeps_usage_but_rejects_partial_transport(monkeypatch):
    runner, calls = run_stream(monkeypatch, event('{"ok":true}') + event(finish="stop", usage=USAGE))
    with pytest.raises(AgentResponseError) as failure:
        runner.complete_json(system_prompt="Return JSON", user_prompt="JSON object")
    assert failure.value.stage == "network"
    assert failure.value.usage == USAGE
    assert len(calls) == 1


def test_done_without_usage_is_rejected(monkeypatch):
    runner, _ = run_stream(monkeypatch, event('{"ok":true}') + event(finish="stop") + DONE)
    with pytest.raises(AgentResponseError) as failure:
        runner.complete_json(system_prompt="Return JSON", user_prompt="JSON object")
    assert failure.value.stage == "network"


@pytest.mark.parametrize("finish,stage", [("length", "response_validation"),
    ("content_filter", "response_validation"), ("insufficient_system_resource", "network")])
def test_non_stop_finish_is_not_a_candidate(monkeypatch, finish, stage):
    runner, calls = run_stream(monkeypatch, event('{"ok":true}') + event(finish=finish, usage=USAGE) + DONE)
    with pytest.raises(AgentResponseError) as failure:
        runner.complete_json(system_prompt="Return JSON", user_prompt="JSON object")
    assert failure.value.stage == stage
    assert failure.value.usage == USAGE
    assert len(calls) == 1


@pytest.mark.parametrize("bad_frame", ['data: {broken}\n\n',
    'data: {"error":{"message":"upstream error"}}\n\n'])
def test_malformed_or_error_event_fails_without_internal_retry(monkeypatch, bad_frame):
    runner, calls = run_stream(monkeypatch, event('{"ok":true}') + bad_frame + DONE)
    with pytest.raises(AgentResponseError) as failure:
        runner.complete_json(system_prompt="Return JSON", user_prompt="JSON object")
    assert failure.value.stage == "network"
    assert len(calls) == 1


def test_content_after_finish_is_rejected(monkeypatch):
    runner, _ = run_stream(monkeypatch, event('{"ok":true}') + event(finish="stop", usage=USAGE) + event(" ") + DONE)
    with pytest.raises(AgentResponseError):
        runner.complete_json(system_prompt="Return JSON", user_prompt="JSON object")


def test_streaming_is_part_of_locked_backend_identity(tmp_path):
    runner = build_agent_runner(backend="api", run_dir=tmp_path, role="code", provider="deepseek",
        endpoint="https://example.invalid/chat/completions", api_key="test-only", model="deepseek-flash",
        reasoning_effort="high", protocol_mode="v4", max_output_tokens=32768, concurrency=5)
    descriptor = backend_descriptor(runner)
    assert descriptor["transport"]["kind"] == "http_sse"
    assert descriptor["transport"]["stream"] is True
    assert descriptor["transport"]["completion_policy"] == "finish_stop_and_done_and_usage"
    assert descriptor["retry_policy"]["max_backend_retries"] == 0


def test_other_providers_keep_json_envelope_transport(monkeypatch):
    envelope = {"choices": [{"message": {"content": '{"ok":true}'}}], "usage": {"total_tokens": 1}}
    calls = []
    def open_request(request, **kwargs):
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps(envelope).encode())
    monkeypatch.setattr("urllib.request.urlopen", open_request)
    runner = OpenAICompatibleAgentRunner(endpoint="https://example.invalid/chat/completions", api_key="test-only", model="other")
    assert runner.complete_json(system_prompt="JSON", user_prompt="JSON").payload == {"ok": True}
    assert "stream" not in calls[0]


def test_absolute_deadline_applies_inside_a_slowly_arriving_line():
    client, server = socket.socketpair()
    reader = client.makefile("rb")
    stopped = threading.Event()

    def send_slow_line():
        try:
            for _ in range(25):
                server.sendall(b":")
                if stopped.wait(0.02):
                    return
            server.sendall(b"\n")
        except OSError:
            pass

    writer = threading.Thread(target=send_slow_line, daemon=True)
    writer.start()
    response = SimpleNamespace(fp=SimpleNamespace(raw=SimpleNamespace(_sock=client)),
                               readline=reader.readline, read1=reader.read1)
    started = time.monotonic()
    try:
        with pytest.raises(AgentResponseError) as failure:
            _read_deepseek_stream(response, deadline=started + 0.12)
        elapsed = time.monotonic() - started
        assert failure.value.stage == "network"
        assert elapsed < 0.30, f"one line bypassed the whole-request deadline: {elapsed:.3f}s"
    finally:
        stopped.set()
        writer.join(timeout=1)
        reader.close()
        client.close()
        server.close()


def test_utf8_characters_can_span_incremental_reads():
    class Fragmented(io.BytesIO):
        def read1(self, size=-1):
            return super().read1(min(size, 3))

    wire = event('{"message":"排程"}') + event(finish="stop", usage=USAGE) + DONE
    answer = _read_deepseek_stream(Fragmented(wire.encode()), deadline=time.monotonic() + 2)
    assert json.loads(answer["choices"][0]["message"]["content"]) == {"message": "排程"}
