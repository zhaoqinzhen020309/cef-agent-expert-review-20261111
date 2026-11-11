import sys
import json
from unittest.mock import patch
import pytest
from cef_wsl_session import NativeBackendSession

def command(source):return [sys.executable,'-I','-u','-c',source]

def test_backend_session_holds_process_until_run_exits():
    with NativeBackendSession(command=command("import sys;print('CEF_BACKEND_READY',flush=True);sys.stdin.buffer.read()")) as session:
        assert session.process.poll() is None
    assert session.process.poll()==0

def test_backend_session_decodes_startup_failure():
    with pytest.raises(RuntimeError,match='ERROR_NO_SYSTEM_RESOURCES'):
        with NativeBackendSession(command=command("import sys;sys.stdout.buffer.write('ERROR_NO_SYSTEM_RESOURCES'.encode('utf-16-le'));sys.exit(1)")):pass

def test_backend_session_releases_process_on_error():
    session=NativeBackendSession(command=command("import sys;print('CEF_BACKEND_READY',flush=True);sys.stdin.buffer.read()"))
    with pytest.raises(ValueError,match='original error'):
        with session:raise ValueError('original error')
    assert session.process.poll()==0


def test_resource_failure_waits_then_starts_same_backend_and_saves_events(tmp_path):
    counter=tmp_path/'counter'
    source=("import sys,pathlib; p=pathlib.Path("+repr(str(counter))+"); "
            "n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); "
            "sys.stdout.buffer.write(('Wsl/Service/CreateInstance/CreateVm/HCS/ERROR_NO_SYSTEM_RESOURCES'.encode('utf-16-le')) if n==0 else b'CEF_BACKEND_READY\\n'); "
            "sys.stdout.flush(); sys.exit(1) if n==0 else sys.stdin.buffer.read()")
    events=tmp_path/'startup.jsonl'
    with patch('cef_wsl_session.time.sleep') as sleep:
        with NativeBackendSession(command=command(source),event_path=events) as session:
            assert session.process.poll() is None
        sleep.assert_called_once_with(5)
    rows=[json.loads(x) for x in events.read_text(encoding='utf-8').splitlines()]
    assert [r['kind'] for r in rows]==['waiting_resources','ready']
    assert counter.read_text()=='2'


def test_resource_wait_can_be_interrupted_without_model_or_tool_work():
    source="import sys;sys.stdout.buffer.write(b'Wsl/Service/CreateInstance/CreateVm/HCS/ERROR_NO_SYSTEM_RESOURCES');sys.exit(1)"
    with patch('cef_wsl_session.time.sleep',side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            with NativeBackendSession(command=command(source)):pass
