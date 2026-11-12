import subprocess
import pytest
from cef import reference_native as native


def test_backend_reports_actual_utf16_wsl_error(monkeypatch):
    monkeypatch.setattr(native,'backend_manifest',lambda: {})
    message='系统资源不足，无法完成请求的服务。\r\nWsl/Service/CreateInstance/CreateVm/HCS/ERROR_NO_SYSTEM_RESOURCES'
    monkeypatch.setattr(native.subprocess,'run',lambda *a,**k: subprocess.CompletedProcess(a,4294967295,message.encode('utf-16-le'),b''))
    with pytest.raises(RuntimeError) as error: native.ensure_backend_ready()
    assert 'ERROR_NO_SYSTEM_RESOURCES' in str(error.value)
    assert '系统资源不足' in str(error.value)
    assert '4294967295' in str(error.value)


def test_concorde_usage_exit_one_is_expected(monkeypatch):
    monkeypatch.setattr(native,'backend_manifest',lambda: {})
    monkeypatch.setattr(native.subprocess,'run',lambda *a,**k: subprocess.CompletedProcess(a,1,b'',b'Usage: concorde [-see below-] [dat_file]'))
    native.ensure_backend_ready()


def test_backend_timeout_is_reported_as_startup_timeout(monkeypatch):
    monkeypatch.setattr(native,'backend_manifest',lambda: {})
    def timeout(*a,**k): raise subprocess.TimeoutExpired(a,30)
    monkeypatch.setattr(native.subprocess,'run',timeout)
    with pytest.raises(RuntimeError,match='30.*Ubuntu-24.04'): native.ensure_backend_ready()
