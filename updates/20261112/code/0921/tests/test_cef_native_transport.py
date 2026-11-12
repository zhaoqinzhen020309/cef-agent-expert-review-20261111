import json
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import pytest
from cef import reference_native as native

def test_cancel_uses_bytes_and_decodes_wsl_utf16(tmp_path):
    request=dict(binary='/solver',pid_path='/pid',cancel_path='/cancel',cancel_name='cancel.flag')
    (tmp_path/'pid.json').write_text('{}')
    request['pid_path']=native.linux_path(tmp_path/'pid.json')
    process=Mock();process.poll.return_value=0
    error='系统资源不足 ERROR_NO_SYSTEM_RESOURCES'.encode('utf-16-le')
    with patch.object(native.subprocess,'run',return_value=Mock(returncode=1,stdout=error,stderr=b'')) as run:
        with pytest.raises(RuntimeError,match='ERROR_NO_SYSTEM_RESOURCES'):
            native.cancel_native(process,request,tmp_path)
    assert isinstance(run.call_args.kwargs['input'],bytes)
    assert not run.call_args.kwargs.get('text',False)

def test_startup_failure_preserves_stdout_and_never_launches_cancel_worker(tmp_path):
    message='系统资源不足 ERROR_NO_SYSTEM_RESOURCES'
    process=Mock(returncode=1);process.wait.return_value=1;process.poll.return_value=1
    def spawn(*args,**kwargs):
        kwargs['stdout'].buffer.write(message.encode('utf-16-le'))
        kwargs['stdout'].flush()
        return process
    with patch.object(native.subprocess,'Popen',side_effect=spawn),patch.object(native.subprocess,'run') as run:
        with pytest.raises(RuntimeError,match='ERROR_NO_SYSTEM_RESOURCES'):
            native.solve_tsp_reference(np.ones((10,10))-np.eye(10),checkpoint_path=tmp_path/'failed.json',console=False)
    run.assert_not_called()

def test_cleanup_failure_does_not_replace_original_interruption(tmp_path):
    process=Mock(returncode=0);process.wait.side_effect=KeyboardInterrupt('original interruption')
    with patch.object(native.subprocess,'Popen',return_value=process),patch.object(native,'cancel_native',side_effect=RuntimeError('cleanup failed')):
        with pytest.raises(KeyboardInterrupt,match='original interruption') as error:
            native.solve_tsp_reference(np.ones((10,10))-np.eye(10),checkpoint_path=tmp_path/'cancel.json',console=False)
    assert any('cleanup failed' in note for note in error.value.__notes__)
