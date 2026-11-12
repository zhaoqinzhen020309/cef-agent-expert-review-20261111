import io,json,sys
from pathlib import Path
import pytest

from cef_log_output import RunLog,equivalent_entrypoints


def test_one_console_and_persistent_human_raw_logs(tmp_path,monkeypatch):
    terminal=io.StringIO();monkeypatch.setattr(sys,'stdout',terminal)
    with RunLog(tmp_path):
        print('data_designer | response=1')
        print('Evaluate data_round_1:old | 128/128 | workers=8')
        path=tmp_path/'rounds/001/data/agent_task/session/facts.jsonl';path.parent.mkdir(parents=True)
        record=dict(seq=0,time=1,kind='tool_started',data=dict(turn=1,call_id='x',call=dict(function=dict(name='read_artifact',arguments='{"artifact_id":"parent.py"}'))))
        path.write_text(json.dumps(record)+'\n')
    visible=terminal.getvalue()
    assert '读取 parent.py' in visible and 'response=1' not in visible
    assert '128/128' in visible
    assert '读取 parent.py' in (tmp_path/'progress.log').read_text(encoding='utf-8')
    assert 'data_designer | response=1' in (tmp_path/'console.raw.log').read_text(encoding='utf-8')


def test_exception_persisted_and_streams_restored(tmp_path,monkeypatch):
    terminal=io.StringIO();monkeypatch.setattr(sys,'stdout',terminal)
    with pytest.raises(RuntimeError,match='test failure'):
        with RunLog(tmp_path):
            print('before failure',end='')
            raise RuntimeError('test failure')
    assert sys.stdout is terminal
    assert 'before failure' in terminal.getvalue()
    assert 'RuntimeError: test failure' in (tmp_path/'console.raw.log').read_text(encoding='utf-8')
    assert 'RuntimeError: test failure' in (tmp_path/'progress.log').read_text(encoding='utf-8')


def test_resume_does_not_duplicate_journal_events(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'stdout',io.StringIO())
    path=tmp_path/'search/agent_tasks/step_0_planner/session/facts.jsonl';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(seq=0,time=1,kind='model_started',data=dict(turn=1)))+'\n')
    with RunLog(tmp_path):pass
    with path.open('a') as f:f.write(json.dumps(dict(seq=1,time=2,kind='model_started',data=dict(turn=2)))+'\n')
    with RunLog(tmp_path):pass
    text=(tmp_path/'progress.log').read_text(encoding='utf-8')
    assert text.count('第1次响应')==1 and text.count('第2次响应')==1


def test_logging_bootstrap_compatibility_excludes_configuration_and_logic_changes():
    prefix="from pathlib import Path\nCONFIG={'workers':8}\n"
    old=prefix+"if __name__=='__main__':\n from scripts.cef_alternating_launcher import main\n raise SystemExit(main(CONFIG))\n"
    new=prefix+"if __name__=='__main__':\n from cef_log_output import launch\n raise SystemExit(launch(CONFIG))\n"
    assert equivalent_entrypoints(old,new)
    assert not equivalent_entrypoints(old,new.replace("'workers':8","'workers':4"))
    assert not equivalent_entrypoints(old,new.replace('launch(CONFIG)','launch({})'))


def test_logger_does_not_create_run_folder_before_launcher(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'stdout',io.StringIO())
    folder=tmp_path/'new_run'
    with RunLog(folder):
        assert not folder.exists()
        print('launching')
        assert not folder.exists()
        folder.mkdir(exist_ok=False)
        print('created')
    assert 'launching' in (folder/'console.raw.log').read_text(encoding='utf-8')


def test_launch_wraps_one_entry_and_restores_launcher(tmp_path,monkeypatch):
    from cef_log_output import launch
    import scripts.cef_alternating_launcher as launcher
    monkeypatch.setattr(sys,'stdout',io.StringIO())
    folder=tmp_path/'new_run'
    def run(config,test,path,resume=False):
        path.mkdir(exist_ok=False)
        print('Round 1 complete')
        return {'valid':True}
    def main(settings,argv):
        assert settings=={'seed':2025} and argv==[]
        assert launcher.run_folder({}, {}, folder)=={'valid':True}
        return 0
    monkeypatch.setattr(launcher,'run_folder',run)
    monkeypatch.setattr(launcher,'main',main)
    validate=launcher.validate
    assert launch({'seed':2025},[])==0
    assert launcher.run_folder is run and launcher.validate is validate
    assert 'Round 1 complete' in (folder/'progress.log').read_text(encoding='utf-8')


def test_startup_error_before_run_creation_is_saved(tmp_path,monkeypatch):
    from cef_log_output import launch
    import scripts.cef_alternating_launcher as launcher
    def fail(settings,argv): raise RuntimeError('WSL/ERROR_NO_SYSTEM_RESOURCES')
    monkeypatch.setattr(launcher,'main',fail)
    with pytest.raises(RuntimeError,match='ERROR_NO_SYSTEM_RESOURCES'):
        launch({'output_dir':tmp_path},[])
    logs=list((tmp_path/'startup_errors').glob('*.log'))
    assert len(logs)==1
    assert 'WSL/ERROR_NO_SYSTEM_RESOURCES' in logs[0].read_text(encoding='utf-8')


@pytest.mark.parametrize('argv',[[],['--resume','saved'],['--check'],['--pause','saved']])
def test_launcher_never_starts_a_wsl_session(tmp_path,monkeypatch,argv):
    from cef_log_output import launch
    import scripts.cef_alternating_launcher as launcher
    import cef_wsl_session
    state={'held':False,'enters':0}
    class Backend:
        def __init__(self,**kwargs):pass
        def __enter__(self):raise AssertionError('active launcher must not start WSL')
        def __exit__(self,*args):state['held']=False
    def main(settings,args):
        assert state['held'] is False
        assert args==argv
        return 0
    monkeypatch.setattr(cef_wsl_session,'NativeBackendSession',Backend)
    monkeypatch.setattr(launcher,'main',main)
    assert launch(dict(model='offline',output_dir=tmp_path),argv)==0
    assert state['held'] is False and state['enters']==0
