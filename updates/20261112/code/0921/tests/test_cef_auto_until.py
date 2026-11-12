from unittest.mock import patch
from cef.store import Archive
from scripts.cef_alternating_launcher import run_until


def test_continue_from_one_to_ten_without_restarting(tmp_path):
    a=Archive(tmp_path/'search');a.add('round_complete',dict(round=1))
    calls=[]
    def run(c,t,folder,resume,operation):
        assert resume and folder==tmp_path
        calls.append(operation)
        if operation=='next':
            a=Archive(folder/'search');a.add('round_complete',dict(round=len(a.of('round_complete'))+1))
    with patch('scripts.cef_alternating_launcher.run_folder',side_effect=run):
        run_until({'manual_rounds':True},{},tmp_path,10)
    assert calls==['next']*9+['evaluate']
    assert len(Archive(tmp_path/'search').of('round_complete'))==10


def test_pending_pause_does_not_start_more_rounds(tmp_path):
    with patch('scripts.cef_alternating_launcher.run_folder') as run:
        run_until({'manual_rounds':True},{},tmp_path,10)
    assert run.call_count==1
