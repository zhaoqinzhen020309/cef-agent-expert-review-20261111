import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cef.store import Archive
from cef.tool_session import ToolSession
from resumeCEFTools import authorize


@pytest.mark.parametrize('role,relative,provider_dir',[
    ('planner','search/agent_tasks/step_0_planner','search'),
    ('data_designer','rounds/001/data/agent_task','data_agent')])
@pytest.mark.parametrize('context_capacity',[None,1000000])
def test_explicit_retry_selects_correct_provider_and_preserves_failed_records(tmp_path,role,relative,provider_dir,context_capacity):
    root=tmp_path
    provider=Archive(root/provider_dir)
    task=root/relative
    model=SimpleNamespace(config={},archive=provider)
    session=ToolSession(task,model,[],None,role=role,system='fixture',payload={},max_turns=1,context_window_tokens=context_capacity)
    session.archive.add('model_started',dict(turn=1,provider_start_seq=0))
    request=provider.add('provider_request',dict(role=role,messages=session.messages,request_parameters=dict(tools=[])))
    provider.add('provider_response',dict(role=role,request_seq=request['seq'],message=None,
        error_chain=['APIConnectionError: offline']))
    config=dict(agent_tool_loop_v1=True,run_dir=str(root/'search'),implementation_source_manifest=[])
    (root/'search_config.json').write_text(json.dumps(config))
    (root/'search').mkdir(exist_ok=True)
    before=provider.path.read_bytes()
    with patch('resumeCEFTools.validate',side_effect=lambda value:value):
        authorize(root,relative,check=True)
        assert not Archive(task/'session').of('retry_authorized')
        authorize(root,relative)
    assert provider.path.read_bytes()==before
    assert len(Archive(task/'session').of('retry_authorized'))==1
    assert not (root/'search/owner.lock').exists()


def test_retry_rejects_task_outside_run_before_access(tmp_path):
    config=dict(agent_tool_loop_v1=True,run_dir=str(tmp_path/'search'),implementation_source_manifest=[])
    (tmp_path/'search_config.json').write_text(json.dumps(config))
    (tmp_path/'search').mkdir()
    with patch('resumeCEFTools.validate',side_effect=lambda value:value):
        with pytest.raises(ValueError):authorize(tmp_path,'../other-task')


def test_live_owner_is_rejected_before_any_archive_is_opened(tmp_path):
    import os
    config=dict(agent_tool_loop_v1=True,run_dir=str(tmp_path/'search'),implementation_source_manifest=[])
    (tmp_path/'search_config.json').write_text(json.dumps(config))
    (tmp_path/'search').mkdir()
    (tmp_path/'search/owner.lock').write_text(str(os.getpid()))
    with patch('resumeCEFTools.validate',side_effect=lambda value:value),patch('resumeCEFTools.Archive') as archive:
        with pytest.raises(FileExistsError):authorize(tmp_path,'search/agent_tasks/step_0_planner',True)
        archive.assert_not_called()
