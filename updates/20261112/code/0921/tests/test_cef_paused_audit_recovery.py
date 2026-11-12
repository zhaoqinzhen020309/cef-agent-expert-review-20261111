import json
from pathlib import Path
import pytest
from cef.cli import freeze_runtime
from cef.data import receipt
from cef.data_recovery import recover_run
from cef.store import Archive
from scripts.cef_alternating_launcher import build_auto_config
from runCEFAuto import CONFIG

def make_paused_source(tmp_path,external_dependency=None):
    source=tmp_path/'paused';source.mkdir()
    c,test=build_auto_config(CONFIG,source,2025)
    if external_dependency is not None:c['reference_backend_manifest']['files'].append(receipt(external_dependency))
    (source/'search_config.json').write_text(json.dumps(c),encoding='utf-8')
    (source/'test_config.json').write_text(json.dumps(test),encoding='utf-8')
    archive=Archive(source/'search');archive.add('manifest',c)
    data=source/'rounds/001/data';data.mkdir(parents=True)
    archive.add('data_round_start',dict(round=1,path=str(data)))
    epoch=source/'search/epoch.json';epoch.write_text('{}')
    archive.add('training_epoch',dict(round=1,artifact=receipt(epoch)))
    (source/'search/result.json').write_text(json.dumps(dict(status='paused',reason='user pause')))
    (source/'search/pause.request').write_text('pause')
    (source/'status_paused.json').write_text(json.dumps(dict(status='paused')))
    Archive(source/'data_agent').add('provider_manifest',dict(fixture=True))
    proposal=dict(rules=[dict(name='fixture',source='def generate(n_cities,seed,parameters):\n return []',parameters={},weight=1)])
    (data/'proposal.json').write_text(json.dumps(proposal))
    (data/'result.json').write_text(json.dumps(dict(historical_selection='old')))
    Archive(data/'agent_task/session').add('submitted',dict(value=proposal))
    task=source/'search/agent_tasks/step_0_planner';task.mkdir(parents=True)
    (task/'bound_materials.json').write_text('{"task":"frozen"}')
    session=Archive(task/'session');session.add('paused',dict(reason='saved after completed effects'))
    freeze_runtime(c['run_dir'],c)
    return source,task


def test_paused_algorithm_stage_can_fork_audit_repair_without_replaying_data(tmp_path):
    source,task=make_paused_source(tmp_path);target=tmp_path/'audit_fixed'
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert recover_run(source,target)==target
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert (target/'search/pause.request').is_file()
    assert (target/'search/agent_tasks/step_0_planner/bound_materials.json').read_bytes()==(task/'bound_materials.json').read_bytes()
    result=json.loads((target/'audit_recovery_result.json').read_text())
    assert result['status']=='paused' and result['model_requests_during_recovery']==0
    assert json.loads((target/'rounds/001/data/result.json').read_text())=={'historical_selection':'old'}


def test_audit_only_repair_cannot_refresh_changed_external_backend_dependencies(tmp_path):
    dependency=tmp_path/'frozen_backend.dll';dependency.write_bytes(b'original frozen dependency')
    source,_=make_paused_source(tmp_path,dependency)
    dependency.write_bytes(b'replaced dependency')
    target=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='reference backend dependency changed'):
        recover_run(source,target)
    assert not target.exists()


def test_failed_native_transport_can_fork_without_replaying_active_epoch(tmp_path):
    source,task=make_paused_source(tmp_path)
    (source/'search/result.json').write_text(json.dumps(dict(status='failed',reason='APIConnectionError: Connection error.')))
    (source/'search/pause.request').unlink()
    (source/'status_paused.json').unlink()
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    target=tmp_path/'transport_fixed'
    assert recover_run(source,target)==target
    assert all(p.read_bytes()==raw for p,raw in before.items())
    result=json.loads((target/'transport_recovery_result.json').read_text())
    assert result['current_round']==1 and result['model_requests_during_recovery']==0
    assert (target/'search/agent_tasks/step_0_planner/bound_materials.json').read_bytes()==(task/'bound_materials.json').read_bytes()
    assert json.loads((target/'rounds/001/data/result.json').read_text())=={'historical_selection':'old'}
