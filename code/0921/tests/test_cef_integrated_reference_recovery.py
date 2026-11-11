import json
from unittest.mock import patch
from pathlib import Path
from cef.cli import freeze_runtime
from cef.config import validate
from cef.store import Archive,canonical
from cef.data_recovery import recover_run
from scripts.cef_alternating_launcher import build_auto_config
from runCEFAuto import CONFIG

def test_integrated_repair_fork_preserves_accepted_submission_without_model_or_reference_calls(tmp_path):
    source=tmp_path/'original';source.mkdir();target=tmp_path/'repaired'
    config,test=build_auto_config(CONFIG,source,2025)
    (source/'search_config.json').write_text(json.dumps(config),encoding='utf-8')
    (source/'test_config.json').write_text(json.dumps(test),encoding='utf-8')
    archive=Archive(source/'search');archive.add('manifest',config)
    data=source/'rounds/001/data';data.mkdir(parents=True)
    archive.add('data_round_start',dict(round=1,path=str(data)))
    (source/'search/result.json').write_text(json.dumps(dict(status='waiting_reference')))
    Archive(source/'data_agent').add('provider_manifest',dict(diagnostic=True))
    proposal=dict(rules=[dict(name='fixed',source='def generate(n_cities,seed,parameters):\n return [[i/n_cities,0.] for i in range(n_cities)]\n',parameters={},weight=1)])
    (data/'proposal.json').write_text(json.dumps(proposal),encoding='utf-8')
    Archive(data/'agent_task/session').add('submitted',dict(value=proposal))
    freeze_runtime(config['run_dir'],config)
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    with patch('cef.data_recovery.DataDesigner.recover',side_effect=AssertionError('wrong legacy pipeline')):
        assert recover_run(source,target)==target
    assert all(p.read_bytes()==raw for p,raw in before.items())
    updated=json.loads((target/'search_config.json').read_text())
    assert validate(updated)['run_dir']==str(target/'search')
    assert Archive(target/'search').of('manifest')[0]==updated
    result=json.loads((target/'recovery_result.json').read_text())
    assert result['status']=='ready_to_resume' and result['model_requests_during_recovery']==0
    assert json.loads((target/'rounds/001/data/proposal.json').read_text())==proposal
    # A backend replacement may repair a previously repaired run. It must
    # retain both generations of provenance without overwriting prior receipts.
    another=tmp_path/'repaired_again'
    assert recover_run(target,another)==another
    assert list(another.glob('recovery_result_previous_*.json'))
