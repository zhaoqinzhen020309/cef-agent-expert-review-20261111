"""External benchmark results must not become research tool materials."""
import json
from test_cef_measured_loop import fixture
from cef.tool_agents import _training_materials
from cef.task_tools import TaskTools

def test_external_test_result_not_in_agent_bindings_or_tool_file_access(tmp_path):
    engine,_,_,_=fixture(tmp_path)
    directory=tmp_path/'round_benchmarks/001';directory.mkdir(parents=True)
    path=directory/'summary.json';path.write_text(json.dumps({'marker':'PRIVATE_TSPLIB_MARKER_7581','gap':.205606}))
    bindings=_training_materials(engine)
    assert 'PRIVATE_TSPLIB_MARKER_7581' not in json.dumps(bindings)
    tools=TaskTools(tmp_path/'isolated_task',bindings,lambda p:{'accepted':False},runtime_v2=True)
    assert 'round_benchmarks' not in json.dumps(tools.dispatch('list_artifacts',{}))
    for identifier in (str(path.resolve()),'../../round_benchmarks/001/summary.json','execution:../../round_benchmarks/001/summary.json'):
        assert tools.dispatch('read_artifact',{'artifact_id':identifier})['ok'] is False
    result=tools.dispatch('run_python',{'code':'result=open(parameters["path"]).read()','parameters':{'path':str(path)},'input_artifact_ids':[]})
    assert result['ok'] is False
