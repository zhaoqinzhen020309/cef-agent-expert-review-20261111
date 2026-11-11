"""Behavior acceptance of the user-approved measured research loop."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cef.alternating import AlternatingEngine
from cef.task_tools import TaskTools
from cef.task_experiments import TaskExperiments
import test_cef_alternating as alternating_tests
from test_cef_integrated import small_fixture, GEN


@pytest.mark.parametrize('rounds',[2,10])
def test_scoring_closes_round_without_an_administrative_planner(tmp_path,rounds):
    _,engine,model,_,case=alternating_tests.AlternatingTests().fixture(str(tmp_path),rounds=rounds)
    engine.config['measured_research_loop_v1']=True
    engine.frozen_config['measured_research_loop_v1']=True
    with patch.object(engine.runner,'case',side_effect=case):result=engine.run()
    assert result['status']=='finished',result['reason']
    assert len(engine.archive.of('round_complete'))==rounds
    assert not any(payload['alternating']['phase']=='round_complete' for role,payload in model.inputs if role=='planner')
    assert len(engine.archive.of('round_feedback'))==rounds
    assert sum(role=='planner' for role,payload in model.inputs)==rounds
    second=next(p for role,p in model.inputs if role=='planner' and p['alternating']['round']==2)
    assert second['previous_round_feedback']['round']==1
    assert second['previous_round_feedback']['winner']['source_hash']
    before=len(model.inputs)
    assert engine.run()['status']=='finished'
    assert len(model.inputs)==before


def fixture(tmp_path):
    config,settings=small_fixture(tmp_path,rounds=1)
    config.update(agent_task_runtime_v2=True,measured_research_loop_v1=True)
    engine=AlternatingEngine(config,None)
    source=Path(config['warm_start_source']).read_text()
    engine.register('warm_start',source)
    tools=TaskTools(tmp_path/'task',{},lambda p:dict(accepted=False),runtime_v2=True)
    proposal=dict(rules=[dict(name='uniform',source=GEN,parameters={},weight=1)],reason='test')
    tools._write('candidate.json',json.dumps(proposal))
    payload=dict(instance_seeds=[41,42],calibration_source_hash=engine.candidates()['warm_start']['source_hash'],
                 solver_seed=2025,solver_seconds=engine.runner.limit,global_round=1)
    context=dict(root=tmp_path/'data',settings=settings,payload=payload,parent_id='warm_start')
    return engine,tools,proposal,context


def test_probe_and_formal_prepare_share_exact_plan_rows_references_and_scoring(tmp_path):
    from cef.integrated_data import IntegratedDataProvider,data_report,candidate_preparation
    engine,tools,proposal,context=fixture(tmp_path)
    experiments=TaskExperiments(engine,tools,data_context=context)
    result=experiments.measure_data_candidate('candidate.json',[1])
    assert result['ok'] and result['measured_case_count']==1
    assert result['scope']=='selected subset of the fixed formal instance plan'
    assert result['cases'][0]['id']=='supplement_0001'
    rules,plan,root=candidate_preparation(context['root'],proposal,context['payload']['instance_seeds'],context['settings'])
    row=IntegratedDataProvider(context['settings'],None).prepare_case(root,plan[1],rules[0],context['settings'])
    receipt=json.loads((tools.results/result['artifact_id'].split(':',1)[1]).read_text())
    assert receipt['prepared_rows']==[row]
    assert receipt['report']['per_instance'][0]['reference_gap_pct']==pytest.approx(result['cases'][0]['reference_gap_pct'])
    before=len(engine.archive.of('execution_start'))
    repeated=experiments.measure_data_candidate('candidate.json',[1])
    assert repeated['cases']==result['cases']
    assert len(engine.archive.of('execution_start'))==before


@pytest.mark.parametrize('indices',[[],[True],[-1],[2],[1,1],['1']])
def test_measurement_rejects_ambiguous_or_nonplan_indices(tmp_path,indices):
    engine,tools,_,context=fixture(tmp_path)
    with pytest.raises(ValueError):TaskExperiments(engine,tools,data_context=context).measure_data_candidate('candidate.json',indices)


def test_unfinished_reference_returns_no_gap_and_does_not_end_data_task(tmp_path):
    from cef.reference_jobs import ReferencePending
    engine,tools,_,context=fixture(tmp_path)
    with patch('cef.integrated_data.IntegratedDataProvider.prepare_case',side_effect=ReferencePending('reference still computing')):
        result=TaskExperiments(engine,tools,data_context=context).measure_data_candidate('candidate.json',[0])
    assert result['ok'] is False
    assert result['pending_cases'][0]['id']=='supplement_0000'
    assert result['cases']==[]
    assert result.get('mean_reference_gap_pct') is None
    assert result['artifact_id'].startswith('execution:')


def test_distance_metric_rebinds_original_matrix_instead_of_rounding_training(tmp_path):
    from cef.data import bind_original_tsplib,TSPLIB_METRIC
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'data/tsplib_unrounded_original_reference_v1/tsp_manifest.json').read_text())
    before=copy.deepcopy(old['train'])
    new=bind_original_tsplib(old)
    assert TSPLIB_METRIC=='tsplib_original_distance_v1'
    assert old['train']==before==new['train']
    assert all(row['evaluation_metric']==TSPLIB_METRIC for row in new['test'])
    row=new['test'][0]
    with np.load(row['path']) as z:assert np.array_equal(z['distance_matrix'],np.floor(z['distance_matrix']))
    assert new['test'][0]['reference']==426
    assert old['test'][0]['path']!=new['test'][0]['path']


def test_data_agent_can_write_measure_read_and_submit_in_one_real_tool_session(tmp_path):
    from cef.tool_agents import run_data_task
    from test_cef_runtime_v2 import Fake,reply
    engine,tools,proposal,context=fixture(tmp_path)
    payload=dict(context['payload'],calibration_solver=engine.archive.read_source(context['payload']['calibration_source_hash']),
                 current_rules=[],instances=2,n_cities=7)
    model=Fake([
        reply('write_file',dict(relative_path='candidate.json',content=json.dumps(proposal)),1),
        reply('measure_data_candidate',dict(candidate_path='candidate.json',case_indices=[0]),2),
        reply('submit_candidate',dict(path='candidate.json'),3)])
    result=run_data_task(engine,model,payload,context['root'],context['settings'])
    assert result==proposal
    measured=json.loads(model.messages[2][-1]['content'])
    assert measured['ok'] and measured['cases'][0]['reference_gap_pct'] is not None
    assert measured['evidence_id'] in engine.known_evidence_ids()
    completed=engine.archive.of('data_candidate_measurement')
    assert len(completed)==1
    assert engine.archive.of('agent_tool_evidence')[-1]['tool']=='measure_data_candidate'


def test_measurement_cache_does_not_claim_one_worker_is_eight_workers(tmp_path):
    engine,tools,_,context=fixture(tmp_path)
    engine.workers=2
    experiments=TaskExperiments(engine,tools,data_context=context)
    experiments.measure_data_candidate('candidate.json',[0])
    first=len(engine.archive.of('execution_start'))
    experiments.measure_data_candidate('candidate.json',[0,1])
    assert len(engine.archive.of('execution_start'))==first+2
    observations=engine.shared_measurements.archive.of('observation')
    assert {r['identity']['contract']['active_parallel_cases'] for r in observations}=={1,2}


def test_context_pressure_prunes_old_tool_output_with_readable_full_evidence(tmp_path):
    from cef.tool_session import ToolSession
    from test_cef_runtime_v2 import Fake,reply
    tools=TaskTools(tmp_path,{'large.json':dict(type='json',value={'text':'x'*5000})},lambda p:dict(accepted=True,value={}),runtime_v2=True)
    calls=[reply('read_artifact',dict(artifact_id='large.json'),1),
           reply('write_file',dict(relative_path='small.txt',content='done'),2),
           reply('submit_candidate',dict(content='{}'),3)]
    calls[1]['usage']=dict(prompt_tokens=900,completion_tokens=10)
    model=Fake(calls);model.config=dict(tool_max_output_tokens=200)
    session=ToolSession(tmp_path,model,tools.schemas,tools.dispatch,role='planner',system='goal',payload={},
        runtime_v2=True,context_window_tokens=1000,context_storage=tools.store_context_result)
    assert session.run()=={}
    assert session.archive.of('context_pruned')
    pruned=next(m for m in model.messages[-1] if m.get('tool_call_id')=='1')
    pointer=json.loads(pruned['content'])
    full=tools.dispatch('read_artifact',dict(artifact_id=pointer['artifact_id']))
    assert 'x'*5000 in full['content']
    assert len(pruned['content'])<1000
    assert not session.archive.of('finalization_started')


def test_original_distance_final_report_has_original_fields_and_zero_gap(tmp_path):
    from cef.data import TSPLIB_METRIC
    from cef.reporting import final_test
    from cef.store import Archive
    from cef.patch import digest
    path=tmp_path/'integer.npz';np.savez(path,distance_matrix=np.array([[0,2,2],[2,0,2],[2,2,0]]))
    rows=[dict(id='triangle',path=str(path),reference=6,evaluation_metric=TSPLIB_METRIC)]
    archive=Archive(tmp_path/'search')
    archive.add('manifest',dict(task='tsp',test_repeats=1,data_manifest={'test':rows},diagnostic_run=True,solver_time_limit_s=.5,evaluation_workers=1))
    source='def solve(instance,time_limit_s,seed):\n    return {"solution":[0,1,2]}\n'
    with patch('cef.reporting.verify_export',return_value=(archive,dict(status='finished',best=dict(source_hash=digest(source))),source)),patch('cef.reporting.audit_test'):
        result=final_test(tmp_path/'search',tmp_path/'test',rows,1,{'triangle:0':2025})
    assert result['mean_original_distance_gap_pct']==0
    assert result['evaluation_metric']==TSPLIB_METRIC
    assert 'mean_unrounded_reference_gap_pct' not in result


def test_crash_after_feedback_commit_resumes_without_any_new_model_request(tmp_path):
    from test_cef_alternating import config,AlgorithmFake,epoch_rows
    c=config(tmp_path,1);c.update(measured_research_loop_v1=True,tool_context_window_tokens=1000000)
    model=AlgorithmFake();provider=lambda e,n,p:epoch_rows(n)
    engine=AlternatingEngine(c,model,provider);add=engine.archive.add
    def crash(kind,value):
        result=add(kind,value)
        if kind=='round_feedback':raise RuntimeError('simulated interruption after durable feedback')
        return result
    with patch.object(engine.archive,'add',side_effect=crash):first=engine.run()
    assert first['status']=='failed'
    calls=len(model.inputs)
    restored=AlternatingEngine(c,model,provider)
    result=restored.run()
    assert result['status']=='finished',result['reason']
    assert len(model.inputs)==calls
    assert len(restored.archive.of('round_feedback'))==len(restored.archive.of('round_complete'))==1


def test_submission_rechecks_coordinates_against_previously_measured_version(tmp_path):
    from cef.tool_agents import run_data_task
    from cef.tool_session import TaskBlocked
    from test_cef_runtime_v2 import Fake,reply
    engine,tools,proposal,context=fixture(tmp_path)
    TaskExperiments(engine,tools,data_context=context).measure_data_candidate('candidate.json',[0])
    payload=dict(context['payload'],calibration_solver=engine.archive.read_source(context['payload']['calibration_source_hash']),
                 current_rules=[],instances=2,n_cities=7)
    model=Fake([reply('submit_candidate',dict(content=json.dumps(proposal)),1),
        reply('report_blocker',dict(reason='generator is not reproducible',evidence=[]),2)])
    changed=np.random.default_rng(999).random((7,2)).tolist()
    with patch('cef.tool_agents.generate_coordinates',return_value=changed),pytest.raises(TaskBlocked):
        run_data_task(engine,model,payload,context['root'],context['settings'])
    result=json.loads(model.messages[-1][-1]['content'])
    assert result['accepted'] is False
    assert 'measured coordinates' in result['error']
