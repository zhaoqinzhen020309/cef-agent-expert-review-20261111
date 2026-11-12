import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from cef.task_experiments import TaskExperiments
from cef.task_tools import TaskTools
from cef.tool_agents import run_data_task
from cef.tool_session import ToolSession
from test_cef_measured_loop import fixture
from test_cef_runtime_v2 import Fake,reply


def enabled_fixture(tmp_path):
    engine,tools,proposal,context=fixture(tmp_path)
    engine.config['exploration_efficiency_v1']=True
    engine.frozen_config['exploration_efficiency_v1']=True
    return engine,tools,proposal,context


def data_payload(engine,context):
    payload=dict(context['payload'],calibration_solver=engine.archive.read_source(context['payload']['calibration_source_hash']),
                 current_rules=[],instances=2,n_cities=7,previous_results=[dict(valid=True,mean_reference_gap_pct=0.)])
    payload['calibration_conditions']=dict(engine.shared_measurements.contract(engine),active_parallel_cases=min(engine.workers,2))
    return payload


def test_data_role_rejects_old_generated_probe_route_and_can_keep_current(tmp_path):
    engine,_,_,context=enabled_fixture(tmp_path)
    model=Fake([reply('run_solver_experiment',dict(source_id='parent.py',generator_id='generator_0.py',seeds=[1]),1),
        reply('submit_candidate',dict(content=json.dumps(dict(action='keep_current',reason='no supported replacement',evidence_ids=[]))),2)])
    result=run_data_task(engine,model,data_payload(engine,context),context['root'],context['settings'])
    assert result['action']=='keep_current'
    error=json.loads(model.messages[1][-1]['content'])
    assert error['ok'] is False and 'measure_data_candidate' in error['error']


def test_generator_source_artifact_is_frozen_for_measurement_and_submission(tmp_path):
    engine,tools,proposal,context=enabled_fixture(tmp_path)
    tools._write('generate.py',proposal['rules'][0]['source'])
    aliased=copy.deepcopy(proposal);aliased['rules'][0]['source_artifact']='generate.py';del aliased['rules'][0]['source']
    tools._write('candidate.json',json.dumps(aliased))
    result=TaskExperiments(engine,tools,data_context=context).measure_data_candidate('candidate.json',[0])
    assert result['ok']
    evidence=json.loads((tools.results/result['artifact_id'].split(':',1)[1]).read_text())
    assert evidence['rules'][0]['source']==proposal['rules'][0]['source']


def test_inline_check_can_save_and_execute_in_one_call(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:dict(accepted=False),runtime_v2=True)
    tools.enable_efficiency('planner')
    result=tools.dispatch('run_python',dict(code='result=3+4',save_as='check.py',input_artifact_ids=[]))
    assert result['ok'] and result['result']==7
    assert (tools.workspace/'check.py').read_text()=='result=3+4'


def test_whole_plan_better_data_completes_task_but_subset_does_not(tmp_path):
    engine,_,proposal,context=enabled_fixture(tmp_path)
    payload=data_payload(engine,context)
    model=Fake([reply('write_file',dict(relative_path='candidate.json',content=json.dumps(proposal)),1),
        reply('measure_data_candidate',dict(candidate_path='candidate.json',case_indices=[0]),2),
        reply('measure_data_candidate',dict(candidate_path='candidate.json',case_indices=[0,1]),3)])
    result=run_data_task(engine,model,payload,context['root'],context['settings'])
    assert result['rules']==proposal['rules']
    assert len(model.messages)==3
    assert not json.loads(model.messages[2][-1]['content']).get('task_completion')
    assert engine.archive.of('agent_task_complete')


def test_batch_runs_predeclared_data_experiments_and_exposes_real_results(tmp_path):
    engine,tools,_,context=enabled_fixture(tmp_path)
    tools.enable_efficiency('data_designer')
    experiments=TaskExperiments(engine,tools,data_context=context)
    tools.experiment_batch=experiments.run_batch
    result=tools.dispatch('run_experiment_batch',dict(question='compare fixed subsets',jobs=[
        dict(tool='measure_data_candidate',arguments=dict(candidate_path='candidate.json',case_indices=[0])),
        dict(tool='measure_data_candidate',arguments=dict(candidate_path='candidate.json',case_indices=[1]))],
        handoff='Candidate in candidate.json; use actual results to decide next design.'))
    assert result['ok'] and len(result['experiments'])==2
    assert all(e['result']['cases'][0]['reference_gap_pct'] is not None for e in result['experiments'])
    assert result['context_checkpoint']['artifact_id'].startswith('execution:')


def test_pending_retry_does_not_mislabel_old_single_case_as_parallel(tmp_path):
    from cef.reference_jobs import ReferencePending
    from cef.integrated_data import IntegratedDataProvider
    engine,tools,_,context=enabled_fixture(tmp_path);engine.workers=2
    original=IntegratedDataProvider.prepare_case;pending=[True]
    def prepare(self,root,item,rule,settings):
        if item['index']==1 and pending[0]:raise ReferencePending('not complete yet')
        return original(self,root,item,rule,settings)
    exp=TaskExperiments(engine,tools,data_context=context)
    with patch.object(IntegratedDataProvider,'prepare_case',prepare):
        first=exp.measure_data_candidate('candidate.json',[0,1]);pending[0]=False
        second=exp.measure_data_candidate('candidate.json',[0,1])
    assert not first['ok'] and second['ok']
    assert all(c['observation_contract']['active_parallel_cases']==2 for c in second['cases'])


def test_plain_stop_also_runs_context_pressure_management(tmp_path):
    tools=TaskTools(tmp_path,{'large.json':dict(type='json',value={'text':'x'*5000})},lambda p:dict(accepted=True,value={}),runtime_v2=True)
    stop=dict(message=dict(role='assistant',content='continue investigating'),finish_reason='stop',usage={'prompt_tokens':2000})
    model=Fake([reply('read_artifact',dict(artifact_id='large.json'),1),stop,
        reply('submit_candidate',dict(content='{}'),3)])
    model.config=dict(tool_max_output_tokens=100)
    session=ToolSession(tmp_path,model,tools.schemas,tools.dispatch,role='planner',system='goal',payload={},
        runtime_v2=True,context_window_tokens=1000,context_storage=tools.store_context_result)
    assert session.run()=={}
    assert session.archive.of('context_pruned') or session.archive.of('context_pressure')


def test_batch_failed_job_stays_failed_with_handoff_and_other_jobs_continue(tmp_path):
    engine,tools,_,context=enabled_fixture(tmp_path)
    experiments=TaskExperiments(engine,tools,data_context=context)
    result=experiments.run_batch('check independent subsets',[
        dict(tool='measure_data_candidate',arguments=dict(candidate_path='candidate.json',case_indices=[999])),
        dict(tool='measure_data_candidate',arguments=dict(candidate_path='candidate.json',case_indices=[0]))],
        handoff='Inspect both actual results before choosing the next investigation.')
    assert not result['ok'] and len(result['experiments'])==2
    assert not result['experiments'][0]['result']['ok'] and result['experiments'][1]['result']['ok']
    assert result['context_checkpoint']['artifact_id']


def test_batch_handoff_restarts_same_task_and_resumes_without_reexecution(tmp_path):
    engine,tools,_,context=enabled_fixture(tmp_path)
    tools.enable_efficiency('data_designer')
    tools.experiment_batch=TaskExperiments(engine,tools,data_context=context).run_batch
    tools.submit=lambda path:dict(accepted=True,value={'finished':True})
    model=Fake([reply('run_experiment_batch',dict(question='measure current candidate',jobs=[
        dict(tool='measure_data_candidate',arguments=dict(candidate_path='candidate.json',case_indices=[0]))],
        handoff='See actual measurement and saved candidate.json.'),1),
        reply('submit_candidate',dict(content='{}'),2)])
    arguments=dict(role='data_designer',system='goal',payload={},runtime_v2=True,
        checkpoint_tools=['checkpoint_context','run_experiment_batch'])
    session=ToolSession(tmp_path/'task',model,tools.schemas,tools.dispatch,**arguments)
    assert session.run()=={'finished':True}
    assert session.archive.of('context_restarted')
    count=len(engine.archive.of('execution_start'))
    assert ToolSession(tmp_path/'task',model,tools.schemas,tools.dispatch,**arguments).run()=={'finished':True}
    assert len(engine.archive.of('execution_start'))==count and len(model.messages)==2


def test_verified_completion_does_not_execute_later_side_effect_calls(tmp_path):
    calls=[]
    def execute(name,args):
        calls.append(name)
        return dict(ok=True,task_completion=dict(accepted=True,value={'rules':[]}))
    first=reply('measure_data_candidate',{},1)
    first['message']['tool_calls'].append(reply('write_file',dict(relative_path='later.py',content='bad'),2)['message']['tool_calls'][0])
    model=Fake([first]);schemas=[{'function':{'name':name}} for name in ('measure_data_candidate','write_file')]
    options=dict(role='data_designer',system='goal',payload={},runtime_v2=True,completion_tools=['measure_data_candidate'])
    session=ToolSession(tmp_path,model,schemas,execute,**options)
    assert session.run()=={'rules':[]}
    assert calls==['measure_data_candidate']
    assert session.archive.of('tool_completed')[-1]['result']['not_executed']
    assert ToolSession(tmp_path,model,schemas,execute,**options).run()=={'rules':[]}
    assert calls==['measure_data_candidate']


@pytest.mark.parametrize('rounds',[2,10])
def test_keep_current_runs_all_algorithm_rounds_without_new_reference_work(tmp_path,rounds):
    from test_cef_integrated import small_fixture
    from test_cef_alternating import AlgorithmFake
    from cef.alternating import AlternatingEngine,DataRoundProvider
    config,settings=small_fixture(tmp_path,rounds=rounds)
    config.update(measured_research_loop_v1=True,exploration_efficiency_v1=True)
    class Data:
        calls=0
        def call(self,role,payload):
            self.calls+=1
            return dict(action='keep_current',reason='retain validated current set',evidence_ids=[])
    data=Data();algorithm=AlgorithmFake()
    engine=AlternatingEngine(config,algorithm,DataRoundProvider(settings,data))
    original_paths=[r['path'] for r in config['data_manifest']['train']]
    with patch('cef.integrated_data.IntegratedDataProvider.prepare_case',side_effect=AssertionError('no new data proposed')):
        result=engine.run()
    assert result['status']=='finished',result['reason']
    assert data.calls==algorithm.edits==rounds
    assert len(engine.archive.of('round_complete'))==rounds
    assert all(r['selection']['decision']=='keep_current' for r in engine.archive.of('data_round_complete'))
    assert [r['path'] for r in engine.rows]==original_paths


def test_repeated_instance_plan_is_rejected_before_automatic_task_completion(tmp_path):
    engine,tools,proposal,context=enabled_fixture(tmp_path)
    proposal['rules'][0]['source']=proposal['rules'][0]['source'].replace('default_rng(seed)','default_rng(123)')
    tools._write('candidate.json',json.dumps(proposal))
    context['payload'].update(previous_results=[dict(valid=True,mean_reference_gap_pct=0.)],
        calibration_conditions=dict(engine.shared_measurements.contract(engine),active_parallel_cases=1))
    result=TaskExperiments(engine,tools,data_context=context).measure_data_candidate('candidate.json',[0,1])
    assert not result['ok'] and '_goal_proposal' not in result
    assert 'duplicate' in result['failed_cases'][0]['error']


def test_explicit_submission_rejects_duplicate_generated_instances_and_allows_repair(tmp_path):
    engine,_,proposal,context=enabled_fixture(tmp_path)
    proposal['rules'][0]['source']=proposal['rules'][0]['source'].replace('default_rng(seed)','default_rng(123)')
    model=Fake([reply('submit_candidate',dict(content=json.dumps(proposal)),1),
        reply('submit_candidate',dict(content=json.dumps(dict(action='keep_current',reason='duplicate generator; retain prior set',evidence_ids=[]))),2)])
    result=run_data_task(engine,model,data_payload(engine,context),context['root'],context['settings'])
    assert result['action']=='keep_current'
    assert 'duplicate' in json.loads(model.messages[1][-1]['content'])['error']


def test_partial_batch_resume_preserves_full_contract_without_repeating_physical_work(tmp_path):
    engine,_,_,_=enabled_fixture(tmp_path);engine.workers=2
    rows=engine.config['data_manifest']['train']
    original=engine.archive.add
    def interrupt(kind,value):
        saved=original(kind,value)
        if kind=='execution':raise RuntimeError('partial view commit')
        return saved
    with patch.object(engine.archive,'add',side_effect=interrupt),pytest.raises(RuntimeError):
        engine.measure_rows('warm_start',rows,'partial')
    before=len(engine.shared_measurements.archive.of('observation'))
    result=engine.measure_rows('warm_start',rows,'partial')
    observations={r['key']:r for r in engine.shared_measurements.archive.of('observation')}
    assert all(observations[r['observation_id']]['identity']['contract']['active_parallel_cases']==2 for r in result)
    assert len(engine.shared_measurements.archive.of('observation'))==len(rows)
    assert before<=len(rows)


@pytest.mark.parametrize('case',['different_conditions','no_improvement'])
def test_full_measurement_without_matching_improvement_keeps_agent_task_open(tmp_path,case):
    engine,_,proposal,context=enabled_fixture(tmp_path)
    payload=data_payload(engine,context)
    if case=='different_conditions':payload['calibration_conditions']['active_parallel_cases']=99
    else:payload['previous_results'][0]['mean_reference_gap_pct']=1e9
    model=Fake([reply('write_file',dict(relative_path='candidate.json',content=json.dumps(proposal)),1),
        reply('measure_data_candidate',dict(candidate_path='candidate.json',case_indices=[0,1]),2),
        reply('submit_candidate',dict(content=json.dumps(dict(action='keep_current',reason='retain old set',evidence_ids=[]))),3)])
    assert run_data_task(engine,model,payload,context['root'],context['settings'])['action']=='keep_current'
    assert not json.loads(model.messages[2][-1]['content']).get('task_completion')


def test_published_worker_examples_execute_under_actual_worker_contract(tmp_path):
    from cef.task_worker import capability_manifest
    tools=TaskTools(tmp_path,{'training_cases.json':dict(type='json',value=[{'id':'a','instance':{}}]),
        'parent.py':dict(type='text',value='def helper(x):\n    return x+1\n')},lambda p:dict(accepted=False),runtime_v2=True)
    tools.enable_efficiency('planner')
    results=[tools.dispatch('run_python',example) for example in capability_manifest()['examples']]
    assert all(r['ok'] for r in results)
    assert results[0]['result']['count']==1 and 'helper' in results[1]['result']
