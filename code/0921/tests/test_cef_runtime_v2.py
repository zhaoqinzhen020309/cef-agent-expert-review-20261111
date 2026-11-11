import copy,json
from pathlib import Path
import numpy as np
import pytest

from cef.store import Archive
from cef.tool_session import ToolSession,TaskPaused,TaskBlocked
from cef.task_tools import TaskTools
from cef.tool_agents import _training_materials
from cef.alternating import AlternatingEngine
from test_cef_integrated import small_fixture,GEN


def reply(name,args,number):
    return dict(message=dict(role='assistant',content=None,tool_calls=[dict(id=str(number),type='function',
        function=dict(name=name,arguments=json.dumps(args)))]),finish_reason='tool_calls')


class Fake:
    config={}
    def __init__(self,replies):self.replies=iter(replies);self.messages=[]
    def exchange(self,messages,tools,role):
        self.messages.append(copy.deepcopy(messages));return next(self.replies)


def session(root,model,tools,**kwargs):
    return ToolSession(root,model,tools.schemas,tools.dispatch,role='planner',system='task',payload={'goal':'deliver'},
        max_turns=None,runtime_v2=True,**kwargs)


def test_free_loop_beyond32_keeps_tools_and_ends_on_valid_delivery(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:dict(accepted=True,value=json.loads(p.read_text())),runtime_v2=True)
    calls=[reply('write_file',dict(relative_path='progress.txt',content=str(i)),i) for i in range(40)]
    calls.append(reply('submit_candidate',dict(content='{"done":true}'),40))
    model=Fake(calls)
    s=session(tmp_path,model,tools)
    assert s.run()=={'done':True}
    assert len(model.messages)==41
    assert not s.archive.of('finalization_started')


def test_low_level_constructor_has_no_hidden_default_turn_limit(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:dict(accepted=True,value={}),runtime_v2=True)
    model=Fake([reply('submit_candidate',dict(content='{}'),1)])
    s=ToolSession(tmp_path,model,tools.schemas,tools.dispatch,role='planner',system='task',payload={},runtime_v2=True)
    assert s.max_turns is None
    assert s.run()=={}


def test_checkpoint_restarts_context_but_keeps_goal_and_persisted_work(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:dict(accepted=True,value=json.loads(p.read_text())),runtime_v2=True)
    model=Fake([reply('write_file',dict(relative_path='proof.txt',content='observation'),1),
        reply('checkpoint_context',dict(summary='Check completed; proof.txt contains the observation. Submit next.'),2),
        reply('submit_candidate',dict(content='{}'),3)])
    s=session(tmp_path,model,tools)
    assert s.run()=={}
    assert len(model.messages[-1])==3
    assert 'deliver' in model.messages[-1][1]['content']
    assert 'proof.txt' in model.messages[-1][-1]['content']
    assert len(s.archive.of('context_restarted'))==1
    assert session(tmp_path,Fake([]),tools).run()=={}


def test_pause_before_next_tool_resumes_without_repeating_first(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:dict(accepted=True,value={}),runtime_v2=True)
    result=reply('write_file',dict(relative_path='a.txt',content='one'),1)
    result['message']['tool_calls']+=reply('submit_candidate',dict(content='{}'),2)['message']['tool_calls']
    model=Fake([result]);paused=[False];calls=[]
    dispatch=tools.dispatch
    def work(name,args):
        calls.append(name);value=dispatch(name,args)
        if name=='write_file':paused[0]=True
        return value
    s=ToolSession(tmp_path,model,tools.schemas,work,role='planner',system='task',payload={},max_turns=None,runtime_v2=True,pause_check=lambda:paused[0])
    with pytest.raises(TaskPaused):s.run()
    paused[0]=False
    resumed=ToolSession(tmp_path,Fake([]),tools.schemas,work,role='planner',system='task',payload={},max_turns=None,runtime_v2=True,pause_check=lambda:paused[0])
    assert resumed.run()=={}
    assert calls==['write_file','submit_candidate']


def test_explicit_blocker_is_not_an_accepted_candidate(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:pytest.fail('must not submit'),runtime_v2=True)
    model=Fake([reply('report_blocker',dict(reason='Required measurement is unavailable',evidence=['execution:x']),1)])
    s=session(tmp_path,model,tools)
    with pytest.raises(TaskBlocked,match='measurement'):s.run()
    assert not s.archive.of('submitted')


def test_plain_completion_without_delivery_gets_actual_postcondition_feedback(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:dict(accepted=True,value={}),runtime_v2=True)
    model=Fake([dict(message=dict(role='assistant',content='I have a plan.'),finish_reason='stop'),
        reply('submit_candidate',dict(content='{}'),2)])
    s=session(tmp_path,model,tools)
    assert s.run()=={}
    assert 'not submitted' in model.messages[1][-1]['content']
    assert not s.archive.of('blocked')
    assert len(s.archive.of('delivery_check_failed'))==1


def test_explicit_provider_refusal_is_not_retried(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:{},runtime_v2=True)
    model=Fake([dict(message=dict(role='assistant',content=None,refusal='provider refusal'),finish_reason='stop')])
    with pytest.raises(TaskBlocked,match='provider refusal'):session(tmp_path,model,tools).run()
    assert len(model.messages)==1


def test_checkpoint_cold_resume_replays_no_completed_effects(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:dict(accepted=True,value={}),runtime_v2=True)
    first=Fake([reply('write_file',dict(relative_path='evidence.txt',content='kept'),1),
        reply('checkpoint_context',dict(summary='Evidence saved; submit next.'),2)])
    s=session(tmp_path,first,tools)
    add=s.archive.add
    def crash(kind,data):
        if kind=='model_started' and data['turn']==3:raise KeyboardInterrupt()
        return add(kind,data)
    s.archive.add=crash
    with pytest.raises(KeyboardInterrupt):s.run()
    next_model=Fake([reply('submit_candidate',dict(content='{}'),3)])
    assert session(tmp_path,next_model,tools).run()=={}
    assert len(next_model.messages[0])==3
    journal=Archive(tmp_path/'session')
    assert len(journal.of('context_restarted'))==1
    assert len([r for r in journal.of('tool_completed') if r['call']['function']['name']=='write_file'])==1


@pytest.mark.parametrize('rounds',[2,10])
def test_outer_pause_then_complete_v2_rounds(tmp_path,rounds):
    from cef.alternating import DataRoundProvider
    from test_cef_alternating import AlgorithmFake
    class Roles:
        config={}
        def __init__(self):self.legacy=AlgorithmFake();self.n=0;self.buffer=''
        def exchange(self,messages,tools,role):
            self.n+=1
            if role=='editor' and getattr(self,'editor_payload',None) is not None:
                content=json.loads(messages[-1]['content'])
                payload=self.editor_payload;payload['parent_code']=content['content'];self.editor_payload=None
                return reply('submit_candidate',dict(content=json.dumps(self.legacy.call(role,payload))),self.n)
            if len(messages)==2:
                self.buffer='';return reply('read_artifact',dict(artifact_id='task.json',limit=16000),self.n)
            result=json.loads(messages[-1]['content']);self.buffer+=result['content']
            if result.get('has_more'):return reply('read_artifact',dict(artifact_id='task.json',offset=len(self.buffer),limit=16000),self.n)
            payload=json.loads(self.buffer)
            if role=='editor' and 'parent_code' not in payload:
                self.editor_payload=payload
                return reply('read_artifact',dict(artifact_id='parent.py',limit=1000000),self.n)
            value=dict(rules=[dict(name='uniform',source=GEN,parameters={},weight=1)]) if role=='data_designer' else self.legacy.call(role,payload)
            return reply('submit_candidate',dict(content=json.dumps(value)),self.n)
    config,settings=small_fixture(tmp_path,rounds=rounds)
    config.update(agent_tool_loop_v1=True,agent_task_runtime_v2=True,tool_max_turns=None)
    settings['agent_tool_loop_v1']=True
    model=Roles();engine=AlternatingEngine(config,model,DataRoundProvider(settings,model))
    pause=engine.archive.root/'pause.request';pause.write_text('pause')
    assert engine.run()['status']=='paused'
    assert model.n==0
    pause.unlink()
    resumed=AlternatingEngine(config,model,DataRoundProvider(settings,model))
    result=resumed.run()
    assert result['status']=='finished',result['reason']
    assert len(resumed.archive.of('round_complete'))==rounds
    assert len(resumed.archive.of('package'))==rounds
    assert len(resumed.archive.of('data_validation_start'))>0
    assert len(resumed.archive.of('data_validation_start'))==len(resumed.archive.of('data_validation_result'))
    for path in resumed.archive.root.parent.rglob('agent_task/session/facts.jsonl'):
        identity=Archive(path.parent).of('identity')[0]
        assert identity['max_turns'] is None and identity['runtime_version']==2


def test_parameters_are_injected_without_exec_or_open(tmp_path):
    tools=TaskTools(tmp_path,{},lambda p:{},runtime_v2=True)
    result=tools.dispatch('run_python',dict(code='result = parameters["count"]',parameters={'count':7},input_artifact_ids=[]))
    assert result['ok'] and result['result']==7
    assert result['parameters']=={'count':7}


def test_all_case_materials_have_same_solver_instance_schema(tmp_path):
    c,_=small_fixture(tmp_path,rounds=1);c['agent_task_runtime_v2']=True
    e=AlternatingEngine(c,None)
    material=_training_materials(e)['training_cases.json']['value']
    assert all('instance' in row and 'coordinates' in row for row in material)
    assert all('distance_matrix' in row['instance'] for row in material)
    assert all('reference' not in row['instance'] for row in material)


def test_new_execution_evidence_is_citable_and_pause_resume_binds_original_inputs(tmp_path):
    from cef.tool_agents import run_algorithm_task
    c,_=small_fixture(tmp_path,rounds=1);c.update(agent_tool_loop_v1=True,agent_task_runtime_v2=True,tool_max_turns=None)
    e=AlternatingEngine(c,None)
    e.register('warm_start',Path(c['warm_start_source']).read_text(),initialization='explicit warm start')
    pause=e.archive.root/'pause.request'
    class Inspect:
        config={}
        def __init__(self):self.n=0;self.evidence=None
        def exchange(self,messages,tools,role):
            self.n+=1
            if self.n==1:return reply('run_python',dict(code='result = 3',input_artifact_ids=[]),1)
            self.evidence=json.loads(messages[-1]['content'])['evidence_id']
            assert self.evidence in e.known_evidence_ids()
            pause.write_text('pause')
            decision=dict(action='inspect',parent_or_target_id='warm_start',evidence_ids=[self.evidence],previous_event_update=[])
            return reply('submit_candidate',dict(content=json.dumps(decision)),2)
    model=Inspect();payload=e.state()
    with pytest.raises(TaskPaused):run_algorithm_task(e,model,'planner',payload,'step_0:planner')
    pause.unlink()
    result=run_algorithm_task(e,Fake([]),'planner',payload,'step_0:planner')
    assert result['evidence_ids']==[model.evidence]
    assert model.evidence in _training_materials(e)
    assert len(e.archive.of('agent_tool_evidence'))==1
    used=e.ledger.used('analysis_seconds')
    assert used>0
    run_algorithm_task(e,Fake([]),'planner',payload,'step_0:planner')
    assert e.ledger.used('analysis_seconds')==used


def test_pause_before_new_measurement_starts_no_worker(tmp_path):
    c,_=small_fixture(tmp_path,rounds=1);c['agent_task_runtime_v2']=True
    e=AlternatingEngine(c,None)
    (e.archive.root/'pause.request').write_text('pause')
    with pytest.raises(TaskPaused):e.measure_rows('warm_start',e.rows,'after_references')
    assert not e.archive.of('execution_start')


def test_generation_failure_is_recorded_with_cost_and_frozen_source(tmp_path):
    from cef.task_experiments import TaskExperiments
    c,_=small_fixture(tmp_path,rounds=1);c['agent_task_runtime_v2']=True
    e=AlternatingEngine(c,None)
    materials=_training_materials(e)
    materials['solver.py']=dict(type='text',value='def solve(instance,time_limit_s,seed):\n    return {"solution":list(range(len(instance["distance_matrix"])))}')
    materials['bad.py']=dict(type='text',value=GEN.replace('return np.random.default_rng(seed).random((n_cities, 2)).tolist()','return []'))
    tools=TaskTools(tmp_path/'task',materials,lambda p:{},runtime_v2=True)
    tools.experiment=TaskExperiments(e,tools).run
    result=tools.dispatch('run_solver_experiment',dict(source_id='solver.py',generator_id='bad.py',seeds=[71]))
    assert result['ok'] is False
    full=json.loads((tools.results/result['artifact_id'].split(':')[1]).read_text())
    assert full['status']=='generation_failed'
    assert e.archive.read_source(full['generator_receipt']['source_hash'])==materials['bad.py']['value']
    assert len(e.archive.of('experimental_generation_start'))==1
    assert len(e.archive.of('experimental_generation'))==1
    assert e.ledger.used('analysis_seconds')>0


def test_v2_launcher_default_is_unlimited_without_user_number(tmp_path):
    from scripts.cef_alternating_launcher import build_auto_config
    from runCEFAuto import CONFIG
    settings=dict(CONFIG);settings.pop('tool_max_turns')
    config,_=build_auto_config(settings,tmp_path/'run',2025)
    assert config['agent_task_runtime_v2'] and config['tool_max_turns'] is None


def test_unknown_experimental_usage_is_reported_not_zeroed(tmp_path):
    c,_=small_fixture(tmp_path,rounds=1);c['agent_task_runtime_v2']=True
    e=AlternatingEngine(c,None)
    e.archive.add('experimental_case_start',dict(experiment_id='interrupted',id='case',seed=1))
    e.archive.add('experimental_generation_start',dict(experiment_id='interrupted',seed=2))
    costs=e.export('paused','interruption')['resource_usage']
    assert costs['experimental_unknown_outcomes']==1
    assert costs['experimental_unknown_generation_outcomes']==1


def test_reproducibility_rejection_is_recorded_as_failed_validation(tmp_path,monkeypatch):
    from cef.tool_agents import run_data_task
    c,settings=small_fixture(tmp_path,rounds=1);c.update(agent_tool_loop_v1=True,agent_task_runtime_v2=True)
    e=AlternatingEngine(c,None)
    outputs=iter([[[0.,0.]]*7,[[1.,1.]]*7])
    monkeypatch.setattr('cef.tool_agents.generate_coordinates',lambda *args:next(outputs))
    proposal=dict(rules=[dict(name='probe',source=GEN,parameters={},weight=1)])
    model=Fake([reply('submit_candidate',dict(content=json.dumps(proposal)),1),
        reply('report_blocker',dict(reason='Reproducibility check failed; cannot deliver this draft.',evidence=[]),2)])
    payload=dict(n_cities=7,instances=1,instance_seeds=[71],current_rules=[],calibration_solver='def solve(instance,time_limit_s,seed): pass')
    with pytest.raises(TaskBlocked):run_data_task(e,model,payload,tmp_path/'data',settings)
    results=e.archive.of('data_validation_result')
    assert len(results)==1 and 'not reproducible' in results[0]['error']


def test_solver_experiment_uses_real_runner_and_does_not_promote(tmp_path):
    from cef.task_experiments import TaskExperiments
    c,_=small_fixture(tmp_path,rounds=1);c['agent_task_runtime_v2']=True;c['solver_time_limit_s']=.5
    e=AlternatingEngine(c,None)
    source='import time\ndef solve(instance,time_limit_s,seed):\n    time.sleep(.12)\n    return {"solution":list(range(len(instance["distance_matrix"])))}\n'
    materials=_training_materials(e)
    materials['solver.py']=dict(type='text',value=source)
    materials['generator.py']=dict(type='text',value=GEN)
    tools=TaskTools(tmp_path/'task',materials,lambda p:{},runtime_v2=True,timeout_s=.05)
    tools.experiment=TaskExperiments(e,tools).run
    out=tools.dispatch('run_solver_experiment',dict(source_id='solver.py',instance_ids=[e.rows[0]['id']]))
    assert out['ok'] and out['cases'][0]['valid']
    assert out['conditions']['solver_time_limit_s']==.5
    assert out['cases'][0]['elapsed']>.1
    assert 'solution' not in out['cases'][0]
    tools.timeout_s=3
    analyzed=tools.dispatch('run_python',dict(code='result = len(inputs['+repr(out['artifact_id'])+']["cases"][0]["solution"])',input_artifact_ids=[out['artifact_id']]))
    assert analyzed['ok'] and analyzed['result']==7
    assert not e.archive.of('measurement') and not e.candidates()
    generated=tools.dispatch('run_solver_experiment',dict(source_id='solver.py',generator_id='generator.py',seeds=[71],n_cities=7))
    assert generated['ok'] and generated['cases'][0]['valid']
    assert generated['cases'][0]['reference_gap_pct'] is None
    assert generated['cases'][0]['objective']>0
    assert len(e.archive.of('experimental_execution'))==2
    full=json.loads((tools.results/out['artifact_id'].split(':')[1]).read_text())
    assert e.archive.read_source(full['source_hash'])==source
    later=TaskTools(tmp_path/'later',{},lambda p:{},runtime_v2=True)
    later.source_loader=e.archive.read_source
    source_id=full['source_receipt']['artifact_id']
    assert later.dispatch('read_artifact',dict(artifact_id=source_id))['content']==source
    loaded=later.dispatch('run_python',dict(code='result = "solve" in load_module('+repr(source_id)+')',input_artifact_ids=[source_id]))
    assert loaded['ok'] and loaded['result'] is True
    assert generated['artifact_id']
    generated_full=json.loads((tools.results/generated['artifact_id'].split(':')[1]).read_text())
    assert e.archive.read_source(generated_full['generator_receipt']['source_hash'])==GEN
    assert len(e.archive.of('experimental_generation'))==1
    assert len(e.archive.of('experimental_case_start'))==2
    assert e.ledger.used('analysis_seconds')>0
    oversized=tools.dispatch('run_solver_experiment',dict(source_id='solver.py',generator_id='generator.py',seeds=[71],n_cities=10000))
    assert oversized['ok'] is False and 'training sizes' in oversized['error']
    with pytest.raises(Exception):TaskExperiments(e,tools).run(source_id='solver.py',instance_ids=['TSPLIB_hidden'])
