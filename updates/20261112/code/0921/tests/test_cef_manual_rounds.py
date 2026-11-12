import json
from pathlib import Path
from unittest.mock import patch
import pytest
from cef.alternating import AlternatingEngine
from test_cef_alternating import config,AlgorithmFake,epoch_rows


def fixture(tmp_path):
    c=config(tmp_path,2)
    c.update(manual_rounds=True,max_iterations=None,measured_research_loop_v1=True)
    model=AlgorithmFake()
    return c,model,lambda e,n,p:epoch_rows(n)


def test_manual_runs_one_round_next_runs_one_more_without_limit(tmp_path):
    c,model,provider=fixture(tmp_path)
    engine=AlternatingEngine(c,model,provider)
    first=engine.run()
    assert first['status']=='paused' and first['completed_iterations']==1
    assert (tmp_path/'rounds/001/round_summary.json').is_file()
    assert (tmp_path/'rounds/001/best_solver.py').is_file()
    count=len(model.inputs)
    restored=AlternatingEngine(c,model,provider)
    assert restored.run()['completed_iterations']==1
    assert len(model.inputs)==count
    second=restored.run(manual_next=True)
    assert second['status']=='paused' and second['completed_iterations']==2
    for n in range(3,12):
        result=AlternatingEngine(c,model,provider).run(manual_next=True)
        assert result['status']=='paused' and result['completed_iterations']==n
    assert model.edits==11


@pytest.mark.parametrize('resume_with_next',[False,True])
def test_partial_round_resume_finishes_requested_round_not_an_extra_round(tmp_path,resume_with_next):
    c,model,provider=fixture(tmp_path);engine=AlternatingEngine(c,model,provider)
    request=engine.request
    def interrupt(role,payload,key):
        if role=='editor':raise RuntimeError('offline')
        return request(role,payload,key)
    with patch.object(engine,'request',side_effect=interrupt):assert engine.run()['status']=='failed'
    result=AlternatingEngine(c,model,provider).run(manual_next=resume_with_next)
    assert result['status']=='paused' and result['completed_iterations']==1
    assert model.edits==1


def test_manual_finish_only_at_completed_boundary_and_freezes_search(tmp_path):
    c,model,provider=fixture(tmp_path);engine=AlternatingEngine(c,model,provider)
    with pytest.raises(ValueError):engine.finish_manual()
    engine.run();calls=len(model.inputs)
    result=engine.finish_manual()
    assert result['status']=='finished' and result['completed_iterations']==1
    assert len(model.inputs)==calls
    frozen_journal=engine.archive.path.read_bytes()
    assert engine.finish_manual()['best']==result['best']
    assert engine.archive.path.read_bytes()==frozen_journal
    with pytest.raises(ValueError):engine.run(manual_next=True)
    assert json.loads((tmp_path/'search/result.json').read_text())['status']=='finished'


def test_manual_finish_rejects_pending_data_round(tmp_path):
    c,model,provider=fixture(tmp_path);engine=AlternatingEngine(c,model,provider);engine.run()
    def pending(*args):raise RuntimeError('pending data investigation')
    engine.data_provider=pending
    assert engine.run(manual_next=True)['status']=='failed'
    with pytest.raises(ValueError):engine.finish_manual()


def test_manual_config_has_no_iteration_cap_and_check_creates_no_run(tmp_path):
    from runCEFAuto import CONFIG
    from scripts.cef_alternating_launcher import build_auto_config,main
    settings=dict(CONFIG,rounds=None,output_dir=tmp_path/'new')
    c,test=build_auto_config(settings,tmp_path/'run',2025)
    assert c['manual_rounds'] and c['max_iterations'] is None
    assert c['max_case_runs'] is None and len(test['instances'])==29
    from cef.config import validate
    with pytest.raises(ValueError):validate(dict(c,manual_rounds=False))
    with pytest.raises(ValueError):validate(dict(c,manual_rounds=True,max_iterations=10))
    assert main(settings,['--check'])==0
    assert not settings['output_dir'].exists()


@pytest.mark.parametrize('flag',['--next','--finish'])
def test_cli_passes_explicit_operation_to_existing_run(tmp_path,flag):
    from scripts.cef_alternating_launcher import main
    from runCEFAuto import CONFIG
    c=dict(alternating_agents=True,manual_rounds=True,run_dir=str(tmp_path/'search'))
    (tmp_path/'search_config.json').write_text(json.dumps(c))
    (tmp_path/'test_config.json').write_text('{}')
    with patch('scripts.cef_alternating_launcher.validate',side_effect=lambda c:c),patch('scripts.cef_alternating_launcher.run_folder') as run:
        assert main(CONFIG,[flag,str(tmp_path)])==0
    assert run.call_args.kwargs['operation']==('next' if flag=='--next' else 'finish')


@pytest.mark.parametrize('operation',['finish','evaluate'])
def test_finish_runs_real_small_benchmark_without_model_auth_or_network(tmp_path,operation):
    import os,numpy as np
    from cef.data import receipt
    from scripts.cef_alternating_launcher import run_folder
    c,model,provider=fixture(tmp_path)
    path=tmp_path/'test.npz';np.savez(path,distance_matrix=np.ones((3,3))-np.eye(3))
    rows=[dict(id='triangle',split='test',**receipt(path),reference=3.)]
    c['data_manifest']['test']=rows
    c.update(test_repeats=1,model='offline-test',api_key_env='NO_TEST_KEY',warm_start_receipt=receipt(Path(c['warm_start_source'])))
    t=dict(instances=rows,test_repeats=1,evaluation_seeds={'triangle:0':2025})
    engine=AlternatingEngine(c,model,provider);assert engine.run()['status']=='paused'
    with patch.dict(os.environ,{},clear=True),patch('scripts.cef_alternating_launcher.load_env'), \
            patch('scripts.cef_alternating_launcher.balance_snapshot',side_effect=AssertionError('No network')), \
            patch('scripts.cef_alternating_launcher.matching_receipt',side_effect=AssertionError('No model gate')), \
            patch('scripts.cef_alternating_launcher.Model',side_effect=AssertionError('No model')), \
            patch('cef.reporting.audit_test'),patch('cef.round_benchmark.audit_test'):
        result=run_folder(c,t,tmp_path,resume=True,operation=operation)
    if operation=='finish':
        assert result['status']=='finished' and result['test']['valid']
        assert result['test']['mean_gap_pct']==0.
        assert (tmp_path/'test/summary.json').is_file()
    else:
        assert result['valid'] and result['mean_gap_pct']==0.
        assert not engine.archive.of('manual_finished')
        assert len(engine.archive.of('round_complete'))==1


def test_launcher_delivers_two_round_pauses_and_boundary_resume_without_model_calls(tmp_path):
    from scripts.cef_alternating_launcher import run_folder
    from cef.data import receipt
    c,model,provider=fixture(tmp_path)
    c['data_manifest']['test']=[]
    c.update(test_repeats=1,model='offline-test',alternating_data_config={},
        warm_start_receipt=receipt(Path(c['warm_start_source'])))
    engine=AlternatingEngine(c,model,provider)
    t=dict(instances=[],test_repeats=1,evaluation_seeds={})
    with patch('scripts.cef_alternating_launcher.load_env'), \
            patch('scripts.cef_alternating_launcher.os.environ',{'apikey':'offline'}), \
            patch('scripts.cef_alternating_launcher.balance_snapshot',return_value={}), \
            patch('scripts.cef_alternating_launcher.matching_receipt',return_value=True), \
            patch('scripts.cef_alternating_launcher.AlternatingEngine',return_value=engine), \
            patch('scripts.cef_alternating_launcher.Model',return_value=model), \
            patch('scripts.cef_alternating_launcher.build_data_model_config',return_value={}), \
            patch('scripts.cef_alternating_launcher.DataRoundProvider',return_value=provider):
        first=run_folder(c,t,tmp_path,resume=True)
        second=run_folder(c,t,tmp_path,resume=True,operation='next')
        calls=len(model.inputs)
        repeated=run_folder(c,t,tmp_path,resume=True)
    assert (first['rounds'],second['rounds'],repeated['rounds'])==(1,2,2)
    assert len(model.inputs)==calls
    assert json.loads((tmp_path/'status_paused.json').read_text())['rounds']==2
    assert len(list(tmp_path.glob('billing_resume_*.json')))==3


def test_common_start_research_supports_manual_rounds_and_final_export(tmp_path):
    from cef.config import freeze_research_contract
    c,_,provider=fixture(tmp_path)
    c=freeze_research_contract(dict(c,common_start_probe=True,probe_states=1,probe_seconds=.04))
    signature='def solve(instance, time_limit_s, seed, *, initial_solution=None, observer=None):'
    Path(c['warm_start_source']).write_text('import time\n'+signature+"\n    route=list(range(len(instance['distance_matrix'])))\n    if observer is not None: observer(route)\n    return {'solution':route}\n")
    class ResearchModel(AlgorithmFake):
        def call(self,role,payload):
            value=super().call(role,payload)
            if role=='editor':
                value['edits'][1]['exact_replacements']=[dict(old=signature,new=signature+'\n    # marker')]
            return value
    engine=AlternatingEngine(c,ResearchModel(),provider)
    result=engine.run()
    assert result['status']=='paused',result['reason']
    assert engine.run(manual_next=True)['completed_iterations']==2
    engine.archive.add('probe_result',dict(event_id='eligible_for_export'))
    with patch.object(engine.research,'brief',side_effect=lambda r:r),patch.object(engine.research,'integration_option',return_value={'available':True}):
        engine.research.export()
        assert engine.finish_manual()['status']=='finished'
    summary=json.loads((tmp_path/'search/integration_summary.json').read_text())
    assert summary['unused_integration_eligible_probes'][0]['status']=='not_attempted_search_finished'


def test_crash_after_round_commit_recovers_summary_without_advancing(tmp_path):
    c,model,provider=fixture(tmp_path);engine=AlternatingEngine(c,model,provider)
    add=engine.archive.add
    def crash(kind,value):
        saved=add(kind,value)
        if kind=='round_complete':raise RuntimeError('stopped after durable round completion')
        return saved
    with patch.object(engine.archive,'add',side_effect=crash):assert engine.run()['status']=='failed'
    calls=len(model.inputs)
    restored=AlternatingEngine(c,model,provider)
    result=restored.run()
    assert result['status']=='paused' and result['completed_iterations']==1
    assert len(model.inputs)==calls
    assert (tmp_path/'rounds/001/best_solver.py').is_file()


def test_crash_after_manual_freeze_resumes_finish_without_reopening_search(tmp_path):
    c,model,provider=fixture(tmp_path);engine=AlternatingEngine(c,model,provider);engine.run()
    with patch.object(engine,'export',side_effect=RuntimeError('crash before final export')),pytest.raises(RuntimeError):engine.finish_manual()
    restored=AlternatingEngine(c,model,provider)
    with pytest.raises(ValueError):restored.run(manual_next=True)
    calls=len(model.inputs)
    assert restored.finish_manual()['status']=='finished'
    assert len(model.inputs)==calls


def test_logging_entrypoint_forwards_finish_operation(tmp_path):
    from contextlib import nullcontext
    import cef_log_output
    from runCEFAuto import CONFIG
    calls=[]
    def run(config,test,folder,resume=False,**kwargs):
        calls.append(dict(resume=resume,**kwargs));return 0
    def main(settings,argv):
        import scripts.cef_alternating_launcher as launcher
        return launcher.run_folder({}, {},tmp_path,True,operation='finish')
    with patch('scripts.cef_alternating_launcher.run_folder',side_effect=run), \
            patch('scripts.cef_alternating_launcher.main',side_effect=main), \
            patch('cef_log_output.RunLog',return_value=nullcontext()):
        assert cef_log_output.launch(CONFIG,['--finish',str(tmp_path)])==0
    assert calls==[dict(resume=True,operation='finish')]
