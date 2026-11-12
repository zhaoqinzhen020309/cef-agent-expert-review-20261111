import json
from unittest.mock import patch
import pytest
from cef.task_experiments import TaskExperiments
from cef.research_progress import progress_view
from test_cef_measured_loop import fixture


def setup(tmp_path):
    engine,tools,_,_=fixture(tmp_path)
    source=engine.archive.read_source(engine.candidates()['warm_start']['source_hash'])
    tools.artifacts.update({'parent.py':dict(type='text',value=source),
        'training_cases.json':dict(type='json',value=[dict(id=k,seed=2025,reference=4.,instance={'distance_matrix':[[0,1,1],[1,0,1],[1,1,0]]}) for k in ('a','b')])})
    experiments=TaskExperiments(engine,tools)
    tools.enable_research(experiments)
    return engine,tools,experiments


def test_paired_tool_executes_both_sources_and_continues_only_new_cases(tmp_path):
    engine,tools,experiments=setup(tmp_path)
    calls=[]
    def case(source,adapter,instance,seed,**kwargs):
        calls.append((seed,kwargs['research_request']))
        return dict(valid=True,objective=3.,fitness=-25.,elapsed=.01,wallclock_s=.01,seed=seed)
    with patch('cef.task_experiments.Runner.case',side_effect=case):
        first=tools.dispatch('compare_sources',dict(reference_source_id='parent.py',candidate_source_id='parent.py',instance_ids=['a'],question='repeatability',solver_seed=71))
        assert first['valid_pairs']==1 and [v[0] for v in calls]==[71,71]
        second=tools.dispatch('continue_comparison',dict(comparison_id=first['artifact_id'],instance_ids=['b'],question='coverage'))
        assert second['paired_cases']==2 and len(calls)==4
        assert len(engine.archive.of('experimental_execution'))==4
        assert tools.dispatch('continue_comparison',dict(comparison_id=first['artifact_id'],instance_ids=['a'],question='duplicate'))['ok'] is False
        assert len(calls)==4


def test_common_start_reaches_runner_without_formal_contrast(tmp_path):
    engine,tools,experiments=setup(tmp_path)
    with patch('cef.task_experiments.Runner.case',return_value=dict(valid=True,objective=3.,fitness=-25.,elapsed=.01,wallclock_s=.01,seed=2025)) as worker:
        result=tools.dispatch('compare_sources',dict(reference_source_id='parent.py',candidate_source_id='parent.py',instance_ids=['a'],question='same path',initial_solutions={'a':[0,1,2]}))
        assert result['valid_pairs']==1
        assert worker.call_args.kwargs['research_request']['initial_solution']==[0,1,2]
        assert not engine.archive.of('contrast')


def test_research_update_persists_and_compacts_without_claiming_proof(tmp_path):
    engine,tools,_=setup(tmp_path)
    r=tools.dispatch('update_research',dict(question='why?',candidate_source_id='parent.py',findings='unresolved hypothesis',open_questions=['timing?'],next_action='measure',evidence_ids=[]))
    assert r['ok'] and r['context_checkpoint']
    assert progress_view(engine.archive)['questions'][0]['next_action']=='measure'
    assert tools._material(r['artifact_id'])[0]=='json'


def test_real_worker_common_start_and_solver_seed(tmp_path):
    engine,tools,exp=setup(tmp_path)
    tools.artifacts['tiny.py']=dict(type='text',value='def solve(instance,time_limit_s,seed,*,observer=None,initial_solution=None):\n    route=list(initial_solution) if initial_solution is not None else [0,1,2]\n    if observer: observer(route)\n    return {"solution":route}\n')
    result=exp.run('tiny.py',instance_ids=['a'],solver_seed=81,trace=True,initial_solutions={'a':[0,1,2]})
    assert result['cases'][0]['valid'],result
    assert result['cases'][0]['seed']==81
    assert result['cases'][0]['initial_solution_verified']


def test_data_reproduction_uses_generator_seed_not_solver_seed(tmp_path):
    engine,tools,exp=setup(tmp_path)
    engine.rows=[dict(id='a',generator_seed=42,rule_version='v')]
    tools.artifacts['training_cases.json']['value'][0]['coordinates']=[[0.,0.],[1.,0.],[0.,1.]]
    exp.data_context=dict(payload=dict(current_rules=[dict(version='v',source='generator',parameters={})]),settings=dict(generator_seconds=1))
    with patch('cef.task_experiments.generate_coordinates',return_value=[[0.,0.],[1.,0.],[0.,1.]]) as generate:
        result=exp.reproduce_training_case('a')
        assert result['identical_to_saved']
        assert generate.call_args.args[3]==42
        assert result['solver_seed']==2025


def test_native_backend_crash_is_not_returned_as_pending_or_zero_gap(tmp_path):
    from cef.reference import ReferenceBackendFailure
    from cef.integrated_data import IntegratedDataProvider
    engine,tools,proposal,context=fixture(tmp_path)
    exp=TaskExperiments(engine,tools,data_context=context)
    with patch.object(IntegratedDataProvider,'prepare_case',side_effect=ReferenceBackendFailure('SCIP memory error','MemoryError')):
        result=exp.measure_data_candidate('candidate.json',[0])
    assert not result['ok']
    assert result['pending_cases']==[]
    assert result['failed_cases'][0]['error_type']=='MemoryError'
    assert result['mean_reference_gap_pct'] is None


def test_data_materials_join_measured_difficulty_with_available_geometry():
    from cef.research_progress import data_observations
    cases=[dict(id='a',training_group='supplement',seed=2025,generator_seed=71,coordinates=[[0,0],[1,0],[0,1]])]
    records=[dict(instance_id='a',valid=True,fitness=.4,execution_id='run1')]
    report=data_observations(cases,records)['cases'][0]
    assert report['coordinates_available'] and report['measured_gap_pct']==[.4]
    assert report['generator_seed']==71 and report['solver_seed']==2025
    assert report['geometry']['span']==[1.,1.]


def test_prepared_package_bypasses_editor_but_still_formally_evaluates(tmp_path):
    import test_cef_alternating as alt
    _,engine,model,_,case=alt.AlternatingTests().fixture(str(tmp_path),rounds=1)
    original=engine.request
    calls=[]
    def request(role,payload,call_id,**kwargs):
        calls.append(role)
        decision=original(role,payload,call_id,**kwargs)
        if role=='planner' and decision['action']=='revise':
            editor=original('editor',engine.editor_context(decision['parent_or_target_id'],decision,2),call_id+':prepare')
            decision['prepared_package']=editor
        return decision
    with patch.object(engine,'request',side_effect=request),patch.object(engine.runner,'case',side_effect=case):
        result=engine.run()
    assert result['status']=='finished',result
    assert 'editor' not in calls
    assert engine.archive.of('contrast') and engine.archive.of('measurement')
