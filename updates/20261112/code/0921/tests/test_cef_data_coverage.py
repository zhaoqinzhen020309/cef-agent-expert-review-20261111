import copy,json
from cef.data_coverage import accumulate,candidate_state
from cef.store import Archive
from cef.task_experiments import TaskExperiments
from test_cef_measured_loop import fixture


def sample(ids):
    return dict(candidate_identity='A',calibration_source_hash='solver',plan=[dict(id='a'),dict(id='b')],rules=[dict(name='rule',source='code',parameters={},weight=1)],
        prepared_rows=[dict(id=k) for k in ids],executions=[dict(instance_id=k,valid=True) for k in ids],
        cases=[dict(id=k,valid=True,reference_gap_pct=.1,observation_contract={'seconds':60}) for k in ids])


def merge(values):
    m=sample([])
    return accumulate(values,candidate_identity='A',plan=m['plan'],rules=m['rules'],calibration_source_hash='solver',contract={'seconds':60})


def test_batches_and_duplicates_have_same_complete_result():
    a,b=sample(['a']),sample(['b'])
    assert merge([a,b])==merge([sample(['a','b'])])
    assert merge([a,b,a])==merge([a,b])
    repeat=sample(['a']);repeat['cases'][0]['reference_gap_pct']=9
    assert merge([a,b,repeat])['cases'][0]['reference_gap_pct']==.1


def test_incompatible_updates_cannot_complete():
    for field,value in [('candidate_identity','B'),('calibration_source_hash','other'),('rules',[]),('plan',[])]:
        b=sample(['b']);b[field]=value
        assert not merge([sample(['a']),b])['complete']
    b=sample(['b']);b['cases'][0]['observation_contract']={'seconds':10}
    assert not merge([sample(['a']),b])['complete']


def test_best_candidate_does_not_follow_working_candidate(tmp_path):
    archive=Archive(tmp_path)
    for index,m in enumerate([sample(['a']),sample(['b'])]):
        path=tmp_path/f'{index}.json';path.write_text(json.dumps(m));archive.add('data_candidate_measurement',dict(candidate_identity='A',artifact_path=str(path)))
    weak=sample(['a']);weak['candidate_identity']='B';weak['rules'][0]['source']='weak code';weak['cases'][0]['reference_gap_pct']=0
    p=tmp_path/'weak.json';p.write_text(json.dumps(weak));archive.add('data_candidate_measurement',dict(candidate_identity='B',artifact_path=str(p)))
    state=candidate_state(archive,'solver',{'seconds':60})
    assert state['best_complete_candidate']['candidate_identity']=='A'
    assert state['best_complete_candidate']['mean_reference_gap_pct']==.1
    assert next(c for c in state['candidates'] if c['candidate_identity']=='B')['complete'] is False


def test_partial_measurements_reach_existing_completion_path(tmp_path):
    from cef.integrated_data import render_data_report
    engine,tools,proposal,context=fixture(tmp_path)
    engine.config['exploration_efficiency_v1']=True
    context['payload']['calibration_conditions']=dict(engine.shared_measurements.contract(engine),active_parallel_cases=1)
    context['payload']['previous_results']=[dict(valid=True,mean_reference_gap_pct=-1)]
    tools.enable_efficiency('data_designer')
    exp=TaskExperiments(engine,tools,data_context=context)
    first=exp.measure_data_candidate('candidate.json',[0])
    assert not first['cumulative_coverage']['complete'] and '_goal_proposal' not in first
    second=exp.measure_data_candidate('candidate.json',[1])
    assert second['cumulative_coverage']['complete']
    assert second['cumulative_coverage']['measured_case_count']==2
    assert '_goal_proposal' in second
