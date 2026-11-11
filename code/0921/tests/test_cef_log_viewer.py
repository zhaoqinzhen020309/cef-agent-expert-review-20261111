import json
from pathlib import Path

from watchCEF import LogView,JournalTail


def event(kind,data):return dict(kind=kind,data=data,time=0)


def test_tool_actions_show_actual_target_not_prompt_or_code(tmp_path):
    view=LogView(tmp_path)
    path=tmp_path/'rounds/001/data/agent_task/session/facts.jsonl'
    lines=view.consume(path,event('tool_started',dict(turn=7,call_id='a',call=dict(function=dict(
        name='run_python',arguments=json.dumps(dict(script_path='check_clusters.py',input_artifact_ids=['training_records.json'])))))))
    text=' '.join(lines)
    assert '数据Agent' in text and '7' in text and 'check_clusters.py' in text
    assert 'training_records.json' in text
    lines=view.consume(path,event('tool_completed',dict(turn=7,call_id='a',call=dict(function=dict(name='run_python')),
        result=dict(ok=False,error='ValueError: sample invalid',wallclock_s=1.2))))
    assert '失败' in ' '.join(lines) and 'sample invalid' in ' '.join(lines)


def test_experiment_conditions_progress_and_no_invented_gap(tmp_path):
    view=LogView(tmp_path)
    artifact=tmp_path/'rounds/001/data/agent_task/.task_results/x.experiment.json'
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps(dict(source_id='parent.py',input_spec=dict(generator_id='g.py',seeds=[1,2]),
        conditions=dict(solver_time_limit_s=20,formal_solver_time_limit_s=60))))
    path=tmp_path/'search/facts.jsonl'
    text=' '.join(view.consume(path,event('experimental_start',dict(experiment_id='x',artifact_path=str(artifact)))))
    assert '20' in text and '60' in text and '诊断' in text and 'g.py' in text
    for i in range(2):
        lines=view.consume(path,event('experimental_execution',dict(experiment_id='x',id=str(i),valid=True,reference_gap_pct=None,elapsed=.5)))
    assert '2/2' in ' '.join(lines)
    lines=view.consume(path,event('experimental_complete',dict(experiment_id='x',artifact_path=str(artifact))))
    assert '无参考gap' in ' '.join(lines)


def test_tail_does_not_repair_or_consume_partial_journal_line(tmp_path):
    path=tmp_path/'facts.jsonl'
    first=json.dumps(event('model_started',dict(turn=1)))+'\n'
    second=json.dumps(event('paused',dict(reason='pause')))+'\n'
    path.write_bytes((first+second[:12]).encode())
    before=path.read_bytes();tail=JournalTail(path)
    assert [r['kind'] for r in tail.poll()]==['model_started']
    assert path.read_bytes()==before
    with path.open('ab') as f:f.write(second[12:].encode())
    assert [r['kind'] for r in tail.poll()]==['paused']
    assert tail.poll()==[]


def test_submission_rejection_visible_and_large_content_hidden(tmp_path):
    view=LogView(tmp_path);path=tmp_path/'search/agent_tasks/step_0_editor/session/facts.jsonl'
    lines=view.consume(path,event('tool_started',dict(turn=3,call_id='s',call=dict(function=dict(name='submit_candidate',arguments=json.dumps(dict(content='SECRET_CODE')))))))
    assert '提交' in ' '.join(lines) and 'SECRET_CODE' not in ' '.join(lines)
    lines=view.consume(path,event('tool_completed',dict(turn=3,call_id='s',call=dict(function=dict(name='submit_candidate')),
        result=dict(accepted=False,error='missing parent_hash'))))
    assert '未通过' in ' '.join(lines) and 'parent_hash' in ' '.join(lines)


def test_generator_failure_is_not_displayed_as_completed_solver_experiment(tmp_path):
    view=LogView(tmp_path)
    view.experiments['failed']=dict(total=3,valid=0,failed=0,gaps=[],actor='数据Agent')
    lines=view.consume(tmp_path/'search/facts.jsonl',event('experimental_complete',dict(experiment_id='failed',status='generation_failed')))
    assert '生成阶段失败' in ' '.join(lines) and '未开始求解' in ' '.join(lines)
