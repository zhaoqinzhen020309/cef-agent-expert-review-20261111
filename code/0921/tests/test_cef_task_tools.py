import hashlib
import json
from pathlib import Path

import pytest

from cef.task_tools import TaskTools


def make_tools(tmp_path, **kwargs):
    return TaskTools(tmp_path, {
        'data': {'type': 'json', 'value': [1, 2, 3]},
        'source': {'type': 'text', 'value': 'def _dedupe(points):\n    return list(dict.fromkeys(points))\n'},
    }, kwargs.pop('submit', lambda path: {'accepted': True, 'value': path.read_text()}), **kwargs)


def write(tools, path, content):
    result = tools.dispatch('write_file', {'relative_path': path, 'content': content})
    assert result['ok'], result


def test_read_pagination_actual_bound_material_and_workspace(tmp_path):
    tools = make_tools(tmp_path)
    first = tools.dispatch('read_artifact', {'artifact_id': 'source', 'offset': 0, 'limit': 10})
    assert first['content'] == 'def _dedup'
    assert first['has_more']
    write(tools, 'check.py', 'print(42)')
    assert tools.dispatch('read_artifact', {'artifact_id': 'check.py'})['content'] == 'print(42)'
    assert tools.dispatch('read_artifact', {'artifact_id': 'missing'})['ok'] is False


@pytest.mark.parametrize('path', ['../escape.py', '/escape.py', 'C:\\escape.py', '\\\\host\\x', 'a/../../x', '.task_results/x', 'a:stream'])
def test_write_rejects_escape_and_internal_paths(tmp_path, path):
    tools = make_tools(tmp_path)
    assert tools.dispatch('write_file', {'relative_path': path, 'content': 'x'})['ok'] is False


def test_actual_script_loader_inputs_and_digests(tmp_path):
    tools = make_tools(tmp_path)
    script = "import numpy as np\nfn = load_module('source')['_dedupe']\nprint(fn([1, 1, 2]))\nresult = int(np.sum(inputs['data']))\n"
    write(tools, 'check.py', script)
    out = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': ['data']})
    assert out['ok'], out
    assert out['result'] == 6
    assert out['stdout'] == '[1, 2]\n'
    assert out['exit_code'] == 0 and not out['timeout']
    assert out['script_sha256'] == hashlib.sha256(script.encode()).hexdigest()
    assert out['source_sha256']['source']
    assert out['input_sha256']['data']
    full = tools.dispatch('read_artifact', {'artifact_id': out['execution_artifact_id']})
    assert json.loads(full['content'])['result'] == 6


def test_workspace_source_loader_and_actual_exception(tmp_path):
    tools = make_tools(tmp_path)
    write(tools, 'candidate.py', 'def fn(x):\n    return x + 3\n')
    write(tools, 'check.py', "result = load_module('candidate.py')['fn'](5)\n")
    out = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert out['result'] == 8, out
    write(tools, 'check.py', "print('before', flush=True)\nraise ValueError('observed')\n")
    out = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert out['ok'] is False and 'ValueError: observed' in out['error']
    assert 'before' in out['stdout'] and 'check.py' in out['stderr']


@pytest.mark.parametrize('script', ["open('secret')", 'import os', "load_module('../secret.py')", "print((1).__class__)", "exec('print(2)')"])
def test_capability_restrictions_observable(tmp_path, script):
    tools = make_tools(tmp_path)
    write(tools, 'check.py', script)
    result = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert result['ok'] is False, result
    assert result['error']


def test_timeout_preserves_output_and_full_output_readable(tmp_path):
    tools = make_tools(tmp_path, timeout_s=2)
    write(tools, 'check.py', "print('started', flush=True)\nwhile True:\n    pass\n")
    result = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert result['timeout'] and not result['ok'], result
    assert 'started' in result['stdout']
    tools.timeout_s = 10
    write(tools, 'check.py', "print('x' * 30000)\n")
    result = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert result['stdout_truncated'] and len(result['stdout']) < 30000
    full = tools.dispatch('read_artifact', {'artifact_id': result['stdout_artifact_id'], 'limit': 40000})
    assert len(full['content']) == 30001


def test_submission_rejection_correction_receipt_and_callback_mutation(tmp_path):
    def submit(path):
        content = path.read_text()
        return {'accepted': content == 'good', 'value': content, 'error': 'invalid' if content != 'good' else None}
    tools = make_tools(tmp_path, submit=submit)
    write(tools, 'candidate.py', 'bad')
    assert not tools.dispatch('submit_candidate', {'path': 'candidate.py'})['accepted']
    write(tools, 'candidate.py', 'good')
    out = tools.dispatch('submit_candidate', {'path': 'candidate.py'})
    assert out['accepted'] and out['value'] == 'good'
    assert out['receipt']['sha256'] == hashlib.sha256(b'good').hexdigest()
    assert Path(out['receipt']['frozen_path']).read_text() == 'good'

    def mutate(path):
        path.write_text('changed')
        return {'accepted': True, 'value': 'invalid'}
    tools = make_tools(tmp_path / 'other', submit=mutate)
    write(tools, 'candidate.py', 'original')
    out = tools.dispatch('submit_candidate', {'path': 'candidate.py'})
    assert not out['accepted'] and 'changed' in out['error']


def test_symlink_escape(tmp_path):
    tools = make_tools(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    try:
        (tools.workspace / 'link').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('symlink privilege unavailable')
    assert not tools.dispatch('write_file', {'relative_path': 'link/x', 'content': 'bad'})['ok']


def test_no_unrequested_inputs_and_material_is_immutable(tmp_path):
    materials = {'data': {'type': 'json', 'value': {'x': 1}}}
    tools = TaskTools(tmp_path, materials, lambda path: {'accepted': False})
    materials['data']['value']['x'] = 999
    assert '999' not in tools.dispatch('read_artifact', {'artifact_id': 'data'})['content']
    write(tools, 'check.py', "result = inputs['data']")
    out = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert not out['ok'] and 'KeyError' in out['error']


def test_submission_exact_bytes_and_cancellation(tmp_path):
    tools = make_tools(tmp_path)
    raw = b'first\r\nsecond\r\n'
    (tools.workspace / 'candidate.py').write_bytes(raw)
    out = tools.dispatch('submit_candidate', {'path': 'candidate.py'})
    assert out['receipt']['sha256'] == hashlib.sha256(raw).hexdigest()
    assert Path(out['receipt']['frozen_path']).read_bytes() == raw

    def cancel(path):
        raise KeyboardInterrupt('cancelled')
    tools.submit = cancel
    with pytest.raises(KeyboardInterrupt):
        tools.dispatch('submit_candidate', {'path': 'candidate.py'})


def test_workspace_root_escape_and_material_input_validation(tmp_path):
    tools = make_tools(tmp_path)
    assert not tools.dispatch('run_python', {'script_path': 'missing.py', 'input_artifact_ids': []})['ok']
    write(tools, 'check.py', 'print(1)')
    assert not tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': ['missing']})['ok']
    assert not tools.dispatch('read_artifact', {'artifact_id': 'execution:../secret'})['ok']
    assert not tools.dispatch('read_artifact', {'artifact_id': 'data', 'offset': -1})['ok']
    assert not tools.dispatch('unknown', {})['ok']
    assert not tools.dispatch('write_file', {'relative_path': 'source', 'content': 'overwrite'})['ok']


def test_bound_loader_restrictions_and_json_result_error(tmp_path):
    tools = TaskTools(tmp_path, {'unsafe': {'type': 'text', 'value': 'import os\ndef f(): return 3'}}, lambda p: {})
    write(tools, 'check.py', "result = load_module('unsafe')['f']()")
    out = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert not out['ok'] and 'import denied' in out['error']
    assert out['exit_code'] == 1
    write(tools, 'check.py', 'result = {1, 2, 3}')
    out = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert not out['ok'] and 'non-JSON result' in out['error']


def test_large_result_retained_as_execution_artifact(tmp_path):
    tools = make_tools(tmp_path)
    write(tools, 'check.py', "result = 'y' * 30000")
    out = tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})
    assert out['ok'] and out['result_truncated'] and out['result'] is None
    full = tools.dispatch('read_artifact', {'artifact_id': out['execution_artifact_id'], 'limit': 100000})
    assert json.loads(full['content'])['result'] == 'y' * 30000


def test_validator_infrastructure_failure_propagates(tmp_path):
    tools = make_tools(tmp_path)
    write(tools, 'candidate.py', 'valid')

    def failed_validator(path):
        raise RuntimeError('reference backend down')

    tools.submit = failed_validator
    with pytest.raises(RuntimeError, match='reference backend down'):
        tools.dispatch('submit_candidate', {'path': 'candidate.py'})

    def unavailable_validator(path):
        raise OSError('reference file unavailable')

    tools.submit = unavailable_validator
    with pytest.raises(RuntimeError, match='infrastructure'):
        tools.dispatch('submit_candidate', {'path': 'candidate.py'})


def test_spawn_infrastructure_failure_propagates(tmp_path, monkeypatch):
    tools = make_tools(tmp_path)
    write(tools, 'candidate.py', 'valid')

    def failed_spawn(*args, **kwargs):
        raise OSError('process setup unavailable')

    monkeypatch.setattr('cef.task_tools.subprocess.Popen', failed_spawn)
    with pytest.raises(RuntimeError, match='infrastructure'):
        tools.dispatch('run_python', {'script_path': 'candidate.py', 'input_artifact_ids': []})


def test_worker_input_utf8_is_independent_of_windows_default_encoding(tmp_path, monkeypatch):
    import subprocess
    tools = make_tools(tmp_path)
    popen = subprocess.Popen
    def legacy_encoding(command, **kwargs):
        kwargs['env']['PYTHONIOENCODING']='gbk'
        return popen(command, **kwargs)
    monkeypatch.setattr('cef.task_tools.subprocess.Popen',legacy_encoding)
    text='本次观察支持；修改路径与接受策略，城市坐标。'
    result=tools.dispatch('run_python',dict(code='result = '+repr(text),input_artifact_ids=[]))
    assert result['ok'] and result['result']==text


def test_worker_startup_failure_is_infrastructure(tmp_path, monkeypatch):
    import subprocess
    import sys
    tools = make_tools(tmp_path)
    write(tools, 'check.py', 'result = 1')
    popen = subprocess.Popen

    def broken_worker(command, **kwargs):
        return popen([sys.executable, '-I', '-c', "raise RuntimeError('worker initialization failed')"], **kwargs)

    monkeypatch.setattr('cef.task_tools.subprocess.Popen', broken_worker)
    with pytest.raises(RuntimeError, match='infrastructure'):
        tools.dispatch('run_python', {'script_path': 'check.py', 'input_artifact_ids': []})


@pytest.mark.parametrize('error', [ValueError('bad'), TypeError('bad'), KeyError('bad'), SyntaxError('bad'), AssertionError('bad')])
def test_callback_validation_errors_observable(tmp_path, error):
    tools = make_tools(tmp_path)
    write(tools, 'candidate.py', 'valid')

    def invalid(path):
        raise error

    tools.submit = invalid
    out = tools.dispatch('submit_candidate', {'path': 'candidate.py'})
    assert not out['accepted'] and type(error).__name__ in out['error']
    assert out['receipt']['sha256']


def completed_write(call_id, path, content):
    call = {'id': call_id, 'function': {'name': 'write_file', 'arguments': json.dumps({'relative_path': path, 'content': content})}}
    return [
        {'kind': 'tool_started', 'data': {'call_id': call_id, 'call': call}},
        {'kind': 'tool_completed', 'data': {'call_id': call_id, 'call': call, 'result': {'ok': True, 'sha256': hashlib.sha256(content.encode()).hexdigest()}}},
    ]


def test_restore_completed_writes_latest_missing_and_mutated(tmp_path):
    tools = make_tools(tmp_path)
    records = completed_write('first', 'check.py', 'old') + completed_write('second', 'check.py', 'print(42)')
    assert tools.restore_completed_writes(records) == 1
    assert (tools.workspace / 'check.py').read_text() == 'print(42)'
    assert tools.restore_completed_writes(records) == 0
    (tools.workspace / 'check.py').write_text('external edit')
    with pytest.raises(RuntimeError, match='changed'):
        tools.restore_completed_writes(records)
    assert (tools.workspace / 'check.py').read_text() == 'external edit'


def test_restore_refuses_unknown_started_execution_and_tampered_record(tmp_path):
    tools = make_tools(tmp_path)
    records = completed_write('first', 'check.py', 'print(42)')
    unknown = {'kind': 'tool_started', 'data': {'call_id': 'unknown', 'call': {'function': {'name': 'run_python'}}}}
    with pytest.raises(RuntimeError, match='unknown started'):
        tools.restore_completed_writes(records + [unknown])
    assert not (tools.workspace / 'check.py').exists()
    records[-1]['data']['result']['sha256'] = 'mismatch'
    with pytest.raises(RuntimeError, match='digest'):
        tools.restore_completed_writes(records)


def test_submission_callback_receives_frozen_copy(tmp_path):
    received = []

    def submit(path):
        received.append(path)
        return {'accepted': True, 'value': path.read_text()}

    tools = make_tools(tmp_path, submit=submit)
    write(tools, 'candidate.py', 'exact source')
    out = tools.dispatch('submit_candidate', {'path': 'candidate.py'})
    assert out['accepted']
    assert received == [Path(out['receipt']['frozen_path'])]
    assert received[0] != tools.workspace / 'candidate.py'


def test_restore_malformed_completed_call_then_corrected_write(tmp_path):
    tools = make_tools(tmp_path)
    records = [
        {'kind': 'tool_started', 'data': {'call_id': 'bad', 'call': {'function': None}}},
        {'kind': 'tool_completed', 'data': {'call_id': 'bad', 'call': {'function': None}, 'result': {'error': 'invalid function'}}},
    ] + completed_write('correct', 'candidate.json', '{"valid": true}')
    records.append({'kind': 'submitted', 'data': {'value': {'valid': True}}})
    assert tools.restore_completed_writes(records) == 1
    assert json.loads((tools.workspace / 'candidate.json').read_text()) == {'valid': True}


def test_direct_code_is_persisted_and_exactly_one_script_source(tmp_path):
    tools = make_tools(tmp_path)
    code = "result = sum(inputs['data'])"
    out = tools.dispatch('run_python', {'code': code, 'input_artifact_ids': ['data']})
    assert out['ok'] and out['result'] == 6, out
    assert out['script_path'] == 'checks/' + hashlib.sha256(code.encode()).hexdigest()[:24] + '.py'
    assert (tools.workspace / out['script_path']).read_text() == code
    assert not tools.dispatch('run_python', {'input_artifact_ids': []})['ok']
    assert not tools.dispatch('run_python', {'code': code, 'script_path': out['script_path'], 'input_artifact_ids': []})['ok']


def test_read_edit_export_workspace_json_without_source_copy(tmp_path):
    tools = make_tools(tmp_path)
    write(tools, 'candidate.json', json.dumps({'source': 'return 1', 'n': 3}))
    code = "import json\np = inputs['candidate.json']\np['source'] = p['source'].replace('1', '2')\nfiles = {'proposal.json': json.dumps(p)}\nresult = p['n']"
    out = tools.dispatch('run_python', {'code': code, 'input_artifact_ids': ['candidate.json']})
    assert out['ok'] and out['result'] == 3, out
    assert json.loads((tools.workspace / 'proposal.json').read_text())['source'] == 'return 2'
    assert out['saved_files'][0]['path'] == 'proposal.json'
    assert out['saved_files'][0]['sha256'] == hashlib.sha256((tools.workspace / 'proposal.json').read_bytes()).hexdigest()
    assert out['input_sha256']['candidate.json']
    follow = tools.dispatch('run_python', {'code': "result=inputs['proposal.json']['source']", 'input_artifact_ids': ['proposal.json']})
    assert follow['result'] == 'return 2'


@pytest.mark.parametrize('files_code', [
    "{'good.txt': 'good', '../bad': 'bad'}", "{'a': 1}", "{1: 'bad'}", "['bad']",
    "{'A.txt': 'one', 'a.txt': 'two'}", "{'.task_results/out': 'bad'}",
    "{'a/b': 'one', 'a\\\\b': 'two'}", "{'big.txt': 'x' * (2 * 1024 * 1024 + 1)}",
])
def test_invalid_exports_do_not_write_any_files(tmp_path, files_code):
    tools = make_tools(tmp_path)
    out = tools.dispatch('run_python', {'code': 'files = ' + files_code, 'input_artifact_ids': []})
    assert not out['ok'], out
    assert not (tools.workspace / 'good.txt').exists()
    assert out['error']


def test_no_exports_on_script_failure_and_workspace_input_escape(tmp_path):
    tools = make_tools(tmp_path)
    out = tools.dispatch('run_python', {'code': "files={'out.txt':'not saved'}\nraise ValueError('fail')", 'input_artifact_ids': []})
    assert not out['ok'] and not (tools.workspace / 'out.txt').exists()
    for artifact_id in ('../secret.json', 'C:\\secret.json', '.task_results/secret.json'):
        out = tools.dispatch('run_python', {'code': 'result=inputs', 'input_artifact_ids': [artifact_id]})
        assert not out['ok']
    write(tools, 'plain.txt', 'actual saved text')
    out = tools.dispatch('run_python', {'code': "result=inputs['plain.txt']", 'input_artifact_ids': ['plain.txt']})
    assert out['result'] == 'actual saved text'


def test_completed_run_exports_and_direct_code_restore(tmp_path):
    tools = make_tools(tmp_path)
    records = completed_write('old', 'candidate.json', '{"old": true}')
    write(tools, 'candidate.json', '{"old": true}')
    code = "files={'candidate.json':'{\"new\": true}', 'actual.txt': 'a\\r\\nb'}"
    args = {'code': code, 'input_artifact_ids': []}
    out = tools.dispatch('run_python', args)
    assert out['ok'], out
    call = {'id': 'run', 'function': {'name': 'run_python', 'arguments': json.dumps(args)}}
    records += [
        {'kind': 'tool_started', 'data': {'call_id': 'run', 'call': call}},
        {'kind': 'tool_completed', 'data': {'call_id': 'run', 'call': call, 'result': out}},
    ]
    # Newer successful export supersedes an earlier completed write.
    assert tools.restore_completed_writes(records) == 0
    (tools.workspace / 'candidate.json').unlink()
    (tools.workspace / 'actual.txt').unlink()
    (tools.workspace / out['script_path']).unlink()
    assert tools.restore_completed_writes(records) == 3
    assert json.loads((tools.workspace / 'candidate.json').read_text()) == {'new': True}
    assert (tools.workspace / 'actual.txt').read_bytes() == b'a\r\nb'
    assert (tools.workspace / out['script_path']).read_text() == code


def test_invalid_export_parent_does_not_partially_write(tmp_path):
    tools = make_tools(tmp_path)
    write(tools, 'existing.txt', 'not directory')
    out = tools.dispatch('run_python', {'code': "files={'good.txt':'good', 'existing.txt/child':'bad'}", 'input_artifact_ids': []})
    assert not out['ok']
    assert not (tools.workspace / 'good.txt').exists()


def test_inline_submission_xor_type_and_deterministic_receipt(tmp_path):
    tools = make_tools(tmp_path)
    content = '{"decision":"ready"}'
    out = tools.dispatch('submit_candidate', {'content': content})
    assert out['accepted'] and out['value'] == content, out
    expected_path = 'submissions/' + hashlib.sha256(content.encode()).hexdigest()[:24] + '.json'
    assert out['receipt']['path'] == expected_path
    assert out['receipt']['sha256'] == hashlib.sha256(content.encode()).hexdigest()
    assert (tools.workspace / expected_path).read_bytes() == content.encode()
    assert Path(out['receipt']['frozen_path']).read_bytes() == content.encode()
    repeated = tools.dispatch('submit_candidate', {'content': content})
    assert repeated['receipt']['path'] == expected_path
    assert repeated['receipt']['sha256'] == out['receipt']['sha256']
    assert not tools.dispatch('submit_candidate', {})['ok']
    assert not tools.dispatch('submit_candidate', {'path': expected_path, 'content': content})['ok']
    assert not tools.dispatch('submit_candidate', {'content': {'decision': 'ready'}})['ok']


def test_inline_submission_validation_error_then_repair_keeps_old_bytes(tmp_path):
    def submit(path):
        value = json.loads(path.read_text())
        if value.get('valid') is not True:
            raise ValueError('valid must be true')
        return {'accepted': True, 'value': value}

    tools = make_tools(tmp_path, submit=submit)
    bad = '{"valid":false}'
    failed = tools.dispatch('submit_candidate', {'content': bad})
    assert not failed['accepted'] and 'valid must be true' in failed['error']
    repaired = tools.dispatch('submit_candidate', {'content': '{"valid":true}'})
    assert repaired['accepted'] and repaired['value'] == {'valid': True}
    assert Path(failed['receipt']['frozen_path']).read_bytes() == bad.encode()
    assert (tools.workspace / failed['receipt']['path']).read_bytes() == bad.encode()
    assert failed['receipt']['path'] != repaired['receipt']['path']


@pytest.mark.parametrize('tool_name', ['write_file', 'run_python'])
def test_restore_skips_rejected_malformed_arguments_before_corrected_write(tmp_path, tool_name):
    tools = make_tools(tmp_path)
    call = {'id': 'bad', 'function': {'name': tool_name, 'arguments': '{broken'}}
    records = [
        {'kind': 'tool_started', 'data': {'call_id': 'bad', 'call': call}},
        {'kind': 'tool_completed', 'data': {'call_id': 'bad', 'call': call, 'result': {'error': {'type': 'JSONDecodeError'}}}},
    ] + completed_write('corrected', 'proposal.json', '{"valid":true}')
    records.append({'kind': 'submitted', 'data': {'value': {'valid': True}}})
    assert tools.restore_completed_writes(records) == 1
    assert json.loads((tools.workspace / 'proposal.json').read_text()) == {'valid': True}


@pytest.mark.parametrize('tool_name', ['write_file', 'run_python'])
def test_restore_claimed_success_with_corrupt_arguments_fails_closed(tmp_path, tool_name):
    tools = make_tools(tmp_path)
    result = {'ok': True, 'sha256': 'claimed', 'script_sha256': 'claimed'}
    call = {'id': 'corrupt', 'function': {'name': tool_name, 'arguments': '{broken'}}
    records = completed_write('earlier', 'should_not_restore.py', 'valid') + [
        {'kind': 'tool_started', 'data': {'call_id': 'corrupt', 'call': call}},
        {'kind': 'tool_completed', 'data': {'call_id': 'corrupt', 'call': call, 'result': result}},
    ]
    with pytest.raises(RuntimeError, match='arguments'):
        tools.restore_completed_writes(records)
    assert not (tools.workspace / 'should_not_restore.py').exists()


@pytest.mark.parametrize('accepted', [True, False])
def test_restore_inline_submission_receipt_for_later_workspace_read(tmp_path, accepted):
    tools = make_tools(tmp_path, submit=lambda path: {'accepted': accepted, 'value': json.loads(path.read_text()), 'error': None if accepted else 'rejected'})
    content = '{"valid":true}'
    args = {'content': content}
    submitted = tools.dispatch('submit_candidate', args)
    path = submitted['receipt']['path']
    call = {'id': 'submit', 'function': {'name': 'submit_candidate', 'arguments': json.dumps(args)}}
    records = [
        {'kind': 'tool_started', 'data': {'call_id': 'submit', 'call': call}},
        {'kind': 'tool_completed', 'data': {'call_id': 'submit', 'call': call, 'result': submitted}},
    ]
    code = 'result = inputs[%r]' % path
    check_args = {'code': code, 'input_artifact_ids': [path]}
    checked = tools.dispatch('run_python', check_args)
    assert checked['result'] == {'valid': True}
    check_call = {'id': 'check', 'function': {'name': 'run_python', 'arguments': json.dumps(check_args)}}
    records += [
        {'kind': 'tool_started', 'data': {'call_id': 'check', 'call': check_call}},
        {'kind': 'tool_completed', 'data': {'call_id': 'check', 'call': check_call, 'result': checked}},
    ]
    (tools.workspace / path).unlink()
    (tools.workspace / checked['script_path']).unlink()
    assert tools.restore_completed_writes(records) == 2
    assert (tools.workspace / path).read_bytes() == content.encode()
    assert tools.dispatch('run_python', check_args)['result'] == {'valid': True}
    tools.submit = lambda path: {'accepted': True, 'value': json.loads(path.read_text())}
    assert tools.dispatch('submit_candidate', {'path': path})['accepted']
    (tools.workspace / path).write_text('external edit')
    with pytest.raises(RuntimeError, match='changed'):
        tools.restore_completed_writes(records)


def test_restore_inline_receipt_hash_mismatch_and_malformed_without_receipt(tmp_path):
    tools = make_tools(tmp_path)
    bad_call = {'id': 'bad', 'function': {'name': 'submit_candidate', 'arguments': '{malformed'}}
    records = [
        {'kind': 'tool_started', 'data': {'call_id': 'bad', 'call': bad_call}},
        {'kind': 'tool_completed', 'data': {'call_id': 'bad', 'call': bad_call, 'result': {'accepted': False, 'error': 'bad arguments'}}},
    ]
    assert tools.restore_completed_writes(records) == 0
    content = '{"valid":false}'
    submitted = tools.dispatch('submit_candidate', {'content': content})
    submitted['receipt']['sha256'] = 'corrupt'
    call = {'id': 'inline', 'function': {'name': 'submit_candidate', 'arguments': json.dumps({'content': content})}}
    records += [
        {'kind': 'tool_started', 'data': {'call_id': 'inline', 'call': call}},
        {'kind': 'tool_completed', 'data': {'call_id': 'inline', 'call': call, 'result': submitted}},
    ]
    with pytest.raises(RuntimeError, match='digest'):
        tools.restore_completed_writes(records)
