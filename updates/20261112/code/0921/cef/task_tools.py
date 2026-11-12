"""Generic tools over explicitly bound task materials and a local workspace.

Bindings are copied at construction. ``read_artifact`` pages the UTF-8 text
representation (JSON bindings are serialized); IDs can also name workspace
files or persisted execution outputs. Checks see ``inputs[id]`` only for IDs
explicitly requested. ``load_module(id_or_relative_py)`` executes the actual
bound source under the same restrictions and returns its callable exports.
``result`` is an optional JSON-compatible check-script global. ``files`` may
export up to 128 UTF-8 text files, with a total encoded size <= 2 MiB; host
validates every path before committing any export on successful execution.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import re
import sys
import tempfile
import time
import uuid

_UNSET = object()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    if isinstance(content, bytes):
        temporary.write_bytes(content)
    else:
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(content)
    os.replace(temporary, path)


def _schema(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False}}}


class TaskTools:
    schemas = [
        _schema('read_artifact', 'Read bound text/JSON, workspace relative file, or execution output by ID. Offset/limit count characters of text (JSON is serialized). Returns content, type, SHA256, has_more. Example: {"artifact_id":"parent_source","offset":0,"limit":8000}.',
                {'artifact_id': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0}, 'limit': {'type': 'integer', 'minimum': 1}}, ['artifact_id']),
        _schema('write_file', 'Persist a UTF-8 script or candidate in the task workspace. Relative paths only. Returns exact saved SHA256. Example: {"relative_path":"check.py","content":"print(1)"}.',
                {'relative_path': {'type': 'string'}, 'content': {'type': 'string'}}, ['relative_path', 'content']),
        _schema('run_python', 'Execute Python using exactly one of code (saved automatically as checks/<sha24>.py) or script_path (existing workspace file). inputs[id] contains requested bound materials OR saved workspace files (.json parsed; other files text); bound IDs take precedence. load_module(bound_text_id or workspace_relative.py) returns actual source callable exports, including underscore helpers. Optional result is JSON-compatible. Optional files={relative_path: UTF8_text} exports files only on success; every path validated first, max 128 files / 2 MiB total. math/random/time/numpy/json allowed; no filesystem/network/process. Returns saved_files paths/SHA256 and actual output/error/digests; full outputs by artifact ID. Example: {"code":"import json\\np=inputs[\'candidate.json\']\\np[\'source\']=p[\'source\'].replace(\'old\',\'new\')\\nfiles={\'proposal.json\':json.dumps(p)}", "input_artifact_ids":["candidate.json"]}.',
                {'script_path': {'type': 'string'}, 'code': {'type': 'string'}, 'input_artifact_ids': {'type': 'array', 'items': {'type': 'string'}}}, ['input_artifact_ids']),
        _schema('submit_candidate', 'Submit exactly one of content (UTF-8 JSON/text string, ideal for small structured decisions/packages) or path (existing workspace candidate; use exported files for large packages). Inline content is saved as submissions/<sha24>.json, then both forms undergo the same frozen host validation. A rejection gives actual errors for correction. Only acceptance ends the task; text alone never bypasses validation. Example: {"content":"{\\"decision\\":\\"ready\\"}"}.',
                {'path': {'type': 'string'}, 'content': {'type': 'string'}}, []),
    ]
    schemas[2]['function']['parameters']['oneOf'] = [{'required': ['code']}, {'required': ['script_path']}]
    schemas[3]['function']['parameters']['oneOf'] = [{'required': ['path']}, {'required': ['content']}]

    def __init__(self, root, artifacts, submit, timeout_s=10,runtime_v2=False):
        self.schemas=copy.deepcopy(type(self).schemas)
        self.runtime_v2=runtime_v2
        self.experiment=None
        self.data_measurement=None
        self.experiment_batch=None
        self.reproduce_case=None
        self.continue_comparison=None
        self.source_comparison=None
        self.research_update=None
        self.efficiency_role=None
        self.source_loader=None
        if runtime_v2:
            self.schemas[0]['function']['description']+=' Immutable source:<sha256> IDs from experiment receipts are also readable.'
            self.schemas[2]['function']['description']+=' In runtime v2 parameters is an optional dict injected as parameters; execution: JSON results can be requested in input_artifact_ids. To load an immutable source:<sha256>, include it in input_artifact_ids then load_module by that ID.'
            self.schemas[2]['function']['parameters']['properties']['parameters']={'type':'object','description':'Explicit check-script parameters, injected as the parameters global.'}
            self.schemas.extend([
                _schema('list_artifacts','List bound material and workspace/execution artifact IDs, sizes and hashes without reading all content.',{},[]),
                _schema('checkpoint_context','Persist your current progress, open questions and next action, then restart model context with the same goal and artifact references. Prior raw execution files stay readable. This is not verification of your interpretation.',{'summary':{'type':'string'}},['summary']),
                _schema('report_blocker','End this task as blocked, without accepting a candidate, when required information or execution is unavailable. Provide reason and actual evidence references.',{'reason':{'type':'string'},'evidence':{'type':'array','items':{'type':'string'}}},['reason','evidence']),
                _schema('run_solver_experiment','Execute a complete source module using the real solver Runner. Choose training instance_ids OR generator_id with explicit seeds. Sources are bound IDs or workspace .py files. Defaults to formal per-case solver time; shorter time_limit_s is explicitly diagnostic. Generated probes have no optimal reference and return objective, not fabricated gap. Does not rank/promote candidates. Full records remain readable by artifact ID.',
                    {'source_id':{'type':'string'},'instance_ids':{'type':'array','items':{'type':'string'}},'generator_id':{'type':'string'},
                     'seeds':{'type':'array','items':{'type':'integer'}},'parameters':{'type':'object'},'n_cities':{'type':'integer'},'time_limit_s':{'type':'number'},'trace':{'type':'boolean'},'solver_seed':{'type':'integer'},'initial_solutions':{'type':'object'}},['source_id'])])
        self.root = Path(root).resolve()
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.results = self.root / '.task_results'
        self.results.mkdir(exist_ok=True)
        self.artifacts = copy.deepcopy(artifacts)
        for key, binding in self.artifacts.items():
            if not isinstance(key, str) or not isinstance(binding, dict) or binding.get('type') not in ('text', 'json') or 'value' not in binding:
                raise ValueError('invalid artifact binding')
            if binding['type'] == 'text' and not isinstance(binding['value'], str):
                raise TypeError('text artifact must contain text')
            _json(binding['value'])
        self.submit = submit
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s must be positive and finite')
        self.timeout_s = timeout_s

    def _path(self, relative):
        if not isinstance(relative, str) or not relative or '\x00' in relative:
            raise ValueError('nonempty relative path required')
        windows = PureWindowsPath(relative)
        path = Path(relative.replace('\\', '/'))
        if (not path.parts or windows.drive or windows.root or path.is_absolute() or ':' in relative
                or '..' in path.parts or path.parts[0].startswith('.')
                or any(part.endswith(('.', ' ')) or PureWindowsPath(part).is_reserved() or part == '.task_results' for part in path.parts)):
            raise ValueError('workspace path escape or reserved path')
        target = self.workspace / path
        # Reject links even when their current target happens to remain inside.
        current = self.workspace
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('symlink workspace path denied')
        if not target.resolve().is_relative_to(self.workspace.resolve()):
            raise ValueError('workspace path escape')
        return target

    def _material(self, artifact_id):
        if artifact_id in self.artifacts:
            binding = self.artifacts[artifact_id]
            value = binding['value']
            return binding['type'], value if binding['type'] == 'text' else _json(value)
        if self.runtime_v2 and artifact_id.startswith('source:'):
            source_hash=artifact_id.split(':',1)[1]
            if not re.fullmatch('[0-9a-f]{64}',source_hash) or self.source_loader is None:
                raise ValueError('source ID must name a saved source:<sha256> receipt')
            return 'text',self.source_loader(source_hash)
        # Execution references are constrained to generated internal names.
        if artifact_id.startswith('execution:'):
            name = artifact_id.split(':', 1)[1]
            if not name or Path(name).name != name or PureWindowsPath(name).name != name or ':' in name:
                raise ValueError('invalid execution artifact ID')
            target = self.results / name
            return 'json' if name.endswith('.json') else 'text', target.read_bytes().decode('utf-8')
        return 'text', self._path(artifact_id).read_bytes().decode('utf-8')

    def dispatch(self, name, args):
        handlers = {'read_artifact': self._read, 'write_file': self._write,
                    'run_python': self._run, 'submit_candidate': self._submit}
        if self.runtime_v2:handlers.update(list_artifacts=self._list,checkpoint_context=self._checkpoint,
            report_blocker=self._block,run_solver_experiment=self._experiment)
        if self.data_measurement is not None:handlers['measure_data_candidate']=self.data_measurement
        if self.experiment_batch is not None:handlers['run_experiment_batch']=self.experiment_batch
        if self.reproduce_case is not None:handlers['reproduce_training_case']=self.reproduce_case
        if self.continue_comparison is not None:handlers['continue_comparison']=self.continue_comparison
        if self.source_comparison is not None:handlers['compare_sources']=self.source_comparison
        if self.research_update is not None:handlers['update_research']=self.research_update
        try:
            if name not in handlers:
                raise ValueError('unknown tool: ' + str(name))
            if not isinstance(args, dict):
                raise TypeError('tool arguments must be an object')
            return handlers[name](**args)
        except (ValueError, TypeError, KeyError, OSError, SyntaxError, ArithmeticError) as error:
            return {'ok': False, 'accepted': False, 'error': type(error).__name__ + ': ' + str(error)}

    def enable_data_measurement(self,callback):
        self.data_measurement=callback
        self.schemas.append(_schema('measure_data_candidate',
            'Measure a saved full rule proposal on selected zero-based indices of this round fixed formal plan. Host uses the same rule weights, parameters, city count, seeds, trusted reference backend, calibration solver and scoring as final preparation. Returns measured reference-gap percentages and numerical reference precision, pending/errors and an evidence artifact. Subset mean is not full-plan mean. Does not select data or end the task. Repeat a pending case to continue retained reference work.',
            {'candidate_path':{'type':'string','description':'Workspace relative .json proposal containing rules with name, source, parameters, positive weight.'},
             'case_indices':{'type':'array','items':{'type':'integer','minimum':0},'minItems':1,'uniqueItems':True}},
            ['candidate_path','case_indices']))

    def enable_efficiency(self,role):
        self.efficiency_role=role
        python=next(s['function'] for s in self.schemas if s['function']['name']=='run_python')
        python['parameters']['properties']['save_as']={'type':'string','description':'Optional workspace path for code: save and execute in this same call instead of separate write_file then run_python.'}
        python['description']='Execute code directly; optional save_as keeps it at a chosen path in the same call. '+python['description']
        if role=='data_designer':
            experiment=next(s['function'] for s in self.schemas if s['function']['name']=='run_solver_experiment')
            for name in ('generator_id','seeds','parameters','n_cities'):
                experiment['parameters']['properties'].pop(name,None)
            experiment['parameters']['required']=['source_id','instance_ids']
            experiment['description']='Diagnose specified existing training instance_ids. For new candidate data use measure_data_candidate, which returns trusted references and measured gaps on the formal fixed plan.'
            measured=next((s['function'] for s in self.schemas if s['function']['name']=='measure_data_candidate'),None)
            if measured:
                measured['description']+=' A fully measured formal plan that improves on the retained set under matching calibration conditions completes this data task; a subset never does. Rule source_artifact may reference an existing Python workspace/bound file instead of copying source text.'
        self.schemas.append(_schema('run_experiment_batch',
            'Run an Agent-chosen list of experiments whose inputs are already decided. Jobs run in order and share the existing worker allowance. Each job names run_solver_experiment or measure_data_candidate and its arguments. Returns actual result summaries and full evidence. If intermediate results must change the next job, use separate calls instead. Optional handoff saves current findings/open questions and restarts context after this complete batch, attaching actual results. A verified complete data goal may finish the task before remaining jobs run.',
            {'question':{'type':'string'},'jobs':{'type':'array','minItems':1,'items':{'type':'object','properties':{
                'tool':{'type':'string','enum':['run_solver_experiment','measure_data_candidate']},'arguments':{'type':'object'}},'required':['tool','arguments'],'additionalProperties':False}},
             'handoff':{'type':'string','description':'Optional research progress, working files and open questions to carry into a fresh context after the batch.'}},['question','jobs']))

    def enable_research(self,experiments):
        if experiments.data_context is not None:
            self.reproduce_case=experiments.reproduce_training_case
            self.schemas.append(_schema('reproduce_training_case','Rebuild an existing supplemental case using its actual original rule version, parameters and generator seed; compare saved coordinates. The solver seed is not used for data generation.',{'instance_id':{'type':'string'}},['instance_id']))
        self.continue_comparison=experiments.continue_comparison
        self.source_comparison=experiments.compare
        self.research_update=experiments.update_research
        self.schemas.extend([
            _schema('continue_comparison','Add new training cases to an existing comparison without rerunning previously measured cases. Use compare_sources for intentional repeat observations.',{'comparison_id':{'type':'string'},'instance_ids':{'type':'array','items':{'type':'string'}},'question':{'type':'string'}},['comparison_id','instance_ids','question']),
            _schema('compare_sources','Compare complete programs on identical training cases and solver seed. Optional initial_solutions maps every case ID to a valid tour for same-path restart, not restoration of internal search state. Returns paired objectives, percentage gaps and evidence. Does not promote candidates.',
                {'reference_source_id':{'type':'string'},'candidate_source_id':{'type':'string'},'instance_ids':{'type':'array','items':{'type':'string'}},'question':{'type':'string'},'solver_seed':{'type':'integer'},'time_limit_s':{'type':'number'},'initial_solutions':{'type':'object'}},['reference_source_id','candidate_source_id','instance_ids','question']),
            _schema('update_research','Persist measured findings and remaining questions for an exact source, choose the next research action, and compact this session to that research progress. Hypotheses remain Agent interpretations. Raw evidence and source remain readable. No reviewer or response limit.',
                {'question':{'type':'string'},'candidate_source_id':{'type':'string'},'findings':{'type':'string'},'open_questions':{'type':'array','items':{'type':'string'}},'next_action':{'type':'string','enum':['modify','measure','change_direction','submit','defer']},'evidence_ids':{'type':'array','items':{'type':'string'}}},['question','candidate_source_id','findings','open_questions','next_action','evidence_ids'])])

    def resolve_data_proposal(self,value):
        if not isinstance(value,dict):raise ValueError('candidate proposal must be a JSON object')
        value=copy.deepcopy(value)
        if value.get('action')=='keep_current':return value
        if not isinstance(value.get('rules'),list):raise ValueError('rules must be a list')
        for rule in value['rules']:
            if not isinstance(rule,dict):raise ValueError('each rule must be an object')
            if 'source_artifact' in rule:
                if 'source' in rule:raise ValueError('provide source OR source_artifact, not both')
                identifier=rule.pop('source_artifact')
                if not isinstance(identifier,str):raise ValueError('source_artifact must name a text artifact')
                kind,text=self._material(identifier)
                if kind!='text':raise ValueError('generator source_artifact must be Python text')
                rule['source']=text
        return value

    def _list(self):
        entries=[]
        for key in self.artifacts:
            kind,text=self._material(key);entries.append(dict(id=key,type=kind,characters=len(text),sha256=_digest(text)))
        for path in self.workspace.rglob('*'):
            if path.is_file():
                relative=path.relative_to(self.workspace).as_posix();self._path(relative)
                entries.append(dict(id=relative,type='workspace',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        entries.extend(dict(id='execution:'+p.name,type='execution',bytes=p.stat().st_size) for p in self.results.iterdir() if p.is_file())
        return dict(ok=True,artifacts=entries)

    def _checkpoint(self,summary):
        if not isinstance(summary,str) or not summary.strip():raise ValueError('nonempty checkpoint summary required')
        value=dict(summary=summary,artifacts=self._list()['artifacts'])
        path=self.results/(uuid.uuid4().hex+'.checkpoint.json');_atomic(path,_json(value))
        return dict(ok=True,context_checkpoint=dict(summary=summary,artifact_id='execution:'+path.name,
            workspace=[r for r in value['artifacts'] if r['type']=='workspace']))

    def store_context_result(self,content):
        """Lossless storage before pruning an old tool result from model history."""
        path=self.results/(_digest(content)+'.context_result.json')
        if path.exists() and path.read_text(encoding='utf-8')!=content:
            raise RuntimeError('context evidence changed')
        if not path.exists():_atomic(path,content)
        return 'execution:'+path.name

    def _block(self,reason,evidence):
        if not isinstance(reason,str) or not reason.strip():raise ValueError('nonempty blocker reason required')
        if not isinstance(evidence,list) or any(not isinstance(v,str) for v in evidence):raise ValueError('evidence must be artifact ID strings')
        return dict(ok=True,blocked=True,reason=reason,evidence=evidence)

    def _experiment(self,**kwargs):
        if self.experiment is None:raise RuntimeError('solver experiment capability is not bound')
        return self.experiment(**kwargs)

    def restore_completed_writes(self, records):
        """Restore missing successful writes from the durable tool-call journal.

        Call only after the session identity check. Unknown started effects or
        changed existing files require intervention; neither is replayed.
        All files are checked before any missing file is reconstructed.
        """
        completed = {record['data']['call_id'] for record in records
                     if record.get('kind') == 'tool_completed'}
        if any(record['data']['call_id'] not in completed for record in records
               if record.get('kind') == 'tool_started'):
            raise RuntimeError('unknown started tool execution; refusing workspace restore')
        latest = {}
        for record in records:
            if record.get('kind') != 'tool_completed':
                continue
            data = record['data']
            call = data.get('call')
            function = call.get('function') if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get('name')
            result = data.get('result')
            if name not in ('write_file', 'run_python', 'submit_candidate') or not isinstance(result, dict):
                continue
            # Rejected model arguments have no recorded write effects. They
            # must not be re-parsed when a later corrected call is restored.
            if name == 'write_file' and result.get('ok') is not True:
                continue
            if name == 'run_python' and result.get('ok') is not True and not result.get('script_sha256') and not result.get('saved_files'):
                continue
            if name == 'submit_candidate' and not isinstance(result.get('receipt'), dict):
                continue
            try:
                args = json.loads(function['arguments'])
                if not isinstance(args, dict):
                    raise TypeError('arguments must be an object')
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError('completed write journal arguments corrupt') from error
            updates = []
            if name == 'write_file' and result.get('ok') is True:
                updates.append((args['relative_path'], args['content'], result['sha256']))
            elif name == 'run_python':
                if isinstance(args.get('code'), str) and result.get('script_sha256'):
                    updates.append((result['script_path'], args['code'], result['script_sha256']))
                if result.get('ok') is True:
                    for saved in result.get('saved_files', []):
                        try:
                            kind, content = self._material(saved['artifact_id'])
                        except OSError as error:
                            raise RuntimeError('completed export artifact missing') from error
                        updates.append((saved['path'], content, saved['sha256']))
            elif name == 'submit_candidate' and 'content' in args:
                content = args['content']
                receipt = result['receipt']
                if not isinstance(content, str) or receipt.get('sha256') != _digest(content):
                    raise RuntimeError('inline submission receipt digest mismatch')
                expected = 'submissions/' + _digest(content)[:24] + '.json'
                if receipt.get('path') != expected:
                    raise RuntimeError('inline submission receipt path mismatch')
                updates.append((expected, content, receipt['sha256']))
            for relative, content, digest in updates:
                if not isinstance(content, str) or _digest(content) != digest:
                    raise RuntimeError('completed write journal digest mismatch')
                canonical = self._path(relative).relative_to(self.workspace).as_posix().casefold()
                latest[canonical] = (relative, content, digest)
        missing = []
        for relative, content, digest in latest.values():
            path = self._path(relative)
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError('workspace file changed since completed write: ' + relative)
            else:
                missing.append((path, content))
        for path, content in missing:
            _atomic(path, content)
        return len(missing)

    def _read(self, artifact_id, offset=0, limit=8000):
        if not isinstance(artifact_id, str):
            raise TypeError('artifact_id must be a string')
        if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 1000000:
            raise ValueError('offset >= 0 and limit in 1..1000000 required')
        kind, content = self._material(artifact_id)
        return {'ok': True, 'artifact_id': artifact_id, 'type': kind,
                'content': content[offset:offset + limit], 'offset': offset,
                'total_characters': len(content), 'has_more': offset + limit < len(content),
                'sha256': _digest(content)}

    def _write(self, relative_path, content):
        if not isinstance(content, str):
            raise TypeError('content must be text')
        target = self._path(relative_path)
        if relative_path in self.artifacts:
            raise ValueError('cannot overwrite bound artifact ID')
        _atomic(target, content)
        return {'ok': True, 'path': relative_path, 'sha256': _digest(content)}

    def _run(self, input_artifact_ids, script_path=None, code=None,parameters=None,save_as=None):
        if parameters is not None and (not self.runtime_v2 or not isinstance(parameters,dict)):
            raise ValueError('parameters requires a runtime-v2 object')
        if not isinstance(input_artifact_ids, list) or any(not isinstance(item, str) for item in input_artifact_ids):
            raise TypeError('input_artifact_ids must be a list of bound IDs')
        if (script_path is None) == (code is None):
            raise ValueError('provide exactly one of code or script_path')
        if save_as is not None and (self.efficiency_role is None or code is None):
            raise ValueError('save_as requires direct code in an efficiency-enabled task')
        if code is not None:
            if not isinstance(code, str):
                raise TypeError('code must be text')
            script_path = save_as if save_as is not None else 'checks/' + _digest(code)[:24] + '.py'
            self._write(script_path, code)
        script = self._path(script_path).read_bytes().decode('utf-8')
        inputs = {}
        input_source_sha256 = {}
        for key in input_artifact_ids:
            if key in self.artifacts:
                inputs[key] = copy.deepcopy(self.artifacts[key]['value'])
                binding = self.artifacts[key]
                input_source_sha256[key] = _digest(binding['value'] if binding['type'] == 'text' else _json(binding['value']))
            elif self.runtime_v2 and key.startswith(('execution:','source:')):
                kind,content=self._material(key)
                inputs[key]=json.loads(content) if kind=='json' else content
                input_source_sha256[key]=_digest(content)
            else:
                # Workspace capability only: execution/internal result IDs are
                # intentionally read through read_artifact, never host paths.
                content = self._path(key).read_bytes().decode('utf-8')
                inputs[key] = json.loads(content) if key.lower().endswith('.json') else content
                input_source_sha256[key] = _digest(content)
        sources = {key: binding['value'] for key, binding in self.artifacts.items() if binding['type'] == 'text'}
        if self.runtime_v2:
            sources.update({key:inputs[key] for key in input_artifact_ids if key.startswith('source:')})
        for path in self.workspace.rglob('*.py'):
            relative = path.relative_to(self.workspace).as_posix()
            self._path(relative)
            if relative not in sources:
                sources[relative] = path.read_bytes().decode('utf-8')
        execution_id = uuid.uuid4().hex
        stdout_path = self.results / (execution_id + '.stdout.txt')
        stderr_path = self.results / (execution_id + '.stderr.txt')
        record_path = self.results / (execution_id + '.json')
        raw_path = self.results / (execution_id + '.worker.json')
        record = {'ok': False, 'result': None, 'error': None, 'timeout': False, 'saved_files': [],
                  'exit_code': None, 'script_path': script_path, 'script_sha256': _digest(script),
                  'input_sha256': {key: _digest(_json(value)) for key, value in inputs.items()},
                  'input_source_sha256': input_source_sha256,
                  'input_digest_semantics': 'input_sha256 hashes canonical JSON of injected value; input_source_sha256 hashes raw UTF-8 workspace/text material, canonical JSON for bound JSON.',
                  'source_sha256': {key: _digest(value) for key, value in sources.items()},
                  'execution_artifact_id': 'execution:' + record_path.name,
                  'stdout_artifact_id': 'execution:' + stdout_path.name,
                  'stderr_artifact_id': 'execution:' + stderr_path.name,
                  'isolation': 'Fresh process, restricted AST/import/builtins, audit IO denial; Windows Job resource and tree limits. Not a hostile-code OS sandbox.'}
        if self.runtime_v2:record['parameters']=parameters or {}
        # Preserve the attempted binding even if process setup fails.
        _atomic(record_path, _json(record))
        job = None
        process = None
        start = time.monotonic()
        env = {key: value for key, value in os.environ.items() if not any(word in key.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PROXY'))}
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')
        try:
            with tempfile.TemporaryDirectory(prefix='cef_task_') as cwd, stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
                process = subprocess.Popen([sys.executable, '-I', '-u', str(Path(__file__).with_name('task_worker.py')), str(raw_path)],
                                           stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, cwd=cwd, env=env)
                if sys.platform == 'win32':
                    from core.windows_job import WindowsJob
                    from core import windows_job as wj
                    import ctypes
                    job = WindowsJob.create()
                    job.assign_pid(process.pid)
                    info = wj._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                    info.BasicLimitInformation.LimitFlags = 0x2000 | 0x100
                    info.ProcessMemoryLimit = 1024 * 1024 * 1024
                    if not wj._kernel32.SetInformationJobObject(job._handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                        raise RuntimeError('task worker memory cap failed')
                payload = dict(script=script, script_path=script_path, inputs=inputs, sources=sources,parameters=parameters or {})
                try:
                    process.communicate((_json(payload) + '\n').encode('utf-8'), timeout=self.timeout_s)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    record.update(timeout=True, error='TimeoutError: task execution deadline')
                record['exit_code'] = process.returncode
                if not record['timeout']:
                    if raw_path.exists():
                        record.update(json.loads(raw_path.read_text(encoding='utf-8')))
                    else:
                        raise RuntimeError('worker exited without a result')
        except Exception as error:
            record.update(ok=False, error=type(error).__name__ + ': ' + str(error))
            _atomic(record_path, _json(record))
            # Spawn/Job/transport failures have an unknown execution outcome.
            # Propagate so ToolSession preserves a started, uncompleted call.
            raise RuntimeError('task execution infrastructure failed: ' + record['error']) from error
        finally:
            if process and process.poll() is None:
                process.kill()
                process.wait()
            if job:
                try:
                    job.terminate_and_wait(timeout_sec=2)
                finally:
                    job.close()
            if process and process.stdin and not process.stdin.closed:
                process.stdin.close()
        record['wallclock_s'] = time.monotonic() - start
        for label, path in [('stdout', stdout_path), ('stderr', stderr_path)]:
            content = path.read_bytes().decode('utf-8', errors='replace') if path.exists() else ''
            record[label] = content
        exports = record.pop('files', {})
        if record['ok']:
            try:
                validated = self._validate_exports(exports)
            except (ValueError, TypeError) as error:
                record.update(ok=False, error=type(error).__name__ + ': ' + str(error))
            else:
                try:
                    for index, (relative, path, content) in enumerate(validated):
                        artifact_path = self.results / (execution_id + '.export.%d.txt' % index)
                        _atomic(artifact_path, content)
                        _atomic(path, content)
                        record['saved_files'].append({'path': relative, 'sha256': _digest(content),
                                                      'artifact_id': 'execution:' + artifact_path.name})
                except OSError as error:
                    record.update(ok=False, error=type(error).__name__ + ': ' + str(error))
                    _atomic(record_path, _json(record))
                    raise RuntimeError('export persistence infrastructure failed: ' + str(error)) from error
        _atomic(record_path, _json(record))
        result = dict(record)
        for label in ('stdout', 'stderr'):
            result[label + '_truncated'] = len(result[label]) > 16000
            result[label] = result[label][:16000]
        result['result_truncated'] = len(_json(result['result'])) > 16000
        if result['result_truncated']:
            result['result'] = None
        return result

    def _validate_exports(self, files):
        if not isinstance(files, dict) or len(files) > 128:
            raise ValueError('files must be an object with at most 128 exports')
        validated, seen, size = [], set(), 0
        for relative, content in files.items():
            if not isinstance(relative, str) or not isinstance(content, str):
                raise TypeError('export paths and contents must be text')
            path = self._path(relative)
            if path.exists() and not path.is_file():
                raise ValueError('export target must be a file: ' + relative)
            if any(parent.exists() and not parent.is_dir() for parent in path.parents if parent.is_relative_to(self.workspace)):
                raise ValueError('export parent is not a directory: ' + relative)
            canonical = path.relative_to(self.workspace).as_posix().casefold()
            if canonical in seen:
                raise ValueError('duplicate export path: ' + relative)
            if relative in self.artifacts:
                raise ValueError('cannot overwrite bound artifact ID')
            seen.add(canonical)
            size += len(content.encode('utf-8'))
            if size > 2 * 1024 * 1024:
                raise ValueError('total exports exceed 2 MiB UTF-8 limit')
            validated.append((relative, path, content))
        targets = [entry[1] for entry in validated]
        if any(left != right and left in right.parents for left in targets for right in targets):
            raise ValueError('export file/directory path collision')
        return validated

    def _submit(self, path=_UNSET, content=_UNSET):
        if (path is _UNSET) == (content is _UNSET):
            raise ValueError('provide exactly one of path or content')
        if content is not _UNSET:
            if not isinstance(content, str):
                raise TypeError('content must be UTF-8 text')
            path = 'submissions/' + _digest(content)[:24] + '.json'
            target = self._path(path)
            if target.exists() and target.read_bytes() != content.encode('utf-8'):
                raise RuntimeError('inline submission file changed after persistence')
            if not target.exists():
                self._write(path, content)
        target = self._path(path)
        content = target.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        frozen = self.results / (uuid.uuid4().hex + '.submitted' + target.suffix)
        _atomic(frozen, content)
        receipt = {'path': path, 'sha256': digest, 'frozen_path': str(frozen)}
        try:
            result = self.submit(frozen)
        except (ValueError, TypeError, KeyError, SyntaxError, AssertionError) as error:
            return {'ok': False, 'accepted': False, 'error': type(error).__name__ + ': ' + str(error), 'receipt': receipt}
        except OSError as error:
            raise RuntimeError('submission infrastructure failed: ' + str(error)) from error
        if target.read_bytes() != content or frozen.read_bytes() != content:
            return {'ok': False, 'accepted': False, 'error': 'candidate changed during submission validation', 'receipt': receipt}
        if not isinstance(result, dict) or type(result.get('accepted')) is not bool:
            raise ValueError('submission callback must return accepted boolean')
        if result['accepted'] and 'value' not in result:
            raise ValueError('accepted submission requires value')
        _json(result)
        return dict(result, ok=result['accepted'], receipt=receipt)
