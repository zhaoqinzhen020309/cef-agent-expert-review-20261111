"""Task checks in a fresh restricted interpreter; not a hostile-code OS sandbox.

Only the trusted loader compiles supplied source. Scripts receive ``inputs``,
``load_module(id)`` and may set ``result``. No model-authored IO is permitted.
"""
import ast
import builtins
import collections
import functools
import heapq
import itertools
import json
import math
import random
import sys
import time
import traceback

import numpy as np

SAFE_MODULES = dict(math=math, random=random, time=time, itertools=itertools,
                    functools=functools, collections=collections, heapq=heapq,
                    numpy=np, json=json)
DENIED_BUILTINS=frozenset({'open','eval','exec','compile','input','breakpoint','help','globals','locals',
                          'vars','getattr','setattr','delattr','__import__'})


def capability_manifest():
    return dict(language='Python',imports=sorted(SAFE_MODULES),
        injected_globals=dict(inputs='dict: requested JSON artifacts are parsed values; text artifacts are str',
            parameters='dict supplied to run_python',load_module='callable returning dict of callable exports',
            result='optional JSON-compatible output',files='optional dict of workspace path -> UTF-8 text'),
        unavailable_builtins=sorted(DENIED_BUILTINS-{'__import__'}),
        denied='filesystem, network, process calls, dunder names/attributes; imports only from declared modules',
        examples=[dict(code="import numpy as np\ncases=inputs['training_cases.json']\nresult={'count':len(cases),'fields':sorted(cases[0])}",
            input_artifact_ids=['training_cases.json']),dict(code="module=load_module('parent.py')\nresult=sorted(module)",
            input_artifact_ids=[])])


def main():
    request = json.loads(sys.stdin.buffer.readline().decode('utf-8'))
    sys.stdout.reconfigure(encoding='utf-8', newline='\n')
    sys.stderr.reconfigure(encoding='utf-8', newline='\n')
    # Preopen before capability restrictions. This handle is never model-visible.
    output = open(sys.argv[1], 'w', encoding='utf-8', newline='\n')
    safe_builtins = {key: value for key, value in vars(builtins).items() if key not in DENIED_BUILTINS}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not level and name.startswith('numpy.') and name in sys.modules:
            return sys.modules[name] if fromlist else np
        if level or name not in SAFE_MODULES:
            raise ImportError('import denied: ' + str(name))
        return SAFE_MODULES[name]

    safe_builtins['__import__'] = safe_import

    def execute(source, filename):
        tree = ast.parse(source, filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                raise ValueError('dunder introspection denied: ' + node.attr)
            if isinstance(node, ast.Name) and node.id.startswith('__'):
                raise ValueError('dunder name denied: ' + node.id)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                if any(name not in SAFE_MODULES for name in names) or getattr(node, 'level', 0):
                    raise ValueError('import denied')
        scope = {'__builtins__': safe_builtins, '__name__': 'cef_task',
                 'inputs': request['inputs'], 'parameters':request.get('parameters',{}),'load_module': load_module}
        exec(compile(tree, filename, 'exec'), scope)
        return scope

    loaded = {}

    def load_module(artifact_id):
        if artifact_id not in request['sources']:
            raise ValueError('source is not a bound text artifact or workspace .py: ' + str(artifact_id))
        if artifact_id not in loaded:
            scope = execute(request['sources'][artifact_id], artifact_id)
            loaded[artifact_id] = {key: value for key, value in scope.items()
                                   if callable(value) and key not in {'load_module'}}
        return dict(loaded[artifact_id])

    def audit(event, args):
        if event == 'open' or event.startswith(('socket.', 'subprocess.', 'os.', 'ctypes.', 'winreg.', 'shutil.')):
            raise PermissionError('task capability denied: ' + event)

    sys.addaudithook(audit)
    random.seed(0)
    np.random.seed(0)
    start = time.monotonic()
    try:
        scope = execute(request['script'], request['script_path'])
        value = scope.get('result')
        # Fail visibly on values that cannot be represented, never fabricate None.
        encoded = json.dumps(value, allow_nan=False, default=json_value)
        files = scope.get('files', {})
        if not isinstance(files, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in files.items()):
            raise TypeError('files must map text paths to text contents')
        if len(files) > 128 or sum(len(value.encode('utf-8')) for value in files.values()) > 2 * 1024 * 1024:
            raise ValueError('exports limited to 128 files and 2 MiB UTF-8 total')
        record = {'ok': True, 'result': json.loads(encoded), 'error': None, 'files': files}
    except BaseException as error:
        locations = ['  %s:%s in %s' % (frame.f_code.co_filename, line, frame.f_code.co_name)
                     for frame, line in traceback.walk_tb(error.__traceback__)]
        message = type(error).__name__ + ': ' + str(error)
        sys.stderr.write('\n'.join(locations) + '\n' + message + '\n')
        record = {'ok': False, 'result': None, 'error': message}
    record['elapsed_s'] = time.monotonic() - start
    output.write(json.dumps(record, allow_nan=False))
    output.close()
    return 0 if record['ok'] else 1


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError('non-JSON result: ' + type(value).__name__)


if __name__ == '__main__':
    sys.exit(main())
