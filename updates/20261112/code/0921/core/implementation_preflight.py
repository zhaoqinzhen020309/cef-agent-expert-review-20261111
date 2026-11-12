"""Cheap source checks before a model response is committed; never run code.

Checks Python lexical names and unambiguous calls to module-level helpers.
Unknown decorators, splatted calls and shadowed helpers are deliberately left
to the isolated evaluator, since their call signatures cannot be inferred here.
"""
import ast
import builtins
import inspect
import symtable


class _ConstantConditional(ast.NodeTransformer):
    def visit_IfExp(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node


def _check_numeric_buffers_and_jit(tree, filename):
    numpy_names = {a.asname or a.name for n in tree.body if isinstance(n, ast.Import)
                   for a in n.names if a.name == 'numpy'}
    jit_names = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'numba'
                 for a in n.names if a.name == 'njit'}
    readonly_numpy = {'argsort', 'argpartition', 'argmin', 'argmax', 'min', 'max', 'sum',
                      'mean', 'std', 'median', 'isfinite', 'isnan', 'isinf', 'all', 'any'}
    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        nodes = list(ast.walk(function))
        jitted = any(isinstance(d.func if isinstance(d, ast.Call) else d, ast.Name)
                     and (d.func if isinstance(d, ast.Call) else d).id in jit_names
                     for d in function.decorator_list)
        if jitted:
            for call in (n for n in nodes if isinstance(n, ast.Call)):
                if (isinstance(call.func, ast.Attribute) and call.func.attr == 'argsort'
                        and isinstance(call.func.value, ast.Name) and call.func.value.id in numpy_names):
                    for keyword in call.keywords:
                        if (keyword.arg == 'kind' and isinstance(keyword.value, ast.Constant)
                                and keyword.value.value not in ('quicksort', 'mergesort')):
                            raise ValueError(f'{filename}:{call.lineno}: Numba argsort does not support kind={keyword.value.value!r}; use mergesort for stable ordering')
        for assignment in (n for n in nodes if isinstance(n, ast.Assign) and len(n.targets) == 1):
            target, value = assignment.targets[0], assignment.value
            if not (isinstance(target, ast.Name) and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute) and value.func.attr in ('empty', 'empty_like')
                    and isinstance(value.func.value, ast.Name) and value.func.value.id in numpy_names):
                continue
            name = target.id
            # Deliberately conservative: opaque helper calls, aliasing and any
            # explicit write make initialization a dynamic evaluator concern.
            escaped = False
            written = False
            read = None
            for node in nodes:
                if isinstance(node, ast.Assign) and node is not assignment:
                    if isinstance(node.value, ast.Name) and node.value.id == name:
                        escaped = True
                    if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                        written = True
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == name:
                    if isinstance(node.ctx, ast.Store):
                        written = True
                    elif isinstance(node.ctx, ast.Load):
                        read = node
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == name:
                        if node.func.attr == 'fill':
                            written = True
                        elif node.func.attr not in ('copy', 'sum', 'mean', 'min', 'max', 'argsort', 'argmin', 'argmax'):
                            escaped = True
                        else:
                            read = node
                    arguments = node.args + [k.value for k in node.keywords]
                    if any(any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(a)) for a in arguments):
                        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                                and node.func.value.id in numpy_names and node.func.attr in readonly_numpy):
                            read = node
                        elif not (isinstance(node.func, ast.Name) and node.func.id in ('len', 'range')):
                            escaped = True
            if read is not None and not written and not escaped:
                raise ValueError(f'{filename}:{read.lineno}: uninitialized NumPy buffer {name!r} is read but never written in {function.name}; initialize the used elements before reading')


def validate_implementation_source(source, filename="solver.py"):
    tree = _ConstantConditional().visit(ast.parse(source, filename=filename))
    _check_numeric_buffers_and_jit(tree, filename)
    symbols = symtable.symtable(ast.unparse(tree), filename, "exec")
    known = set(dir(builtins)) | {"__name__", "__file__", "__package__", "__doc__",
                                 "__annotations__", "__builtins__", "__spec__", "__loader__", "__cached__"}
    known.update(s.get_name() for s in symbols.get_symbols()
                 if s.is_assigned() or s.is_imported() or s.is_namespace())
    wildcard = any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
                   for n in ast.walk(tree))
    if not wildcard:
        pending = list(symbols.get_children())
        while pending:
            table = pending.pop()
            pending.extend(table.get_children())
            for symbol in table.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known:
                    raise ValueError(f"{filename}: undefined global name {symbol.get_name()!r} in {table.get_name()}")
    definitions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            definitions.setdefault(node.name, []).append(node)
    # A local binding with the same name makes global call resolution ambiguous.
    shadowed = set()
    pending = list(symbols.get_children())
    while pending:
        table = pending.pop()
        pending.extend(table.get_children())
        shadowed.update(s.get_name() for s in table.get_symbols() if s.is_local())
    reassigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    signatures = {}
    for name, nodes in definitions.items():
        if len(nodes) != 1 or name in shadowed or name in reassigned:
            continue
        node = nodes[0]
        def cpu_jit(decorator):
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            return isinstance(target, ast.Name) and target.id == "njit"
        if any(not cpu_jit(d) for d in node.decorator_list):
            continue
        params = []
        positional = node.args.posonlyargs + node.args.args
        required = len(positional) - len(node.args.defaults)
        for i, arg in enumerate(positional):
            kind = inspect.Parameter.POSITIONAL_ONLY if i < len(node.args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
            params.append(inspect.Parameter(arg.arg, kind, default=inspect.Parameter.empty if i < required else None))
        if node.args.vararg:
            params.append(inspect.Parameter(node.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            params.append(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                                           default=inspect.Parameter.empty if default is None else None))
        if node.args.kwarg:
            params.append(inspect.Parameter(node.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
        signatures[name] = inspect.Signature(params)
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id not in signatures:
            continue
        if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
            continue
        try:
            signatures[call.func.id].bind(*([None] * len(call.args)), **{k.arg: None for k in call.keywords})
        except TypeError as exc:
            raise ValueError(f"{filename}:{call.lineno}: helper {call.func.id} call does not match its signature: {exc}") from exc
