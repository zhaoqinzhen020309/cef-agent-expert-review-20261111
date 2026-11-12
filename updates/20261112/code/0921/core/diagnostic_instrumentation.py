from __future__ import annotations

import ast
import time
from pathlib import Path


COUNTER_NAMES = (
    "initial_solutions",
    "two_opt_attempts",
    "two_opt_accepts",
    "relocate_attempts",
    "relocate_accepts",
    "or_opt_attempts",
    "or_opt_accepts",
    "double_bridge_calls",
    "main_loop_iterations",
)


def build_diagnostic_spec(candidate_path: Path) -> dict[str, object]:
    tree = ast.parse(
        candidate_path.read_text(encoding="utf-8"),
        filename=str(candidate_path),
    )
    return _diagnostic_spec(tree)


def instrument_diagnostic_tree(tree: ast.Module) -> ast.Module:
    return DiagnosticInstrumenter(_diagnostic_spec(tree)).visit(tree)


def finish_diagnostics(
    counters: object,
    *,
    deadline: float,
    effective_iteration_limit: int | None = None,
    current_time: float | None = None,
) -> dict[str, object]:
    if not isinstance(counters, dict):
        raise RuntimeError("diagnostic instrumentation did not initialize counters")
    record = {name: int(counters.get(name, 0)) for name in COUNTER_NAMES}
    observed_time = time.time() if current_time is None else current_time
    if observed_time >= deadline - 0.01:
        reason = "deadline_reached"
    else:
        reason = "solver_returned_before_deadline"
    if effective_iteration_limit is not None:
        record["effective_iteration_limit"] = int(effective_iteration_limit)
    return {**record, "termination_reason": reason}


def _diagnostic_spec(tree: ast.Module) -> dict[str, object]:
    attempt_lines: dict[int, str] = {}
    accept_lines: dict[int, str] = {}
    iteration_lines: set[int] = set()
    call_kinds = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.args.args or node.args.args[0].arg != "self":
            continue
        method_kind = _operator_from_method(node.name)
        if "greedy" in node.name:
            call_kinds[node.name] = "initial_solutions"
        if node.name in {"_kick", "double_bridge"}:
            call_kinds[node.name] = "double_bridge_calls"
        for descendant in ast.walk(node):
            if isinstance(descendant, (ast.Assign, ast.AnnAssign)):
                targets = (
                    descendant.targets
                    if isinstance(descendant, ast.Assign)
                    else [descendant.target]
                )
                if any(
                    isinstance(target, ast.Name) and target.id == "delta"
                    for target in targets
                ):
                    kind = method_kind or _operator_from_expression(descendant)
                    if kind is not None:
                        attempt_lines[descendant.lineno] = kind
            if isinstance(descendant, ast.If) and _tests_delta(descendant.test):
                preceding = [
                    (line, kind)
                    for line, kind in attempt_lines.items()
                    if node.lineno <= line < descendant.lineno
                ]
                kind = (
                    method_kind
                    or _operator_from_expression(descendant)
                    or (max(preceding)[1] if preceding else None)
                )
                if kind is not None and descendant.body:
                    accept_lines[descendant.body[0].lineno] = kind
            if (
                node.name == "solve"
                and isinstance(descendant, ast.For)
                and _is_main_iteration_loop(descendant)
                and descendant.body
            ):
                iteration_lines.add(descendant.body[0].lineno)
    return {
        "attempt_lines": attempt_lines,
        "accept_lines": accept_lines,
        "iteration_lines": iteration_lines,
        "call_kinds": call_kinds,
    }


class DiagnosticInstrumenter(ast.NodeTransformer):
    def __init__(self, spec: dict[str, object]) -> None:
        self.attempt_lines = spec["attempt_lines"]
        self.accept_lines = spec["accept_lines"]
        self.iteration_lines = spec["iteration_lines"]
        self.call_kinds = spec["call_kinds"]

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if not isinstance(value, ast.AST):
                        new_values.append(value)
                        continue
                    value = self.visit(value)
                    if value is None:
                        continue
                    values = value if isinstance(value, list) else [value]
                    for current in values:
                        if isinstance(current, ast.stmt):
                            new_values.extend(self._line_counters(current))
                        new_values.append(current)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        if not node.args.args or node.args.args[0].arg != "self":
            return node
        prefix = []
        if node.name == "solve":
            prefix.append(_diagnostic_initializer(node))
        call_counter = self.call_kinds.get(node.name)
        if call_counter is not None:
            prefix.append(_counter_increment(call_counter, node))
        insert_at = int(bool(node.body and _is_docstring(node.body[0])))
        node.body[insert_at:insert_at] = prefix
        return node

    def _line_counters(self, statement: ast.stmt) -> list[ast.stmt]:
        line = getattr(statement, "lineno", -1)
        names = []
        attempt = self.attempt_lines.get(line)
        if attempt is not None:
            names.append(f"{attempt}_attempts")
        accepted = self.accept_lines.get(line)
        if accepted is not None:
            names.append(f"{accepted}_accepts")
        if line in self.iteration_lines:
            names.append("main_loop_iterations")
        return [_counter_increment(name, statement) for name in names]


def _diagnostic_initializer(anchor: ast.AST) -> ast.stmt:
    statement = ast.Assign(
        targets=[
            ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr="_ahd_diagnostics",
                ctx=ast.Store(),
            )
        ],
        value=ast.Dict(
            keys=[ast.Constant(value=name) for name in COUNTER_NAMES],
            values=[ast.Constant(value=0) for _ in COUNTER_NAMES],
        ),
    )
    return ast.copy_location(statement, anchor)


def _counter_increment(name: str, anchor: ast.AST) -> ast.stmt:
    target = ast.Subscript(
        value=ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr="_ahd_diagnostics",
            ctx=ast.Load(),
        ),
        slice=ast.Constant(value=name),
        ctx=ast.Store(),
    )
    return ast.copy_location(
        ast.AugAssign(target=target, op=ast.Add(), value=ast.Constant(value=1)),
        anchor,
    )


def _operator_from_method(name: str) -> str | None:
    lowered = name.lower()
    if "two_opt" in lowered or "2_opt" in lowered:
        return "two_opt"
    if "relocate" in lowered:
        return "relocate"
    if "or_opt" in lowered:
        return "or_opt"
    return None


def _operator_from_expression(node: ast.AST) -> str | None:
    names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    if {"segment_length", "first", "last"} & names:
        return "or_opt"
    if {"prev_node", "after", "after_next"} & names:
        return "relocate"
    if {"a", "b", "c", "d"}.issubset(names):
        return "two_opt"
    return None


def _tests_delta(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Name) and item.id == "delta"
        for item in ast.walk(node)
    )


def _is_main_iteration_loop(node: ast.For) -> bool:
    names = {item.id for item in ast.walk(node.iter) if isinstance(item, ast.Name)}
    return bool(names & {"iteration_limit", "limit"})


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )
