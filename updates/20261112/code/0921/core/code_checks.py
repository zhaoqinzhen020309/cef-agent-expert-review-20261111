from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from agent_ahd.core.research_records import ScopeSpec
from agent_ahd.problems.base import EntryContract

ENTRY_CLASS_NAME = "SearchStrategy"
ENTRY_METHOD_NAME = "solve"
ENTRY_METHOD_PARAMETERS = (
    "self",
    "distance_matrix",
    "deadline",
    "iteration_limit",
    "seed",
)
DEFAULT_ENTRY_CONTRACT = EntryContract(
    class_name=ENTRY_CLASS_NAME,
    method_name=ENTRY_METHOD_NAME,
    parameters=ENTRY_METHOD_PARAMETERS,
)
SAFE_IMPORT_ROOTS = {
    "__future__",
    "array",
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "functools",
    "heapq",
    "itertools",
    "math",
    "numpy",
    "operator",
    "random",
    "statistics",
    "time",
    "typing",
}
PROTECTED_IMPORT_ROOTS = {"agent_ahd"}
UNSAFE_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
UNSAFE_ATTRIBUTE_NAMES = {
    "connect",
    "fork",
    "fromfile",
    "genfromtxt",
    "load",
    "loadtxt",
    "memmap",
    "popen",
    "request",
    "save",
    "spawn",
    "system",
    "tofile",
    "urlopen",
}


@dataclass(frozen=True)
class CodeCheckResult:
    valid: bool
    error_type: str | None = None
    error_message: str | None = None


def check_candidate_source(
    source_path: str | Path,
    candidate_root: str | Path | None = None,
    entry_contract: EntryContract = DEFAULT_ENTRY_CONTRACT,
) -> CodeCheckResult:
    path = Path(source_path).resolve()
    if candidate_root is not None:
        root = Path(candidate_root).resolve()
        if not path.is_relative_to(root):
            return CodeCheckResult(
                False,
                "candidate_path_escape",
                f"candidate source {path} is outside {root}",
            )

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return CodeCheckResult(False, "syntax_error", str(exc))
    except OSError as exc:
        return CodeCheckResult(False, "read_error", str(exc))

    entry_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == entry_contract.class_name
    ]
    if len(entry_classes) != 1:
        return CodeCheckResult(
            False,
            "missing_entry_class",
            f"candidate must define exactly one class {entry_contract.class_name}",
        )

    entry_check = _check_entry_class(entry_classes[0], entry_contract)
    if not entry_check.valid:
        return entry_check

    root = Path(candidate_root).resolve() if candidate_root is not None else path.parent
    return _check_candidate_files(root)


def _check_entry_class(
    node: ast.ClassDef, entry_contract: EntryContract
) -> CodeCheckResult:
    if node.bases or node.keywords or node.decorator_list:
        return CodeCheckResult(
            False,
            "invalid_entry_class",
            f"{entry_contract.class_name} must be a complete class without inheritance",
        )

    solve_methods = [
        item
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == entry_contract.method_name
    ]
    if len(solve_methods) != 1:
        return CodeCheckResult(
            False,
            "invalid_entry_method",
            f"{entry_contract.class_name} must define exactly one {entry_contract.method_name} method",
        )

    method = solve_methods[0]
    positional = [*method.args.posonlyargs, *method.args.args]
    names = tuple(argument.arg for argument in positional)
    if (
        names != entry_contract.parameters
        or method.args.vararg is not None
        or method.args.kwarg is not None
        or method.args.kwonlyargs
        or method.args.defaults
        or method.args.kw_defaults
        or method.decorator_list
    ):
        expected = ", ".join(entry_contract.parameters)
        return CodeCheckResult(
            False,
            "invalid_entry_method",
            f"{entry_contract.method_name} must have signature ({expected})",
        )

    return CodeCheckResult(True)


def changed_symbols_in_scope(
    before_root: str | Path,
    after_root: str | Path,
    scope: ScopeSpec,
) -> tuple[str, ...]:
    before_symbols, after_symbols = _tree_symbol_fingerprints(
        Path(before_root),
        Path(after_root),
    )
    known = set(before_symbols) | set(after_symbols)
    changed = {
        name
        for name in known
        if before_symbols.get(name) != after_symbols.get(name)
    }
    if scope.kind == "full_solver":
        return tuple(sorted(changed))
    allowed = set(scope.symbols)
    return tuple(sorted(
        name for name in changed if _symbol_is_allowed(name, allowed, known)
    ))


def check_scope_tree_change(
    before_root: str | Path,
    after_root: str | Path,
    scope: ScopeSpec,
) -> CodeCheckResult:
    if scope.kind == "full_solver":
        return CodeCheckResult(True)
    try:
        before_symbols, after_symbols = _tree_symbol_fingerprints(
            Path(before_root),
            Path(after_root),
        )
    except SyntaxError as exc:
        return CodeCheckResult(False, "syntax_error", str(exc))
    except OSError as exc:
        return CodeCheckResult(False, "read_error", str(exc))
    known = set(before_symbols) | set(after_symbols)
    changed = {
        name
        for name in known
        if before_symbols.get(name) != after_symbols.get(name)
    }
    allowed = set(scope.symbols)
    outside = sorted(
        name
        for name in changed
        if not _symbol_is_allowed(name, allowed, known)
    )
    if outside:
        return CodeCheckResult(
            False,
            "scope_violation",
            f"changed symbols outside declared scope: {outside}",
        )
    return CodeCheckResult(True)


def check_ablation_reversion(
    parent_root: str | Path,
    mechanism_root: str | Path,
    ablation_root: str | Path,
    scope: ScopeSpec,
) -> CodeCheckResult:
    scope_check = check_scope_tree_change(mechanism_root, ablation_root, scope)
    if not scope_check.valid:
        return CodeCheckResult(
            False,
            "invalid_ablation",
            scope_check.error_message,
        )
    try:
        parent, mechanism = _tree_symbol_fingerprints(
            Path(parent_root),
            Path(mechanism_root),
            other_roots=(Path(ablation_root),),
        )
        _parent_again, ablation = _tree_symbol_fingerprints(
            Path(parent_root),
            Path(ablation_root),
            other_roots=(Path(mechanism_root),),
        )
    except (OSError, SyntaxError) as exc:
        return CodeCheckResult(False, "invalid_ablation", str(exc))
    known = set(parent) | set(mechanism) | set(ablation)
    allowed = set(scope.symbols)
    mechanism_delta = {
        symbol
        for symbol in known
        if parent.get(symbol) != mechanism.get(symbol)
        and (
            scope.kind == "full_solver"
            or _symbol_is_allowed(symbol, allowed, known)
        )
    }
    if not mechanism_delta:
        return CodeCheckResult(
            False,
            "invalid_ablation",
            "registered ablation scope contains no parent-to-candidate change",
        )
    mismatches = []
    for symbol in known:
        expected = (
            parent.get(symbol)
            if symbol in mechanism_delta
            else mechanism.get(symbol)
        )
        if ablation.get(symbol) != expected:
            mismatches.append(symbol)
    if mismatches:
        return CodeCheckResult(
            False,
            "invalid_ablation",
            "ablation does not exactly restore the registered mechanism delta: "
            f"{sorted(mismatches)}",
        )
    return CodeCheckResult(True)


def _tree_symbol_fingerprints(
    before: Path,
    after: Path,
    *,
    other_roots: tuple[Path, ...] = (),
) -> tuple[dict[str, str], dict[str, str]]:
    # Ablation compares three trees. Use the same path universe for both
    # comparisons, including a helper that the mechanism added then removed.
    # Otherwise an absent file becomes an empty AST on one side and a missing
    # dictionary key on the other, incorrectly rejecting an exact restoration.
    relative_paths = {
        path.relative_to(root)
        for root in (before, after, *other_roots)
        if root.is_dir()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    before_symbols: dict[str, str] = {}
    after_symbols: dict[str, str] = {}
    for relative in sorted(relative_paths):
        for root, target in ((before, before_symbols), (after, after_symbols)):
            path = root / relative
            tree = (
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                if path.is_file()
                else ast.Module(body=[], type_ignores=[])
            )
            prefix = "" if relative.as_posix() == "solver.py" else f"{relative.as_posix()}:"
            target.update({
                f"{prefix}{name}": fingerprint
                for name, fingerprint in symbol_fingerprints(tree).items()
            })
    return before_symbols, after_symbols


def symbol_fingerprints(tree: ast.Module) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    fingerprints["module:imports"] = ast.dump(ast.Module(body=imports, type_ignores=[]))
    module_other = [
        node
        for node in tree.body
        if not isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        )
    ]
    fingerprints["module:other"] = ast.dump(
        ast.Module(body=module_other, type_ignores=[]),
        include_attributes=False,
    )
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fingerprints[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            class_other = [
                item
                for item in node.body
                if not isinstance(
                    item,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.Assign,
                        ast.AnnAssign,
                    ),
                )
            ]
            fingerprints[f"{node.name}:structure"] = repr((
                tuple(ast.dump(item, include_attributes=False) for item in node.bases),
                tuple(
                    ast.dump(item, include_attributes=False) for item in node.keywords
                ),
                tuple(
                    ast.dump(item, include_attributes=False)
                    for item in node.decorator_list
                ),
                tuple(
                    ast.dump(item, include_attributes=False) for item in class_other
                ),
            ))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fingerprints[f"{node.name}.{item.name}"] = ast.dump(
                        item, include_attributes=False
                    )
                elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                    fingerprints[f"{node.name}:attributes"] = fingerprints.get(
                        f"{node.name}:attributes", ""
                    ) + ast.dump(item, include_attributes=False)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            fingerprints["module:constants"] = fingerprints.get(
                "module:constants", ""
            ) + ast.dump(node, include_attributes=False)
    return fingerprints


def _symbol_is_allowed(
    symbol: str,
    allowed: set[str],
    known_symbols: set[str],
) -> bool:
    if symbol in allowed:
        return True
    short_name = symbol.rsplit(".", 1)[-1]
    if short_name not in allowed:
        return False
    matching_symbols = {
        candidate
        for candidate in known_symbols
        if candidate.rsplit(".", 1)[-1] == short_name
    }
    return matching_symbols == {symbol}


def _check_candidate_files(root: Path) -> CodeCheckResult:
    local_roots = {path.stem for path in root.glob("*.py")}
    local_roots.update(
        path.name for path in root.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    )

    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            return CodeCheckResult(False, "syntax_error", str(exc))
        except OSError as exc:
            return CodeCheckResult(False, "read_error", str(exc))

        result = _check_tree_safety(tree, path, local_roots)
        if not result.valid:
            return result
    return CodeCheckResult(True)


def _check_tree_safety(
    tree: ast.AST,
    path: Path,
    local_roots: set[str],
) -> CodeCheckResult:
    allowed_roots = SAFE_IMPORT_ROOTS | local_roots
    for node in ast.walk(tree):
        imported_roots: list[str] = []
        if isinstance(node, ast.Import):
            imported_roots = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported_roots = [(node.module or "").split(".", 1)[0]]
        # Permit only the pure CPU compiler decorator, not arbitrary Numba APIs.
        # The existing candidate AST and process restrictions still apply.
        njit_import = (isinstance(node, ast.ImportFrom) and node.level == 0
                       and node.module == "numba" and all(alias.name == "njit" for alias in node.names))
        for imported in imported_roots:
            if imported == "numba" and njit_import:
                continue
            if imported in PROTECTED_IMPORT_ROOTS or imported not in allowed_roots:
                return CodeCheckResult(
                    False,
                    "unsafe_import",
                    f"{path.name} imports disallowed module {imported!r}",
                )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in UNSAFE_CALL_NAMES:
                return CodeCheckResult(
                    False,
                    "unsafe_call",
                    f"{path.name} calls disallowed builtin {node.func.id!r}",
                )
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in UNSAFE_ATTRIBUTE_NAMES:
                return CodeCheckResult(
                    False,
                    "unsafe_attribute",
                    f"{path.name} accesses disallowed attribute {node.attr!r}",
                )
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            return CodeCheckResult(
                False,
                "unsafe_name",
                f"{path.name} accesses __builtins__",
            )
    return CodeCheckResult(True)
