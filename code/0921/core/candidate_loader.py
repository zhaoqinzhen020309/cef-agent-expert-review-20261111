from __future__ import annotations

import ast
import importlib.util
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Callable

from agent_ahd.core.code_checks import (
    DEFAULT_ENTRY_CONTRACT,
    check_candidate_source,
)
from agent_ahd.problems.base import CandidateLoadResult, EntryContract


def load_candidate_strategy(
    candidate_id: str,
    candidate_path: str | Path,
    candidate_root: str | Path | None = None,
    entry_contract: EntryContract = DEFAULT_ENTRY_CONTRACT,
    source_transform: Callable[[ast.Module], ast.Module] | None = None,
) -> CandidateLoadResult:
    path = Path(candidate_path).resolve()
    root = Path(candidate_root).resolve() if candidate_root is not None else path.parent
    check = check_candidate_source(
        path,
        candidate_root=root,
        entry_contract=entry_contract,
    )
    if not check.valid:
        return CandidateLoadResult(
            candidate_id=candidate_id,
            valid=False,
            error_type=check.error_type,
            error_message=check.error_message,
        )

    try:
        module = _load_module_from_path(path, root, source_transform)
        strategy_cls = getattr(module, entry_contract.class_name, None)
    except Exception as exc:
        return CandidateLoadResult(
            candidate_id=candidate_id,
            valid=False,
            error_type="import_error",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    if not isinstance(strategy_cls, type):
        return CandidateLoadResult(
            candidate_id=candidate_id,
            valid=False,
            error_type="invalid_entry_class",
            error_message=f"{entry_contract.class_name} is not a class",
        )

    return CandidateLoadResult(
        candidate_id=candidate_id,
        valid=True,
        strategy_cls=strategy_cls,
    )


def _load_module_from_path(
    path: Path,
    candidate_root: Path,
    source_transform: Callable[[ast.Module], ast.Module] | None = None,
) -> ModuleType:
    module_name = f"agent_ahd_candidate_{path.stem}_{uuid.uuid4().hex}"
    if source_transform is not None:
        module = ModuleType(module_name)
        module.__file__ = str(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        tree = source_transform(tree)
        ast.fix_missing_locations(tree)
        with _candidate_import_path(candidate_root):
            sys.modules[module_name] = module
            try:
                exec(compile(tree, str(path), "exec"), module.__dict__)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        return module

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    with _candidate_import_path(candidate_root):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return module


@contextmanager
def _candidate_import_path(candidate_root: Path):
    value = str(candidate_root)
    inserted = value not in sys.path
    if inserted:
        sys.path.insert(0, value)

    try:
        yield
    finally:
        if inserted and value in sys.path:
            sys.path.remove(value)
