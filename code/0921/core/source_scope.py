from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from agent_ahd.core.research_records import (
    InheritanceChannel,
    ScopeBoundaryEvidence,
    ScopeSpec,
)


@dataclass(frozen=True)
class SymbolBlock:
    qualified_name: str
    relative_path: str
    start_line: int
    end_line: int
    source: str
    node: ast.AST


@dataclass(frozen=True)
class SourceExposure:
    channel: InheritanceChannel
    source_bundle: dict[str, str]
    editable_symbols: tuple[str, ...]
    dependency_symbols: tuple[str, ...]
    complete_tree_sha256: str
    exposed_sha256: str

    def to_record(self) -> dict:
        return {
            "channel": self.channel.value,
            "editable_symbols": list(self.editable_symbols),
            "dependency_symbols": list(self.dependency_symbols),
            "source_bundle_keys": sorted(self.source_bundle),
            "complete_tree_sha256": self.complete_tree_sha256,
            "exposed_sha256": self.exposed_sha256,
            "exposes_complete_tree": (
                self.channel == InheritanceChannel.FULL_SOURCE
            ),
        }


def build_source_exposure(
    root: str | Path,
    *,
    scope: ScopeSpec,
    channel: InheritanceChannel,
) -> SourceExposure:
    source_root = Path(root).resolve()
    full_bundle = _full_source_bundle(source_root)
    complete_hash = _bundle_hash(full_bundle)
    if channel == InheritanceChannel.LOCAL_SOURCE:
        if scope.kind != "symbol_set":
            raise ValueError("local_source requires an explicit symbol_set scope")
        bundle, dependencies = _local_source_bundle(source_root, scope.symbols)
    else:
        bundle = full_bundle
        dependencies = ()
    return SourceExposure(
        channel=channel,
        source_bundle=bundle,
        editable_symbols=scope.symbols,
        dependency_symbols=dependencies,
        complete_tree_sha256=complete_hash,
        exposed_sha256=_bundle_hash(bundle),
    )


def locate_symbol(root: str | Path, qualified_name: str) -> SymbolBlock:
    blocks = _symbol_blocks(Path(root).resolve())
    try:
        return blocks[qualified_name]
    except KeyError as exc:
        raise ValueError(f"unknown source symbol: {qualified_name}") from exc


def source_symbol_names(root: str | Path) -> tuple[str, ...]:
    """Return the exact symbol namespace accepted by local-source exposure."""
    return tuple(sorted(_symbol_blocks(Path(root).resolve())))


def derive_scope_boundary_evidence(
    root: str | Path,
    *,
    candidate_id: str,
    source_scope: ScopeSpec,
    supporting_evaluation_ids: tuple[str, ...],
) -> ScopeBoundaryEvidence | None:
    if source_scope.kind != "symbol_set":
        return None
    exposure = build_source_exposure(
        root,
        scope=source_scope,
        channel=InheritanceChannel.LOCAL_SOURCE,
    )
    if not exposure.dependency_symbols:
        return None
    approved_scope = ScopeSpec(
        "symbol_set",
        tuple(sorted({*source_scope.symbols, *exposure.dependency_symbols})),
    )
    identity = {
        "candidate_id": candidate_id,
        "source_scope": source_scope.symbols,
        "approved_scope": approved_scope.symbols,
        "supporting_evaluation_ids": supporting_evaluation_ids,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return ScopeBoundaryEvidence(
        evidence_id=f"scope_boundary:{candidate_id}:{digest}",
        candidate_id=candidate_id,
        source_scope=source_scope,
        approved_scope=approved_scope,
        reason_kind="symbol_dependency",
        supporting_evaluation_ids=supporting_evaluation_ids,
        producer="ast_dependency",
    )


def _full_source_bundle(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _local_source_bundle(
    root: Path,
    requested_symbols: tuple[str, ...],
) -> tuple[dict[str, str], tuple[str, ...]]:
    blocks = _symbol_blocks(root)
    unknown = set(requested_symbols) - set(blocks)
    if unknown:
        raise ValueError(f"unknown local source symbols: {sorted(unknown)}")
    selected = set(requested_symbols)
    pending = list(requested_symbols)
    while pending:
        current = blocks[pending.pop()]
        for dependency in _dependencies(current, blocks):
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)

    bundle: dict[str, str] = {}
    paths = sorted({blocks[name].relative_path for name in selected})
    for relative in paths:
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            ast.get_source_segment(source, node) or ""
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        if imports:
            bundle[f"{relative}::imports"] = "\n".join(imports)
    for name in sorted(selected):
        block = blocks[name]
        bundle[f"{block.relative_path}::{name}"] = block.source
    dependencies = tuple(sorted(selected - set(requested_symbols)))
    return bundle, dependencies


def _symbol_blocks(root: Path) -> dict[str, SymbolBlock]:
    result: dict[str, SymbolBlock] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        relative = path.relative_to(root).as_posix()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _add_block(result, node.name, relative, source, node)
            elif isinstance(node, ast.ClassDef):
                _add_block(result, node.name, relative, source, node)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _add_block(
                            result,
                            f"{node.name}.{child.name}",
                            relative,
                            source,
                            child,
                        )
    return result


def _add_block(
    result: dict[str, SymbolBlock],
    name: str,
    relative: str,
    source: str,
    node: ast.AST,
) -> None:
    if name in result:
        raise ValueError(f"ambiguous source symbol: {name}")
    lines = source.splitlines(keepends=True)
    start = int(getattr(node, "lineno"))
    end = int(getattr(node, "end_lineno"))
    result[name] = SymbolBlock(
        qualified_name=name,
        relative_path=relative,
        start_line=start,
        end_line=end,
        source="".join(lines[start - 1 : end]),
        node=node,
    )


def _dependencies(
    block: SymbolBlock,
    blocks: dict[str, SymbolBlock],
) -> tuple[str, ...]:
    owner = block.qualified_name.rsplit(".", 1)[0] if "." in block.qualified_name else None
    names = set()
    for node in ast.walk(block.node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
                and owner is not None
            ):
                names.add(f"{owner}.{node.func.attr}")
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return tuple(sorted(name for name in names if name in blocks))


def _bundle_hash(bundle: dict[str, str]) -> str:
    encoded = json.dumps(
        bundle,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
