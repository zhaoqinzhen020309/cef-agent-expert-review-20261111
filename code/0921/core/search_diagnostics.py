from __future__ import annotations

import ast
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_ahd.core.archive import ArchiveSnapshot
from agent_ahd.core.code_checks import symbol_fingerprints
from agent_ahd.core.research_records import Observation, SearchAction
from agent_ahd.core.search_state import SearchState
from agent_ahd.core.source_scope import source_symbol_names


@dataclass(frozen=True)
class SourceChangeProfile:
    candidate_id: str
    kind: str
    changed_symbols: tuple[str, ...]
    added_symbols: tuple[str, ...]
    removed_symbols: tuple[str, ...]
    call_edges_added: tuple[str, ...]
    call_edges_removed: tuple[str, ...]

    @property
    def structural(self) -> bool:
        return self.kind.startswith("structural_")


@dataclass(frozen=True)
class SearchDiagnostics:
    parent_dominance_rate: float
    dominant_parent_id: str | None
    effective_root_count: int
    behavior_cluster_count: int
    invalid_candidate_rate: float
    structural_modification_ratio: float
    parameter_tuning_ratio: float
    rounds_without_global_improvement: int
    source_changes: dict[str, SourceChangeProfile]
    inheritance_channel_counts: dict[str, int]
    action_statistics: dict[str, dict[str, float | int]]
    parent_ast_distance: dict[str, float]
    parent_behavior_distance: dict[str, float]
    modified_symbol_count: dict[str, int]
    visible_candidate_ids: tuple[str, ...]
    retained_candidate_ids: tuple[str, ...]
    entered_candidate_ids: tuple[str, ...]
    evicted_candidate_ids: tuple[str, ...]
    visible_root_counts: dict[str, int]
    scope_transition_counts: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)

    def observations(self) -> dict[str, Observation]:
        values = {
            "parent_dominance_rate": self.parent_dominance_rate,
            "dominant_parent_id": self.dominant_parent_id,
            "effective_root_count": self.effective_root_count,
            "behavior_cluster_count": self.behavior_cluster_count,
            "invalid_candidate_rate": self.invalid_candidate_rate,
            "structural_modification_ratio": self.structural_modification_ratio,
            "parameter_tuning_ratio": self.parameter_tuning_ratio,
            "rounds_without_global_improvement": self.rounds_without_global_improvement,
            "inheritance_channel_counts": self.inheritance_channel_counts,
            "action_statistics": self.action_statistics,
            "mean_parent_ast_distance": _mean_or_zero(
                self.parent_ast_distance.values()
            ),
            "mean_parent_behavior_distance": _mean_or_zero(
                self.parent_behavior_distance.values()
            ),
            "mean_modified_symbol_count": _mean_or_zero(
                self.modified_symbol_count.values()
            ),
            "retained_candidate_ids": self.retained_candidate_ids,
            "entered_candidate_ids": self.entered_candidate_ids,
            "evicted_candidate_ids": self.evicted_candidate_ids,
            "visible_root_counts": self.visible_root_counts,
            "scope_transition_counts": self.scope_transition_counts,
        }
        observations = {
            "search:population_state": Observation(
                observation_id="search:population_state",
                kind="search_state",
                statement=(
                    f"Parent dominance is {self.parent_dominance_rate:.3f}; "
                    f"{self.effective_root_count} root branches and "
                    f"{self.behavior_cluster_count} behavior clusters remain visible."
                ),
                values=values,
            ),
            "search:change_mix": Observation(
                observation_id="search:change_mix",
                kind="search_state",
                statement=(
                    f"Structural changes account for "
                    f"{self.structural_modification_ratio:.3f} of classified candidates; "
                    f"parameter-only changes account for {self.parameter_tuning_ratio:.3f}."
                ),
                values=values,
            ),
            "search:validity": Observation(
                observation_id="search:validity",
                kind="search_state",
                statement=(
                    f"Invalid candidates account for {self.invalid_candidate_rate:.3f}; "
                    f"the global best has not improved for "
                    f"{self.rounds_without_global_improvement} completed rounds."
                ),
                values=values,
            ),
        }
        return observations


def build_search_diagnostics(
    state: SearchState,
    snapshot: ArchiveSnapshot,
    *,
    run_dir: str | Path,
    baseline_path: str | Path,
    previous_visible_candidate_ids: set[str] | None = None,
) -> SearchDiagnostics:
    run_root = Path(run_dir).resolve()
    baseline = Path(baseline_path).resolve()
    evaluated_nodes = [
        node
        for node in state.nodes
        if node.status != "pending" and node.action != SearchAction.ABLATE.value
        and node.round_index >= 0
    ]
    inherited = [node for node in evaluated_nodes if node.primary_parent_id is not None]
    parent_counts = Counter(node.primary_parent_id for node in inherited)
    dominant_parent_id = None
    parent_dominance = 0.0
    if parent_counts:
        dominant_parent_id, count = min(
            parent_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        parent_dominance = count / len(inherited)

    visible_roots = {
        state.require_node(candidate_id).root_id
        for candidate_id in snapshot.visible_candidate_ids
        if _has_node(state, candidate_id)
    }
    source_changes = {}
    for node in evaluated_nodes:
        if node.valid is not True:
            continue
        candidate_path = run_root / node.solver_relpath
        if not candidate_path.is_file():
            continue
        before_path = baseline
        if node.primary_parent_id is not None:
            parent = state.require_node(node.primary_parent_id)
            before_path = run_root / parent.solver_relpath
        if before_path.is_file():
            source_changes[node.node_id] = classify_source_change(
                node.node_id,
                before_path,
                candidate_path,
            )

    classified = list(source_changes.values())
    structural_ratio = (
        sum(item.structural for item in classified) / len(classified)
        if classified
        else 0.0
    )
    parameter_ratio = (
        sum(item.kind == "parameter_tuning" for item in classified) / len(classified)
        if classified
        else 0.0
    )
    invalid_rate = (
        sum(node.valid is False for node in evaluated_nodes) / len(evaluated_nodes)
        if evaluated_nodes
        else 0.0
    )
    decisions = {decision.decision_id: decision for decision in state.decisions}
    inheritance_counts = Counter()
    action_totals = Counter()
    action_valid = Counter()
    action_improved = Counter()
    scope_transitions = Counter()
    parent_ast_distance = {}
    parent_behavior_distance = {}
    modified_symbol_count = {}
    for node in evaluated_nodes:
        decision = decisions.get(node.decision_id)
        if decision is not None:
            inheritance_counts[decision.inheritance_channel.value] += 1
        if node.action is not None:
            action_totals[node.action] += 1
        if node.valid is True and node.action is not None:
            action_valid[node.action] += 1
            if node.primary_parent_id is not None:
                parent = state.require_node(node.primary_parent_id)
                if (
                    node.score is not None
                    and parent.score is not None
                    and node.score < parent.score
                ):
                    action_improved[node.action] += 1
        if decision is not None:
            parent_scope = "baseline"
            if node.primary_parent_id is not None:
                raw_parent_scope = state.require_node(node.primary_parent_id).scope
                parent_scope = (
                    str(raw_parent_scope.get("kind", "unknown"))
                    if raw_parent_scope
                    else "unknown"
                )
            scope_transitions[
                f"{parent_scope}->{decision.scope.kind}"
            ] += 1
        change = source_changes.get(node.node_id)
        if change is not None:
            modified_symbol_count[node.node_id] = (
                len(change.changed_symbols)
                + len(change.added_symbols)
                + len(change.removed_symbols)
            )
        if node.primary_parent_id is None or node.valid is not True:
            continue
        child_path = run_root / node.solver_relpath
        parent = state.require_node(node.primary_parent_id)
        parent_path = run_root / parent.solver_relpath
        if child_path.is_file() and parent_path.is_file():
            parent_ast_distance[node.node_id] = normalized_ast_distance(
                parent_path,
                child_path,
            )
        child_profile = snapshot.profiles.get(node.node_id)
        parent_profile = snapshot.profiles.get(node.primary_parent_id)
        if child_profile is not None and parent_profile is not None:
            parent_behavior_distance[node.node_id] = normalized_behavior_distance(
                parent_profile.per_instance_mean,
                child_profile.per_instance_mean,
            )
    current_visible = snapshot.visible_candidate_ids
    previous_visible = previous_visible_candidate_ids or set()
    visible_root_counts = Counter(
        state.require_node(candidate_id).root_id
        for candidate_id in current_visible
        if _has_node(state, candidate_id)
    )
    return SearchDiagnostics(
        parent_dominance_rate=parent_dominance,
        dominant_parent_id=dominant_parent_id,
        effective_root_count=len(visible_roots),
        behavior_cluster_count=len(snapshot.candidate_clusters),
        invalid_candidate_rate=invalid_rate,
        structural_modification_ratio=structural_ratio,
        parameter_tuning_ratio=parameter_ratio,
        rounds_without_global_improvement=state.rounds_without_global_improvement,
        source_changes=source_changes,
        inheritance_channel_counts=dict(sorted(inheritance_counts.items())),
        action_statistics={
            action: {
                "count": action_totals[action],
                "valid_count": action_valid[action],
                "valid_rate": action_valid[action] / action_totals[action],
                "improvement_count": action_improved[action],
                "improvement_rate": action_improved[action] / action_totals[action],
            }
            for action in sorted(action_totals)
        },
        parent_ast_distance=parent_ast_distance,
        parent_behavior_distance=parent_behavior_distance,
        modified_symbol_count=modified_symbol_count,
        visible_candidate_ids=tuple(sorted(current_visible)),
        retained_candidate_ids=tuple(sorted(current_visible & previous_visible)),
        entered_candidate_ids=tuple(sorted(current_visible - previous_visible)),
        evicted_candidate_ids=tuple(sorted(previous_visible - current_visible)),
        visible_root_counts=dict(sorted(visible_root_counts.items())),
        scope_transition_counts=dict(sorted(scope_transitions.items())),
    )


def normalized_ast_distance(first_path: str | Path, second_path: str | Path) -> float:
    first = _source_symbols(_source_trees(Path(first_path)))
    second = _source_symbols(_source_trees(Path(second_path)))
    symbols = set(first) | set(second)
    if not symbols:
        return 0.0
    return sum(first.get(name) != second.get(name) for name in symbols) / len(symbols)


def normalized_behavior_distance(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("behavior vectors must be non-empty and aligned")
    scale = max(max(abs(value) for value in first + second), 1e-12)
    return sum(abs(left - right) for left, right in zip(first, second)) / (
        len(first) * scale
    )


def _mean_or_zero(values) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0


def classify_source_change(
    candidate_id: str,
    before_path: str | Path,
    after_path: str | Path,
) -> SourceChangeProfile:
    before_trees = _source_trees(Path(before_path))
    after_trees = _source_trees(Path(after_path))
    before_symbols = _source_symbols(before_trees)
    after_symbols = _source_symbols(after_trees)
    changed = tuple(sorted(
        name
        for name in set(before_symbols) & set(after_symbols)
        if before_symbols[name] != after_symbols[name]
    ))
    added = tuple(sorted(set(after_symbols) - set(before_symbols)))
    removed = tuple(sorted(set(before_symbols) - set(after_symbols)))
    before_edges = _source_call_edges(before_trees)
    after_edges = _source_call_edges(after_trees)
    edges_added = tuple(sorted(after_edges - before_edges))
    edges_removed = tuple(sorted(before_edges - after_edges))

    before_dump = _source_dump(before_trees)
    after_dump = _source_dump(after_trees)
    if before_dump == after_dump:
        kind = "unchanged"
    elif _source_without_constants(before_trees) == _source_without_constants(
        after_trees
    ):
        kind = "parameter_tuning"
    elif (
        set(before_trees) != set(after_trees)
        or added
        or removed
        or edges_added
        or edges_removed
    ):
        kind = "structural_pipeline"
    else:
        kind = "structural_logic"
    return SourceChangeProfile(
        candidate_id=candidate_id,
        kind=kind,
        changed_symbols=changed,
        added_symbols=added,
        removed_symbols=removed,
        call_edges_added=edges_added,
        call_edges_removed=edges_removed,
    )


def extract_source_structure(source_path: str | Path) -> dict[str, list[str]]:
    entry = Path(source_path).resolve()
    trees = _source_trees(entry)
    return {
        "files": sorted(trees),
        "symbols": list(source_symbol_names(entry.parent)),
        "diagnostic_symbols": sorted(_source_symbols(trees)),
        "call_edges": sorted(_source_call_edges(trees)),
    }


def source_structure_fingerprint(source_path: str | Path) -> str:
    trees = _source_trees(Path(source_path))
    payload = repr(_source_without_constants(trees)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_trees(entry_path: Path) -> dict[str, ast.Module]:
    entry = entry_path.resolve()
    if entry.name != "solver.py":
        return {"solver.py": ast.parse(entry.read_text(encoding="utf-8"))}
    return {
        path.relative_to(entry.parent).as_posix(): ast.parse(
            path.read_text(encoding="utf-8")
        )
        for path in sorted(entry.parent.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _source_symbols(trees: dict[str, ast.Module]) -> dict[str, str]:
    result = {}
    for relative_path, tree in trees.items():
        for name, fingerprint in symbol_fingerprints(tree).items():
            qualified = name if relative_path == "solver.py" else f"{relative_path}:{name}"
            result[qualified] = fingerprint
    return result


def _source_call_edges(trees: dict[str, ast.Module]) -> set[str]:
    result = set()
    for relative_path, tree in trees.items():
        prefix = "" if relative_path == "solver.py" else f"{relative_path}:"
        result.update(f"{prefix}{edge}" for edge in _call_edges(tree))
    return result


def _source_dump(trees: dict[str, ast.Module]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (relative_path, ast.dump(tree, include_attributes=False))
        for relative_path, tree in sorted(trees.items())
    )


def _source_without_constants(
    trees: dict[str, ast.Module],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (relative_path, _without_constants(tree))
        for relative_path, tree in sorted(trees.items())
    )


def _call_edges(tree: ast.Module) -> set[str]:
    edges = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        caller = node.name
        for descendant in ast.walk(node):
            if not isinstance(descendant, ast.Call):
                continue
            if isinstance(descendant.func, ast.Name):
                callee = descendant.func.id
            elif isinstance(descendant.func, ast.Attribute):
                callee = descendant.func.attr
            else:
                continue
            edges.add(f"{caller}->{callee}")
    return edges


def _without_constants(tree: ast.Module) -> str:
    normalized = _ConstantNormalizer().visit(ast.parse(ast.unparse(tree)))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


class _ConstantNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        if node.value is None or isinstance(node.value, bool):
            return node
        return ast.copy_location(ast.Constant(value=f"<{type(node.value).__name__}>"), node)


def _has_node(state: SearchState, candidate_id: str) -> bool:
    try:
        state.require_node(candidate_id)
    except KeyError:
        return False
    return True
