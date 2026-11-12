from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, pstdev

from agent_ahd.core.archive import build_archive_snapshot
from agent_ahd.core.engine import (
    _archive_exclusions,
    _candidate_file_hashes,
    _canonical_instance_ids,
    _structural_signatures,
    load_planner_engine_config,
    verify_planner_run_integrity,
)
from agent_ahd.core.search_diagnostics import classify_source_change
from agent_ahd.core.staged_evaluation import NONINFERIORITY_TOLERANCE_PP
from agent_ahd.core.protocol_versions import PROTOCOL_VERSIONS
from agent_ahd.problems.base import ProblemAdapter


FINAL_SELECTION_RULE = (
    "complete_validation_confirmation_seeds_then_"
    "sequential_paired_bootstrap_mean_cvar_worst_seed_std_then_"
    "complexity_runtime"
)
FINAL_SELECTION_RULE_VERSION = PROTOCOL_VERSIONS["selection"]
FINAL_CONFIRMATION_FILENAME = "final_confirmation.json"
FROZEN_DIRECTORY = "frozen"
STABLE_CONTROL_ID = "__stable_control__"


def required_confirmation_evaluation_units(adapter, config) -> int:
    if config.experiment_phase != "formal":
        return 0
    evaluated_sources = config.confirmation_finalist_count + 1
    return (
        evaluated_sources
        * len(config.confirmation_seeds)
        * len(adapter.validation_manifest.instance_ids)
    )


def build_finalist_ids(
    run_dir: str | Path,
    *,
    state,
    adapter: ProblemAdapter,
    config,
) -> tuple[str, ...]:
    root = Path(run_dir).resolve()
    snapshot = _training_archive_snapshot(
        root,
        state=state,
        total_rounds=config.total_rounds,
        base_seeds=config.base_seeds,
        tail_fraction=config.tail_fraction,
        instance_ids=_canonical_instance_ids(
            state, adapter, config, split="train"
        ),
        instance_families=adapter.train_manifest.family_by_instance,
        view_size=max(config.confirmation_finalist_count, 1),
    )
    profiles = list(snapshot.profiles.values())
    if not profiles:
        raise RuntimeError("final confirmation requires valid training candidates")

    by_mean = sorted(profiles, key=lambda item: (item.mean_gap, item.candidate_id))
    by_cvar = sorted(
        profiles,
        key=lambda item: (item.cvar, item.worst_gap, item.mean_gap, item.candidate_id),
    )
    by_worst = sorted(
        profiles,
        key=lambda item: (item.worst_gap, item.mean_gap, item.candidate_id),
    )
    selected: list[str] = []
    stable_slots = int(config.include_stable_control_in_shortlist)
    search_slots = config.confirmation_finalist_count - stable_slots
    if search_slots <= 0:
        raise ValueError("final confirmation requires at least one searched candidate")

    def add(candidate_id: str) -> None:
        if candidate_id not in selected and len(selected) < search_slots:
            selected.append(candidate_id)

    add(by_mean[0].candidate_id)
    add(by_cvar[0].candidate_id)
    add(by_worst[0].candidate_id)
    structurally_distinct = _best_structurally_distinct(
        root,
        state,
        by_mean,
    )
    if structurally_distinct is not None:
        add(structurally_distinct)
    for profile in by_mean:
        add(profile.candidate_id)
    if len(selected) != search_slots:
        raise RuntimeError(
            f"final confirmation requires {search_slots} valid searched candidates"
        )
    if config.include_stable_control_in_shortlist:
        return (*selected, STABLE_CONTROL_ID)
    return tuple(selected)


def select_training_roles(
    run_dir: str | Path,
    *,
    state,
    total_rounds: int,
    base_seeds: tuple[int, ...],
    tail_fraction: float,
    instance_ids: tuple[str, ...],
    instance_families: dict[str, str],
) -> dict:
    snapshot = _training_archive_snapshot(
        Path(run_dir).resolve(),
        state=state,
        total_rounds=total_rounds,
        base_seeds=base_seeds,
        tail_fraction=tail_fraction,
        instance_ids=instance_ids,
        instance_families=instance_families,
        view_size=2,
    )
    profiles = list(snapshot.profiles.values())
    if not profiles:
        raise RuntimeError("training-role selection requires valid candidates")
    mean = min(profiles, key=lambda item: (item.mean_gap, item.candidate_id))
    tail = min(
        profiles,
        key=lambda item: (
            item.cvar,
            item.worst_gap,
            item.mean_gap,
            item.candidate_id,
        ),
    )
    candidate_ids = list(dict.fromkeys((mean.candidate_id, tail.candidate_id)))
    return {
        "roles": {"mean": mean.candidate_id, "tail": tail.candidate_id},
        "candidate_ids": candidate_ids,
        "profiles": {
            candidate_id: asdict(snapshot.profiles[candidate_id])
            for candidate_id in candidate_ids
        },
    }


def _training_archive_snapshot(
    root: Path,
    *,
    state,
    total_rounds: int,
    base_seeds: tuple[int, ...],
    tail_fraction: float,
    instance_ids: tuple[str, ...],
    instance_families: dict[str, str],
    view_size: int,
):
    _require_completed_search(state, total_rounds)
    primary_records = [
        record
        for record in state.evaluations
        if record.split == "train" and record.seeds == base_seeds
    ]
    return build_archive_snapshot(
        primary_records,
        split="train",
        view_size=view_size,
        tail_fraction=tail_fraction,
        excluded_candidate_ids=_archive_exclusions(state),
        canonical_instance_ids=instance_ids,
        instance_families=instance_families,
        structural_signatures=_structural_signatures(state, root),
    )


def select_final_candidate(
    run_dir: str | Path,
    *,
    adapter: ProblemAdapter,
) -> dict:
    root = Path(run_dir).resolve()
    state = verify_planner_run_integrity(root, adapter)
    config = load_planner_engine_config(root)
    _require_completed_search(state, config.total_rounds)
    confirmation = _load_confirmation_manifest(root)
    requested_finalist_ids = set(confirmation["finalist_ids"])
    finalist_ids = set(
        confirmation.get("eligible_finalist_ids", requested_finalist_ids)
    )
    if not finalist_ids or not finalist_ids <= requested_finalist_ids:
        raise RuntimeError("final confirmation has invalid eligible finalists")
    confirmation_records = [
        record
        for record in state.evaluations
        if record.evaluation_id.startswith("final_confirm_")
    ]
    source_complexity = {
        candidate_id: _source_complexity(root, state, candidate_id)
        for candidate_id in finalist_ids
    }
    selection = select_validation_records(
        confirmation_records,
        candidate_ids=finalist_ids,
        required_seeds=set(config.confirmation_seeds),
        required_instance_ids=_canonical_instance_ids(
            state,
            adapter,
            config,
            split="validation",
        ),
        tail_fraction=config.tail_fraction,
        source_complexity=source_complexity,
    )
    return {
        "rule": FINAL_SELECTION_RULE,
        "required_split": "validation",
        "required_instance_ids": list(selection.pop("required_instance_ids")),
        "required_seeds": list(config.confirmation_seeds),
        **selection,
    }


def select_validation_records(
    records,
    *,
    candidate_ids: set[str],
    required_seeds: set[int],
    required_instance_ids: tuple[str, ...],
    tail_fraction: float,
    source_complexity: dict[str, int],
) -> dict:
    if not candidate_ids:
        raise RuntimeError("validation selection requires eligible candidates")
    if STABLE_CONTROL_ID not in candidate_ids:
        raise RuntimeError("final selection requires the stable control")
    record_list = list(records)
    snapshot = build_archive_snapshot(
        record_list,
        split="validation",
        canonical_instance_ids=required_instance_ids,
        tail_fraction=tail_fraction,
    )
    eligible = [
        profile
        for profile in snapshot.profiles.values()
        if profile.candidate_id in candidate_ids
        and set(profile.seeds) == required_seeds
    ]
    if {item.candidate_id for item in eligible} != candidate_ids:
        raise RuntimeError(
            "finalists do not have complete validation on all confirmation seeds"
        )
    records_by_id = {
        record.candidate_id: record
        for record in record_list
        if record.candidate_id in candidate_ids
    }
    if set(records_by_id) != candidate_ids:
        raise RuntimeError("validation selection records are incomplete")
    stable_profile = next(
        (
            profile
            for profile in eligible
            if profile.candidate_id == STABLE_CONTROL_ID
        ),
        None,
    )
    if stable_profile is None:
        raise RuntimeError("stable control does not have complete validation")
    control_threshold = stable_profile.mean_gap + NONINFERIORITY_TOLERANCE_PP
    control_noninferior = [
        profile
        for profile in eligible
        if profile.mean_gap <= control_threshold
    ]
    best_mean = min(profile.mean_gap for profile in control_noninferior)
    noninferior = [
        profile
        for profile in control_noninferior
        if profile.mean_gap <= best_mean + NONINFERIORITY_TOLERANCE_PP
    ]
    metric_values = {
        "mean_gap": lambda profile: profile.mean_gap,
        "cvar": lambda profile: profile.cvar,
        "worst_gap": lambda profile: profile.worst_gap,
        "seed_std_gap": lambda profile: profile.seed_std_gap,
    }
    tied = list(noninferior)
    metric_stages = {}
    mean_tied_ids: list[str] = []
    for metric_name, value in metric_values.items():
        winner = min(
            tied,
            key=lambda profile: (value(profile), profile.candidate_id),
        )
        confidence_intervals = {
            profile.candidate_id: _paired_metric_confidence_interval(
                records_by_id[profile.candidate_id],
                records_by_id[winner.candidate_id],
                metric_name=metric_name,
                tail_fraction=tail_fraction,
            )
            for profile in tied
        }
        tied = [
            profile
            for profile in tied
            if confidence_intervals[profile.candidate_id][0] <= 0.0
            <= confidence_intervals[profile.candidate_id][1]
        ]
        metric_stages[metric_name] = {
            "winner_id": winner.candidate_id,
            "tied_candidate_ids": sorted(
                profile.candidate_id for profile in tied
            ),
            "paired_difference_ci95": {
                candidate_id: list(interval)
                for candidate_id, interval in sorted(confidence_intervals.items())
            },
        }
        if metric_name == "mean_gap":
            mean_tied_ids = sorted(profile.candidate_id for profile in tied)
        if len(tied) == 1:
            break
    selected = min(
        tied,
        key=lambda profile: (
            source_complexity[profile.candidate_id],
            profile.total_runtime_sec,
            profile.candidate_id,
        ),
    )
    return {
        "selection_rule_version": FINAL_SELECTION_RULE_VERSION,
        "candidate_id": selected.candidate_id,
        "required_instance_ids": tuple(selected.instance_ids),
        "profile": asdict(selected),
        "eligible_candidate_ids": sorted(item.candidate_id for item in eligible),
        "mean_noninferior_candidate_ids": sorted(
            item.candidate_id for item in noninferior
        ),
        "stable_control_noninferior_candidate_ids": sorted(
            item.candidate_id for item in control_noninferior
        ),
        "mean_noninferiority_tolerance_pp": NONINFERIORITY_TOLERANCE_PP,
        "mean_winner_id": metric_stages["mean_gap"]["winner_id"],
        "statistically_tied_candidate_ids": mean_tied_ids,
        "metric_selection_stages": metric_stages,
        "source_complexity": dict(source_complexity),
    }


def freeze_selected_candidate(
    run_dir: str | Path,
    *,
    state,
    selection: dict,
) -> Path:
    root = Path(run_dir).resolve()
    candidate_id = selection["candidate_id"]
    if candidate_id == STABLE_CONTROL_ID:
        source_root = root / "controls" / "stable"
        solver_name = _stable_solver_path(root).name
    else:
        node = state.require_node(candidate_id)
        source_root = (root / node.solver_relpath).resolve().parent
        solver_name = Path(node.solver_relpath).name
    destination = root / FROZEN_DIRECTORY / candidate_id
    if destination.exists():
        raise FileExistsError(f"frozen candidate already exists: {destination}")
    for source in sorted(source_root.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        target = destination / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    solver_path = destination / solver_name
    if not solver_path.is_file():
        raise RuntimeError("frozen candidate has no solver entry file")
    protocol = json.loads(
        (root / "planner_config.json").read_text(encoding="utf-8")
    )
    selection_contract = {
        "rule": selection["rule"],
        "required_seeds": selection["required_seeds"],
        "required_instance_ids": selection["required_instance_ids"],
    }
    manifest = {
        "candidate_id": candidate_id,
        "selection_rule": selection["rule"],
        "selection_contract": selection_contract,
        "selection_rule_sha256": _json_sha256(selection_contract),
        "locked_protocol_hashes": protocol["locked_artifacts"],
        "source_files": _candidate_file_hashes(destination),
        "solver_relpath": solver_path.relative_to(root).as_posix(),
    }
    _write_json(destination / "manifest.json", manifest)
    return solver_path


def load_frozen_solver(run_dir: str | Path, candidate_id: str) -> Path:
    root = Path(run_dir).resolve()
    frozen_root = root / FROZEN_DIRECTORY / candidate_id
    manifest_path = frozen_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("final candidate has not been frozen")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_id") != candidate_id:
        raise RuntimeError("frozen candidate manifest id mismatch")
    actual = _candidate_file_hashes(frozen_root)
    actual.pop("manifest.json", None)
    if actual != manifest.get("source_files"):
        raise RuntimeError("frozen candidate source hash mismatch")
    protocol = json.loads(
        (root / "planner_config.json").read_text(encoding="utf-8")
    )
    if manifest.get("locked_protocol_hashes") != protocol.get("locked_artifacts"):
        raise RuntimeError("frozen candidate protocol hash mismatch")
    if _json_sha256(manifest.get("selection_contract")) != manifest.get(
        "selection_rule_sha256"
    ):
        raise RuntimeError("frozen candidate selection rule hash mismatch")
    solver_path = root / manifest["solver_relpath"]
    if not solver_path.is_file() or not solver_path.is_relative_to(frozen_root):
        raise RuntimeError("frozen solver path is invalid")
    return solver_path


def _require_completed_search(state, total_rounds: int) -> None:
    if state.pending_node_ids:
        raise RuntimeError("final selection requires no pending candidates")
    if (
        state.current_round + 1 != total_rounds
        or state.completed_rounds != list(range(total_rounds))
    ):
        raise RuntimeError(
            "final selection requires all configured rounds to finish"
        )


def _load_confirmation_manifest(root: Path) -> dict:
    path = root / FINAL_CONFIRMATION_FILENAME
    if not path.is_file():
        raise RuntimeError("final confirmation has not been completed")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") not in {"evaluated", "complete"}:
        raise RuntimeError("final confirmation is not complete")
    if not data.get("finalist_ids"):
        raise RuntimeError("final confirmation has no finalists")
    return data


def _best_structurally_distinct(root: Path, state, by_mean) -> str | None:
    champion = state.require_node(by_mean[0].candidate_id)
    champion_path = root / champion.solver_relpath
    for profile in by_mean[1:]:
        node = state.require_node(profile.candidate_id)
        change = classify_source_change(
            profile.candidate_id,
            champion_path,
            root / node.solver_relpath,
        )
        if change.structural:
            return profile.candidate_id
    return None


def _stable_solver_path(root: Path) -> Path:
    protocol = json.loads(
        (root / "planner_config.json").read_text(encoding="utf-8")
    )
    stable = protocol.get("stable_control")
    if not stable:
        raise RuntimeError("stable control was not preregistered")
    path = root / stable["solver_relpath"]
    if not path.is_file() or not path.is_relative_to(root / "controls" / "stable"):
        raise RuntimeError("stable control path is invalid")
    actual = _candidate_file_hashes(path.parent)
    if actual != stable.get("source_files"):
        raise RuntimeError("stable control source hash mismatch")
    return path


def _candidate_source_root(root: Path, state, candidate_id: str) -> Path:
    if candidate_id == STABLE_CONTROL_ID:
        return _stable_solver_path(root).parent
    return (root / state.require_node(candidate_id).solver_relpath).resolve().parent


def _source_complexity(root: Path, state, candidate_id: str) -> int:
    source_root = _candidate_source_root(root, state, candidate_id)
    return source_tree_complexity(source_root)


def source_tree_complexity(source_root: str | Path) -> int:
    root = Path(source_root).resolve()
    statement_count = 0
    for source in root.rglob("*.py"):
        if "__pycache__" in source.parts:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"))
        statement_count += sum(
            isinstance(node, ast.stmt) for node in ast.walk(tree)
        )
    return statement_count


def _paired_metric_confidence_interval(
    first,
    second,
    *,
    metric_name: str,
    tail_fraction: float,
) -> tuple[float, float]:
    if first.seeds != second.seeds or first.instance_ids != second.instance_ids:
        raise RuntimeError("final confirmation records are not paired")
    if len(first.gap_matrix) != len(second.gap_matrix) or any(
        len(first_row) != len(second_row)
        for first_row, second_row in zip(first.gap_matrix, second.gap_matrix)
    ):
        raise RuntimeError("final confirmation gap matrices are not aligned")
    if not first.gap_matrix or not first.gap_matrix[0]:
        raise RuntimeError("final confirmation has no paired gap values")
    generator = random.Random(0)
    seed_count = len(first.gap_matrix)
    instance_count = len(first.gap_matrix[0])
    differences = []
    for _ in range(2000):
        seed_indices = [generator.randrange(seed_count) for _ in range(seed_count)]
        instance_indices = [
            generator.randrange(instance_count) for _ in range(instance_count)
        ]
        first_value = _resampled_metric(
            first.gap_matrix,
            seed_indices,
            instance_indices,
            metric_name=metric_name,
            tail_fraction=tail_fraction,
        )
        second_value = _resampled_metric(
            second.gap_matrix,
            seed_indices,
            instance_indices,
            metric_name=metric_name,
            tail_fraction=tail_fraction,
        )
        differences.append(first_value - second_value)
    differences.sort()
    return differences[49], differences[1949]


def _resampled_metric(
    matrix,
    seed_indices: list[int],
    instance_indices: list[int],
    *,
    metric_name: str,
    tail_fraction: float,
) -> float:
    per_instance = [
        fmean(matrix[seed_index][instance_index] for seed_index in seed_indices)
        for instance_index in instance_indices
    ]
    if metric_name == "mean_gap":
        return float(fmean(per_instance))
    if metric_name == "cvar":
        tail_count = max(1, math.ceil(len(per_instance) * tail_fraction))
        return float(fmean(sorted(per_instance, reverse=True)[:tail_count]))
    if metric_name == "worst_gap":
        return float(max(per_instance))
    if metric_name == "seed_std_gap":
        seed_means = [
            fmean(
                matrix[seed_index][instance_index]
                for instance_index in instance_indices
            )
            for seed_index in seed_indices
        ]
        return float(pstdev(seed_means)) if len(seed_means) > 1 else 0.0
    raise ValueError(f"unsupported final-selection metric: {metric_name}")


def _json_sha256(data: object) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
