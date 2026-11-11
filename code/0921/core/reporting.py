from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from agent_ahd.core.archive import build_archive_snapshot
from agent_ahd.core.attempt_ledger import ATTEMPT_LEDGER_FILENAME, AttemptLedger
from agent_ahd.core.engine import (
    BASELINE_CONTROL_ID,
    _archive_exclusions,
    _budget_usage,
    _canonical_instance_ids,
    _model_call_attempts,
    _model_call_limit,
    _model_token_usage,
    _node_model_call_attempts,
    _structural_signatures,
    _wall_clock_elapsed,
    load_planner_engine_config,
    verify_planner_run_integrity,
)
from agent_ahd.core.search_diagnostics import build_search_diagnostics
from agent_ahd.problems.base import ProblemAdapter


def build_run_report(
    run_dir: str | Path,
    *,
    adapter: ProblemAdapter,
) -> dict:
    root = Path(run_dir).resolve()
    state = verify_planner_run_integrity(root, adapter)
    config = load_planner_engine_config(root)
    structural_signatures = _structural_signatures(state, root)
    train = build_archive_snapshot(
        state.evaluations,
        split="train",
        excluded_candidate_ids=_archive_exclusions(state),
        canonical_instance_ids=_canonical_instance_ids(
            state,
            adapter,
            config,
            split="train",
        ),
        instance_families=adapter.train_manifest.family_by_instance,
        structural_signatures=structural_signatures,
    )
    validation = build_archive_snapshot(
        state.evaluations,
        split="validation",
        excluded_candidate_ids=_archive_exclusions(state),
        canonical_instance_ids=_canonical_instance_ids(
            state,
            adapter,
            config,
            split="validation",
        ),
        instance_families=adapter.validation_manifest.family_by_instance,
        structural_signatures=structural_signatures,
    )
    baseline_train = build_archive_snapshot(
        [
            record
            for record in state.evaluations
            if record.candidate_id == BASELINE_CONTROL_ID
        ],
        split="train",
        canonical_instance_ids=_canonical_instance_ids(
            state,
            adapter,
            config,
            split="train",
        ),
        instance_families=adapter.train_manifest.family_by_instance,
    )
    diagnostics = build_search_diagnostics(
        state,
        train,
        run_dir=root,
        baseline_path=adapter.baseline_path,
    )
    trajectory = _improvement_trajectory(state, train.profiles)
    finalized_evidence = [
        item for item in state.evidence if item.status != "pending"
    ]
    ablated = [
        item
        for item in finalized_evidence
        if item.ablation_id is not None
        and item.status in {"supported", "inconclusive"}
    ]
    ablated_supported = [item for item in ablated if item.status == "supported"]
    migrated = [item for item in finalized_evidence if item.source_action == "CROSS"]
    migrated_supported = [item for item in migrated if item.status == "supported"]
    mechanism_outcomes = [
        item
        for item in finalized_evidence
        if item.source_action != "ABLATE"
    ]
    generalization = {}
    for candidate_id in sorted(set(train.profiles) & set(validation.profiles)):
        generalization[candidate_id] = (
            validation.profiles[candidate_id].mean_gap
            - train.profiles[candidate_id].mean_gap
        )

    budget_usage = _budget_usage(state, adapter, config, run_dir=root)
    from agent_ahd.core.final_selection import (
        required_confirmation_evaluation_units,
    )

    confirmation_limit = required_confirmation_evaluation_units(adapter, config)
    model_calls = _model_call_attempts(root)
    model_tokens = _model_token_usage(root)
    wall_clock_sec = _wall_clock_elapsed(root)
    protocol = json.loads(
        (root / "planner_config.json").read_text(encoding="utf-8")
    )
    return {
        "run_id": state.run_id,
        "policy": protocol["policy"],
        "planner_variant": protocol.get("planner_variant"),
        "dataset_protocol_id": protocol["dataset_protocol"]["protocol_id"],
        "completed_rounds": list(state.completed_rounds),
        "candidate_count": sum(node.round_index >= 0 for node in state.nodes),
        "evaluation_record_count": sum(
            item.candidate_id != BASELINE_CONTROL_ID for item in state.evaluations
        ),
        "control_evaluation_count": sum(
            item.candidate_id == BASELINE_CONTROL_ID for item in state.evaluations
        ),
        "model_call_count": model_calls,
        "model_calls_to_train_best": _model_calls_to_train_best(
            root,
            state,
            train,
        ),
        "budget": {
            "limits": {
                "candidates": config.resolved_candidate_budget,
                "candidate_evaluation_units": (
                    config.resolved_candidate_evaluation_unit_budget
                ),
                "diagnostic_evaluation_units": (
                    config.resolved_diagnostic_evaluation_unit_budget
                ),
                "confirmation_evaluation_units": confirmation_limit,
                "evaluation_units": (
                    config.resolved_candidate_evaluation_unit_budget
                    + config.resolved_diagnostic_evaluation_unit_budget
                    + confirmation_limit
                ),
                "model_calls": _model_call_limit(config),
                "model_tokens": config.model_token_budget,
                "wall_clock_sec": config.wall_clock_budget_sec,
            },
            "used": {
                **budget_usage,
                "model_calls": model_calls,
                "model_tokens": model_tokens,
                "wall_clock_sec": wall_clock_sec,
            },
            "remaining": {
                "candidates": config.resolved_candidate_budget
                - budget_usage["candidates"],
                "candidate_evaluation_units": (
                    config.resolved_candidate_evaluation_unit_budget
                    - budget_usage["candidate_evaluation_units"]
                ),
                "diagnostic_evaluation_units": (
                    config.resolved_diagnostic_evaluation_unit_budget
                    - budget_usage["diagnostic_evaluation_units"]
                ),
                "confirmation_evaluation_units": (
                    confirmation_limit
                    - budget_usage["confirmation_evaluation_units"]
                ),
                "evaluation_units": (
                    config.resolved_candidate_evaluation_unit_budget
                    + config.resolved_diagnostic_evaluation_unit_budget
                    + confirmation_limit
                    - budget_usage["evaluation_units"]
                ),
                "model_calls": _model_call_limit(config) - model_calls,
                "model_tokens": (
                    None
                    if config.model_token_budget is None
                    else config.model_token_budget - model_tokens
                ),
                "wall_clock_sec": (
                    None
                    if config.wall_clock_budget_sec is None
                    else config.wall_clock_budget_sec - wall_clock_sec
                ),
            },
        },
        "quality": {
            "train_best": _best_profile(train),
            "train_cvar10_best": _cvar_best_profile(train),
            "train_worst_best": _worst_best_profile(train),
            "validation_best": _best_profile(validation),
            "validation_cvar10_best": _cvar_best_profile(validation),
            "validation_worst_best": _worst_best_profile(validation),
            "generalization_gap_by_candidate": generalization,
            "train_best_vs_baseline": _best_vs_baseline(train, baseline_train),
            "train_profiles": {
                candidate_id: asdict(profile)
                for candidate_id, profile in sorted(train.profiles.items())
            },
            "validation_profiles": {
                candidate_id: asdict(profile)
                for candidate_id, profile in sorted(validation.profiles.items())
            },
        },
        "search": {
            **{
                key: value
                for key, value in diagnostics.to_dict().items()
                if key != "source_changes"
            },
            "improvement_trajectory": trajectory,
            "improving_round_count": sum(item["improved"] for item in trajectory),
            "max_consecutive_non_improving_rounds": _max_stagnation(trajectory),
            "source_change_kinds": {
                candidate_id: profile.kind
                for candidate_id, profile in diagnostics.source_changes.items()
            },
            "multi_archive_retention_rate": _multi_archive_retention_rate(train),
        },
        "evidence": {
            "status_counts": _status_counts(state.evidence),
            "explanation_ablation_consistency": (
                len(ablated_supported) / len(ablated)
                if ablated
                else None
            ),
            "mechanism_migration_success_rate": (
                len(migrated_supported) / len(migrated) if migrated else None
            ),
            "new_mechanism_success_rate": (
                sum(item.status == "supported" for item in mechanism_outcomes)
                / len(mechanism_outcomes)
                if mechanism_outcomes
                else None
            ),
            "records": [asdict(item) for item in state.evidence],
        },
    }


def write_run_report(
    run_dir: str | Path,
    *,
    adapter: ProblemAdapter,
) -> Path:
    root = Path(run_dir).resolve()
    path = root / "research_report.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(build_run_report(root, adapter=adapter), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _best_profile(snapshot) -> dict | None:
    if not snapshot.mean_elites:
        return None
    candidate_id = snapshot.mean_elites[0]
    return asdict(snapshot.profiles[candidate_id])


def _cvar_best_profile(snapshot) -> dict | None:
    if not snapshot.cvar_elites:
        return None
    candidate_id = snapshot.cvar_elites[0]
    return asdict(snapshot.profiles[candidate_id])


def _worst_best_profile(snapshot) -> dict | None:
    if not snapshot.worst_elites:
        return None
    candidate_id = snapshot.worst_elites[0]
    return asdict(snapshot.profiles[candidate_id])


def _multi_archive_retention_rate(snapshot) -> float | None:
    complementary = (
        set(snapshot.cvar_elites)
        | set(snapshot.worst_elites)
        | set(snapshot.structural_elites)
    )
    if not complementary:
        return None
    non_mean = complementary - set(snapshot.mean_elites)
    return len(non_mean) / len(complementary)


def _best_vs_baseline(snapshot, baseline_snapshot) -> dict | None:
    if not snapshot.mean_elites or BASELINE_CONTROL_ID not in baseline_snapshot.profiles:
        return None
    candidate = snapshot.profiles[snapshot.mean_elites[0]]
    baseline = baseline_snapshot.profiles[BASELINE_CONTROL_ID]
    if candidate.instance_ids != baseline.instance_ids:
        raise ValueError("best candidate and baseline use different training instances")
    deltas = [
        baseline_gap - candidate_gap
        for baseline_gap, candidate_gap in zip(
            baseline.per_instance_mean,
            candidate.per_instance_mean,
        )
    ]
    return {
        "candidate_id": candidate.candidate_id,
        "wins": sum(delta > 1e-12 for delta in deltas),
        "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "losses": sum(delta < -1e-12 for delta in deltas),
        "baseline_minus_candidate_mean_gap": (
            baseline.mean_gap - candidate.mean_gap
        ),
    }


def _improvement_trajectory(state, profiles) -> list[dict]:
    best = None
    trajectory = []
    for round_index in sorted(state.completed_rounds):
        scores = [
            profiles[node.node_id].mean_gap
            for node in state.nodes
            if node.round_index == round_index and node.node_id in profiles
        ]
        round_best = min(scores) if scores else None
        improved = round_best is not None and (best is None or round_best < best - 1e-12)
        if improved:
            best = round_best
        trajectory.append({
            "round_index": round_index,
            "round_best": round_best,
            "global_best": best,
            "improved": improved,
        })
    return trajectory


def _max_stagnation(trajectory: list[dict]) -> int:
    current = 0
    maximum = 0
    for item in trajectory:
        current = 0 if item["improved"] else current + 1
        maximum = max(maximum, current)
    return maximum


def _model_calls_to_train_best(root: Path, state, snapshot) -> int | None:
    if not snapshot.mean_elites:
        return None
    best = state.require_node(snapshot.mean_elites[0])
    if (root / ATTEMPT_LEDGER_FILENAME).is_file():
        node_rounds = {node.node_id: node.round_index for node in state.nodes}
        count = 0
        for record in AttemptLedger(root).records():
            if record["event_type"] != "attempt_terminal":
                continue
            logical_call_id = record["payload"]["logical_call_id"]
            parts = logical_call_id.split(":")
            if len(parts) == 3 and parts[0] == "round" and parts[2] == "controller":
                try:
                    call_round = int(parts[1])
                except ValueError as exc:
                    raise RuntimeError(
                        "ledger controller logical call has an invalid round"
                    ) from exc
            elif len(parts) == 3 and parts[0] in {"candidate", "scalar"}:
                try:
                    call_round = node_rounds[parts[1]]
                except KeyError as exc:
                    raise RuntimeError(
                        "ledger candidate logical call has no search-state node"
                    ) from exc
            else:
                raise RuntimeError(
                    f"ledger has an unknown logical call identity: {logical_call_id}"
                )
            if call_round <= best.round_index:
                count += 1
        return count
    controller_calls = sum(
        (
            root
            / "rounds"
            / f"round_{round_index:03d}"
            / "controller_attempt.json"
        ).is_file()
        for round_index in range(best.round_index + 1)
    )
    code_calls = 0
    for node in state.nodes:
        if node.round_index > best.round_index:
            continue
        workspace = root / node.solver_relpath
        code_calls += _node_model_call_attempts(workspace.parent)
    return int(controller_calls + code_calls)


def _status_counts(evidence) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in evidence:
        result[item.status] = result.get(item.status, 0) + 1
    return result
