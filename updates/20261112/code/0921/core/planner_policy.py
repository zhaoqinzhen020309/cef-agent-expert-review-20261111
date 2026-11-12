from __future__ import annotations

from dataclasses import dataclass

from agent_ahd.core.research_records import (
    EvidenceRecord,
    Observation,
    PlannerDecision,
    RoundPlan,
    ScopeBoundaryEvidence,
    ScopeSpec,
    SearchAction,
)
from agent_ahd.core.search_state import SearchState, SolverNode


_PARENT_ACTIONS = {
    SearchAction.EXPLOIT,
    SearchAction.REPAIR,
    SearchAction.CROSS,
    SearchAction.EXPAND_SCOPE,
    SearchAction.NARROW_SCOPE,
    SearchAction.ABLATE,
    SearchAction.RETEST,
}
_SCOPE_PRESERVING_ACTIONS = {
    SearchAction.EXPLOIT,
    SearchAction.REPAIR,
    SearchAction.CROSS,
    SearchAction.RETEST,
}
_TARGET_CLUSTER_CONTRACTS = {
    action: (
        "required_failure_clusters"
        if action == SearchAction.REPAIR
        else "registered_failure_clusters"
        if action == SearchAction.ABLATE
        else "empty"
    )
    for action in SearchAction
}
_PERFORMANCE_EVIDENCE_KINDS = {
    "candidate_profile",
    "evaluation_diagnostic",
    "failure_cluster",
    "mechanism_evidence",
}
_GENERATION_CONTEXT_KINDS = {
    SearchAction.EXPLOIT: frozenset({
        "candidate_profile",
        "evaluation_diagnostic",
        "mechanism_evidence",
        "source_structure",
    }),
    SearchAction.REPAIR: frozenset({
        "candidate_profile",
        "evaluation_diagnostic",
        "failure_cluster",
        "mechanism_evidence",
        "source_structure",
    }),
    SearchAction.CROSS: frozenset({
        "candidate_profile",
        "mechanism_evidence",
        "source_structure",
    }),
    SearchAction.DIVERSIFY: frozenset({
        "problem",
        "baseline_control",
        "failure_cluster",
        "source_structure",
    }),
    SearchAction.EXPAND_SCOPE: frozenset({
        "candidate_profile",
        "evaluation_diagnostic",
        "mechanism_evidence",
        "source_structure",
    }),
    SearchAction.NARROW_SCOPE: frozenset({
        "candidate_profile",
        "evaluation_diagnostic",
        "mechanism_evidence",
        "source_structure",
    }),
    SearchAction.ABLATE: frozenset({
        "candidate_profile",
        "failure_cluster",
        "mechanism_evidence",
        "source_structure",
    }),
    SearchAction.RETEST: frozenset(),
}
_GENERATION_CONTEXT_LINK_CONTRACTS = {
    SearchAction.EXPLOIT: {
        "candidate_profile": "primary_parent_only",
        "evaluation_diagnostic": "primary_parent_only",
        "mechanism_evidence": "primary_parent_only",
        "source_structure": "primary_parent_only",
    },
    SearchAction.REPAIR: {
        "candidate_profile": "primary_parent_only",
        "evaluation_diagnostic": "primary_parent_only",
        "mechanism_evidence": "primary_parent_only",
        "source_structure": "primary_parent_only",
    },
    SearchAction.CROSS: {
        "candidate_profile": "primary_parent_only",
        "mechanism_evidence": "targeted_secondary_mechanism_only",
        "source_structure": "primary_parent_only",
    },
    SearchAction.DIVERSIFY: {
        "source_structure": "baseline_only",
    },
    SearchAction.EXPAND_SCOPE: {
        "candidate_profile": "primary_parent_only",
        "evaluation_diagnostic": "primary_parent_only",
        "mechanism_evidence": "primary_parent_only",
        "source_structure": "primary_parent_only",
    },
    SearchAction.NARROW_SCOPE: {
        "candidate_profile": "primary_parent_only",
        "evaluation_diagnostic": "primary_parent_only",
        "mechanism_evidence": "primary_parent_only",
        "source_structure": "primary_parent_only",
    },
    SearchAction.ABLATE: {
        "candidate_profile": "primary_parent_only",
        "mechanism_evidence": "targeted_primary_mechanism_only",
        "source_structure": "primary_parent_only",
    },
    SearchAction.RETEST: {},
}


@dataclass(frozen=True)
class MaterializedPlan:
    candidate_nodes: tuple[SolverNode, ...]
    retest_decisions: tuple[PlannerDecision, ...]


def generation_context_contracts() -> dict[str, list[str]]:
    """Return the exact observation kinds each action may expose to code agents."""
    return {
        action.value: sorted(_GENERATION_CONTEXT_KINDS[action])
        for action in SearchAction
    }


def generation_context_link_contracts() -> dict[str, dict[str, str]]:
    """Return which candidate each linked observation must describe."""
    return {
        action.value: dict(_GENERATION_CONTEXT_LINK_CONTRACTS[action])
        for action in SearchAction
    }


def target_cluster_contracts() -> dict[str, str]:
    """Return each action's exact target-cluster requirement."""
    return {
        action.value: _TARGET_CLUSTER_CONTRACTS[action]
        for action in SearchAction
    }


def materialize_round_plan(
    state: SearchState,
    plan: RoundPlan,
) -> MaterializedPlan:
    nodes = []
    retests = []
    for decision in plan.decisions:
        if decision.action == SearchAction.RETEST:
            state.decisions.append(decision)
            retests.append(decision)
            continue
        if decision.action == SearchAction.DIVERSIFY:
            generation_mode = "fresh"
        elif decision.action == SearchAction.CROSS:
            generation_mode = "cross_reference_redesign"
        elif decision.action in {SearchAction.EXPAND_SCOPE, SearchAction.ABLATE}:
            generation_mode = "redesign"
        else:
            generation_mode = "continue"
        node_id = f"node_{state.next_node_index:04d}"
        nodes.append(
            state.allocate_node(
                round_index=plan.round_index,
                generation_mode=generation_mode,
                solver_relpath=f"nodes/{node_id}/solver.py",
                primary_parent_id=decision.primary_parent_id,
                reference_node_id=decision.secondary_parent_id,
                parent_selection="planner",
                reference_reason=(
                    "validated_mechanism" if decision.secondary_parent_id else None
                ),
                decision=decision,
            )
        )
    return MaterializedPlan(tuple(nodes), tuple(retests))


def validate_round_plan(
    plan: RoundPlan,
    *,
    round_index: int,
    batch_size: int,
    visible_candidate_ids: set[str],
    observation_ids: set[str],
    observation_records: dict[str, Observation | dict] | None = None,
    evidence_records: dict[str, EvidenceRecord] | None = None,
    scope_boundary_evidence: dict[str, ScopeBoundaryEvidence] | None = None,
    failure_cluster_ids: set[str],
    pending_hypothesis_ids: set[str],
    existing_decision_ids: set[str] | None = None,
    transfer_candidate_ids: set[str] | None = None,
    transfer_hypotheses: dict[str, str] | None = None,
    ablation_targets: dict[
        str,
        tuple[str, ScopeSpec, tuple[str, ...]],
    ] | None = None,
    candidate_scopes: dict[str, ScopeSpec] | None = None,
    candidate_source_structures: dict[str, dict] | None = None,
    allowed_retest_seeds: set[int] | None = None,
    allowed_retest_instances: dict[str, set[str]] | None = None,
    completed_retests: (
        set[tuple[str, str, tuple[int, ...], tuple[str, ...]]] | None
    ) = None,
    allowed_actions: set[SearchAction] | None = None,
) -> dict:
    transfer_candidate_ids = transfer_candidate_ids or set()
    scope_boundary_evidence = scope_boundary_evidence or {}
    existing_decision_ids = existing_decision_ids or set()
    transfer_hypotheses = transfer_hypotheses or {}
    ablation_targets = ablation_targets or {}
    candidate_scopes = candidate_scopes or {}
    candidate_source_structures = candidate_source_structures or {}
    allowed_retest_seeds = allowed_retest_seeds or set()
    allowed_retest_instances = allowed_retest_instances or {}
    completed_retests = completed_retests or set()
    allowed_actions = (
        set(SearchAction) if allowed_actions is None else allowed_actions
    )
    evidence_records = evidence_records or {}
    if plan.round_index != round_index:
        raise ValueError(
            f"plan round {plan.round_index} does not match expected round {round_index}"
        )
    if plan.is_stop:
        return {
            "round_index": plan.round_index,
            "candidate_decision_count": 0,
            "diagnostic_decision_count": 0,
            "stopped": True,
            "stop_reason": plan.stop_reason,
            "decisions": [],
        }
    candidate_decisions = [
        decision
        for decision in plan.decisions
        if decision.action != SearchAction.RETEST
    ]
    if len(candidate_decisions) > batch_size:
        raise ValueError(
            f"planner returned {len(candidate_decisions)} candidate decisions, "
            f"maximum is {batch_size}"
        )
    decision_ids = [decision.decision_id for decision in plan.decisions]
    if any(not decision_id.strip() for decision_id in decision_ids):
        raise ValueError("decision ids cannot be empty")
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("decision ids must be unique")
    reused = set(decision_ids) & existing_decision_ids
    if reused:
        raise ValueError(f"decision ids were already used: {sorted(reused)}")
    planned_retests = set()
    validation_records = []
    for decision in plan.decisions:
        if decision.action not in allowed_actions:
            raise ValueError(
                f"{decision.action.value} is unavailable under the remaining budget"
            )
        if not decision.observation_ids:
            raise ValueError(f"{decision.action.value} requires objective observations")
        unknown_observations = set(decision.observation_ids) - observation_ids
        if unknown_observations:
            raise ValueError(
                f"unknown observation ids: {sorted(unknown_observations)}"
            )
        unknown_context = set(decision.generation_context_ids) - observation_ids
        if unknown_context:
            raise ValueError(
                f"unknown generation context ids: {sorted(unknown_context)}"
            )
        ungrounded_context = (
            set(decision.generation_context_ids) - set(decision.observation_ids)
        )
        if ungrounded_context:
            raise ValueError(
                "generation context must be selected from the controller's objective "
                f"observations: {sorted(ungrounded_context)}"
            )
        unknown_clusters = (
            set(decision.target_cluster_ids) - failure_cluster_ids
        )
        if unknown_clusters and decision.action != SearchAction.ABLATE:
            raise ValueError(
                "target_cluster_ids accepts failure clusters only; candidate behavior "
                f"clusters or unknown ids are invalid: {sorted(unknown_clusters)}"
            )
        target_splits = {
            cluster_id.split(":", 1)[0]
            for cluster_id in decision.target_cluster_ids
            if ":" in cluster_id
        }
        if len(target_splits) > 1:
            raise ValueError("one experiment can target only one evaluation split")
        if (
            decision.target_cluster_ids
            and _TARGET_CLUSTER_CONTRACTS[decision.action] == "empty"
        ):
            raise ValueError(
                f"{decision.action.value} cannot target failure clusters; use REPAIR"
            )

        if decision.action == SearchAction.DIVERSIFY:
            if decision.primary_parent_id is not None or decision.secondary_parent_id is not None:
                raise ValueError("DIVERSIFY cannot expose parent source")
        elif decision.action in _PARENT_ACTIONS:
            if decision.primary_parent_id not in visible_candidate_ids:
                raise ValueError(
                    f"primary parent is not visible: {decision.primary_parent_id}"
                )

        if decision.scope.kind != "full_solver" and candidate_source_structures:
            source_id = decision.primary_parent_id or "baseline"
            known_symbols = set(
                candidate_source_structures.get(source_id, {}).get("symbols", ())
            )
            unknown = [
                symbol
                for symbol in decision.scope.symbols
                if not _known_scope_symbol(symbol, known_symbols)
            ]
            if unknown:
                raise ValueError(f"scope names unknown source symbols: {sorted(unknown)}")

        if decision.action == SearchAction.CROSS:
            if decision.secondary_parent_id not in visible_candidate_ids:
                raise ValueError(
                    f"secondary parent is not visible: {decision.secondary_parent_id}"
                )
            if decision.secondary_parent_id == decision.primary_parent_id:
                raise ValueError("CROSS requires two different candidates")
            if decision.secondary_parent_id not in transfer_candidate_ids:
                raise ValueError("CROSS requires a transfer-qualified mechanism")
            if (
                decision.target_hypothesis_id is None
                or transfer_hypotheses.get(decision.target_hypothesis_id)
                != decision.secondary_parent_id
            ):
                raise ValueError(
                    "CROSS must name a transfer-qualified mechanism from its "
                    "secondary parent"
                )
        elif decision.secondary_parent_id is not None:
            raise ValueError(f"{decision.action.value} cannot use a secondary parent")

        if observation_records is not None:
            selected = {
                observation_id: observation_records[observation_id]
                for observation_id in decision.observation_ids
            }
            _validate_action_evidence(decision, selected, evidence_records)
            generation_context = {
                observation_id: observation_records[observation_id]
                for observation_id in decision.generation_context_ids
            }
            _validate_generation_context(decision, generation_context)
            validation_records.append({
                "decision_id": decision.decision_id,
                "action": decision.action.value,
                "observation_ids": list(decision.observation_ids),
                "generation_context_ids": list(decision.generation_context_ids),
                "evidence_ids": [
                    value
                    for value in decision.observation_ids
                    if value.startswith("evidence:")
                ],
                "parent_scope": (
                    _scope_record(candidate_scopes.get(decision.primary_parent_id))
                    if decision.primary_parent_id is not None
                    else None
                ),
                "requested_scope": _scope_record(decision.scope),
                "scope_transition": _scope_transition(
                    decision,
                    candidate_scopes.get(decision.primary_parent_id),
                ),
            })

        if decision.action == SearchAction.REPAIR and not decision.target_cluster_ids:
            raise ValueError("REPAIR requires a target failure cluster")
        if decision.action == SearchAction.ABLATE:
            if decision.target_hypothesis_id not in pending_hypothesis_ids:
                raise ValueError("ABLATE requires a pending hypothesis")
            required = ablation_targets.get(decision.target_hypothesis_id)
            if required is None:
                raise ValueError("ABLATE target has no registered experiment contract")
            candidate_id, scope, target_clusters = required
            if decision.primary_parent_id != candidate_id:
                raise ValueError(
                    "ABLATE must use the registered mechanism-on candidate"
                )
            if decision.scope != scope:
                raise ValueError("ABLATE must use the registered ablation scope")
            if decision.target_cluster_ids != target_clusters:
                raise ValueError(
                    "ABLATE must preserve the registered target clusters"
                )
        elif (
            decision.action != SearchAction.CROSS
            and decision.target_hypothesis_id is not None
        ):
            raise ValueError(f"{decision.action.value} cannot target an old hypothesis")

        if decision.hypothesis is not None:
            raise ValueError(
                "the controller cannot predeclare a mechanism hypothesis; "
                "the code agent owns that decision"
            )
        if decision.action == SearchAction.RETEST:
            if decision.retest_split != "train":
                raise ValueError("RETEST is restricted to the training split")
            if not decision.retest_seeds:
                raise ValueError("RETEST requires explicit new seeds")
            if len(set(decision.retest_seeds)) != len(decision.retest_seeds):
                raise ValueError("RETEST seeds must be unique")
            if len(set(decision.retest_instance_ids)) != len(
                decision.retest_instance_ids
            ):
                raise ValueError("RETEST instances must be unique")
            if allowed_retest_seeds and not set(decision.retest_seeds) <= allowed_retest_seeds:
                raise ValueError("RETEST uses seeds outside the diagnostic pool")
            allowed_instances = allowed_retest_instances.get(
                decision.retest_split,
                set(),
            )
            if decision.retest_instance_ids and not set(
                decision.retest_instance_ids
            ) <= allowed_instances:
                raise ValueError("RETEST uses unknown instances")
            retest_key = (
                decision.primary_parent_id,
                decision.retest_split,
                tuple(sorted(decision.retest_seeds)),
                tuple(sorted(decision.retest_instance_ids)),
            )
            if retest_key in planned_retests:
                raise ValueError("RETEST duplicates another planned RETEST")
            planned_retests.add(retest_key)
            if retest_key in completed_retests:
                raise ValueError("RETEST exactly duplicates an existing evaluation")
        elif (
            decision.retest_split is not None
            or decision.retest_seeds
            or decision.retest_instance_ids
        ):
            raise ValueError(f"{decision.action.value} cannot define retest parameters")

        if decision.action in _SCOPE_PRESERVING_ACTIONS:
            parent_scope = candidate_scopes.get(decision.primary_parent_id)
            if parent_scope is None:
                raise ValueError(
                    f"{decision.action.value} requires a recorded parent scope"
                )
            if decision.scope != parent_scope:
                raise ValueError(
                    f"{decision.action.value} must preserve parent scope; use "
                    "EXPAND_SCOPE or NARROW_SCOPE to change it"
                )
        elif decision.action in {SearchAction.EXPAND_SCOPE, SearchAction.NARROW_SCOPE}:
            parent_scope = candidate_scopes.get(decision.primary_parent_id)
            if parent_scope is None:
                raise ValueError(
                    f"{decision.action.value} requires a recorded parent scope"
                )
            if (
                decision.action == SearchAction.EXPAND_SCOPE
                and not _is_strictly_broader(decision.scope, parent_scope)
            ):
                raise ValueError("EXPAND_SCOPE requires a broader scope than its parent")
            if decision.action == SearchAction.EXPAND_SCOPE:
                boundary = scope_boundary_evidence.get(
                    decision.scope_evidence_id or ""
                )
                if boundary is None:
                    raise ValueError("EXPAND_SCOPE requires scope boundary evidence")
                if boundary.candidate_id != decision.primary_parent_id:
                    raise ValueError("scope boundary evidence candidate mismatch")
                if boundary.source_scope != parent_scope:
                    raise ValueError("scope boundary evidence source scope mismatch")
                if boundary.approved_scope != decision.scope:
                    raise ValueError("EXPAND_SCOPE exceeds its approved boundary")
            if (
                decision.action == SearchAction.NARROW_SCOPE
                and not _is_strictly_narrower(decision.scope, parent_scope)
            ):
                raise ValueError("NARROW_SCOPE requires a narrower scope than its parent")
        if (
            decision.action != SearchAction.EXPAND_SCOPE
            and decision.scope_evidence_id is not None
        ):
            raise ValueError("scope_evidence_id is valid only for EXPAND_SCOPE")
    return {
        "round_index": plan.round_index,
        "candidate_decision_count": len(candidate_decisions),
        "diagnostic_decision_count": len(plan.decisions) - len(candidate_decisions),
        "stopped": False,
        "stop_reason": None,
        "decisions": validation_records,
    }


def _validate_action_evidence(
    decision: PlannerDecision,
    selected: dict[str, Observation | dict],
    evidence_records: dict[str, EvidenceRecord],
) -> None:
    kinds_by_record = tuple(
        _observation_kind(record) for record in selected.values()
    )
    values = [_observation_values(record) for record in selected.values()]
    linked_performance_evidence = any(
        kind in _PERFORMANCE_EVIDENCE_KINDS
        and _links_candidate(value, decision.primary_parent_id)
        for kind, value in zip(kinds_by_record, values)
    )

    if decision.action == SearchAction.DIVERSIFY:
        return

    if decision.action == SearchAction.REPAIR:
        matching_clusters = [
            value
            for kind, value in zip(
                (_observation_kind(record) for record in selected.values()),
                values,
            )
            if kind == "failure_cluster"
            and value.get("cluster_id") in decision.target_cluster_ids
        ]
        if not matching_clusters:
            raise ValueError("REPAIR must select its target failure-cluster observation")
        if not any(
            decision.primary_parent_id in value.get("failing_candidate_ids", ())
            for value in matching_clusters
        ):
            raise ValueError("REPAIR parent is not failing on the selected cluster")
        return

    if decision.action in {SearchAction.CROSS, SearchAction.ABLATE}:
        expected_status = (
            "supported" if decision.action == SearchAction.CROSS else "pending"
        )
        matching = [
            evidence_records.get(observation_id)
            for observation_id in selected
            if observation_id.startswith("evidence:")
        ]
        if not any(
            record is not None
            and record.status == expected_status
            and record.hypothesis_id == decision.target_hypothesis_id
            and record.candidate_id
            == (
                decision.secondary_parent_id
                if decision.action == SearchAction.CROSS
                else decision.primary_parent_id
            )
            for record in matching
        ):
            raise ValueError(
                f"{decision.action.value} must select {expected_status} evidence "
                "for the named mechanism"
            )
        return

    if not linked_performance_evidence:
        raise ValueError(
            f"{decision.action.value} requires performance evidence linked to "
            "primary parent"
        )


def _validate_generation_context(
    decision: PlannerDecision,
    selected: dict[str, Observation | dict],
) -> None:
    records = [
        (_observation_kind(record), _observation_values(record))
        for record in selected.values()
    ]
    allowed_kinds = _GENERATION_CONTEXT_KINDS[decision.action]
    disallowed_kinds = sorted({
        kind for kind, _ in records if kind not in allowed_kinds
    })
    if disallowed_kinds:
        if "candidate_cluster" in disallowed_kinds:
            raise ValueError(
                "candidate behavior clusters are controller evidence, not code-agent "
                "generation context"
            )
        if decision.action == SearchAction.DIVERSIFY:
            raise ValueError(
                "DIVERSIFY generation context may contain only problem, baseline, "
                "and failure-profile observations; disallowed kinds: "
                f"{disallowed_kinds}"
            )
        raise ValueError(
            f"{decision.action.value} generation context contains disallowed "
            f"observation kinds: {disallowed_kinds}"
        )

    if decision.action == SearchAction.RETEST:
        return

    for kind, values in records:
        link_contract = _GENERATION_CONTEXT_LINK_CONTRACTS[
            decision.action
        ].get(kind)
        if link_contract == "baseline_only":
            if values.get("candidate_id") not in {
                "baseline",
                "__baseline_control__",
            }:
                raise ValueError(
                    "DIVERSIFY generation context cannot expose candidate source"
                )
            continue
        if link_contract == "primary_parent_only" and not _links_candidate(
            values,
            decision.primary_parent_id,
        ):
            raise ValueError(
                f"{decision.action.value} generation context exposes a non-parent "
                "candidate"
            )
        if link_contract not in {
            "targeted_primary_mechanism_only",
            "targeted_secondary_mechanism_only",
        }:
            continue
        expected_candidate = (
            decision.secondary_parent_id
            if link_contract == "targeted_secondary_mechanism_only"
            else decision.primary_parent_id
        )
        if (
            values.get("candidate_id") != expected_candidate
            or values.get("hypothesis_id") != decision.target_hypothesis_id
        ):
            raise ValueError(
                f"{decision.action.value} generation context exposes an unrelated "
                "mechanism"
            )


def _observation_kind(record: Observation | dict) -> str:
    return record.kind if isinstance(record, Observation) else str(record["kind"])


def _observation_values(record: Observation | dict) -> dict:
    return record.values if isinstance(record, Observation) else dict(record.get("values", {}))


def _links_candidate(values: dict, candidate_id: str | None) -> bool:
    if candidate_id is None:
        return False
    return candidate_id in {
        values.get("candidate_id"),
        values.get("parent_id"),
    } or candidate_id in {
        *values.get("candidate_ids", ()),
        *values.get("failing_candidate_ids", ()),
    }


def _known_scope_symbol(symbol: str, known: set[str]) -> bool:
    if symbol in known:
        return True
    return sum(item.rsplit(".", 1)[-1] == symbol for item in known) == 1


def _is_strictly_broader(requested: ScopeSpec, parent: ScopeSpec) -> bool:
    if parent.kind == "full_solver":
        return False
    if requested.kind == "full_solver":
        return True
    return set(requested.symbols) > set(parent.symbols)


def _is_strictly_narrower(requested: ScopeSpec, parent: ScopeSpec) -> bool:
    if requested.kind == "full_solver":
        return False
    if parent.kind == "full_solver":
        return True
    return set(requested.symbols) < set(parent.symbols)


def _scope_record(scope: ScopeSpec | None) -> dict | None:
    if scope is None:
        return None
    return {"kind": scope.kind, "symbols": list(scope.symbols)}


def _scope_transition(
    decision: PlannerDecision,
    parent_scope: ScopeSpec | None,
) -> str:
    if decision.action == SearchAction.DIVERSIFY:
        return "initial"
    if decision.action == SearchAction.EXPAND_SCOPE:
        return "expand"
    if decision.action == SearchAction.NARROW_SCOPE:
        return "narrow"
    if decision.action == SearchAction.ABLATE:
        return "ablation"
    if parent_scope is not None and decision.scope == parent_scope:
        return "same"
    return "unknown"
