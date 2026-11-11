from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from agent_ahd.core.mechanism_transfer import MechanismTransferRecord
from agent_ahd.core.planner_policy import validate_round_plan
from agent_ahd.core.research_records import (
    ActionCost,
    EvidenceRecord,
    InheritanceChannel,
    ControllerSelection,
    PlannerDecision,
    RoundPlan,
    ScopeBoundaryEvidence,
    ScopeSpec,
    SearchAction,
)


@dataclass(frozen=True)
class DecisionOption:
    option_id: str
    decision: PlannerDecision
    cost: ActionCost = ActionCost()

    def to_dict(self) -> dict:
        return {
            "option_id": self.option_id,
            "decision": asdict(self.decision),
            "cost": asdict(self.cost),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionOption":
        normalized = dict(data)
        normalized.setdefault("cost", {})
        _require_fields(
            normalized,
            {"option_id", "decision", "cost"},
            "decision option",
        )
        return cls(
            option_id=str(normalized["option_id"]),
            decision=PlannerDecision.from_dict(normalized["decision"]),
            cost=ActionCost.from_dict(normalized["cost"]),
        )


@dataclass(frozen=True)
class DecisionCatalog:
    round_index: int
    available_actions: tuple[SearchAction, ...]
    candidate_options: tuple[DecisionOption, ...]
    diagnostic_options: tuple[DecisionOption, ...] = ()

    def to_dict(self) -> dict:
        return {
            "round_index": self.round_index,
            "available_actions": [
                action.value for action in self.available_actions
            ],
            "candidate_options": [
                option.to_dict() for option in self.candidate_options
            ],
            "diagnostic_options": [
                option.to_dict() for option in self.diagnostic_options
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionCatalog":
        _require_fields(
            data,
            {
                "round_index",
                "available_actions",
                "candidate_options",
                "diagnostic_options",
            },
            "decision catalog",
        )
        return cls(
            round_index=int(data["round_index"]),
            available_actions=tuple(
                SearchAction(value)
                for value in data["available_actions"]
            ),
            candidate_options=tuple(
                DecisionOption.from_dict(item)
                for item in data["candidate_options"]
            ),
            diagnostic_options=tuple(
                DecisionOption.from_dict(item)
                for item in data["diagnostic_options"]
            ),
        )


def controller_selection_schema(
    catalog: DecisionCatalog,
    *,
    candidate_count: int,
) -> dict:
    candidate_ids = [option.option_id for option in catalog.candidate_options]
    if not candidate_ids:
        raise ValueError("decision catalog has no candidate options")
    diagnostic_ids = [
        option.option_id for option in catalog.diagnostic_options
    ]
    diagnostic_items = {"type": "string"}
    if diagnostic_ids:
        diagnostic_items["enum"] = diagnostic_ids
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "round_index",
            "candidate_option_ids",
            "diagnostic_option_ids",
            "stop",
            "stop_reason",
        ],
        "properties": {
            "round_index": {
                "type": "integer",
                "const": catalog.round_index,
            },
            "candidate_option_ids": {
                "type": "array",
                "minItems": 0,
                "maxItems": candidate_count,
                "items": {
                    "type": "string",
                    "enum": candidate_ids,
                },
            },
            "diagnostic_option_ids": {
                "type": "array",
                "maxItems": len(diagnostic_ids),
                "items": diagnostic_items,
            },
            "stop": {"type": "boolean"},
            "stop_reason": {
                "anyOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "null"},
                ]
            },
        },
    }


def materialize_controller_selection(
    selection: ControllerSelection,
    catalog: DecisionCatalog,
) -> RoundPlan:
    if selection.round_index != catalog.round_index:
        raise ValueError(
            f"selection round {selection.round_index} does not match catalog "
            f"round {catalog.round_index}"
        )
    candidates = {
        option.option_id: option for option in catalog.candidate_options
    }
    diagnostics = {
        option.option_id: option for option in catalog.diagnostic_options
    }
    unknown_candidates = set(selection.candidate_option_ids) - set(candidates)
    if unknown_candidates:
        raise ValueError(
            f"unknown candidate option ids: {sorted(unknown_candidates)}"
        )
    unknown_diagnostics = set(selection.diagnostic_option_ids) - set(
        diagnostics
    )
    if unknown_diagnostics:
        raise ValueError(
            f"unknown diagnostic option ids: {sorted(unknown_diagnostics)}"
        )
    if len(set(selection.diagnostic_option_ids)) != len(
        selection.diagnostic_option_ids
    ):
        raise ValueError("diagnostic option ids must be unique")

    if selection.stop:
        return RoundPlan(
            round_index=catalog.round_index,
            decisions=(),
            stop_reason=selection.stop_reason,
        )

    decisions = [
        replace(
            candidates[option_id].decision,
            decision_id=(
                f"round_{catalog.round_index:03d}_candidate_{index:02d}_"
                f"{option_id}"
            ),
            cost=candidates[option_id].cost,
        )
        for index, option_id in enumerate(selection.candidate_option_ids)
    ]
    decisions.extend(
        replace(
            diagnostics[option_id].decision,
            decision_id=(
                f"round_{catalog.round_index:03d}_diagnostic_{index:02d}_"
                f"{option_id}"
            ),
            cost=diagnostics[option_id].cost,
        )
        for index, option_id in enumerate(selection.diagnostic_option_ids)
    )
    return RoundPlan(catalog.round_index, tuple(decisions))


def build_decision_catalog(
    *,
    round_index: int,
    available_actions: tuple[SearchAction, ...],
    observations: dict[str, dict],
    failure_clusters: dict[str, dict],
    evidence: Iterable[EvidenceRecord],
    scope_boundary_evidence: Iterable[ScopeBoundaryEvidence] = (),
    mechanism_records: Iterable[MechanismTransferRecord] = (),
    candidate_scopes: dict[str, ScopeSpec],
    candidate_source_structures: dict[str, dict],
    ablation_targets: dict[
        str,
        tuple[str, ScopeSpec, tuple[str, ...]],
    ] | None = None,
    diagnostic_seeds: tuple[int, ...] = (),
    retest_instance_ids: tuple[str, ...] = (),
    completed_retests: set[
        tuple[str, str, tuple[int, ...], tuple[str, ...]]
    ] | None = None,
) -> DecisionCatalog:
    evidence = tuple(evidence)
    scope_boundary_evidence = tuple(scope_boundary_evidence)
    mechanism_records = tuple(mechanism_records)
    ablation_targets = ablation_targets or {}
    completed_retests = completed_retests or set()
    candidate_options = []
    diagnostic_options = []
    visible = tuple(sorted(candidate_scopes))

    if SearchAction.EXPLOIT in available_actions:
        for candidate_id in visible:
            context = _parent_context(observations, candidate_id)
            candidate_options.append(
                _option(
                    round_index,
                    SearchAction.EXPLOIT,
                    primary_parent_id=candidate_id,
                    observation_ids=context,
                    generation_context_ids=context,
                    scope=candidate_scopes[candidate_id],
                )
            )

    if SearchAction.REPAIR in available_actions:
        for cluster_id, cluster in sorted(failure_clusters.items()):
            for candidate_id in sorted(cluster["failing_candidate_ids"]):
                if candidate_id not in candidate_scopes:
                    continue
                observation_ids = _parent_context(
                    observations,
                    candidate_id,
                    cluster_id=cluster_id,
                )
                candidate_options.append(
                    _option(
                        round_index,
                        SearchAction.REPAIR,
                        primary_parent_id=candidate_id,
                        target_cluster_ids=(cluster_id,),
                        observation_ids=observation_ids,
                        generation_context_ids=observation_ids,
                        scope=candidate_scopes[candidate_id],
                    )
                )

    if SearchAction.CROSS in available_actions:
        for item in sorted(
            (
                record
                for record in mechanism_records
                if record.transfer_eligible
            ),
            key=lambda record: (record.candidate_id, record.hypothesis_id),
        ):
            if item.candidate_id not in candidate_scopes:
                continue
            evidence_id = _evidence_observation_id(item)
            for primary_id in visible:
                if primary_id == item.candidate_id:
                    continue
                context = _parent_context(
                    observations,
                    primary_id,
                    evidence_ids=(evidence_id,),
                    include_parent_mechanisms=False,
                )
                generation_context = _parent_context(
                    observations,
                    primary_id,
                    evidence_ids=(evidence_id,),
                    include_diagnostics=False,
                    include_parent_mechanisms=False,
                )
                candidate_options.append(
                    _option(
                        round_index,
                        SearchAction.CROSS,
                        primary_parent_id=primary_id,
                        secondary_parent_id=item.candidate_id,
                        target_hypothesis_id=item.hypothesis_id,
                        observation_ids=context,
                        generation_context_ids=generation_context,
                        scope=candidate_scopes[primary_id],
                    )
                )

    if SearchAction.DIVERSIFY in available_actions:
        baseline_context = _baseline_context(observations)
        candidate_options.append(
            _option(
                round_index,
                SearchAction.DIVERSIFY,
                observation_ids=baseline_context,
                generation_context_ids=baseline_context,
                scope=ScopeSpec("full_solver"),
            )
        )
        for cluster_id in sorted(failure_clusters):
            cluster_observation = f"cluster:{cluster_id}"
            if cluster_observation not in observations:
                continue
            context = tuple(sorted((*baseline_context, cluster_observation)))
            candidate_options.append(
                _option(
                    round_index,
                    SearchAction.DIVERSIFY,
                    observation_ids=context,
                    generation_context_ids=context,
                    scope=ScopeSpec("full_solver"),
                    identity_context=cluster_id,
                )
            )

    if SearchAction.EXPAND_SCOPE in available_actions:
        for boundary in sorted(
            scope_boundary_evidence,
            key=lambda item: (item.candidate_id, item.evidence_id),
        ):
            candidate_id = boundary.candidate_id
            if candidate_id not in candidate_scopes:
                continue
            parent_scope = candidate_scopes[candidate_id]
            if boundary.source_scope != parent_scope:
                continue
            context = _parent_context(observations, candidate_id)
            candidate_options.append(
                _option(
                    round_index,
                    SearchAction.EXPAND_SCOPE,
                    primary_parent_id=candidate_id,
                    observation_ids=context,
                    generation_context_ids=context,
                    scope=boundary.approved_scope,
                    scope_evidence_id=boundary.evidence_id,
                )
            )

    if SearchAction.NARROW_SCOPE in available_actions:
        for candidate_id in visible:
            parent_scope = candidate_scopes[candidate_id]
            known = _source_symbols(
                candidate_source_structures,
                candidate_id,
            )
            if parent_scope.kind == "full_solver":
                symbol_sets = [known]
                if len(known) > 1:
                    symbol_sets.extend(
                        tuple(item for item in known if item != removed)
                        for removed in known
                    )
            else:
                symbol_sets = [
                    tuple(
                        item
                        for item in parent_scope.symbols
                        if item != removed
                    )
                    for removed in parent_scope.symbols
                    if len(parent_scope.symbols) > 1
                ]
            context = _parent_context(observations, candidate_id)
            for symbols in dict.fromkeys(symbol_sets):
                if not symbols:
                    continue
                candidate_options.append(
                    _option(
                        round_index,
                        SearchAction.NARROW_SCOPE,
                        primary_parent_id=candidate_id,
                        observation_ids=context,
                        generation_context_ids=context,
                        scope=ScopeSpec("symbol_set", tuple(sorted(symbols))),
                    )
                )

    if SearchAction.ABLATE in available_actions:
        for item in sorted(
            (record for record in evidence if record.status == "pending"),
            key=lambda record: (record.candidate_id, record.hypothesis_id),
        ):
            target = ablation_targets.get(item.hypothesis_id)
            if target is None:
                continue
            candidate_id, scope, target_clusters = target
            if candidate_id not in candidate_scopes:
                continue
            context = _parent_context(
                observations,
                candidate_id,
                cluster_ids=target_clusters,
                evidence_ids=(_evidence_observation_id(item),),
                include_parent_mechanisms=False,
            )
            generation_context = _parent_context(
                observations,
                candidate_id,
                cluster_ids=target_clusters,
                evidence_ids=(_evidence_observation_id(item),),
                include_diagnostics=False,
                include_parent_mechanisms=False,
            )
            candidate_options.append(
                _option(
                    round_index,
                    SearchAction.ABLATE,
                    primary_parent_id=candidate_id,
                    target_cluster_ids=target_clusters,
                    target_hypothesis_id=item.hypothesis_id,
                    observation_ids=context,
                    generation_context_ids=generation_context,
                    scope=scope,
                )
            )

    if SearchAction.RETEST in available_actions:
        for candidate_id in visible:
            observations_for_parent = _parent_context(
                observations,
                candidate_id,
                include_source=False,
                include_parent_mechanisms=False,
            )
            for seed in sorted(diagnostic_seeds):
                retest_key = (
                    candidate_id,
                    "train",
                    (seed,),
                    tuple(sorted(retest_instance_ids)),
                )
                if retest_key in completed_retests:
                    continue
                diagnostic_options.append(
                    _option(
                        round_index,
                        SearchAction.RETEST,
                        primary_parent_id=candidate_id,
                        observation_ids=observations_for_parent,
                        scope=candidate_scopes[candidate_id],
                        retest_split="train",
                        retest_seeds=(seed,),
                        retest_instance_ids=tuple(
                            sorted(retest_instance_ids)
                        ),
                    )
                )

    catalog = DecisionCatalog(
        round_index=round_index,
        available_actions=available_actions,
        candidate_options=_unique_options(candidate_options),
        diagnostic_options=_unique_options(diagnostic_options),
    )
    for option in catalog.candidate_options + catalog.diagnostic_options:
        validate_catalog_option(
            option,
            catalog=catalog,
            observations=observations,
            failure_clusters=failure_clusters,
            evidence=evidence,
            scope_boundary_evidence=scope_boundary_evidence,
            mechanism_records=mechanism_records,
            candidate_scopes=candidate_scopes,
            candidate_source_structures=candidate_source_structures,
            ablation_targets=ablation_targets,
            diagnostic_seeds=diagnostic_seeds,
            retest_instance_ids=retest_instance_ids,
            completed_retests=completed_retests,
        )
    return catalog


def validate_catalog_option(
    option: DecisionOption,
    *,
    catalog: DecisionCatalog,
    observations: dict[str, dict],
    failure_clusters: dict[str, dict],
    evidence: Iterable[EvidenceRecord],
    scope_boundary_evidence: Iterable[ScopeBoundaryEvidence] = (),
    mechanism_records: Iterable[MechanismTransferRecord] = (),
    candidate_scopes: dict[str, ScopeSpec],
    candidate_source_structures: dict[str, dict],
    ablation_targets: dict[
        str,
        tuple[str, ScopeSpec, tuple[str, ...]],
    ] | None = None,
    diagnostic_seeds: tuple[int, ...] = (),
    retest_instance_ids: tuple[str, ...] = (),
    completed_retests: set[
        tuple[str, str, tuple[int, ...], tuple[str, ...]]
    ] | None = None,
) -> None:
    evidence = tuple(evidence)
    scope_boundary_evidence = tuple(scope_boundary_evidence)
    mechanism_records = tuple(mechanism_records)
    ablation_targets = ablation_targets or {}
    completed_retests = completed_retests or set()
    decision = option.decision
    validate_round_plan(
        RoundPlan(catalog.round_index, (decision,)),
        round_index=catalog.round_index,
        batch_size=0 if decision.action == SearchAction.RETEST else 1,
        visible_candidate_ids=set(candidate_scopes),
        observation_ids=set(observations),
        observation_records=observations,
        evidence_records={
            _evidence_observation_id(item): item for item in evidence
        },
        scope_boundary_evidence={
            item.evidence_id: item for item in scope_boundary_evidence
        },
        failure_cluster_ids=set(failure_clusters),
        pending_hypothesis_ids={
            item.hypothesis_id for item in evidence if item.status == "pending"
        },
        transfer_candidate_ids={
            item.candidate_id
            for item in mechanism_records
            if item.transfer_eligible
        },
        transfer_hypotheses={
            item.hypothesis_id: item.candidate_id
            for item in mechanism_records
            if item.transfer_eligible
        },
        ablation_targets=ablation_targets,
        candidate_scopes=candidate_scopes,
        candidate_source_structures=candidate_source_structures,
        allowed_retest_seeds=set(diagnostic_seeds),
        allowed_retest_instances={
            "train": set(retest_instance_ids),
        },
        completed_retests=completed_retests,
        allowed_actions=set(catalog.available_actions),
    )


def _option(
    round_index: int,
    action: SearchAction,
    *,
    scope: ScopeSpec,
    primary_parent_id: str | None = None,
    secondary_parent_id: str | None = None,
    target_cluster_ids: tuple[str, ...] = (),
    observation_ids: tuple[str, ...] = (),
    generation_context_ids: tuple[str, ...] = (),
    target_hypothesis_id: str | None = None,
    retest_split: str | None = None,
    retest_seeds: tuple[int, ...] = (),
    retest_instance_ids: tuple[str, ...] = (),
    identity_context: str | None = None,
    scope_evidence_id: str | None = None,
) -> DecisionOption:
    identity_payload = {
        "action": action.value,
        "primary_parent_id": primary_parent_id,
        "secondary_parent_id": secondary_parent_id,
        "target_cluster_ids": target_cluster_ids,
        "target_hypothesis_id": target_hypothesis_id,
        "scope": {
            "kind": scope.kind,
            "symbols": scope.symbols,
        },
        "retest_split": retest_split,
        "retest_seeds": retest_seeds,
        "retest_instance_ids": retest_instance_ids,
        "identity_context": identity_context,
        "scope_evidence_id": scope_evidence_id,
    }
    if action == SearchAction.DIVERSIFY:
        inheritance_channel = InheritanceChannel.INDEPENDENT
    elif action == SearchAction.CROSS:
        inheritance_channel = InheritanceChannel.MECHANISM_ONLY
    elif scope.kind == "symbol_set":
        inheritance_channel = InheritanceChannel.LOCAL_SOURCE
    else:
        inheritance_channel = InheritanceChannel.FULL_SOURCE
    identity_payload["inheritance_channel"] = inheritance_channel.value
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    option_id = f"option_{action.value.lower()}_{digest}"
    return DecisionOption(
        option_id=option_id,
        decision=PlannerDecision(
            decision_id=f"catalog_round_{round_index}_{option_id}",
            action=action,
            scope=scope,
            primary_parent_id=primary_parent_id,
            secondary_parent_id=secondary_parent_id,
            target_cluster_ids=target_cluster_ids,
            observation_ids=observation_ids,
            generation_context_ids=generation_context_ids,
            target_hypothesis_id=target_hypothesis_id,
            retest_split=retest_split,
            retest_seeds=retest_seeds,
            retest_instance_ids=retest_instance_ids,
            inheritance_channel=inheritance_channel,
            scope_evidence_id=scope_evidence_id,
        ),
    )


def _parent_context(
    observations: dict[str, dict],
    candidate_id: str,
    *,
    cluster_id: str | None = None,
    cluster_ids: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
    include_source: bool = True,
    include_diagnostics: bool = True,
    include_parent_mechanisms: bool = True,
) -> tuple[str, ...]:
    selected = []
    for observation_id, record in sorted(observations.items()):
        values = record.get("values", {})
        kind = record.get("kind")
        if (
            values.get("candidate_id") == candidate_id
            and (
                kind == "candidate_profile"
                or (
                    include_diagnostics
                    and kind == "evaluation_diagnostic"
                )
            )
        ):
            selected.append(observation_id)
        elif (
            include_source
            and kind == "source_structure"
            and values.get("candidate_id") == candidate_id
        ):
            selected.append(observation_id)
        elif (
            include_parent_mechanisms
            and kind == "mechanism_evidence"
            and values.get("candidate_id") == candidate_id
        ):
            selected.append(observation_id)
    for target_cluster in (
        (cluster_id,) if cluster_id is not None else cluster_ids
    ):
        observation_id = f"cluster:{target_cluster}"
        if observation_id in observations:
            selected.append(observation_id)
    selected.extend(
        observation_id
        for observation_id in evidence_ids
        if observation_id in observations
    )
    return tuple(sorted(set(selected)))


def _baseline_context(observations: dict[str, dict]) -> tuple[str, ...]:
    return tuple(sorted(
        observation_id
        for observation_id, record in observations.items()
        if record.get("kind") == "baseline_control"
        or (
            record.get("kind") in {"problem", "source_structure"}
            and record.get("values", {}).get("candidate_id") == "baseline"
        )
    ))


def _source_symbols(
    structures: dict[str, dict],
    candidate_id: str,
) -> tuple[str, ...]:
    return tuple(sorted(set(
        str(symbol)
        for symbol in structures.get(candidate_id, {}).get("symbols", ())
    )))


def _unique_options(
    options: list[DecisionOption],
) -> tuple[DecisionOption, ...]:
    by_id = {option.option_id: option for option in options}
    return tuple(by_id[option_id] for option_id in sorted(by_id))


def _evidence_observation_id(record: EvidenceRecord) -> str:
    return (
        record.evidence_id
        if record.evidence_id.startswith("evidence:")
        else f"evidence:{record.evidence_id}"
    )


def _require_fields(
    data: dict,
    expected: set[str],
    label: str,
) -> None:
    actual = set(data)
    if actual != expected:
        raise ValueError(
            f"{label} fields do not match contract: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
