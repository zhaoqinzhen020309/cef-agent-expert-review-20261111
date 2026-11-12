from __future__ import annotations

import math
import random
from collections.abc import Sequence

from agent_ahd.core.search_state import SearchState, SolverNode


_EPSILON = 1e-12
_MIN_EXPONENT = -700.0


def annealing_acceptance_probability(
    *,
    child_gap: float,
    parent_gap: float,
    temperature: float,
) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if child_gap <= parent_gap:
        return 1.0
    return _positive_exp(-(child_gap - parent_gap) / temperature)


def boltzmann_weights(gaps: Sequence[float], *, temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not gaps:
        return []
    best = min(gaps)
    return [_positive_exp(-(gap - best) / temperature) for gap in gaps]


def plan_initial_round(state: SearchState) -> list[SolverNode]:
    if state.nodes:
        raise RuntimeError("initial round requires an empty search state")
    return [
        _allocate_fresh(state, round_index=0, selection_reason="initial_fresh")
        for _ in range(state.config.initial_size)
    ]


def finalize_round(state: SearchState, round_index: int) -> None:
    round_nodes = [node for node in state.nodes if node.round_index == round_index]
    if not round_nodes:
        raise RuntimeError(f"round {round_index} has no nodes")
    if any(node.status == "pending" for node in round_nodes):
        raise RuntimeError(f"round {round_index} has pending nodes")

    for node in round_nodes:
        _update_child_mode_weights(state, node)

    old_best = state.require_node(state.best_node_id) if state.best_node_id else None
    valid_nodes = [node for node in state.nodes if node.valid and node.score is not None]
    if not valid_nodes:
        state.finish_round(round_index)
        return

    best = min(valid_nodes, key=lambda node: (node.score, node.node_id))
    improved = old_best is None or best.score < old_best.score - _EPSILON
    state.best_node_id = best.node_id

    if old_best is not None:
        if improved:
            state.rounds_without_global_improvement = 0
            state.temperature *= state.config.temperature_decay
        else:
            state.rounds_without_global_improvement += 1
            if (
                state.rounds_without_global_improvement
                >= state.config.global_stagnation_rounds
            ):
                state.temperature += state.config.reheat_increment
                state.rounds_without_global_improvement = 0
                state.force_fresh_next_round = True
            else:
                state.temperature *= state.config.temperature_decay

    state.finish_round(round_index)


def plan_next_round(state: SearchState) -> list[SolverNode]:
    if state.pending_node_ids:
        raise RuntimeError("cannot plan while nodes are pending")
    if state.current_round < 0:
        raise RuntimeError("initial round is not complete")

    valid_nodes = _valid_nodes(state)
    if not valid_nodes:
        raise RuntimeError("search has no valid parent nodes")

    rng = state.restore_rng()
    next_round = state.current_round + 1
    if state.force_fresh_next_round:
        fresh_reasons = ["stagnation_fresh"]
    elif should_open_fresh_root(state):
        fresh_reasons = ["progressive_fresh"]
    else:
        fresh_reasons = []
    fresh_slots = len(fresh_reasons)
    inherited_slots = state.config.batch_size - fresh_slots

    accepted = _accepted_latest_children(state, rng)
    accepted_selected = _weighted_sample_nodes(
        accepted,
        min(inherited_slots, len(accepted)),
        state.temperature,
        rng,
    )
    selected: list[tuple[SolverNode, str]] = [
        (node, "annealing_accept") for node in accepted_selected
    ]

    if len(selected) < inherited_slots:
        selected_ids = {node.node_id for node, _reason in selected}
        historical = [node for node in valid_nodes if node.node_id not in selected_ids]
        historical_selected = _weighted_sample_nodes(
            historical,
            min(inherited_slots - len(selected), len(historical)),
            state.temperature,
            rng,
        )
        selected.extend((node, "history_boltzmann") for node in historical_selected)

    fresh_reasons.extend(
        "capacity_fresh" for _ in range(inherited_slots - len(selected))
    )
    planned = [
        _allocate_child(state, parent, next_round, rng, selection_reason)
        for parent, selection_reason in selected
    ]
    planned.extend(
        _allocate_fresh(state, next_round, selection_reason)
        for selection_reason in fresh_reasons
    )

    if len(planned) != state.config.batch_size:
        raise RuntimeError(
            f"policy planned {len(planned)} jobs, expected {state.config.batch_size}"
        )
    state.force_fresh_next_round = False
    state.capture_rng(rng)
    return planned


def should_open_fresh_root(state: SearchState) -> bool:
    evaluated_count = sum(node.status != "pending" for node in state.nodes)
    root_count = sum(node.primary_parent_id is None for node in state.nodes)
    root_budget = max(
        state.config.initial_size,
        math.floor(
            max(evaluated_count, 1) ** state.config.progressive_widening_exponent
        ),
    )
    return root_count < root_budget


def branch_stagnation_count(state: SearchState, root_id: str) -> int:
    branch = sorted(
        (
            node
            for node in state.nodes
            if node.root_id == root_id and node.valid and node.score is not None
        ),
        key=lambda node: (node.round_index, node.node_id),
    )
    best = math.inf
    count = 0
    for node in branch:
        if node.score < best - _EPSILON:
            best = node.score
            count = 0
        else:
            count += 1
    return count


def select_reference_node(
    state: SearchState,
    primary: SolverNode,
    rng: random.Random,
) -> SolverNode | None:
    candidates = [
        node
        for node in _valid_nodes(state)
        if node.root_id != primary.root_id
    ]
    sampled = _weighted_sample_nodes(candidates, 1, state.temperature, rng)
    return sampled[0] if sampled else None


def _accepted_latest_children(
    state: SearchState,
    rng: random.Random,
) -> list[SolverNode]:
    accepted = []
    for child in state.nodes:
        if (
            child.round_index != state.current_round
            or not child.valid
            or child.score is None
            or child.primary_parent_id is None
        ):
            continue
        parent = state.require_node(child.primary_parent_id)
        if not parent.valid or parent.score is None:
            continue
        probability = annealing_acceptance_probability(
            child_gap=child.score,
            parent_gap=parent.score,
            temperature=state.temperature,
        )
        if rng.random() < probability:
            accepted.append(child)
    return accepted


def _weighted_sample_nodes(
    nodes: Sequence[SolverNode],
    count: int,
    temperature: float,
    rng: random.Random,
) -> list[SolverNode]:
    pool = list(nodes)
    selected: list[SolverNode] = []
    while pool and len(selected) < count:
        weights = boltzmann_weights(
            [node.score for node in pool if node.score is not None],
            temperature=temperature,
        )
        if len(weights) != len(pool):
            raise ValueError("weighted sampling requires scored nodes")
        index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        selected.append(pool.pop(index))
    return selected


def _allocate_child(
    state: SearchState,
    parent: SolverNode,
    round_index: int,
    rng: random.Random,
    selection_reason: str,
) -> SolverNode:
    base_mode = _select_generation_mode(parent, state.temperature, rng)
    reference = None
    if (
        branch_stagnation_count(state, parent.root_id)
        >= state.config.branch_stagnation_attempts
    ):
        reference = select_reference_node(state, parent, rng)
    generation_mode = (
        f"cross_reference_{base_mode}" if reference is not None else base_mode
    )
    return state.allocate_node(
        round_index=round_index,
        generation_mode=generation_mode,
        solver_relpath=_next_solver_relpath(state),
        primary_parent_id=parent.node_id,
        reference_node_id=reference.node_id if reference is not None else None,
        parent_selection=selection_reason,
        reference_reason="branch_stagnation" if reference is not None else None,
    )


def _allocate_fresh(
    state: SearchState,
    round_index: int,
    selection_reason: str,
) -> SolverNode:
    return state.allocate_node(
        round_index=round_index,
        generation_mode="fresh",
        solver_relpath=_next_solver_relpath(state),
        parent_selection=selection_reason,
    )


def _next_solver_relpath(state: SearchState) -> str:
    node_id = f"node_{state.next_node_index:04d}"
    return f"nodes/{node_id}/solver.py"


def _select_generation_mode(
    parent: SolverNode,
    temperature: float,
    rng: random.Random,
) -> str:
    modes = ("continue", "redesign")
    raw = [parent.mode_weights[mode] for mode in modes]
    maximum = max(raw)
    weights = [
        _positive_exp((value - maximum) / max(temperature, _EPSILON))
        for value in raw
    ]
    return rng.choices(modes, weights=weights, k=1)[0]


def _update_child_mode_weights(state: SearchState, child: SolverNode) -> None:
    if (
        not child.valid
        or child.score is None
        or child.primary_parent_id is None
    ):
        return
    parent = state.require_node(child.primary_parent_id)
    if not parent.valid or parent.score is None:
        return

    mode = "redesign" if "redesign" in child.generation_mode else "continue"
    reward = (parent.score - child.score) / max(abs(parent.score), _EPSILON)
    if reward >= 0:
        child.mode_weights[mode] += reward
    else:
        child.mode_weights[mode] -= state.config.negative_reward_scale * abs(reward)


def _valid_nodes(state: SearchState) -> list[SolverNode]:
    return [node for node in state.nodes if node.valid and node.score is not None]


def _positive_exp(exponent: float) -> float:
    return math.exp(max(exponent, _MIN_EXPONENT))
