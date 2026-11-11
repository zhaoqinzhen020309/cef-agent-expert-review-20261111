"""Pure, versioned training-state selection and paired diagnostic arithmetic.

These summaries provide a heuristic integration signal, never a significance
test or a replacement for full-candidate measurements. Execution stays in Engine.
"""
from copy import deepcopy
import math
import random

from .patch import digest
from .store import canonical


RESEARCH_VERSION = "common_start_v1"
TRACE_POLICY = {"max_snapshots": 6, "fractions": [0.1, 0.25, 0.5, 0.75]}


def _number(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _objective(state):
    return state.get("objective", state.get("externally_scored_objective"))


def select_states(snapshots, rows, source_hash, seed, count):
    """Freeze up to eight distinct train instances, with fixed phase quotas.

    Engine supplies row.instance_hash computed from TaskAdapter.solver_input;
    row.sha256 may hash an on-disk container and is deliberately not substituted.
    Inline instances are independently checked against that same canonical hash.
    Missing eligible late states leave empty slots, never duplicated padding.
    """
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("state count must be an integer in 1..8")
    if type(seed) is not int or not isinstance(source_hash, str) or not source_hash:
        raise ValueError("frozen source hash and integer selection seed required")
    whitelist = {}
    for row in rows:
        instance_id = row.get("id")
        if not isinstance(instance_id, str) or not instance_id or instance_id in whitelist:
            raise ValueError("unique nonempty instance IDs required")
        if row.get("split") != "train":
            raise ValueError("state selection accepts only training rows")
        if not isinstance(row.get("instance_hash"), str) or not row["instance_hash"]:
            raise ValueError("trusted solver-input instance_hash required")
        if "instance" in row:
            instance = {k: v for k, v in row["instance"].items() if k != "reference"}
            if digest(canonical(instance)) != row["instance_hash"]:
                raise ValueError("inline instance hash mismatch")
        whitelist[instance_id] = row
    pools = {"early": {}, "late": {}}
    seen = set()
    for item in snapshots:
        if item.get("source_hash") != source_hash:
            continue
        instance_id = item.get("instance_id")
        row = whitelist.get(instance_id)
        if row is None or item.get("split", "train") != "train":
            raise ValueError("snapshot outside training whitelist")
        if item.get("instance_hash") != row["instance_hash"]:
            raise ValueError("snapshot instance hash mismatch")
        snapshot_id = item.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id or snapshot_id in seen:
            raise ValueError("unique nonempty snapshot IDs required")
        seen.add(snapshot_id)
        if (not isinstance(item.get("source_candidate_id"), str) or not item["source_candidate_id"]
                or type(item.get("source_seed")) is not int
                or not _number(item.get("elapsed_s"))):
            raise ValueError("invalid snapshot provenance")
        phase = item.get("phase")
        if phase not in pools or not _number(_objective(item), positive=True):
            raise ValueError("invalid snapshot phase/objective")
        if "stagnation_s" in item and not _number(item["stagnation_s"]):
            raise ValueError("invalid stagnation duration")
        solution = item.get("solution")
        if (not isinstance(solution, list) or not solution
                or any(type(x) is not int for x in solution)
                or sorted(solution) != list(range(len(solution)))):
            raise ValueError("snapshot solution must be a complete permutation")
        if "instance" in row:
            instance = row["instance"]
            matrix = instance.get("distance_matrix", instance.get("processing_times"))
            if matrix is None or len(solution) != len(matrix):
                raise ValueError("snapshot permutation has wrong instance size")
        if digest(canonical(solution)) != item.get("solution_hash"):
            raise ValueError("snapshot solution hash mismatch")
        pools[phase].setdefault(instance_id, []).append(item)
    order = sorted(whitelist)
    random.Random(seed).shuffle(order)
    if any(row.get('training_group')=='supplement' for row in rows):
        # Interleave groups after deterministic shuffling; existing phase quotas remain.
        original=[id for id in order if whitelist[id].get('training_group')!='supplement']
        supplement=[id for id in order if whitelist[id].get('training_group')=='supplement']
        order=[items[i] for i in range(max(len(original),len(supplement)))
               for items in (original,supplement) if i<len(items)]
    available = set(pools["early"]) | set(pools["late"])
    target = min(count, len(available))
    late_quota = target // 2
    reserve_late = min(late_quota, len(pools["late"]))
    selected, used = [], set()
    for phase, quota in (("early", (target + 1) // 2), ("late", late_quota)):
        phase_selected = 0
        for instance_id in order:
            options = pools[phase].get(instance_id, [])
            if instance_id in used or not options or phase_selected >= quota:
                continue
            if (phase == "early" and instance_id in pools["late"]
                    and len(set(pools["late"]) - used - {instance_id}) < reserve_late):
                continue
            # Earliest early observation, latest late observation; no score ranking.
            options = sorted(options, key=lambda s: (s["elapsed_s"], s["snapshot_id"]))
            selected.append(deepcopy(options[0] if phase == "early" else options[-1]))
            used.add(instance_id)
            phase_selected += 1
    return selected


def cache_key(reference_hash, candidate_hash, states, seeds, seconds,
              implementation, workers, environment):
    """Bind the exact frozen manifest, execution conditions and interface version."""
    return digest(canonical(dict(research_version=RESEARCH_VERSION,
        trace_policy=TRACE_POLICY, reference_hash=reference_hash,
        candidate_hash=candidate_hash, states=states, seeds=seeds, seconds=seconds,
        implementation=implementation, workers=workers, environment=environment)))


def summarize(records, states, seeds, tolerance=1e-12):
    """Require one execution per (state, restart seed, role), including failures.

    Missing, duplicate, unexpected or invalid executions make the whole batch
    inconclusive. Per-state gains remain visible but are never averaged over only
    surviving successes to produce a batch signal. Unknown resource use stays None.
    """
    if not _number(tolerance):
        raise ValueError("finite nonnegative tolerance required")
    states, seeds, records = list(states), list(seeds), list(records)
    ids = [s["snapshot_id"] for s in states]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError("unique state IDs and restart seeds required")
    if any(type(seed) is not int for seed in seeds):
        raise ValueError("integer restart seeds required")
    starts = {s["snapshot_id"]: _objective(s) for s in states}
    if any(not _number(value, positive=True) for value in starts.values()):
        raise ValueError("positive finite state objectives required")
    expected = {(sid, seed, role) for sid in ids for seed in seeds
                for role in ("reference", "candidate")}
    indexed, errors, run_results = {}, [], []
    valid_runs = 0
    for record in records:
        result = deepcopy(record)
        key = (record.get("snapshot_id"), record.get("restart_seed"), record.get("role"))
        issues = []
        if key not in expected:
            issues.append("unexpected execution key")
        if key in indexed:
            issues.append("duplicate execution key")
        valid = (record.get("valid") is True
                 and _number(record.get("start_objective"), positive=True)
                 and _number(record.get("objective")))
        if key[0] in starts and record.get("start_objective") != starts[key[0]]:
            valid = False
            issues.append("start objective differs from frozen state")
        if not valid:
            issues.append("invalid or unknown execution outcome")
        result["gain_ratio"] = ((record["start_objective"] - record["objective"])
                                / record["start_objective"]) if valid else None
        result["summary_errors"] = issues
        run_results.append(result)
        if valid and not issues:
            valid_runs += 1
        errors.extend(f"{key}: {issue}" for issue in issues)
        if key not in indexed:
            indexed[key] = result
    missing = expected - set(indexed)
    errors.extend(f"{key}: missing execution" for key in sorted(missing))
    if not states or not seeds:
        errors.append("empty state or restart-seed batch")
    state_results = []
    for sid in ids:
        by_role = {role: [indexed.get((sid, seed, role)) for seed in seeds]
                   for role in ("reference", "candidate")}
        complete = bool(seeds) and all(result is not None and not result["summary_errors"]
            and result["gain_ratio"] is not None for runs in by_role.values() for result in runs)
        item = dict(snapshot_id=sid, valid=complete, reference_gain_ratio=None,
                    candidate_gain_ratio=None, delta=None)
        if complete:
            for role, runs in by_role.items():
                item[f"{role}_gain_ratio"] = math.fsum(r["gain_ratio"] for r in runs) / len(seeds)
            item["delta"] = item["candidate_gain_ratio"] - item["reference_gain_ratio"]
        state_results.append(item)
    deltas = [s["delta"] for s in state_results if s["delta"] is not None]
    positive = sum(delta > tolerance for delta in deltas)
    negative = sum(delta < -tolerance for delta in deltas)
    tied = len(deltas) - positive - negative
    all_valid = not errors and len(records) == len(expected) and len(deltas) == len(states)
    mean_delta = math.fsum(deltas) / len(states) if all_valid else None
    status = ("supported" if mean_delta > tolerance and positive > len(states) / 2
              else "not_supported") if all_valid else "inconclusive"
    usage, usage_unknown = {}, {}
    for field in ("elapsed", "wallclock_s", "cpu_seconds"):
        known = [record[field] for record in records if _number(record.get(field))]
        usage_unknown[field] = len(records) - len(known)
        usage[field] = math.fsum(known) if len(known) == len(records) else None
    return dict(research_version=RESEARCH_VERSION, status=status,
        metric="gain_ratio", statistical_claim=False, tolerance=tolerance,
        state_results=state_results, mean_delta=mean_delta, positive_states=positive,
        negative_states=negative, tied_states=tied, valid_runs=valid_runs,
        total_runs=len(records), expected_runs=len(expected), usage=usage,
        usage_unknown=usage_unknown, errors=errors, run_results=run_results)


def agent_report(summary, reference_id, candidate_id, research_purpose, experiment=None):
    """Create a deterministic, readable measurement report for the Planner.

    ``research_purpose`` records why the Agent requested the comparison; it does
    not redefine what the runner measured. Deltas remain ratios internally and
    are also rendered as percent of the frozen starting objective for humans.
    """
    states = summary.get("state_results", [])
    valid_states = [state for state in states if state.get("valid") and state.get("delta") is not None]
    wins = sum(state["delta"] > summary.get("tolerance", 1e-12) for state in valid_states)
    losses = sum(state["delta"] < -summary.get("tolerance", 1e-12) for state in valid_states)
    ties = len(valid_states) - wins - losses
    mean_delta = summary.get("mean_delta") if summary.get("status") != "inconclusive" else None
    mean_percent = None if mean_delta is None else 100.0 * mean_delta

    if mean_delta is None:
        direction = "inconclusive"
        outcome = (f"The comparison is incomplete or invalid: {summary.get('valid_runs', 0)} of "
                   f"{summary.get('expected_runs', 0)} expected executions are valid. No directional conclusion can be drawn.")
    elif summary.get("status") == "supported":
        direction = "candidate_favored"
        outcome = f"The candidate gained more on {wins} of {len(valid_states)} measured starting states."
    elif mean_delta < -summary.get("tolerance", 1e-12) and losses > len(valid_states) / 2:
        direction = "reference_favored"
        outcome = f"The reference gained more on {losses} of {len(valid_states)} measured starting states."
    else:
        direction = "mixed_or_tied"
        outcome = ("Results were mixed or tied; there is no directional conclusion under the configured "
                   "majority rule.")

    if mean_percent is None:
        mean_text = "unavailable"
    else:
        mean_text = f"{mean_percent:+.6g}% of the frozen starting objective"
    conditions = dict(experiment or {})
    condition_text = (f"Same frozen training starts; {conditions['states']} states; "
        f"restart seeds {conditions['restart_seeds']}; {conditions['time_limit_s']} seconds per continuation."
        if all(key in conditions for key in ('states','restart_seeds','time_limit_s'))
        else "The execution conditions are listed in the experiment fields.")
    return dict(
        research_purpose=research_purpose,
        reference_candidate_id=reference_id,
        candidate_id=candidate_id,
        comparison=(f"{candidate_id} minus {reference_id}; both complete solvers continued from the same "
                    "frozen training starting solutions."),
        conditions=condition_text,
        metric_definition=("For each starting solution, gain ratio = (starting objective - final objective) / "
            "starting objective. Delta is candidate minus reference gain ratio. Positive favors the "
            "candidate; negative favors the reference. The displayed percent is a percentage of the frozen "
            "starting objective, not a TSPLIB gap."),
        mean_candidate_minus_reference_gain_ratio=mean_delta,
        mean_candidate_minus_reference_gain_percent_of_start=mean_percent,
        candidate_better_states=wins, reference_better_states=losses, tied_states=ties,
        direction=direction,
        execution=dict(completed=summary.get("status") != "inconclusive",
            valid_runs=summary.get("valid_runs", 0), total_runs=summary.get("total_runs", 0),
            expected_runs=summary.get("expected_runs", 0), errors=list(summary.get("errors", []))),
        experiment=conditions,
        report=(f"{outcome} Mean candidate-minus-reference gain difference: {mean_text}. "
                f"Per starting state: candidate better {wins}, reference better {losses}, tied {ties}."),
        limits=("This is a small paired diagnostic on training starting solutions, comparing complete solver "
            "continuations. It does not measure TSPLIB gap, isolate individual edits, establish causality, or "
            "provide a statistical significance claim."),
    )
