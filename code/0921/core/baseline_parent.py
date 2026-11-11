"""Expose the measured baseline as a parent without inventing a proposal.

The seed aliases an existing training evaluation; it is never a new experiment,
never accesses validation, and does not consume a new proposal slot.
"""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json

from agent_ahd.core.search_state import SolverNode

BASELINE_ID = "__baseline_control__"
SEED_ID = "seed_baseline"


def ensure_baseline_parent(state, run_dir, *, seeds, instance_ids):
    records = [r for r in state.evaluations if r.candidate_id == BASELINE_ID
               and r.split == "train" and r.seeds == tuple(seeds)
               and r.instance_ids == tuple(instance_ids)]
    if len(records) != 1 or not records[0].valid:
        raise RuntimeError("baseline parent requires one valid, matching training receipt")
    original = records[0]
    alias = replace(original, candidate_id=SEED_ID, evaluation_id="eval_seed_baseline_000")
    root = Path(run_dir)
    source = root / "controls/baseline/solver.py"
    code = source.read_bytes()
    digest = hashlib.sha256(code).hexdigest()
    if digest != state.baseline_sha256:
        raise RuntimeError("baseline parent source changed")
    target = root / "nodes" / SEED_ID / "solver.py"
    receipt_path = root / "baseline_parent.json"
    receipt = {"schema": "baseline_parent_alias_v1", "candidate_id": SEED_ID,
               "source_sha256": digest, "source_evaluation_id": original.evaluation_id,
               "source_evaluation_sha256": hashlib.sha256(json.dumps(asdict(original), sort_keys=True).encode()).hexdigest(),
               "evaluation_reused": True, "new_candidate_proposals": 0,
               "new_evaluation_units": 0, "origin": "supplied_baseline_not_agent_generated"}
    existing = [n for n in state.nodes if n.node_id == SEED_ID]
    aliases = [r for r in state.evaluations if r.evaluation_id == alias.evaluation_id]
    if existing:
        if len(existing) != 1 or aliases != [alias] or existing[0].round_index != -1:
            raise RuntimeError("baseline parent identity changed")
        if not target.exists() or target.read_bytes() != code:
            raise RuntimeError("baseline parent copy changed")
        if not receipt_path.exists() or json.loads(receipt_path.read_text()) != receipt:
            raise RuntimeError("baseline parent provenance changed")
        return False
    if aliases or state.current_round >= 0 or any(n.round_index >= 0 for n in state.nodes):
        raise RuntimeError("baseline parent must be registered before the first proposal")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != code:
        raise RuntimeError("baseline parent copy changed")
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise RuntimeError("baseline parent provenance changed")
    target.write_bytes(code)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    node = SolverNode(node_id=SEED_ID, round_index=-1, root_id=SEED_ID,
                      solver_relpath=f"nodes/{SEED_ID}/solver.py", generation_mode="fresh",
                      parent_selection="supplied_baseline_training_receipt",
                      scope={"kind": "full_solver", "symbols": []},
                      solver_sha256=digest, solver_file_sha256={"solver.py": digest})
    state.nodes.append(node)
    state.record_evaluation_record(alias)
    state.best_node_id = SEED_ID
    return True
