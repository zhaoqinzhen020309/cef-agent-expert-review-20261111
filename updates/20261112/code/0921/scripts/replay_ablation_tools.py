"""Replay exact source operations on copied pilot artifacts, never rewriting results."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.agents.candidate_agent import apply_code_implementation_response
from agent_ahd.core.code_checks import check_ablation_reversion
from agent_ahd.core.deterministic_tools import exact_ablation_response
from agent_ahd.core.research_records import PlannerDecision


def write_bundle(root, bundle):
    root.mkdir(parents=True, exist_ok=False)
    for name, source in bundle.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")


def main():
    run = ROOT / "runs/guidance_pilot_v1/tsp_gls_on"
    destination = ROOT / "analysis_outputs/current_audit/exact_ablation_replay_v2"
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError("replay destination must be empty")
    state = json.loads((run / "search_state.json").read_text(encoding="utf-8"))
    records = []
    for candidate_id in ("node_0009", "node_0011"):
        decision_path = run / "nodes" / candidate_id / "planner_decision.json"
        decision = PlannerDecision.from_dict(json.loads(decision_path.read_text(encoding="utf-8")))
        task_path = run / "rounds/round_003/agent_tasks" / (candidate_id + ".md")
        task = task_path.read_text(encoding="utf-8")
        data = json.loads(task)
        auxiliary = data["experiment"]["auxiliary_mechanism"]
        parent_bundle = auxiliary["mechanism_off_source"]
        mechanism_bundle = data["source_bundle"]
        parent_id = auxiliary["evidence"]["parent_id"]
        nodes = state["nodes"]
        if isinstance(nodes, list):
            nodes = {row["node_id"]: row for row in nodes}
        parent_source = (run / "controls/baseline/solver.py" if parent_id == "__baseline_control__"
                         else run / nodes[parent_id]["solver_relpath"])
        if parent_source.read_text(encoding="utf-8") != parent_bundle["solver.py"]:
            raise ValueError("registered parent source disagrees with immutable evaluated version")
        workspace = destination / candidate_id
        parent, mechanism, restored = [workspace / label for label in ("parent", "mechanism", "restored")]
        write_bundle(parent, parent_bundle)
        write_bundle(mechanism, mechanism_bundle)
        write_bundle(restored, mechanism_bundle)
        response = exact_ablation_response(task, decision, mechanism)
        apply_code_implementation_response(restored, response, decision=decision)
        check = check_ablation_reversion(parent, mechanism, restored, decision.scope)
        if not check.valid:
            raise ValueError(check.error_message)
        records.append({"candidate_id": candidate_id, "registered_parent_id": parent_id,
                        "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
                        "exact_reversion_valid": check.valid, "model_calls": 0,
                        "usage": response.usage})
    (destination / "receipt.json").write_text(json.dumps({
        "evidence_type": "copied_source_operation_replay_only",
        "original_pilot_results_unchanged": True,
        "not_a_new_solver_evaluation": True, "cases": records}, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(records), "exact_reversion_passed": True, "model_calls": 0}))


if __name__ == "__main__":
    main()
