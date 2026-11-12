"""Freeze the engineering winner using training receipts, then validate once."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.problems.registry import get_problem_adapter
from agent_ahd.problems.base import EvaluationBudget
from agent_ahd.core.parallel_evaluator import evaluate_tasks


def main():
    root = ROOT / "runs/constructive_training_controls_v1"
    registry = json.loads((root / "registry.json").read_text())
    records = [json.loads((root / f"rollout_{a}_{b}/train.json").read_text()) for a, b in registry["grid"]]
    best = min((r for r in records if r["valid"]), key=lambda r: (r["score"], r["candidate_id"]))
    source = root / best["candidate_id"] / "solver.py"
    lock = {"candidate_id": best["candidate_id"], "train_gap": best["score"],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "validation_used_for_selection": False,
            "selection": "all four preregistered training receipts; lowest mean then name",
            "evidence_type": "human_engineering_control_not_agent_search_or_AHD_replication"}
    lock_path = root / "training_selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("selection changed")
    lock_path.write_text(json.dumps(lock, indent=2))
    protocol = ROOT / "data/bundles/guidance_v1/tsp_constructive/protocol.json"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != registry["protocol_sha256"]:
        raise ValueError("control protocol changed")
    adapter = get_problem_adapter("tsp_constructive", dataset_protocol_path=protocol)
    output = root / "heldout_selected.json"
    if not output.exists():
        task = adapter.make_task(candidate_id=best["candidate_id"], candidate_path=source, split="validation",
                 budget=EvaluationBudget(n_instances=192, time_limit=2, ite_max=1000, seeds=(66201,)))
        result = evaluate_tasks([task], max_workers=1)[0].to_record()
        output.write_text(json.dumps(result, indent=2))
    result = json.loads(output.read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != lock["source_sha256"]:
        raise ValueError("selected source changed")
    groups = {}
    if result["valid"]:
        by_case = np.asarray(result["gap_matrix"]).mean(axis=0)
        for identifier, gap in zip(result["instance_ids"], by_case, strict=True):
            family = adapter.validation_manifest.family_by_instance[identifier]
            groups.setdefault(family, []).append(float(gap))
    summary = {**lock, "valid": result["valid"], "validation_gap_percent": result["score"],
               "groups": {name: {"count": len(gaps), "mean_gap_percent": float(np.mean(gaps))} for name, gaps in groups.items()},
               "independent_agent_search_repetitions": 0,
               "target_number_met_by_engineering_control": bool(result["valid"] and result["score"] <= 13)}
    (root / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
