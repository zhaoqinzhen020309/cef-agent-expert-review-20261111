"""Publish the training-selected engineering kernel without changing Agent baselines."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / "runs/fssp_batch_training_controls_v1"
    registration = json.loads((run / "registry.json").read_text())
    results = {name: json.loads((run / (name + ".json")).read_text()) for name in registration["sources"]}
    valid = [name for name, result in results.items() if result["valid"]]
    selected = min(valid, key=lambda name: (results[name]["score"], name))
    source = registration["sources"][selected]
    if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("training source changed")
    output = ROOT / "problems/fssp/engineering_controls/batched_ig.py"
    if output.exists():
        raise FileExistsError(output)
    shutil.copy2(source["path"], output)
    receipt = {"evidence_type": "human_engineering_control_not_agent_discovery",
               "selected_by": "mean_training_gap_only", "selection_candidates": list(results),
               "selected": selected, "source_sha256": source["sha256"], "budget": registration["budget"],
               "training_gap_percent": results[selected]["score"], "validation_status": "NOT_RUN",
               "algorithm_family": "Iterated Greedy; local implementation, not published-code reproduction",
               "reference": "https://doi.org/10.1016/j.ejor.2005.12.009",
               "whole_solver_auxiliary_space": "O(B*n*m), bounded batch B=24; input O(n*m)",
               "batch_equivalence": "same random job order and first-improvement tie handling as scalar insertion LS"}
    output.with_suffix(".provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "train_gap": results[selected]["score"]}))


if __name__ == "__main__":
    main()
