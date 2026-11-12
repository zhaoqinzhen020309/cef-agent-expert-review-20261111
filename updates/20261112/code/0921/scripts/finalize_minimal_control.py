"""Lock a training-selected engineering control, then evaluate only its tiny manifest."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.problems.registry import get_problem_adapter
from agent_ahd.problems.base import EvaluationBudget
from agent_ahd.core.parallel_evaluator import evaluate_tasks
from scripts.run_campaign import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registration", type=Path)
    parser.add_argument("--seeds", type=int, required=True)
    args = parser.parse_args()
    registration = args.registration.resolve()
    root = registration.parent
    spec = json.loads(registration.read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(spec["protocol"]) != spec["protocol_sha256"]:
        raise ValueError("minimal protocol changed")
    candidates = []
    for name, source in spec["sources"].items():
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("control source changed")
        result = json.loads((root / (name + ".json")).read_text())
        if result["valid"]:
            candidates.append((result["score"], name))
    if not candidates:
        raise RuntimeError("no valid control")
    score, name = min(candidates)
    source = spec["sources"][name]
    adapter = get_problem_adapter(spec["problem"], dataset_protocol_path=spec["protocol"])
    budget = {**spec["budget"], "n_instances": len(adapter.validation_manifest.instance_ids),
              "seeds": list(range(71600, 71600 + args.seeds))}
    lock = {"candidate_id": name, "training_gap_percent": score, "source": source,
            "rule": "minimum_training_mean_then_name", "validation_used_for_selection": False,
            "evidence_type": "human_engineering_control_on_reused_minimal_validation_not_new_agent_discovery",
            "validation_budget": budget, "registry_sha256": sha(registration),
            "evaluator_source_sha256": {p: sha(ROOT / p) for p in ("core/eoh_timeout.py", "core/code_checks.py",
                "core/candidate_loader.py", "core/parallel_evaluator.py", "evaluators/manifest_evaluator.py")}}
    lock_path = root / "training_selection_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("selection or evaluation protocol changed")
    write(lock_path, lock)
    result_path = root / "minimal_validation.json"
    if not result_path.exists():
        frozen = root / "selected_artifact/solver.py"
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(Path(source["path"]).read_bytes())
        task = adapter.make_task(candidate_id=name, candidate_path=frozen, split="validation", budget=EvaluationBudget(**budget))
        result = evaluate_tasks([task], max_workers=1)[0]
        write(result_path, result.to_record())
    result = json.loads(result_path.read_text())
    print(json.dumps({"selected": name, "train_gap_percent": score, "valid": result["valid"], "validation_gap_percent": result["score"]}), flush=True)


if __name__ == "__main__":
    main()
