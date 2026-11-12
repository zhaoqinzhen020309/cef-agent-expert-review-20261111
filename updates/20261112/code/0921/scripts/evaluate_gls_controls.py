"""Training-only engineering control comparison, separate from Agent results."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.problems.registry import get_problem_adapter
from agent_ahd.problems.base import EvaluationBudget
from agent_ahd.core.parallel_evaluator import evaluate_tasks


def main():
    root = ROOT / "runs/gls_training_controls_v1"
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / "registry.json"
    protocol = ROOT / "data/bundles/v2/tsp_gls/protocol.json"
    sources = {"optimized_gls": ROOT / "problems/tsp_gls/optimized/solver.py",
               "historical_ils_optimized": ROOT / "problems/tsp_gls/engineering_controls/historical_ils_optimized.py"}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    registry = {"evidence_type": "training_only_human_engineering_controls_not_agent_search",
                "protocol_sha256": sha(protocol), "seconds": 10, "seeds": [61001, 61002], "instances": 64,
                "sources": {k: {"path": str(p), "sha256": sha(p)} for k, p in sources.items()},
                "runtime_sha256": {s: sha(ROOT / s) for s in ["core/eoh_timeout.py", "evaluators/manifest_evaluator.py", "core/parallel_evaluator.py"]}}
    if registry_path.exists() and json.loads(registry_path.read_text()) != registry:
        raise ValueError("control experiment changed")
    registry_path.write_text(json.dumps(registry, indent=2))
    adapter = get_problem_adapter("tsp_gls", dataset_protocol_path=protocol)
    tasks = [adapter.make_task(candidate_id=name, candidate_path=path, split="train",
                              budget=EvaluationBudget(n_instances=64, time_limit=10, ite_max=1000, seeds=(61001, 61002)))
             for name, path in sources.items() if not (root / f"{name}.json").exists()]
    for result in evaluate_tasks(tasks, max_workers=2):
        (root / f"{result.candidate_id}.json").write_text(json.dumps(result.to_record(), indent=2))
        print(json.dumps({"control": result.candidate_id, "valid": result.valid, "train_gap": result.score}), flush=True)
    for name, path in sources.items():
        if sha(path) != registry["sources"][name]["sha256"]:
            raise ValueError("control source changed during evaluation")


if __name__ == "__main__":
    main()
