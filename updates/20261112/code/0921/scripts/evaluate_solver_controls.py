"""Task-neutral training-only evaluator for registered engineering controls."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registration", type=Path)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    root = args.registration.resolve().parent
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    expected = {registration["protocol"]: registration["protocol_sha256"],
                **{v["path"]: v["sha256"] for v in registration["sources"].values()}}
    if any(sha(path) != value for path, value in expected.items()):
        raise ValueError("registered source changed")
    adapter = get_problem_adapter(registration["problem"], dataset_protocol_path=registration["protocol"])
    tasks = [adapter.make_task(candidate_id=name, candidate_path=Path(source["path"]), split="train",
                              budget=EvaluationBudget(**registration["budget"]))
             for name, source in registration["sources"].items() if not (root / (name + ".json")).exists()]
    for result in evaluate_tasks(tasks, max_workers=2):
        (root / (result.candidate_id + ".json")).write_text(json.dumps(result.to_record(), indent=2))
        print(json.dumps({"name": result.candidate_id, "valid": result.valid, "train_gap": result.score}), flush=True)
    if any(sha(path) != value for path, value in expected.items()):
        raise ValueError("registered source changed during evaluation")


if __name__ == "__main__":
    main()
