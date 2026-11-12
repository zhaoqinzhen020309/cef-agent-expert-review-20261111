"""Evaluate external-method code with the same frozen worker as our method."""
from pathlib import Path
import json
import sys


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    sys.path.insert(0, request["runtime"])
    from agent_ahd.problems.registry import get_problem_adapter
    from agent_ahd.problems.base import EvaluationBudget
    from agent_ahd.core.parallel_evaluator import evaluate_tasks
    adapter = get_problem_adapter(request["problem"], dataset_protocol_path=Path(request["protocol"]))
    task = adapter.make_task(candidate_id=request["candidate_id"], candidate_path=Path(request["candidate_path"]),
                             budget=EvaluationBudget(**request["budget"]), split=request["split"])
    result = evaluate_tasks([task], max_workers=1)[0].to_record()
    Path(request["output"]).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
