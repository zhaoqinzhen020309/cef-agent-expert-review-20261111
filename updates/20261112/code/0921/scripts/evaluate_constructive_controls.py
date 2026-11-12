"""Four declared human engineering variants; training selection only."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.problems.registry import get_problem_adapter
from agent_ahd.problems.base import EvaluationBudget
from agent_ahd.core.parallel_evaluator import evaluate_tasks
import numpy as np


def main():
    root = ROOT / "runs/constructive_training_controls_v1"
    root.mkdir(parents=True, exist_ok=True)
    source = ROOT / "problems/tsp_constructive/engineering_controls/bounded_rollout.py"
    protocol = ROOT / "data/bundles/guidance_v1/tsp_constructive/protocol.json"
    registry = {"evidence_type": "human_engineering_controls_not_agent_search", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(), "grid": [[6, 3], [8, 4], [12, 4], [16, 8]],
                "seconds": 2, "instances": 64, "seed": 66001,
                "scale_check": "one independent unlabelled synthetic N=200 input; runtime and feasibility only, no validation references"}
    reg = root / "registry.json"
    if reg.exists() and json.loads(reg.read_text()) != registry:
        raise ValueError("control registration changed")
    reg.write_text(json.dumps(registry, indent=2))
    adapter = get_problem_adapter("tsp_constructive", dataset_protocol_path=protocol)
    tasks = []
    for first, second in registry["grid"]:
        name = f"rollout_{first}_{second}"
        directory = root / name
        directory.mkdir(exist_ok=True)
        candidate = directory / "solver.py"
        candidate.write_text(source.read_text().replace("FIRST_CHOICES = 8", f"FIRST_CHOICES = {first}")
                             .replace("SECOND_CHOICES = 4", f"SECOND_CHOICES = {second}"))
        if not (directory / "train.json").exists():
            tasks.append(adapter.make_task(candidate_id=name, candidate_path=candidate, split="train",
                         budget=EvaluationBudget(n_instances=64, time_limit=2, ite_max=1000, seeds=(66001,))))
    for result in evaluate_tasks(tasks, max_workers=2):
        (root / result.candidate_id / "train.json").write_text(json.dumps(result.to_record(), indent=2))
        print(json.dumps({"control": result.candidate_id, "valid": result.valid, "train_gap": result.score,
                          "total_instance_seconds": result.runtime_sec}), flush=True)
    # Input-only scale check is kept separate from gap measurements.
    import importlib.util
    rng = np.random.RandomState(66002)
    coordinates = rng.uniform(size=(200, 2))
    distances = np.sqrt(np.sum((coordinates[:, None] - coordinates[None, :]) ** 2, axis=-1))
    scale = {}
    for first, second in registry["grid"]:
        name = f"rollout_{first}_{second}"
        spec = importlib.util.spec_from_file_location(name, root / name / "solver.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        strategy = module.SearchStrategy()
        current, unvisited, tour = 0, list(range(1, 200)), [0]
        started = time.perf_counter()
        while unvisited:
            current = strategy.select_next_node(current, 0, unvisited, distances)
            unvisited.remove(current)
            tour.append(current)
        scale[name] = {"wall_seconds": time.perf_counter() - started, "feasible": sorted(tour) == list(range(200))}
    (root / "unlabelled_scale_check.json").write_text(json.dumps(scale, indent=2))
    print(json.dumps({"scale_check": scale}), flush=True)


if __name__ == "__main__":
    main()
