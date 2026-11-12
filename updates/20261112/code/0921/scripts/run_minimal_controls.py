"""Train-only engineering controls on frozen minimal subsets, never full data."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.problems.registry import get_problem_adapter
from agent_ahd.problems.base import EvaluationBudget
from agent_ahd.core.parallel_evaluator import evaluate_tasks
from scripts.run_campaign import write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def previous_training_winner(run):
    state = json.loads((run / "search_state.json").read_text())
    config = json.loads((run / "planner_config.json").read_text())["engine"]
    rows = [r for r in state["evaluations"] if r["valid"] and r["split"] == "train" and r["seeds"] == config["base_seeds"]]
    best = min(rows, key=lambda r: sum(map(sum, r["gap_matrix"])) / sum(map(len, r["gap_matrix"])))
    if best["candidate_id"] == "__baseline_control__":
        return run / "controls/baseline/solver.py"
    node = next(n for n in state["nodes"] if n["node_id"] == best["candidate_id"])
    return run / node["solver_relpath"]


def main():
    root = ROOT / "runs/minimal_controls_v2"
    registration = root / "registry.json"
    if registration.exists():
        registry = json.loads(registration.read_text())
    else:
        registry = {"evidence_type": "human_engineering_and_previous_training_controls_not_new_agent_discovery", "tasks": {}}
        for kind in ("tsp_gls", "tsp_constructive", "cvrp", "fssp"):
            protocol_path = ROOT / "data/bundles/minimal_v1" / kind / "protocol.json"
            protocol = json.loads(protocol_path.read_text())
            sources = {"original_baseline": Path(protocol["task"]["baseline_path"])}
            if kind == "tsp_gls":
                sources["optimized_ils"] = ROOT / "problems/tsp_gls/engineering_controls/historical_ils_optimized.py"
            elif kind == "tsp_constructive":
                sources["bounded_rollout"] = ROOT / "problems/tsp_constructive/engineering_controls/bounded_rollout.py"
            elif kind == "cvrp":
                sources["previous_agent_train_winner"] = previous_training_winner(ROOT / "runs/pilot_cvrp_v2")
            else:
                sources["batched_ig"] = ROOT / "problems/fssp/engineering_controls/batched_ig.py"
                base = ROOT / "problems/fssp/engineering_controls/compiled_ig.py"
                for destroy in (4, 8):
                    target = root / "sources" / f"compiled_ig_d{destroy}.py"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(base.read_text().replace("DESTROY_SIZE = 4", f"DESTROY_SIZE = {destroy}"), encoding="utf-8")
                    sources[f"compiled_ig_d{destroy}"] = target
            budget = {"n_instances": len(protocol["train"]["instance_ids"]), "time_limit": 2 if kind == "tsp_constructive" else 10,
                      "ite_max": 100000 if kind == "fssp" else 1000,
                      "seeds": [71001] if kind == "tsp_constructive" else [71001, 71002]}
            frozen_sources = {}
            for name, path in sources.items():
                target = root / "candidates" / kind / name / "solver.py"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                frozen_sources[name] = {"path": str(target), "sha256": sha(target), "origin": str(path)}
            registry["tasks"][kind] = {"protocol": str(protocol_path), "protocol_sha256": sha(protocol_path), "budget": budget,
                                     "sources": frozen_sources}
        write(registration, registry)
    tasks = []
    for kind, entry in registry["tasks"].items():
        if sha(entry["protocol"]) != entry["protocol_sha256"]:
            raise ValueError("minimal protocol changed")
        adapter = get_problem_adapter(kind, dataset_protocol_path=entry["protocol"])
        for name, source in entry["sources"].items():
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("control source changed")
            identifier = kind + "__" + name
            if not (root / (identifier + ".json")).exists():
                tasks.append(adapter.make_task(candidate_id=identifier, candidate_path=Path(source["path"]), split="train",
                                               budget=EvaluationBudget(**entry["budget"])))
    for result in evaluate_tasks(tasks, max_workers=2):
        write(root / (result.candidate_id + ".json"), result.to_record())
        print(json.dumps({"candidate": result.candidate_id, "valid": result.valid, "train_gap": result.score,
                          "error_type": result.error_type}), flush=True)
    selections = {}
    for kind, entry in registry["tasks"].items():
        rows = [(name, json.loads((root / (kind + "__" + name + ".json")).read_text())) for name in entry["sources"]]
        valid = [(name, result) for name, result in rows if result["valid"]]
        if not valid:
            selections[kind] = {"status": "NO_VALID_CONTROL"}
            continue
        name, result = min(valid, key=lambda row: (row[1]["score"], row[0]))
        selections[kind] = {"status": "TRAINING_SELECTED", "name": name, "source": entry["sources"][name],
                            "training_gap_percent": result["score"], "validation_used_for_selection": False}
    write(root / "training_selection.json", selections)


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    main()
