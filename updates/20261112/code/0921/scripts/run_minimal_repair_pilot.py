"""Four training proposals with static repair checks; fixed minimal data only.

The seed is chosen strictly on registered training scores. Finalize only a
strictly better training candidate; otherwise keep the separately evaluated
engineering seed and report that no new algorithm was accepted.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_campaign import digest, summarize_run, write
from scripts.snapshot_runtime import snapshot
from scripts.prepare_minimal_protocols import write_once
from scripts.run_research_campaign import balance

CAMPAIGN = ROOT / "runs/minimal_repair_pilot_v1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def register():
    source_registry = ROOT / "runs/minimal_fssp_tiebreak_v1/registry.json"
    spec = json.loads(source_registry.read_text())
    scores = []
    for name, source in spec["sources"].items():
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("registered source changed")
        result = json.loads((source_registry.parent / (name + ".json")).read_text())
        if result["valid"]:
            scores.append((result["score"], name))
    score, name = min(scores)
    source = spec["sources"][name]
    seed = ROOT / "problems/fssp/minimal_repair_seed_v1/solver.py"
    seed.parent.mkdir(parents=True, exist_ok=True)
    data = Path(source["path"]).read_bytes()
    if seed.exists() and seed.read_bytes() != data:
        raise ValueError("seed changed")
    seed.write_bytes(data)
    parent = ROOT / "data/bundles/minimal_v1/fssp/protocol.json"
    protocol = json.loads(parent.read_text())
    directory = ROOT / "data/bundles/minimal_repair_v1/fssp"
    protocol["protocol_id"] = "fssp_minimal_repair_v1"
    protocol["data_root"] = str(directory)
    protocol["task"]["baseline_path"] = str(seed)
    protocol["metadata"].update(parent_protocol=str(parent), baseline_training_gap_percent=score,
                                baseline_source_registry=str(source_registry), validation_used_for_selection=False)
    for split in ("train", "validation"):
        manifest = json.loads(Path(protocol[split]["source_path"]).read_text())
        manifest["contract_id"] = protocol["protocol_id"]
        target = directory / (split + ".json")
        write_once(target, manifest)
        protocol[split]["source_path"] = str(target)
    write_once(directory / "protocol.json", protocol)
    runtime = ROOT / "runs/runtime_minimal_repair_v1"
    snapshot(runtime)
    runtime_protocol = runtime / "protocols/minimal_repair_v1/fssp/protocol.json"
    registration = {"schema": "minimal_repair_pilot_v1", "runtime": str(runtime),
        "snapshot_sha256": sha(runtime / "snapshot.json"), "protocol": str(runtime_protocol),
        "protocol_sha256": sha(runtime_protocol), "source": source, "source_registry_sha256": sha(source_registry),
        "training_seed_selection": name, "training_gap_percent": score, "validation_used_for_selection": False,
        "model": "deepseek-flash", "reasoning_effort": "none", "max_tokens": 16384, "candidate_budget": 4,
        "model_token_cap": 800000, "solver_seconds": 30, "solver_seeds": [71400, 71401],
        "final_seeds": [71600, 71601, 71602], "full_validation_scheduled": False,
        "finalize_rule": "new training winner must beat baseline by more than 1e-6 pp; otherwise retain separately evaluated engineering seed",
        "runner_sha256": sha(__file__)}
    registration["sha256"] = digest(registration)
    write_once(CAMPAIGN / "registry.json", registration)
    return registration


def load():
    spec = json.loads((CAMPAIGN / "registry.json").read_text())
    if spec["sha256"] != digest({k: v for k, v in spec.items() if k != "sha256"}) or sha(__file__) != spec["runner_sha256"]:
        raise ValueError("pilot registration changed")
    runtime = Path(spec["runtime"])
    if sha(runtime / "snapshot.json") != spec["snapshot_sha256"] or sha(spec["protocol"]) != spec["protocol_sha256"]:
        raise ValueError("frozen pilot identity changed")
    for p, expected in json.loads((runtime / "snapshot.json").read_text())["source_sha256"].items():
        if sha(runtime / "agent_ahd" / p) != expected:
            raise ValueError("frozen runtime changed")
    return spec


def stage(spec, phase, extra):
    base = [sys.executable, "-m", "agent_ahd.experiments.run_task_search", phase,
            "--run-dir", str(CAMPAIGN / "job")]
    prefix = CAMPAIGN / phase
    with prefix.with_suffix(".stdout.log").open("a") as out, prefix.with_suffix(".stderr.log").open("a") as err:
        process = subprocess.Popen(base + extra, cwd=spec["runtime"], stdout=out, stderr=err,
            env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}, creationflags=subprocess.CREATE_NO_WINDOW)
        write(prefix.with_suffix(".process.json"), {"pid": process.pid, "started_at": time.time()})
        code = process.wait()
    write(prefix.with_suffix(".exit.json"), {"exit_code": code, "finished_at": time.time()})
    if code:
        raise RuntimeError(f"{phase} failed with exit {code}")


def run(spec):
    from agent_ahd.core.engine import _PlannerRunFileLock
    with _PlannerRunFileLock(CAMPAIGN):
        if (CAMPAIGN / "exit.json").exists():
            return
        funds = balance()
        if funds < 5 + 8 * spec["model_token_cap"] / 1e6:
            raise RuntimeError("resource admission failed")
        try:
            if not (CAMPAIGN / "job/planner_config.json").exists():
                stage(spec, "init", ["--protocol", spec["protocol"], "--problem", "fssp", "--phase", "benchmark",
                    "--rounds", "2", "--batch-size", "2", "--instances", "3", "--seconds", "30", "--iterations", "10000000",
                    "--workers", "2", "--seed", "71400", "--solver-seeds", "2", "--final-seeds", "3", "--token-budget", "800000",
                    "--max-tokens", "16384", "--wall-seconds", "14400", "--baseline-parent", "--deterministic-ablation"])
            if summarize_run(CAMPAIGN / "job")["status"] != "COMPLETED_SEARCH":
                stage(spec, "run", ["--env-file", str(ROOT / ".env"), "--model", spec["model"], "--reasoning-effort", "none"])
            progress = summarize_run(CAMPAIGN / "job")
            if progress["status"] != "COMPLETED_SEARCH":
                raise RuntimeError("fixed proposal budget not completed")
            if progress["best_train_gap"] < progress["baseline_train_gap"] - 1e-6:
                stage(spec, "finalize", [])
                result = {"status": "VALIDATED_NEW_TRAINING_WINNER", "validation_source": str(CAMPAIGN / "job/validation_summary.json")}
            else:
                result = {"status": "NO_IMPROVING_CANDIDATE", "validation_not_repeated": True,
                    "retained_engineering_source_sha256": spec["source"]["sha256"],
                    "engineering_validation": str(ROOT / "runs/minimal_fssp_tiebreak_v1/minimal_validation.json")}
        except Exception as exc:
            result = {"status": "FAILED_EXECUTION", "error": str(exc)}
        result.update(finished_at=time.time(), search=summarize_run(CAMPAIGN / "job"))
        write(CAMPAIGN / "exit.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "run", "status"])
    args = parser.parse_args()
    if args.command == "register":
        print(json.dumps(register(), indent=2))
    elif args.command == "run":
        run(load())
    else:
        load()
        print(json.dumps(summarize_run(CAMPAIGN / "job"), indent=2))
