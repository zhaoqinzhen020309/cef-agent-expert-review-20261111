"""Preregister, resume and report a bounded multi-domain research campaign.

Pilot gates never turn a failed/unfinished search into scientific evidence.
Each subprocess uses an immutable runtime snapshot and the existing call ledger.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.core.engine import _PlannerRunFileLock


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def summarize_run(root):
    state_path = root / "search_state.json"
    if not state_path.exists():
        return {"status": "NOT_RUN"}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = json.loads((root / "planner_config.json").read_text(encoding="utf-8"))["engine"]
    proposals = [n for n in state["nodes"] if n["round_index"] >= 0]
    ids = {n["node_id"] for n in proposals}
    scores = {r["candidate_id"]: float(np.mean(r["gap_matrix"])) for r in state["evaluations"]
              if r["valid"] and r["split"] == "train" and r["seeds"] == config["base_seeds"]}
    complete = state["current_round"] + 1 == config["total_rounds"] and len(proposals) == config["candidate_budget"] and not state.get("stopped_reason")
    terminals = []
    ledger = root / "attempt_ledger.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if row["event_type"] == "attempt_terminal":
                terminals.append(row["payload"])
    result = {"status": "COMPLETED_SEARCH" if complete else "INCOMPLETE", "round": state["current_round"],
              "proposal_slots": len(proposals), "valid_candidates": len(ids & scores.keys()),
              "candidate_budget": config["candidate_budget"], "best_train_gap": min(scores.values()) if scores else None,
              "baseline_train_gap": scores.get("__baseline_control__"), "model_attempts": len(terminals),
              "model_tokens": sum(r.get("model_tokens", 0) for r in terminals), "stop_reason": state.get("stopped_reason")}
    final = root / "validation_summary.json"
    if final.exists():
        result["validation"] = json.loads(final.read_text())
    return result


def register(root, runtime, *, model="deepseek-v4-pro", token_cap=32000000):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "registry.json").exists():
        raise FileExistsError("campaign registry is immutable")
    bundles = []
    for kind in ("tsp_constructive", "tsp_gls", "cvrp", "fssp"):
        count = 5 if kind in {"cvrp", "fssp"} else 1
        for fold in range(count):
            relative = Path(kind) / (f"fold_{fold}/protocol.json" if count == 5 else "protocol.json")
            path = runtime / "protocols/v2" / relative
            if not path.exists() and kind == "tsp_constructive":
                path = runtime / "protocols/tsp_constructive/protocol.json"
            if not path.exists():
                raise FileNotFoundError(f"task protocol not ready: {path}")
            protocol = json.loads(path.read_text())
            bundles.append({"kind": kind, "fold": fold, "protocol": str(path), "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "train_count": len(protocol["train"]["instance_ids"])})
    jobs = []
    for bundle in bundles:
        kind, fold = bundle["kind"], bundle["fold"]
        phases = ["pilot", "benchmark"] if fold == 0 else ["benchmark"]
        for phase in phases:
            repeats = 1 if phase == "pilot" else 3
            for repeat in range(repeats):
                for framework in ("on", "off"):
                    name = f"{phase}_{kind}_fold{fold}_repeat{repeat}_{framework}"
                    jobs.append({**bundle, "id": name, "phase": phase, "framework": framework, "repeat": repeat,
                                 "seed": 51000 + repeat * 1000 + fold * 10,
                                 "rounds": 2 if phase == "pilot" else 25, "batch_size": 2 if phase == "pilot" else 4,
                                 "instances": min(bundle["train_count"], {"fssp": 11, "cvrp": 10}.get(kind, 8)) if phase == "pilot" else bundle["train_count"],
                                 "seconds": 2 if phase == "pilot" or kind == "tsp_constructive" else 10,
                                 "iterations": 1000, "solver_seeds": 1 if kind == "tsp_constructive" else 2,
                                 "final_seeds": 1 if kind == "tsp_constructive" else 10,
                                 "token_budget": 1000000 if phase == "pilot" else 6000000})
    registry = {"schema": "multi_domain_campaign_v1", "runtime": str(runtime), "model": model,
                "jobs": jobs, "pilot_gate": "ON and OFF each complete the fixed budget with >=75% valid proposals",
                "gate_scope": "execution_feasibility_only; negative method effects never exclude a domain", "max_parallel_jobs": 2,
                "benchmark_comparison": "same_model_same_inputs_same_solver_budget; literature not automatically comparable",
                "full_campaign_model_token_cap": token_cap,
                "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                                "cpu_count": os.cpu_count(), "numpy": np.__version__, "BLAS_threads_per_process": 1},
                "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "requested_targets": {"tsp_gls_percent": .2, "tsp_constructive_percent": 13.0, "fssp_percent_provisional": .14}}
    registry["sha256"] = digest(registry)
    write(root / "registry.json", registry)
    return registry


def load_registry(root):
    registry = json.loads((root / "registry.json").read_text())
    if registry["sha256"] != digest({k: v for k, v in registry.items() if k != "sha256"}):
        raise ValueError("campaign registry changed")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != registry["orchestrator_sha256"]:
        raise ValueError("registered orchestrator changed; restore it or register a new campaign")
    snapshot = json.loads((Path(registry["runtime"]) / "snapshot.json").read_text())
    for relative, expected in snapshot["source_sha256"].items():
        if hashlib.sha256((Path(registry["runtime"]) / "agent_ahd" / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen runtime source changed")
    for job in registry["jobs"]:
        if hashlib.sha256(Path(job["protocol"]).read_bytes()).hexdigest() != job["protocol_sha256"]:
            raise ValueError("task protocol changed")
    return registry


def stage(registry, job, path, command, env_file):
    argv = [sys.executable, "-m", "agent_ahd.experiments.run_task_search", command, "--run-dir", str(path)]
    if command == "init":
        argv += ["--protocol", job["protocol"], "--problem", job["kind"], "--phase", job["phase"],
                 "--framework", job["framework"], "--workers", "2", "--max-tokens", "32768", "--wall-seconds", "86400"]
        for option in ("rounds", "batch_size", "instances", "seconds", "iterations", "seed", "solver_seeds", "final_seeds", "token_budget"):
            argv.extend(["--" + option.replace("_", "-"), str(job[option])])
    if command == "run":
        argv += ["--env-file", str(env_file), "--model", registry["model"]]
    logs = path.parent / "launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = logs / f"{job['id']}_{command}"
    with prefix.with_suffix(".stdout.log").open("a", encoding="utf-8") as out, prefix.with_suffix(".stderr.log").open("a", encoding="utf-8") as err:
        process = subprocess.Popen(argv, cwd=registry["runtime"], stdout=out, stderr=err,
                                   env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1",
                                        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        write(prefix.with_suffix(".process.json"), {"pid": process.pid, "started_at": time.time(), "command": command})
        code = process.wait()
    write(prefix.with_suffix(".exit.json"), {"exit_code": code, "finished_at": time.time()})
    if code != 0:
        raise RuntimeError(f"{command} exited {code}; inspect {prefix}.stderr.log")


def run_job(registry, job, root, env_file):
    path = root / "jobs" / job["id"]
    try:
        if not (path / "planner_config.json").exists():
            stage(registry, job, path, "init", env_file)
        result = summarize_run(path)
        if result["status"] != "COMPLETED_SEARCH":
            stage(registry, job, path, "run", env_file)
        result = summarize_run(path)
        if job["phase"] == "benchmark" and result["status"] == "COMPLETED_SEARCH":
            stage(registry, job, path, "finalize", env_file)
            result = summarize_run(path)
    except Exception as exc:
        result = {**summarize_run(path), "execution_error": f"{type(exc).__name__}: {exc}"}
    write(root / "receipts" / (job["id"] + ".json"), result)
    return job["id"], result


def report(root, registry):
    jobs = {job["id"]: summarize_run(root / "jobs" / job["id"]) for job in registry["jobs"]}
    gates = {}
    for kind in ("tsp_gls", "tsp_constructive", "cvrp", "fssp"):
        on = jobs[f"pilot_{kind}_fold0_repeat0_on"]
        off = jobs[f"pilot_{kind}_fold0_repeat0_off"]
        ready = all(r["status"] == "COMPLETED_SEARCH" and r["valid_candidates"] / r["candidate_budget"] >= .75 for r in (on, off))
        gates[kind] = "PASS" if ready else "NOT_ELIGIBLE" if all(r["status"] == "COMPLETED_SEARCH" for r in (on, off)) else "PENDING"
    result = {"registry_sha256": registry["sha256"], "gates": gates, "jobs": jobs,
              "completed_searches": sum(r["status"] == "COMPLETED_SEARCH" for r in jobs.values()),
              "total_searches": len(jobs), "model_tokens": sum(r.get("model_tokens", 0) for r in jobs.values())}
    write(root / "progress.json", result)
    return result


def run(root, registry, env_file, pilots_only):
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    with _PlannerRunFileLock(root):
        for phase in ("pilot", "benchmark"):
            progress = report(root, registry)
            if phase == "benchmark" and pilots_only:
                break
            todo = [j for j in registry["jobs"] if j["phase"] == phase and
                    (phase == "pilot" or progress["gates"][j["kind"]] == "PASS") and
                    not (progress["jobs"][j["id"]]["status"] == "COMPLETED_SEARCH" and
                         (phase == "pilot" or "validation" in progress["jobs"][j["id"]]))]
            # Fixed batches bound the maximum outstanding resource reservations.
            for offset in range(0, len(todo), registry["max_parallel_jobs"]):
                progress = report(root, registry)
                batch = todo[offset:offset + registry["max_parallel_jobs"]]
                reservation = sum(j["token_budget"] for j in batch)
                if progress["model_tokens"] + reservation > registry["full_campaign_model_token_cap"]:
                    write(root / "budget_stop.json", {"status": "BLOCKED_RESOURCE_CAP", "used": progress["model_tokens"], "next_batch_reservation": reservation})
                    return
                with ThreadPoolExecutor(max_workers=registry["max_parallel_jobs"]) as pool:
                    futures = [pool.submit(run_job, registry, job, root, env_file) for job in batch]
                    for future in as_completed(futures):
                        identifier, result = future.result()
                        print(json.dumps({"job": identifier, **result}), flush=True)
                        report(root, registry)
    report(root, registry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "run", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--pilots-only", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--token-cap", type=int, default=32000000)
    args = parser.parse_args()
    root = args.root.resolve()
    registry = register(root, args.runtime.resolve(), model=args.model, token_cap=args.token_cap) if args.command == "register" else load_registry(root)
    if args.command == "run":
        run(root, registry, args.env_file.resolve(), args.pilots_only)
    else:
        p = report(root, registry)
        print(json.dumps({k: v for k, v in p.items() if k != "jobs"}, indent=2))


if __name__ == "__main__":
    main()
