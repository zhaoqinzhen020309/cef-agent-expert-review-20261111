"""Bounded minimal-data optimization and acceptance, with no full-data jobs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from scripts.run_campaign import write, summarize_run, digest
from scripts.run_research_campaign import balance
from scripts.prepare_minimal_protocols import write_once
from scripts.snapshot_runtime import snapshot

CAMPAIGN = ROOT / "runs/minimal_campaign_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    controls = json.loads((ROOT / "runs/minimal_controls_v2/training_selection.json").read_text())
    # This upgrade uses only already recorded training scores; no FSSP validation has been run.
    refinement_root = ROOT / "runs/minimal_fssp_refinements_v1"
    refinement_registry = json.loads((refinement_root / "registry.json").read_text())
    for name, source in refinement_registry["sources"].items():
        result = json.loads((refinement_root / (name + ".json")).read_text())
        if result["valid"] and result["score"] < controls["fssp"]["training_gap_percent"]:
            controls["fssp"] = {"status": "TRAINING_SELECTED", "name": name, "source": source,
                                 "training_gap_percent": result["score"], "selection_budget_seconds": 10,
                                 "new_evaluation_budget_seconds": 30, "validation_used_for_selection": False}
    for kind, selected in controls.items():
        if selected["status"] != "TRAINING_SELECTED":
            raise ValueError(f"no valid training control: {kind}")
        origin = Path(selected["source"]["path"])
        if sha(origin) != selected["source"]["sha256"]:
            raise ValueError("training selected source changed")
        seed = ROOT / "problems" / kind / "minimal_seed_v2/solver.py"
        if seed.exists() and seed.read_bytes() != origin.read_bytes():
            raise ValueError("frozen seed changed")
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_bytes(origin.read_bytes())
        parent = ROOT / "data/bundles/minimal_v1" / kind / "protocol.json"
        protocol = json.loads(parent.read_text())
        protocol["protocol_id"] = kind + "_minimal_seeded_v2"
        directory = ROOT / "data/bundles/minimal_seeded_v2" / kind
        protocol["data_root"] = str(directory)
        protocol["task"]["baseline_path"] = str(seed)
        protocol["metadata"].update(warm_start=selected, parent_protocol=str(parent),
                                     baseline_interpretation="training-selected engineering seed, shared by every compared search method")
        for split in ("train", "validation"):
            manifest = json.loads(Path(protocol[split]["source_path"]).read_text())
            manifest["contract_id"] = protocol["protocol_id"]
            target = directory / (split + ".json")
            write_once(target, manifest)
            protocol[split]["source_path"] = str(target)
        write_once(directory / "protocol.json", protocol)
    runtime = ROOT / "runs/runtime_minimal_v2"
    snapshot(runtime)
    jobs = []
    for kind in ("tsp_gls", "fssp", "cvrp"):
        protocol = runtime / "protocols/minimal_seeded_v2" / kind / "protocol.json"
        data = json.loads(protocol.read_text())
        job = {"id": kind, "kind": kind, "method": "framework", "protocol": str(protocol),
               "protocol_sha256": sha(protocol), "train_count": len(data["train"]["instance_ids"]),
               "validation_count": len(data["validation"]["instance_ids"]), "candidates": 8 if kind == "cvrp" else 4,
               "seconds": {"fssp": 30, "tsp_gls": 5, "cvrp": 2}[kind], "iterations": 100000 if kind == "fssp" else 1000,
               "train_seeds": 1 if kind == "tsp_constructive" else 2, "validation_seeds": 10 if kind == "cvrp" else 3,
               "seed": 71400, "token_budget": 1000000}
        jobs.append(job)
    for method in ("funsearch", "eoh", "reevo", "mcts_ahd"):
        jobs.append({**jobs[2],
                     "id": "cvrp_" + method, "method": method})
    registry = {"schema": "minimal_research_campaign_v2", "acceptance_scope": "fixed_minimal_validation_only",
                "original_full_validation_scheduled": False, "runtime": str(runtime), "snapshot_sha256": sha(runtime / "snapshot.json"),
                "warm_start": controls, "jobs": jobs, "model": "deepseek-flash", "reasoning_effort": "none",
                "max_output_tokens": 16384, "global_model_token_cap": 4000000, "balance_reserve_cny": 5,
                "max_parallel_jobs": 2, "search_repetitions": 1, "new_agent_proposals_per_method": "4 GLS/FSSP, 8 CVRP every method",
                "libraries": "same NumPy/standard library and CPU njit support; compilation inside solve timed",
                "evaluation_budget_note": "solver seed repeats are not independent Agent search repeats",
                "files_sha256": {name: sha(ROOT / name) for name in ["scripts/run_minimal_campaign_v2.py", "scripts/run_minimal_comparator_v2.py",
                    "scripts/minimal_comparator_contract.py", "scripts/comparator_eval_worker.py"]}}
    registry["sha256"] = digest(registry)
    write_once(CAMPAIGN / "registry.json", registry)
    return registry


def load():
    registry = json.loads((CAMPAIGN / "registry.json").read_text())
    if registry["sha256"] != digest({k: v for k, v in registry.items() if k != "sha256"}):
        raise ValueError("minimal registration changed")
    for name, expected in registry["files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("minimal runner changed")
    runtime = Path(registry["runtime"])
    if sha(runtime / "snapshot.json") != registry["snapshot_sha256"]:
        raise ValueError("minimal snapshot manifest changed")
    for name, expected in json.loads((runtime / "snapshot.json").read_text())["source_sha256"].items():
        if sha(runtime / "agent_ahd" / name) != expected:
            raise ValueError("minimal frozen source changed")
    for job in registry["jobs"]:
        if sha(job["protocol"]) != job["protocol_sha256"]:
            raise ValueError("minimal task protocol changed")
    return registry


def report(registry):
    rows = {}
    for job in registry["jobs"]:
        run = CAMPAIGN / "jobs" / job["id"]
        if job["method"] == "framework":
            row = summarize_run(run)
        else:
            path = run / "summary.json"
            row = json.loads(path.read_text()) if path.exists() else {"status": "NOT_RUN"}
            progress = run / "progress.json"
            if progress.exists():
                saved = json.loads(progress.read_text())
                row.update(model_tokens=saved["tokens"], proposal_slots=len(saved["proposals"]))
                if not path.exists():
                    row["status"] = "INCOMPLETE"
        receipt = CAMPAIGN / "receipts" / (job["id"] + ".json")
        if receipt.exists():
            row["execution"] = json.loads(receipt.read_text())
        rows[job["id"]] = row
    output = {"observed_at": time.time(), "acceptance_scope": registry["acceptance_scope"], "jobs": rows,
              "model_tokens": sum(r.get("model_tokens", r.get("tokens", 0)) for r in rows.values()),
              "completed_searches": sum(r["status"] == "COMPLETED_SEARCH" for r in rows.values()), "total_searches": len(rows)}
    write(CAMPAIGN / "progress.json", output)
    return output


def stage(registry, job, phase, command, cwd):
    log = CAMPAIGN / "logs" / (job["id"] + "_" + phase)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.with_suffix(".stdout.log").open("a", encoding="utf-8") as out, log.with_suffix(".stderr.log").open("a", encoding="utf-8") as err:
        process = subprocess.Popen(command, cwd=cwd, stdout=out, stderr=err,
                  env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"},
                  creationflags=subprocess.CREATE_NO_WINDOW)
        write(log.with_suffix(".process.json"), {"pid": process.pid, "started_at": time.time(), "stage": phase})
        code = process.wait()
    write(log.with_suffix(".exit.json"), {"exit_code": code, "finished_at": time.time()})
    if code:
        raise RuntimeError(f"{phase} failed, exit={code}; see {log}.stderr.log")


def run_job(registry, job):
    run = CAMPAIGN / "jobs" / job["id"]
    runtime = Path(registry["runtime"])
    try:
        if job["method"] == "framework":
            base = [sys.executable, "-m", "agent_ahd.experiments.run_task_search"]
            if not (run / "planner_config.json").exists():
                init = base + ["init", "--run-dir", str(run), "--problem", job["kind"], "--protocol", job["protocol"],
                    "--phase", "benchmark", "--rounds", str(job["candidates"] // 2), "--batch-size", "2", "--instances", str(job["train_count"]),
                    "--seconds", str(job["seconds"]), "--iterations", str(job["iterations"]), "--workers", "2",
                    "--seed", str(job["seed"]), "--solver-seeds", str(job["train_seeds"]), "--final-seeds", str(job["validation_seeds"]),
                    "--token-budget", str(job["token_budget"]), "--max-tokens", str(registry["max_output_tokens"]),
                    "--wall-seconds", "14400", "--deterministic-ablation"]
                stage(registry, job, "init", init, runtime)
            if summarize_run(run)["status"] != "COMPLETED_SEARCH":
                stage(registry, job, "run", base + ["run", "--run-dir", str(run), "--env-file", str(ROOT / ".env"),
                    "--model", registry["model"], "--reasoning-effort", registry["reasoning_effort"]], runtime)
            if not (run / "validation_summary.json").exists():
                stage(registry, job, "finalize", base + ["finalize", "--run-dir", str(run)], runtime)
        else:
            if run.exists() and any(run.iterdir()):
                raise RuntimeError("external method has existing artifacts; no automatic unsafe retry")
            command = [str(ROOT / ".baseline-venv/Scripts/python.exe"), str(ROOT / "scripts/run_minimal_comparator_v2.py"),
                "--method", job["method"], "--run-dir", str(run), "--protocol", job["protocol"], "--runtime", str(runtime),
                "--candidates", str(job["candidates"]), "--instances", str(job["train_count"]), "--seconds", str(job["seconds"]),
                "--iterations", str(job["iterations"]), "--seed", str(job["seed"]), "--solver-seeds", str(job["train_seeds"]),
                "--final-seeds", str(job["validation_seeds"]), "--model", registry["model"], "--reasoning-effort", registry["reasoning_effort"],
                "--max-tokens", str(registry["max_output_tokens"]), "--token-budget", str(job["token_budget"]),
                "--api-call-budget", "48", "--wall-seconds", "14400"]
            stage(registry, job, "external", command, ROOT)
        result = {"status": "RETURNED", "finished_at": time.time()}
    except Exception as exc:
        result = {"status": "FAILED_EXECUTION", "error": str(exc), "finished_at": time.time()}
    write(CAMPAIGN / "receipts" / (job["id"] + ".json"), result)
    return result


def run(registry):
    from agent_ahd.core.engine import _PlannerRunFileLock
    with _PlannerRunFileLock(CAMPAIGN):
        for start in range(0, len(registry["jobs"]), 2):
            progress = report(registry)
            if (CAMPAIGN / "STOP.json").exists():
                write(CAMPAIGN / "exit.json", {"status": "STOP_REQUESTED", "time": time.time()})
                return
            jobs = [j for j in registry["jobs"][start:start + 2] if progress["jobs"][j["id"]].get("execution", {}).get("status") != "RETURNED"]
            if not jobs:
                continue
            required = sum(j["token_budget"] for j in jobs)
            funds = balance()
            if progress["model_tokens"] + required > registry["global_model_token_cap"] or funds < 5 + 8 * required / 1e6:
                write(CAMPAIGN / "exit.json", {"status": "RESOURCE_STOP", "time": time.time(), "balance_cny": funds, "model_tokens": progress["model_tokens"]})
                return
            write(CAMPAIGN / "active_batch.json", {"jobs": [j["id"] for j in jobs], "time": time.time(), "balance_cny": funds})
            with ThreadPoolExecutor(max_workers=2) as executor:
                pending = {executor.submit(run_job, registry, job): job for job in jobs}
                failures = []
                for future in as_completed(pending):
                    result = future.result()
                    print(json.dumps({"job": pending[future]["id"], **result}), flush=True)
                    report(registry)
                    if result["status"] != "RETURNED":
                        failures.append(pending[future]["id"])
            if failures:
                write(CAMPAIGN / "exit.json", {"status": "FAILED_BATCH_REVIEW_REQUIRED", "jobs": failures, "time": time.time()})
                return
        write(CAMPAIGN / "exit.json", {"status": "QUEUE_RETURNED", "time": time.time()})
        report(registry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "run", "status"])
    args = parser.parse_args()
    if args.command == "register":
        registry = register()
        print(json.dumps({"registered": str(CAMPAIGN), "jobs": len(registry["jobs"])}))
    elif args.command == "run":
        run(load())
    else:
        print(json.dumps(report(load()), indent=2))
