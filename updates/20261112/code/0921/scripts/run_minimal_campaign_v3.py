"""Fixed minimal experiments after exposing the measured baseline as a parent.

An immutable runtime and registration protect every running experiment. Tasks
share model, baseline and solver budgets. No original full validation is queued.
"""
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

CAMPAIGN = ROOT / "runs/minimal_campaign_v3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    if (CAMPAIGN / "registry.json").exists():
        raise FileExistsError("registration already exists")
    controls = {}
    for kind, trial in (("fssp", "minimal_fssp_partial_v1"), ("cvrp", "minimal_cvrp_portfolio_v1")):
        directory = ROOT / "runs" / trial
        spec = json.loads((directory / "registry.json").read_text())
        scores = []
        for name, source in spec["sources"].items():
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("registered engineering source changed")
            result = json.loads((directory / (name + ".json")).read_text())
            if result["valid"]:
                scores.append((result["score"], name))
        value, name = min(scores)
        source = spec["sources"][name]
        controls[kind] = {"name": name, "source": source, "training_gap_percent": value,
                          "validation_used_for_selection": False, "origin": "training-selected engineering baseline"}
        seed = ROOT / "problems" / kind / "minimal_seed_v3/solver.py"
        seed.parent.mkdir(parents=True, exist_ok=True)
        data = Path(source["path"]).read_bytes()
        if seed.exists() and seed.read_bytes() != data:
            raise ValueError("seed already exists with different content")
        seed.write_bytes(data)
        parent = ROOT / "data/bundles/minimal_v1" / kind / "protocol.json"
        protocol = json.loads(parent.read_text())
        directory = ROOT / "data/bundles/minimal_v3" / kind
        protocol["protocol_id"] = kind + "_minimal_v3"
        protocol["data_root"] = str(directory)
        protocol["task"]["baseline_path"] = str(seed)
        protocol["metadata"].update(warm_start=controls[kind], parent_protocol=str(parent),
            baseline_interpretation="same training-selected source supplied to all methods")
        for split in ("train", "validation"):
            manifest = json.loads(Path(protocol[split]["source_path"]).read_text())
            manifest["contract_id"] = protocol["protocol_id"]
            target = directory / (split + ".json")
            write_once(target, manifest)
            protocol[split]["source_path"] = str(target)
        write_once(directory / "protocol.json", protocol)
    runtime = ROOT / "runs/runtime_minimal_v3"
    snapshot(runtime)
    jobs = []
    for kind, method in (("fssp", "framework"), ("cvrp", "framework"),
                         ("cvrp", "funsearch"), ("cvrp", "eoh"), ("cvrp", "reevo"), ("cvrp", "mcts_ahd")):
        protocol = runtime / "protocols/minimal_v3" / kind / "protocol.json"
        data = json.loads(protocol.read_text())
        jobs.append({"id": kind if method == "framework" else kind + "_" + method,
            "kind": kind, "method": method, "protocol": str(protocol), "protocol_sha256": sha(protocol),
            "train_count": len(data["train"]["instance_ids"]), "validation_count": len(data["validation"]["instance_ids"]),
            "candidates": 8, "seconds": 30 if kind == "fssp" else 2,
            "iterations": 10000000 if kind == "fssp" else 1000,
            "train_seeds": 2, "validation_seeds": 3 if kind == "fssp" else 10,
            "seed": 71400, "token_budget": 1000000, "baseline_parent": method == "framework"})
    registry = {"schema": "minimal_campaign_v3", "acceptance_scope": "fixed_minimal_validation_only",
        "runtime": str(runtime), "snapshot_sha256": sha(runtime / "snapshot.json"), "jobs": jobs,
        "warm_start": controls, "model": "deepseek-flash", "reasoning_effort": "none", "max_output_tokens": 16384,
        "global_model_token_cap": 6000000, "max_parallel_jobs": 2, "search_repetitions": 1,
        "original_full_validation_scheduled": False,
        "claim_boundary": "one small-budget search per method; not a paper-table reproduction or multi-run superiority claim",
        "files_sha256": {p: sha(ROOT / p) for p in ("scripts/run_minimal_campaign_v3.py",
            "scripts/run_minimal_comparator_v2.py", "scripts/minimal_comparator_contract.py", "scripts/comparator_eval_worker.py")}}
    registry["sha256"] = digest(registry)
    write_once(CAMPAIGN / "registry.json", registry)
    return registry


def load():
    spec = json.loads((CAMPAIGN / "registry.json").read_text())
    if spec["sha256"] != digest({k: v for k, v in spec.items() if k != "sha256"}):
        raise ValueError("registration changed")
    for p, expected in spec["files_sha256"].items():
        if sha(ROOT / p) != expected:
            raise ValueError("runner changed")
    runtime = Path(spec["runtime"])
    if sha(runtime / "snapshot.json") != spec["snapshot_sha256"]:
        raise ValueError("snapshot changed")
    for p, expected in json.loads((runtime / "snapshot.json").read_text())["source_sha256"].items():
        if sha(runtime / "agent_ahd" / p) != expected:
            raise ValueError("runtime changed")
    for job in spec["jobs"]:
        if sha(job["protocol"]) != job["protocol_sha256"]:
            raise ValueError("protocol changed")
    return spec


def report(spec):
    rows = {}
    for job in spec["jobs"]:
        root = CAMPAIGN / "jobs" / job["id"]
        if job["method"] == "framework":
            row = summarize_run(root)
        else:
            path = root / "summary.json"
            row = json.loads(path.read_text()) if path.exists() else {"status": "NOT_RUN"}
            if (root / "progress.json").exists():
                saved = json.loads((root / "progress.json").read_text())
                row.update(model_tokens=saved["tokens"], proposal_slots=len(saved["proposals"]))
                if not path.exists():
                    row["status"] = "INCOMPLETE"
        receipt = CAMPAIGN / "receipts" / (job["id"] + ".json")
        if receipt.exists():
            row["execution"] = json.loads(receipt.read_text())
        rows[job["id"]] = row
    result = {"observed_at": time.time(), "acceptance_scope": spec["acceptance_scope"], "jobs": rows,
              "model_tokens": sum(r.get("model_tokens", r.get("tokens", 0)) for r in rows.values()),
              "completed_searches": sum(r["status"] == "COMPLETED_SEARCH" for r in rows.values()), "total_searches": len(rows)}
    write(CAMPAIGN / "progress.json", result)
    return result


def stage(job, phase, command, cwd):
    prefix = CAMPAIGN / "logs" / (job["id"] + "_" + phase)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".stdout.log").open("a") as out, prefix.with_suffix(".stderr.log").open("a") as err:
        proc = subprocess.Popen(command, cwd=cwd, stdout=out, stderr=err,
            env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}, creationflags=subprocess.CREATE_NO_WINDOW)
        write(prefix.with_suffix(".process.json"), {"pid": proc.pid, "started_at": time.time(), "stage": phase})
        code = proc.wait()
    write(prefix.with_suffix(".exit.json"), {"exit_code": code, "finished_at": time.time()})
    if code:
        raise RuntimeError(f"{phase} exit={code}; see {prefix}.stderr.log")


def run_job(spec, job):
    root, runtime = CAMPAIGN / "jobs" / job["id"], Path(spec["runtime"])
    if (CAMPAIGN / "STOP.json").exists():
        result = {"status": "NOT_STARTED_STOP_REQUESTED", "finished_at": time.time()}
        write(CAMPAIGN / "receipts" / (job["id"] + ".json"), result)
        return result
    try:
        if job["method"] == "framework":
            base = [sys.executable, "-m", "agent_ahd.experiments.run_task_search"]
            if not (root / "planner_config.json").exists():
                stage(job, "init", base + ["init", "--run-dir", str(root), "--problem", job["kind"], "--protocol", job["protocol"],
                    "--phase", "benchmark", "--rounds", "4", "--batch-size", "2", "--instances", str(job["train_count"]),
                    "--seconds", str(job["seconds"]), "--iterations", str(job["iterations"]), "--workers", "2", "--seed", str(job["seed"]),
                    "--solver-seeds", "2", "--final-seeds", str(job["validation_seeds"]), "--token-budget", str(job["token_budget"]),
                    "--max-tokens", "16384", "--wall-seconds", "14400", "--deterministic-ablation", "--baseline-parent"], runtime)
            if summarize_run(root)["status"] != "COMPLETED_SEARCH":
                stage(job, "run", base + ["run", "--run-dir", str(root), "--env-file", str(ROOT / ".env"),
                    "--model", spec["model"], "--reasoning-effort", "none"], runtime)
            if not (root / "validation_summary.json").exists():
                stage(job, "finalize", base + ["finalize", "--run-dir", str(root)], runtime)
        else:
            if root.exists() and any(root.iterdir()):
                raise RuntimeError("existing external run; no automatic model replay")
            command = [str(ROOT / ".baseline-venv/Scripts/python.exe"), str(ROOT / "scripts/run_minimal_comparator_v2.py"),
                "--method", job["method"], "--run-dir", str(root), "--protocol", job["protocol"], "--runtime", str(runtime),
                "--candidates", "8", "--instances", str(job["train_count"]), "--seconds", str(job["seconds"]),
                "--iterations", str(job["iterations"]), "--seed", str(job["seed"]), "--solver-seeds", "2", "--final-seeds", "10",
                "--model", spec["model"], "--reasoning-effort", "none", "--max-tokens", "16384", "--token-budget", str(job["token_budget"]),
                "--api-call-budget", "48", "--wall-seconds", "14400"]
            stage(job, "external", command, ROOT)
        result = {"status": "RETURNED", "finished_at": time.time()}
    except Exception as exc:
        result = {"status": "FAILED_EXECUTION", "error": str(exc), "finished_at": time.time()}
    write(CAMPAIGN / "receipts" / (job["id"] + ".json"), result)
    return result


def run(spec):
    from agent_ahd.core.engine import _PlannerRunFileLock
    with _PlannerRunFileLock(CAMPAIGN):
        progress = report(spec)
        jobs = [j for j in spec["jobs"] if progress["jobs"][j["id"]].get("execution", {}).get("status") != "RETURNED"]
        required = sum(j["token_budget"] for j in jobs)
        funds = balance()
        if progress["model_tokens"] + required > spec["global_model_token_cap"] or funds < 5 + 8 * required / 1e6:
            write(CAMPAIGN / "exit.json", {"status": "RESOURCE_STOP", "time": time.time()})
            return
        write(CAMPAIGN / "admission.json", {"time": time.time(), "tokens_reserved": required, "available_balance_cny": funds})
        failures = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {pool.submit(run_job, spec, j): j for j in jobs}
            for future in as_completed(pending):
                job, result = pending[future], future.result()
                print(json.dumps({"job": job["id"], **result}), flush=True)
                if result["status"] != "RETURNED":
                    failures.append(job["id"])
                report(spec)
        write(CAMPAIGN / "exit.json", {"status": "FAILED_JOBS" if failures else "QUEUE_RETURNED", "jobs": failures, "time": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "run", "status"])
    args = parser.parse_args()
    if args.command == "register":
        spec = register()
        print(json.dumps({"registered": str(CAMPAIGN), "jobs": len(spec["jobs"])}))
    elif args.command == "run":
        run(load())
    else:
        print(json.dumps(report(load()), indent=2))
