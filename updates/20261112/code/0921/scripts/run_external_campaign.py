"""Registered CVRP common-interface reruns of four published search methods.

The primary campaign is a prerequisite. Missing/invalid folds never disappear
from results, and solver seeds are never treated as independent search repeats.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.core.engine import _PlannerRunFileLock
from scripts.run_campaign import digest, write
from scripts.run_research_campaign import balance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "registry.json").exists():
        raise FileExistsError("external registration already exists")
    jobs = []
    for repeat in range(3):
        for method in ("funsearch", "eoh", "reevo", "mcts_ahd"):
            for fold in range(5):
                protocol = ROOT / f"runs/runtime_v4/protocols/guidance_v1/cvrp/fold_{fold}/protocol.json"
                jobs.append({"id": f"{method}_fold{fold}_repeat{repeat}", "method": method, "repeat": repeat, "fold": fold,
                             "protocol": str(protocol), "protocol_sha256": sha(protocol), "candidates": 32,
                             "instances": 10, "seconds": 10, "iterations": 1000, "solver_seeds": 2, "final_seeds": 10,
                             "seed": 64000 + 1000 * repeat + 10 * fold, "token_budget": 2000000})
    paths = ["scripts/run_external_campaign.py", "scripts/run_external_comparator.py", "scripts/comparator_eval_worker.py",
             "scripts/comparator_contract.py", "third_party/llm4ad_manifest.json", "runs/runtime_v4/snapshot.json"]
    registry = {"schema": "external_cvrp_common_interface_v1", "model": "deepseek-flash", "jobs": jobs,
                "primary_campaign": str(ROOT / "runs/research_campaign_v2"), "source_sha256": {name: sha(ROOT / name) for name in paths},
                "token_cap": 32000000, "maximum_parallel_jobs": 2, "balance_reserve_cny": 5,
                "baseline_adaptation": "entire solver as a function; identical public baseline and operating budget in prompts",
                "not_claimed": "paper-specific ACO hosts or original published table reproduction",
                "validation": "all 100 X instances through five fixed folds; 10 seeds; three independent search repeats",
                "acknowledgment": "LLM4AD, Liu et al., arXiv:2412.17287"}
    registry["sha256"] = digest(registry)
    write(root / "registry.json", registry)
    return registry


def load(root):
    registry = json.loads((root / "registry.json").read_text())
    if digest({k: v for k, v in registry.items() if k != "sha256"}) != registry["sha256"]:
        raise ValueError("external registration changed")
    for path, expected in registry["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("external runtime changed")
    for job in registry["jobs"]:
        if sha(job["protocol"]) != job["protocol_sha256"]:
            raise ValueError("external task protocol changed")
    return registry


def report(root, registry):
    statuses, total_tokens = {}, 0
    grouped = {}
    for job in registry["jobs"]:
        run = root / "jobs" / job["id"]
        summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {"status": "INCOMPLETE" if run.exists() else "NOT_RUN"}
        if (run / "progress.json").exists():
            progress = json.loads((run / "progress.json").read_text())
            total_tokens += progress["tokens"]
            summary.update(proposals=len(progress["proposals"]), tokens=progress["tokens"])
        statuses[job["id"]] = summary
        grouped.setdefault((job["method"], job["repeat"]), []).append(job)
    repeats = {}
    for key, jobs in grouped.items():
        values, families, seen, invalid = [], {}, set(), None
        for job in jobs:
            run = root / "jobs" / job["id"]
            if statuses[job["id"]]["status"] != "COMPLETED_SEARCH" or not (run / "training_selection_lock.json").exists():
                invalid = "NOT_COMPLETE"
                break
            lock = json.loads((run / "training_selection_lock.json").read_text())
            if lock["validation_used_for_selection"]:
                raise ValueError("validation contaminated selection")
            artifact = run / lock["candidate_id"]
            if sha(artifact / "solver.py") != lock["source_sha256"]:
                raise ValueError("selected source changed")
            result = json.loads((artifact / "validation.json").read_text())
            if not result["valid"]:
                invalid = "INVALID_VALIDATION"
                break
            protocol = json.loads(Path(job["protocol"]).read_text())
            ids = protocol["validation"]["instance_ids"]
            gaps = np.asarray(result["gap_matrix"], dtype=float)
            expected_seeds = list(range(job["seed"] + 200, job["seed"] + 200 + 10))
            if result["instance_ids"] != ids or gaps.shape != (10, len(ids)) or result["seeds"] != expected_seeds or not np.isfinite(gaps).all():
                raise ValueError("heldout seed/instance denominator differs from registration")
            means = gaps.mean(axis=0)
            for instance, gap in zip(ids, means, strict=True):
                if instance in seen:
                    raise ValueError("heldout instance repeated across folds")
                seen.add(instance)
                values.append(float(gap))
                family = protocol["validation"]["family_by_instance"][instance]
                families.setdefault(family, []).append(float(gap))
        if invalid is None and len(seen) != 100:
            raise ValueError("incomplete X-set denominator")
        repeats[key] = {"status": invalid or "COMPLETE", "mean_gap_percent": None if invalid else float(np.mean(values)),
                        "groups": {} if invalid else {k: {"count": len(v), "gap_percent": float(np.mean(v))} for k, v in families.items()}}
    aggregate = {}
    for method in sorted({k[0] for k in repeats}):
        rows = [repeats[(method, repeat)] for repeat in range(3)]
        complete = all(r["status"] == "COMPLETE" for r in rows)
        gaps = [r["mean_gap_percent"] for r in rows] if complete else []
        aggregate[method] = {"status": "COMPLETE" if complete else "NOT_COMPLETE", "independent_repeats": rows,
                             "mean_gap_percent": float(np.mean(gaps)) if complete else None,
                             "sample_variance": float(np.var(gaps, ddof=1)) if complete else None}
    result = {"registry_sha256": registry["sha256"], "jobs": statuses, "model_tokens": total_tokens,
              "completed_searches": sum(row["status"] == "COMPLETED_SEARCH" for row in statuses.values()),
              "expected_searches": len(statuses), "results": aggregate, "superiority": "NOT_ESTABLISHED"}
    primary = Path(registry["primary_campaign"]) / "results.json"
    if primary.exists():
        our = json.loads(primary.read_text())["results"].get("cvrp_on", {})
        if our.get("status") == "COMPLETE" and all(row["status"] == "COMPLETE" for row in aggregate.values()):
            result["superiority"] = "LOWER_REGISTERED_MEAN_ALL_FOUR" if all(our["mean_gap_percent"] < r["mean_gap_percent"] for r in aggregate.values()) else "NOT_MET"
            result["superiority_scope"] = "mean comparison only; no significance or original-paper-host claim"
    write(root / "results.json", result)
    return result


def run_one(root, job):
    run = root / "jobs" / job["id"]
    if run.exists():
        return {"job": job["id"], "status": "PRESERVED_EXISTING_RUN"}
    command = [str(ROOT / ".baseline-venv/Scripts/python.exe"), str(ROOT / "scripts/run_external_comparator.py"),
               "--method", job["method"], "--run-dir", str(run), "--protocol", job["protocol"]]
    for option in ("candidates", "instances", "seconds", "iterations", "solver_seeds", "final_seeds", "seed", "token_budget"):
        command += ["--" + option.replace("_", "-"), str(job[option])]
    logs = root / "launcher_logs"
    logs.mkdir(exist_ok=True)
    prefix = logs / job["id"]
    with prefix.with_suffix(".out.log").open("a", encoding="utf-8") as out, prefix.with_suffix(".err.log").open("a", encoding="utf-8") as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                   env={**os.environ, "PYTHONUTF8": "1"}, creationflags=subprocess.CREATE_NO_WINDOW)
        write(prefix.with_suffix(".process.json"), {"pid": process.pid, "started_at": time.time()})
        code = process.wait()
    result = {"job": job["id"], "exit_code": code, "finished_at": time.time()}
    write(prefix.with_suffix(".exit.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "run", "status"])
    parser.add_argument("--root", type=Path, default=ROOT / "runs/external_cvrp_campaign_v1")
    args = parser.parse_args()
    root = args.root.resolve()
    registry = register(root) if args.command == "register" else load(root)
    if args.command == "run":
        with _PlannerRunFileLock(root):
            primary = Path(registry["primary_campaign"]) / "results.json"
            data = json.loads(primary.read_text()) if primary.exists() else {}
            if data.get("results", {}).get("cvrp_on", {}).get("status") != "COMPLETE":
                write(root / "waiting.json", {"status": "WAITING_PRIMARY_CVRP_REPEATS", "time": time.time()})
                return
            pending = [j for j in registry["jobs"] if not (root / "jobs" / j["id"]).exists()]
            for offset in range(0, len(pending), 2):
                progress = report(root, registry)
                if (root / "DRAIN_REQUEST.json").exists():
                    return
                funds = balance()
                batch = pending[offset:offset + 2]
                reserved = sum(j["token_budget"] for j in batch)
                if progress["model_tokens"] + reserved > registry["token_cap"] or funds < reserved / 1000000 * 8 + registry["balance_reserve_cny"]:
                    write(root / "resource_stop.json", {"status": "TOKEN_OR_ACCOUNT_CAP", "reported_tokens": progress["model_tokens"], "balance_cny": funds})
                    return
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for future in as_completed([pool.submit(run_one, root, job) for job in batch]):
                        print(json.dumps(future.result()), flush=True)
                        report(root, registry)
    result = report(root, registry)
    print(json.dumps({k: result[k] for k in ("completed_searches", "expected_searches", "model_tokens", "superiority")}, indent=2))


if __name__ == "__main__":
    main()
