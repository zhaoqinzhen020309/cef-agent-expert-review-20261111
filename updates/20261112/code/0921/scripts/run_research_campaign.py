"""Frozen multi-domain repetitions with balance checks and safe batch draining.

No quality-based exclusion: pilot gates concern executability only. Every planned
fold/repetition stays in reports, including failed, missing and budget-stopped jobs.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.core.engine import _PlannerRunFileLock
from agent_ahd.core.local_config import load_deepseek_env
from scripts.run_campaign import digest, summarize_run, write, run_job
from scripts.report_campaign import aggregate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balance():
    credentials = load_deepseek_env(ROOT / ".env")
    req = urllib.request.Request(credentials["base_url"] + "/user/balance",
              headers={"Authorization": "Bearer " + credentials["api_key"]})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=30) as connection:
        data = json.load(connection)
    if not data.get("is_available"):
        return 0.0
    amounts = [float(r["total_balance"]) for r in data.get("balance_infos", []) if r["currency"] == "CNY"]
    if len(amounts) != 1:
        raise ValueError("CNY balance cannot be verified")
    return amounts[0]


def register(root, runtime, cap):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "registry.json"
    if path.exists():
        raise FileExistsError("registration is immutable; resume the existing campaign")
    jobs = []
    # Engineering checks for the two new domains; no validation is inspected.
    for kind in ("cvrp", "fssp"):
        for framework in ("on", "off"):
            jobs.append({"kind": kind, "fold": 0, "repeat": 0, "framework": framework,
                         "phase": "pilot", "rounds": 2, "batch_size": 4,
                         "instances": 10 if kind == "cvrp" else 11,
                         "seconds": 2, "token_budget": 1000000})
    # Main results first, then equal-budget framework ablations; folds are fixed.
    for framework in ("on", "off"):
        for repeat in range(3):
            for kind in ("tsp_constructive", "tsp_gls", "cvrp", "fssp"):
                for fold in range(5 if kind in {"cvrp", "fssp"} else 1):
                    jobs.append({"kind": kind, "fold": fold, "repeat": repeat, "framework": framework,
                                 "phase": "benchmark", "rounds": 25 if kind == "tsp_constructive" else 8,
                                 "batch_size": 4, "instances": {"tsp_constructive": 64, "tsp_gls": 64, "cvrp": 10, "fssp": 11}[kind],
                                 "seconds": 2 if kind == "tsp_constructive" else 10,
                                 "token_budget": 6000000 if kind == "tsp_constructive" else 2000000})
    for job in jobs:
        kind = job["kind"]
        job["id"] = f"{job['phase']}_{kind}_fold{job['fold']}_repeat{job['repeat']}_{job['framework']}"
        relative = Path(kind) / (f"fold_{job['fold']}/protocol.json" if kind in {"cvrp", "fssp"} else "protocol.json")
        protocol = runtime / "protocols/guidance_v1" / relative
        job.update(protocol=str(protocol), protocol_sha256=sha(protocol), seed=64000 + 1000 * job["repeat"] + 10 * job["fold"],
                   iterations=1000, solver_seeds=1 if kind == "tsp_constructive" else 2,
                   final_seeds=1 if kind == "tsp_constructive" else 10)
    registry = {"schema": "resource_aware_research_campaign_v2", "runtime": str(runtime), "model": "deepseek-flash",
                "jobs": jobs, "max_parallel_jobs": 2, "full_campaign_model_token_cap": cap,
                "balance_reserve_cny": 5, "conservative_cny_per_million_tokens": 8,
                "pricing_source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
                "pilot_gate": "completed fixed budget and at least 75 percent valid; never gate on quality gain",
                "tsp_pilot_root": str(ROOT / "runs/guidance_pilot_v1"),
                "validation_use": "training-only selection; validation reused across protocol versions, no blind claim",
                "train_subsets": "CVRP first 10 size-stratified and FSSP first 11 group-stratified of each fold; all validation cases retained",
                "baseline_policy": "same registered original optimized baseline for every arm; human ILS control stays separate",
                "orchestrator_sha256": sha(__file__), "dependency_sha256": {p: sha(ROOT / p) for p in ["scripts/run_campaign.py", "scripts/report_campaign.py"]},
                "runtime_snapshot_sha256": sha(runtime / "snapshot.json")}
    registry["sha256"] = digest(registry)
    write(path, registry)
    return registry


def load(root):
    registry = json.loads((root / "registry.json").read_text())
    if registry["sha256"] != digest({k: v for k, v in registry.items() if k != "sha256"}):
        raise ValueError("registration changed")
    if registry["orchestrator_sha256"] != sha(__file__):
        raise ValueError("registered launcher changed")
    for name, expected in registry["dependency_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("registered launcher dependency changed")
    runtime = Path(registry["runtime"])
    if sha(runtime / "snapshot.json") != registry["runtime_snapshot_sha256"]:
        raise ValueError("runtime manifest changed")
    for name, expected in json.loads((runtime / "snapshot.json").read_text())["source_sha256"].items():
        if sha(runtime / "agent_ahd" / name) != expected:
            raise ValueError("frozen runtime changed")
    for job in registry["jobs"]:
        if sha(job["protocol"]) != job["protocol_sha256"]:
            raise ValueError("protocol changed")
    return registry


def gate(rows):
    if any(row.get("status") != "COMPLETED_SEARCH" for row in rows):
        return "PENDING"
    return "PASS" if all(row["valid_candidates"] / row["candidate_budget"] >= .75 for row in rows) else "FAILED_FEASIBILITY"


def status(root, registry):
    rows = {job["id"]: summarize_run(root / "jobs" / job["id"]) for job in registry["jobs"]}
    gates = {kind: gate([rows[f"pilot_{kind}_fold0_repeat0_{arm}"] for arm in ("on", "off")]) for kind in ("cvrp", "fssp")}
    for kind in ("tsp_gls", "tsp_constructive"):
        gates[kind] = gate([summarize_run(Path(registry["tsp_pilot_root"]) / f"{kind}_{arm}") for arm in ("on", "off")])
    result = {"registry_sha256": registry["sha256"], "gates": gates, "jobs": rows,
              "completed_searches": sum(r["status"] == "COMPLETED_SEARCH" for r in rows.values()),
              "total_searches": len(rows), "model_tokens": sum(r.get("model_tokens", 0) for r in rows.values()), "updated_at": time.time()}
    write(root / "progress.json", result)
    results = aggregate(root, registry)
    results["literature_baseline_reruns"] = "SEPARATE_REGISTERED_COMPARATOR_CAMPAIGN_REQUIRED"
    write(root / "results.json", results)
    return result


def admission(batch, progress, registry, current_balance):
    # Reserve full per-job caps before launch, never spend past current funds.
    reserve = sum(j["token_budget"] for j in batch)
    if progress["model_tokens"] + reserve > registry["full_campaign_model_token_cap"]:
        return "TOKEN_CAP"
    cost = reserve / 1000000 * registry["conservative_cny_per_million_tokens"]
    if current_balance - registry["balance_reserve_cny"] < cost:
        return "ACCOUNT_BALANCE"
    return None


def run(root, registry):
    attempted = set()
    with _PlannerRunFileLock(root):
        while True:
            progress = status(root, registry)
            if (root / "DRAIN_REQUEST.json").exists():
                write(root / "exit.json", {"status": "DRAINED_AT_BATCH_BOUNDARY", "time": time.time()})
                return
            pending = [j for j in registry["jobs"] if j["id"] not in attempted
                       and not (progress["jobs"][j["id"]]["status"] == "COMPLETED_SEARCH"
                                and (j["phase"] == "pilot" or "validation" in progress["jobs"][j["id"]]))]
            ready = [j for j in pending if j["phase"] == "pilot" or progress["gates"][j["kind"]] == "PASS"]
            # Prioritize actual TSP benchmarks as soon as their pilot gate opens.
            ready.sort(key=lambda j: (0 if j["phase"] == "benchmark" and j["kind"].startswith("tsp") else 1,
                                      registry["jobs"].index(j)))
            if not ready:
                if any(j["kind"].startswith("tsp") and progress["gates"][j["kind"]] == "PENDING" for j in pending):
                    write(root / "waiting.json", {"status": "WAITING_REGISTERED_PILOTS", "time": time.time()})
                    time.sleep(30)
                    continue
                write(root / "exit.json", {"status": "QUEUE_FINISHED_OR_GATED", "remaining": len(pending), "time": time.time()})
                return
            try:
                funds = balance()
            except Exception as exc:
                write(root / "resource_stop.json", {"status": "BALANCE_CHECK_UNAVAILABLE", "error_type": type(exc).__name__, "time": time.time()})
                return
            batch = ready[:registry["max_parallel_jobs"]]
            while len(batch) > 1 and admission(batch, progress, registry, funds):
                batch.pop()
            reason = admission(batch, progress, registry, funds)
            if reason:
                write(root / "resource_stop.json", {"status": reason, "balance_cny": funds,
                      "reported_tokens": progress["model_tokens"], "next_job": batch[0]["id"], "time": time.time()})
                return
            write(root / "active_batch.json", {"jobs": [j["id"] for j in batch], "time": time.time(), "balance_before_cny": funds})
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = [pool.submit(run_job, registry, job, root, ROOT / ".env") for job in batch]
                for future in as_completed(futures):
                    identifier, result = future.result()
                    attempted.add(identifier)
                    print(json.dumps({"job": identifier, **result}), flush=True)
                    status(root, registry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "run", "status"])
    parser.add_argument("--root", type=Path, default=ROOT / "runs/research_campaign_v2")
    parser.add_argument("--runtime", type=Path, default=ROOT / "runs/runtime_v4")
    parser.add_argument("--token-cap", type=int, default=32000000)
    args = parser.parse_args()
    root = args.root.resolve()
    registry = register(root, args.runtime.resolve(), args.token_cap) if args.command == "register" else load(root)
    if args.command == "run":
        run(root, registry)
    else:
        result = status(root, registry)
        print(json.dumps({k: v for k, v in result.items() if k != "jobs"}, indent=2))


if __name__ == "__main__":
    main()
