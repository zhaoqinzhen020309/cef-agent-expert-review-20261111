"""Supervise authorized registered queues; never edit protocols or mint results."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.core.engine import _PlannerRunFileLock
from scripts.run_campaign import write
from scripts.run_research_campaign import load, status, balance
from scripts.run_external_campaign import load as load_external, report as report_external


def matching_process(script):
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            argv = process.info["cmdline"] or []
            if process.info["name"].lower() == "python.exe" and any(str(arg).replace("\\", "/").endswith(script) for arg in argv) and "run" in argv:
                found.append({"pid": process.pid, "created_at": process.info["create_time"]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def launch(script, log_root):
    with (log_root / (Path(script).stem + ".out.log")).open("a", encoding="utf-8") as out, (log_root / (Path(script).stem + ".err.log")).open("a", encoding="utf-8") as err:
        process = subprocess.Popen([sys.executable, str(ROOT / script), "run"], cwd=ROOT, stdout=out, stderr=err,
                                   env={**os.environ, "PYTHONUTF8": "1"}, creationflags=subprocess.CREATE_NO_WINDOW)
    return process


def main():
    watch = ROOT / "runs/research_supervisor"
    watch.mkdir(parents=True, exist_ok=True)
    primary_root = ROOT / "runs/research_campaign_v2"
    external_root = ROOT / "runs/external_cvrp_campaign_v1"
    primary, external = load(primary_root), load_external(external_root)
    last_funds_check, external_launched = 0, False
    started = time.time()
    with _PlannerRunFileLock(watch):
        while time.time() - started < 7 * 86400:
            p = status(primary_root, primary)
            e = report_external(external_root, external)
            active = matching_process("run_research_campaign.py")
            external_active = matching_process("run_external_campaign.py")
            heartbeat = {"time": time.time(), "primary_processes": active, "external_processes": external_active,
                         "primary_completed": p["completed_searches"], "primary_total": p["total_searches"],
                         "external_completed": e["completed_searches"], "external_total": e["expected_searches"],
                         "status": "RUNNING" if active or external_active else "INSPECTING_TERMINAL_STATE"}
            if (watch / "STOP.json").exists() or (primary_root / "DRAIN_REQUEST.json").exists():
                heartbeat["status"] = "SUPERVISION_STOP_REQUESTED_ACTIVE_JOBS_UNCHANGED"
                write(watch / "heartbeat.json", heartbeat)
                return
            if not active:
                stop_path = primary_root / "resource_stop.json"
                stop = json.loads(stop_path.read_text()) if stop_path.exists() else {}
                if stop.get("status") == "ACCOUNT_BALANCE":
                    heartbeat["status"] = "WAITING_ACCOUNT_BALANCE"
                    if time.time() - last_funds_check >= 300:
                        last_funds_check = time.time()
                        try:
                            funds = balance()
                            job = next(j for j in primary["jobs"] if j["id"] == stop["next_job"])
                            if funds >= 5 + 8 * job["token_budget"] / 1000000:
                                launch("scripts/run_research_campaign.py", watch)
                                heartbeat["status"] = "RESUMED_AFTER_ACCOUNT_REPLENISHMENT"
                        except Exception as exc:
                            heartbeat["balance_error_type"] = type(exc).__name__
                elif stop.get("status"):
                    heartbeat["status"] = "RESOURCE_STOP_REQUIRES_REVIEW"
                elif (primary_root / "exit.json").exists():
                    results = json.loads((primary_root / "results.json").read_text())
                    cvrp_complete = results["results"].get("cvrp_on", {}).get("status") == "COMPLETE"
                    if cvrp_complete and not external_active and not external_launched:
                        launch("scripts/run_external_campaign.py", watch)
                        external_launched = True
                        heartbeat["status"] = "STARTED_REGISTERED_EXTERNAL_COMPARATORS"
                    elif not external_active and external_launched:
                        heartbeat["status"] = "QUEUES_RETURNED_INSPECT_RESULTS_AND_RESOURCE_STOPS"
                        write(watch / "heartbeat.json", heartbeat)
                        return
                    elif not cvrp_complete:
                        heartbeat["status"] = "PRIMARY_RETURNED_WITH_INCOMPLETE_CVRP"
                else:
                    heartbeat["status"] = "UNEXPECTED_PRIMARY_EXIT_REVIEW_REQUIRED"
            write(watch / "heartbeat.json", heartbeat)
            # Updating documentation reads saved receipts only and never calls models.
            with (watch / "report_refresh.log").open("a", encoding="utf-8") as output:
                subprocess.run([sys.executable, str(ROOT / "scripts/write_current_results.py")], cwd=ROOT,
                               stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
            time.sleep(30)
        write(watch / "heartbeat.json", {"status": "SUPERVISION_WINDOW_ENDED", "time": time.time()})


if __name__ == "__main__":
    main()
