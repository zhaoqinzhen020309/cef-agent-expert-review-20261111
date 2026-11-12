"""Read process and exit receipts separately from scientific search completion."""
from collections import Counter
from pathlib import Path
import json
import re
import time

import psutil


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _process_state(receipt):
    try:
        process = psutil.Process(receipt["pid"])
        # An unrelated process can reuse a recorded PID after a search exits.
        if abs(process.create_time() - receipt["started_at"]) > 3:
            return "EXITED"
        return "ALIVE" if process.is_running() else "EXITED"
    except psutil.NoSuchProcess:
        return "EXITED"
    except psutil.AccessDenied:
        return "UNKNOWN"


def _failure(log):
    if not log.exists():
        return {"failure_kind": "EXIT_NONZERO"}
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 65536))
        tail = handle.read().decode("utf-8", errors="replace")
    match = re.search(r"prompt exceeds max_input_bytes: actual=(\d+), limit=(\d+)", tail)
    if match:
        return {"failure_kind": "PROMPT_INPUT_LIMIT", "actual_bytes": int(match[1]),
                "limit_bytes": int(match[2])}
    if "model budget cannot reserve the call group" in tail:
        return {"failure_kind": "MODEL_BUDGET_GROUP_RESERVATION"}
    return {"failure_kind": "EXIT_NONZERO"}


def inspect_execution(root: Path, registry: dict, progress: dict) -> dict:
    """Do not alter scientific results, immutable registries, or existing receipts."""
    rows = {}
    for job in registry["jobs"]:
        job_id = job["id"]
        scientific = progress["jobs"].get(job_id, {})
        stages = []
        for stage in ("init", "run", "finalize"):
            prefix = root / "jobs/launcher_logs" / f"{job_id}_{stage}"
            process = _read(prefix.with_suffix(".process.json"))
            if process:
                stages.append((process["started_at"], stage, prefix, process))
        row = {"status": "NOT_RUN", "scientific_status": scientific.get("status", "NOT_RUN")}
        if stages:
            started, stage, prefix, process = max(stages, key=lambda item: item[0])
            terminal = _read(prefix.with_suffix(".exit.json"))
            row.update(stage=stage, pid=process["pid"], started_at=started)
            # Only the terminal receipt belonging to the latest launch is relevant.
            if terminal and terminal["finished_at"] >= started:
                row.update(exit_code=terminal["exit_code"], finished_at=terminal["finished_at"])
                if terminal["exit_code"]:
                    row.update(status="FAILED_EXECUTION", **_failure(prefix.with_suffix(".stderr.log")))
                elif scientific.get("status") == "COMPLETED_SEARCH":
                    row["status"] = "RETURNED_SEARCH_COMPLETE"
                else:
                    row["status"] = "RETURNED_INCOMPLETE"
            else:
                process_state = _process_state(process)
                row["status"] = {"ALIVE": "RUNNING", "EXITED": "EXITED_WITHOUT_RECEIPT",
                                 "UNKNOWN": "PROCESS_UNVERIFIED"}[process_state]
        elif scientific.get("status") != "NOT_RUN":
            row["status"] = "INCOMPLETE_WITHOUT_PROCESS_RECEIPT"
        for key in ("proposal_slots", "candidate_budget", "valid_candidates", "model_tokens"):
            if key in scientific:
                row[key] = scientific[key]
        rows[job_id] = row
    counts = dict(Counter(row["status"] for row in rows.values()))
    drain = _read(root / "DRAIN_REQUEST.json")
    queue_state = ("DRAIN_REQUESTED_ACTIVE_JOB_RUNNING" if counts.get("RUNNING") else "DRAIN_REQUESTED_NO_ACTIVE_JOB") if drain else "INSPECT_JOB_STATUSES"
    return {"observed_at_epoch_sec": time.time(), "queue_state": queue_state,
            "counts": counts, "drain_request": drain, "jobs": rows}
