"""One-time retirement of campaign_v1 after already dispatched pilots finish."""
from pathlib import Path
import json
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_campaign import load_registry, report


def main():
    root = (ROOT / "runs/campaign_v1").resolve()
    process = psutil.Process(23156)
    command = process.cmdline()
    expected_script = (ROOT / "scripts/run_campaign.py").resolve()
    if not any(Path(part).resolve() == expected_script for part in command[1:] if part.endswith(".py")):
        raise RuntimeError("PID does not belong to the expected campaign orchestrator")
    if "--root" not in command or Path(command[command.index("--root") + 1]).resolve() != root:
        raise RuntimeError("campaign path mismatch")
    children = process.children()
    for child in children:
        args = child.cmdline()
        if "--run-dir" not in args or not Path(args[args.index("--run-dir") + 1]).resolve().is_relative_to(root / "jobs"):
            raise RuntimeError("unexpected active child; do not suspend")
        if not Path(args[args.index("--run-dir") + 1]).name.startswith("pilot_"):
            raise RuntimeError("benchmark already dispatched; manual reconciliation required")
    receipt = {"status": "DRAINING_PILOTS", "pid": process.pid, "process_created": process.create_time(),
               "active_child_pids": [c.pid for c in children], "started_at": time.time(),
               "reason": "CVRP feasibility gate failed; next version adds explicit resource facts and training-episode guidance",
               "registered_benchmark_results": "NOT_RUN; original registration and pilot evidence retained"}
    target = root / "retirement.json"
    target.write_text(json.dumps(receipt, indent=2))
    process.suspend()  # Children finish independently; no new jobs are dispatched.
    print(json.dumps({"status": "DRAINING_PILOTS", "children": receipt["active_child_pids"]}), flush=True)
    while any(c.is_running() for c in children):
        time.sleep(2)
    process.terminate()
    process.wait(timeout=30)
    progress = report(root, load_registry(root))
    receipt.update(status="RETIRED_AFTER_DRAIN", finished_at=time.time(), completed_searches=progress["completed_searches"])
    target.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
