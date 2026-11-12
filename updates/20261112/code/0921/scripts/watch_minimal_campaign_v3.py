"""Refresh status and reports only; no model calls or job restarts."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/minimal_campaign_v3"


def main():
    started = time.time()
    while time.time() - started < 8 * 3600:
        if (CAMPAIGN / "WATCH_STOP.json").exists():
            return
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/report_minimal_results.py")],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"}, creationflags=subprocess.CREATE_NO_WINDOW)
        terminal = CAMPAIGN / "exit.json"
        receipt = json.loads(terminal.read_text()) if terminal.exists() else None
        record = {"time": time.time(), "state": receipt["status"] if receipt else "MONITORING",
                  "terminal_receipt": receipt, "report_exit_code": completed.returncode,
                  "full_validation_scheduled": False, "automatic_model_retry": False}
        tmp = CAMPAIGN / "heartbeat.tmp"
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(CAMPAIGN / "heartbeat.json")
        if completed.returncode:
            (CAMPAIGN / "report_error.log").write_text(completed.stderr, encoding="utf-8")
        if receipt:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
