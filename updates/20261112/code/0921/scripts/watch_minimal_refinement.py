"""Read-only supervision: refresh minimal reports, record failures, never retry APIs."""
from pathlib import Path
import json
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_campaign import write


def main():
    root = ROOT / "runs/minimal_refinement_v1"
    began = time.time()
    while time.time() - began < 8 * 3600:
        processes = []
        for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                args = process.info["cmdline"] or []
                if (process.info["name"] or "").lower() == "python.exe" and "run" in args and any(
                    str(a).replace("\\", "/").endswith("scripts/run_minimal_refinement.py") for a in args):
                    processes.append({"pid": process.pid, "created_at": process.info["create_time"]})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        terminal_path = root / "exit.json"
        terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
        state = "ACTIVE" if processes else "EXITED_WITHOUT_RECEIPT"
        if terminal:
            state = terminal["status"]
        with (root / "report_refresh.log").open("a", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/report_minimal_results.py")], cwd=ROOT,
                                    stdout=log, stderr=log, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
        failures = []
        for path in (root / "receipts").glob("*.json"):
            record = json.loads(path.read_text())
            if record["status"] == "FAILED_EXECUTION":
                failures.append({"job": path.stem, "error": record.get("error")})
        if failures:
            state = "EXECUTION_FAILURE_REQUIRES_REVIEW"
        write(root / "heartbeat.json", {"time": time.time(), "state": state, "processes": processes,
              "terminal_receipt": terminal, "execution_failures": failures, "report_exit_code": result.returncode,
              "full_validation_scheduled": False, "automatic_model_retry": False})
        if terminal or not processes or (root / "WATCH_STOP.json").exists():
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
