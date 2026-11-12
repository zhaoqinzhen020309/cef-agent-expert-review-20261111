"""Four real external-method smoke searches, with a hard 6M total token cap."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.core.engine import _PlannerRunFileLock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "runs/comparator_pilot_v2")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    methods = ["funsearch", "eoh", "reevo", "mcts_ahd"]
    registry = {"methods": methods, "model": "deepseek-flash", "candidate_budget": args.candidates, "instances": 10,
                "seconds": args.seconds, "solver_seeds": 2, "token_budget_per_method": 1500000,
                "phase": "training_only_external_method_feasibility_pilot",
                "bridge_sha256": hashlib.sha256((ROOT / "scripts/run_external_comparator.py").read_bytes()).hexdigest(),
                "contract_helper_sha256": hashlib.sha256((ROOT / "scripts/comparator_contract.py").read_bytes()).hexdigest(),
                "worker_sha256": hashlib.sha256((ROOT / "scripts/comparator_eval_worker.py").read_bytes()).hexdigest()}
    path = root / "registry.json"
    if path.exists() and json.loads(path.read_text()) != registry:
        raise ValueError("external pilot source or registration changed")
    path.write_text(json.dumps(registry, indent=2))
    def run(method):
        target = root / method
        if (target / "summary.json").exists():
            return json.loads((target / "summary.json").read_text())
        if target.exists() and any(target.iterdir()):
            return {"method": method, "status": "INTERRUPTED_UPSTREAM_STATE_NOT_RESUMABLE"}
        command = [str(ROOT / ".baseline-venv/Scripts/python.exe"), str(ROOT / "scripts/run_external_comparator.py"),
                   "--method", method, "--run-dir", str(target), "--protocol",
                   str(ROOT / "runs/runtime_v4/protocols/guidance_v1/cvrp/fold_0/protocol.json"),
                   "--instances", "10", "--candidates", str(args.candidates), "--seconds", str(args.seconds), "--solver-seeds", "2",
                   "--token-budget", "1500000", "--wall-seconds", "14400", "--no-finalize"]
        with (root / f"{method}.out.log").open("a", encoding="utf-8") as out, (root / f"{method}.err.log").open("a", encoding="utf-8") as err:
            p = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                                 env={**os.environ, "PYTHONUTF8": "1"}, creationflags=subprocess.CREATE_NO_WINDOW)
            (root / f"{method}.process.json").write_text(json.dumps({"pid": p.pid, "started_at": time.time()}))
            code = p.wait()
        result = {"method": method, "exit_code": code}
        (root / f"{method}.exit.json").write_text(json.dumps(result))
        return result
    with _PlannerRunFileLock(root), ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(run, methods):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
