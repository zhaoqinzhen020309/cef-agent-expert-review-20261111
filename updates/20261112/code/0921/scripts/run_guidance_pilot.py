"""Paired guidance ablation using identical task, model, inputs and budgets."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.core.engine import _PlannerRunFileLock
from scripts.run_campaign import summarize_run


def dump(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    root = ROOT / "runs/guidance_pilot_v1"
    runtime = ROOT / "runs/runtime_v4"
    root.mkdir(parents=True, exist_ok=True)
    jobs = [{"id": kind + "_" + guidance, "kind": kind, "guidance": guidance,
             "instances": 64 if kind == "tsp_constructive" else 16,
             "solver_seeds": 1 if kind == "tsp_constructive" else 2,
             "protocol": str(runtime / "protocols/guidance_v1" / kind / "protocol.json")}
            for kind in ("tsp_gls", "tsp_constructive") for guidance in ("on", "off")]
    registry = {"model": "deepseek-flash", "framework_mode": "on", "rounds": 4, "batch_size": 3,
                "seconds": 2, "seed": 62001, "jobs": jobs, "runtime": str(runtime),
                "scope": "engineering_pilot; guidance_on_vs_off; no formal superiority claim",
                "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "protocol_sha256": {j["protocol"]: hashlib.sha256(Path(j["protocol"]).read_bytes()).hexdigest() for j in jobs}}
    path = root / "registry.json"
    if path.exists() and json.loads(path.read_text()) != registry:
        raise ValueError("guidance pilot registration changed")
    dump(path, registry)
    snapshots = json.loads((runtime / "snapshot.json").read_text())
    for relative, expected in snapshots["source_sha256"].items():
        if hashlib.sha256((runtime / "agent_ahd" / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen source changed")

    def run(job):
        run_dir = root / job["id"]
        logs = root / "launcher_logs"
        logs.mkdir(exist_ok=True)
        for stage in ("init", "run", "finalize"):
            if stage == "init" and (run_dir / "planner_config.json").exists():
                continue
            if stage == "run" and summarize_run(run_dir)["status"] == "COMPLETED_SEARCH":
                continue
            if stage == "finalize" and (run_dir / "validation_summary.json").exists():
                continue
            command = [sys.executable, "-m", "agent_ahd.experiments.run_task_search", stage, "--run-dir", str(run_dir)]
            if stage == "init":
                command += ["--protocol", job["protocol"], "--problem", job["kind"], "--rounds", "4", "--batch-size", "3",
                            "--instances", str(job["instances"]), "--solver-seeds", str(job["solver_seeds"]),
                            "--final-seeds", str(job["solver_seeds"]), "--seconds", "2", "--iterations", "1000",
                            "--workers", "2", "--seed", "62001", "--research-guidance", job["guidance"],
                            "--token-budget", "2000000", "--wall-seconds", "21600"]
            if stage == "run":
                command += ["--model", registry["model"], "--env-file", str(ROOT / ".env")]
            prefix = logs / f"{job['id']}_{stage}"
            with prefix.with_suffix(".out.log").open("a", encoding="utf-8") as out, prefix.with_suffix(".err.log").open("a", encoding="utf-8") as err:
                process = subprocess.Popen(command, cwd=runtime, stdout=out, stderr=err,
                    env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                dump(prefix.with_suffix(".process.json"), {"pid": process.pid, "started_at": time.time()})
                code = process.wait()
            dump(prefix.with_suffix(".exit.json"), {"exit_code": code, "finished_at": time.time()})
            if code:
                return {"job": job["id"], "stage": stage, "exit_code": code, **summarize_run(run_dir)}
        return {"job": job["id"], **summarize_run(run_dir)}

    with _PlannerRunFileLock(root), ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            result = future.result()
            dump(root / (result["job"] + ".receipt.json"), result)
            dump(root / "progress.json", {j["id"]: summarize_run(root / j["id"]) for j in jobs})
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
