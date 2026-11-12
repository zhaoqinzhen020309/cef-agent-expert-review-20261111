"""Differential benchmark of the optional, historical ILS engineering control."""
from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.optimize_historical_ils import SOURCE, TARGET


def load(path):
    spec = importlib.util.spec_from_file_location("benchmark_solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SearchStrategy()


def main():
    old, new = load(SOURCE), load(TARGET)
    records = []
    for n in (50, 100, 200):
        xy = np.random.default_rng(77 + n).random((n, 2))
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        times, routes = {}, {}
        for label, solver in (("original", old), ("optimized", new)):
            started = time.perf_counter()
            routes[label] = solver.solve(distances, time.time() + 120, 10, 2024)
            times[label] = time.perf_counter() - started
        assert routes["original"] == routes["optimized"]
        records.append({"n": n, "iterations": 10, "seconds": times,
                        "speedup": times["original"] / times["optimized"], "route_equal": True})
    report = {"evidence_type": "fixed_work_engineering_control_not_agent_search_result", "records": records}
    target = ROOT / "analysis_outputs/current_audit/historical_ils_benchmark.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
