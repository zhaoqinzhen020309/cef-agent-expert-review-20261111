"""Paired computational benchmarks; these are not Agent-quality evidence."""
from pathlib import Path
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.problems.tsp_gls.problem import SearchStrategy as Original
from agent_ahd.problems.tsp_gls.optimized.solver import SearchStrategy as Optimized
from agent_ahd.core.archive import _cluster_labeled_rows


def measure(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def main():
    records = []
    for n in (100, 500, 1000):
        xy = np.random.default_rng(n).random((n, 2))
        d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        timings = {"original": [], "optimized": []}
        for repeat in range(3):
            outputs = {}
            for label, cls in (("original", Original), ("optimized", Optimized)) if repeat % 2 == 0 else (("optimized", Optimized), ("original", Original)):
                outputs[label], duration = measure(lambda: cls().nearest_neighbor_2end(d, 0))
                timings[label].append(duration)
            np.testing.assert_array_equal(outputs["original"], outputs["optimized"])
        row = {"component": "nearest_neighbor", "n": n, "seconds": timings,
               "speedup": statistics.median(timings["original"]) / statistics.median(timings["optimized"]),
               "outputs_equal": True}
        records.append(row)
        print(json.dumps(row), flush=True)
    for n in (30, 100):
        xy = np.random.default_rng(n).random((n, 2))
        d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        routes, timings = {}, {}
        for label, cls in (("original", Original), ("optimized", Optimized)):
            s = cls()
            route = s.nearest_neighbor_2end(d, 0).astype(int)
            cost = s.tour_cost_2end(d, route)
            nn = np.argsort(d, axis=1)[:, 1:101]
            routes[label], timings[label] = measure(lambda: s.guided_local_search(d, nn, route, cost, time.time() + 300, 10))
        np.testing.assert_array_equal(routes["original"][0], routes["optimized"][0])
        row = {"component": "gls_10_iterations", "n": n, "seconds": timings,
               "speedup": timings["original"] / timings["optimized"], "outputs_equal": True,
               "objective": routes["optimized"][1]}
        records.append(row)
        print(json.dumps(row), flush=True)
    spec = importlib.util.spec_from_file_location("reference_test", ROOT / "tests/test_runtime_contracts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = np.random.default_rng(71).normal(size=(500, 256))
    labels = [f"c{i}" for i in range(len(matrix))]
    memory = {}
    for label, fn in (("original", module.reference_cluster), ("optimized", _cluster_labeled_rows)):
        tracemalloc.start()
        fn(matrix, labels, max_clusters=10)
        memory[label] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    records.append({"component": "behavior_clustering", "shape": list(matrix.shape), "tracemalloc_peak_bytes": memory,
                    "memory_reduction_fraction": 1 - memory["optimized"] / memory["original"]})
    output = ROOT / "analysis_outputs/current_audit/optimization_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"evidence_type": "computational_benchmark_not_search_quality", "records": records}, indent=2))


if __name__ == "__main__":
    main()
