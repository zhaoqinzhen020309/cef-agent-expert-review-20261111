from pathlib import Path
import json
import pickle
import sys
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.evaluators.manifest_evaluator import _load_inputs, read_manifest


def main():
    path = ROOT / "data/bundles/v2/cvrp/fold_0/train.json"
    rows = read_manifest(path)["instances"]
    tracemalloc.start()
    manifest, inputs, selected = _load_inputs(path, tuple(r["id"] for r in rows))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    result = {"instances": len(rows), "dense_distance_bytes_if_loaded_together": sum(r["size"] ** 2 * 8 for r in rows),
              "largest_distance_bytes": max(r["size"] ** 2 * 8 for r in rows),
              "lazy_preflight_tracemalloc_peak_bytes": peak, "worker_input_pickle_bytes": len(pickle.dumps(inputs)),
              "memory_bound": "O(instance metadata + seeds*instances*solution size + workers*largest_instance_array)",
              "solver_input_kind": "fresh numeric copy; file paths remain inside trusted evaluator"}
    output = ROOT / "analysis_outputs/current_audit/input_memory_benchmark.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
