"""Prepare the agreed training-only FSSP inputs; never evaluate a solver."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from agent_ahd.problems.base import EntryContract


DATA_SEED = 20261024
METRIC = {"name": "makespan", "unit": "processing_time", "aggregation": "mean", "direction": "minimize"}
CONTRACT_ID = "fssp_random64_n50_U01_raw_makespan_v1"


def prepare_training_protocol(output, *, solver_path):
    output, solver_path = Path(output).resolve(), Path(solver_path).resolve()
    if not solver_path.is_file():
        raise FileNotFoundError(solver_path)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    rng = np.random.Generator(np.random.PCG64(DATA_SEED))
    rows = []
    for index in range(64):
        machines = int(rng.integers(2, 21))
        matrix = rng.uniform(0.0, 1.0, size=(50, machines)).astype("<f8", copy=False)
        source = inputs / f"train_{index:04d}.npz"
        np.savez(source, processing_times=matrix)
        rows.append({"id": f"train_{index:04d}", "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "family": f"uniform_U01_n50_m{machines}", "size": 50,
            "source": "prepare_fssp_search_v1:PCG64:20261024"})
    entry = asdict(EntryContract("SearchStrategy", "solve",
        ("self", "processing_times", "deadline", "iteration_limit", "seed"),
        python_random_seed_policy="reset_per_instance"))
    manifest_path = output / "train.json"
    manifest = {"schema": "task_dataset_v1", "kind": "fssp", "split": "train",
        "contract_id": CONTRACT_ID, "metric": METRIC, "entry_contract": entry, "instances": rows}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    protocol = {"protocol_id": CONTRACT_ID, "data_root": str(output), "training_only": True,
        "metric": METRIC,
        "task": {"name": "fssp", "baseline_path": str(solver_path), "entry_contract": entry,
            "evaluator_module": "agent_ahd.evaluators.manifest_evaluator",
            "description": "Minimize arithmetic mean raw makespan on the fixed training instances. "
                "Return a permutation of all jobs for the permutation flow-shop scheduling problem. "
                "Processing times are nonnegative float64 values; every job visits all machines in order. "
                "A complete initial solver is supplied and the complete program may be modified. "
                "The objective is not reference-normalized.",
            "operating_spec": {"jobs": 50, "machines_min": 2, "machines_max": 20,
                "processing_times": "uniform [0,1)", "instance_count": 64,
                "solver_time_limit_seconds": 60, "outer_iteration_limit": 1000}},
        "train": {"source_path": str(manifest_path), "instance_ids": [row["id"] for row in rows]},
        "validation": {"source_path": None, "instance_ids": []},
        "blind": {"commitment": "not_registered:no_final_test_in_this_search"},
        "metadata": {"data_seed": DATA_SEED, "generator": "PCG64", "dtype": "float64",
            "machine_sampling": "discrete uniform integers 2 through 20 inclusive; implementation choice",
            "paper_source": "https://arxiv.org/html/2401.02051v3",
            "alignment_limits": ["Follows paper body U(0,1), not the integer-valued public example data.",
                "Whole-solver 60s includes NEH and cold JIT; upstream starts its GLS timer after NEH.",
                "Python random reset per instance; no added NumPy reset, so NumPy randomness remains stochastic.",
                "Final testing is deferred; this run establishes training search outcomes only."]}}
    path = output / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    args = parser.parse_args()
    path = prepare_training_protocol(args.output, solver_path=args.solver)
    print(json.dumps({"status": "prepared_not_evaluated", "protocol": str(path)}), flush=True)


if __name__ == "__main__":
    main()
