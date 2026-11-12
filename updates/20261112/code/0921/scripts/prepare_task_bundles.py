"""Build immutable-input task manifests from local and public benchmarks."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import json
import pickle
import re
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "data/bundles"
REVISION = "v1"
sys.path.insert(0, str(ROOT))
from agent_ahd.evaluators.manifest_evaluator import sha256_file, objective
from agent_ahd.evaluators.tsplib_protocol import load_tsplib_instance, TSPLIB29_INSTANCES, CANONICAL_OPTIMAL_COSTS
from agent_ahd.problems.base import EntryContract


ENTRIES = {
    "tsp_gls": EntryContract("SearchStrategy", "solve", ("self", "distance_matrix", "deadline", "iteration_limit", "seed")),
    "tsp_constructive": EntryContract("SearchStrategy", "select_next_node", ("self", "current_node", "destination_node", "unvisited_nodes", "distance_matrix"), "constructive_tsp"),
    "cvrp": EntryContract("SearchStrategy", "solve", ("self", "instance", "deadline", "iteration_limit", "seed"), "mapping", "routes"),
    "fssp": EntryContract("SearchStrategy", "solve", ("self", "processing_times", "deadline", "iteration_limit", "seed")),
}
DESCRIPTIONS = {
    "tsp_gls": "Minimize the cost of a symmetric TSP Hamiltonian cycle. Return a permutation of all nodes. Input is the original distance matrix. Deadline is an absolute time.time timestamp; return before it. Initialization, local search, perturbation, acceptance and restart may be redesigned. Costs and feasibility are computed externally.",
    "tsp_constructive": "Design ONLY the next-city selection policy f(current_node, destination_node, unvisited_nodes, distance_matrix)->integer node. A fixed external greedy loop starts at depot 0 and appends the selected unvisited city, then closes the cycle. No local search or replacement of the completed tour is available. Minimize total Euclidean length. Helper functions are permitted within the policy. Never modify inputs.",
    "cvrp": "Minimize symmetric CVRP total distance. Input dict has distance_matrix, demands, capacity. Depot is 0. Return list[list[int]] of nonempty customer routes, EXCLUDING depot. Visit every customer exactly once, never exceed capacity on any route. Vehicle count is unbounded (the X filename k is not a fleet constraint). Return before absolute time.time deadline. External evaluator validates feasibility and computes gap to frozen BKS.",
    "fssp": "Minimize permutation flow shop makespan. Input processing_times has shape (jobs, machines); all jobs traverse machines in the same order, without preemption. Return one permutation of job indices, shared by all machines. Minimize makespan under the standard unlimited-buffer model. Return before absolute time.time deadline. External evaluator validates and scores the permutation.",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refuse to overwrite changed data contract: {path}")
    path.write_text(text, encoding="utf-8")


def instance_record(root, identifier, arrays, reference, family, source, reference_kind="optimal"):
    path = root / "instances" / (identifier + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        np.savez(path, **arrays)
    else:
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[k], v) for k, v in arrays.items()):
                raise ValueError("existing instance content differs")
    return {"id": identifier, "path": str(path.resolve()), "sha256": sha256_file(path),
            "reference": float(reference), "reference_kind": reference_kind, "family": family, "source": source,
            "size": int(len(arrays.get("distance_matrix", arrays.get("processing_times"))))}


def bundle(kind, root, train, validation, *, baseline=None, metadata=None):
    contract = f"{kind}_standalone_{REVISION}"
    manifests = {}
    for split, rows in (("train", train), ("validation", validation)):
        manifest = {"schema": "task_dataset_v1", "kind": kind, "split": split, "contract_id": contract,
                    "entry_contract": asdict(ENTRIES[kind]), "execution": "one_fresh_process_per_candidate_batch; parent_deadline_per_instance_seed",
                    "instances": rows}
        path = root / f"{split}.json"
        write_json(path, manifest)
        manifests[split] = {"format": "task_dataset_v1", "source_path": str(path.resolve()),
                            "instance_ids": [r["id"] for r in rows], "family_by_instance": {r["id"]: r["family"] for r in rows}}
    protocol = {"protocol_id": contract, "data_root": str(root.resolve()),
                "task": {"name": kind, "description": DESCRIPTIONS[kind], "entry_contract": asdict(ENTRIES[kind]),
                         "baseline_path": str((baseline or ROOT / "problems" / kind / "problem.py").resolve()),
                         "evaluator_module": "agent_ahd.evaluators.manifest_evaluator"},
                **manifests, "blind": {"commitment": "not_registered:no_blind_claim"},
                "metadata": metadata or {}}
    write_json(root / "protocol.json", protocol)
    return root / "protocol.json"


def tsp_gls():
    root = BUNDLE_ROOT / "tsp_gls"
    source = ROOT / "problems/tsp_gls/TrainingData/TSPAEL64.pkl"
    with source.open("rb") as handle:
        data = pickle.load(handle)  # User-supplied historical data, never model output.
    for d, tour, cost in zip(data["distance_matrix"], data["optimal_tour"], data["cost"], strict=True):
        actual = objective("tsp_gls", d, list(map(int, tour)))
        if abs(actual - cost) > 1e-7:
            raise ValueError("historical reference tour does not reproduce its stored objective")
    train = [instance_record(root, f"train_{i:04d}", {"distance_matrix": d}, data["cost"][i], "random_euclidean_tsp100",
                              {"path": str(source), "sha256": sha256_file(source)}) for i, d in enumerate(data["distance_matrix"])]
    validation = []
    for name in TSPLIB29_INSTANCES:
        path = ROOT / "data/raw/tsplib" / name
        d, meta = load_tsplib_instance(path)
        validation.append(instance_record(root, path.stem, {"distance_matrix": d}, CANONICAL_OPTIMAL_COSTS[name],
                                           re.sub(r"[^a-z]", "", path.stem.lower()),
                                           {"url": json.loads(path.with_suffix(".tsp.source.json").read_text())["url"], "distance": meta}))
    return bundle("tsp_gls", root, train, validation, baseline=ROOT / "problems/tsp_gls/optimized/solver.py",
                  metadata={"validation_seen_historically": True, "target_gap_percent": 0.2,
                            "original_baseline": str(ROOT / "problems/tsp_gls/problem.py")})


def parse_cvrp(path):
    text = path.read_text()
    n = int(re.search(r"DIMENSION\s*:\s*(\d+)", text)[1])
    cap = int(re.search(r"CAPACITY\s*:\s*(\d+)", text)[1])
    if not re.search(r"EDGE_WEIGHT_TYPE\s*:\s*EUC_2D", text):
        raise ValueError("CVRP parser only supports EUC_2D")
    section = None
    coords, demands, depots = {}, {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if line.endswith("SECTION"):
            section = line
            continue
        if not line or line == "EOF":
            continue
        values = line.split()
        if section == "NODE_COORD_SECTION":
            coords[int(values[0])] = list(map(float, values[1:3]))
        elif section == "DEMAND_SECTION":
            demands[int(values[0])] = float(values[1])
        elif section == "DEPOT_SECTION" and values[0] != "-1":
            depots.append(int(values[0]))
    if set(coords) != set(range(1, n + 1)) or set(demands) != set(coords) or depots != [1]:
        raise ValueError("invalid or unsupported CVRP node/depot numbering")
    xy = np.asarray([coords[i] for i in range(1, n + 1)])
    # TSPLIB EUC_2D uses floor(d+0.5), not bankers rounding.
    distance = np.floor(np.linalg.norm(xy[:, None] - xy[None, :], axis=2) + 0.5)
    return {"distance_matrix": distance, "demands": np.asarray([demands[i] for i in range(1, n + 1)]), "capacity": np.asarray(cap)}


def cvrp():
    root = BUNDLE_ROOT / "cvrp"
    records = []
    paths = sorted((ROOT / "data/raw/cvrp").glob("X-*.vrp"), key=lambda p: int(re.search(r"-n(\d+)", p.name)[1]))
    if len(paths) != 100:
        raise ValueError(f"X collection requires 100 instances; found {len(paths)}")
    for path in paths:
        item = parse_cvrp(path)
        solution = path.with_suffix(".sol").read_text()
        reference = float(re.search(r"Cost\s*:?\s*([\d.]+)", solution, re.I)[1])
        routes = [list(map(int, line.split(":", 1)[1].split())) for line in solution.splitlines() if line.lower().startswith("route")]
        if abs(objective("cvrp", item, routes) - reference) > 1e-6:
            raise ValueError(f"BKS route does not reproduce reference: {path.name}")
        n = len(item["distance_matrix"]) - 1
        band = min(4, n // 200)
        records.append(instance_record(root, path.stem, item, reference, f"size_band_{band}",
                                       json.loads(path.with_suffix(".vrp.source.json").read_text()), "best_known"))
    for fold in range(5):
        # Round-robin by size, fixed before any candidate evaluation.
        validation = [r for i, r in enumerate(records) if i % 5 == fold]
        train = [r for i, r in enumerate(records) if i % 5 != fold]
        if REVISION != "v1":
            train = stratify(train)
        bundle("cvrp", root / f"fold_{fold}", train, validation,
               metadata={"fold": fold, "folds": 5, "split": "size_sorted_round_robin", "seeds_per_instance": 10,
                         "pilot_time_sec": 2, "formal_time_sec": 10, "literature_time_matched": False})
    return root


def fssp():
    root = BUNDLE_ROOT / "fssp"
    raw = ROOT / "data/raw/fssp"
    records = []
    with (raw / "Taillard_UB_Schedules.csv").open(newline="") as handle:
        references = list(csv.DictReader(handle))
    for row in references[:110]:
        identifier = "ta" + row["Ta"].zfill(3)
        values = list(map(int, (raw / "taillard" / (identifier + ".dat")).read_text().split()))
        n, m = values[:2]
        p = np.asarray(values[2:]).reshape(m, n).T
        route = [int(v) - 1 for v in row["Permutation"].split()]
        reference = float(row["UB"])
        if objective("fssp", p, route) != reference:
            raise ValueError(f"published FSSP schedule does not reproduce makespan: {identifier}")
        records.append(instance_record(root, identifier, {"processing_times": p}, reference, f"{n}_{m}",
                                       {"url": "https://zenodo.org/records/4542886", "csv_sha256": sha256_file(raw / "Taillard_UB_Schedules.csv")},
                                       "optimal" if row["Optimal"] == "1" else "best_known"))
    for fold in range(5):
        train = [r for i, r in enumerate(records) if i % 5 != fold]
        if REVISION != "v1":
            train = stratify(train)
        bundle("fssp", root / f"fold_{fold}", train,
               [r for i, r in enumerate(records) if i % 5 == fold],
               metadata={"fold": fold, "folds": 5, "target_gap_percent": 0.14,
                         "reference_version": "Zenodo 4542886", "literature_reference_matched": False,
                         "target_status": "provisional_until_same_reference_and_budget_verified"})
    return root


def stratify(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(row)
    return [group[index] for index in range(max(map(len, groups.values())))
            for group in groups.values() if index < len(group)]


def main():
    global BUNDLE_ROOT, REVISION
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["tsp_gls", "cvrp", "fssp"])
    parser.add_argument("--revision", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    REVISION = args.revision
    if REVISION != "v1":
        BUNDLE_ROOT = BUNDLE_ROOT / REVISION
    print({"tsp_gls": tsp_gls, "cvrp": cvrp, "fssp": fssp}[args.kind]())


if __name__ == "__main__":
    main()
