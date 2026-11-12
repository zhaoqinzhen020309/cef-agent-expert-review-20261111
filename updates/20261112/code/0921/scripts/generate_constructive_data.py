"""ReEvo's published uniform generator + exact MILP reference certificates.

Unfinished exact solves stay BLOCKED and never become invented optimum costs.
This reproduces the public ReEvo generator, not unreleased AHD Agent raw data.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import time
import warnings
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_task_bundles import instance_record, bundle


def solve_reference(job):
    identifier, xy, seconds = job
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix, vstack
    root = ROOT / "data/bundles/tsp_constructive"
    receipt = root / "certificates" / (identifier + ".json")
    if receipt.is_file():
        old = json.loads(receipt.read_text())
        if old.get("status") == "optimal":
            return old
        history = receipt.with_name(receipt.stem + f".attempt_{old['seconds_limit']:g}s.json")
        if not history.exists():
            history.write_text(receipt.read_text())
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    n = len(d)
    a, b = np.triu_indices(n, 1)
    variables = len(a)
    columns = np.tile(np.arange(variables), 2)
    constraints = coo_matrix((np.ones(2 * variables), (np.r_[a, b], columns)), shape=(n, variables)).tocsr()
    lower, upper = np.full(n, 2.0), np.full(n, 2.0)
    started = time.monotonic()
    cuts = 0
    record = {"id": identifier, "status": "BLOCKED", "reason": "exact_reference_not_finished",
              "solver": "scipy.optimize.milp/HiGHS", "seconds_limit": seconds}
    while time.monotonic() - started < seconds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = milp(d[a, b], integrality=np.ones(variables), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(constraints, lower, upper),
                          options={"time_limit": max(.01, seconds - (time.monotonic() - started)),
                                   "mip_rel_gap": 0.0, "threads": 1})
        if result.x is None:
            break
        edges = np.flatnonzero(result.x > .5)
        neighbors = [[] for _ in range(n)]
        for index in edges:
            neighbors[a[index]].append(int(b[index]))
            neighbors[b[index]].append(int(a[index]))
        if any(len(items) != 2 for items in neighbors):
            break
        unseen, components = set(range(n)), []
        while unseen:
            stack = [min(unseen)]
            component = set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(neighbors[node])
            unseen -= component
            components.append(sorted(component))
        if len(components) == 1:
            tour, previous, node = [0], -1, 0
            for _ in range(n - 1):
                nxt = next(v for v in neighbors[node] if v != previous)
                tour.append(nxt)
                previous, node = node, nxt
            cost = float(sum(d[tour[i], tour[(i + 1) % n]] for i in range(n)))
            if result.status == 0 and result.mip_gap <= 1e-7 and len(set(tour)) == n:
                row = instance_record(root, identifier, {"distance_matrix": d}, cost, f"N={n}",
                      {"generator": "https://github.com/ai4co/reevo/blob/main/problems/tsp_constructive/gen_inst.py",
                       "generator_seed": 1234, "reference_solver": "scipy.optimize.milp", "raw_AHD_dataset_match": False})
                record.update(status="optimal", reason=None, reference=cost, lower_bound=float(result.mip_dual_bound),
                              relative_solver_gap=float(result.mip_gap), tour=tour, record=row)
            break
        if result.status != 0:
            break
        new_rows = []
        for component in components:
            member = np.zeros(n, dtype=bool)
            member[component] = True
            indices = np.flatnonzero(member[a] & member[b])
            new_rows.append(coo_matrix((np.ones(len(indices)), (np.zeros(len(indices), dtype=int), indices)), shape=(1, variables)).tocsr())
            lower = np.r_[lower, -np.inf]
            upper = np.r_[upper, len(component) - 1]
        constraints = vstack([constraints, *new_rows], format="csr")
        cuts += len(new_rows)
    record.update(seconds=time.monotonic() - started, subtour_cuts=cuts)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    rng = np.random.RandomState(1234)
    jobs = [(f"train50_{i:04d}", xy, args.seconds) for i, xy in enumerate(rng.rand(64, 50, 2))]
    # The public generator consumes validation N=20 before N=50/100/200.
    for n in (20, 50, 100, 200):
        data = rng.rand(64, n, 2)
        if n != 20:
            jobs.extend((f"val{n}_{i:04d}", xy, args.seconds) for i, xy in enumerate(data))
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_reference, job): job[0] for job in jobs}
        for future in as_completed(futures):
            record = future.result()
            results[record["id"]] = record
            print(json.dumps({"done": len(results), "total": len(jobs), "id": record["id"], "status": record["status"], "seconds": record["seconds"]}), flush=True)
    root = ROOT / "data/bundles/tsp_constructive"
    blocked = [identifier for identifier, record in results.items() if record["status"] != "optimal"]
    (root / "reference_status.json").write_text(json.dumps({"status": "BLOCKED" if blocked else "READY", "blocked": blocked, "certified": len(jobs) - len(blocked)}, indent=2))
    if not blocked:
        train = [results[j[0]]["record"] for j in jobs if j[0].startswith("train")]
        validation = [results[j[0]]["record"] for j in jobs if j[0].startswith("val")]
        print(bundle("tsp_constructive", root, train, validation,
                     metadata={"design_instances": 64, "validation_counts": {"50": 64, "100": 64, "200": 64},
                               "candidate_budget": 100, "target_gap_percent": 13.0,
                               "AHD_raw_dataset_match": False, "comparison": "distribution_matched_only"}))


if __name__ == "__main__":
    main()
