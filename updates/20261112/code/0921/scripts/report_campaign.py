"""Aggregate independent searches, preserving missing/invalid folds and seeds."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_campaign import load_registry, report, write


def collect_repeat(root, jobs):
    """One independent search repeat includes every preregistered fold."""
    values, baseline, families, ids = [], [], {}, set()
    for job in sorted(jobs, key=lambda j: j["fold"]):
        run = root / "jobs" / job["id"]
        summary_path = run / "validation_summary.json"
        if not summary_path.exists():
            return {"status": "NOT_COMPLETE", "missing_job": job["id"]}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["phase"] != "benchmark" or summary["selection"]["validation_used_for_selection"]:
            raise ValueError("only benchmark results with training-only selection are eligible")
        protocol = json.loads(Path(job["protocol"]).read_text(encoding="utf-8"))
        manifest = json.loads(Path(protocol["validation"]["source_path"]).read_text(encoding="utf-8"))
        rows = manifest["instances"]
        for label, destination in (("selected", values), ("baseline", baseline)):
            result = json.loads((run / f"heldout_{label}.json").read_text(encoding="utf-8"))
            if not result["valid"]:
                return {"status": "INVALID", "job": job["id"], "label": label,
                        "error_type": result.get("error_type"), "error_message": result.get("error_message")}
            if result["instance_ids"] != [row["id"] for row in rows]:
                raise ValueError("heldout instance order differs from registration")
            gaps = np.asarray(result["gap_matrix"], dtype=float)
            if gaps.shape != (job["final_seeds"], len(rows)) or not np.isfinite(gaps).all():
                raise ValueError("incomplete seed/instance denominator")
            means = gaps.mean(axis=0)
            destination.extend(means.tolist())
            if label == "selected":
                for row, mean in zip(rows, means):
                    if row["id"] in ids:
                        raise ValueError("validation instance appears in multiple folds")
                    ids.add(row["id"])
                    families.setdefault(row["family"], []).append(float(mean))
    return {"status": "COMPLETE", "instance_count": len(ids), "gap_percent": float(np.mean(values)),
            "baseline_gap_percent": float(np.mean(baseline)),
            "groups": {key: {"count": len(val), "gap_percent": float(np.mean(val))}
                       for key, val in sorted(families.items())}}


def aggregate(root, registry):
    groups = {}
    for job in registry["jobs"]:
        if job["phase"] == "benchmark":
            groups.setdefault((job["kind"], job["framework"], job["repeat"]), []).append(job)
    repeats = {key: collect_repeat(root, jobs) for key, jobs in groups.items()}
    result = {}
    for kind, framework in sorted({key[:2] for key in groups}):
        rows = [r for key, r in sorted(repeats.items()) if key[:2] == (kind, framework)]
        complete = all(row["status"] == "COMPLETE" for row in rows)
        gap = [row["gap_percent"] for row in rows] if complete else []
        result[f"{kind}_{framework}"] = {
            "status": "COMPLETE" if complete else "NOT_COMPLETE",
            "expected_independent_repeats": len(rows), "repeats": rows,
            "mean_gap_percent": float(np.mean(gap)) if complete else None,
            "sample_variance": float(np.var(gap, ddof=1)) if complete and len(gap) > 1 else None,
            "uncertainty_unit": "independent_algorithm_search_repeat",
        }
    targets = {}
    for kind, target in (("tsp_gls", .2), ("tsp_constructive", 13.), ("fssp", .14)):
        row = result.get(kind + "_on", {})
        targets[kind] = {"target_percent": target,
                         "status": ("MET_ON_REGISTERED_DATA" if row["mean_gap_percent"] <= target else "NOT_MET")
                         if row.get("status") == "COMPLETE" and row["expected_independent_repeats"] >= 3 else "NOT_COMPLETE",
                         "literature_same_condition_claim": False}
    return {"registry_sha256": registry["sha256"], "results": result, "targets": targets,
            "literature_baseline_reruns": "NOT_RUN", "CVRP_superiority_claim": "NOT_ESTABLISHED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    registry = load_registry(root)
    progress = report(root, registry)
    result = aggregate(root, registry)
    write(root / "results.json", result)
    lines = ["# Registered campaign results", "",
             f"Completed searches: {progress['completed_searches']}/{progress['total_searches']}.",
             f"Reported model tokens: {progress['model_tokens']}.", "",
             "| Task / framework | Status | Mean gap % | Independent-search sample variance |",
             "| --- | --- | --- | --- |"]
    for name, row in result["results"].items():
        lines.append(f"| {name} | {row['status']} | {row['mean_gap_percent']} | {row['sample_variance']} |")
    lines.extend(["", "Missing or invalid folds are never dropped from aggregate results.",
                  "Literature baseline reruns: NOT_RUN. CVRP superiority: NOT_ESTABLISHED.",
                  "These results do not imply matching the original AHD samples, paper runtimes, or BKS versions."])
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["targets"], indent=2))


if __name__ == "__main__":
    main()
