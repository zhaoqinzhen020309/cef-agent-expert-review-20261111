"""Missing scientific evidence must not become a favorable aggregate."""
import json
from pathlib import Path

import pytest

from scripts.report_campaign import collect_repeat, aggregate


def fixture(tmp_path, *, valid=True, seeds=2):
    protocol = tmp_path / "protocol.json"
    manifest = tmp_path / "validation.json"
    manifest.write_text(json.dumps({"instances": [{"id": "x", "family": "N=50"}]}))
    protocol.write_text(json.dumps({"validation": {"source_path": str(manifest)}}))
    job = {"id": "job", "protocol": str(protocol), "fold": 0, "final_seeds": 2,
           "kind": "tsp_constructive", "framework": "on", "repeat": 0, "phase": "benchmark"}
    run = tmp_path / "jobs/job"
    run.mkdir(parents=True)
    (run / "validation_summary.json").write_text(json.dumps({"phase": "benchmark", "selection": {"validation_used_for_selection": False}}))
    for label in ("selected", "baseline"):
        (run / f"heldout_{label}.json").write_text(json.dumps({"valid": valid, "instance_ids": ["x"], "gap_matrix": [[2.]] * seeds}))
    return job


def test_missing_fold_does_not_disappear(tmp_path):
    job = fixture(tmp_path)
    missing = {**job, "id": "absent", "fold": 1}
    assert collect_repeat(tmp_path, [job, missing])["status"] == "NOT_COMPLETE"


def test_invalid_solver_is_not_a_missing_value_to_average(tmp_path):
    job = fixture(tmp_path, valid=False)
    assert collect_repeat(tmp_path, [job])["status"] == "INVALID"


def test_seed_denominator_checked(tmp_path):
    job = fixture(tmp_path, seeds=1)
    with pytest.raises(ValueError, match="denominator"):
        collect_repeat(tmp_path, [job])


def test_seed_repeats_are_not_independent_searches(tmp_path):
    job = fixture(tmp_path)
    results = aggregate(tmp_path, {"sha256": "test", "jobs": [job]})
    row = results["results"]["tsp_constructive_on"]
    assert row["mean_gap_percent"] == 2.
    assert row["sample_variance"] is None
    assert row["expected_independent_repeats"] == 1


def test_duplicate_heldout_instance_rejected(tmp_path):
    job = fixture(tmp_path)
    with pytest.raises(ValueError, match="multiple folds"):
        collect_repeat(tmp_path, [job, {**job, "fold": 1}])
