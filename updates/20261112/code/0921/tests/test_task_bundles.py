from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pytest

from scripts.prepare_task_bundles import bundle, instance_record, ENTRIES
from agent_ahd.problems.registry import get_problem_adapter
from agent_ahd.problems.base import EvaluationBudget
from agent_ahd.core.parallel_evaluator import evaluate_tasks


def setup_bundle(tmp_path, kind="tsp_constructive", *, same=False, reference_kind="optimal"):
    a = np.array([[0., 1, 2], [1, 0, 1], [2, 1, 0]])
    b = a if same else a * 2
    first = instance_record(tmp_path, "train_a", {"distance_matrix": a}, 4, "N=3", "test fixture", reference_kind)
    second = instance_record(tmp_path, "val_a", {"distance_matrix": b}, 8, "N=3", "test fixture", reference_kind)
    path = bundle(kind, tmp_path, [first], [second])
    return get_problem_adapter(kind, dataset_protocol_path=path)


def test_end_to_end_manifest_evaluation(tmp_path):
    adapter = setup_bundle(tmp_path)
    isolated = tmp_path / "candidate"
    isolated.mkdir()
    target = isolated / "solver.py"
    target.write_text(adapter.baseline_path.read_text())
    task = adapter.make_task(candidate_id="baseline", candidate_path=target,
           budget=EvaluationBudget(n_instances=1, time_limit=1, seeds=(101, 102)), split="validation")
    result = evaluate_tasks([task])[0]
    assert result.valid, result.error_message
    assert result.gap_matrix == [[0.0], [0.0]]
    assert result.instance_ids == ["val_a"]


def test_no_blind_access_or_unknown_id(tmp_path):
    adapter = setup_bundle(tmp_path)
    common = dict(candidate_id="c", candidate_path=adapter.baseline_path, budget=EvaluationBudget(n_instances=1))
    with pytest.raises(PermissionError):
        adapter.make_task(**common, split="blind")
    with pytest.raises(ValueError):
        adapter.make_task(**common, split="validation", instance_ids=("train_a",))


def test_renamed_identical_data_is_not_a_holdout(tmp_path):
    with pytest.raises(ValueError, match="content overlaps"):
        setup_bundle(tmp_path, same=True)


def test_changed_instance_rejected(tmp_path):
    adapter = setup_bundle(tmp_path)
    path = tmp_path / "instances/val_a.npz"
    np.savez(path, distance_matrix=np.eye(3))
    task = adapter.make_task(candidate_id="c", candidate_path=adapter.baseline_path,
                             budget=EvaluationBudget(n_instances=1), split="validation")
    with pytest.raises(ValueError, match="hash changed"):
        evaluate_tasks([task])


def test_entry_contract_mismatch_rejected(tmp_path):
    adapter = setup_bundle(tmp_path)
    path = tmp_path / "validation.json"
    data = json.loads(path.read_text())
    data["entry_contract"]["method_name"] = "solve"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="entry contracts differ"):
        get_problem_adapter("tsp_constructive", dataset_protocol_path=tmp_path / "protocol.json")


@pytest.mark.parametrize("reference_kind,valid", [("best_known", True), ("optimal", False)])
def test_improving_bks_is_allowed_but_beating_declared_optimum_is_rejected(tmp_path, reference_kind, valid):
    setup_bundle(tmp_path, reference_kind=reference_kind)
    path = tmp_path / "validation.json"
    content = json.loads(path.read_text())
    content["instances"][0]["reference"] = 10.0  # The toy tour actually costs 8.
    path.write_text(json.dumps(content))
    adapter = get_problem_adapter("tsp_constructive", dataset_protocol_path=tmp_path / "protocol.json")
    task = adapter.make_task(candidate_id="baseline", candidate_path=adapter.baseline_path,
                             budget=EvaluationBudget(n_instances=1, time_limit=1, seeds=(101,)), split="validation")
    result = evaluate_tasks([task])[0]
    assert result.valid is valid
    if valid:
        assert result.score == pytest.approx(-20.0)
    else:
        assert result.error_type == "invalid_solution"
