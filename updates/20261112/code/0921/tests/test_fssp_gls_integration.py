"""Synthetic, offline contract checks, never a performance experiment."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from agent_ahd.problems.base import EntryContract, EvaluationBudget


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "problems/fssp/neh_gls_v1/solver.py"
PREPARER = ROOT / "scripts/prepare_fssp_gls_protocol.py"
ENTRY = EntryContract("SearchStrategy", "solve", (
    "self", "processing_times", "deadline", "iteration_limit", "seed"))


def preparation_module():
    assert PREPARER.is_file(), "the independent GLS protocol preparer is missing"
    spec = importlib.util.spec_from_file_location("prepare_gls_acceptance", PREPARER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_protocol(tmp_path, *, stale=False):
    from agent_ahd.evaluators.manifest_evaluator import objective

    root = tmp_path / "source"
    root.mkdir()
    recorded_root = tmp_path / "old_location" if stale else root
    protocol = {"protocol_id": "fssp_unit_v1", "data_root": str(recorded_root),
                "task": {"name": "fssp", "description": "Minimize FSSP makespan; full solver.",
                         "baseline_path": str(ROOT / "problems/fssp/problem.py"),
                         "entry_contract": asdict(ENTRY),
                         "evaluator_module": "agent_ahd.evaluators.manifest_evaluator"},
                "blind": {"commitment": "not_registered:no_blind_claim"},
                "metadata": {"unit_fixture": True}}
    for split, offset in (("train", 0), ("validation", 1)):
        matrix = np.array([[2., 3.], [4., 1.], [1., 2.]]) + offset
        data_path = root / (split + ".npz")
        np.savez(data_path, processing_times=matrix)
        row = {"id": split + "_unit", "family": "3_2", "size": 3,
               "path": str(recorded_root / data_path.name),
               "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
               "reference": objective("fssp", matrix, [0, 1, 2]),
               "reference_kind": "best_known", "source": "synthetic unit fixture"}
        manifest = {"schema": "task_dataset_v1", "kind": "fssp", "split": split,
                    "contract_id": protocol["protocol_id"], "entry_contract": asdict(ENTRY),
                    "instances": [row]}
        target = root / (split + ".json")
        target.write_text(json.dumps(manifest), encoding="utf-8")
        protocol[split] = {"format": "task_dataset_v1", "source_path": str(recorded_root / target.name),
                           "instance_ids": [row["id"]], "family_by_instance": {row["id"]: row["family"]}}
    path = root / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return path, recorded_root, root


def test_prepare_independent_protocol_preserves_data_and_original(tmp_path):
    base, _, source_root = fixture_protocol(tmp_path)
    originals = {p.name: p.read_bytes() for p in source_root.iterdir()}
    output = tmp_path / "gls"
    target = preparation_module().prepare_protocol(base, output, solver_path=SOURCE)
    prepared = json.loads(target.read_text(encoding="utf-8"))
    original = json.loads(base.read_text(encoding="utf-8"))
    assert Path(prepared["task"]["baseline_path"]) == SOURCE
    assert prepared["task"]["entry_contract"] == original["task"]["entry_contract"]
    assert prepared["metadata"] == original["metadata"]
    assert prepared["blind"] == original["blind"]
    assert prepared["protocol_id"] != original["protocol_id"]
    for split in ("train", "validation"):
        actual = json.loads(Path(prepared[split]["source_path"]).read_text(encoding="utf-8"))
        expected = json.loads((source_root / (split + ".json")).read_text(encoding="utf-8"))
        assert actual["instances"] == expected["instances"]
        assert actual["contract_id"] == prepared["protocol_id"]
        assert prepared[split]["instance_ids"] == original[split]["instance_ids"]
    assert {p.name: p.read_bytes() for p in source_root.iterdir()} == originals
    frozen = {p.name: p.read_bytes() for p in output.iterdir()}
    assert preparation_module().prepare_protocol(base, output, solver_path=SOURCE) == target
    assert {p.name: p.read_bytes() for p in output.iterdir()} == frozen


def test_stale_paths_require_explicit_root_relocation(tmp_path):
    base, old_root, new_root = fixture_protocol(tmp_path, stale=True)
    module = preparation_module()
    output = tmp_path / "gls"
    with pytest.raises(FileNotFoundError):
        module.prepare_protocol(base, output, solver_path=SOURCE)
    assert not output.exists()
    target = module.prepare_protocol(base, output, solver_path=SOURCE,
                                     old_root=old_root, new_root=new_root)
    content = json.loads(target.read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        rows = json.loads(Path(content[split]["source_path"]).read_text(encoding="utf-8"))["instances"]
        assert Path(rows[0]["path"]) == new_root / (split + ".npz")
        assert hashlib.sha256(Path(rows[0]["path"]).read_bytes()).hexdigest() == rows[0]["sha256"]


def test_changed_instance_is_rejected_before_writing(tmp_path):
    base, _, source_root = fixture_protocol(tmp_path)
    (source_root / "train.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        preparation_module().prepare_protocol(base, tmp_path / "gls", solver_path=SOURCE)
    assert not (tmp_path / "gls").exists()


def test_in_place_or_unrelated_output_is_not_overwritten(tmp_path):
    base, _, source_root = fixture_protocol(tmp_path)
    module = preparation_module()
    with pytest.raises(ValueError, match="source"):
        module.prepare_protocol(base, source_root, solver_path=SOURCE)
    output = tmp_path / "gls"
    output.mkdir()
    foreign = output / "notes.txt"
    foreign.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.prepare_protocol(base, output, solver_path=SOURCE)
    assert [p.name for p in output.iterdir()] == ["notes.txt"]
    assert foreign.read_text(encoding="utf-8") == "keep"


def test_invalid_split_or_other_task_is_not_relabelled_as_fssp(tmp_path):
    base, _, source_root = fixture_protocol(tmp_path)
    original = json.loads(base.read_text(encoding="utf-8"))
    changed = deepcopy(original)
    changed["task"]["name"] = "cvrp"
    base.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="fssp"):
        preparation_module().prepare_protocol(base, tmp_path / "gls", solver_path=SOURCE)
    base.write_text(json.dumps(original), encoding="utf-8")
    manifest_path = source_root / "validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["instances"] = json.loads((source_root / "train.json").read_text())["instances"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        preparation_module().prepare_protocol(base, tmp_path / "gls", solver_path=SOURCE)
    assert not (tmp_path / "gls").exists()


@pytest.mark.parametrize("overlap_kind", ["identifier", "content"])
def test_consistent_manifest_with_cross_split_overlap_is_rejected(tmp_path, overlap_kind):
    base, _, source_root = fixture_protocol(tmp_path)
    protocol = json.loads(base.read_text(encoding="utf-8"))
    train_row = json.loads((source_root / "train.json").read_text(encoding="utf-8"))["instances"][0]
    manifest_path = source_root / "validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["instances"][0]
    if overlap_kind == "identifier":
        row["id"] = train_row["id"]
    else:
        row["path"] = train_row["path"]
        row["sha256"] = train_row["sha256"]
        row["reference"] = train_row["reference"]
    protocol["validation"]["instance_ids"] = [row["id"]]
    protocol["validation"]["family_by_instance"] = {row["id"]: row["family"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    base.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        preparation_module().prepare_protocol(base, tmp_path / "gls", solver_path=SOURCE)
    assert not (tmp_path / "gls").exists()


def test_full_source_parent_feedback_and_resume_with_offline_provider(tmp_path):
    from agent_ahd.agents.agent_runner import AgentResponse
    from agent_ahd.agents.candidate_agent import (
        HYPOTHESIS_AGENT_SYSTEM_PROMPT, build_planner_candidate_task,
        prepare_search_candidate_workspace, request_code_hypothesis,
        request_code_implementation, validate_code_implementation_response,
    )
    from agent_ahd.core.baseline_parent import ensure_baseline_parent, SEED_ID
    from agent_ahd.core.code_checks import changed_symbols_in_scope
    from agent_ahd.core.decision_catalog import build_decision_catalog
    from agent_ahd.core.parallel_evaluator import evaluate_tasks
    from agent_ahd.core.research_records import EvaluationRecord, PlannerDecision, ScopeSpec, SearchAction
    from agent_ahd.core.search_state import SearchConfig, SearchState, save_search_state, load_search_state
    from agent_ahd.problems.registry import get_problem_adapter

    base, _, _ = fixture_protocol(tmp_path)
    protocol = preparation_module().prepare_protocol(base, tmp_path / "gls", solver_path=SOURCE)
    adapter = get_problem_adapter("fssp", dataset_protocol_path=protocol)
    budget = EvaluationBudget(n_instances=1, time_limit=30, ite_max=2, seeds=(401,))
    run = tmp_path / "offline_contract"
    control = run / "controls/baseline/solver.py"
    control.parent.mkdir(parents=True)
    shutil.copy2(SOURCE, control)
    baseline = evaluate_tasks([adapter.make_task(candidate_id="__baseline_control__",
        candidate_path=control, budget=budget, split="train")], max_workers=1)[0]
    assert baseline.valid, baseline.error_message
    digest = hashlib.sha256(control.read_bytes()).hexdigest()
    state = SearchState.create(run_id="offline", baseline_path=str(SOURCE),
        baseline_sha256=digest, config=SearchConfig(batch_size=1, initial_size=1))

    def record(result, identifier):
        return EvaluationRecord(identifier, result.candidate_id, "train", tuple(result.seeds),
            tuple(result.instance_ids), tuple(tuple(row) for row in result.gap_matrix),
            tuple(tuple(row) for row in result.runtime_matrix), result.valid,
            total_runtime_sec=result.runtime_sec)

    state.evaluations.append(record(baseline, "eval_baseline_control_000"))
    assert ensure_baseline_parent(state, run, seeds=budget.resolved_seeds,
                                  instance_ids=adapter.train_manifest.instance_ids)
    scope = ScopeSpec("full_solver")

    def request(parent_id, parent_source, feedback, round_index):
        observation_id = "profile:" + parent_id
        observations = {observation_id: {"kind": "candidate_profile", "statement": "Measured unit fixture",
            "values": {"candidate_id": parent_id, "mean_gap": feedback["score"]}}}
        decision = PlannerDecision("unit_" + str(round_index), SearchAction.EXPLOIT, scope,
            primary_parent_id=parent_id, observation_ids=(observation_id,),
            generation_context_ids=(observation_id,))
        node = state.allocate_node(round_index=round_index, generation_mode="continue",
            solver_relpath=f"nodes/node_{state.next_node_index:04d}/solver.py",
            primary_parent_id=parent_id, decision=decision)
        workspace = prepare_search_candidate_workspace(run_dir=run, node=node,
            baseline_path=SOURCE, primary_solver_path=parent_source, primary_result=feedback)
        task = build_planner_candidate_task(decision=decision, candidate_id=node.node_id,
            workspace=workspace.root, problem_context=adapter.problem_context(),
            observations=observations, target_cluster_context={}, auxiliary_mechanism=None)
        return node, decision, workspace, task

    seed = state.require_node(SEED_ID)
    node, decision, workspace, task = request(SEED_ID, run / seed.solver_relpath,
                                             baseline.to_record(), 0)
    incoming = json.loads(task)
    assert incoming["source_bundle"]["solver.py"] == SOURCE.read_text(encoding="utf-8")
    assert incoming["edit_contract"]["scope"] == {"kind": "full_solver", "symbols": []}
    modified = SOURCE.read_text(encoding="utf-8").replace("    return order\n", "    return order[::-1]\n")
    modified = modified.replace("iteration % 50 == 0", "iteration % 49 == 0")

    class OfflineProvider:
        model = "offline-contract-fixture"

        def __init__(self):
            self.requests = []

        def complete_json(self, **kwargs):
            self.requests.append(kwargs)
            if kwargs["system_prompt"] == HYPOTHESIS_AGENT_SYSTEM_PROMPT:
                payload = {"hypothesis": {"target_problem": "offline full-source acceptance",
                    "planned_change": "Change construction and restart locations for a scope contract test.",
                    "predicted_effects": {"mean_gap": "decrease"}, "regression_risks": ["fixture only"],
                    "ablation_scope": {"kind": "full_solver", "symbols": []}}}
            else:
                payload = {"solver_py": modified, "helper_files": [], "delete_files": []}
            return AgentResponse(payload, json.dumps(payload), {})

    provider = OfflineProvider()
    hypothesis, _ = request_code_hypothesis(provider, task=task, decision=decision,
        source_symbols=("sum_and_order", "SearchStrategy.solve"))
    implementation = AgentResponse({"solver_py": modified, "helper_files": [], "delete_files": []}, "", {})
    validate_code_implementation_response(workspace.root, implementation, decision=decision, hypothesis=hypothesis)
    request_code_implementation(provider, task=task, workspace=workspace.root,
                                hypothesis=hypothesis, decision=decision)
    changes = changed_symbols_in_scope(control.parent, workspace.root, scope)
    assert {"sum_and_order", "SearchStrategy.solve"} <= set(changes)
    child = evaluate_tasks([adapter.make_task(candidate_id=node.node_id,
        candidate_path=workspace.solver_path, budget=budget, split="train")], max_workers=1)[0]
    assert child.valid, child.error_message
    node.solver_sha256 = hashlib.sha256(workspace.solver_path.read_bytes()).hexdigest()
    node.solver_file_sha256 = {"solver.py": node.solver_sha256}
    state.record_evaluation_record(record(child, "eval_child_unit"))
    state.finish_round(0)
    checkpoint = run / "search_state.json"
    save_search_state(checkpoint, state)
    state = load_search_state(checkpoint)
    evaluated = next(r for r in state.evaluations if r.evaluation_id == "eval_child_unit")
    _, next_decision, _, next_task = request(node.node_id, workspace.solver_path,
        {"score": evaluated.mean_gap}, 1)
    request_code_hypothesis(provider, task=next_task, decision=next_decision,
                           source_symbols=("sum_and_order", "SearchStrategy.solve"))
    next_input = json.loads(provider.requests[-1]["user_prompt"])
    assert next_input["source_bundle"]["solver.py"] == modified
    assert next_input["evidence"]["profile:" + node.node_id]["values"]["mean_gap"] == evaluated.mean_gap
    profiles = {"profile:" + n.node_id: {"kind": "candidate_profile", "statement": "Measured fixture",
        "values": {"candidate_id": n.node_id, "mean_gap": n.score}}
        for n in state.nodes if n.valid}
    catalog = build_decision_catalog(round_index=1, available_actions=(SearchAction.EXPLOIT,),
        observations=profiles, failure_clusters={}, evidence=(),
        candidate_scopes={n.node_id: scope for n in state.nodes if n.valid}, candidate_source_structures={})
    assert {SEED_ID, node.node_id} <= {option.decision.primary_parent_id for option in catalog.candidate_options}
    assert len(provider.requests) == 3
