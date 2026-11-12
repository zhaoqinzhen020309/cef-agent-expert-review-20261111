"""Raw score integration, with explicit units and no fabricated reference gaps."""
from types import SimpleNamespace
from dataclasses import asdict, replace
import hashlib
import json
import numpy as np

import pytest

from agent_ahd.core import engine
from agent_ahd.core.archive import build_archive_snapshot
from agent_ahd.core.research_records import EvaluationRecord
from agent_ahd.core.search_state import SearchConfig, SearchState
from agent_ahd.problems.base import EvaluationResult


def raw_record(identifier, values):
    result = EvaluationResult(candidate_id=identifier, valid=True, score=sum(values)/len(values),
        gap_unit="not_applicable", runtime_sec=0.2, objective_name="makespan", objective_unit="processing_time",
        metric_name="makespan", metric_unit="processing_time",
        objective_matrix=[values], seeds=[2024], instance_ids=["a", "b"], runtime_matrix=[[0.1, 0.1]])
    record = engine._result_record(result, candidate_id=identifier, evaluation_id="eval_"+identifier,
        split="train", requested_seeds=(2024,), requested_instance_ids=("a", "b"))
    assert record.valid, "raw-objective results are still being rejected for lacking a gap matrix"
    return record


def test_raw_objective_roundtrip_never_labels_cost_as_gap():
    record = raw_record("candidate", [2.0, 4.0])
    assert record.gap_matrix == ()
    assert record.mean_gap is None
    assert record.mean_score == 3.0
    assert record.score_matrix == ((2.0, 4.0),)
    assert (record.metric_name, record.metric_unit) == ("makespan", "processing_time")
    assert EvaluationRecord.from_dict(record.to_dict()) == record


def test_state_and_archive_rank_raw_mean_makespan():
    state = SearchState.create(run_id="unit", baseline_path="unused", baseline_sha256="0"*64,
                               config=SearchConfig(batch_size=2, initial_size=2))
    nodes = [state.allocate_node(round_index=0, generation_mode="fresh", solver_relpath=f"{i}/solver.py") for i in range(2)]
    for node, values in zip(nodes, ([1.0, 9.0], [4.0, 4.0])):
        state.record_evaluation_record(raw_record(node.node_id, values))
    assert [node.score for node in nodes] == [5.0, 4.0]
    assert nodes[0].instance_gaps is None
    assert nodes[0].instance_scores == [1.0, 9.0]
    snapshot = build_archive_snapshot(state.evaluations, canonical_instance_ids=("a", "b"))
    assert snapshot.mean_elites[0] == nodes[1].node_id
    profile = snapshot.profiles[nodes[1].node_id]
    assert profile.mean_score == 4.0
    observation = snapshot.observations[f"candidate:{nodes[1].node_id}:profile"]
    assert observation.values["mean_score"] == 4.0
    assert "mean_gap" not in observation.values
    assert "gap" not in observation.statement.lower()
    assert SearchState.from_dict(state.to_dict()).nodes[0].instance_scores == [1.0, 9.0]


def test_mixed_metric_archive_is_rejected():
    raw = raw_record("raw", [1.0, 2.0])
    gap = EvaluationRecord("gap", "gap", "train", (2024,), ("a", "b"), ((1.0, 2.0),), (), True)
    with pytest.raises(ValueError, match="metric"):
        build_archive_snapshot([raw, gap])


def test_legacy_percentage_gap_keeps_original_scale():
    record = EvaluationRecord("gap", "gap", "train", (2024,), ("a", "b"), ((1.5, 2.5),), (), True)
    assert record.mean_score == record.mean_gap == 2.0
    snapshot = build_archive_snapshot([record])
    assert snapshot.profiles["gap"].mean_gap == 2.0
    assert snapshot.observations["candidate:gap:profile"].values["mean_gap"] == 2.0


def test_raw_ranking_is_not_reference_normalized():
    first = raw_record("A", [10.0, 2000.0])
    second = raw_record("B", [20.0, 1500.0])
    assert first.mean_score == 1005.0 and second.mean_score == 760.0
    assert build_archive_snapshot([first, second]).mean_elites[0] == "B"
    # A reference-normalized average would incorrectly favor A for this objective.
    assert ((10/10-1)+(2000/1000-1))/2 < ((20/10-1)+(1500/1000-1))/2


def test_old_dual_matrix_evaluator_still_optimizes_percentage_gap():
    result = EvaluationResult("legacy", True, 2.0, "percent", 0.1,
        seeds=[2024], instance_ids=["a", "b"], gap_matrix=[[1.5, 2.5]],
        objective_matrix=[[100.0, 200.0]], objective_name="makespan", objective_unit="cost")
    record = engine._result_record(result, candidate_id="legacy", evaluation_id="eval", split="train")
    assert record.metric_name == "optimality_gap" and record.mean_score == 2.0
    result.score = 150.0
    with pytest.raises(ValueError, match="matrix mean"):
        engine._result_record(result, candidate_id="legacy", evaluation_id="eval", split="train")


def test_raw_dual_matrix_result_is_rejected_before_conversion():
    result = EvaluationResult("raw", True, 3.0, "not_applicable", 0.1,
        seeds=[2024], instance_ids=["a", "b"], gap_matrix=[[1.0, 2.0]],
        objective_matrix=[[2.0, 4.0]], objective_name="makespan", objective_unit="processing_time",
        metric_name="makespan", metric_unit="processing_time")
    with pytest.raises(ValueError, match="gap matrix"):
        engine._result_record(result, candidate_id="raw", evaluation_id="eval", split="train")


def raw_protocol(tmp_path):
    from agent_ahd.problems.base import EntryContract
    entry = EntryContract("SearchStrategy", "solve",
        ("self", "processing_times", "deadline", "iteration_limit", "seed"))
    solver = tmp_path / "solver.py"
    solver.write_text("class SearchStrategy:\n    def solve(self, processing_times, deadline, iteration_limit, seed):\n        return [0, 1, 2]\n")
    source = tmp_path / "unit.npz"
    np.savez(source, processing_times=np.array([[2., 3.], [4., 1.], [1., 2.]]))
    metric = {"name": "makespan", "unit": "processing_time", "aggregation": "mean", "direction": "minimize"}
    manifest = {"schema": "task_dataset_v1", "kind": "fssp", "split": "train",
        "contract_id": "raw_unit", "metric": metric, "entry_contract": asdict(entry), "instances": [
        {"id": "unit", "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
         "family": "3_2", "size": 3, "source": "synthetic"}]}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(manifest))
    protocol = {"protocol_id": "raw_unit", "data_root": str(tmp_path), "training_only": True,
        "metric": metric, "task": {"name": "fssp", "description": "Minimize mean makespan.",
        "baseline_path": str(solver), "entry_contract": asdict(entry),
        "evaluator_module": "agent_ahd.evaluators.manifest_evaluator"},
        "train": {"source_path": str(path), "instance_ids": ["unit"]},
        "validation": {"source_path": None, "instance_ids": []},
        "blind": {"commitment": "not_registered:no_final_test_in_this_search"}}
    return protocol, path, solver


def test_real_raw_manifest_evaluation_without_references(tmp_path):
    from agent_ahd.evaluators.manifest_evaluator import evaluate_candidate_file
    from agent_ahd.problems.base import EvaluationBudget
    protocol, manifest, solver = raw_protocol(tmp_path)
    result = evaluate_candidate_file("raw", solver, EvaluationBudget(n_instances=1, time_limit=2, ite_max=1),
        candidate_timeout=15, data_path=manifest, instance_ids=("unit",))
    assert result.valid, result.error_message
    # Machine completion after each job: (2,5), (6,7), (7,9).
    assert result.objective_matrix == [[9.0]]
    assert result.score == 9.0 and result.gap_matrix is None
    assert result.metric_name == "makespan" and result.metric_unit == "processing_time"
    assert result.gap_unit == "not_applicable"


def test_training_only_adapter_and_metric_fingerprint(tmp_path):
    from agent_ahd.problems.manifest_adapter import ManifestAdapter
    from agent_ahd.problems.base import EvaluationBudget
    protocol, _, solver = raw_protocol(tmp_path)
    adapter = ManifestAdapter("fssp", protocol)
    assert adapter.validation_manifest.instance_ids == ()
    assert adapter.validation_manifest.source_path is None
    with pytest.raises(PermissionError, match="training.only"):
        adapter.make_task(candidate_id="raw", candidate_path=solver,
            budget=EvaluationBudget(n_instances=1), split="validation")
    task = adapter.make_task(candidate_id="raw", candidate_path=solver,
        budget=EvaluationBudget(n_instances=1), split="train")
    fingerprint = engine._evaluation_fingerprint(task, source_sha256="a"*64)
    assert (fingerprint.metric_name, fingerprint.metric_unit) == ("makespan", "processing_time")
    changed = replace(task, metric_name="optimality_gap", metric_unit="percent", gap_unit="percent")
    assert engine._evaluation_fingerprint(changed, source_sha256="a"*64).cache_key != fingerprint.cache_key
    assert engine._evaluation_task_batch_fingerprint([task]) != engine._evaluation_task_batch_fingerprint([changed])


def test_raw_feedback_and_memory_use_declared_units():
    from agent_ahd.agents.controller_agent import build_controller_context, build_controller_prompt_context
    from agent_ahd.core.research_guidance import retrieve_training_episodes
    from agent_ahd.core.research_records import ScopeSpec, SearchAction
    parent = raw_record("__baseline_control__", [10., 20.])
    child = raw_record("candidate", [9., 18.])
    node = SimpleNamespace(node_id="candidate", round_index=0, valid=True, primary_parent_id=None,
        hypothesis_id=None, solver_sha256="a"*64, error_type=None)
    state = SimpleNamespace(nodes=[node], evaluations=[parent, child], hypotheses=[])
    config = SimpleNamespace(n_instances=2, base_seeds=(2024,))
    adapter = SimpleNamespace(train_manifest=SimpleNamespace(instance_ids=("a", "b")), dataset_protocol={})
    memory = retrieve_training_episodes(state, config, adapter, before_round=1)
    assert memory["baseline_mean_score"] == 15.0
    assert memory["episodes"][0]["mean_score"] == 13.5
    assert memory["metric_name"] == "makespan" and memory["metric_unit"] == "processing_time"
    assert "gap" not in json.dumps(memory).lower()
    snapshot = build_archive_snapshot([parent, child], excluded_candidate_ids={"__baseline_control__"})
    context = build_controller_context(round_index=1, batch_size=1, problem_context="Minimize mean makespan.",
        snapshot=snapshot, evidence=(), remaining_candidate_budget=1,
        remaining_candidate_evaluation_units=2, remaining_diagnostic_evaluation_units=0,
        remaining_model_calls=10, candidate_scopes={"candidate": ScopeSpec("full_solver")},
        allowed_actions={SearchAction.EXPLOIT}, split_instance_ids={"train": ("a", "b"), "validation": ()})
    visible = json.dumps(build_controller_prompt_context(context))
    assert '"mean_score": 13.5' in visible and "processing_time" in visible
    assert "mean_gap" not in visible and "per_instance_gap" not in visible


def test_raw_pairing_rejects_different_metric():
    from agent_ahd.core.evidence import _require_paired
    raw = raw_record("raw", [1., 2.])
    gap = EvaluationRecord("gap", "gap", "train", (2024,), ("a", "b"), ((1., 2.),), (), True)
    with pytest.raises(ValueError, match="metric"):
        _require_paired(raw, gap)


def test_raw_hypothesis_request_only_offers_score_predictions():
    from agent_ahd.agents.agent_runner import AgentResponse
    from agent_ahd.agents.candidate_agent import request_code_hypothesis
    from agent_ahd.core.research_records import PlannerDecision, ScopeSpec, SearchAction
    request = {}
    decision = PlannerDecision("raw", SearchAction.DIVERSIFY, ScopeSpec("full_solver"))

    class Provider:
        def complete_json(self, **kwargs):
            request.update(kwargs)
            payload = {"hypothesis": {"target_problem": "makespan", "planned_change": "Change construction.",
                "predicted_effects": {"mean_score": "decrease"}, "regression_risks": [],
                "ablation_scope": {"kind": "full_solver", "symbols": []}}}
            return AgentResponse(payload, json.dumps(payload), {})

    task = json.dumps({"resource_facts": {"metric": {"name": "makespan", "unit": "processing_time"}}})
    hypothesis, _ = request_code_hypothesis(Provider(), task=task, decision=decision, source_symbols=("solve",))
    schema = json.dumps(request["output_schema"])
    assert "mean_score" in schema and "mean_gap" not in schema
    assert "mean_gap" not in request["system_prompt"]
    assert hypothesis.predicted_effects == {"mean_score": "decrease"}


def test_receipt_result_must_match_task_metric_and_instance_contract():
    result = EvaluationResult("raw", True, 3.0, "not_applicable", 0.1,
        seeds=[2024], instance_ids=["a", "b"], objective_matrix=[[2., 4.]],
        objective_name="makespan", objective_unit="processing_time", metric_name="makespan", metric_unit="processing_time")
    with pytest.raises(ValueError, match="metric"):
        engine._result_record(result, candidate_id="raw", evaluation_id="eval", split="train",
                              expected_metric_name="optimality_gap", expected_metric_unit="percent")
    with pytest.raises(ValueError, match="instances"):
        engine._result_record(result, candidate_id="raw", evaluation_id="eval", split="train",
                              requested_instance_ids=("a", "c"))


def test_state_cache_cannot_reuse_another_metric(tmp_path):
    from agent_ahd.problems.base import CandidateTask, EvaluationBudget
    record = raw_record("raw", [2., 4.])
    state = SimpleNamespace(nodes=[SimpleNamespace(node_id="raw", solver_sha256="a"*64)], evaluations=[record])
    task = CandidateTask("raw", tmp_path / "solver.py", "unused", "unused",
        EvaluationBudget(n_instances=2, seeds=(2024,)), instance_ids=("a", "b"))
    assert engine._matching_cached_state_record(state, source_sha256="a"*64, task=task) is None
    task = replace(task, metric_name="makespan", metric_unit="processing_time", gap_unit="not_applicable")
    assert engine._matching_cached_state_record(state, source_sha256="a"*64, task=task) == record


def test_raw_hypothesis_failure_records_actual_request_identity(tmp_path):
    from agent_ahd.agents.candidate_agent import CandidateWorkspace, hypothesis_system_prompt, build_code_hypothesis_schema
    from agent_ahd.core.research_records import PlannerDecision, SearchAction, ScopeSpec
    decision = PlannerDecision("raw", SearchAction.DIVERSIFY, ScopeSpec("full_solver"))
    task = json.dumps({"resource_facts": {"metric": {"name": "makespan", "unit": "processing_time"}}})
    job = {"task": task, "decision": decision, "source_symbols": ("solve",),
           "control_workspace": CandidateWorkspace("raw", tmp_path, tmp_path / "solver.py")}
    engine.PlannerSearchEngine._write_generation_error(job, "hypothesis", ValueError("unit fixture failure"))
    record = json.loads((tmp_path / "generation_error.json").read_text())
    assert record["system_prompt_sha256"] == engine._text_sha256(hypothesis_system_prompt(task))
    schema = build_code_hypothesis_schema(decision, source_symbols=("solve",), task=task)
    assert record["output_schema_sha256"] == engine._text_sha256(json.dumps(schema, sort_keys=True))


def test_raw_target_cluster_context_uses_score_units():
    snapshot = build_archive_snapshot([raw_record("parent", [10., 20.]), raw_record("other", [9., 25.])])
    context = engine._cluster_gap_context(snapshot, ("a", "b"), "parent")
    assert context["parent_per_instance_score"] == {"a": 10., "b": 20.}
    assert context["archive_best_per_instance_score"] == {"a": 9., "b": 20.}
    assert context["metric_name"] == "makespan" and context["metric_unit"] == "processing_time"
    assert "gap" not in json.dumps(context)


def test_raw_parent_feedback_and_replay_keep_score_identity(tmp_path):
    from agent_ahd.core.search_state import SolverNode
    from agent_ahd.core.replay import _to_result
    from agent_ahd.problems.base import CandidateTask, EvaluationBudget
    node = SolverNode("raw", 0, "raw", "raw/solver.py", "fresh", valid=True, score=3.0,
        metric_name="makespan", metric_unit="processing_time", gap_unit="not_applicable", instance_scores=[2., 4.])
    feedback = engine._node_result(node)
    assert feedback["metric_name"] == "makespan" and feedback["instance_scores"] == [2., 4.]
    task = CandidateTask("raw", tmp_path / "solver.py", "unused", "unused", EvaluationBudget(n_instances=2),
        instance_ids=("a", "b"), metric_name="makespan", metric_unit="processing_time", gap_unit="not_applicable")
    result = _to_result(raw_record("raw", [2., 4.]), task)
    assert result.score == 3.0 and result.objective_matrix == [[2., 4.]]
    assert result.gap_matrix is None and result.instance_gaps is None
    assert result.metric_name == "makespan" and result.metric_unit == "processing_time"


def test_paired_lookup_does_not_choose_another_metric():
    raw = raw_record("raw", [2., 4.])
    gap = EvaluationRecord("gap", "raw", "train", (2024,), ("a", "b"), ((1., 2.),), (), True)
    assert engine._paired_record(SimpleNamespace(evaluations=[raw, gap]), "raw", raw) == raw
