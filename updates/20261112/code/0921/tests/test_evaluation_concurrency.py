"""Offline scheduling checks; no LLM calls or optimization benchmarks."""
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from types import SimpleNamespace

import pytest
import numpy as np

from agent_ahd.agents.candidate_agent import CandidateWorkspace
from agent_ahd.core import engine
from agent_ahd.core.code_checks import CodeCheckResult
from agent_ahd.core.research_records import PlannerDecision, RoundPlan, ScopeSpec, SearchAction
from agent_ahd.core.search_state import SearchConfig, SearchState
from agent_ahd.core.stage_journal import CandidateStageJournal
from agent_ahd.evaluators import manifest_evaluator
from agent_ahd.problems.base import CandidateTask, EntryContract, EvaluationBudget, EvaluationResult


BASELINE = "def solve(x):\n    return x\n"
CHILD = "def solve(x):\n    return x[::-1]\n"


def make_round(root, module=engine, *, workers=3):
    root.mkdir(parents=True)
    baseline = root / "baseline.py"
    baseline.write_text(BASELINE, encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    state = SearchState.create(run_id="offline", baseline_path=str(baseline),
        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        config=SearchConfig(batch_size=3, initial_size=3))
    jobs, nodes, decisions, dispatched = {}, [], [], []
    for index, source in enumerate((CHILD, CHILD, CHILD.replace("x[::-1]", "x[::2]"))):
        decision = PlannerDecision(f"decision_{index}", SearchAction.DIVERSIFY, ScopeSpec("full_solver"))
        node = state.allocate_node(round_index=0, generation_mode="fresh",
            solver_relpath=f"nodes/node_{index:04d}/solver.py", decision=decision)
        workspace_root = root / "nodes" / node.node_id
        workspace_root.mkdir(parents=True)
        solver = workspace_root / "solver.py"
        solver.write_text(source, encoding="utf-8")
        snapshot = root / "snapshots" / node.node_id
        snapshot.mkdir(parents=True)
        (snapshot / "solver.py").write_text(BASELINE, encoding="utf-8")
        control = root / "controls" / node.node_id
        control.mkdir(parents=True)
        journal = CandidateStageJournal(control / "stages.json", candidate_id=node.node_id)
        journal.begin("generation", input_fingerprint="locked_" + node.node_id)
        journal.finish("generation", input_fingerprint="locked_" + node.node_id,
                       output_fingerprint="generated_" + node.node_id)
        jobs[node.node_id] = {"workspace": CandidateWorkspace(node.node_id, workspace_root, solver),
            "control_workspace": CandidateWorkspace(node.node_id, control, control / "solver.py"), "snapshot_root": snapshot,
            "decision": decision, "stage_journal": journal}
        nodes.append(node)
        decisions.append(decision)

    def make_task(**kwargs):
        return CandidateTask(**kwargs, evaluator_module="agent_ahd.evaluators.manifest_evaluator",
            evaluator_function="evaluate_candidate_file", gap_unit="percent",
            contract_id="offline", instance_ids=("unit",), data_path=manifest)

    def evaluate(tasks, *, max_workers):
        dispatched.append([task.candidate_id for task in tasks])
        assert max_workers == workers
        return [EvaluationResult(candidate_id=task.candidate_id, valid=True, score=1.0,
            gap_unit="percent", runtime_sec=0.25, split="train", seeds=[2024],
            instance_ids=["unit"], gap_matrix=[[1.0]], runtime_matrix=[[0.25]]) for task in tasks]

    subject = module.PlannerSearchEngine(run_dir=root,
        adapter=SimpleNamespace(baseline_path=baseline, make_task=make_task),
        config=module.PlannerEngineConfig(evaluation_workers=workers, n_instances=1),
        controller_runner=None, code_runner=None, evaluation_function=evaluate)
    round_path = root / "rounds/round_000"
    round_path.mkdir(parents=True)
    plan = RoundPlan(0, tuple(decisions))
    return subject, state, plan, tuple(nodes), jobs, round_path, dispatched


def evaluate_round(case, *, errors=None):
    subject, state, plan, nodes, jobs, path, _ = case
    return subject._evaluate_round(state, plan, nodes, (), jobs, errors or {}, path)


def reverse_checks(monkeypatch, module, count=3):
    original = module._check_workspace_scope
    barrier = threading.Barrier(count, timeout=5)
    gates = [threading.Event() for _ in range(count)]
    gates[-1].set()
    completed = []

    def check(snapshot, workspace, scope):
        index = int(Path(workspace).name.split("_")[-1])
        barrier.wait()
        assert gates[index].wait(5), "concurrent checks did not all start"
        result = original(snapshot, workspace, scope)
        completed.append(index)
        if index:
            gates[index - 1].set()
        return result

    monkeypatch.setattr(module, "_check_workspace_scope", check)
    return completed


def test_source_checks_overlap_without_reordering(tmp_path, monkeypatch):
    case = make_round(tmp_path / "run")
    completed = reverse_checks(monkeypatch, engine)
    owner = threading.get_ident()
    writes = []
    original_write = engine._write_candidate_artifacts

    def write(path, artifact):
        assert threading.get_ident() == owner
        writes.append(artifact.candidate_id)
        return original_write(path, artifact)

    monkeypatch.setattr(engine, "_write_candidate_artifacts", write)
    records = evaluate_round(case)
    _, state, _, nodes, jobs, _, dispatched = case
    identifiers = [node.node_id for node in nodes]
    assert completed == [2, 1, 0]
    assert writes == identifiers
    assert dispatched == [[identifiers[0], identifiers[2]]]
    assert [record.candidate_id for record in state.evaluations] == identifiers
    assert all(records[identifier].valid for identifier in identifiers)
    assert jobs[identifiers[1]]["stage_journal"].record("development_evaluation")["metadata"] == {
        "cache_hit": True, "same_batch_reuse": True, "valid": True}


def test_scope_rejection_does_not_collect_source(tmp_path, monkeypatch):
    case = make_round(tmp_path / "run")
    monkeypatch.setattr(engine, "_check_workspace_scope",
        lambda *args: CodeCheckResult(False, "scope_violation", "scope first"))

    def forbidden(**kwargs):
        raise AssertionError("artifact collection must be short-circuited")

    monkeypatch.setattr(engine, "collect_candidate_artifact", forbidden)
    records = evaluate_round(case)
    assert all(record.error_message == "scope first" for record in records.values())
    assert case[-1] == []


def test_ablation_rejection_does_not_collect_source(tmp_path, monkeypatch):
    subject, state, _, nodes, jobs, _, _ = make_round(tmp_path / "run")
    node = nodes[1]
    node.primary_parent_id = nodes[0].node_id
    job = jobs[node.node_id]
    job["decision"] = replace(job["decision"], action=SearchAction.ABLATE)
    monkeypatch.setattr(engine, "_check_workspace_scope", lambda *args: CodeCheckResult(True))
    monkeypatch.setattr(engine, "check_ablation_reversion",
        lambda *args: CodeCheckResult(False, "invalid_ablation", "reversion first"))

    def forbidden(**kwargs):
        raise AssertionError("artifact collection must be short-circuited")

    monkeypatch.setattr(engine, "collect_candidate_artifact", forbidden)
    scope, ablation, artifact = subject._inspect_candidate_source(state, node, job)
    assert scope.valid and ablation.error_message == "reversion first"
    assert artifact is None


def test_later_exception_keeps_original_publication_order(tmp_path, monkeypatch):
    case = make_round(tmp_path / "run")
    reverse_checks(monkeypatch, engine)
    collect = engine.collect_candidate_artifact

    def fail_second(**kwargs):
        if kwargs["workspace"].candidate_id == "node_0001":
            raise OSError("unhandled source read failure")
        return collect(**kwargs)

    monkeypatch.setattr(engine, "collect_candidate_artifact", fail_second)
    with pytest.raises(OSError, match="unhandled source read failure"):
        evaluate_round(case)
    jobs = case[4]
    assert jobs["node_0000"]["stage_journal"].record("static_check")["status"] == "complete"
    assert jobs["node_0001"]["stage_journal"].record("static_check")["status"] == "running"
    assert jobs["node_0002"]["stage_journal"].record("static_check") is None
    assert case[-1] == []


@pytest.mark.parametrize("failure", ["usage", "missing_validation", "excess_candidates"])
def test_closure_gates_precede_source_checks(tmp_path, monkeypatch, failure):
    subject, state, plan, nodes, jobs, path, dispatched = make_round(tmp_path / "run")
    before = state.to_dict()
    monkeypatch.setattr(engine, "_model_usage_fingerprint", lambda *args: "usage")
    monkeypatch.setattr(engine, "_model_token_usage_complete", lambda *args: failure != "usage")
    monkeypatch.setattr(engine, "_model_call_attempts", lambda *args: 0)
    monkeypatch.setattr(engine, "_model_token_usage", lambda *args: 0)

    def forbidden(*args):
        raise AssertionError("source inspection started before the closure gates passed")

    monkeypatch.setattr(subject, "_inspect_candidate_source", forbidden)
    messages = {"usage": "complete model token usage", "missing_validation": "locked budget validation",
                "excess_candidates": "exceeds the candidates budget"}
    if failure == "excess_candidates":
        planned = {"candidates": subject.config.resolved_candidate_budget + 1,
                   "candidate_evaluation_units": 0, "diagnostic_evaluation_units": 0, "model_calls": 0}
        monkeypatch.setattr(engine, "_planned_budget", lambda *args: planned)
        (path / "decision_validation.json").write_text(json.dumps({"planned_budget": planned,
            "budget_usage_before": dict.fromkeys(planned, 0), "model_calls_before": 0}), encoding="utf-8")
    with pytest.raises(RuntimeError, match=messages[failure]):
        subject._close_round(state, round_index=0, round_path=path, plan=plan, jobs=jobs,
                             nodes=nodes, retests=(), generation_errors={})
    assert state.to_dict() == before
    assert not dispatched
    assert all(job["stage_journal"].record("static_check") is None for job in jobs.values())


def reference_module():
    """Optionally audit against a frozen pre-edit engine, otherwise serial scheduling."""
    source = os.environ.get("FROZEN_0921_ENGINE")
    if not source:
        return engine
    spec = importlib.util.spec_from_file_location("frozen_0921_engine", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("recovery", ["fresh", "cache", "receipt", "ambiguous", "generation_error"])
def test_round_matches_serial_reference(tmp_path, recovery):
    root = tmp_path / "run"

    def run(module, workers):
        case = make_round(root, module, workers=workers)
        error = None
        if recovery in {"cache", "receipt", "ambiguous"}:
            evaluate_round(case)
            for name in ("evaluation_receipt.json", "evaluation_attempt.json", "evaluations.json"):
                path = case[5] / name
                if (recovery == "cache" or (recovery == "ambiguous" and name != "evaluation_attempt.json")):
                    path.unlink(missing_ok=True)
            if recovery in {"receipt", "ambiguous"}:
                # Restore the unscored state while retaining the completed stage ledger.
                case[1].evaluations.clear()
                for node in case[3]:
                    node.evaluation_ids.clear()
                    node.status, node.valid, node.score = "pending", None, None
                shutil.rmtree(root / "evaluation_cache")
        try:
            evaluate_round(case, errors={"node_0001": "generation failed"} if recovery == "generation_error" else None)
        except module.TransactionAmbiguityError as exc:
            error = (type(exc).__name__, str(exc))
        snapshot = {str(p.relative_to(root)): json.loads(p.read_text(encoding="utf-8"))
            for p in root.rglob("*.json")}
        state = case[1].to_dict()
        return state, snapshot, case[-1], error

    expected = run(reference_module(), 1)
    assert root.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(root)
    actual = run(engine, 3)
    assert actual == expected


def make_manifest(root):
    rows = []
    for index in range(3):
        source = root / f"instance_{index}.npz"
        np.savez(source, processing_times=np.arange(6).reshape(3, 2) + index)
        rows.append({"id": f"unit_{index}", "path": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "reference": 10.0,
            "reference_kind": "best_known", "family": "3_2", "source": "synthetic"})
    entry = EntryContract("SearchStrategy", "solve",
        ("self", "processing_times", "deadline", "iteration_limit", "seed"))
    manifest = {"schema": "task_dataset_v1", "kind": "fssp", "split": "train",
        "contract_id": "offline", "entry_contract": asdict(entry), "instances": rows}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, tuple(row["id"] for row in rows)


def reverse_hashes(monkeypatch, *, corrupt=()):
    original = manifest_evaluator.sha256_file
    barrier = threading.Barrier(3, timeout=5)
    gates = [threading.Event() for _ in range(3)]
    gates[-1].set()
    completed = []

    def check(source):
        index = int(Path(source).stem.split("_")[-1])
        barrier.wait()
        assert gates[index].wait(5), "hash checks did not overlap"
        result = original(source)
        completed.append(index)
        if index:
            gates[index - 1].set()
        return "0" * 64 if index in corrupt else result

    monkeypatch.setattr(manifest_evaluator, "sha256_file", check)
    return completed


def test_input_hashes_overlap_without_reordering(tmp_path, monkeypatch):
    path, ids = make_manifest(tmp_path)
    with monkeypatch.context() as patches:
        completed = reverse_hashes(patches)
        manifest, inputs, rows = manifest_evaluator._load_inputs(path, ids, max_workers=3)
    assert completed == [2, 1, 0]
    assert [row["id"] for row in rows] == list(ids)
    assert [item.record["id"] for item in inputs] == list(ids)
    for index, item in enumerate(inputs):
        np.testing.assert_array_equal(item.materialize(), np.arange(6).reshape(3, 2) + index)
    assert manifest["kind"] == "fssp"


@pytest.mark.parametrize("corrupt", [(), (0, 2)])
def test_batch_preflight_finishes_before_any_solver(tmp_path, monkeypatch, corrupt):
    path, ids = make_manifest(tmp_path)
    completed = reverse_hashes(monkeypatch, corrupt=corrupt)
    calls, loaded_objects = [], []

    def evaluate(candidate_id, candidate_path, budget, timeout, split, loaded, diagnostics):
        assert completed == [2, 1, 0]
        calls.append(candidate_id)
        loaded_objects.append(loaded)
        return candidate_id

    monkeypatch.setattr(manifest_evaluator, "_evaluate_loaded", evaluate)
    tasks = [CandidateTask(f"candidate_{i}", tmp_path / "unused.py", "unused", "unused",
        EvaluationBudget(n_instances=3, time_limit=1, ite_max=1),
        instance_ids=ids, data_path=path) for i in range(3)]
    if corrupt:
        with pytest.raises(ValueError, match="hash changed: unit_0"):
            manifest_evaluator.evaluate_task_batch(tasks, max_workers=3)
        assert calls == []
    else:
        assert manifest_evaluator.evaluate_task_batch(tasks, max_workers=3) == [task.candidate_id for task in tasks]
        assert len(loaded_objects) == 3
        assert all(item is loaded_objects[0] for item in loaded_objects)


def test_default_input_preflight_stays_on_caller_thread(tmp_path, monkeypatch):
    path, ids = make_manifest(tmp_path)
    original = manifest_evaluator.sha256_file
    owner = threading.get_ident()
    checked = []

    def check(source):
        assert threading.get_ident() == owner
        checked.append(Path(source).stem)
        return original(source)

    monkeypatch.setattr(manifest_evaluator, "sha256_file", check)
    manifest_evaluator._load_inputs(path, ids)
    assert checked == ["instance_0", "instance_1", "instance_2"]


def test_input_hash_is_rechecked_at_actual_use(tmp_path):
    path, ids = make_manifest(tmp_path)
    _, inputs, _ = manifest_evaluator._load_inputs(path, ids, max_workers=3)
    Path(inputs[0].source_path).write_bytes(b"changed after preflight")
    with pytest.raises(ValueError, match="hash changed: unit_0"):
        inputs[0].materialize()


def test_parallel_input_values_match_serial(tmp_path):
    path, ids = make_manifest(tmp_path)
    expected = manifest_evaluator._load_inputs(path, ids[::-1])
    actual = manifest_evaluator._load_inputs(path, ids[::-1], max_workers=3)
    assert actual[0] == expected[0]
    assert actual[2] == expected[2]
    assert actual[1] == expected[1]
    for left, right in zip(actual[1], expected[1]):
        np.testing.assert_array_equal(left.materialize(), right.materialize())
