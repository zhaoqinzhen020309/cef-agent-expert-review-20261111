from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace as NS
import hashlib
import json

import pytest
from agent_ahd.core.baseline_parent import ensure_baseline_parent, SEED_ID
from agent_ahd.core.engine import _budget_usage, _verify_node_sources, PlannerEngineConfig
from agent_ahd.core.search_state import SearchState, SearchConfig, load_search_state, save_search_state
from agent_ahd.core.research_records import EvaluationRecord, SearchAction, ScopeSpec
from agent_ahd.core.decision_catalog import build_decision_catalog


def example(tmp_path):
    target = tmp_path / "controls/baseline/solver.py"
    target.parent.mkdir(parents=True)
    target.write_text("class SearchStrategy:\n    def solve(self, x):\n        return x\n")
    state = SearchState.create(run_id="test", baseline_path=str(target),
        baseline_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), config=SearchConfig())
    state.evaluations.append(EvaluationRecord("eval_baseline_control_000", "__baseline_control__",
        "train", (7,), ("a", "b"), ((1., 3.),), ((.1, .2),), True))
    return state


def test_seed_becomes_editable_parent_without_charging_or_counting_proposal(tmp_path):
    state = example(tmp_path)
    assert ensure_baseline_parent(state, tmp_path, seeds=(7,), instance_ids=("a", "b"))
    node = state.require_node(SEED_ID)
    assert node.score == 2 and node.round_index == -1
    assert state.next_node_index == 0 and state.pending_node_ids == []
    usage = _budget_usage(state, NS(), PlannerEngineConfig(), run_dir=tmp_path)
    assert usage["candidates"] == usage["evaluation_units"] == 0
    catalog = build_decision_catalog(round_index=0, available_actions=(SearchAction.EXPLOIT,),
        observations={"profile:seed": {"kind": "candidate_profile", "statement": "Measured training profile",
                      "values": {"candidate_id": SEED_ID, "mean_gap": 2.}}}, failure_clusters={}, evidence=(),
        candidate_scopes={SEED_ID: ScopeSpec("full_solver")}, candidate_source_structures={})
    assert any(o.decision.primary_parent_id == SEED_ID for o in catalog.candidate_options)
    _verify_node_sources(tmp_path, state)
    receipt = json.loads((tmp_path / "baseline_parent.json").read_text())
    assert receipt["evaluation_reused"] and receipt["new_candidate_proposals"] == 0


def test_seed_survives_state_resume_and_rejects_source_tampering(tmp_path):
    state = example(tmp_path)
    ensure_baseline_parent(state, tmp_path, seeds=(7,), instance_ids=("a", "b"))
    save_search_state(tmp_path / "search_state.json", state)
    resumed = load_search_state(tmp_path / "search_state.json")
    assert not ensure_baseline_parent(resumed, tmp_path, seeds=(7,), instance_ids=("a", "b"))
    assert len(resumed.nodes) == 1 and len(resumed.evaluations) == 2
    (tmp_path / resumed.nodes[0].solver_relpath).write_text("changed = True")
    with pytest.raises(RuntimeError, match="copy changed"):
        ensure_baseline_parent(resumed, tmp_path, seeds=(7,), instance_ids=("a", "b"))


def test_unmatched_training_receipt_cannot_seed_search(tmp_path):
    state = example(tmp_path)
    with pytest.raises(RuntimeError, match="matching training"):
        ensure_baseline_parent(state, tmp_path, seeds=(99,), instance_ids=("a", "b"))
    assert not state.nodes
    assert not (tmp_path / "baseline_parent.json").exists()


def test_actual_seed_retest_is_charged_and_does_not_break_resume(tmp_path):
    state = example(tmp_path)
    ensure_baseline_parent(state, tmp_path, seeds=(7,), instance_ids=("a", "b"))
    seed_evaluation = state.evaluations[-1]
    retest = replace(seed_evaluation, evaluation_id="retest_seed_baseline_001", seeds=(8,))
    state.record_evaluation_record(retest, update_node=False)
    assert not ensure_baseline_parent(state, tmp_path, seeds=(7,), instance_ids=("a", "b"))
    usage = _budget_usage(state, NS(), PlannerEngineConfig(enable_baseline_parent=True), run_dir=tmp_path)
    assert usage["candidates"] == usage["candidate_evaluation_units"] == 0
    assert usage["diagnostic_evaluation_units"] == usage["evaluation_units"] == 2
