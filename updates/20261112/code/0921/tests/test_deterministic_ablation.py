import json
from types import SimpleNamespace as NS
from pathlib import Path

import pytest
from agent_ahd.core.deterministic_tools import exact_ablation_response, EXACT_ABLATION_TOOL
from agent_ahd.core.code_checks import check_ablation_reversion
from agent_ahd.core.research_records import ScopeSpec, SearchAction, InheritanceChannel
from agent_ahd.agents.candidate_agent import apply_code_implementation_response, CandidateWorkspace
from agent_ahd.core.engine import PlannerSearchEngine
from agent_ahd.core.prompt_projection import PROMPT_PROJECTION_VERSION


def example(tmp_path):
    decision = NS(action=SearchAction.ABLATE, scope=ScopeSpec("full_solver"), inheritance_channel=InheritanceChannel.FULL_SOURCE)
    original = "class SearchStrategy:\n    def solve(self, x, deadline, iteration_limit, seed):\n        return x\n"
    altered = original.replace("return x", "return list(reversed(x))")
    parent, mechanism, candidate = [tmp_path / name for name in ("parent", "mechanism", "candidate")]
    for path in (parent, mechanism, candidate):
        path.mkdir()
    (parent / "solver.py").write_text(original)
    for path in (mechanism, candidate):
        (path / "solver.py").write_text(altered)
        (path / "new_helper.py").write_text("FLAG = 1\n")
    task = json.dumps({"projection_version": PROMPT_PROJECTION_VERSION, "role": "hypothesis",
                       "experiment": {"auxiliary_mechanism": {"mechanism_off_source": {"solver.py": original}}},
                       "source_bundle": {}, "edit_contract": {}, "problem_facts": {}})
    return decision, task, parent, mechanism, candidate


def test_exact_tool_restores_changes_and_removes_new_helper(tmp_path):
    decision, task, parent, mechanism, candidate = example(tmp_path)
    response = exact_ablation_response(task, decision, mechanism)
    assert response.usage["total_tokens"] == 0
    assert response.usage["tool_name"] == EXACT_ABLATION_TOOL
    apply_code_implementation_response(candidate, response, decision=decision)
    assert check_ablation_reversion(parent, mechanism, candidate, decision.scope).valid


def test_tool_never_broadens_a_scoped_intervention(tmp_path):
    decision, task, _, mechanism, _ = example(tmp_path)
    decision.scope = ScopeSpec("symbol_set", ("SearchStrategy.solve",))
    assert exact_ablation_response(task, decision, mechanism) is None


def test_tool_rejects_escaping_source_paths(tmp_path):
    decision, task, _, mechanism, _ = example(tmp_path)
    data = json.loads(task)
    data["experiment"]["auxiliary_mechanism"]["mechanism_off_source"]["../outside.py"] = "x = 1"
    with pytest.raises(ValueError, match="source path"):
        exact_ablation_response(json.dumps(data), decision, mechanism)


def test_engine_publishes_and_resumes_tool_commit_without_model_call(tmp_path):
    decision, task, _, mechanism, candidate = example(tmp_path)
    class NoModelCalls:
        model = "configured-but-unused"
        def complete_json(self, **kwargs):
            raise AssertionError("exact restoration must not call the model")
    engine = object.__new__(PlannerSearchEngine)
    engine.run_dir = tmp_path
    engine.config = NS(enable_deterministic_ablation=True)
    engine.code_runner = NoModelCalls()
    job = {"control_workspace": CandidateWorkspace("n", candidate, candidate / "solver.py"),
           "decision": decision, "task": task, "snapshot_root": mechanism, "node": NS(solver_relpath="candidate/solver.py")}
    first = engine._run_implementation_transaction("n", job)
    assert engine._run_implementation_transaction("n", job) == first
    record = json.loads((candidate / "agent_response.json").read_text())
    assert record["model"] == EXACT_ABLATION_TOOL
    assert record["usage"]["execution_backend"] == "deterministic_tool"
    assert not (tmp_path / "attempt_ledger.jsonl").exists()
