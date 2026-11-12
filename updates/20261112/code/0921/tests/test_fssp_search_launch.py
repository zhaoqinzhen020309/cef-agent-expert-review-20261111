"""Launch acceptance with generated data and an offline provider only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "problems/fssp/neh_gls_eoh_rng_v1/solver.py"
PREPARER = ROOT / "scripts/prepare_fssp_search.py"


def load_source(path, name):
    assert path.is_file(), str(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_protocol_has_all_64_paper_style_inputs(tmp_path):
    module = load_source(PREPARER, "prepare_fssp_search_test")
    path = module.prepare_training_protocol(tmp_path / "data", solver_path=SOLVER)
    protocol = json.loads(path.read_text())
    assert protocol["training_only"] is True
    assert protocol["validation"] == {"source_path": None, "instance_ids": []}
    assert protocol["metric"] == {"name": "makespan", "unit": "processing_time", "aggregation": "mean", "direction": "minimize"}
    manifest = json.loads(Path(protocol["train"]["source_path"]).read_text())
    assert len(manifest["instances"]) == 64
    assert protocol["train"]["instance_ids"] == [f"train_{i:04d}" for i in range(64)]
    rng = np.random.Generator(np.random.PCG64(20261024))
    for row in manifest["instances"]:
        machines = int(rng.integers(2, 21))
        expected = rng.uniform(0.0, 1.0, size=(50, machines))
        source = Path(row["path"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
        with np.load(source, allow_pickle=False) as payload:
            matrix = payload["processing_times"]
        np.testing.assert_array_equal(matrix, expected)
        assert matrix.dtype == np.float64
        assert "reference" not in row and "reference_kind" not in row
    assert protocol["metadata"]["data_seed"] == 20261024


def test_preparer_does_not_overwrite_an_existing_directory(tmp_path):
    module = load_source(PREPARER, "prepare_fssp_no_overwrite_test")
    target = tmp_path / "keep"
    target.mkdir()
    (target / "notes.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        module.prepare_training_protocol(target, solver_path=SOLVER)
    assert (target / "notes.txt").read_text() == "keep"
    assert list(target.iterdir()) == [target / "notes.txt"]


def test_eoh_rng_solver_seeds_python_without_resetting_numpy(monkeypatch):
    module = load_source(SOLVER, "fssp_eoh_rng_test")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        np.random.seed(711)
        before = np.random.get_state()
        original = random.seed
        calls = []

        def seed(value):
            calls.append(value)
            return original(value)

        def forbidden(*args, **kwargs):
            raise AssertionError("EOH adapter must not add a per-instance NumPy reset")

        with monkeypatch.context() as patches:
            patches.setattr(module.random, "seed", seed)
            patches.setattr(module.np.random, "seed", forbidden)
            for _ in range(2):
                solution = module.SearchStrategy().solve(np.array([[2., 3.], [4., 1.]]), time.time()+10, 0, 2024)
                assert sorted(solution) == [0, 1]
                assert random.random() == random.Random(2024).random()
        assert calls == [2024, 2024]
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_offline_provider_real_round_checkpoint_and_next_feedback(tmp_path):
    from test_raw_objective_search import raw_protocol
    from agent_ahd.agents.agent_runner import AgentResponse
    from agent_ahd.core.agent_protocol import wrap_agent_runner
    from agent_ahd.core.catalog_encoding import decode_catalog
    from agent_ahd.core.engine import PlannerEngineConfig, PlannerSearchEngine, initialize_planner_run
    from agent_ahd.core.search_state import load_search_state
    from agent_ahd.problems.manifest_adapter import ManifestAdapter

    task_root = tmp_path / "task"
    task_root.mkdir()
    protocol, manifest_path, _ = raw_protocol(task_root)
    protocol["task"]["entry_contract"]["python_random_seed_policy"] = "reset_per_instance"
    manifest = json.loads(manifest_path.read_text())
    manifest["entry_contract"] = protocol["task"]["entry_contract"]
    manifest_path.write_text(json.dumps(manifest))
    adapter = ManifestAdapter("fssp", protocol)
    run = tmp_path / "run"
    config = PlannerEngineConfig(problem_name="fssp", experiment_phase="benchmark", protocol_mode="v4",
        controller_stop_policy="fixed_rounds", total_rounds=2, batch_size=1, candidate_budget=2,
        model_call_budget=6, model_token_budget=200000, wall_clock_budget_sec=120,
        n_instances=1, time_limit=2, ite_max=1, agent_workers=2, evaluation_workers=2,
        enable_baseline_parent=True, enable_research_guidance=True, agent_call_max_tokens=4000,
        agent_call_max_wall_clock_sec=10)
    initialize_planner_run(run, adapter=adapter, config=config)
    requests = []

    class Provider:
        model = "offline-raw-objective-provider"

        def complete_json(self, **kwargs):
            requests.append(kwargs)
            context = json.loads(kwargs["user_prompt"])
            schema = kwargs["output_schema"]["properties"]
            if "candidate_option_ids" in schema:
                catalog = decode_catalog(context["decision_catalog"])
                desired = "seed_baseline" if context["round_index"] == 0 else "node_0000"
                options = [o for o in catalog["candidate_options"] if o["primary_parent_id"] == desired
                           and o["scope"]["kind"] == "full_solver" and o["action"] == "EXPLOIT"]
                assert options, f"expected parent unavailable: {desired}"
                payload = {"round_index": context["round_index"], "candidate_option_ids": [options[0]["option_id"]],
                    "diagnostic_option_ids": [], "stop": False, "stop_reason": None}
            elif "hypothesis" in schema:
                payload = {"hypothesis": {"target_problem": "offline fixture", "planned_change": "Change job ordering.",
                    "predicted_effects": {"mean_score": "decrease"}, "regression_risks": ["fixture only"],
                    "ablation_scope": {"kind": "full_solver", "symbols": []}}}
            else:
                source = context["source_bundle"]["solver.py"]
                if "return [0, 1, 2]" in source:
                    source = source.replace("return [0, 1, 2]", "return [2, 1, 0]")
                else:
                    assert "return [2, 1, 0]" in source
                    source = source.replace("return [2, 1, 0]", "return [1, 0, 2]")
                payload = {"solver_py": source, "helper_files": [], "delete_files": []}
            return AgentResponse(payload, json.dumps(payload), {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20})

    def subject():
        controller = wrap_agent_runner(run_dir=run, runner=Provider(), config=config, role="controller")
        code = wrap_agent_runner(run_dir=run, runner=Provider(), config=config, role="code")
        return PlannerSearchEngine(run_dir=run, adapter=adapter, config=config,
                                   controller_runner=controller, code_runner=code)

    first = subject()
    first._ensure_model_protocol()
    first._ensure_baseline_control(first.load_state())
    assert first.run_round()
    checkpoint = load_search_state(run / "search_state.json")
    assert checkpoint.current_round == 0
    assert len(requests) == 3
    resumed = subject().run()
    assert resumed.current_round == 1 and resumed.completed_rounds == [0, 1]
    assert len(requests) == 6
    assert len([n for n in resumed.nodes if n.round_index >= 0]) == 2
    assert all(r.split == "train" and r.metric_name == "makespan" and not r.gap_matrix for r in resumed.evaluations)
    second_controller = json.loads(requests[3]["user_prompt"])
    memory = second_controller["training_episode_memory"]
    assert memory["episodes"][0]["candidate_id"] == "node_0000"
    expected = next(r.mean_score for r in resumed.evaluations if r.candidate_id == "node_0000")
    assert memory["episodes"][0]["mean_score"] == expected
    assert "mean_gap" not in json.dumps(second_controller)
    second_hypothesis = json.loads(requests[4]["user_prompt"])
    assert "return [2, 1, 0]" in second_hypothesis["source_bundle"]["solver.py"]
    assert "processing_time" in json.dumps(second_hypothesis)
    subject().run()
    assert len(requests) == 6, "completed search was reissued on resume"


def test_host_resets_python_seed_for_every_candidate_instance(tmp_path):
    from agent_ahd.core.eoh_timeout import execute_candidate_with_timeout
    from agent_ahd.problems.base import EntryContract, EvaluationBudget
    source = tmp_path / "solver.py"
    source.write_text("import random\nclass SearchStrategy:\n"
        "    def solve(self, processing_times, deadline, iteration_limit, seed):\n"
        "        order = list(range(len(processing_times)))\n"
        "        random.shuffle(order)\n        return order\n")
    entry = EntryContract("SearchStrategy", "solve", ("self", "processing_times", "deadline", "iteration_limit", "seed"),
                          python_random_seed_policy="reset_per_instance")
    expected = list(range(8))
    random.Random(2024).shuffle(expected)
    result = execute_candidate_with_timeout(candidate_id="no_self_seed", candidate_path=source,
        instances=[np.ones((8, 2)), np.ones((8, 2))],
        budget=EvaluationBudget(n_instances=2, time_limit=2, ite_max=1, seeds=(2024,)), timeout=15,
        entry_contract=entry, collect_diagnostics=False)
    assert result.valid, result.error_message
    assert result.routes_by_seed == [[expected, expected]]


def test_host_keeps_numpy_stream_between_instance_calls(tmp_path):
    from agent_ahd.core.eoh_timeout import execute_candidate_with_timeout
    from agent_ahd.problems.base import EntryContract, EvaluationBudget
    source = tmp_path / "solver.py"
    source.write_text("import numpy as np\nnp.random.seed(713)\nclass SearchStrategy:\n"
        "    def solve(self, processing_times, deadline, iteration_limit, seed):\n"
        "        return np.random.permutation(len(processing_times)).tolist()\n")
    entry = EntryContract("SearchStrategy", "solve", ("self", "processing_times", "deadline", "iteration_limit", "seed"),
                          python_random_seed_policy="reset_per_instance")
    rng = np.random.RandomState(713)
    expected = [rng.permutation(8).tolist(), rng.permutation(8).tolist()]
    result = execute_candidate_with_timeout(candidate_id="numpy_stream", candidate_path=source,
        instances=[np.ones((8, 2)), np.ones((8, 2))],
        budget=EvaluationBudget(n_instances=2, time_limit=2, ite_max=1, seeds=(2024,)), timeout=15,
        entry_contract=entry, collect_diagnostics=False)
    assert result.valid, result.error_message
    assert result.routes_by_seed == [expected]
