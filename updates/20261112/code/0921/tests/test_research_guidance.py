from types import SimpleNamespace as NS
import json

import pytest

from agent_ahd.core.research_guidance import retrieve_training_episodes, guidance_for_candidate, resource_facts
from agent_ahd.core.research_records import EvaluationRecord
from agent_ahd.core.engine import PlannerEngineConfig
from agent_ahd.agents.candidate_agent import build_code_implementation_task
from agent_ahd.core.prompt_projection import PROMPT_PROJECTION_VERSION, compact_repair_context


def fixture():
    config = NS(n_instances=2, base_seeds=(1,), time_limit=2, ite_max=100)
    adapter = NS(train_manifest=NS(instance_ids=("a", "b")),
                 dataset_protocol={"task": {"operating_spec": {"max_input_size": 200}}})
    def evaluation(name, gap, split="train", seeds=(1,), ids=("a", "b")):
        return EvaluationRecord("eval_" + name, name, split, seeds, ids, ((gap, gap),), ((.1, .2),), True)
    nodes, hypotheses, evaluations = [], [], [evaluation("__baseline_control__", 10)]
    for i in range(20):
        identifier = f"node_{i:04d}"
        nodes.append(NS(node_id=identifier, round_index=i, valid=True, primary_parent_id=None,
                        hypothesis_id=identifier, solver_sha256=f"sha{i}", error_type=None))
        hypotheses.append(NS(hypothesis_id=identifier, planned_change="hypothesis_" + str(i)))
        evaluations.append(evaluation(identifier, 8 if i == 0 else 11 + i))
    evaluations.append(evaluation("node_0000", -999, split="validation"))
    evaluations.append(evaluation("node_0000", -888, seeds=(99,)))
    state = NS(nodes=nodes, hypotheses=hypotheses, evaluations=evaluations)
    return state, config, adapter


def test_memory_is_bounded_and_preserves_best_and_recent_failure():
    state, config, adapter = fixture()
    memory = retrieve_training_episodes(state, config, adapter, before_round=20)
    assert len(memory["episodes"]) == 6
    assert {"node_0000", "node_0019"} <= {e["candidate_id"] for e in memory["episodes"]}
    best = memory["episodes"][0]
    assert best["mean_gap_percent"] == 8
    assert best["paired_mean_delta"] == -2
    assert best["baseline_mean_delta"] == -2 and best["beats_supplied_baseline"] is True
    assert best["failed_hypothesis_untrusted"] is None
    assert best["causal_mechanism_status"] != "supported"


def test_memory_excludes_validation_retests_and_current_batch():
    state, config, adapter = fixture()
    memory = retrieve_training_episodes(state, config, adapter, before_round=1)
    assert len(memory["episodes"]) == 1
    serialized = json.dumps(memory)
    assert "999" not in serialized and "888" not in serialized and "node_0001" not in serialized


def test_resource_contract_uses_declared_spec_not_validation_manifest():
    _, config, adapter = fixture()
    assert resource_facts(adapter, config)["single_instance_seed_seconds"] == 2
    assert resource_facts(adapter, config)["declared_operating_spec"] == {"max_input_size": 200}


def test_guidance_and_resource_facts_survive_implementation_and_repair():
    state, config, adapter = fixture()
    memory = retrieve_training_episodes(state, config, adapter, before_round=20)
    guidance = guidance_for_candidate(memory, action="EXPLOIT", slot=1, stagnation=2)
    payload = {"projection_version": PROMPT_PROJECTION_VERSION, "role": "hypothesis",
               "resource_facts": resource_facts(adapter, config), "research_guidance": guidance,
               "problem_facts": {}, "source_bundle": {}, "edit_contract": {}}
    task = build_code_implementation_task(json.dumps(payload), None)
    result = json.loads(task)
    assert result["resource_facts"] == payload["resource_facts"]
    assert "preserve_measured_strength" in result["implementation_guidance"]["skills"]
    repaired = compact_repair_context("implementation", task)
    assert repaired["resource_facts"] == payload["resource_facts"]
    assert "implementation_guidance" in repaired


def test_guidance_cannot_contaminate_legacy_scope_or_off_arms():
    with pytest.raises(ValueError, match="separate framework-on"):
        PlannerEngineConfig(enable_research_guidance=True, framework_mode="off")
