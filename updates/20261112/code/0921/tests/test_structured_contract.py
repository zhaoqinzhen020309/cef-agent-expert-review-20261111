import numpy as np
import pytest

from agent_ahd.core.eoh_timeout import execute_candidate_with_timeout
from agent_ahd.core.solution_contract import normalize_structured_solution
from agent_ahd.problems.base import EntryContract, EvaluationBudget


def test_subset_output_survives_real_worker_and_input_is_not_mutated(tmp_path):
    source = tmp_path / "solver.py"
    source.write_text('''import numpy as np
class SubsetSolver:
    def select(self, instance, deadline, iteration_limit, seed):
        selected = np.flatnonzero(instance["weights"] <= instance["capacity"])
        instance["weights"][:] = 0
        return {"selected": selected[:1], "metadata": {"seed": seed}}
''')
    instance = {"weights": np.array([8, 3, 4]), "capacity": 5}
    contract = EntryContract("SubsetSolver", "select", ("self", "instance", "deadline", "iteration_limit", "seed"),
                             input_kind="mapping", output_kind="structured")
    result = execute_candidate_with_timeout(candidate_id="custom_subset", candidate_path=source, instances=[instance],
                 budget=EvaluationBudget(n_instances=1, time_limit=2, seeds=(22, 23)), timeout=30, entry_contract=contract)
    assert result.valid, result.error_message
    assert result.routes_by_seed == [[{"selected": [1], "metadata": {"seed": 22}}],
                                    [{"selected": [1], "metadata": {"seed": 23}}]]
    np.testing.assert_array_equal(instance["weights"], [8, 3, 4])
    # Feasibility is a separate trusted-evaluator decision, not serialization.
    for row in result.routes_by_seed:
        assert instance["weights"][row[0]["selected"]].sum() <= instance["capacity"]


def test_structured_transport_rejects_nonfinite_objects_and_excessive_outputs():
    for output in ({"x": np.nan}, {1: "bad key"}, {"x": object()}, np.array([object()]), complex(1, 2)):
        with pytest.raises(ValueError):
            normalize_structured_solution(output)
    with pytest.raises(ValueError, match="limits"):
        normalize_structured_solution({"values": list(range(10))}, max_items=5)
    with pytest.raises(ValueError, match="limits"):
        normalize_structured_solution([[[1]]], max_depth=2)
    circular = []
    circular.append(circular)
    with pytest.raises(ValueError, match="limits"):
        normalize_structured_solution(circular)
