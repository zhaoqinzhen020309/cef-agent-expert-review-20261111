import ast
from scripts.comparator_contract import make_template, describe_task


def test_function_only_prompt_retains_public_budget_and_range(tmp_path):
    source = tmp_path / "baseline.py"
    source.write_text("class SearchStrategy:\n    def solve(self, x, deadline, iterations, seed):\n        return x\n")
    protocol = {"task": {"description": "Return a feasible result.", "baseline_path": str(source),
                         "operating_spec": {"max_input_size": 1001},
                         "entry_contract": {"class_name": "SearchStrategy", "method_name": "solve",
                                            "parameters": ["self", "x", "deadline", "iterations", "seed"]}}}
    program = make_template(protocol, 10)
    function = ast.parse(program).body[0]
    assert ast.get_docstring(function) == describe_task(protocol, 10)
    assert "1001" in ast.get_docstring(function)
    assert "10 seconds" in ast.get_docstring(function)
    namespace = {}
    exec(program, namespace)
    assert namespace["design_algorithm"]([1, 2], 999, 5, 1) == [1, 2]
