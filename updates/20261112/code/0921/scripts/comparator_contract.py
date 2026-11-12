"""Put the complete public contract in both task text and function-only prompts."""
from pathlib import Path
import json
import textwrap


def describe_task(protocol, seconds):
    task = protocol["task"]
    return (task["description"] + "\nPublic operating range: " + json.dumps(task.get("operating_spec", {}))
            + f"\nEach complete instance has {seconds} seconds including initialization, input copying and all policy calls."
            + " Use only standard library and NumPy; place helper functions and imports inside the evolved function."
            + " Never read files or network. A bounded implementation must return before the absolute deadline.")


def make_template(protocol, seconds):
    contract = protocol["task"]["entry_contract"]
    params = contract["parameters"][1:]
    source = Path(protocol["task"]["baseline_path"]).read_text(encoding="utf-8")
    # A literal first expression is a docstring, including when json.dumps uses
    # escaped newlines. Upstream preserves it in the FunSearch versioned prompt.
    docstring = "    " + json.dumps(describe_task(protocol, seconds))
    body = textwrap.indent(source, "    ")
    call = f"{contract['class_name']}().{contract['method_name']}({', '.join(params)})"
    return f"def design_algorithm({', '.join(params)}):\n{docstring}\n{body}\n    return {call}\n"
