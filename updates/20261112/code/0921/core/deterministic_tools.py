"""Deterministic implementations of exact operations selected by the planner."""
from pathlib import Path, PurePosixPath
import hashlib
import json

from agent_ahd.agents.agent_runner import AgentResponse

EXACT_ABLATION_TOOL = "tool:restore_registered_full_ablation_v1"


def exact_ablation_response(task, decision, source_root):
    """Restore a registered full-solver parent verbatim, with no model call.

Narrow symbol interventions deliberately use the existing scoped path. This
tool never broadens a scope or promotes an observation to causal evidence.
The normal source, scope and ablation checks still run after publication.
"""
    if (decision.action.value != "ABLATE" or decision.scope.kind != "full_solver"
        or decision.inheritance_channel.value != "full_source"):
        return None
    content = json.loads(task)
    source = content["experiment"]["auxiliary_mechanism"]["mechanism_off_source"]
    if not isinstance(source, dict) or "solver.py" not in source:
        raise ValueError("exact ablation requires the registered mechanism-off source")
    for name, text in source.items():
        path = PurePosixPath(name)
        if (not isinstance(text, str) or path.is_absolute() or ".." in path.parts
            or "\\" in name or ":" in name or path.suffix != ".py"):
            raise ValueError("invalid registered ablation source path or content")
    present = {p.relative_to(source_root).as_posix() for p in Path(source_root).rglob("*.py") if "__pycache__" not in p.parts}
    payload = {"solver_py": source["solver.py"],
               "helper_files": [{"path": name, "content": text} for name, text in sorted(source.items()) if name != "solver.py"],
               "delete_files": sorted(present - source.keys())}
    source_hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sorted(source.items())}
    return AgentResponse(payload, json.dumps(payload), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                         "execution_backend": "deterministic_tool", "tool_name": EXACT_ABLATION_TOOL,
                         "restored_source_sha256": source_hashes})
