"""Predeclare a small training-only IG refinement grid; preserve all outcomes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / "runs/minimal_fssp_refinements_v1"
    protocol = ROOT / "data/bundles/minimal_v1/fssp/protocol.json"
    base = (ROOT / "problems/fssp/engineering_controls/compiled_ig.py").read_text()
    variants = {
        "destroy2": base.replace("DESTROY_SIZE = 4", "DESTROY_SIZE = 2"),
        "cooler": base.replace("TEMPERATURE_FACTOR = 0.4", "TEMPERATURE_FACTOR = 0.2"),
        "warmer": base.replace("TEMPERATURE_FACTOR = 0.4", "TEMPERATURE_FACTOR = 0.8"),
        "continuous_sweep": base.replace("                improved = True\n                break", "                improved = True"),
        "reverse_machines": base.replace("p = np.ascontiguousarray(processing_times, dtype=np.float64)", "p = np.ascontiguousarray(processing_times[:, ::-1], dtype=np.float64)").replace("return best.tolist()", "return best[::-1].tolist()"),
    }
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    registry = {"problem": "fssp", "protocol": str(protocol), "protocol_sha256": sha(protocol),
                "evidence_type": "human_engineering_training_grid_not_agent_search", "validation_read": False,
                "budget": {"n_instances": 3, "time_limit": 10, "ite_max": 100000, "seeds": [71001, 71002]}, "sources": {}}
    for name, text in variants.items():
        path = root / "sources" / name / "solver.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != text:
            raise ValueError("registered refinement source changed")
        path.write_text(text, encoding="utf-8")
        registry["sources"][name] = {"path": str(path), "sha256": sha(path)}
    path = root / "registry.json"
    if path.exists() and json.loads(path.read_text()) != registry:
        raise ValueError("refinement grid changed")
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
