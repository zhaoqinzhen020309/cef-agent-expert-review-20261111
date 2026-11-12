"""Register a bounded training-only comparison before evaluating any variant."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / "runs/minimal_fssp_partial_v1"
    if root.exists():
        raise FileExistsError(root)
    protocol = ROOT / "data/bundles/minimal_v1/fssp/protocol.json"
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    base = (ROOT / "problems/fssp/engineering_controls/partial_neutral_ig.py").read_text()
    variants = {
        "baseline_blocks": (ROOT / "problems/fssp/minimal_refinement_seed_v1/solver.py").read_text(),
        "partial_strict": base.replace("    NEUTRAL_MOVES = True", "    NEUTRAL_MOVES = False"),
        "partial_neutral": base,
    }
    spec = {"problem": "fssp", "protocol": str(protocol), "protocol_sha256": sha(protocol),
            "evidence_type": "human_engineering_training_only_selection_on_reused_minimal_data",
            "budget": {"n_instances": 3, "time_limit": 30, "ite_max": 10000000, "seeds": [71400, 71401]},
            "selection_rule": "minimum_training_mean_then_name", "sources": {},
            "hypothesis": "Optimize the retained partial permutation before reconstruction; compare strict descent with bounded equal-cost moves.",
            "reference": "https://www.sciencedirect.com/science/article/pii/S0305054819302096"}
    for name, code in variants.items():
        path = root / "sources" / name / "solver.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        spec["sources"][name] = {"path": str(path), "sha256": sha(path)}
    (root / "registry.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(root / "registry.json")


if __name__ == "__main__":
    main()
