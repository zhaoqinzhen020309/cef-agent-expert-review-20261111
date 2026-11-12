"""Compare an explicit symmetric tie heuristic, not a paper-table reproduction."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

TIE_KERNEL = '''
@njit
def guided_position(costs, count, excluded, job, p, forward, backward):
    value = np.min(costs[:count])
    best, best_secondary, ties = 0, np.inf, 0
    for i in range(count):
        if costs[i] != value:
            continue
        head, tail, secondary = 0.0, 0.0, 0.0
        for k in range(p.shape[1]):
            head = max(head, forward[i, k + 1]) + p[job, k]
            secondary += head
        for k in range(p.shape[1] - 1, -1, -1):
            tail = max(tail, backward[i, k]) + p[job, k]
            secondary += tail
        if secondary < best_secondary:
            best, best_secondary, ties = i, secondary, 1
        elif secondary == best_secondary and i != excluded:
            ties += 1
            if best == excluded or np.random.randint(ties) == 0:
                best = i
    return best, value

'''


def main():
    root = ROOT / "runs/minimal_fssp_tiebreak_v1"
    if root.exists():
        raise FileExistsError(root)
    protocol = ROOT / "data/bundles/minimal_v1/fssp/protocol.json"
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    original = (ROOT / "problems/fssp/engineering_controls/partial_neutral_ig.py").read_text()
    base = original.replace("@njit\ndef local_search", TIE_KERNEL + "\n@njit\ndef local_search")
    construction = base.replace("best_position(costs, count + 1, -1)",
                                "guided_position(costs, count + 1, -1, job, p, forward, backward)")
    all_ties = construction.replace("best_position(costs, count, old)",
                                   "guided_position(costs, count, old, job, p, forward, backward)")
    variants = {"random_ties": original, "symmetric_reconstruction": construction, "symmetric_all": all_ties}
    spec = {"problem": "fssp", "protocol": str(protocol), "protocol_sha256": sha(protocol),
            "evidence_type": "human_engineering_training_only_selection_on_reused_minimal_data",
            "budget": {"n_instances": 3, "time_limit": 30, "ite_max": 10000000, "seeds": [71400, 71401]},
            "selection_rule": "minimum_training_mean_then_name", "sources": {},
            "hypothesis": "Among equal makespans, minimize sum of forward and reversed completion estimates, preserving reversal symmetry and O(n*m) workspace.",
            "inspiration_not_exact_reproduction": "https://doi.org/10.1016/j.ejor.2021.05.014"}
    for name, code in variants.items():
        path = root / "sources" / name / "solver.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        spec["sources"][name] = {"path": str(path), "sha256": sha(path)}
    (root / "registry.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(root / "registry.json")


if __name__ == "__main__":
    main()
