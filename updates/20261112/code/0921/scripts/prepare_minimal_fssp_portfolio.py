"""Two reconstruction directions with separate incumbents under one time cap."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOLVE = '''    def solve(self, processing_times, deadline, iteration_limit, seed):
        original = np.ascontiguousarray(processing_times, dtype=np.float64)
        n, m = original.shape
        if n == 0 or m == 0:
            return list(range(n))
        orientations = (original, np.ascontiguousarray(original[:, ::-1]))
        forward, backward = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1))
        costs = np.empty(n + 1)
        currents, values = [], []
        for chain, p in enumerate(orientations):
            seq, val = initialize(p, (seed + 104729 * chain) % (2 ** 32), forward, backward, costs, True)
            currents.append(seq)
            values.append(val)
        best_chain = int(values[1] < values[0])
        best_value = values[best_chain]
        best = currents[best_chain].copy()
        rng = np.random.default_rng(seed)
        temperature = max(1e-9, 0.4 * original.mean() / 10)
        prefix = np.cumsum(original, axis=1) - original
        suffix = np.cumsum(original[:, ::-1], axis=1)[:, ::-1] - original
        lower_bound = max(original.sum(axis=1).max(), (original.sum(axis=0) + prefix.min(axis=0) + suffix.min(axis=0)).max())
        for iteration in range(iteration_limit):
            if time.time() >= deadline - 0.2 or best_value <= lower_bound:
                break
            chain = iteration % 2
            candidate, candidate_value = step(currents[chain], orientations[chain], 4,
                forward, backward, costs, True, True)
            if candidate_value < best_value:
                best, best_value, best_chain = candidate.copy(), candidate_value, chain
            delta = candidate_value - values[chain]
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                currents[chain], values[chain] = candidate, candidate_value
        return (best[::-1] if best_chain else best).tolist()
'''


def main():
    root = ROOT / "runs/minimal_fssp_portfolio_v1"
    if root.exists():
        raise FileExistsError(root)
    origin = ROOT / "runs/minimal_fssp_tiebreak_v1/selected_artifact/solver.py"
    source = origin.read_text()
    portfolio = source[:source.index("    def solve(self,")] + SOLVE
    protocol = ROOT / "data/bundles/minimal_v1/fssp/protocol.json"
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    spec = {"problem": "fssp", "protocol": str(protocol), "protocol_sha256": sha(protocol),
        "evidence_type": "human_engineering_training_only_selection_on_reused_minimal_data",
        "budget": {"n_instances": 3, "time_limit": 30, "ite_max": 10000000, "seeds": [71400, 71401]},
        "selection_rule": "minimum_training_mean_then_name", "sources": {},
        "hypothesis": "Keep two directional trajectories with separate incumbents, sharing the same time cap and reusable workspaces; no reheating or stronger ruin."}
    for name, code in {"symmetric_single": source, "two_directions": portfolio}.items():
        path = root / "sources" / name / "solver.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        spec["sources"][name] = {"path": str(path), "sha256": sha(path)}
    (root / "registry.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(root / "registry.json")


if __name__ == "__main__":
    main()
