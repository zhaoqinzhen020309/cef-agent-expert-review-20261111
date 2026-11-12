"""Predeclare stronger perturbation controls; source depends on training only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

ADAPTIVE_SOLVE = '''    def solve(self, processing_times, deadline, iteration_limit, seed):
        start = time.time()
        p = np.ascontiguousarray(processing_times, dtype=np.float64)
        n, m = p.shape
        if n == 0 or m == 0:
            return list(range(n))
        forward = np.zeros((n + 1, m + 1))
        backward = np.zeros((n + 1, m + 1))
        costs = np.empty(n + 1)
        current, value = initialize(p, seed % (2 ** 32), forward, backward, costs, True)
        best, best_value = current.copy(), value
        rng = np.random.default_rng(seed)
        base_temperature = max(1e-9, 0.4 * p.mean() / 10)
        prefix = np.cumsum(p, axis=1) - p
        suffix = np.cumsum(p[:, ::-1], axis=1)[:, ::-1] - p
        lower_bound = max(p.sum(axis=1).max(), (p.sum(axis=0) + prefix.min(axis=0) + suffix.min(axis=0)).max())
        initialized = time.time() - start
        iterations, improvements, stagnant = 0, 0, 0
        reversed_machines = False
        for iteration in range(iteration_limit):
            if time.time() >= deadline - 0.2 or best_value <= lower_bound:
                break
            # Reversing all machines and the permutation preserves makespan.
            # This changes the greedy reconstruction direction without changing
            # the problem or discarding the incumbent.
            if iteration > 0 and iteration % 128 == 0:
                p = np.ascontiguousarray(p[:, ::-1])
                current = current[::-1].copy()
                best = best[::-1].copy()
                reversed_machines = not reversed_machines
            destroy = 4 if stagnant < 64 else int(rng.integers(5, min(n, 12) + 1)) if n >= 5 else n
            temperature = base_temperature * (1.0 + min(stagnant / 512.0, 2.0))
            candidate, candidate_value = step(current, p, destroy, forward,
                backward, costs, True, True)
            iterations += 1
            stagnant += 1
            if candidate_value < best_value:
                best, best_value = candidate.copy(), candidate_value
                improvements += 1
                stagnant = 0
            delta = candidate_value - value
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                current, value = candidate, candidate_value
            if stagnant > 0 and stagnant % 512 == 0:
                current, value = best.copy(), best_value
        if reversed_machines:
            best = best[::-1].copy()
        self.statistics = {"iterations": iterations, "improvements": improvements,
                           "initialization_seconds": initialized, "best_objective": best_value}
        return best.tolist()
'''


def main():
    root = ROOT / "runs/minimal_fssp_adaptive_v1"
    if root.exists():
        raise FileExistsError(root)
    protocol = ROOT / "data/bundles/minimal_v1/fssp/protocol.json"
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source = (ROOT / "problems/fssp/engineering_controls/partial_neutral_ig.py").read_text()
    adaptive = source[:source.index("    def solve(self,")] + ADAPTIVE_SOLVE
    variants = {"partial_neutral_d4": source,
                "partial_neutral_d8": source.replace("    DESTROY_SIZE = 4", "    DESTROY_SIZE = 8"),
                "adaptive_bidirectional": adaptive}
    spec = {"problem": "fssp", "protocol": str(protocol), "protocol_sha256": sha(protocol),
            "evidence_type": "human_engineering_training_only_selection_on_reused_minimal_data",
            "budget": {"n_instances": 3, "time_limit": 30, "ite_max": 10000000, "seeds": [71400, 71401]},
            "selection_rule": "minimum_training_mean_then_name", "sources": {},
            "hypothesis": "Test stronger ruin and exact reversal symmetry after observed training stagnation; preserve reusable insertion buffers and incumbent."}
    for name, code in variants.items():
        path = root / "sources" / name / "solver.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        spec["sources"][name] = {"path": str(path), "sha256": sha(path)}
    (root / "registry.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(root / "registry.json")


if __name__ == "__main__":
    main()
