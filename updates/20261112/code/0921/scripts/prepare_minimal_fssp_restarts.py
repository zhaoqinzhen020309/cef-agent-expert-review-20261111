"""Predeclare training-only adaptive IG policies, without changing held-out cases."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

SOLVE = '''    def solve(self, processing_times, deadline, iteration_limit, seed):
        p = np.ascontiguousarray(processing_times, dtype=np.float64)
        n, m = p.shape
        if n == 0 or m == 0:
            return list(range(n))
        orientations = (p, np.ascontiguousarray(p[:, ::-1]))
        orientation = 0
        forward, backward = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1))
        costs = np.empty(n + 1)
        current, value = initialize(p, seed % (2 ** 32), forward, backward, costs)
        best, best_value = current.copy(), value
        rng = np.random.default_rng(seed)
        base_temperature = max(1e-9, self.TEMPERATURE_FACTOR * p.mean() / 10)
        prefix = np.cumsum(p, axis=1) - p
        suffix = np.cumsum(p[:, ::-1], axis=1)[:, ::-1] - p
        lower_bound = max(p.sum(axis=1).max(), (p.sum(axis=0) + prefix.min(axis=0) + suffix.min(axis=0)).max())
        stagnant = 0
        strengths = np.asarray([2, 3, 4, 4, 4, 6, 8])
        for iteration in range(iteration_limit):
            if time.time() >= deadline - 0.2 or best_value <= lower_bound:
                break
            if self.DUAL and stagnant > 0 and stagnant % 256 == 0:
                orientation = 1 - orientation
                p = orientations[orientation]
                current = current[::-1].copy()
                best = best[::-1].copy()
            restarted = stagnant >= self.RESTART_AFTER
            if restarted:
                candidate, candidate_value = step(best, p, min(n, max(8, n // 5)), forward, backward, costs)
                current, value = candidate.copy(), candidate_value
                stagnant = 0
            else:
                strength = int(strengths[rng.integers(len(strengths))])
                candidate, candidate_value = step(current, p, strength, forward, backward, costs)
            if candidate_value < best_value:
                best, best_value = candidate.copy(), candidate_value
                stagnant = 0
            else:
                stagnant += 1
            temperature = base_temperature * (1 + min(stagnant / 256.0, 4.0))
            delta = candidate_value - value
            if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                current, value = candidate, candidate_value
        return (best[::-1] if orientation else best).tolist()
'''


def main():
    root = ROOT / "runs/minimal_fssp_restarts_v1"
    protocol = ROOT / "data/bundles/minimal_v1/fssp/protocol.json"
    original = (ROOT / "problems/fssp/engineering_controls/compiled_ig.py").read_text()
    prefix = original[:original.index("    def solve(self, processing_times,")]
    variants = {"baseline_cooler": original.replace("TEMPERATURE_FACTOR = 0.4", "TEMPERATURE_FACTOR = 0.2")}
    for name, dual, restart, temperature in (("adaptive", False, 768, .4), ("dual", True, 768, .4), ("frequent_restart", True, 256, .2)):
        source = prefix.replace("TEMPERATURE_FACTOR = 0.4", f"TEMPERATURE_FACTOR = {temperature}")
        source += f"    DUAL = {dual}\n    RESTART_AFTER = {restart}\n\n" + SOLVE
        variants[name] = source
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    registry = {"problem": "fssp", "protocol": str(protocol), "protocol_sha256": sha(protocol),
                "evidence_type": "training_selected_human_engineering_after_initial_minimal_validation; no blind claim",
                "candidate_selection": "training_mean_only", "budget": {"n_instances": 3, "time_limit": 30, "ite_max": 100000,
                                                                           "seeds": [71400, 71401]}, "sources": {}}
    for name, source in variants.items():
        path = root / "sources" / name / "solver.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(source, encoding="utf-8")
        registry["sources"][name] = {"path": str(path), "sha256": sha(path)}
    (root / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(root / "registry.json")


if __name__ == "__main__":
    main()
