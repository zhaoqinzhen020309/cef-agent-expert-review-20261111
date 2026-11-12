"""Register literature-inspired IG engineering controls; never Agent evidence.

Algorithm family: Ruiz & Stutzle, EJOR 177 (2007), 2033-2049,
https://doi.org/10.1016/j.ejor.2005.12.009. Local implementation, not paper-code reproduction.
"""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOLVE = '''
    def solve(self, processing_times, deadline, iteration_limit, seed):
        import math
        p = np.asarray(processing_times, dtype=float)
        n, machines = p.shape
        if n == 0 or machines == 0:
            return list(range(n))
        rng = Random(seed)
        order = np.argsort(-p.sum(axis=1), kind="stable").tolist()
        sequence = []
        for index, job in enumerate(order):
            if not self._time_ok(deadline):
                return sequence + order[index:]
            costs = self.insertion_costs(sequence, job, p)
            sequence.insert(int(np.argmin(costs)), job)
        current, value = self._local_search(sequence, self.makespan(sequence, p), p, deadline, rng)
        best, best_value = current[:], value
        prefix = np.cumsum(p, axis=1) - p
        suffix = np.cumsum(p[:, ::-1], axis=1)[:, ::-1] - p
        lower_bound = max(float(p.sum(axis=1).max()), float((p.sum(axis=0) + prefix.min(axis=0) + suffix.min(axis=0)).max()))
        temperature = max(1e-9, self.TEMPERATURE_FACTOR * float(p.mean()) / 10.0)
        for _ in range(iteration_limit):
            if not self._time_ok(deadline) or best_value <= lower_bound:
                break
            removed = rng.sample(current, min(self.DESTROY_SIZE, n))
            chosen = set(removed)
            candidate = [job for job in current if job not in chosen]
            for job in removed:
                if not self._time_ok(deadline):
                    return best
                costs = self.insertion_costs(candidate, job, p)
                positions = np.flatnonzero(costs == costs.min()).tolist()
                candidate.insert(rng.choice(positions), job)
            candidate, candidate_value = self._local_search(candidate, self.makespan(candidate, p), p, deadline, rng)
            if candidate_value < best_value:
                best, best_value = candidate[:], candidate_value
            delta = candidate_value - value
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current, value = candidate, candidate_value
        return best
'''


def main():
    root = ROOT / "runs/fssp_ig_training_controls_v1"
    if root.exists():
        raise FileExistsError("control registration already exists")
    root.mkdir(parents=True)
    base = ROOT / "problems/fssp/engineering_controls/vectorized_ils.py"
    source = base.read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    solve = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "solve")
    lines = source.splitlines(keepends=True)
    lines[solve.lineno - 1:solve.end_lineno] = [SOLVE + "\n"]
    # A reverse view avoids the extra dense suffix table; arithmetic is unchanged.
    source = "".join(lines).replace("        backward = np.zeros_like(forward)\n        backward[:count] = reverse[1:][::-1, ::-1]",
                                   "        backward = reverse[::-1, ::-1]")
    protocol = ROOT / "data/bundles/guidance_v1/fssp/fold_0/protocol.json"
    grid = [(4, .4), (6, .4), (4, .8), (8, .4)]
    sources = {}
    for destroy, temperature in grid:
        name = f"ig_d{destroy}_t{temperature}"
        directory = root / name
        directory.mkdir()
        candidate = directory / "solver.py"
        candidate.write_text(source.replace("    TIME_MARGIN = 0.05", f"    TIME_MARGIN = 0.15\n    DESTROY_SIZE = {destroy}\n    TEMPERATURE_FACTOR = {temperature}"))
        sources[name] = {"path": str(candidate), "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}
    registry = {"evidence_type": "human_engineering_IG_control_not_agent_search", "problem": "fssp",
                "protocol": str(protocol), "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "sources": sources, "budget": {"n_instances": 11, "time_limit": 10, "ite_max": 1000, "seeds": [68001, 68002]},
                "base_kernel_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "algorithm_family_source": "https://doi.org/10.1016/j.ejor.2005.12.009", "grid": grid}
    (root / "registry.json").write_text(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
