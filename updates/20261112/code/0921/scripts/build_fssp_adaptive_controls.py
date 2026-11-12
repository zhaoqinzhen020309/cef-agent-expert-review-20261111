"""Register a bounded engineering grid; only training data informs selection."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOLVE = '''
    def _construct(self, p, deadline):
        order = np.argsort(-p.sum(axis=1), kind="stable").tolist()
        sequence = []
        for index, job in enumerate(order):
            if not self._time_ok(deadline):
                return sequence + order[index:]
            sequence.insert(int(np.argmin(self.insertion_costs(sequence, job, p))), job)
        return sequence

    def solve(self, processing_times, deadline, iteration_limit, seed):
        import math
        p = np.asarray(processing_times, dtype=float)
        n, machines = p.shape
        if n == 0 or machines == 0:
            return list(range(n))
        rng = Random(seed)
        sequence = self._construct(p, deadline)
        value = self.makespan(sequence, p)
        if self._time_ok(deadline):
            dual = self._construct(p[:, ::-1], deadline)[::-1]
            dual_value = self.makespan(dual, p)
            if dual_value < value:
                sequence, value = dual, dual_value
        current, value = self._local_search(sequence, value, p, deadline, rng)
        best, best_value, stale, reverse = current[:], value, 0, False
        elites = [(best_value, best[:])]
        prefix = np.cumsum(p, axis=1) - p
        suffix = np.cumsum(p[:, ::-1], axis=1)[:, ::-1] - p
        lower_bound = max(float(p.sum(axis=1).max()), float((p.sum(axis=0) + prefix.min(axis=0) + suffix.min(axis=0)).max()))
        base_temperature = max(1e-9, self.TEMPERATURE_FACTOR * float(p.mean()) / 10.0)
        for iteration in range(iteration_limit):
            if not self._time_ok(deadline) or best_value <= lower_bound:
                break
            if iteration and iteration % 10 == 0:
                # Reversing both machine order and schedule preserves makespan.
                p = p[:, ::-1]
                current.reverse()
                best.reverse()
                elites = [(score, order[::-1]) for score, order in elites]
                reverse = not reverse
            strength = self.DESTROY_SIZE
            if self.VARIANT in ("adaptive_dual", "elite"):
                strength = 4 + 4 * ((stale // 12) % 3)
            if self.VARIANT == "elite" and stale and stale % 20 == 0:
                value, donor = rng.choice(elites)
                current = donor[:]
            removed = rng.sample(current, min(strength, n))
            chosen = set(removed)
            candidate = [job for job in current if job not in chosen]
            if self.VARIANT == "reduced_search" and len(candidate) > 1:
                candidate, _ = self._local_search(candidate, self.makespan(candidate, p), p, deadline, rng)
            interrupted = False
            for job in removed:
                if not self._time_ok(deadline):
                    interrupted = True
                    break
                costs = self.insertion_costs(candidate, job, p)
                candidate.insert(rng.choice(np.flatnonzero(costs == costs.min()).tolist()), job)
            if interrupted:
                break
            candidate, candidate_value = self._local_search(candidate, self.makespan(candidate, p), p, deadline, rng)
            stale += 1
            if candidate_value < best_value:
                best, best_value, stale = candidate[:], candidate_value, 0
            if self.VARIANT == "elite" and not any(candidate == order for _, order in elites):
                elites.append((candidate_value, candidate[:]))
                elites.sort(key=lambda item: item[0])
                elites = elites[:5]
            temperature = base_temperature * (2 if self.VARIANT in ("adaptive_dual", "elite") and stale >= 8 else 1)
            delta = candidate_value - value
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current, value = candidate, candidate_value
        return best[::-1] if reverse else best
'''


def main():
    root = ROOT / "runs/fssp_adaptive_training_controls_v1"
    if root.exists():
        raise FileExistsError("adaptive control grid already registered")
    root.mkdir(parents=True)
    parent = ROOT / "problems/fssp/engineering_controls/batched_ig.py"
    source = parent.read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    target = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "solve")
    lines = source.splitlines(keepends=True)
    lines[target.lineno - 1:target.end_lineno] = [SOLVE + "\n"]
    source = "".join(lines)
    sources = {}
    for variant in ("dual_neutral", "adaptive_dual", "reduced_search", "elite"):
        directory = root / variant
        directory.mkdir()
        path = directory / "solver.py"
        path.write_text(source.replace("    TIME_MARGIN = 0.15", f"    TIME_MARGIN = 0.15\n    VARIANT = {variant!r}"))
        sources[variant] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    protocol = ROOT / "data/bundles/guidance_v1/fssp/fold_0/protocol.json"
    registry = {"evidence_type": "human_engineering_training_grid_not_agent_search", "problem": "fssp",
                "protocol": str(protocol), "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "sources": sources, "budget": {"n_instances": 11, "time_limit": 10, "ite_max": 1000, "seeds": [68001, 68002]},
                "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
                "selection_rule": "mean training gap only; keep all failures and no validation during design",
                "reference_family": "https://doi.org/10.1016/j.ejor.2005.12.009",
                "complexity": "bounded 24-job batch O(B*n*m) memory plus <=5 O(n) elite schedules"}
    (root / "registry.json").write_text(json.dumps(registry, indent=2))
    print(json.dumps({"registered": len(sources), "root": str(root)}))


if __name__ == "__main__":
    main()
