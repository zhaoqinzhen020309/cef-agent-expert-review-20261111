"""Batch independent insertion DPs while preserving first-improvement order."""
from pathlib import Path
import ast
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
METHODS = '''
    @staticmethod
    def _batch_completion(processing):
        batch, count, machines = processing.shape
        table = np.zeros((batch, count + 1, machines))
        previous = np.zeros((batch, count))
        for machine in range(machines):
            cumulative = np.cumsum(processing[:, :, machine], axis=1)
            offsets = np.zeros_like(cumulative)
            offsets[:, 1:] = cumulative[:, :-1]
            previous = cumulative + np.maximum.accumulate(previous - offsets, axis=1)
            table[:, 1:, machine] = previous
        return table

    def _batch_removed_costs(self, sequence, positions, p):
        seq = np.asarray(sequence, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        count, batch = len(seq), len(positions)
        keep = np.broadcast_to(np.arange(count), (batch, count))
        keep = keep[keep != positions[:, None]].reshape(batch, count - 1)
        remaining = seq[keep]
        processing = p[remaining]
        forward = self._batch_completion(processing)
        backward = self._batch_completion(processing[:, ::-1, ::-1])[:, ::-1, ::-1]
        finish = np.zeros((batch, count))
        costs = np.zeros_like(finish)
        jobs = seq[positions]
        for machine in range(p.shape[1]):
            finish = np.maximum(finish, forward[:, :, machine]) + p[jobs, machine, None]
            costs = np.maximum(costs, finish + backward[:, :, machine])
        return costs

    def _local_search(self, sequence, initial_ms, processing_times, deadline, rng):
        seq, best_ms = list(sequence), initial_ms
        improved = True
        while improved and self._time_ok(deadline):
            improved = False
            indices = list(range(len(seq)))
            rng.shuffle(indices)
            for start in range(0, len(seq), self.CANDIDATE_BATCH):
                if not self._time_ok(deadline):
                    return seq, best_ms
                positions = indices[start:start + self.CANDIDATE_BATCH]
                costs = self._batch_removed_costs(seq, positions, processing_times)
                minima = np.min(costs, axis=1)
                eligible = np.flatnonzero(minima + 1e-12 < best_ms)
                if len(eligible):
                    # Same first improving job and first best insertion as scalar LS.
                    row = int(eligible[0])
                    job = seq.pop(positions[row])
                    seq.insert(int(np.argmin(costs[row])), job)
                    best_ms = float(minima[row])
                    improved = True
                    break
        return seq, best_ms
'''


def main():
    root = ROOT / "runs/fssp_batch_training_controls_v1"
    if root.exists():
        raise FileExistsError("batch experiment already registered")
    root.mkdir(parents=True)
    parent = ROOT / "runs/fssp_ig_training_controls_v1/ig_d8_t0.4/solver.py"
    text = parent.read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef))
    target = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_local_search")
    lines = text.splitlines(keepends=True)
    lines[target.lineno - 1:target.end_lineno] = [METHODS + "\n"]
    text = "".join(lines)
    sources = {}
    for batch in (8, 24):
        name = f"ig_batch_{batch}"
        directory = root / name
        directory.mkdir()
        source = directory / "solver.py"
        source.write_text(text.replace("    TIME_MARGIN = 0.15", f"    TIME_MARGIN = 0.15\n    CANDIDATE_BATCH = {batch}"))
        sources[name] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    protocol = ROOT / "data/bundles/guidance_v1/fssp/fold_0/protocol.json"
    registry = {"evidence_type": "human_engineering_batched_exact_kernel_not_agent_search", "problem": "fssp",
                "protocol": str(protocol), "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "sources": sources, "budget": {"n_instances": 11, "time_limit": 10, "ite_max": 1000, "seeds": [68001, 68002]},
                "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(), "grid": [8, 24],
                "complexity": "same O(n*m) per removal, O(B*n*m) workspace with fixed B; fewer Python dispatches"}
    (root / "registry.json").write_text(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
