"""Accelerate the previous Agent FSSP solver without changing its search rules."""
from pathlib import Path
import ast
import hashlib
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]

KERNEL = '''
    @staticmethod
    def _completion_table(processing):
        count, machines = processing.shape
        table = np.zeros((count + 1, machines))
        previous = np.zeros(count)
        for machine in range(machines):
            cumulative = np.cumsum(processing[:, machine])
            offsets = np.empty(count)
            if count:
                offsets[0] = 0
                offsets[1:] = cumulative[:-1]
            previous = cumulative + np.maximum.accumulate(previous - offsets)
            table[1:, machine] = previous
        return table

    def insertion_costs(self, sequence, job, processing_times):
        processing = processing_times[np.asarray(sequence, dtype=np.int64)]
        count, machines = processing.shape
        forward = self._completion_table(processing)
        reverse = self._completion_table(processing[::-1, ::-1])
        backward = np.zeros_like(forward)
        backward[:count] = reverse[1:][::-1, ::-1]
        finish = np.zeros(count + 1)
        costs = np.zeros(count + 1)
        for machine in range(machines):
            finish = np.maximum(finish, forward[:, machine]) + processing_times[job, machine]
            costs = np.maximum(costs, finish + backward[:, machine])
        return costs

    def makespan(self, sequence, processing_times):
        if len(sequence) == 0 or processing_times.shape[1] == 0:
            return 0.0
        return float(self._completion_table(processing_times[np.asarray(sequence, dtype=np.int64)])[-1, -1])
'''


def main():
    run = ROOT / "runs/pilot_fssp_v2"
    state = json.loads((run / "search_state.json").read_text())
    node = next(n for n in state["nodes"] if n["node_id"] == state["best_node_id"])
    source = run / node["solver_relpath"]
    text = source.read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "SearchStrategy")
    lines = text.splitlines(keepends=True)
    targets = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"insertion_costs", "makespan"}]
    for node in sorted(targets, key=lambda n: n.lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [KERNEL + "\n"] if node.name == "insertion_costs" else []
    output = ROOT / "problems/fssp/engineering_controls/vectorized_ils.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# Human kernel optimization of a previous Agent solver; not a new Agent search result.\n" + "".join(lines), encoding="utf-8")
    receipt = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "change": "Replace scalar max-plus DP with exact prefix maximum scans; preserve insertion scores and search rules",
               "complexity": "O(n*m) arithmetic and workspace per insertion; Python loops reduced from O(n*m) to O(m)"}
    output.with_suffix(".provenance.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
