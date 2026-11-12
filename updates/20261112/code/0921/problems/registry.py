"""Problem selection is an adapter concern, never a controller branch."""
from pathlib import Path
import json


def get_problem_adapter(name, *, dataset_protocol_path=None, dataset_protocol=None, data_root=None):
    if name == "tsp_gls" and (dataset_protocol or {}).get("task") is None:
        if dataset_protocol_path is None or "task" not in json.loads(Path(dataset_protocol_path).read_text(encoding="utf-8")):
            from agent_ahd.problems.tsp_gls.adapter import TSPGLSAdapter, _read_dataset_protocol
            protocol = dataset_protocol
            if dataset_protocol_path is not None:
                protocol = _read_dataset_protocol(dataset_protocol_path)
            return TSPGLSAdapter(protocol, data_root=data_root)
    from agent_ahd.problems.manifest_adapter import ManifestAdapter
    if dataset_protocol_path is not None and dataset_protocol is not None:
        raise ValueError("supply one dataset protocol")
    if dataset_protocol_path is not None:
        path = Path(dataset_protocol_path).resolve()
        dataset_protocol = json.loads(path.read_text(encoding="utf-8"))
        data_root = data_root or path.parent
    return ManifestAdapter(name, dataset_protocol, data_root=data_root)


def load_run_adapter(run_dir):
    root = Path(run_dir).resolve()
    config = json.loads((root / "planner_config.json").read_text(encoding="utf-8"))
    adapter = get_problem_adapter(config["problem"], dataset_protocol=config["dataset_protocol"],
                                  data_root=config["dataset_protocol"].get("data_root"))
    state = json.loads((root / "search_state.json").read_text(encoding="utf-8"))
    path = Path(state["baseline_path"])
    adapter.baseline_path = path.resolve() if path.is_absolute() else (root / path).resolve()
    return adapter
