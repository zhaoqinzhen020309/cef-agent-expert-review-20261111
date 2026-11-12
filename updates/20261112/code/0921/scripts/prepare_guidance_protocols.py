"""Freeze declared deployment shapes without reading validation outcomes."""
from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_task_bundles import write_json

# Public task requirements, not learned from validation scores.
OPERATING_SPECS = {
    "tsp_constructive": {"design_sizes": [50], "deployment_sizes": [50, 100, 200],
                         "policy_calls_per_instance": "n-1", "max_input_size": 200,
                         "interface_restriction": "next city only; no completed-tour local search"},
    "tsp_gls": {"max_input_size": 1000, "design_size": 100,
                "deployment_family": "fixed TSPLIB29; integer distances with ties"},
    "cvrp": {"max_input_size": 1001, "customer_count_range": [100, 1000],
             "capacity_feasibility": "every route demand <= capacity"},
    "fssp": {"max_input_size": 200, "jobs_range": [20, 200], "machine_counts": [5, 10, 20]},
}


def main():
    paths = [ROOT / "data/bundles/tsp_constructive/protocol.json"]
    paths.extend((ROOT / "data/bundles/v2").glob("**/protocol.json"))
    for path in sorted(paths):
        original = json.loads(path.read_text(encoding="utf-8"))
        kind = original["task"]["name"]
        root = ROOT / "data/bundles/guidance_v1" / kind
        if "fold_" in path.parent.name:
            root /= path.parent.name
        protocol = copy.deepcopy(original)
        protocol["protocol_id"] = kind + "_resource_contract_v1"
        protocol["data_root"] = str(root)
        protocol["task"]["operating_spec"] = OPERATING_SPECS[kind]
        protocol.setdefault("metadata", {})["parent_protocol"] = str(path)
        for split in ("train", "validation"):
            manifest = json.loads(Path(original[split]["source_path"]).read_text(encoding="utf-8"))
            manifest["contract_id"] = protocol["protocol_id"]
            target = root / f"{split}.json"
            write_json(target, manifest)
            protocol[split]["source_path"] = str(target)
        write_json(root / "protocol.json", protocol)
        print(root / "protocol.json")


if __name__ == "__main__":
    main()
