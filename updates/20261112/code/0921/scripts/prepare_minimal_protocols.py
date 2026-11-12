"""Freeze tiny, score-independent subsets under the user's revised acceptance."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"frozen minimal protocol differs: {path}")
    path.write_text(text, encoding="utf-8")


def select(kind, split, rows):
    ordered = sorted(rows, key=lambda row: (row["size"], row["id"]))
    if kind == "tsp_constructive":
        if split == "train":
            return ordered[:4]
        return [next(r for r in ordered if r["size"] == n) for n in (50, 100, 200)]
    if kind == "tsp_gls":
        if split == "train":
            return ordered[:3]
        return [next(r for r in ordered if lo <= r["size"] <= hi)
                for lo, hi in ((50, 99), (100, 149), (150, 200))]
    families = [f"size_band_{i}" for i in range(5)] if kind == "cvrp" else ["20_5", "50_10", "100_20"]
    return [next(r for r in ordered if r["family"] == family) for family in families]


def main():
    root = ROOT / "data/bundles/minimal_v1"
    record = {"schema": "minimal_acceptance_v1", "acceptance_scope": "fixed_minimal_validation_only",
              "full_validation_requested": False, "selection_rule": "first by size then ID in predeclared strata; no scores read",
              "interpretation": "small-sample development acceptance; not full-benchmark or paper superiority evidence",
              "historical_exposure": "subsets of previously used public validation pools; no blind claim", "tasks": {}}
    for kind in ("tsp_constructive", "tsp_gls", "cvrp", "fssp"):
        original_dir = ROOT / "data/bundles/guidance_v1" / kind
        if kind in ("cvrp", "fssp"):
            original_dir /= "fold_0"
        source = original_dir / "protocol.json"
        protocol = json.loads(source.read_text(encoding="utf-8"))
        protocol["protocol_id"] = kind + "_minimal_acceptance_v1"
        protocol["data_root"] = str(root / kind)
        protocol["metadata"].update(acceptance_scope=record["acceptance_scope"], parent_protocol=str(source),
                                    full_validation_requested=False)
        task_record = {}
        selected_hashes = {}
        for split in ("train", "validation"):
            manifest = json.loads(Path(protocol[split]["source_path"]).read_text(encoding="utf-8"))
            rows = select(kind, split, manifest["instances"])
            manifest.update(instances=deepcopy(rows), contract_id=protocol["protocol_id"])
            target = root / kind / (split + ".json")
            write_once(target, manifest)
            protocol[split].update(source_path=str(target), instance_ids=[r["id"] for r in rows],
                                   family_by_instance={r["id"]: r["family"] for r in rows})
            task_record[split] = [{k: r[k] for k in ("id", "family", "size", "sha256")} for r in rows]
            selected_hashes[split] = {r["sha256"] for r in rows}
        if selected_hashes["train"] & selected_hashes["validation"]:
            raise ValueError("minimal train/validation content overlaps")
        target = root / kind / "protocol.json"
        write_once(target, protocol)
        task_record.update(protocol=str(target), protocol_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        record["tasks"][kind] = task_record
    write_once(root / "selection.json", record)
    print(json.dumps({kind: {s: [r['id'] for r in row[s]] for s in ('train', 'validation')}
                      for kind, row in record['tasks'].items()}, indent=2))


if __name__ == "__main__":
    main()
