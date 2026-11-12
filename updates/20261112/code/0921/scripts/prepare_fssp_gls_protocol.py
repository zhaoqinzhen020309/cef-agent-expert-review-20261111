"""Prepare a new FSSP input protocol without initializing or running a search.

This utility is stdlib-only and never imports a solver or a model backend.
Existing absolute paths are relocated only with an explicit old/new root pair.
Start a later search with the existing --baseline-parent option; no search
settings or evaluation budgets are chosen here.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import math


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOLVER = ROOT / "problems/fssp/neh_gls_v1/solver.py"
ENTRY = {"class_name": "SearchStrategy", "method_name": "solve",
         "parameters": ["self", "processing_times", "deadline", "iteration_limit", "seed"],
         "input_kind": "array", "output_kind": "permutation"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matches_entry(value):
    if not isinstance(value, dict):
        return False
    normalized = dict(value)
    normalized.setdefault("python_random_seed_policy", "solver_managed")
    return normalized == {**ENTRY, "python_random_seed_policy": "solver_managed"}


def prepare_protocol(base_protocol, output_dir, *, solver_path=DEFAULT_SOLVER,
                     old_root=None, new_root=None):
    if (old_root is None) != (new_root is None):
        raise ValueError("old_root and new_root must be supplied together")
    base = Path(base_protocol).resolve()
    output = Path(output_dir).resolve()
    solver = Path(solver_path).resolve()
    if not solver.is_file():
        raise FileNotFoundError(solver)
    source = _json(base)
    if source.get("task", {}).get("name") != "fssp":
        raise ValueError("base protocol must describe fssp")
    if not _matches_entry(source["task"].get("entry_contract")):
        raise ValueError("base fssp entry contract differs from the existing solver interface")
    if not source.get("protocol_id") or not source.get("blind", {}).get("commitment"):
        raise ValueError("source protocol requires an id and explicit blind commitment")
    origin = Path(old_root).resolve() if old_root is not None else None
    destination = Path(new_root).resolve() if new_root is not None else None

    def resolve(value, anchor):
        path = Path(value)
        if not path.is_absolute():
            path = anchor / path
        path = path.resolve()
        if origin is not None and path.is_relative_to(origin):
            path = destination / path.relative_to(origin)
        return path

    recorded_root = Path(source.get("data_root", base.parent))
    if not recorded_root.is_absolute():
        recorded_root = base.parent / recorded_root
    prepared = deepcopy(source)
    prepared["protocol_id"] = source["protocol_id"] + "__neh_gls_v1"
    prepared["data_root"] = str(output)
    prepared["task"]["baseline_path"] = str(solver)
    documents = {}
    selected = {}
    source_dirs = {base.parent}
    for split in ("train", "validation"):
        manifest_path = resolve(source[split]["source_path"], recorded_root)
        source_dirs.add(manifest_path.parent)
        manifest = _json(manifest_path)
        if (manifest.get("schema") != "task_dataset_v1" or manifest.get("kind") != "fssp"
                or manifest.get("split") != split or not _matches_entry(manifest.get("entry_contract"))
                or manifest.get("contract_id") != source["protocol_id"]):
            raise ValueError("source manifest task, split or contract mismatch")
        rows = manifest.get("instances", [])
        if not rows:
            raise ValueError("source manifest has no instances")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or ids != source[split]["instance_ids"]:
            raise ValueError("source instance identifiers or order differ")
        families = {row["id"]: row["family"] for row in rows}
        if families != source[split]["family_by_instance"]:
            raise ValueError("source instance families differ")
        hashes = set()
        for row in rows:
            path = resolve(row["path"], manifest_path.parent)
            if not path.is_file():
                raise FileNotFoundError(path)
            if _digest(path) != row["sha256"]:
                raise ValueError(f"instance content hash changed: {row['id']}")
            reference = float(row["reference"])
            if (row.get("reference_kind") not in {"optimal", "best_known"}
                    or not math.isfinite(reference) or reference <= 0
                    or not row.get("source") or not row.get("family")):
                raise ValueError("source instance reference or provenance is invalid")
            hashes.add(row["sha256"])
            row["path"] = str(path)
        selected[split] = (set(ids), hashes)
        manifest["contract_id"] = prepared["protocol_id"]
        prepared[split]["source_path"] = str(output / (split + ".json"))
        documents[split + ".json"] = manifest
    if output in source_dirs:
        raise ValueError("output must be separate from source protocol and manifests")
    if (selected["train"][0] & selected["validation"][0]
            or selected["train"][1] & selected["validation"][1]):
        raise ValueError("source train and validation instances overlap")
    documents["protocol.json"] = prepared
    encoded = {name: (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
               for name, value in documents.items()}
    if output.exists():
        for existing in output.iterdir():
            if (not existing.is_file() or existing.name not in encoded
                    or existing.read_bytes() != encoded[existing.name]):
                raise FileExistsError(f"refuse to overwrite existing protocol output: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        target = output / name
        if not target.exists():
            target.write_bytes(content)
    return output / "protocol.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--new-root", type=Path)
    args = parser.parse_args()
    path = prepare_protocol(args.base_protocol, args.output, solver_path=args.solver,
                            old_root=args.old_root, new_root=args.new_root)
    print(json.dumps({"protocol": str(path), "required_search_option": "--baseline-parent",
                      "search_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
