"""Task bundle: description + baseline + entry contract + evaluator + splits."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import copy
import json

from agent_ahd.problems.base import CandidateTask, DatasetManifest, EntryContract, ProblemSpec


class ManifestAdapter:
    def __init__(self, name: str, protocol: dict, *, data_root=None):
        if not isinstance(protocol, dict) or not protocol.get("protocol_id"):
            raise ValueError("a versioned task protocol is required")
        self.name = name
        self.dataset_protocol = copy.deepcopy(protocol)
        self.data_root = Path(data_root or protocol.get("data_root", ".")).resolve()
        task = self.dataset_protocol["task"]
        if task["name"] != name:
            raise ValueError("task name does not match requested problem")
        self.baseline_path = self._path(task["baseline_path"])
        if not self.baseline_path.is_file():
            raise FileNotFoundError(self.baseline_path)
        task["baseline_path"] = str(self.baseline_path)
        entry = dict(task["entry_contract"])
        entry["parameters"] = tuple(entry["parameters"])
        self.entry_contract = EntryContract(**entry)
        from agent_ahd.evaluators.manifest_evaluator import metric_contract
        self.metric = metric_contract(protocol)
        self.metric_name, self.metric_unit = self.metric["name"], self.metric["unit"]
        gap_unit = "percent" if self.metric_name == "optimality_gap" else "not_applicable"
        self.spec = ProblemSpec(name, self.baseline_path.parent, self.entry_contract.class_name,
                                "solver.py", gap_unit, task["evaluator_module"])
        self._description = str(task["description"])
        if not self._description.strip():
            raise ValueError("problem description must not be empty")
        self.train_manifest = self._manifest("train")
        self.training_only = protocol.get("training_only", False)
        if type(self.training_only) is not bool:
            raise ValueError("training_only must be boolean")
        if self.training_only:
            validation = protocol.get("validation", {})
            if validation.get("source_path") is not None or validation.get("instance_ids"):
                raise ValueError("training-only protocol cannot register validation data")
            self.validation_manifest = DatasetManifest("validation", None, (), False)
        else:
            self.validation_manifest = self._manifest("validation")
        if set(self.train_manifest.instance_ids) & set(self.validation_manifest.instance_ids):
            raise ValueError("train and validation identifiers overlap")
        from agent_ahd.evaluators.manifest_evaluator import read_manifest
        train = read_manifest(self.train_manifest.source_path)
        validation = read_manifest(self.validation_manifest.source_path) if not self.training_only else {"instances": []}
        # Renaming the same instance must not bypass split separation.
        if {r["sha256"] for r in train["instances"]} & {r["sha256"] for r in validation["instances"]}:
            raise ValueError("train and validation data content overlaps")
        self.blind_commitment = str(protocol.get("blind", {}).get("commitment", ""))
        if not self.blind_commitment:
            raise ValueError("opaque blind commitment (or explicit no-blind marker) required")
        self.blind_manifest_path = self.blind_data_path = None

    def _path(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.data_root / path).resolve()

    def _manifest(self, split):
        from agent_ahd.evaluators.manifest_evaluator import read_manifest
        record = self.dataset_protocol[split]
        path = self._path(record["source_path"])
        content = read_manifest(path)
        from agent_ahd.evaluators.manifest_evaluator import metric_contract
        if metric_contract(content) != self.metric:
            raise ValueError("manifest metric differs from the task protocol")
        if content["split"] != split or content["kind"] != self.name:
            raise ValueError("manifest split/task mismatch")
        manifest_entry = dict(content["entry_contract"])
        manifest_entry["parameters"] = tuple(manifest_entry["parameters"])
        if EntryContract(**manifest_entry) != self.entry_contract:
            raise ValueError("manifest and task entry contracts differ")
        ids = tuple(r["id"] for r in content["instances"])
        families = {r["id"]: r["family"] for r in content["instances"]}
        if tuple(record["instance_ids"]) != ids:
            raise ValueError("protocol instance order differs from manifest")
        record.update(source_path=str(path), family_by_instance=families)
        return DatasetManifest(split, path, ids, split == "train", families)

    def problem_context(self):
        from agent_ahd.evaluators.manifest_evaluator import read_manifest
        rows = read_manifest(self.train_manifest.source_path)["instances"]
        scope = {"training_families": sorted(set(self.train_manifest.family_by_instance.values())),
                 "training_size_range": [min(r["size"] for r in rows), max(r["size"] for r in rows)]}
        return (self._description + "\nFixed entry contract: " + json.dumps(asdict(self.entry_contract))
                + "\nLocked metric: " + json.dumps(self.metric)
                + "\nDesign data coverage: " + json.dumps(scope)
                + "\nAll initialization and preprocessing count against the solver deadline. "
                  "Preserve a valid incumbent and return it with a conservative time margin.")

    def make_task(self, *, candidate_id, candidate_path, budget, split, instance_ids=()):
        if self.training_only and split != "train":
            raise PermissionError("training-only protocol cannot evaluate other splits")
        if split not in {"train", "validation"}:
            raise PermissionError("search cannot access final/blind data")
        manifest = self.train_manifest if split == "train" else self.validation_manifest
        ids = tuple(instance_ids) if instance_ids else manifest.instance_ids[:budget.n_instances]
        if len(ids) != len(set(ids)) or not set(ids) <= set(manifest.instance_ids):
            raise ValueError("duplicate or unknown instance identifiers")
        if not ids:
            raise ValueError("empty evaluation")
        ids = tuple(i for i in manifest.instance_ids if i in set(ids))[:budget.n_instances]
        return CandidateTask(candidate_id, Path(candidate_path), self.dataset_protocol["task"]["evaluator_module"],
                             self.dataset_protocol["task"].get("evaluator_function", "evaluate_candidate_file"),
                             budget, self.spec.gap_unit, self.dataset_protocol["protocol_id"], split, ids, manifest.source_path,
                             metric_name=self.metric_name, metric_unit=self.metric_unit)
