from __future__ import annotations

import json
from pathlib import Path

from agent_ahd.problems.base import (
    CandidateTask,
    DatasetManifest,
    EntryContract,
    EvaluationBudget,
    ProblemSpec,
)
from agent_ahd.evaluators.tsplib_protocol import TSPLIB29_INSTANCES


class TSPGLSAdapter:
    name = "tsp_gls"

    def __init__(
        self,
        dataset_protocol: dict | None = None,
        *,
        data_root: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent
        project_root = root.parents[2]
        resolved_data_root = (
            Path(data_root).resolve() if data_root is not None else project_root
        )
        tsplib_data_path = (
            project_root
            / "examples"
            / "user_tsp_gls_OPRO"
            / "TestingData"
            / "TSPLib"
        )
        protocol = dataset_protocol or _default_dataset_protocol(
            root,
            tsplib_data_path,
        )
        self.dataset_protocol = _validate_dataset_protocol(
            protocol,
            data_root=resolved_data_root,
        )
        self.data_root = resolved_data_root
        train = self.dataset_protocol["train"]
        validation = self.dataset_protocol["validation"]
        blind = self.dataset_protocol["blind"]
        self.split_formats = {
            "train": str(train["format"]),
            "validation": str(validation["format"]),
        }
        self.baseline_path = root / "problem.py"
        self.spec = ProblemSpec(
            name=self.name,
            root_dir=root,
            entry_class_name="SearchStrategy",
            solver_filename="solver.py",
            gap_unit="percent",
            evaluator_name="tsp_gls_evaluator",
        )
        self.entry_contract = EntryContract(
            class_name="SearchStrategy",
            method_name="solve",
            parameters=(
                "self",
                "distance_matrix",
                "deadline",
                "iteration_limit",
                "seed",
            ),
        )
        self.train_manifest = DatasetManifest(
            split="train",
            source_path=Path(str(train["source_path"])),
            instance_ids=tuple(str(item) for item in train["instance_ids"]),
            exposed_to_search=True,
            family_by_instance=dict(train["family_by_instance"]),
        )
        self.validation_manifest = DatasetManifest(
            split="validation",
            source_path=Path(str(validation["source_path"])),
            instance_ids=tuple(
                str(item) for item in validation["instance_ids"]
            ),
            exposed_to_search=False,
            family_by_instance=dict(validation["family_by_instance"]),
        )
        self.blind_commitment = str(blind.get("commitment") or "")
        self.blind_manifest_path = (
            Path(str(blind["manifest_path"]))
            if "manifest_path" in blind
            else None
        )
        self.blind_data_path = (
            Path(str(blind["data_path"])) if "data_path" in blind else None
        )

    def problem_context(self) -> str:
        return (
            "Design a complete TSP solver. The fixed interface is "
            "SearchStrategy.solve(distance_matrix, deadline, iteration_limit, seed) "
            "-> a permutation containing every node exactly once. Route cost, "
            "feasibility, datasets, metric, and budgets are computed externally from "
            "the original distance matrix. Internal initialization, neighborhoods, "
            "local-search control, perturbation, acceptance, memory, restart, and "
            "scheduling logic may be redesigned."
        )

    def make_task(
        self,
        *,
        candidate_id: str,
        candidate_path: Path,
        budget: EvaluationBudget,
        split: str,
        instance_ids: tuple[str, ...] = (),
    ) -> CandidateTask:
        if split == "train":
            manifest = self.train_manifest
            evaluator_module = "agent_ahd.evaluators.tsp_gls_evaluator"
            evaluator_function = "evaluate_candidate_file"
        elif split == "validation":
            manifest = self.validation_manifest
            if self.split_formats["validation"] == "tsplib":
                evaluator_module = "agent_ahd.evaluators.canonical_tsplib_evaluator"
                evaluator_function = "evaluate_candidate_manifest"
            else:
                evaluator_module = "agent_ahd.evaluators.tsp_gls_evaluator"
                evaluator_function = "evaluate_candidate_file"
        elif split == "blind":
            raise PermissionError(
                "blind evaluation is unavailable to the search engine; use the final evaluator"
            )
        else:
            raise ValueError(f"unknown split: {split}")
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError(f"duplicate {split} instance ids")
        selected_ids = tuple(instance_ids) if instance_ids else manifest.instance_ids
        unknown_ids = set(selected_ids) - set(manifest.instance_ids)
        if unknown_ids:
            raise ValueError(
                f"unknown {split} instance ids: {sorted(unknown_ids)}"
            )
        selected_set = set(selected_ids)
        selected_ids = tuple(
            instance_id
            for instance_id in manifest.instance_ids
            if instance_id in selected_set
        )[: budget.n_instances]
        return CandidateTask(
            candidate_id=candidate_id,
            candidate_path=candidate_path,
            evaluator_module=evaluator_module,
            evaluator_function=evaluator_function,
            budget=budget,
            gap_unit=self.spec.gap_unit,
            contract_id="tsp_gls_complete_solver_v1",
            split=split,
            instance_ids=selected_ids,
            data_path=manifest.source_path,
        )


def get_problem_adapter(
    name: str,
    *,
    dataset_protocol_path: str | Path | None = None,
    dataset_protocol: dict | None = None,
    data_root: str | Path | None = None,
):
    if name == "tsp_gls":
        if dataset_protocol_path is not None and dataset_protocol is not None:
            raise ValueError("provide either dataset_protocol_path or dataset_protocol")
        protocol = dataset_protocol
        if dataset_protocol_path is not None:
            protocol = _read_dataset_protocol(dataset_protocol_path)
        return TSPGLSAdapter(protocol, data_root=data_root)
    raise KeyError(f"unknown problem adapter: {name}")


def load_run_adapter(run_dir: str | Path):
    run_path = Path(run_dir).resolve()
    protocol_path = run_path / "planner_config.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    adapter = get_problem_adapter(
        str(protocol["problem"]),
        dataset_protocol=protocol["dataset_protocol"],
        data_root=protocol["dataset_protocol"].get("data_root"),
    )
    state = json.loads((run_path / "search_state.json").read_text(encoding="utf-8"))
    baseline_path = state.get("baseline_path")
    if not isinstance(baseline_path, str) or not baseline_path:
        raise ValueError("run state requires a frozen baseline_path")
    source = Path(baseline_path)
    adapter.baseline_path = (
        source.resolve() if source.is_absolute() else (run_path / source).resolve()
    )
    return adapter


def _read_dataset_protocol(path: str | Path) -> dict:
    source = Path(path).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("dataset protocol must be a JSON object")
    resolved = json.loads(json.dumps(data))
    data_root = Path(str(resolved.get("data_root", source.parent))).resolve()
    for split in ("train", "validation"):
        value = Path(str(resolved[split]["source_path"]))
        resolved[split]["source_path"] = str(
            value.resolve() if value.is_absolute() else (data_root / value).resolve()
        )
    for field in ("manifest_path", "data_path"):
        value = Path(str(resolved["blind"][field]))
        resolved["blind"][field] = str(value.resolve() if value.is_absolute() else (data_root / value).resolve())
    resolved["data_root"] = str(data_root)
    return resolved


def _default_dataset_protocol(root: Path, tsplib_data_path: Path) -> dict:
    train_ids = tuple(f"train_{index:04d}" for index in range(64))
    return {
        "protocol_id": "tsp100_main_v1",
        "train": {
            "format": "pickle",
            "source_path": str((root / "TrainingData" / "TSPAEL64.pkl").resolve()),
            "instance_ids": list(train_ids),
            "family_by_instance": {
                instance_id: "random_euclidean_tsp100"
                for instance_id in train_ids
            },
        },
        "validation": {
            "format": "tsplib",
            "source_path": str(tsplib_data_path.resolve()),
            "instance_ids": list(TSPLIB29_INSTANCES),
            "family_by_instance": {
                instance_id: _tsplib_family(instance_id)
                for instance_id in TSPLIB29_INSTANCES
            },
        },
        "blind": {
            "commitment": (
                "sha256:e5b5cdb6e9140b362d1da76ae634210a63756ddb"
                "b659e4f41c6830531bb3c115"
            ),
            "manifest_path": str((root / "BlindData" / "manifest.json").resolve()),
            "data_path": str(tsplib_data_path.resolve()),
        },
    }


def _validate_dataset_protocol(
    protocol: dict,
    *,
    data_root: str | Path | None = None,
) -> dict:
    value = json.loads(json.dumps(protocol))
    root = (
        Path(data_root).resolve()
        if data_root is not None
        else Path(__file__).resolve().parents[3]
    )
    unknown = set(value) - {"protocol_id", "train", "validation", "blind", "metadata", "data_root"}
    if unknown:
        raise ValueError(f"unknown dataset protocol fields: {sorted(unknown)}")
    if not str(value.get("protocol_id", "")).strip():
        raise ValueError("dataset protocol requires protocol_id")
    for split in ("train", "validation"):
        record = value.get(split)
        if not isinstance(record, dict):
            raise ValueError(f"dataset protocol requires {split}")
        expected = {"format", "source_path", "instance_ids", "family_by_instance"}
        if set(record) != expected:
            raise ValueError(f"{split} dataset fields must be {sorted(expected)}")
        if record["format"] not in {"pickle", "tsplib"}:
            raise ValueError(f"unsupported {split} dataset format")
        if split == "train" and record["format"] != "pickle":
            raise ValueError("training data must use the indexed pickle contract")
        raw_source = Path(str(record["source_path"]))
        source = (
            raw_source.resolve()
            if raw_source.is_absolute()
            else (root / raw_source).resolve()
        )
        if not source.exists():
            raise FileNotFoundError(source)
        record["source_path"] = str(source)
        instance_ids = tuple(str(item) for item in record["instance_ids"])
        if not instance_ids or len(set(instance_ids)) != len(instance_ids):
            raise ValueError(f"{split} instance ids must be non-empty and unique")
        families = {
            str(key): str(family)
            for key, family in dict(record["family_by_instance"]).items()
        }
        if set(families) != set(instance_ids):
            raise ValueError(f"{split} requires one family for every instance")
        record["instance_ids"] = list(instance_ids)
        record["family_by_instance"] = families
    blind = value.get("blind")
    if not isinstance(blind, dict):
        raise ValueError("dataset protocol requires blind commitment metadata")
    allowed_blind_fields = {"commitment", "manifest_path", "data_path"}
    if set(blind) - allowed_blind_fields:
        raise ValueError("blind dataset contains unknown fields")
    has_manifest = "manifest_path" in blind
    has_data = "data_path" in blind
    if has_manifest != has_data:
        raise ValueError("blind manifest_path and data_path must be supplied together")
    if has_manifest:
        for field in ("manifest_path", "data_path"):
            raw_path = Path(str(blind[field]))
            path = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (root / raw_path).resolve()
            )
            if not path.exists():
                raise FileNotFoundError(path)
            blind[field] = str(path)
    commitment = blind.get("commitment")
    if commitment is not None and not str(commitment).strip():
        raise ValueError("blind commitment cannot be empty")
    train_ids = set(value["train"]["instance_ids"])
    validation_ids = set(value["validation"]["instance_ids"])
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(
            f"train and validation instances must be disjoint: {sorted(overlap)}"
        )
    return value


def _tsplib_family(instance_id: str) -> str:
    stem = Path(instance_id).stem
    family = "".join(character for character in stem if character.isalpha())
    return family.lower() or "unknown"
