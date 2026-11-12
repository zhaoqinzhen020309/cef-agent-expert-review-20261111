from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, Callable


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    root_dir: Path
    entry_class_name: str
    solver_filename: str
    gap_unit: str
    evaluator_name: str


@dataclass(frozen=True)
class EntryContract:
    class_name: str
    method_name: str
    parameters: tuple[str, ...]
    input_kind: str = "array"
    output_kind: str = "permutation"
    python_random_seed_policy: str = "solver_managed"

    def __post_init__(self) -> None:
        if self.python_random_seed_policy not in {"solver_managed", "reset_per_instance"}:
            raise ValueError("unsupported Python random seed policy")
        if self.input_kind not in {"array", "mapping", "constructive_tsp"}:
            raise ValueError("unsupported execution input kind")
        if self.output_kind not in {"permutation", "routes", "structured"}:
            raise ValueError("unsupported execution output kind")


@dataclass(frozen=True)
class DatasetManifest:
    split: str
    source_path: Path | None
    instance_ids: tuple[str, ...]
    exposed_to_search: bool
    family_by_instance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.family_by_instance) - set(self.instance_ids)
        if unknown:
            raise ValueError(f"families reference unknown instances: {sorted(unknown)}")


@dataclass(frozen=True)
class EvaluationBudget:
    n_instances: int = 64
    time_limit: float = 10.0
    ite_max: int = 1000
    seed: int = 2024
    seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.n_instances <= 0:
            raise ValueError("n_instances must be positive")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.ite_max <= 0:
            raise ValueError("ite_max must be positive")
        if len(set(self.resolved_seeds)) != len(self.resolved_seeds):
            raise ValueError("evaluation seeds must be unique")

    @property
    def resolved_seeds(self) -> tuple[int, ...]:
        return self.seeds or (self.seed,)


@dataclass(frozen=True)
class CandidateTask:
    candidate_id: str
    candidate_path: Path
    evaluator_module: str
    evaluator_function: str
    budget: EvaluationBudget
    gap_unit: str = "unknown"
    contract_id: str = "unknown"
    split: str = "train"
    instance_ids: tuple[str, ...] = ()
    data_path: Path | None = None
    collect_diagnostics: bool = False
    metric_name: str = "optimality_gap"
    metric_unit: str = "percent"


@dataclass(frozen=True)
class LazyInstance:
    """Trusted loader descriptor. Numeric inputs alone are passed to candidates."""
    loader: Callable
    source_path: str
    record: dict
    kind: str

    def materialize(self):
        return self.loader(self.source_path, self.record, self.kind)


@dataclass
class CandidateLoadResult:
    candidate_id: str
    valid: bool
    strategy_cls: type | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class CandidateExecutionResult:
    candidate_id: str
    valid: bool
    routes: list[object] | None
    runtime_sec: float
    routes_by_seed: list[list[object]] | None = None
    runtime_matrix: list[list[float]] | None = None
    termination_matrix: list[list[str]] | None = None
    diagnostics_matrix: list[list[dict[str, object]]] | None = None
    seeds: list[int] | None = None
    error_type: str | None = None
    error_message: str | None = None
    timeout_timing: dict[str, object] | None = None


@dataclass
class EvaluationResult:
    candidate_id: str
    valid: bool
    score: float | None
    gap_unit: str
    runtime_sec: float
    contract_id: str = "unknown"
    error_type: str | None = None
    error_message: str | None = None
    instance_gaps: list[float] | None = None
    split: str = "train"
    seeds: list[int] | None = None
    instance_ids: list[str] | None = None
    gap_matrix: list[list[float]] | None = None
    runtime_matrix: list[list[float]] | None = None
    termination_matrix: list[list[str]] | None = None
    diagnostics_matrix: list[list[dict[str, object]]] | None = None
    objective_name: str | None = None
    objective_unit: str | None = None
    per_instance_objectives: list[float] | None = None
    objective_matrix: list[list[float]] | None = None
    timeout_timing: dict[str, object] | None = None
    metric_name: str = "optimality_gap"
    metric_unit: str = "percent"

    def to_record(self) -> dict:
        return asdict(self)


class ProblemAdapter(Protocol):
    name: str
    spec: ProblemSpec
    entry_contract: EntryContract
    baseline_path: Path
    train_manifest: DatasetManifest
    validation_manifest: DatasetManifest
    blind_commitment: str
    blind_manifest_path: Path | None
    blind_data_path: Path | None

    def problem_context(self) -> str:
        ...

    def make_task(
        self,
        *,
        candidate_id: str,
        candidate_path: Path,
        budget: EvaluationBudget,
        split: str,
        instance_ids: tuple[str, ...] = (),
    ) -> CandidateTask:
        ...
