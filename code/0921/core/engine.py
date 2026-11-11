from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import threading
import time
import uuid
from math import isclose
from statistics import fmean
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from agent_ahd.agents.agent_runner import AgentResponse, AgentResponseError, AgentRunner
from agent_ahd.agents.candidate_agent import (
    CODE_AGENT_SYSTEM_PROMPT,
    GENERATED_ARTIFACT_NAMES,
    HYPOTHESIS_AGENT_SYSTEM_PROMPT,
    hypothesis_system_prompt,
    CandidateArtifact,
    CandidateWorkspace,
    apply_code_implementation_response,
    build_code_hypothesis_schema,
    build_code_implementation_schema,
    build_code_implementation_task,
    build_planner_candidate_task,
    collect_candidate_artifact,
    parse_code_hypothesis_response,
    prepare_search_candidate_workspace,
    request_code_hypothesis,
    request_code_implementation_response,
)
from agent_ahd.agents.controller_agent import (
    CONTROLLER_SYSTEM_PROMPT,
    build_controller_context,
    build_controller_prompt_context,
)
from agent_ahd.core.archive import (
    ArchiveSnapshot,
    build_archive_snapshot,
    restrict_archive_snapshot_to_mean,
)
from agent_ahd.core.attempt_ledger import (
    ATTEMPT_LEDGER_FILENAME,
    LEDGER_VERSION,
    AttemptLedger,
)
from agent_ahd.core.agent_protocol import is_explicit_infrastructure_failure
from agent_ahd.core.backend_factory import backend_descriptor
from agent_ahd.core.code_checks import (
    CodeCheckResult,
    changed_symbols_in_scope,
    check_ablation_reversion,
    check_scope_tree_change,
)
from agent_ahd.core.decision_catalog import (
    DecisionCatalog,
    controller_selection_schema,
    materialize_controller_selection,
)
from agent_ahd.core.evidence import update_hypothesis_evidence
from agent_ahd.core.eoh_timeout import (
    candidate_execution_contract,
    probe_candidate_process_tree_capability,
)
from agent_ahd.core.mechanism_transfer import (
    MechanismTransferRecord,
    build_mechanism_transfer_record,
    create_mechanism_transfer_registry,
    load_mechanism_transfer_registry,
    registry_compatibility_context,
)
from agent_ahd.core.parallel_evaluator import evaluate_tasks
from agent_ahd.core.prompt_projection import (
    PROMPT_PROJECTION_VERSION,
    prompt_contract,
    projection_source_sha256,
)
from agent_ahd.core.preflight import (
    build_root_contract,
    collect_environment_fingerprint,
    validate_root_contract,
    verify_dependency_lock,
)
from agent_ahd.core.planner_policy import materialize_round_plan, validate_round_plan
from agent_ahd.core.protocol_versions import CURRENT_PROTOCOL_SCHEMA, PROTOCOL_VERSIONS
from agent_ahd.core.research_records import (
    ControllerSelection,
    EvaluationRecord,
    EvidenceRecord,
    InheritanceChannel,
    PlannerDecision,
    RoundPlan,
    ScopeSpec,
    SearchAction,
)
from agent_ahd.core.source_scope import (
    build_source_exposure,
    derive_scope_boundary_evidence,
)
from agent_ahd.core.search_state import (
    SearchConfig,
    SearchState,
    SolverNode,
    load_search_state,
    save_search_state,
)
from agent_ahd.core.search_diagnostics import (
    build_search_diagnostics,
    classify_source_change,
    extract_source_structure,
    source_structure_fingerprint,
)
from agent_ahd.core.stage_journal import (
    CLOSURE_GATE_ORDER,
    TERMINAL_STAGE_STATUSES,
    CandidateStageJournal,
    RoundAttemptJournal,
    RoundClosureJournal,
)
from agent_ahd.core.staged_evaluation import (
    EvaluationCache,
    EvaluationFingerprint,
    quick_screen_record,
)
from agent_ahd.problems.base import (
    CandidateTask,
    EvaluationBudget,
    EvaluationResult,
    ProblemAdapter,
)


PLANNER_CONFIG_FILENAME = "planner_config.json"
MODEL_PROTOCOL_FILENAME = "model_protocol.json"
RESOURCE_CLOCK_FILENAME = "resource_clock.json"
STATE_FILENAME = "search_state.json"
BASELINE_CONTROL_ID = "__baseline_control__"
BLIND_EVALUATION_LOCK_FILENAME = "blind_evaluation_lock.json"
FINAL_CONFIRMATION_FILENAME = "final_confirmation.json"
PROTOCOL_GATE_FILENAME = "protocol_gate.json"
ROOT_PROTOCOL_FILENAME = "protocol_contract.json"
MECHANISM_TRANSFER_REGISTRY_FILENAME = "mechanism_transfer_registry.json"
EXPERIMENT_PHASES = {"pilot", "formal", "benchmark", "legacy"}
CONTROLLER_STOP_POLICIES = {"adaptive", "fixed_rounds"}
GATE_MODES = {"observe_only", "feasibility_stop"}
FRAMEWORK_MODES = {"on", "off"}
SEARCH_POLICIES = {"scalar_baseline", "planner"}
PLANNER_VARIANTS = {
    "full",
    "without_evidence",
    "without_risk_archive",
    "without_controlled_ablation",
}


def _primary_inheritance_source(
    decision: PlannerDecision,
    *,
    parent_source: Path | None,
    parent_id: str | None,
    baseline_path: str | Path,
) -> tuple[Path, str]:
    """Resolve source access before creating a candidate workspace."""
    baseline = Path(baseline_path).resolve()
    if decision.inheritance_channel == InheritanceChannel.INDEPENDENT:
        return baseline, BASELINE_CONTROL_ID
    if parent_source is None:
        return baseline, BASELINE_CONTROL_ID
    return Path(parent_source).resolve(), str(parent_id)


def _record_quick_screen_skip(
    journal: CandidateStageJournal,
    source_sha256: str,
) -> None:
    record = quick_screen_record()
    if not journal.begin("quick_screen", input_fingerprint=source_sha256):
        return
    journal.finish(
        "quick_screen",
        input_fingerprint=source_sha256,
        output_fingerprint=_text_sha256(json.dumps(record, sort_keys=True)),
        status="skipped",
        metadata=record,
    )


def _reject_and_skip_candidate_stages(
    journal: CandidateStageJournal,
    *,
    static_input: str,
    error: str,
) -> None:
    static = journal.record("static_check")
    if static is not None and static["status"] == "running":
        journal.finish(
            "static_check",
            input_fingerprint=static_input,
            output_fingerprint=_text_sha256(error),
            status="rejected",
            metadata={"error": error},
        )
    previous = _text_sha256(error)
    for stage in ("quick_screen", "development_evaluation", "evidence_validation"):
        journal.begin(stage, input_fingerprint=previous)
        journal.finish(
            stage,
            input_fingerprint=previous,
            output_fingerprint=previous,
            status="skipped",
            metadata={"reason": "candidate rejected before this stage"},
        )


def _finish_evidence_validation_stages(
    jobs: dict[str, dict],
    evidence_updates: list[EvidenceRecord],
) -> None:
    evidence_by_candidate = {
        item.candidate_id: item for item in evidence_updates
    }
    for candidate_id, job in jobs.items():
        journal = job["stage_journal"]
        if journal.record("evidence_validation") is not None:
            continue
        development = journal.record("development_evaluation")
        if development is None:
            raise RuntimeError("evidence stage has no development predecessor")
        input_fingerprint = development["output_fingerprint"]
        journal.begin("evidence_validation", input_fingerprint=input_fingerprint)
        evidence = evidence_by_candidate.get(candidate_id)
        if evidence is None or evidence.status == "pending":
            metadata = {
                "reason": "ordinary candidate awaits elite promotion",
                "evidence_id": evidence.evidence_id if evidence else None,
            }
            status = "skipped"
        else:
            metadata = {
                "evidence_id": evidence.evidence_id,
                "evidence_status": evidence.status,
            }
            status = "complete"
        journal.finish(
            "evidence_validation",
            input_fingerprint=input_fingerprint,
            output_fingerprint=_text_sha256(json.dumps(metadata, sort_keys=True)),
            status=status,
            metadata=metadata,
        )


def _evaluation_fingerprint(
    task: CandidateTask,
    *,
    source_sha256: str,
) -> EvaluationFingerprint:
    manifest_payload = {
        "split": task.split,
        "data_path": str(task.data_path.resolve()) if task.data_path else None,
        "instance_ids": task.instance_ids,
    }
    diagnostics_payload = {
        "collect_diagnostics": task.collect_diagnostics,
        "schema": "diagnostics_matrix_v1",
    }
    return EvaluationFingerprint(
        source_sha256=source_sha256,
        evaluator_sha256=_evaluator_source_sha256(
            task.evaluator_module,
            task.evaluator_function,
        ),
        manifest_sha256=_text_sha256(
            json.dumps(manifest_payload, sort_keys=True)
        ),
        split=task.split,
        instance_ids=task.instance_ids,
        seeds=task.budget.resolved_seeds,
        time_limit_sec=task.budget.time_limit,
        iteration_limit=task.budget.ite_max,
        metric_name=task.metric_name,
        metric_unit=task.metric_unit,
        diagnostics_schema_sha256=_text_sha256(
            json.dumps(diagnostics_payload, sort_keys=True)
        ),
        stage="development_evaluation",
    )


def _evaluator_source_sha256(module_name: str, function_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot resolve evaluator module: {module_name}")
    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise RuntimeError(f"evaluator module has no source file: {module_name}")
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(b"\0")
    digest.update(function_name.encode("utf-8"))
    return digest.hexdigest()


def _matching_cached_state_record(
    state: SearchState,
    *,
    source_sha256: str,
    task: CandidateTask,
) -> EvaluationRecord | None:
    candidate_ids = {
        node.node_id
        for node in state.nodes
        if node.solver_sha256 == source_sha256
    }
    for record in reversed(state.evaluations):
        if (
            record.candidate_id in candidate_ids
            and record.split == task.split
            and record.seeds == task.budget.resolved_seeds
            and record.instance_ids == task.instance_ids
            and (record.metric_name, record.metric_unit) == (task.metric_name, task.metric_unit)
        ):
            return record
    return None


def _hydrate_locked_plan(state: SearchState, plan: RoundPlan) -> None:
    decisions_by_id = {item.decision_id: item for item in plan.decisions}
    for index, existing in enumerate(state.decisions):
        locked = decisions_by_id.get(existing.decision_id)
        if locked is not None:
            state.decisions[index] = locked
    known_hypotheses = {item.hypothesis_id for item in state.hypotheses}
    for decision in plan.decisions:
        if decision.hypothesis is None:
            continue
        if decision.hypothesis.hypothesis_id not in known_hypotheses:
            state.hypotheses.append(decision.hypothesis)
            known_hypotheses.add(decision.hypothesis.hypothesis_id)
        for node in state.nodes:
            if node.decision_id == decision.decision_id:
                node.hypothesis_id = decision.hypothesis.hypothesis_id


def _previous_visible_candidate_ids(
    run_dir: Path,
    completed_round: int,
) -> set[str]:
    if completed_round < 0:
        return set()
    path = (
        run_dir
        / "rounds"
        / f"round_{completed_round:03d}"
        / "search_diagnostics.json"
    )
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("visible_candidate_ids", ()))


def _serialize_controller_prompt(context: dict) -> str:
    return json.dumps(context, sort_keys=True, separators=(",", ":"))


class TransactionAmbiguityError(RuntimeError):
    """A dispatched external side effect cannot be proven safe to replay."""


class _PlannerRunFileLock:
    """Hold an OS-released exclusive byte lock for one planner run."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir.resolve() / ".planner_transaction.lock"
        self._handle = None

    def __enter__(self) -> "_PlannerRunFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                f"planner run is already active: {self.path.parent}"
            ) from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


@dataclass(frozen=True)
class PlannerEngineConfig:
    problem_name: str = "tsp_gls"
    experiment_phase: str = "pilot"
    controller_stop_policy: str = "adaptive"
    framework_mode: str = "on"
    enable_research_guidance: bool = False
    enable_baseline_parent: bool = False
    enable_deterministic_ablation: bool = False
    planner_variant: str = "full"
    total_rounds: int = 20
    batch_size: int = 5
    archive_view_size: int = 5
    tail_fraction: float = 0.1
    agent_workers: int = 5
    evaluation_workers: int = 5
    n_instances: int = 64
    time_limit: float = 10.0
    ite_max: int = 1000
    base_seeds: tuple[int, ...] = (2024,)
    diagnostic_seeds: tuple[int, ...] = (2025,)
    confirmation_seeds: tuple[int, ...] = (2027, 2028)
    blind_seeds: tuple[int, ...] = (2029, 2030, 2031)
    candidate_budget: int | None = None
    candidate_evaluation_unit_budget: int | None = None
    diagnostic_evaluation_unit_budget: int = 0
    confirmation_finalist_count: int = 5
    include_stable_control_in_shortlist: bool = True
    confirmation_evaluation_unit_budget: int | None = None
    model_call_budget: int | None = None
    model_token_budget: int | None = None
    wall_clock_budget_sec: float | None = None
    protocol_mode: str = "compatibility"
    agent_call_max_tokens: int = 50_000
    agent_call_max_wall_clock_sec: float = 3_600.0
    inheritance_experiment_arm: str | None = None
    inheritance_donor_id: str | None = None
    inheritance_scaffold_id: str | None = None
    scope_experiment_arm: str | None = None
    scope_experiment_parent_id: str | None = None
    scope_experiment_initial_symbols: tuple[str, ...] = ()
    gate_mode: str = "observe_only"
    min_valid_candidates: int | None = None
    min_exact_unique_candidates: int | None = None
    min_structural_unique_candidates: int | None = None

    def __post_init__(self) -> None:
        if self.experiment_phase not in EXPERIMENT_PHASES:
            raise ValueError(f"unknown experiment phase: {self.experiment_phase}")
        if self.experiment_phase == "benchmark":
            if self.controller_stop_policy != "fixed_rounds" or self.candidate_budget is None:
                raise ValueError("benchmark requires preregistered fixed rounds and proposal budget")
            if self.model_token_budget is None or self.wall_clock_budget_sec is None:
                raise ValueError("benchmark requires explicit model and wall-clock budgets")
            if self.diagnostic_evaluation_unit_budget != 0:
                raise ValueError("benchmark counts all evaluation proposals in the candidate budget; extra retests are disabled")
        if self.controller_stop_policy not in CONTROLLER_STOP_POLICIES:
            raise ValueError(
                "unknown controller stop policy: "
                f"{self.controller_stop_policy}"
            )
        if self.framework_mode not in FRAMEWORK_MODES:
            raise ValueError(f"unknown framework mode: {self.framework_mode}")
        if type(self.enable_research_guidance) is not bool:
            raise ValueError("enable_research_guidance must be boolean")
        if type(self.enable_baseline_parent) is not bool:
            raise ValueError("enable_baseline_parent must be boolean")
        if self.enable_baseline_parent and (self.inheritance_experiment_arm or self.scope_experiment_arm):
            raise ValueError("baseline parent requires a separate experiment without legacy inheritance/scope arms")
        if self.enable_research_guidance and (self.framework_mode == "off" or self.inheritance_experiment_arm or self.scope_experiment_arm):
            raise ValueError("research guidance requires a separate framework-on experiment without legacy inheritance/scope arms")
        if type(self.enable_deterministic_ablation) is not bool:
            raise ValueError("enable_deterministic_ablation must be boolean")
        if self.enable_deterministic_ablation and (self.framework_mode == "off" or self.inheritance_experiment_arm or self.scope_experiment_arm):
            raise ValueError("deterministic ablation requires a separate framework-on experiment")
        if self.planner_variant not in PLANNER_VARIANTS:
            raise ValueError(f"unknown planner variant: {self.planner_variant}")
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"unknown gate mode: {self.gate_mode}")
        if self.total_rounds <= 0 or self.batch_size <= 0:
            raise ValueError("round and batch counts must be positive")
        if self.agent_workers <= 0 or self.evaluation_workers <= 0:
            raise ValueError("agent and evaluation worker counts must be positive")
        if self.n_instances <= 0 or self.time_limit <= 0 or self.ite_max <= 0:
            raise ValueError("evaluation budget values must be positive")
        if self.archive_view_size <= 0:
            raise ValueError("archive_view_size must be positive")
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        seed_groups = (
            self.base_seeds,
            self.diagnostic_seeds,
            self.confirmation_seeds,
            self.blind_seeds,
        )
        if any(not seeds for seeds in seed_groups):
            raise ValueError("evaluation seed sets cannot be empty")
        if self.candidate_budget is not None and self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")
        resolved_candidate_budget = (
            self.candidate_budget or self.total_rounds * self.batch_size
        )
        for field_name in (
            "min_valid_candidates",
            "min_exact_unique_candidates",
            "min_structural_unique_candidates",
        ):
            threshold = getattr(self, field_name)
            if threshold is None:
                continue
            if isinstance(threshold, bool) or not isinstance(threshold, int):
                raise ValueError(f"{field_name} must be an integer")
            if threshold <= 0 or threshold > resolved_candidate_budget:
                raise ValueError(
                    f"{field_name} must be within the proposal-slot budget"
                )
        if (
            self.candidate_evaluation_unit_budget is not None
            and self.candidate_evaluation_unit_budget <= 0
        ):
            raise ValueError("candidate_evaluation_unit_budget must be positive")
        if self.diagnostic_evaluation_unit_budget < 0:
            raise ValueError("diagnostic_evaluation_unit_budget cannot be negative")
        if (
            self.experiment_phase == "formal"
            and self.diagnostic_evaluation_unit_budget == 0
        ):
            raise ValueError("formal runs require a diagnostic evaluation budget")
        if self.confirmation_finalist_count <= 0:
            raise ValueError("confirmation_finalist_count must be positive")
        if (
            self.experiment_phase == "formal"
            and not self.include_stable_control_in_shortlist
        ):
            raise ValueError("formal runs cannot disable the stable control")
        if (
            self.confirmation_evaluation_unit_budget is not None
            and self.confirmation_evaluation_unit_budget < 0
        ):
            raise ValueError("confirmation evaluation budget cannot be negative")
        if (
            self.experiment_phase == "formal"
            and self.confirmation_evaluation_unit_budget == 0
        ):
            raise ValueError("formal runs require a confirmation evaluation budget")
        if self.model_call_budget is not None and self.model_call_budget <= 0:
            raise ValueError("model_call_budget must be positive")
        if self.model_token_budget is not None and self.model_token_budget <= 0:
            raise ValueError("model_token_budget must be positive")
        if self.wall_clock_budget_sec is not None and self.wall_clock_budget_sec <= 0:
            raise ValueError("wall_clock_budget_sec must be positive")
        if self.protocol_mode not in {"compatibility", "v4"}:
            raise ValueError("protocol_mode must be compatibility or v4")
        if type(self.agent_call_max_tokens) is not int or self.agent_call_max_tokens <= 0:
            raise ValueError("agent_call_max_tokens must be a positive integer")
        if (
            not isinstance(self.agent_call_max_wall_clock_sec, (int, float))
            or isinstance(self.agent_call_max_wall_clock_sec, bool)
            or self.agent_call_max_wall_clock_sec <= 0
        ):
            raise ValueError("agent_call_max_wall_clock_sec must be positive")
        inheritance_arms = {
            "full_source",
            "local_source",
            "mechanism_only",
            "agent_choice",
        }
        if (
            self.inheritance_experiment_arm is not None
            and self.inheritance_experiment_arm not in inheritance_arms
        ):
            raise ValueError(
                "unknown inheritance experiment arm: "
                f"{self.inheritance_experiment_arm}"
            )
        if self.inheritance_experiment_arm is not None:
            if not self.inheritance_donor_id:
                raise ValueError("inheritance experiments require a donor id")
            if self.inheritance_experiment_arm in {
                "mechanism_only",
                "agent_choice",
            } and not self.inheritance_scaffold_id:
                raise ValueError(
                    "mechanism inheritance experiments require a scaffold id"
                )
        scope_arms = {
            "S1_fixed_single",
            "S2_fixed_full",
            "S3_dynamic",
        }
        if (
            self.scope_experiment_arm is not None
            and self.scope_experiment_arm not in scope_arms
        ):
            raise ValueError(
                "unknown scope experiment arm: "
                f"{self.scope_experiment_arm}"
            )
        if self.scope_experiment_arm is not None:
            if self.inheritance_experiment_arm is not None:
                raise ValueError(
                    "inheritance and scope experiments are mutually exclusive"
                )
            if not self.scope_experiment_parent_id:
                raise ValueError("scope experiments require a parent id")
            if len(self.scope_experiment_initial_symbols) != 1:
                raise ValueError(
                    "scope experiments require exactly one initial symbol"
                )
        if self.framework_mode == "off" and (
            self.inheritance_experiment_arm is not None
            or self.scope_experiment_arm is not None
        ):
            raise ValueError(
                "framework off mode cannot be combined with inheritance or scope arms"
            )
        seed_sets = [
            set(self.base_seeds),
            set(self.diagnostic_seeds),
            set(self.confirmation_seeds),
            set(self.blind_seeds),
        ]
        if any(
            len(seed_set) != len(seed_values)
            for seed_set, seed_values in zip(
                seed_sets,
                seed_groups,
            )
        ):
            raise ValueError("seed sets cannot contain duplicates")
        if any(
            seed_sets[first] & seed_sets[second]
            for first in range(len(seed_sets))
            for second in range(first + 1, len(seed_sets))
        ):
            raise ValueError(
                "base, diagnostic, confirmation, and blind seeds must be disjoint"
            )

    def base_budget(self) -> EvaluationBudget:
        return EvaluationBudget(
            n_instances=self.n_instances,
            time_limit=self.time_limit,
            ite_max=self.ite_max,
            seed=self.base_seeds[0],
            seeds=self.base_seeds,
        )

    @property
    def resolved_candidate_budget(self) -> int:
        return self.candidate_budget or self.total_rounds * self.batch_size

    @property
    def resolved_candidate_evaluation_unit_budget(self) -> int:
        return self.candidate_evaluation_unit_budget or (
            self.resolved_candidate_budget * len(self.base_seeds) * self.n_instances
        )

    @property
    def resolved_diagnostic_evaluation_unit_budget(self) -> int:
        return self.diagnostic_evaluation_unit_budget

    @property
    def resolved_model_call_budget(self) -> int:
        return self.model_call_budget or self.total_rounds * (1 + 2 * self.batch_size)

    @property
    def resolved_protocol_model_call_budget(self) -> int:
        return self.resolved_model_call_budget * 3

    @property
    def controller_stop_is_disabled(self) -> bool:
        return (
            self.experiment_phase == "formal"
            or self.controller_stop_policy == "fixed_rounds"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PlannerEngineConfig":
        payload = dict(data)
        payload.setdefault("experiment_phase", "legacy")
        payload.setdefault("controller_stop_policy", "adaptive")
        payload.setdefault("framework_mode", "on")
        legacy_budget = payload.pop("evaluation_unit_budget", None)
        payload.setdefault("candidate_evaluation_unit_budget", legacy_budget)
        payload["base_seeds"] = tuple(payload["base_seeds"])
        payload["diagnostic_seeds"] = tuple(
            payload.get("diagnostic_seeds", (2025,))
        )
        payload["confirmation_seeds"] = tuple(payload["confirmation_seeds"])
        payload["blind_seeds"] = tuple(
            payload.get("blind_seeds", (2029, 2030, 2031))
        )
        payload.setdefault("inheritance_experiment_arm", None)
        payload.setdefault("inheritance_donor_id", None)
        payload.setdefault("inheritance_scaffold_id", None)
        payload.setdefault("scope_experiment_arm", None)
        payload.setdefault("scope_experiment_parent_id", None)
        payload["scope_experiment_initial_symbols"] = tuple(
            payload.get("scope_experiment_initial_symbols", ())
        )
        payload.setdefault("protocol_mode", "compatibility")
        payload.setdefault("agent_call_max_tokens", 50_000)
        payload.setdefault("agent_call_max_wall_clock_sec", 3_600.0)
        return cls(**payload)


def initialize_planner_run(
    run_dir: str | Path,
    *,
    adapter: ProblemAdapter,
    config: PlannerEngineConfig,
    stable_candidate_path: str | Path | None = None,
    policy: str = "planner",
    initial_size: int | None = None,
    mechanism_transfer_registry_path: str | Path | None = None,
) -> SearchState:
    run_path = Path(run_dir).resolve()
    dependency_lock_path = Path(__file__).resolve().parents[1] / "requirements.runtime.lock"
    locked_dependencies = verify_dependency_lock(dependency_lock_path)
    free_space_bytes = shutil.disk_usage(run_path.parent).free
    if free_space_bytes < 512 * 1024 * 1024:
        raise RuntimeError("preflight requires at least 512 MiB free space")
    execution_contract = candidate_execution_contract()
    process_tree_contract = execution_contract["process_tree_contract"]
    process_tree_probe = probe_candidate_process_tree_capability()
    if (
        process_tree_probe.get("status") != "passed"
        or process_tree_probe.get("backend") != process_tree_contract["backend"]
        or process_tree_probe.get("assignment")
        != process_tree_contract["assignment"]
    ):
        raise RuntimeError("candidate process-tree capability probe changed")
    sandbox_capabilities = {
        "profile_version": PROTOCOL_VERSIONS["sandbox"],
        "backend": "direct_process",
        "filesystem_isolation": False,
        "network_isolation": False,
        "process_tree_isolation": True,
        "process_tree_probe": dict(process_tree_probe),
        "restricted_identity": False,
        "details": (
            "Candidate evaluation binds the worker to "
            f"{process_tree_contract['backend']} before candidate import and "
            "requires the complete process tree to be empty before returning; "
            "filesystem, network, and restricted-identity isolation remain absent.",
        ),
    }
    if run_path.exists() and any(run_path.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_path}")
    if policy not in SEARCH_POLICIES:
        raise ValueError(f"unknown search policy: {policy}")
    from agent_ahd.core.final_selection import (
        STABLE_CONTROL_ID,
        required_confirmation_evaluation_units,
    )

    stable_source = (
        Path(stable_candidate_path).resolve()
        if stable_candidate_path is not None
        else None
    )
    if config.experiment_phase == "formal" and stable_source is None:
        raise ValueError("formal runs require a preregistered stable candidate")
    if stable_source is not None and not stable_source.is_file():
        raise FileNotFoundError(f"stable candidate does not exist: {stable_source}")

    locked_artifacts = _protocol_artifact_hashes(adapter)
    train_manifest_record = _manifest_record(adapter.train_manifest)
    validation_manifest_record = _manifest_record(adapter.validation_manifest)
    registry_expected_context = registry_compatibility_context(
        locked_artifacts=locked_artifacts,
        train_manifest=train_manifest_record,
        validation_manifest=validation_manifest_record,
    )
    source_registry = None
    if mechanism_transfer_registry_path is None:
        registry_payload = create_mechanism_transfer_registry(
            (),
            registry_context=registry_expected_context,
        )
    else:
        source_registry = Path(mechanism_transfer_registry_path).resolve()
        load_mechanism_transfer_registry(
            source_registry,
            expected_context=registry_expected_context,
        )
        registry_payload = json.loads(source_registry.read_text(encoding="utf-8"))

    required_confirmation_units = required_confirmation_evaluation_units(
        adapter,
        config,
    )
    configured_confirmation_units = config.confirmation_evaluation_unit_budget
    if (
        config.experiment_phase == "formal"
        and configured_confirmation_units is None
    ):
        raise ValueError("formal runs require an explicit confirmation evaluation budget")
    if (
        config.experiment_phase == "formal"
        and configured_confirmation_units < required_confirmation_units
    ):
        raise ValueError(
            "confirmation evaluation budget is smaller than the complete "
            f"shortlist protocol: {configured_confirmation_units} < "
            f"{required_confirmation_units}"
        )
    run_path.mkdir(parents=True, exist_ok=True)
    state = SearchState.create(
        run_id=run_path.name,
        baseline_path=str(adapter.baseline_path.resolve()),
        baseline_sha256=_file_sha256(adapter.baseline_path),
        config=SearchConfig(
            batch_size=config.batch_size,
            initial_size=initial_size or config.batch_size,
            random_seed=config.base_seeds[0],
        ),
    )
    baseline_control_root = run_path / "controls" / "baseline"
    baseline_control_root.mkdir(parents=True)
    shutil.copy2(adapter.baseline_path, baseline_control_root / "solver.py")
    baseline_control = {
        "solver_relpath": "controls/baseline/solver.py",
        "source_files": _candidate_file_hashes(baseline_control_root),
    }
    stable_control = None
    if stable_source is not None:
        stable_control_root = run_path / "controls" / "stable"
        for source in sorted(stable_source.parent.rglob("*.py")):
            if "__pycache__" in source.parts:
                continue
            target = stable_control_root / source.relative_to(stable_source.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        stable_entry = stable_control_root / stable_source.name
        stable_control = {
            "candidate_id": STABLE_CONTROL_ID,
            "solver_relpath": stable_entry.relative_to(run_path).as_posix(),
            "source_files": _candidate_file_hashes(stable_control_root),
        }

    registry_path = run_path / MECHANISM_TRANSFER_REGISTRY_FILENAME
    if source_registry is None:
        _write_json(registry_path, registry_payload)
    else:
        shutil.copy2(source_registry, registry_path)
        load_mechanism_transfer_registry(
            registry_path,
            expected_hash=str(registry_payload["registry_hash"]),
            expected_context=registry_expected_context,
        )

    source_bundle_id = _text_sha256(
        json.dumps(locked_artifacts, sort_keys=True, separators=(",", ":"))
    )
    environment_fingerprint = collect_environment_fingerprint()
    protocol = {
        "protocol_schema": (
            CURRENT_PROTOCOL_SCHEMA
            if config.protocol_mode == "v4"
            else PROTOCOL_VERSIONS["schema"]
        ),
        "protocol_versions": dict(PROTOCOL_VERSIONS),
        "preflight": {
            "dependency_lock_sha256": _file_sha256(dependency_lock_path),
            "locked_dependencies": locked_dependencies,
            "minimum_free_space_bytes": 512 * 1024 * 1024,
            "source_bundle_id": source_bundle_id,
            "environment": environment_fingerprint,
            "data_hashes": {
                "train": train_manifest_record,
                "validation": validation_manifest_record,
            },
        },
        "sandbox_capabilities": sandbox_capabilities,
        "candidate_execution": execution_contract,
        "policy": policy,
        "initialized_at_epoch_sec": time.time(),
        "planner_variant": (
            config.planner_variant if policy == "planner" else None
        ),
        "engine": config.to_dict(),
        "problem": adapter.name,
        "problem_contract": {
            **asdict(adapter.spec),
            "root_dir": str(adapter.spec.root_dir),
        },
        "entry_contract": asdict(adapter.entry_contract),
        "baseline_sha256": state.baseline_sha256,
        "baseline_control": baseline_control,
        "stable_control": stable_control,
        "mechanism_transfer_registry": {
            "relpath": MECHANISM_TRANSFER_REGISTRY_FILENAME,
            "registry_hash": registry_payload["registry_hash"],
        },
        "dataset_protocol": _search_dataset_protocol(
            adapter.dataset_protocol,
            data_root=getattr(adapter, "data_root", None),
        ),
        "train_manifest": train_manifest_record,
        "validation_manifest": validation_manifest_record,
        "blind_test": {
            "access": "gatekeeper_only",
            "commitment": adapter.blind_commitment,
        },
        "confirmation_protocol": {
            "finalist_count": config.confirmation_finalist_count,
            "include_stable_control_in_shortlist": (
                config.include_stable_control_in_shortlist
            ),
            "seeds": list(config.confirmation_seeds),
            "instance_ids": list(adapter.validation_manifest.instance_ids),
            "objective_semantics": {
                "problem": adapter.name,
                "evaluator_module": adapter.spec.evaluator_name,
                "description": adapter.problem_context(),
            },
            "reserved_evaluation_units": required_confirmation_units,
        },
        "locked_artifacts": locked_artifacts,
    }
    root_contract = build_root_contract(
        config={
            key: value
            for key, value in protocol.items()
            if key not in {"initialized_at_epoch_sec", "locked_artifacts"}
        },
        source_files={
            "locked_artifacts": locked_artifacts,
            "baseline_control": baseline_control["source_files"],
            "stable_control": (
                stable_control["source_files"] if stable_control is not None else None
            ),
            "mechanism_transfer_registry": registry_payload["registry_hash"],
        },
        non_blind_manifest={
            "train": protocol["train_manifest"],
            "validation": protocol["validation_manifest"],
        },
        environment=environment_fingerprint,
        blind_commitment=adapter.blind_commitment,
    )
    protocol["root_protocol_hash"] = root_contract["root_hash"]
    _write_json(run_path / PLANNER_CONFIG_FILENAME, protocol)
    _write_json(run_path / ROOT_PROTOCOL_FILENAME, root_contract)
    _write_json(
        run_path / "preflight.json",
        {
            "status": "passed",
            "free_space_bytes_at_initialization": free_space_bytes,
            **protocol["preflight"],
        },
    )
    save_search_state(run_path / STATE_FILENAME, state)
    return state


class PlannerSearchEngine:
    def __init__(
        self,
        *,
        run_dir: str | Path,
        adapter: ProblemAdapter,
        config: PlannerEngineConfig,
        controller_runner: AgentRunner,
        code_runner: AgentRunner,
        evaluation_function: Callable = evaluate_tasks,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.adapter = adapter
        self.config = config
        self.controller_runner = controller_runner
        self.code_runner = code_runner
        self.evaluation_function = evaluation_function
        self._transaction_mutex = threading.RLock()
        self._transaction_lock_depth = 0

    def run(self) -> SearchState:
        with self._transaction_scope():
            return self._run_locked()

    def _run_locked(self) -> SearchState:
        self._require_search_open()
        state = self.load_state()
        state.require_search_open()
        if (self.run_dir / PROTOCOL_GATE_FILENAME).exists():
            return state
        self._ensure_model_protocol()
        self._ensure_baseline_control(state)
        while state.stopped_reason is None:
            if (
                self._resumable_round_index(state) is None
                and state.current_round + 1 >= self.config.total_rounds
            ):
                break
            self.run_round()
            state = self.load_state()
            if (self.run_dir / PROTOCOL_GATE_FILENAME).exists():
                break
        return state

    @contextmanager
    def _transaction_scope(self):
        with self._transaction_mutex:
            if self._transaction_lock_depth:
                self._transaction_lock_depth += 1
                try:
                    yield
                finally:
                    self._transaction_lock_depth -= 1
                return
            with _PlannerRunFileLock(self.run_dir):
                self._transaction_lock_depth = 1
                try:
                    yield
                finally:
                    self._transaction_lock_depth = 0

    def _ensure_model_protocol(self) -> None:
        has_model_attempts = _model_call_attempts(self.run_dir) > 0
        clock_path = self.run_dir / RESOURCE_CLOCK_FILENAME
        if has_model_attempts and not clock_path.is_file():
            raise RuntimeError("resource clock is missing during resume")
        _start_resource_clock(self.run_dir)
        path = self.run_dir / MODEL_PROTOCOL_FILENAME
        protocol = {
            "controller_model": self.controller_runner.model,
            "controller_reasoning_effort": getattr(
                self.controller_runner,
                "reasoning_effort",
                None,
            ),
            "code_model": self.code_runner.model,
            "code_reasoning_effort": getattr(
                self.code_runner,
                "reasoning_effort",
                None,
            ),
            "response_format": "json_schema",
            "controller_response_format": "json_schema",
            "hypothesis_response_format": "json_schema",
            "code_response_format": "json_schema",
            "protocol_schema": (
                CURRENT_PROTOCOL_SCHEMA
                if self.config.protocol_mode == "v4"
                else PROTOCOL_VERSIONS["schema"]
            ),
            "local_schema_dialect": (
                "json-schema-draft-2020-12"
                if self.config.protocol_mode == "v4"
                else None
            ),
            "max_repair_attempts_per_logical_call": (
                1 if self.config.protocol_mode == "v4" else 0
            ),
            "max_recovery_attempts_per_logical_call": (
                1 if self.config.protocol_mode == "v4" else 0
            ),
            "attempt_semantics_version": (
                "primary_recovery_repair_v1"
                if self.config.protocol_mode == "v4"
                else None
            ),
            "attempt_ledger_version": (
                LEDGER_VERSION if self.config.protocol_mode == "v4" else None
            ),
            "prompt_projection_version": (
                PROMPT_PROJECTION_VERSION
                if self.config.protocol_mode == "v4"
                else None
            ),
            "prompt_projection_source_sha256": (
                projection_source_sha256()
                if self.config.protocol_mode == "v4"
                else None
            ),
            "prompt_contracts": (
                {
                    role: prompt_contract(
                        role,
                        max_output_tokens=self.config.agent_call_max_tokens,
                    )
                    for role in ("controller", "hypothesis", "implementation")
                }
                if self.config.protocol_mode == "v4"
                else None
            ),
        }
        if self.config.protocol_mode == "v4":
            protocol["backend_descriptors"] = {
                "controller": backend_descriptor(self.controller_runner),
                "code": backend_descriptor(self.code_runner),
            }
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != protocol:
                raise RuntimeError("model protocol changed during the run")
            return
        if has_model_attempts:
            raise RuntimeError("model protocol is missing during resume")
        _write_json(path, protocol)

    def load_state(self) -> SearchState:
        state = verify_planner_run_integrity(self.run_dir, self.adapter)
        for round_index in state.completed_rounds:
            round_path = self.run_dir / "rounds" / f"round_{round_index:03d}"
            summary_path = round_path / "round_closure.json"
            journal_path = round_path / "round_closure_journal.json"
            if summary_path.is_file() and not journal_path.is_file():
                raise RuntimeError(
                    f"completed round {round_index} has no closure gate journal"
                )
            if not summary_path.is_file():
                continue
            journal = RoundClosureJournal(journal_path, round_index=round_index)
            records = journal.records()
            if [record["name"] for record in records] != list(CLOSURE_GATE_ORDER):
                raise RuntimeError(
                    f"completed round {round_index} has missing closure gate records"
                )
            if any(record.get("status") != "complete" for record in records):
                raise RuntimeError(
                    f"completed round {round_index} has incomplete closure gates"
                )
            records_by_name = {record["name"]: record for record in records}
            for gate, filename in (
                ("evaluation", "evaluations.json"),
                ("evidence", "evidence_updates.json"),
            ):
                output_path = round_path / filename
                if not output_path.is_file():
                    raise RuntimeError(
                        f"completed round {round_index} has no {gate} stage output"
                    )
                if (
                    records_by_name[gate]["output_fingerprint"]
                    != _file_sha256(output_path)
                ):
                    raise RuntimeError(
                        f"completed round {round_index} {gate} output changed"
                    )
        return state

    def run_round(self, state: SearchState | None = None) -> bool:
        with self._transaction_scope():
            return self._run_round_locked(state)

    def _begin_round_call_group(self, *, round_index: int) -> None:
        if self.config.protocol_mode != "v4":
            return
        worst_case_wall_clock = 6 * self.config.agent_call_max_wall_clock_sec
        if (
            self.config.wall_clock_budget_sec is not None
            and _wall_clock_elapsed(self.run_dir) + worst_case_wall_clock
            > self.config.wall_clock_budget_sec
        ):
            raise RuntimeError(
                "wall-clock budget cannot reserve the complete planner round"
            )
        runners = self._protocol_runners()
        reservation_group_id = f"round:{round_index}:agent-calls"
        logical_calls = 1 + 2 * self.config.batch_size
        for runner in runners:
            runner.begin_call_group(
                reservation_group_id,
                logical_calls=logical_calls,
            )

    def _end_round_call_group(self, *, round_index: int) -> None:
        if self.config.protocol_mode != "v4":
            return
        reservation_group_id = f"round:{round_index}:agent-calls"
        for runner in self._protocol_runners():
            runner.end_call_group(reservation_group_id)

    def _protocol_runners(self) -> tuple[AgentRunner, ...]:
        runners = tuple({
            id(runner): runner
            for runner in (self.controller_runner, self.code_runner)
        }.values())
        if any(not getattr(runner, "protocol_runner", False) for runner in runners):
            raise RuntimeError("v4 planner requires protocol-wrapped agent runners")
        return runners

    def _run_round_locked(self, state: SearchState | None = None) -> bool:
        self._require_search_open()
        self._ensure_model_protocol()
        locked_state = self.load_state()
        if state is not None and state.run_id != locked_state.run_id:
            raise RuntimeError("provided search state does not match the locked run")
        state = locked_state
        state.require_search_open()
        resumable_round = self._resumable_round_index(state)
        if (self.run_dir / PROTOCOL_GATE_FILENAME).exists() and resumable_round is None:
            raise RuntimeError("search is blocked by the experiment protocol gate")
        if resumable_round is not None:
            round_index = resumable_round
            round_path = self.run_dir / "rounds" / f"round_{round_index:03d}"
            self._begin_round_call_group(round_index=round_index)
            if (round_path / "round_plan.json").is_file():
                return self._resume_pending_round(state, round_index=round_index)
            if round_index != state.current_round + 1:
                raise RuntimeError("pre-plan round index does not follow locked state")
        else:
            if state.current_round + 1 >= self.config.total_rounds:
                raise RuntimeError("all configured search rounds are complete")
            round_index = state.current_round + 1
            round_path = self.run_dir / "rounds" / f"round_{round_index:03d}"
            if round_path.exists():
                raise FileExistsError(f"round directory already exists: {round_path}")
            round_path.mkdir(parents=True)
            self._begin_round_call_group(round_index=round_index)
        round_attempt = RoundAttemptJournal(
            round_path / "round_attempt.json",
            round_index=round_index,
        )
        round_attempt.append(
            "round_created",
            fingerprint=_text_sha256(f"{state.run_id}:{round_index}"),
            metadata={"run_id": state.run_id},
        )

        structural_signatures = _structural_signatures(state, self.run_dir)
        snapshot = build_archive_snapshot(
            _primary_training_records(state.evaluations, self.config),
            view_size=self.config.archive_view_size,
            tail_fraction=self.config.tail_fraction,
            excluded_candidate_ids=_archive_exclusions(state),
            canonical_instance_ids=_canonical_instance_ids(
                state,
                self.adapter,
                self.config,
                split="train",
            ),
            instance_families=self.adapter.train_manifest.family_by_instance,
            structural_signatures=structural_signatures,
        )
        validation_snapshot = build_archive_snapshot(
            (),
            split="validation",
            view_size=self.config.archive_view_size,
            tail_fraction=self.config.tail_fraction,
            excluded_candidate_ids=_archive_exclusions(state),
            canonical_instance_ids=_canonical_instance_ids(
                state,
                self.adapter,
                self.config,
                split="validation",
            ),
            instance_families=self.adapter.validation_manifest.family_by_instance,
        )
        diagnostics = build_search_diagnostics(
            state,
            snapshot,
            run_dir=self.run_dir,
            baseline_path=self.adapter.baseline_path,
            previous_visible_candidate_ids=_previous_visible_candidate_ids(
                self.run_dir,
                state.current_round,
            ),
        )
        controller_snapshot = _controller_snapshot(snapshot, self.config)
        controller_evidence = _controller_evidence(
            state.evidence,
            controller_snapshot,
            self.config,
        )
        allowed_actions = _planner_actions(self.config)
        budget_usage = _budget_usage(
            state,
            self.adapter,
            self.config,
            run_dir=self.run_dir,
        )
        model_calls_used = _model_call_attempts(self.run_dir)
        model_tokens_used = _model_token_usage(self.run_dir)
        wall_clock_used = _wall_clock_elapsed(self.run_dir)
        _write_json(
            round_path / "archive_snapshot.json",
            {
                "train": _snapshot_record(snapshot),
                "validation": {"status": "held_out_until_final_confirmation"},
            },
        )
        _write_json(round_path / "search_diagnostics.json", diagnostics.to_dict())
        visible = _inheritance_experiment_visible_ids(
            controller_snapshot.visible_candidate_ids
            | _evidence_candidate_ids(controller_evidence),
            self.config,
        )
        candidate_scopes = {
            candidate_id: ScopeSpec.from_dict(state.require_node(candidate_id).scope)
            for candidate_id in visible
            if state.require_node(candidate_id).scope is not None
        }
        source_structures = {
            "baseline": extract_source_structure(self.adapter.baseline_path),
            **{
                candidate_id: extract_source_structure(
                    self.run_dir / state.require_node(candidate_id).solver_relpath
                )
                for candidate_id in visible
            },
        }
        scope_boundary_evidence = []
        for candidate_id, source_scope in sorted(candidate_scopes.items()):
            supporting_ids = tuple(sorted(
                record.evaluation_id
                for record in state.evaluations
                if record.candidate_id == candidate_id
                and record.split == "train"
                and record.valid
            ))
            if not supporting_ids:
                continue
            boundary = derive_scope_boundary_evidence(
                (self.run_dir / state.require_node(candidate_id).solver_relpath).parent,
                candidate_id=candidate_id,
                source_scope=source_scope,
                supporting_evaluation_ids=supporting_ids,
            )
            if boundary is not None:
                scope_boundary_evidence.append(boundary)
        scope_boundary_evidence = tuple(scope_boundary_evidence)
        _write_json(
            round_path / "scope_boundary_evidence.json",
            [item.to_dict() for item in scope_boundary_evidence],
        )
        mechanism_records = self._mechanism_transfer_records(
            state,
            controller_evidence,
        )
        _write_json(
            round_path / "mechanism_transfer_records.json",
            [record.to_dict() for record in mechanism_records],
        )
        failure_clusters = _failure_cluster_registry(
            controller_snapshot,
            validation_snapshot,
        )
        ablation_targets = {
            item.hypothesis_id: (
                item.candidate_id,
                _require_hypothesis(
                    state,
                    item.hypothesis_id,
                ).ablation_scope,
                item.target_cluster_ids,
            )
            for item in controller_evidence
            if item.status == "pending"
        }
        completed_retests = _completed_retests(
            state,
            self.adapter,
            self.config,
        )
        controller_context = build_controller_context(
            round_index=round_index,
            batch_size=self.config.batch_size,
            problem_context=self.adapter.problem_context(),
            snapshot=controller_snapshot,
            validation_snapshot=validation_snapshot,
            evidence=controller_evidence,
            scope_boundary_evidence=scope_boundary_evidence,
            mechanism_records=mechanism_records,
            remaining_candidate_budget=max(
                self.config.resolved_candidate_budget - budget_usage["candidates"],
                0,
            ),
            remaining_candidate_evaluation_units=max(
                self.config.resolved_candidate_evaluation_unit_budget
                - budget_usage["candidate_evaluation_units"],
                0,
            ),
            remaining_diagnostic_evaluation_units=max(
                self.config.resolved_diagnostic_evaluation_unit_budget
                - budget_usage["diagnostic_evaluation_units"],
                0,
            ),
            remaining_model_calls=(
                _model_call_limit(self.config) - model_calls_used
            ),
            model_calls_used=model_calls_used,
            remaining_model_tokens=(
                None
                if self.config.model_token_budget is None
                else max(self.config.model_token_budget - model_tokens_used, 0)
            ),
            model_tokens_used=model_tokens_used,
            remaining_wall_clock_sec=(
                None
                if self.config.wall_clock_budget_sec is None
                else max(self.config.wall_clock_budget_sec - wall_clock_used, 0.0)
            ),
            diagnostic_seeds=self.config.diagnostic_seeds,
            split_instance_ids={
                "train": self.adapter.train_manifest.instance_ids,
            },
            extra_observations={
                **diagnostics.observations(),
                **_baseline_observations(state),
                **_diagnostic_evaluation_observations(
                    state,
                    candidate_ids=visible,
                    adapter=self.adapter,
                    config=self.config,
                ),
            },
            candidate_scopes=candidate_scopes,
            candidate_source_structures=source_structures,
            allowed_actions=allowed_actions,
            rounds_without_global_improvement=(
                state.rounds_without_global_improvement
            ),
            ablation_targets=ablation_targets,
            completed_retests=completed_retests,
            candidate_evaluation_units_per_candidate=(
                len(self.config.base_seeds)
                * len(
                    self.adapter.train_manifest.instance_ids[
                        : self.config.n_instances
                    ]
                )
            ),
        )
        if self.config.framework_mode == "off":
            controller_context = _apply_framework_off_contract(controller_context)
        controller_context = _apply_inheritance_experiment_contract(
            controller_context,
            self.config,
        )
        controller_context = _apply_scope_experiment_contract(
            controller_context,
            self.config,
        )
        if self.config.controller_stop_policy == "fixed_rounds":
            controller_context = _apply_fixed_round_controller_contract(
                controller_context,
                batch_size=self.config.batch_size,
            )
        elif self.config.controller_stop_is_disabled:
            controller_context = _disable_controller_stop(controller_context)
        from agent_ahd.core.research_guidance import resource_facts, retrieve_training_episodes
        controller_context["resource_facts"] = resource_facts(self.adapter, self.config)
        if self.config.enable_research_guidance:
            controller_context["training_episode_memory"] = retrieve_training_episodes(
                state, self.config, self.adapter, before_round=round_index)
        controller_context_path = round_path / "controller_context.json"
        if controller_context_path.is_file():
            controller_context = json.loads(
                controller_context_path.read_text(encoding="utf-8")
            )
        else:
            _write_json(controller_context_path, controller_context)
        catalog_record = controller_context["decision_catalog"]
        _write_json(round_path / "decision_catalog.json", catalog_record)
        catalog = DecisionCatalog.from_dict(catalog_record)
        controller_prompt_context = build_controller_prompt_context(
            controller_context
        )
        _write_json(
            round_path / "controller_prompt_context.json",
            controller_prompt_context,
        )
        controller_prompt = _serialize_controller_prompt(
            controller_prompt_context
        )
        attempt_path = round_path / "controller_attempt.json"
        logical_call_id = f"round:{round_index}:controller"
        controller_identity = {
            "model": self.controller_runner.model,
            "logical_call_id": logical_call_id,
            "system_prompt_sha256": _text_sha256(CONTROLLER_SYSTEM_PROMPT),
            "user_prompt_sha256": _text_sha256(controller_prompt),
            "output_schema_sha256": _text_sha256(
                json.dumps(controller_context["output_schema"], sort_keys=True)
            ),
            "decision_catalog_sha256": _text_sha256(
                json.dumps(catalog_record, sort_keys=True)
            ),
        }
        existing_attempt = (
            json.loads(attempt_path.read_text(encoding="utf-8"))
            if attempt_path.is_file()
            else None
        )
        if existing_attempt is not None and any(
            existing_attempt.get(field) != value
            for field, value in controller_identity.items()
        ):
            raise RuntimeError("controller locked inputs changed during recovery")
        controller_attempt = {
            "status": "started",
            **controller_identity,
            "budget_usage_before": budget_usage,
            "model_calls_before": model_calls_used,
            "model_tokens_before": model_tokens_used,
            "wall_clock_before": wall_clock_used,
            "payload": None,
            "raw_content": None,
            "usage": {},
            "failure_stage": None,
            "error_type": None,
            "error_message": None,
        }
        cached_response = None
        if existing_attempt is not None:
            locked_budget_usage = existing_attempt.get("budget_usage_before")
            locked_model_calls = existing_attempt.get("model_calls_before")
            if not isinstance(locked_budget_usage, dict) or not isinstance(
                locked_model_calls,
                int,
            ):
                raise RuntimeError("controller recovery budget snapshot is incomplete")
            budget_usage = locked_budget_usage
            model_calls_used = locked_model_calls
            payload = existing_attempt.get("payload")
            raw_content = existing_attempt.get("raw_content")
            usage = existing_attempt.get("usage")
            if (
                isinstance(payload, dict)
                and isinstance(raw_content, str)
                and isinstance(usage, dict)
            ):
                cached_response = AgentResponse(payload, raw_content, usage)
            elif not getattr(
                self.controller_runner,
                "supports_durable_attempt_recovery",
                False,
            ):
                raise TransactionAmbiguityError(
                    "ambiguous controller attempt; direct backend replay is forbidden"
                )
            elif existing_attempt.get("status") == "failed" and (
                existing_attempt.get("failure_stage") != "network"
            ):
                raise RuntimeError("controller attempt previously failed")
        if cached_response is None:
            _write_json(attempt_path, controller_attempt)

        def validate_controller_response(response: AgentResponse):
            selection = ControllerSelection.from_dict(
                response.payload,
                candidate_count=self.config.batch_size,
            )
            plan = materialize_controller_selection(selection, catalog)
            if self.config.controller_stop_policy == "fixed_rounds":
                candidate_count = sum(
                    decision.action != SearchAction.RETEST
                    for decision in plan.decisions
                )
                diagnostic_count = len(plan.decisions) - candidate_count
                if (
                    plan.is_stop
                    or candidate_count != self.config.batch_size
                    or diagnostic_count != 0
                ):
                    raise ValueError(
                        "fixed-round controller policy requires exactly "
                        f"{self.config.batch_size} candidate decisions and no "
                        "diagnostic decisions"
                    )
            decision_validation = validate_round_plan(
                plan,
                round_index=round_index,
                batch_size=self.config.batch_size,
                visible_candidate_ids=visible,
                observation_ids=set(controller_context["observations"]),
                observation_records=controller_context["observations"],
                evidence_records={
                    (
                        item.evidence_id
                        if item.evidence_id.startswith("evidence:")
                        else f"evidence:{item.evidence_id}"
                    ): item
                    for item in controller_evidence
                },
                scope_boundary_evidence={
                    item.evidence_id: item for item in scope_boundary_evidence
                },
                failure_cluster_ids=set(failure_clusters),
                pending_hypothesis_ids={
                    item.hypothesis_id
                    for item in controller_evidence
                    if item.status == "pending"
                },
                existing_decision_ids={item.decision_id for item in state.decisions},
                transfer_candidate_ids={
                    item.candidate_id
                    for item in mechanism_records
                    if item.transfer_eligible
                },
                transfer_hypotheses={
                    item.hypothesis_id: item.candidate_id
                    for item in mechanism_records
                    if item.transfer_eligible
                },
                ablation_targets=ablation_targets,
                candidate_scopes=candidate_scopes,
                candidate_source_structures=source_structures,
                allowed_retest_seeds=set(self.config.diagnostic_seeds),
                allowed_retest_instances={
                    "train": set(self.adapter.train_manifest.instance_ids),
                },
                completed_retests=completed_retests,
                allowed_actions={
                    SearchAction(action) for action in controller_context["actions"]
                },
            )
            planned_budget = _planned_budget(plan, self.adapter, self.config)
            checks = (
                (
                    "candidate",
                    budget_usage["candidates"],
                    planned_budget["candidates"],
                    self.config.resolved_candidate_budget,
                ),
                (
                    "candidate evaluation",
                    budget_usage["candidate_evaluation_units"],
                    planned_budget["candidate_evaluation_units"],
                    self.config.resolved_candidate_evaluation_unit_budget,
                ),
                (
                    "diagnostic evaluation",
                    budget_usage["diagnostic_evaluation_units"],
                    planned_budget["diagnostic_evaluation_units"],
                    self.config.resolved_diagnostic_evaluation_unit_budget,
                ),
            )
            for label, used, planned, limit in checks:
                if used + planned > limit:
                    raise ValueError(f"round plan exceeds the {label} budget")
            if (
                model_calls_used
                + _reserved_model_calls(self.config, planned_budget["model_calls"])
                > _model_call_limit(self.config)
            ):
                raise ValueError("round plan exceeds the model-call budget")
            decision_validation.update({
                "planned_budget": planned_budget,
                "budget_usage_before": budget_usage,
                "model_calls_before": model_calls_used,
            })
            return selection, plan, decision_validation, planned_budget

        failure_stage = "request"
        try:
            controller_response = cached_response
            if controller_response is None:
                current_model_calls = _model_call_attempts(self.run_dir)
                current_model_tokens = _model_token_usage(self.run_dir)
                current_wall_clock = _wall_clock_elapsed(self.run_dir)
                if current_model_calls >= _model_call_limit(self.config):
                    raise RuntimeError("model-call budget is exhausted")
                if (
                    self.config.model_token_budget is not None
                    and current_model_tokens >= self.config.model_token_budget
                ):
                    raise RuntimeError("model-token budget is exhausted")
                if (
                    self.config.wall_clock_budget_sec is not None
                    and current_wall_clock >= self.config.wall_clock_budget_sec
                ):
                    raise RuntimeError("wall-clock budget is exhausted")
                call_kwargs = {}
                if getattr(self.controller_runner, "protocol_runner", False):
                    call_kwargs["response_validator"] = (
                        lambda response: validate_controller_response(response)
                    )
                controller_response = self.controller_runner.complete_json(
                    system_prompt=CONTROLLER_SYSTEM_PROMPT,
                    user_prompt=controller_prompt,
                    output_schema=controller_context["output_schema"],
                    logical_call_id=logical_call_id,
                    **call_kwargs,
                )
            controller_attempt.update({
                "status": "returned",
                "payload": controller_response.payload,
                "raw_content": controller_response.raw_content,
                "usage": controller_response.usage,
            })
            _write_json(attempt_path, controller_attempt)
            round_attempt.append(
                "controller_returned",
                fingerprint=_text_sha256(json.dumps({
                    "payload": controller_response.payload,
                    "raw_content": controller_response.raw_content,
                    "usage": controller_response.usage,
                }, sort_keys=True)),
                metadata={"logical_call_id": logical_call_id},
            )
            failure_stage = "parse"
            (
                selection,
                plan,
                decision_validation,
                planned_budget,
            ) = validate_controller_response(controller_response)
            _write_json(
                round_path / "controller_selection.json",
                asdict(selection),
            )
            failure_stage = "materialization"
            _write_json(
                round_path / "materialized_plan.json",
                plan.to_dict(),
            )
            failure_stage = "validation"
        except Exception as exc:
            if isinstance(exc, AgentResponseError):
                failure_stage = exc.stage
                controller_attempt.update({
                    "raw_content": exc.raw_content,
                    "usage": exc.usage,
                })
            controller_attempt.update({
                "status": "failed",
                "failure_stage": failure_stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            _write_json(attempt_path, controller_attempt)
            raise
        controller_attempt["status"] = "validated"
        _write_json(attempt_path, controller_attempt)
        _write_json(round_path / "decision_validation.json", decision_validation)
        _write_json(round_path / "round_plan.json", plan.to_dict())
        round_attempt.append(
            "plan_validated",
            fingerprint=_file_sha256(round_path / "round_plan.json"),
            metadata={
                "decision_validation_sha256": _file_sha256(
                    round_path / "decision_validation.json"
                )
            },
        )

        if plan.is_stop:
            if self.config.controller_stop_is_disabled:
                raise RuntimeError("STOP is disabled by the controller stop policy")
            if not _model_token_usage_complete(self.run_dir):
                raise RuntimeError("STOP requires complete model token usage")
            stop_resource_gate = _resource_budget_gate(
                self.run_dir,
                self.config,
            )
            if stop_resource_gate is not None:
                _write_idempotent_json(
                    self.run_dir / PROTOCOL_GATE_FILENAME,
                    {**stop_resource_gate, "round_index": round_index},
                    label="protocol gate",
                )
                raise RuntimeError(stop_resource_gate["reason"])
            state.stop_search(
                round_index=round_index,
                reason=plan.stop_reason or "controller requested STOP",
            )
            _write_json(
                round_path / "stop.json",
                {
                    "round_index": round_index,
                    "reason": state.stopped_reason,
                    "planned_budget": planned_budget,
                },
            )
            save_search_state(self.run_dir / STATE_FILENAME, state)
            self._end_round_call_group(round_index=round_index)
            return False

        materialized = materialize_round_plan(state, plan)
        save_search_state(self.run_dir / STATE_FILENAME, state)
        round_attempt.append(
            "candidates_materialized",
            fingerprint=_text_sha256(json.dumps({
                node.node_id: node.decision_id
                for node in materialized.candidate_nodes
            }, sort_keys=True)),
            metadata={"candidate_ids": [
                node.node_id for node in materialized.candidate_nodes
            ]},
        )
        generation_jobs = self._prepare_generation_jobs(
            state,
            snapshot,
            validation_snapshot,
            plan,
            materialized.candidate_nodes,
            round_path,
            observations=controller_context["observations"],
            mechanism_records=mechanism_records,
        )
        hypothesis_errors = self._run_hypothesis_agents(generation_jobs)
        locked_plan = self._lock_code_hypotheses(
            state,
            plan,
            generation_jobs,
            hypothesis_errors,
        )
        _write_json(round_path / "locked_experiments.json", locked_plan.to_dict())
        save_search_state(self.run_dir / STATE_FILENAME, state)
        round_attempt.append(
            "hypotheses_locked",
            fingerprint=_file_sha256(round_path / "locked_experiments.json"),
            metadata={},
        )
        generation_errors = self._run_implementation_agents(
            generation_jobs,
            hypothesis_errors,
        )
        return self._close_round(
            state,
            round_index=round_index,
            round_path=round_path,
            plan=locked_plan,
            jobs=generation_jobs,
            nodes=materialized.candidate_nodes,
            retests=materialized.retest_decisions,
            generation_errors=generation_errors,
        )

    def _resume_pending_round(self, state: SearchState, *, round_index: int) -> bool:
        round_path = self.run_dir / "rounds" / f"round_{round_index:03d}"
        plan_path = round_path / "round_plan.json"
        if not plan_path.is_file():
            raise RuntimeError("pending round has no locked round plan")
        decision_validation_path = round_path / "decision_validation.json"
        if not decision_validation_path.is_file():
            raise RuntimeError("pending round has no locked budget validation")
        original_plan = RoundPlan.from_dict(
            json.loads(plan_path.read_text(encoding="utf-8"))
        )
        round_attempt = RoundAttemptJournal(
            round_path / "round_attempt.json",
            round_index=round_index,
        )
        if "round_created" not in round_attempt.stages():
            round_attempt.append(
                "round_created",
                fingerprint=_text_sha256(f"{state.run_id}:{round_index}"),
                metadata={"run_id": state.run_id},
            )
        attempt_path = round_path / "controller_attempt.json"
        if "controller_returned" not in round_attempt.stages():
            if not attempt_path.is_file():
                raise RuntimeError("pending round has no controller response")
            controller_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            if not isinstance(controller_attempt.get("payload"), dict):
                raise RuntimeError("pending round controller response is incomplete")
            round_attempt.append(
                "controller_returned",
                fingerprint=_text_sha256(json.dumps({
                    "payload": controller_attempt["payload"],
                    "raw_content": controller_attempt.get("raw_content"),
                    "usage": controller_attempt.get("usage") or {},
                }, sort_keys=True)),
                metadata={
                    "logical_call_id": controller_attempt.get(
                        "logical_call_id",
                        f"round:{round_index}:controller",
                    )
                },
            )
        if "plan_validated" not in round_attempt.stages():
            round_attempt.append(
                "plan_validated",
                fingerprint=_file_sha256(plan_path),
                metadata={
                    "decision_validation_sha256": _file_sha256(
                        decision_validation_path
                    )
                },
            )

        expected_decision_ids = {
            decision.decision_id for decision in original_plan.decisions
        }
        existing_round_decisions = {
            decision.decision_id
            for decision in state.decisions
            if decision.decision_id in expected_decision_ids
        }
        existing_nodes = [
            node for node in state.nodes if node.round_index == round_index
        ]
        if not existing_round_decisions and not existing_nodes:
            materialized = materialize_round_plan(state, original_plan)
            save_search_state(self.run_dir / STATE_FILENAME, state)
            nodes = materialized.candidate_nodes
        else:
            if existing_round_decisions != expected_decision_ids:
                raise RuntimeError("pending round was only partially materialized")
            expected_node_decisions = {
                decision.decision_id
                for decision in original_plan.decisions
                if decision.action != SearchAction.RETEST
            }
            if {node.decision_id for node in existing_nodes} != expected_node_decisions:
                raise RuntimeError("pending round candidate materialization changed")
            nodes = tuple(existing_nodes)
        round_attempt.append(
            "candidates_materialized",
            fingerprint=_text_sha256(json.dumps({
                node.node_id: node.decision_id for node in nodes
            }, sort_keys=True)),
            metadata={"candidate_ids": [node.node_id for node in nodes]},
        )

        locked_path = round_path / "locked_experiments.json"
        snapshot, validation_snapshot = self._recovery_snapshots(state)
        controller_context = json.loads(
            (round_path / "controller_context.json").read_text(encoding="utf-8")
        )
        jobs = self._prepare_generation_jobs(
            state,
            snapshot,
            validation_snapshot,
            original_plan,
            nodes,
            round_path,
            observations=controller_context["observations"],
        )

        if locked_path.is_file():
            plan = RoundPlan.from_dict(
                json.loads(locked_path.read_text(encoding="utf-8"))
            )
            if plan.round_index != original_plan.round_index:
                raise RuntimeError("locked experiment round changed during resume")
            _hydrate_locked_plan(state, plan)
            for node_id, job in jobs.items():
                node = state.require_node(node_id)
                if node.hypothesis_id is not None:
                    job["hypothesis"] = _require_hypothesis(
                        state,
                        node.hypothesis_id,
                    )
            hypothesis_errors = {
                node_id: job["stage_journal"].record("generation")["metadata"].get(
                    "error",
                    "generation was previously rejected",
                )
                for node_id, job in jobs.items()
                if job["stage_journal"].record("generation")["status"] == "rejected"
            }
        else:
            hypothesis_errors = self._run_hypothesis_agents(jobs)
            plan = self._lock_code_hypotheses(
                state,
                original_plan,
                jobs,
                hypothesis_errors,
            )
            _write_json(locked_path, plan.to_dict())
            save_search_state(self.run_dir / STATE_FILENAME, state)
        round_attempt.append(
            "hypotheses_locked",
            fingerprint=_file_sha256(locked_path),
            metadata={},
        )
        retests = tuple(
            decision
            for decision in plan.decisions
            if decision.action == SearchAction.RETEST
        )
        generation_errors = self._run_implementation_agents(
            jobs,
            hypothesis_errors,
        )
        return self._close_round(
            state,
            round_index=round_index,
            round_path=round_path,
            plan=plan,
            jobs=jobs,
            nodes=nodes,
            retests=retests,
            generation_errors=generation_errors,
        )

    def _close_round(
        self,
        state: SearchState,
        *,
        round_index: int,
        round_path: Path,
        plan: RoundPlan,
        jobs: dict[str, dict],
        nodes: tuple[SolverNode, ...],
        retests: tuple[PlannerDecision, ...],
        generation_errors: dict[str, str],
    ) -> bool:
        journal = RoundClosureJournal(
            round_path / "round_closure_journal.json",
            round_index=round_index,
        )
        usage_input = _model_usage_fingerprint(self.run_dir)
        usage_record = journal.record("usage_completeness")
        if usage_record is None:
            journal.begin("usage_completeness", input_fingerprint=usage_input)
        elif usage_record["input_fingerprint"] != usage_input:
            raise RuntimeError("usage changed after round closure started")
        if not _model_token_usage_complete(self.run_dir):
            raise RuntimeError("round closure requires complete model token usage")
        usage_metadata = {
            "model_calls": _model_call_attempts(self.run_dir),
            "model_tokens": _model_token_usage(self.run_dir),
        }
        journal.finish(
            "usage_completeness",
            input_fingerprint=usage_input,
            output_fingerprint=_text_sha256(json.dumps(usage_metadata, sort_keys=True)),
            metadata=usage_metadata,
        )

        validation_path = round_path / "decision_validation.json"
        if not validation_path.is_file():
            raise RuntimeError("round closure requires locked budget validation")
        decision_validation = json.loads(validation_path.read_text(encoding="utf-8"))
        planned_budget = decision_validation.get("planned_budget")
        usage_before = decision_validation.get("budget_usage_before")
        model_calls_before = decision_validation.get("model_calls_before")
        if (
            not isinstance(planned_budget, dict)
            or not isinstance(usage_before, dict)
            or not isinstance(model_calls_before, int)
        ):
            raise RuntimeError("round closure requires complete locked budget validation")
        if planned_budget != _planned_budget(plan, self.adapter, self.config):
            raise RuntimeError("locked planned budget does not match the round plan")
        locked_limits = {
            "candidates": self.config.resolved_candidate_budget,
            "candidate_evaluation_units": (
                self.config.resolved_candidate_evaluation_unit_budget
            ),
            "diagnostic_evaluation_units": (
                self.config.resolved_diagnostic_evaluation_unit_budget
            ),
        }
        for name, limit in locked_limits.items():
            before = usage_before.get(name)
            planned = planned_budget.get(name)
            if not isinstance(before, int) or not isinstance(planned, int):
                raise RuntimeError(f"locked {name} budget is incomplete")
            if before + planned > limit:
                raise RuntimeError(f"locked round plan exceeds the {name} budget")
        if (
            model_calls_before
            + _reserved_model_calls(
                self.config,
                int(planned_budget.get("model_calls", -1)),
            )
            > _model_call_limit(self.config)
        ):
            raise RuntimeError("locked round plan exceeds the model-call budget")
        budget_input = _text_sha256(
            json.dumps(
                {
                    "plan": plan.to_dict(),
                    "planned_budget": planned_budget,
                    "usage_before": usage_before,
                    "model_calls_before": model_calls_before,
                },
                sort_keys=True,
            )
        )
        journal.begin("planned_budget", input_fingerprint=budget_input)
        journal.finish(
            "planned_budget",
            input_fingerprint=budget_input,
            output_fingerprint=_text_sha256(
                json.dumps(decision_validation, sort_keys=True)
            ),
            metadata={
                "planned_budget": planned_budget,
                "budget_usage_before": usage_before,
                "model_calls_before": model_calls_before,
            },
        )

        for node_id, job in sorted(jobs.items()):
            generation = job["stage_journal"].record("generation")
            if (
                generation is not None
                and generation.get("status") == "complete"
            ):
                verified = _verified_candidate_commit(job["control_workspace"])
                if verified is None:
                    raise RuntimeError(
                        f"candidate {node_id} has no committed source version"
                    )
                commit, workspace = verified
                job["workspace"] = workspace
                job["node"].solver_relpath = workspace.solver_path.relative_to(
                    self.run_dir
                ).as_posix()
                if generation.get("output_fingerprint") != commit.get(
                    "source_fingerprint"
                ):
                    raise RuntimeError(
                        f"candidate {node_id} generation fingerprint changed"
                    )

        evaluation_path = round_path / "evaluations.json"
        evaluation_input = _text_sha256(
            json.dumps(
                {
                    "plan": plan.to_dict(),
                    "generation": {
                        node_id: job["stage_journal"].record("generation")
                        for node_id, job in sorted(jobs.items())
                    },
                },
                sort_keys=True,
            )
        )
        evaluation_gate = journal.record("evaluation")
        if evaluation_gate is None:
            if evaluation_path.exists():
                raise RuntimeError("evaluation output exists without a gate record")
            journal.begin("evaluation", input_fingerprint=evaluation_input)
            records = self._evaluate_round(
                state,
                plan,
                nodes,
                retests,
                jobs,
                generation_errors,
                round_path,
            )
        else:
            if evaluation_gate.get("input_fingerprint") != evaluation_input:
                raise RuntimeError("evaluation gate input changed during resume")
            if evaluation_path.is_file():
                records = _load_round_evaluation_records(evaluation_path)
                _hydrate_round_evaluation_records(state, records)
            elif evaluation_gate.get("status") == "running":
                records = self._evaluate_round(
                    state,
                    plan,
                    nodes,
                    retests,
                    jobs,
                    generation_errors,
                    round_path,
                )
            else:
                raise RuntimeError("evaluation gate has no stage output")
        if not evaluation_path.is_file():
            raise RuntimeError("evaluation stage did not write its output")
        evaluation_output = _file_sha256(evaluation_path)
        if (
            evaluation_gate is not None
            and evaluation_gate.get("status") == "complete"
            and evaluation_gate.get("output_fingerprint") != evaluation_output
        ):
            raise RuntimeError("evaluation stage output changed during resume")
        usage = _budget_usage(
            state,
            self.adapter,
            self.config,
            run_dir=self.run_dir,
        )
        reconciliation = _reconcile_round_budget(
            usage_before=usage_before,
            planned_budget=planned_budget,
            usage_after=usage,
        )
        save_search_state(self.run_dir / STATE_FILENAME, state)
        journal.finish(
            "evaluation",
            input_fingerprint=evaluation_input,
            output_fingerprint=evaluation_output,
            metadata={
                "evaluation_ids": [
                    record.evaluation_id for record in records.values()
                ],
                "budget_reconciliation": reconciliation,
            },
        )

        evidence_path = round_path / "evidence_updates.json"
        evidence_input = _text_sha256(
            json.dumps(
                {key: record.to_dict() for key, record in sorted(records.items())},
                sort_keys=True,
            )
        )
        evidence_gate = journal.record("evidence")
        if evidence_gate is None:
            if evidence_path.exists():
                raise RuntimeError("evidence output exists without a gate record")
            journal.begin("evidence", input_fingerprint=evidence_input)
            evidence_updates = self._update_evidence(state, plan, records)
            _finish_evidence_validation_stages(jobs, evidence_updates)
            _write_json(evidence_path, [asdict(item) for item in evidence_updates])
        else:
            if evidence_gate.get("input_fingerprint") != evidence_input:
                raise RuntimeError("evidence gate input changed during resume")
            if evidence_path.is_file():
                evidence_updates = [
                    EvidenceRecord.from_dict(item)
                    for item in json.loads(evidence_path.read_text(encoding="utf-8"))
                ]
                _hydrate_evidence_updates(state, evidence_updates)
                _finish_evidence_validation_stages(jobs, evidence_updates)
            else:
                raise RuntimeError("evidence gate has no stage output")
        evidence_output = _file_sha256(evidence_path)
        if (
            evidence_gate is not None
            and evidence_gate.get("status") == "complete"
            and evidence_gate.get("output_fingerprint") != evidence_output
        ):
            raise RuntimeError("evidence stage output changed during resume")
        save_search_state(self.run_dir / STATE_FILENAME, state)
        journal.finish(
            "evidence",
            input_fingerprint=evidence_input,
            output_fingerprint=evidence_output,
            metadata={
                "evidence_ids": [item.evidence_id for item in evidence_updates],
            },
        )

        quality_gate = _candidate_quality_gate(state, self.run_dir, self.config)
        resource_gate = _resource_budget_gate(self.run_dir, self.config)
        if resource_gate is not None:
            resource_gate = {
                **resource_gate,
                "category": "resource",
                "reason_kind": resource_gate["reason"],
            }
        gate_input = _text_sha256(
            json.dumps(
                {
                    "usage": usage,
                    "evidence_ids": [item.evidence_id for item in evidence_updates],
                    "gate_mode": self.config.gate_mode,
                    "quality_thresholds": {
                        "valid": self.config.min_valid_candidates,
                        "exact_unique": self.config.min_exact_unique_candidates,
                        "structural_unique": self.config.min_structural_unique_candidates,
                    },
                    "proposal_slots": [
                        {
                            "node_id": node.node_id,
                            "valid": node.valid,
                            "solver_sha256": node.solver_sha256,
                        }
                        for node in state.nodes
                        if node.round_index >= 0
                    ],
                },
                sort_keys=True,
            )
        )
        gate_record = journal.record("invalidity_resource_gate")
        protocol_gate_path = self.run_dir / PROTOCOL_GATE_FILENAME
        if gate_record is None and protocol_gate_path.is_file():
            raise RuntimeError("protocol gate output has no closure gate record")
        journal.begin("invalidity_resource_gate", input_fingerprint=gate_input)
        if gate_record is not None and gate_record.get("status") == "complete":
            gate = gate_record["metadata"].get("gate")
            if gate is not None and not protocol_gate_path.is_file():
                raise RuntimeError("completed resource gate has no gate output")
            if gate is None and protocol_gate_path.is_file():
                raise RuntimeError("unexpected protocol gate after passed closure gate")
        elif protocol_gate_path.is_file():
            gate_output = json.loads(protocol_gate_path.read_text(encoding="utf-8"))
            if gate_output.get("round_index") != round_index:
                raise RuntimeError("protocol gate round changed during resume")
            gate = {
                key: value
                for key, value in gate_output.items()
                if key != "round_index"
            }
        else:
            gate = resource_gate if resource_gate is not None else quality_gate
        ledger_limits = {
            "candidates": self.config.resolved_candidate_budget,
            "candidate_evaluation_units": (
                self.config.resolved_candidate_evaluation_unit_budget
            ),
            "diagnostic_evaluation_units": (
                self.config.resolved_diagnostic_evaluation_unit_budget
            ),
        }
        ledger_overages = {
            **reconciliation["overages"],
            **{
                name: {"used": usage[name], "limit": limit}
                for name, limit in ledger_limits.items()
                if usage[name] > limit
            },
        }
        if ledger_overages:
            raise RuntimeError(
                "evaluation budget ledger overage: "
                + json.dumps(ledger_overages, sort_keys=True)
            )
        if gate is not None:
            _write_idempotent_json(
                protocol_gate_path,
                {**gate, "round_index": round_index},
                label="protocol gate",
            )
        gate_metadata = {"gate": gate, "usage": usage}
        journal.finish(
            "invalidity_resource_gate",
            input_fingerprint=gate_input,
            output_fingerprint=_text_sha256(json.dumps(gate_metadata, sort_keys=True)),
            metadata=gate_metadata,
        )

        completion_input = _text_sha256(
            json.dumps(
                [
                    record
                    for record in journal.records()
                    if record["name"] != "round_complete"
                ],
                sort_keys=True,
            )
        )
        journal.begin("round_complete", input_fingerprint=completion_input)
        if round_index not in state.completed_rounds:
            self._finalize_round(state, round_index)
            save_search_state(self.run_dir / STATE_FILENAME, state)
        journal.finish(
            "round_complete",
            input_fingerprint=completion_input,
            output_fingerprint=_file_sha256(self.run_dir / STATE_FILENAME),
            metadata={"completed_round": round_index},
        )
        _write_json(
            round_path / "round_closure.json",
            {
                "round_index": round_index,
                "ordered_gates": [
                    {
                        **record,
                        "status": (
                            "terminated"
                            if record["name"] == "invalidity_resource_gate"
                            and record["metadata"].get("gate") is not None
                            else "passed"
                        ),
                    }
                    for record in journal.records()
                ],
            },
        )
        RoundAttemptJournal(
            round_path / "round_attempt.json",
            round_index=round_index,
        ).append(
            "round_closed",
            fingerprint=_file_sha256(round_path / "round_closure.json"),
            metadata={"completed_round": round_index},
        )
        self._end_round_call_group(round_index=round_index)
        return True

    def _resumable_round_index(self, state: SearchState) -> int | None:
        if state.pending_node_ids:
            indices = {
                state.require_node(node_id).round_index
                for node_id in state.pending_node_ids
            }
            if len(indices) != 1:
                raise RuntimeError("pending candidates span multiple rounds")
            return next(iter(indices))
        candidates = []
        for path in sorted((self.run_dir / "rounds").glob("round_*")):
            try:
                round_index = int(path.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if (
                round_index in state.completed_rounds
                and (path / "round_closure.json").is_file()
            ):
                attempt_path = path / "round_attempt.json"
                if not attempt_path.is_file():
                    continue
                attempt = RoundAttemptJournal(
                    attempt_path,
                    round_index=round_index,
                )
                if "round_closed" in attempt.stages():
                    continue
            candidates.append(round_index)
        if len(candidates) > 1:
            raise RuntimeError("multiple incomplete round closures exist")
        return candidates[0] if candidates else None

    def _recovery_snapshots(
        self,
        state: SearchState,
    ) -> tuple[ArchiveSnapshot, ArchiveSnapshot]:
        snapshot = build_archive_snapshot(
            _primary_training_records(state.evaluations, self.config),
            view_size=self.config.archive_view_size,
            tail_fraction=self.config.tail_fraction,
            excluded_candidate_ids=_archive_exclusions(state),
            canonical_instance_ids=_canonical_instance_ids(
                state,
                self.adapter,
                self.config,
                split="train",
            ),
            instance_families=self.adapter.train_manifest.family_by_instance,
            structural_signatures=_structural_signatures(state, self.run_dir),
        )
        validation = build_archive_snapshot(
            (),
            split="validation",
            view_size=self.config.archive_view_size,
            tail_fraction=self.config.tail_fraction,
            excluded_candidate_ids=_archive_exclusions(state),
            canonical_instance_ids=_canonical_instance_ids(
                state,
                self.adapter,
                self.config,
                split="validation",
            ),
            instance_families=self.adapter.validation_manifest.family_by_instance,
        )
        return snapshot, validation

    def _require_search_open(self) -> None:
        if (self.run_dir / FINAL_CONFIRMATION_FILENAME).exists():
            raise RuntimeError(
                "search is frozen because final confirmation has started"
            )
        if (self.run_dir / BLIND_EVALUATION_LOCK_FILENAME).exists():
            raise RuntimeError(
                "search is frozen because blind evaluation has started"
            )

    def _ensure_baseline_control(self, state: SearchState) -> None:
        with self._transaction_scope():
            self._ensure_baseline_control_locked(state)

    def _ensure_baseline_control_locked(self, state: SearchState) -> None:
        self._ensure_baseline_evaluation(
            state,
            split="train",
            seeds=self.config.base_seeds,
            instance_ids=self.adapter.train_manifest.instance_ids[
                : self.config.n_instances
            ],
            evaluation_id="eval_baseline_control_000",
            output_name="evaluation.json",
        )
        if self.config.enable_baseline_parent:
            from agent_ahd.core.baseline_parent import ensure_baseline_parent
            changed = ensure_baseline_parent(state, self.run_dir, seeds=self.config.base_seeds,
                instance_ids=self.adapter.train_manifest.instance_ids[:self.config.n_instances])
            if changed:
                save_search_state(self.run_dir / STATE_FILENAME, state)

    def _ensure_baseline_evaluation(
        self,
        state: SearchState,
        *,
        split: str,
        seeds: tuple[int, ...],
        instance_ids: tuple[str, ...],
        evaluation_id: str,
        output_name: str,
    ) -> None:
        matches = [
            record
            for record in state.evaluations
            if record.candidate_id == BASELINE_CONTROL_ID
            and record.split == split
            and record.seeds == seeds
            and record.instance_ids == instance_ids
        ]
        if matches:
            if len(matches) != 1 or not matches[0].valid:
                raise RuntimeError(f"baseline {split} control is ambiguous or invalid")
            return
        candidate_path = self.run_dir / "controls" / "baseline" / "solver.py"
        if _file_sha256(candidate_path) != state.baseline_sha256:
            raise RuntimeError("baseline control source changed")
        task = self.adapter.make_task(
            candidate_id=BASELINE_CONTROL_ID,
            candidate_path=candidate_path,
            budget=EvaluationBudget(
                n_instances=len(instance_ids),
                time_limit=self.config.time_limit,
                ite_max=self.config.ite_max,
                seed=seeds[0],
                seeds=seeds,
            ),
            split=split,
            instance_ids=instance_ids,
        )
        control_root = self.run_dir / "controls" / "baseline"
        attempt_path = control_root / f"{split}_evaluation_attempt.json"
        receipt_path = control_root / f"{split}_evaluation_receipt.json"
        task_fingerprint = _evaluation_task_batch_fingerprint([task])
        attempt = (
            json.loads(attempt_path.read_text(encoding="utf-8"))
            if attempt_path.is_file()
            else None
        )
        if attempt is not None and (
            set(attempt) != {
                "candidate_ids",
                "receipt_sha256",
                "status",
                "task_fingerprint",
                "version",
            }
            or attempt.get("version") != 1
            or attempt.get("task_fingerprint") != task_fingerprint
            or attempt.get("candidate_ids") != [BASELINE_CONTROL_ID]
            or attempt.get("status") not in {"dispatched", "returned"}
        ):
            raise RuntimeError(f"baseline {split} evaluation attempt changed")
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("version") != 1
                or receipt.get("task_fingerprint") != task_fingerprint
                or not isinstance(receipt.get("results"), list)
            ):
                raise RuntimeError(f"baseline {split} evaluation receipt changed")
            if attempt is None:
                raise RuntimeError(f"baseline {split} receipt has no dispatch attempt")
            receipt_sha256 = _file_sha256(receipt_path)
            if attempt.get("receipt_sha256") not in {None, receipt_sha256}:
                raise RuntimeError(f"baseline {split} evaluation receipt changed")
            results = [EvaluationResult(**item) for item in receipt["results"]]
            if attempt.get("status") != "returned":
                _write_json(attempt_path, {
                    "version": 1,
                    "status": "returned",
                    "task_fingerprint": task_fingerprint,
                    "candidate_ids": [BASELINE_CONTROL_ID],
                    "receipt_sha256": receipt_sha256,
                })
        else:
            if attempt is not None:
                raise TransactionAmbiguityError(
                    f"ambiguous baseline {split} evaluation attempt; "
                    "automatic redispatch is forbidden"
                )
            _write_json(attempt_path, {
                "version": 1,
                "status": "dispatched",
                "task_fingerprint": task_fingerprint,
                "candidate_ids": [BASELINE_CONTROL_ID],
                "receipt_sha256": None,
            })
            results = self.evaluation_function(
                [task],
                max_workers=self.config.evaluation_workers,
            )
            _write_evaluation_receipt(
                receipt_path,
                task_fingerprint=task_fingerprint,
                results=results,
            )
            _write_json(attempt_path, {
                "version": 1,
                "status": "returned",
                "task_fingerprint": task_fingerprint,
                "candidate_ids": [BASELINE_CONTROL_ID],
                "receipt_sha256": _file_sha256(receipt_path),
            })
        if len(results) != 1:
            raise RuntimeError(f"baseline {split} evaluator returned the wrong result count")
        record = _result_record(
            results[0],
            candidate_id=BASELINE_CONTROL_ID,
            evaluation_id=evaluation_id,
            split=split,
            requested_seeds=task.budget.resolved_seeds,
            requested_instance_ids=task.instance_ids,
            expected_metric_name=task.metric_name,
            expected_metric_unit=task.metric_unit,
        )
        output_path = control_root / output_name
        _write_idempotent_json(
            output_path,
            record.to_dict(),
            label=f"baseline {split} evaluation",
        )
        if not record.valid:
            raise RuntimeError(
                f"baseline {split} control failed: {record.error_type}: "
                f"{record.error_message}"
            )
        state.evaluations.append(record)
        save_search_state(self.run_dir / STATE_FILENAME, state)

    def _prepare_generation_jobs(
        self,
        state: SearchState,
        snapshot: ArchiveSnapshot,
        validation_snapshot: ArchiveSnapshot,
        plan: RoundPlan,
        nodes: tuple[SolverNode, ...],
        round_path: Path,
        observations: dict[str, object],
        mechanism_records: tuple[MechanismTransferRecord, ...] | None = None,
    ) -> dict[str, dict]:
        mechanism_records = (
            mechanism_records
            if mechanism_records is not None
            else self._mechanism_transfer_records(state, state.evidence)
        )
        decision_by_id = {item.decision_id: item for item in plan.decisions}
        failure_clusters = _failure_cluster_registry(snapshot, validation_snapshot)
        jobs = {}
        for node in nodes:
            decision = decision_by_id[node.decision_id]
            nodes_root = (self.run_dir / "nodes").resolve()
            control_root = (nodes_root / node.node_id).resolve()
            if control_root.parent != nodes_root:
                raise RuntimeError(f"candidate control path escapes run: {node.node_id}")
            control_workspace = CandidateWorkspace(
                candidate_id=node.node_id,
                root=control_root,
                solver_path=control_root / "solver.py",
            )
            snapshot_root = round_path / "source_snapshots" / node.node_id
            task_path = round_path / "agent_tasks" / f"{node.node_id}.md"
            if control_root.exists():
                if not snapshot_root.exists() or not task_path.is_file():
                    raise RuntimeError(
                        f"incomplete candidate preparation cannot be recovered: {node.node_id}"
                    )
                for staging in control_root.glob(".candidate-stage-*"):
                    _discard_generated_tree(
                        staging,
                        parent=control_root,
                        name_prefix=".candidate-stage-",
                    )
                task = task_path.read_text(encoding="utf-8")
                journal = CandidateStageJournal(
                    control_root / "stage_journal.json",
                    candidate_id=node.node_id,
                )
                generation_input = _text_sha256(task)
                generation_record = journal.record("generation")
                if generation_record is None:
                    journal.begin(
                        "generation",
                        input_fingerprint=generation_input,
                    )
                    generation_record = journal.record("generation")
                verified = _verified_candidate_commit(control_workspace)
                if verified is None:
                    workspace = control_workspace
                else:
                    _, workspace = verified
                    node.solver_relpath = workspace.solver_path.relative_to(
                        self.run_dir
                    ).as_posix()
                source_entry = snapshot_root / "solver.py"
                job = {
                    "node": node,
                    "decision": decision,
                    "control_workspace": control_workspace,
                    "workspace": workspace,
                    "snapshot_root": snapshot_root,
                    "task": task,
                    "source_symbols": tuple(
                        extract_source_structure(source_entry)["symbols"]
                    ),
                    "stage_journal": journal,
                    "generation_input_fingerprint": generation_input,
                }
                if node.hypothesis_id is not None:
                    job["hypothesis"] = _require_hypothesis(
                        state,
                        node.hypothesis_id,
                    )
                jobs[node.node_id] = job
                continue

            _discard_partial_candidate_preparation(
                nodes_root=nodes_root,
                node_id=node.node_id,
                snapshot_root=snapshot_root,
                task_path=task_path,
            )
            parent = state.require_node(node.primary_parent_id) if node.primary_parent_id else None
            reference = (
                state.require_node(node.reference_node_id) if node.reference_node_id else None
            )
            parent_source = (
                self.run_dir / parent.solver_relpath if parent else None
            )
            primary_source, source_origin_id = _primary_inheritance_source(
                decision,
                parent_source=parent_source,
                parent_id=parent.node_id if parent else None,
                baseline_path=self.adapter.baseline_path,
            )
            preparation_id = uuid.uuid4().hex
            temporary_control = nodes_root / (
                f".prepare-{node.node_id}-{preparation_id}"
            )
            temporary_snapshot = snapshot_root.parent / (
                f".prepare-{node.node_id}-{preparation_id}"
            )
            temporary_task = task_path.parent / (
                f".{node.node_id}.{preparation_id}.tmp"
            )
            temporary_node = replace(
                node,
                solver_relpath=(temporary_control / "solver.py")
                .relative_to(self.run_dir)
                .as_posix(),
            )
            try:
                workspace = prepare_search_candidate_workspace(
                    run_dir=self.run_dir,
                    node=temporary_node,
                    baseline_path=self.adapter.baseline_path,
                    primary_solver_path=primary_source if parent else None,
                    primary_result=_node_result(parent) if parent else None,
                    reference_solver_path=(
                        self.run_dir / reference.solver_relpath
                    ) if reference else None,
                    reference_result=_node_result(reference) if reference else None,
                    write_reference_diff=False,
                    inheritance_channel=decision.inheritance_channel,
                    source_origin_id=source_origin_id,
                )
                _copy_python_sources(workspace.root, temporary_snapshot)
                auxiliary = self._auxiliary_mechanism(
                    state,
                    decision,
                    mechanism_records=mechanism_records,
                )
                if auxiliary is not None:
                    _write_json(workspace.root / "auxiliary_mechanism.json", auxiliary)
                    if decision.action == SearchAction.CROSS:
                        _write_json(
                            workspace.root / "mechanism_transfer_record.json",
                            auxiliary["mechanism_transfer_record"],
                        )
                _write_json(workspace.root / "planner_decision.json", asdict(decision))
                target_context = {}
                for cluster_id in decision.target_cluster_ids:
                    current = failure_clusters.get(cluster_id)
                    if current is not None:
                        split, current_snapshot, current_cluster = current
                        target_context[cluster_id] = {
                            **asdict(current_cluster),
                            "cluster_id": cluster_id,
                            "split": split,
                            **_cluster_gap_context(
                                current_snapshot,
                                current_cluster.instance_ids,
                                node.primary_parent_id,
                            ),
                        }
                        continue
                    if decision.action != SearchAction.ABLATE:
                        raise RuntimeError(f"target cluster disappeared: {cluster_id}")
                    evidence = next(
                        item
                        for item in reversed(state.evidence)
                        if item.hypothesis_id == decision.target_hypothesis_id
                        and item.status == "pending"
                    )
                    target_context[cluster_id] = {
                        "cluster_id": cluster_id,
                        "instance_ids": list(evidence.target_instance_ids),
                        "family_ids": [],
                        "failing_candidate_ids": [evidence.candidate_id],
                        "historical": True,
                        **_cluster_gap_context(
                            (
                                validation_snapshot
                                if cluster_id.startswith("validation:")
                                else snapshot
                            ),
                            evidence.target_instance_ids,
                            node.primary_parent_id,
                        ),
                    }
                from agent_ahd.core.research_guidance import resource_facts, retrieve_training_episodes, guidance_for_candidate
                research_guidance = None
                if self.config.enable_research_guidance:
                    memory = retrieve_training_episodes(state, self.config, self.adapter, before_round=node.round_index)
                    research_guidance = guidance_for_candidate(memory, action=decision.action.value,
                        slot=int(node.node_id.rsplit("_", 1)[-1]), stagnation=state.rounds_without_global_improvement)
                task = build_planner_candidate_task(
                    decision=decision,
                    candidate_id=node.node_id,
                    workspace=workspace.root,
                    problem_context=self.adapter.problem_context(),
                    observations={
                        item: observations[item]
                        for item in decision.observation_ids
                    },
                    target_cluster_context=target_context,
                    auxiliary_mechanism=auxiliary,
                    resource_facts=resource_facts(self.adapter, self.config),
                    research_guidance=research_guidance,
                )
                exposure = build_source_exposure(
                    workspace.root,
                    scope=decision.scope,
                    channel=decision.inheritance_channel,
                )
                _write_json(
                    workspace.root / "source_exposure.json",
                    exposure.to_record(),
                )
                generation_input = _text_sha256(task)
                CandidateStageJournal(
                    workspace.root / "stage_journal.json",
                    candidate_id=node.node_id,
                ).begin("generation", input_fingerprint=generation_input)
                task_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_task.write_text(task, encoding="utf-8")
                snapshot_root.parent.mkdir(parents=True, exist_ok=True)
                temporary_snapshot.replace(snapshot_root)
                temporary_task.replace(task_path)
                temporary_control.replace(control_root)
            finally:
                _discard_generated_tree(
                    temporary_control,
                    parent=nodes_root,
                    name_prefix=f".prepare-{node.node_id}-",
                )
                _discard_generated_tree(
                    temporary_snapshot,
                    parent=snapshot_root.parent,
                    name_prefix=f".prepare-{node.node_id}-",
                )
                _discard_generated_file(
                    temporary_task,
                    parent=task_path.parent,
                    name_prefix=f".{node.node_id}.",
                )
            workspace = control_workspace
            jobs[node.node_id] = {
                "node": node,
                "decision": decision,
                "control_workspace": control_workspace,
                "workspace": workspace,
                "snapshot_root": snapshot_root,
                "task": task,
                "source_symbols": tuple(
                    extract_source_structure(snapshot_root / "solver.py")["symbols"]
                ),
            }
            journal = CandidateStageJournal(
                control_root / "stage_journal.json",
                candidate_id=node.node_id,
            )
            jobs[node.node_id]["stage_journal"] = journal
            jobs[node.node_id]["generation_input_fingerprint"] = generation_input
        return jobs

    def _run_hypothesis_agents(self, jobs: dict[str, dict]) -> dict[str, str]:
        errors: dict[str, str] = {
            node_id: job["stage_journal"].record("generation")["metadata"].get(
                "error",
                "generation was previously rejected",
            )
            for node_id, job in jobs.items()
            if job["stage_journal"].record("generation")["status"] == "rejected"
        }
        if not jobs:
            return errors
        cached = set()
        for node_id, job in jobs.items():
            if node_id in errors:
                continue
            response_path = (
                job["control_workspace"].root / "hypothesis_response.json"
            )
            if not response_path.is_file():
                continue
            schema = build_code_hypothesis_schema(
                job["decision"],
                source_symbols=job["source_symbols"],
                task=jobs[node_id]["task"],
            )
            response = _load_agent_response_record(
                response_path,
                model=self.code_runner.model,
                logical_call_id=f"candidate:{node_id}:hypothesis",
                system_prompt=hypothesis_system_prompt(jobs[node_id]["task"]),
                user_prompt=job["task"],
                output_schema=schema,
            )
            job["hypothesis"] = parse_code_hypothesis_response(
                job["decision"],
                response,
                task=jobs[node_id]["task"],
            )
            _finish_agent_attempt(
                job["control_workspace"].root / "hypothesis_attempt.json",
                identity=_agent_attempt_identity(
                    model=self.code_runner.model,
                    logical_call_id=f"candidate:{node_id}:hypothesis",
                    system_prompt=hypothesis_system_prompt(jobs[node_id]["task"]),
                    user_prompt=job["task"],
                    output_schema=schema,
                ),
                response_path=response_path,
            )
            cached.add(node_id)
        runnable = {
            node_id: job
            for node_id, job in jobs.items()
            if node_id not in errors
            and node_id not in cached
            and job["stage_journal"].record("generation")["status"] == "running"
        }
        for node_id, job in runnable.items():
            if job["decision"].action in {SearchAction.ABLATE, SearchAction.RETEST}:
                continue
            schema = build_code_hypothesis_schema(
                job["decision"],
                source_symbols=job["source_symbols"],
                task=jobs[node_id]["task"],
            )
            _begin_agent_attempt(
                job["control_workspace"].root / "hypothesis_attempt.json",
                identity=_agent_attempt_identity(
                    model=self.code_runner.model,
                    logical_call_id=f"candidate:{node_id}:hypothesis",
                    system_prompt=hypothesis_system_prompt(jobs[node_id]["task"]),
                    user_prompt=job["task"],
                    output_schema=schema,
                ),
                allow_recovery=getattr(
                    self.code_runner,
                    "supports_durable_attempt_recovery",
                    False,
                ),
                role="hypothesis",
            )
        with ThreadPoolExecutor(max_workers=self.config.agent_workers) as executor:
            futures = {
                executor.submit(
                    request_code_hypothesis,
                    self.code_runner,
                    task=job["task"],
                    decision=job["decision"],
                    source_symbols=job["source_symbols"],
                    logical_call_id=f"candidate:{node_id}:hypothesis",
                ): node_id
                for node_id, job in runnable.items()
            }
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    hypothesis, hypothesis_response = future.result()
                    jobs[node_id]["hypothesis"] = hypothesis
                    if hypothesis_response is not None:
                        response_path = (
                            jobs[node_id]["control_workspace"].root
                            / "hypothesis_response.json"
                        )
                        try:
                            _write_json(
                                response_path,
                                {
                                    "model": self.code_runner.model,
                                    "logical_call_id": (
                                        f"candidate:{node_id}:hypothesis"
                                    ),
                                    "payload": hypothesis_response.payload,
                                    "raw_content": hypothesis_response.raw_content,
                                    "usage": hypothesis_response.usage,
                                    "system_prompt_sha256": _text_sha256(
                                        hypothesis_system_prompt(jobs[node_id]["task"])
                                    ),
                                    "user_prompt_sha256": _text_sha256(
                                        jobs[node_id]["task"]
                                    ),
                                    "output_schema_sha256": _text_sha256(
                                        json.dumps(
                                            build_code_hypothesis_schema(
                                                jobs[node_id]["decision"],
                                                source_symbols=jobs[node_id][
                                                    "source_symbols"
                                                ],
                                                task=jobs[node_id]["task"],
                                            ),
                                            sort_keys=True,
                                        )
                                    ),
                                },
                            )
                        except OSError as exc:
                            raise TransactionAmbiguityError(
                                "hypothesis response returned but could not be persisted"
                            ) from exc
                        _finish_agent_attempt(
                            jobs[node_id]["control_workspace"].root
                            / "hypothesis_attempt.json",
                            identity=_agent_attempt_identity(
                                model=self.code_runner.model,
                                logical_call_id=f"candidate:{node_id}:hypothesis",
                                system_prompt=hypothesis_system_prompt(jobs[node_id]["task"]),
                                user_prompt=jobs[node_id]["task"],
                                output_schema=build_code_hypothesis_schema(
                                    jobs[node_id]["decision"],
                                    source_symbols=jobs[node_id]["source_symbols"],
                                    task=jobs[node_id]["task"],
                                ),
                            ),
                            response_path=response_path,
                        )
                except Exception as exc:
                    job = jobs[node_id]
                    if isinstance(exc, TransactionAmbiguityError):
                        raise
                    _fail_agent_attempt(
                        job["control_workspace"].root / "hypothesis_attempt.json",
                        identity=_agent_attempt_identity(
                            model=self.code_runner.model,
                            logical_call_id=f"candidate:{node_id}:hypothesis",
                            system_prompt=hypothesis_system_prompt(jobs[node_id]["task"]),
                            user_prompt=job["task"],
                            output_schema=build_code_hypothesis_schema(
                                job["decision"],
                                source_symbols=job["source_symbols"],
                                task=jobs[node_id]["task"],
                            ),
                        ),
                        exc=exc,
                    )
                    if is_explicit_infrastructure_failure(exc):
                        raise
                    if (
                        isinstance(exc, AgentResponseError)
                        and exc.stage in {"network", "orphan"}
                    ):
                        raise
                    errors[node_id] = f"{type(exc).__name__}: {exc}"
                    journal = job["stage_journal"]
                    current = journal.record("generation")
                    if current is not None and current["status"] == "running":
                        journal.finish(
                            "generation",
                            input_fingerprint=job["generation_input_fingerprint"],
                            output_fingerprint=_text_sha256(errors[node_id]),
                            status="rejected",
                            metadata={"error": errors[node_id]},
                        )
                    self._write_generation_error(jobs[node_id], "hypothesis", exc)
        return errors

    def _run_implementation_agents(
        self,
        jobs: dict[str, dict],
        prior_errors: dict[str, str],
    ) -> dict[str, str]:
        errors = dict(prior_errors)
        runnable = {
            node_id: job
            for node_id, job in jobs.items()
            if node_id not in errors
            and job["stage_journal"].record("generation")["status"] == "running"
        }
        with ThreadPoolExecutor(max_workers=self.config.agent_workers) as executor:
            futures = {
                executor.submit(
                    self._run_implementation_transaction,
                    node_id,
                    job,
                ): node_id
                for node_id, job in runnable.items()
            }
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    commit = future.result()
                    job = jobs[node_id]
                    job["stage_journal"].finish(
                        "generation",
                        input_fingerprint=job["generation_input_fingerprint"],
                        output_fingerprint=commit["source_fingerprint"],
                    )
                except Exception as exc:
                    if isinstance(exc, TransactionAmbiguityError):
                        raise
                    if is_explicit_infrastructure_failure(exc):
                        raise
                    if (
                        isinstance(exc, AgentResponseError)
                        and exc.stage in {"network", "orphan"}
                    ):
                        raise
                    errors[node_id] = f"{type(exc).__name__}: {exc}"
                    job = jobs[node_id]
                    journal = job["stage_journal"]
                    current = journal.record("generation")
                    if current is not None and current["status"] == "running":
                        journal.finish(
                            "generation",
                            input_fingerprint=job["generation_input_fingerprint"],
                            output_fingerprint=_text_sha256(errors[node_id]),
                            status="rejected",
                            metadata={"error": errors[node_id]},
                        )
                    self._write_generation_error(jobs[node_id], "implementation", exc)
        return errors

    def _run_implementation_transaction(
        self,
        node_id: str,
        job: dict,
    ) -> dict:
        control_workspace = job["control_workspace"]
        logical_call_id = f"candidate:{node_id}:implementation"
        hypothesis = job.get("hypothesis")
        implementation_task = build_code_implementation_task(
            job["task"],
            hypothesis,
            job["decision"],
        )
        schema = build_code_implementation_schema(job["decision"])
        tool_response = None
        if self.config.enable_deterministic_ablation:
            from agent_ahd.core.deterministic_tools import exact_ablation_response
            tool_response = exact_ablation_response(job["task"], job["decision"], job["snapshot_root"])
        implementation_model = tool_response.usage["tool_name"] if tool_response is not None else self.code_runner.model
        response_path = control_workspace.root / "agent_response.json"
        attempt_path = control_workspace.root / "implementation_attempt.json"
        attempt_identity = _agent_attempt_identity(
            model=implementation_model,
            logical_call_id=logical_call_id,
            system_prompt=CODE_AGENT_SYSTEM_PROMPT,
            user_prompt=implementation_task,
            output_schema=schema,
        )
        verified = _verified_candidate_commit(control_workspace)
        if verified is not None:
            committed, workspace = verified
            _load_agent_response_record(
                response_path,
                model=implementation_model,
                logical_call_id=logical_call_id,
                system_prompt=CODE_AGENT_SYSTEM_PROMPT,
                user_prompt=implementation_task,
                output_schema=schema,
            )
            _finish_agent_attempt(
                attempt_path,
                identity=attempt_identity,
                response_path=response_path,
            )
            job["workspace"] = workspace
            job["node"].solver_relpath = workspace.solver_path.relative_to(
                self.run_dir
            ).as_posix()
            return committed

        response = None
        if response_path.is_file():
            response = _load_agent_response_record(
                response_path,
                model=implementation_model,
                logical_call_id=logical_call_id,
                system_prompt=CODE_AGENT_SYSTEM_PROMPT,
                user_prompt=implementation_task,
                output_schema=schema,
            )
            _finish_agent_attempt(
                attempt_path,
                identity=attempt_identity,
                response_path=response_path,
            )

        staging_root = control_workspace.root / (
            f".candidate-stage-{uuid.uuid4().hex}"
        )
        try:
            _copy_python_sources(job["snapshot_root"], staging_root)
            if response is None:
                _begin_agent_attempt(
                    attempt_path,
                    identity=attempt_identity,
                    allow_recovery=tool_response is not None or getattr(
                        self.code_runner,
                        "supports_durable_attempt_recovery",
                        False,
                    ),
                    role="implementation",
                )
                try:
                    response = tool_response if tool_response is not None else request_code_implementation_response(
                        self.code_runner,
                        task=job["task"],
                        hypothesis=hypothesis,
                        decision=job["decision"],
                        logical_call_id=logical_call_id,
                        validation_workspace=staging_root,
                    )
                except Exception as exc:
                    _fail_agent_attempt(
                        attempt_path,
                        identity=attempt_identity,
                        exc=exc,
                    )
                    raise
                try:
                    _write_json(response_path, {
                        "model": implementation_model,
                        "logical_call_id": logical_call_id,
                        "payload": response.payload,
                        "raw_content": response.raw_content,
                        "usage": response.usage,
                        "system_prompt_sha256": _text_sha256(
                            CODE_AGENT_SYSTEM_PROMPT
                        ),
                        "user_prompt_sha256": _text_sha256(implementation_task),
                        "output_schema_sha256": _text_sha256(
                            json.dumps(schema, sort_keys=True)
                        ),
                    })
                except OSError as exc:
                    raise TransactionAmbiguityError(
                        "implementation response returned but could not be persisted"
                    ) from exc
                _finish_agent_attempt(
                    attempt_path,
                    identity=attempt_identity,
                    response_path=response_path,
                )
            apply_code_implementation_response(
                staging_root,
                response,
                decision=job["decision"],
            )
            source_files = _candidate_file_hashes(staging_root)
            source_fingerprint = _text_sha256(
                json.dumps(source_files, sort_keys=True)
            )
            versions_path = control_workspace.root / "versions"
            if versions_path.is_symlink():
                raise RuntimeError("candidate versions directory cannot be a symlink")
            versions_path.mkdir(parents=True, exist_ok=True)
            versions_root = versions_path.resolve()
            if versions_root.parent != control_workspace.root.resolve():
                raise RuntimeError("candidate versions directory escapes workspace")
            version_root = versions_root / source_fingerprint
            if version_root.exists():
                if (
                    version_root.is_symlink()
                    or _source_tree_contains_symlink(version_root)
                    or _candidate_file_hashes(version_root) != source_files
                ):
                    raise RuntimeError("candidate immutable source version changed")
            else:
                staging_root.replace(version_root)
            commit = _write_candidate_commit(
                control_workspace,
                source_root=version_root,
                source_files=source_files,
            )
            verified = _verified_candidate_commit(control_workspace)
            if verified is None:
                raise RuntimeError("candidate commit was not published")
            _, workspace = verified
            job["workspace"] = workspace
            job["node"].solver_relpath = workspace.solver_path.relative_to(
                self.run_dir
            ).as_posix()
            return commit
        finally:
            _discard_generated_tree(
                staging_root,
                parent=control_workspace.root,
                name_prefix=".candidate-stage-",
            )

    @staticmethod
    def _write_generation_error(job: dict, stage: str, exc: Exception) -> None:
        response_error = exc if isinstance(exc, AgentResponseError) else None
        if stage == "hypothesis":
            system_prompt = hypothesis_system_prompt(job["task"])
            user_prompt = job["task"]
            output_schema = build_code_hypothesis_schema(
                job["decision"],
                source_symbols=job["source_symbols"],
                task=job["task"],
            )
        else:
            system_prompt = CODE_AGENT_SYSTEM_PROMPT
            user_prompt = build_code_implementation_task(
                job["task"],
                job.get("hypothesis"),
                job["decision"],
            )
            output_schema = build_code_implementation_schema(job["decision"])
        _write_json(
            job["control_workspace"].root / "generation_error.json",
            {
                "stage": stage,
                "failure_stage": (
                    response_error.stage if response_error is not None else stage
                ),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "raw_content": (
                    response_error.raw_content
                    if response_error is not None
                    else None
                ),
                "payload": (
                    response_error.payload
                    if response_error is not None
                    else None
                ),
                "usage": (
                    response_error.usage if response_error is not None else {}
                ),
                "system_prompt_sha256": _text_sha256(system_prompt),
                "user_prompt_sha256": _text_sha256(user_prompt),
                "output_schema_sha256": _text_sha256(
                    json.dumps(output_schema, sort_keys=True)
                ),
                "replay_error": (
                    response_error is None or response_error.payload is None
                ),
            },
        )

    def _lock_code_hypotheses(
        self,
        state: SearchState,
        plan: RoundPlan,
        jobs: dict[str, dict],
        generation_errors: dict[str, str],
    ) -> RoundPlan:
        job_by_decision = {
            job["decision"].decision_id: job for job in jobs.values()
        }
        locked_decisions = []
        for decision in plan.decisions:
            job = job_by_decision.get(decision.decision_id)
            if job is None or job["node"].node_id in generation_errors:
                locked_decisions.append(decision)
                continue
            hypothesis = job.get("hypothesis")
            locked = replace(decision, hypothesis=hypothesis)
            locked_decisions.append(locked)
            job["decision"] = locked
            _write_json(
                job["control_workspace"].root / "planner_decision.json",
                asdict(locked),
            )
            if hypothesis is None:
                continue
            if any(
                item.hypothesis_id == hypothesis.hypothesis_id
                for item in state.hypotheses
            ):
                raise RuntimeError(
                    f"duplicate hypothesis id: {hypothesis.hypothesis_id}"
                )
            state.hypotheses.append(hypothesis)
            job["node"].hypothesis_id = hypothesis.hypothesis_id
            for index, existing in enumerate(state.decisions):
                if existing.decision_id == decision.decision_id:
                    state.decisions[index] = locked
                    break
            else:
                raise RuntimeError(
                    f"materialized decision disappeared: {decision.decision_id}"
                )
        return RoundPlan(
            round_index=plan.round_index,
            decisions=tuple(locked_decisions),
        )

    def _inspect_candidate_source(
        self,
        state: SearchState,
        node: SolverNode,
        job: dict,
    ) -> tuple[CodeCheckResult, CodeCheckResult | None, CandidateArtifact | None]:
        """Read immutable sources only; the owner publishes results in node order."""
        scope_result = _check_workspace_scope(
            job["snapshot_root"], job["workspace"].root, job["decision"].scope,
        )
        if not scope_result.valid:
            return scope_result, None, None
        ablation_result = None
        if job["decision"].action == SearchAction.ABLATE:
            mechanism_node = state.require_node(node.primary_parent_id)
            mechanism_parent = (
                self.run_dir / "rounds" / f"round_{mechanism_node.round_index:03d}"
                / "source_snapshots" / mechanism_node.node_id
            )
            ablation_result = check_ablation_reversion(
                mechanism_parent, job["snapshot_root"], job["workspace"].root,
                job["decision"].scope,
            )
            if not ablation_result.valid:
                return scope_result, ablation_result, None
        artifact = collect_candidate_artifact(
            workspace=job["workspace"],
            baseline_path=self.adapter.baseline_path,
            comparison_path=(
                self.run_dir / state.require_node(node.primary_parent_id).solver_relpath
                if node.primary_parent_id else self.adapter.baseline_path
            ),
            comparison_label=node.primary_parent_id or "baseline",
        )
        return scope_result, ablation_result, artifact

    def _evaluate_round(
        self,
        state: SearchState,
        plan: RoundPlan,
        nodes: tuple[SolverNode, ...],
        retests: tuple[PlannerDecision, ...],
        jobs: dict[str, dict],
        generation_errors: dict[str, str],
        round_path: Path,
    ) -> dict[str, EvaluationRecord]:
        records: dict[str, EvaluationRecord] = {}
        receipt_path = round_path / "evaluation_receipt.json"
        attempt_path = round_path / "evaluation_attempt.json"
        receipt = (
            json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt_path.is_file()
            else None
        )
        attempt = (
            json.loads(attempt_path.read_text(encoding="utf-8"))
            if attempt_path.is_file()
            else None
        )
        receipt_candidate_ids = {
            str(record.get("candidate_id"))
            for record in (receipt or {}).get("results", [])
            if isinstance(record, dict) and record.get("candidate_id") is not None
        }
        submitted_artifacts: dict[str, CandidateArtifact] = {}
        tasks = []
        task_targets: dict[str, tuple[str, str, str, str]] = {}
        evaluation_cache = EvaluationCache(self.run_dir / "evaluation_cache")
        cache_fingerprints: dict[str, EvaluationFingerprint] = {}
        pending_cache_leaders: dict[str, str] = {}
        duplicate_waiters: dict[str, list[tuple[str, str]]] = {}
        next_evaluation_index = {
            node.node_id: len(node.evaluation_ids)
            for node in state.nodes
        }
        historical_file_hashes = [
            node.solver_file_sha256
            for node in state.nodes
            if node.solver_file_sha256 is not None
        ]

        with ThreadPoolExecutor(max_workers=self.config.evaluation_workers) as pool:
            inspections = {
                node.node_id: pool.submit(
                    self._inspect_candidate_source, state, node, jobs[node.node_id],
                )
                for node in nodes if node.node_id not in generation_errors
            }
            # Completion order must never choose cache leaders or mutate journals.
            for node in nodes:
                job = jobs[node.node_id]
                journal = job["stage_journal"]
                generation_record = journal.record("generation")
                static_input = generation_record["output_fingerprint"]
                static_started = journal.begin(
                    "static_check",
                    input_fingerprint=static_input,
                )
                if node.node_id in generation_errors:
                    records[node.node_id] = _invalid_record(
                        node.node_id,
                        f"eval_{node.node_id}_000",
                        "generation_error",
                        generation_errors[node.node_id],
                    )
                    _reject_and_skip_candidate_stages(
                        journal,
                        static_input=static_input,
                        error=records[node.node_id].error_message or "generation error",
                    )
                    continue
                scope_result, ablation_result, artifact = inspections[node.node_id].result()
                if not scope_result.valid:
                    records[node.node_id] = _invalid_record(
                        node.node_id,
                        f"eval_{node.node_id}_000",
                        scope_result.error_type or "scope_violation",
                        scope_result.error_message or "scope check failed",
                    )
                    _reject_and_skip_candidate_stages(
                        journal,
                        static_input=static_input,
                        error=records[node.node_id].error_message or "scope check failed",
                    )
                    continue
                if ablation_result is not None:
                    if not ablation_result.valid:
                        records[node.node_id] = _invalid_record(
                            node.node_id,
                            f"eval_{node.node_id}_000",
                            ablation_result.error_type or "invalid_ablation",
                            ablation_result.error_message or "ablation check failed",
                        )
                        _reject_and_skip_candidate_stages(
                            journal,
                            static_input=static_input,
                            error=records[node.node_id].error_message or "ablation check failed",
                        )
                        continue
                _write_candidate_artifacts(job["control_workspace"], artifact)
                submitted_artifacts[node.node_id] = artifact
                node.solver_sha256 = artifact.source_sha256
                node.solver_file_sha256 = artifact.file_sha256
                ablation_contract = _check_ablation_contract(
                    job["snapshot_root"],
                    job["workspace"].root,
                    job.get("hypothesis"),
                )
                if not ablation_contract.valid:
                    records[node.node_id] = _invalid_record(
                        node.node_id,
                        f"eval_{node.node_id}_000",
                        ablation_contract.error_type or "ablation_scope_without_change",
                        ablation_contract.error_message or "ablation scope has no changed symbol",
                    )
                elif not artifact.changed:
                    records[node.node_id] = _invalid_record(
                        node.node_id,
                        f"eval_{node.node_id}_000",
                        "unchanged_candidate",
                        "candidate is identical to its generation source",
                    )
                else:
                    historical_file_hashes.append(artifact.file_sha256)
                    if static_started:
                        journal.finish(
                            "static_check",
                            input_fingerprint=static_input,
                            output_fingerprint=artifact.source_sha256,
                            metadata={
                                "source_sha256": artifact.source_sha256,
                                "file_sha256": artifact.file_sha256,
                            },
                        )
                    _record_quick_screen_skip(journal, artifact.source_sha256)
                    task = self.adapter.make_task(
                        candidate_id=node.node_id,
                        candidate_path=job["workspace"].solver_path,
                        budget=self.config.base_budget(),
                        split="train",
                    )
                    evaluation_id = (
                        f"eval_{node.node_id}_{next_evaluation_index[node.node_id]:03d}"
                    )
                    next_evaluation_index[node.node_id] += 1
                    fingerprint = _evaluation_fingerprint(
                        task,
                        source_sha256=artifact.source_sha256,
                    )
                    development_input = fingerprint.cache_key
                    development_started = journal.begin(
                        "development_evaluation",
                        input_fingerprint=development_input,
                    )
                    job["development_input_fingerprint"] = development_input
                    job["development_stage_started"] = development_started
                    cached = (
                        None
                        if task.candidate_id in receipt_candidate_ids
                        else evaluation_cache.load(fingerprint)
                    )
                    if cached is None:
                        cached_record = (
                            None
                            if task.candidate_id in receipt_candidate_ids
                            else _matching_cached_state_record(
                                state,
                                source_sha256=artifact.source_sha256,
                                task=task,
                            )
                        )
                        if cached_record is not None:
                            evaluation_cache.store(fingerprint, cached_record.to_dict())
                            cached = cached_record.to_dict()
                    if cached is not None:
                        record = replace(
                            EvaluationRecord.from_dict(cached),
                            evaluation_id=evaluation_id,
                            candidate_id=node.node_id,
                        )
                        records[node.node_id] = record
                        if development_started:
                            journal.finish(
                                "development_evaluation",
                                input_fingerprint=development_input,
                                output_fingerprint=_text_sha256(
                                    json.dumps(record.to_dict(), sort_keys=True)
                                ),
                                status="complete" if record.valid else "rejected",
                                metadata={"cache_hit": True, "valid": record.valid},
                            )
                    elif fingerprint.cache_key in pending_cache_leaders:
                        leader = pending_cache_leaders[fingerprint.cache_key]
                        duplicate_waiters.setdefault(leader, []).append(
                            (node.node_id, evaluation_id)
                        )
                    else:
                        pending_cache_leaders[fingerprint.cache_key] = task.candidate_id
                        cache_fingerprints[task.candidate_id] = fingerprint
                        tasks.append(task)
                        task_targets[task.candidate_id] = (
                            node.node_id,
                            "train",
                            evaluation_id,
                            node.node_id,
                        )
                if (
                    node.node_id in records
                    and journal.record("static_check")["status"] == "running"
                ):
                    _reject_and_skip_candidate_stages(
                        journal,
                        static_input=static_input,
                        error=records[node.node_id].error_message or "static check failed",
                    )
        for decision in retests:
            target = state.require_node(decision.primary_parent_id)
            job_id = f"retest_{decision.decision_id}"
            budget = EvaluationBudget(
                n_instances=self.config.n_instances,
                time_limit=self.config.time_limit,
                ite_max=self.config.ite_max,
                seed=decision.retest_seeds[0],
                seeds=decision.retest_seeds,
            )
            task = self.adapter.make_task(
                candidate_id=job_id,
                candidate_path=self.run_dir / target.solver_relpath,
                budget=budget,
                split=decision.retest_split,
                instance_ids=decision.retest_instance_ids,
            )
            tasks.append(task)
            evaluation_id = (
                f"eval_{target.node_id}_{next_evaluation_index[target.node_id]:03d}"
            )
            next_evaluation_index[target.node_id] += 1
            task_targets[job_id] = (
                target.node_id,
                decision.retest_split,
                evaluation_id,
                job_id,
            )

        task_fingerprint = _evaluation_task_batch_fingerprint(tasks)
        candidate_ids = [task.candidate_id for task in tasks]
        if attempt is not None:
            if set(attempt) != {
                "candidate_ids",
                "receipt_sha256",
                "status",
                "task_fingerprint",
                "version",
            }:
                raise RuntimeError("evaluation attempt fields are invalid")
            if (
                attempt.get("version") != 1
                or attempt.get("task_fingerprint") != task_fingerprint
                or attempt.get("candidate_ids") != candidate_ids
                or attempt.get("status") not in {"dispatched", "returned"}
            ):
                raise RuntimeError("evaluation attempt identity changed")
        if receipt is None:
            if attempt is not None:
                raise TransactionAmbiguityError(
                    "ambiguous evaluation attempt; automatic redispatch is forbidden"
                )
            if tasks:
                _write_json(attempt_path, {
                    "version": 1,
                    "status": "dispatched",
                    "task_fingerprint": task_fingerprint,
                    "candidate_ids": candidate_ids,
                    "receipt_sha256": None,
                })
            evaluated = self.evaluation_function(
                tasks,
                max_workers=self.config.evaluation_workers,
            ) if tasks else []
            _write_evaluation_receipt(
                receipt_path,
                task_fingerprint=task_fingerprint,
                results=evaluated,
            )
            if tasks:
                _write_json(attempt_path, {
                    "version": 1,
                    "status": "returned",
                    "task_fingerprint": task_fingerprint,
                    "candidate_ids": candidate_ids,
                    "receipt_sha256": _file_sha256(receipt_path),
                })
        else:
            if receipt.get("version") != 1:
                raise RuntimeError("evaluation receipt version is invalid")
            if receipt.get("task_fingerprint") != task_fingerprint:
                raise RuntimeError("evaluation receipt task fingerprint changed")
            raw_results = receipt.get("results")
            if not isinstance(raw_results, list):
                raise RuntimeError("evaluation receipt results are invalid")
            if tasks:
                if attempt is None:
                    raise RuntimeError("evaluation receipt has no dispatch attempt")
                receipt_sha256 = _file_sha256(receipt_path)
                locked_receipt = attempt.get("receipt_sha256")
                if locked_receipt not in {None, receipt_sha256}:
                    raise RuntimeError("evaluation receipt changed after dispatch")
                if attempt.get("status") != "returned" or locked_receipt is None:
                    _write_json(attempt_path, {
                        "version": 1,
                        "status": "returned",
                        "task_fingerprint": task_fingerprint,
                        "candidate_ids": candidate_ids,
                        "receipt_sha256": receipt_sha256,
                    })
            evaluated = [EvaluationResult(**record) for record in raw_results]
        tasks_by_id = {task.candidate_id: task for task in tasks}
        for result in evaluated:
            target_id, split, evaluation_id, key = task_targets[result.candidate_id]
            task = tasks_by_id[result.candidate_id]
            record = _result_record(
                result,
                candidate_id=target_id,
                evaluation_id=evaluation_id,
                split=split,
                requested_seeds=task.budget.resolved_seeds,
                requested_instance_ids=task.instance_ids,
                expected_metric_name=task.metric_name,
                expected_metric_unit=task.metric_unit,
            )
            submitted = submitted_artifacts.get(target_id)
            if submitted is not None:
                current_hashes = _candidate_file_hashes(
                    jobs[target_id]["workspace"].root
                )
                if current_hashes != submitted.file_sha256:
                    record = _invalid_record(
                        target_id,
                        evaluation_id,
                        "source_changed_during_evaluation",
                        "candidate source changed after submission hashing",
                        total_runtime_sec=result.runtime_sec,
                        split=split,
                        seeds=task.budget.resolved_seeds,
                        instance_ids=task.instance_ids,
                    )
            records[key] = record
            fingerprint = cache_fingerprints.get(result.candidate_id)
            if fingerprint is not None:
                evaluation_cache.store(fingerprint, record.to_dict())
            if target_id in jobs:
                job = jobs[target_id]
                if job.get("development_stage_started", True):
                    job["stage_journal"].finish(
                        "development_evaluation",
                        input_fingerprint=job["development_input_fingerprint"],
                        output_fingerprint=_text_sha256(
                            json.dumps(record.to_dict(), sort_keys=True)
                        ),
                        status="complete" if record.valid else "rejected",
                        metadata={"valid": record.valid},
                    )
            for duplicate_id, duplicate_evaluation_id in duplicate_waiters.get(
                result.candidate_id,
                (),
            ):
                duplicate_record = replace(
                    record,
                    evaluation_id=duplicate_evaluation_id,
                    candidate_id=duplicate_id,
                )
                records[duplicate_id] = duplicate_record
                duplicate_job = jobs[duplicate_id]
                if duplicate_job.get("development_stage_started", True):
                    duplicate_job["stage_journal"].finish(
                        "development_evaluation",
                        input_fingerprint=duplicate_job[
                            "development_input_fingerprint"
                        ],
                        output_fingerprint=_text_sha256(
                            json.dumps(duplicate_record.to_dict(), sort_keys=True)
                        ),
                        status="complete" if duplicate_record.valid else "rejected",
                        metadata={
                            "cache_hit": True,
                            "same_batch_reuse": True,
                            "valid": duplicate_record.valid,
                        },
                    )

        for node in nodes:
            record = records[node.node_id]
            state.record_evaluation_record(record)
        for decision in retests:
            key = f"retest_{decision.decision_id}"
            state.record_evaluation_record(records[key], update_node=False)
        for key, record in records.items():
            if key.startswith("evidence:"):
                state.record_evaluation_record(record, update_node=False)

        _write_json(
            round_path / "evaluations.json",
            {
                "records": {
                    key: record.to_dict() for key, record in sorted(records.items())
                }
            },
        )
        return records

    def _update_evidence(
        self,
        state: SearchState,
        plan: RoundPlan,
        round_records: dict[str, EvaluationRecord],
    ) -> list[EvidenceRecord]:
        updates = []
        decision_by_id = {item.decision_id: item for item in plan.decisions}
        for node in state.nodes:
            if node.round_index != plan.round_index or node.decision_id is None:
                continue
            decision = decision_by_id[node.decision_id]
            if decision.action == SearchAction.ABLATE:
                existing = next(
                    (
                        item
                        for item in reversed(state.evidence)
                        if item.hypothesis_id == decision.target_hypothesis_id
                        and item.status == "pending"
                    ),
                    None,
                )
                if existing is None:
                    raise RuntimeError("ablation target evidence disappeared")
                ablation_record = _decision_evidence_record(
                    round_records,
                    node.node_id,
                )
                if not round_records[node.node_id].valid or not ablation_record.valid:
                    failed_record = (
                        round_records[node.node_id]
                        if not round_records[node.node_id].valid
                        else ablation_record
                    )
                    update = EvidenceRecord(
                        evidence_id=(
                            f"evidence:{existing.hypothesis_id}:"
                            f"ablation_attempt:{node.node_id}"
                        ),
                        hypothesis_id=existing.hypothesis_id,
                        parent_id=existing.candidate_id,
                        candidate_id=node.node_id,
                        status="invalid",
                        paired_statistics={},
                        ablation_id=node.node_id,
                        target_instance_ids=existing.target_instance_ids,
                        target_cluster_ids=existing.target_cluster_ids,
                        source_action=SearchAction.ABLATE.value,
                        failure_reason=(
                            f"{failed_record.error_type}: "
                            f"{failed_record.error_message}"
                        ),
                    )
                    state.evidence.append(update)
                    updates.append(update)
                    continue
                hypothesis = _require_hypothesis(state, existing.hypothesis_id)
                parent_record = _paired_record(state, existing.parent_id, ablation_record)
                candidate_record = _paired_record(state, existing.candidate_id, ablation_record)
                update = update_hypothesis_evidence(
                    hypothesis,
                    parent_record,
                    candidate_record,
                    ablation=ablation_record,
                    target_instance_ids=existing.target_instance_ids,
                    source_action=existing.source_action,
                )
                state.evidence.remove(existing)
                state.evidence.append(update)
                updates.append(update)
            elif decision.hypothesis is not None:
                child_record = _decision_evidence_record(
                    round_records,
                    node.node_id,
                )
                if not round_records[node.node_id].valid or not child_record.valid:
                    failed_record = (
                        round_records[node.node_id]
                        if not round_records[node.node_id].valid
                        else child_record
                    )
                    update = EvidenceRecord(
                        evidence_id=(
                            f"evidence:{decision.hypothesis.hypothesis_id}:"
                            f"{node.node_id}"
                        ),
                        hypothesis_id=decision.hypothesis.hypothesis_id,
                        parent_id=node.primary_parent_id or BASELINE_CONTROL_ID,
                        candidate_id=node.node_id,
                        status="invalid",
                        paired_statistics={},
                        target_cluster_ids=decision.target_cluster_ids,
                        source_action=decision.action.value,
                        failure_reason=(
                            f"{failed_record.error_type}: {failed_record.error_message}"
                        ),
                    )
                    state.evidence.append(update)
                    updates.append(update)
                    continue
                parent_id = node.primary_parent_id or BASELINE_CONTROL_ID
                parent_record = _paired_record(state, parent_id, child_record)
                target_instances = _target_instances_from_plan(
                    plan,
                    decision,
                    self.run_dir / "rounds" / f"round_{plan.round_index:03d}"
                    / "archive_snapshot.json",
                )
                update = update_hypothesis_evidence(
                    decision.hypothesis,
                    parent_record,
                    child_record,
                    target_instance_ids=target_instances,
                    source_action=decision.action.value,
                )
                state.evidence.append(update)
                updates.append(update)
        return updates

    def _mechanism_transfer_records(
        self,
        state: SearchState,
        evidence_records,
    ) -> tuple[MechanismTransferRecord, ...]:
        protocol = json.loads(
            (self.run_dir / PLANNER_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        registry = protocol.get("mechanism_transfer_registry")
        if not isinstance(registry, dict):
            raise RuntimeError("mechanism transfer registry is not locked")
        return load_mechanism_transfer_registry(
            self.run_dir / str(registry["relpath"]),
            expected_hash=str(registry["registry_hash"]),
        )

    def _auxiliary_mechanism(
        self,
        state: SearchState,
        decision: PlannerDecision,
        *,
        mechanism_records: tuple[MechanismTransferRecord, ...],
    ) -> dict | None:
        if decision.action == SearchAction.ABLATE:
            hypothesis = _require_hypothesis(state, decision.target_hypothesis_id)
            evidence = next(
                item
                for item in reversed(state.evidence)
                if item.hypothesis_id == decision.target_hypothesis_id
                and item.status == "pending"
            )
            if decision.primary_parent_id != evidence.candidate_id:
                raise ValueError(
                    "ABLATE primary parent must be the mechanism-on candidate"
                )
            mechanism_node = state.require_node(evidence.candidate_id)
            parent_snapshot = (
                self.run_dir
                / "rounds"
                / f"round_{mechanism_node.round_index:03d}"
                / "source_snapshots"
                / mechanism_node.node_id
            )
            return {
                "hypothesis": asdict(hypothesis),
                "evidence": asdict(evidence),
                "mechanism_off_source": {
                    path.relative_to(parent_snapshot).as_posix(): path.read_text(
                        encoding="utf-8"
                    )
                    for path in sorted(parent_snapshot.rglob("*.py"))
                },
                "instruction": (
                    "Restore the registered scope exactly from mechanism_off_source "
                    "while preserving every change outside that scope."
                ),
            }
        if decision.action != SearchAction.CROSS:
            return None
        record = next(
            item
            for item in mechanism_records
            if item.hypothesis_id == decision.target_hypothesis_id
            and item.candidate_id == decision.secondary_parent_id
            and item.transfer_eligible
        )
        return {
            "mechanism_transfer_record": record.to_dict(),
            "instruction": (
                "Reimplement only the transfer-qualified mechanism record "
                "without access to its source, diff, variable names, or control flow."
            ),
        }

    def _finalize_round(self, state: SearchState, round_index: int) -> None:
        if state.pending_node_ids:
            raise RuntimeError(f"planner round has pending nodes: {state.pending_node_ids}")
        snapshot = build_archive_snapshot(
            state.evaluations,
            view_size=self.config.archive_view_size,
            tail_fraction=self.config.tail_fraction,
            excluded_candidate_ids=_archive_exclusions(state),
            canonical_instance_ids=_canonical_instance_ids(
                state,
                self.adapter,
                self.config,
                split="train",
            ),
            instance_families=self.adapter.train_manifest.family_by_instance,
            structural_signatures=_structural_signatures(state, self.run_dir),
        )
        if snapshot.mean_elites:
            previous_best = (
                state.require_node(state.best_node_id).score
                if state.best_node_id is not None
                else None
            )
            new_best_score = snapshot.profiles[snapshot.mean_elites[0]].mean_score
            if previous_best is None or new_best_score < previous_best - 1e-12:
                state.rounds_without_global_improvement = 0
            else:
                state.rounds_without_global_improvement += 1
            state.best_node_id = snapshot.mean_elites[0]
            for candidate_id, profile in snapshot.profiles.items():
                try:
                    state.require_node(candidate_id).score = profile.mean_score
                except KeyError:
                    from agent_ahd.core.final_selection import STABLE_CONTROL_ID

                    if candidate_id != STABLE_CONTROL_ID:
                        raise
        state.finish_round(round_index, require_nodes=False)


def load_planner_engine_config(run_dir: str | Path) -> PlannerEngineConfig:
    data = json.loads(
        (Path(run_dir).resolve() / PLANNER_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    if data.get("policy") not in SEARCH_POLICIES:
        raise ValueError("run has an unknown search policy")
    return PlannerEngineConfig.from_dict(data["engine"])


def verify_planner_run_integrity(
    run_dir: str | Path,
    adapter: ProblemAdapter,
) -> SearchState:
    root = Path(run_dir).resolve()
    state = load_search_state(root / STATE_FILENAME)
    if state.baseline_sha256 != _file_sha256(adapter.baseline_path):
        raise RuntimeError("baseline source changed after run initialization")
    protocol = json.loads(
        (root / PLANNER_CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    baseline_control = protocol.get("baseline_control")
    baseline_root = root / "controls" / "baseline"
    if (
        not isinstance(baseline_control, dict)
        or _candidate_file_hashes(baseline_root)
        != baseline_control.get("source_files")
    ):
        raise RuntimeError("baseline control source changed after run initialization")
    stable = protocol.get("stable_control")
    if stable is not None:
        stable_root = root / "controls" / "stable"
        stable_solver = root / stable["solver_relpath"]
        if (
            not stable_solver.is_file()
            or not stable_solver.is_relative_to(stable_root)
            or _candidate_file_hashes(stable_root) != stable.get("source_files")
        ):
            raise RuntimeError("stable control source changed after run initialization")
    contract_path = root / ROOT_PROTOCOL_FILENAME
    if not contract_path.is_file():
        raise RuntimeError("root protocol contract is missing")
    root_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_root_contract(root_contract)
    if protocol.get("root_protocol_hash") != root_contract["root_hash"]:
        raise RuntimeError("planner config root protocol hash mismatch")
    locked_config = {
        key: value
        for key, value in protocol.items()
        if key not in {"initialized_at_epoch_sec", "locked_artifacts", "root_protocol_hash"}
    }
    if root_contract.get("config") != locked_config:
        raise RuntimeError("planner config differs from the locked root contract")
    if protocol.get("protocol_versions") != dict(PROTOCOL_VERSIONS):
        raise RuntimeError("planner config protocol versions are unsupported")
    if protocol.get("candidate_execution") != candidate_execution_contract():
        raise RuntimeError("candidate execution protocol is unsupported")
    expected_protocol_schema = (
        CURRENT_PROTOCOL_SCHEMA
        if (protocol.get("engine") or {}).get("protocol_mode") == "v4"
        else PROTOCOL_VERSIONS["schema"]
    )
    if protocol.get("protocol_schema") != expected_protocol_schema:
        raise RuntimeError("planner config protocol schema is unsupported")
    expected = protocol.get("locked_artifacts", {})
    actual = _protocol_artifact_hashes(adapter)
    if actual != expected:
        changed = sorted(
            name
            for name in set(actual) | set(expected)
            if actual.get(name) != expected.get(name)
        )
        raise RuntimeError(f"locked experiment artifacts changed: {changed}")
    dependency_lock_path = Path(__file__).resolve().parents[1] / "requirements.runtime.lock"
    locked_dependencies = verify_dependency_lock(dependency_lock_path)
    preflight = protocol.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("status") not in {None, "passed"}:
        raise RuntimeError("planner preflight record is missing or failed")
    if preflight.get("dependency_lock_sha256") != _file_sha256(dependency_lock_path):
        raise RuntimeError("dependency lock changed after run initialization")
    if preflight.get("locked_dependencies") != locked_dependencies:
        raise RuntimeError("locked dependency versions changed after run initialization")
    _verify_node_sources(root, state)
    return state


def _result_record(
    result: EvaluationResult,
    *,
    candidate_id: str,
    evaluation_id: str,
    split: str,
    requested_seeds: tuple[int, ...] = (),
    requested_instance_ids: tuple[str, ...] = (),
    expected_metric_name: str | None = None,
    expected_metric_unit: str | None = None,
) -> EvaluationRecord:
    if expected_metric_name is not None and (result.metric_name, result.metric_unit) != (expected_metric_name, expected_metric_unit):
        raise ValueError("evaluator result metric differs from the submitted task")
    if result.metric_name != "optimality_gap" and result.gap_matrix:
        raise ValueError("raw objective result must not contain a gap matrix")
    matrix = result.gap_matrix if result.metric_name == "optimality_gap" else result.objective_matrix
    valid = bool(result.valid and matrix and result.instance_ids and result.seeds)
    if valid:
        if requested_seeds and tuple(result.seeds) != requested_seeds:
            raise ValueError("evaluator seeds differ from the submitted task")
        if requested_instance_ids and tuple(result.instance_ids) != requested_instance_ids:
            raise ValueError("evaluator instances differ from the submitted task")
        measured_mean = fmean(value for row in matrix for value in row)
        if result.score is None or not isclose(result.score, measured_mean, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("evaluator score differs from the declared metric matrix mean")
    return EvaluationRecord(
        evaluation_id=evaluation_id,
        candidate_id=candidate_id,
        split=split,
        seeds=tuple(result.seeds or ()) if valid else requested_seeds,
        instance_ids=(
            tuple(result.instance_ids or ()) if valid else requested_instance_ids
        ),
        gap_matrix=(
            tuple(tuple(float(value) for value in row) for row in result.gap_matrix)
            if valid and result.metric_name == "optimality_gap"
            else ()
        ),
        runtime_matrix=(
            tuple(tuple(float(value) for value in row) for row in result.runtime_matrix)
            if valid and result.runtime_matrix
            else ()
        ),
        valid=valid,
        total_runtime_sec=result.runtime_sec,
        error_type=result.error_type if not valid else None,
        error_message=result.error_message if not valid else None,
        objective_matrix=tuple(tuple(float(value) for value in row)
                               for row in result.objective_matrix or ()) if valid else (),
        objective_name=result.objective_name,
        objective_unit=result.objective_unit,
        metric_name=result.metric_name,
        metric_unit=result.metric_unit,
        timeout_timing=(
            dict(result.timeout_timing)
            if not valid and result.timeout_timing is not None
            else None
        ),
    )


def _invalid_record(
    candidate_id: str,
    evaluation_id: str,
    error_type: str,
    error_message: str,
    *,
    total_runtime_sec: float = 0.0,
    split: str = "train",
    seeds: tuple[int, ...] = (),
    instance_ids: tuple[str, ...] = (),
) -> EvaluationRecord:
    return EvaluationRecord(
        evaluation_id=evaluation_id,
        candidate_id=candidate_id,
        split=split,
        seeds=seeds,
        instance_ids=instance_ids,
        gap_matrix=(),
        runtime_matrix=(),
        valid=False,
        total_runtime_sec=total_runtime_sec,
        error_type=error_type,
        error_message=error_message,
    )


def _check_workspace_scope(source_root: Path, workspace_root: Path, scope):
    return check_scope_tree_change(source_root, workspace_root, scope)


def _check_ablation_contract(
    source_root: Path,
    workspace_root: Path,
    hypothesis,
) -> CodeCheckResult:
    if hypothesis is None:
        return CodeCheckResult(True)
    try:
        changed = changed_symbols_in_scope(
            source_root,
            workspace_root,
            hypothesis.ablation_scope,
        )
    except (OSError, SyntaxError) as exc:
        return CodeCheckResult(False, "ablation_scope_error", str(exc))
    if changed:
        return CodeCheckResult(True)
    return CodeCheckResult(
        False,
        "ablation_scope_without_change",
        "registered ablation scope does not identify any changed symbol",
    )


def _paired_record(
    state: SearchState,
    candidate_id: str,
    template: EvaluationRecord,
) -> EvaluationRecord:
    for record in reversed(state.evaluations):
        if (
            record.candidate_id == candidate_id
            and record.valid
            and record.split == template.split
            and record.seeds == template.seeds
            and record.instance_ids == template.instance_ids
            and (record.metric_name, record.metric_unit) == (template.metric_name, template.metric_unit)
        ):
            return record
    raise RuntimeError(
        f"no paired evaluation for {candidate_id} on {template.split}, "
        f"seeds={template.seeds}"
    )


def _decision_evidence_record(
    records: dict[str, EvaluationRecord],
    node_id: str,
) -> EvaluationRecord:
    return records[node_id]


def _require_hypothesis(state: SearchState, hypothesis_id: str):
    for hypothesis in state.hypotheses:
        if hypothesis.hypothesis_id == hypothesis_id:
            return hypothesis
    raise RuntimeError(f"unknown hypothesis: {hypothesis_id}")


def _target_instances_from_plan(
    plan: RoundPlan,
    decision: PlannerDecision,
    archive_path: Path,
) -> tuple[str, ...]:
    if not decision.target_cluster_ids:
        return ()
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    instances = []
    for cluster_id in decision.target_cluster_ids:
        split, raw_cluster_id = cluster_id.split(":", 1)
        instances.extend(
            archive[split]["failure_clusters"][raw_cluster_id]["instance_ids"]
        )
    return tuple(dict.fromkeys(instances))


def _failure_cluster_registry(
    train: ArchiveSnapshot,
    validation: ArchiveSnapshot,
) -> dict[str, tuple[str, ArchiveSnapshot, object]]:
    return {
        f"{split}:{cluster_id}": (split, snapshot, cluster)
        for split, snapshot in (("train", train), ("validation", validation))
        for cluster_id, cluster in snapshot.failure_clusters.items()
    }


def _budget_usage(
    state: SearchState,
    adapter: ProblemAdapter,
    config: PlannerEngineConfig,
    *,
    run_dir: str | Path | None = None,
) -> dict[str, int]:
    proposal_nodes = [node for node in state.nodes if node.round_index >= 0]
    candidate_ids = {node.node_id for node in proposal_nodes}
    confirmation_evaluations = _current_confirmation_evaluations(state, run_dir)
    evaluations = [
        record for record in state.evaluations if record.candidate_id in candidate_ids
    ]
    primary_evaluations = [
        record
        for record in evaluations
        if record.candidate_id in candidate_ids
        and record.split == "train"
        and record.evaluation_id == f"eval_{record.candidate_id}_000"
    ]
    diagnostic_evaluations = [
        record
        for record in evaluations
        if record not in primary_evaluations
        and not record.evaluation_id.startswith("final_confirm_")
    ]
    # The baseline alias costs no new evaluation. Actual retests of that parent
    # still spend diagnostic budget, although the parent is not a proposal.
    if config.enable_baseline_parent:
        from agent_ahd.core.baseline_parent import SEED_ID
        diagnostic_evaluations.extend(record for record in state.evaluations
            if record.candidate_id == SEED_ID
            and record.evaluation_id != "eval_seed_baseline_000"
            and not record.evaluation_id.startswith("final_confirm_"))
    candidate_evaluation_units = sum(
        evaluation_unit_charge(
            seeds=record.seeds,
            instance_ids=record.instance_ids,
            started=True,
        )
        for record in primary_evaluations
    )
    diagnostic_evaluation_units = sum(
        evaluation_unit_charge(
            seeds=record.seeds,
            instance_ids=record.instance_ids,
            started=True,
        )
        for record in diagnostic_evaluations
    )
    confirmation_evaluation_units = sum(
        evaluation_unit_charge(
            seeds=record.seeds,
            instance_ids=record.instance_ids,
            started=True,
        )
        for record in confirmation_evaluations
    )
    return {
        "candidates": len(proposal_nodes),
        "evaluation_records": len(evaluations),
        "candidate_evaluation_units": candidate_evaluation_units,
        "diagnostic_evaluation_units": diagnostic_evaluation_units,
        "confirmation_evaluation_units": confirmation_evaluation_units,
        "evaluation_units": (
            candidate_evaluation_units
            + diagnostic_evaluation_units
            + confirmation_evaluation_units
        ),
    }


def _current_confirmation_evaluations(
    state: SearchState,
    run_dir: str | Path | None,
) -> list[EvaluationRecord]:
    if run_dir is None:
        return []
    confirmation_path = Path(run_dir).resolve() / "final_confirmation.json"
    if not confirmation_path.is_file():
        return []
    manifest = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("final confirmation manifest must be an object")
    candidate_ids = set()
    for field in ("finalist_ids", "control_ids"):
        values = manifest.get(field)
        if not isinstance(values, list) or any(type(value) is not str for value in values):
            raise RuntimeError(f"final confirmation {field} must be a string list")
        candidate_ids.update(values)
    expected = {
        f"final_confirm_{candidate_id}": candidate_id
        for candidate_id in candidate_ids
    }
    records = []
    seen = set()
    for record in state.evaluations:
        expected_candidate = expected.get(record.evaluation_id)
        if expected_candidate is None:
            continue
        if record.candidate_id != expected_candidate:
            raise RuntimeError("final confirmation record candidate mismatch")
        if record.evaluation_id in seen:
            raise RuntimeError("duplicate final confirmation evaluation record")
        seen.add(record.evaluation_id)
        records.append(record)
    return records


def _load_round_evaluation_records(path: Path) -> dict[str, EvaluationRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, dict) or not records:
        raise RuntimeError("evaluation stage output is incomplete")
    return {
        str(key): EvaluationRecord.from_dict(value)
        for key, value in records.items()
    }


def _hydrate_round_evaluation_records(
    state: SearchState,
    records: dict[str, EvaluationRecord],
) -> None:
    existing = {record.evaluation_id: record for record in state.evaluations}
    for key, record in records.items():
        current = existing.get(record.evaluation_id)
        if current is not None:
            if current != record:
                raise RuntimeError(
                    f"evaluation record changed during resume: {record.evaluation_id}"
                )
            continue
        state.record_evaluation_record(
            record,
            update_node=not key.startswith(("retest_", "evidence:")),
        )


def _hydrate_evidence_updates(
    state: SearchState,
    updates: list[EvidenceRecord],
) -> None:
    for update in updates:
        matches = [
            item for item in state.evidence if item.evidence_id == update.evidence_id
        ]
        if matches:
            if len(matches) != 1 or matches[0] != update:
                raise RuntimeError(
                    f"evidence record changed during resume: {update.evidence_id}"
                )
            continue
        if update.ablation_id is not None:
            state.evidence = [
                item
                for item in state.evidence
                if not (
                    item.hypothesis_id == update.hypothesis_id
                    and item.status == "pending"
                )
            ]
        state.evidence.append(update)


def _reconcile_round_budget(
    *,
    usage_before: dict,
    planned_budget: dict,
    usage_after: dict,
) -> dict:
    charged_fields = (
        "candidates",
        "candidate_evaluation_units",
        "diagnostic_evaluation_units",
    )
    charged = {
        name: int(usage_after.get(name, 0)) - int(usage_before.get(name, 0))
        for name in charged_fields
    }
    if any(value < 0 for value in charged.values()):
        raise RuntimeError("budget ledger usage decreased during round closure")
    if charged["candidates"] != int(planned_budget.get("candidates", -1)):
        raise RuntimeError("candidate budget ledger does not match the locked plan")
    overages = {
        name: {
            "charged": charged[name],
            "planned": int(planned_budget.get(name, 0)),
        }
        for name in charged_fields[1:]
        if charged[name] > int(planned_budget.get(name, 0))
    }
    return {
        "usage_before": {
            name: int(usage_before.get(name, 0)) for name in charged_fields
        },
        "planned": {
            name: int(planned_budget.get(name, 0)) for name in charged_fields
        },
        "charged": charged,
        "uncharged": {
            name: max(int(planned_budget.get(name, 0)) - charged[name], 0)
            for name in charged_fields[1:]
        },
        "usage_after": {
            name: int(usage_after.get(name, 0)) for name in charged_fields
        },
        "overages": overages,
    }


def evaluation_unit_charge(
    *,
    seeds: tuple[int, ...],
    instance_ids: tuple[str, ...],
    started: bool,
) -> int:
    """Return the deterministic charge for one requested evaluation."""
    if not started:
        return 0
    return len(seeds) * len(instance_ids)


def _candidate_quality_gate(
    state: SearchState,
    run_dir: Path,
    config: PlannerEngineConfig,
) -> dict | None:
    """Stop only when a frozen final threshold is mathematically unreachable."""
    proposal_nodes = [node for node in state.nodes if node.round_index >= 0]
    if any(getattr(node, "status", None) == "pending" for node in proposal_nodes):
        raise RuntimeError("candidate quality gate requires terminal proposal slots")
    used = len(proposal_nodes)
    remaining = config.resolved_candidate_budget - used
    if remaining < 0:
        raise RuntimeError("proposal-slot budget is overdrawn")

    valid = [node for node in proposal_nodes if node.valid is True]
    counts = {
        "valid_candidates": len(valid),
        "exact_unique_candidates": len(
            {node.solver_sha256 for node in valid if node.solver_sha256}
        ),
    }
    if config.min_structural_unique_candidates is not None:
        counts["structural_unique_candidates"] = len(
            {
                source_structure_fingerprint(run_dir / node.solver_relpath)
                for node in valid
            }
        )
    else:
        counts["structural_unique_candidates"] = 0

    thresholds = {
        "valid_candidates": config.min_valid_candidates,
        "exact_unique_candidates": config.min_exact_unique_candidates,
        "structural_unique_candidates": config.min_structural_unique_candidates,
    }
    maximum_reachable = {
        name: count + remaining for name, count in counts.items()
    }
    unreachable = [
        name
        for name, threshold in thresholds.items()
        if threshold is not None and maximum_reachable[name] < threshold
    ]
    if config.gate_mode == "observe_only" or not unreachable:
        return None
    return {
        "status": "blocked",
        "category": "quality",
        "reason": "candidate_quality_threshold_unreachable",
        "reason_kind": "quality_threshold_unreachable",
        "proposal_slots_used": used,
        "proposal_slots_remaining": remaining,
        "counts": counts,
        "maximum_reachable": maximum_reachable,
        "thresholds": thresholds,
        "unreachable_metrics": unreachable,
    }


def _archive_exclusions(state: SearchState) -> set[str]:
    return {BASELINE_CONTROL_ID} | {
        node.node_id
        for node in state.nodes
        if node.action == SearchAction.ABLATE.value
    }


def _primary_training_records(evaluations, config: PlannerEngineConfig) -> list:
    return [
        record
        for record in evaluations
        if record.split == "train" and record.seeds == config.base_seeds
    ]


def _planned_budget(
    plan: RoundPlan,
    adapter: ProblemAdapter,
    config: PlannerEngineConfig,
) -> dict[str, int]:
    candidates = 0
    candidate_units = 0
    diagnostic_units = 0
    model_calls = 1
    if plan.is_stop:
        return {
            "candidates": 0,
            "candidate_evaluation_units": 0,
            "diagnostic_evaluation_units": 0,
            "evaluation_units": 0,
            "model_calls": model_calls,
        }
    for decision in plan.decisions:
        if decision.action == SearchAction.RETEST:
            diagnostic_units += decision.cost.diagnostic_evaluation_units
            continue
        candidates += 1
        model_calls += decision.cost.model_calls
        candidate_units += decision.cost.candidate_evaluation_units
    return {
        "candidates": candidates,
        "candidate_evaluation_units": candidate_units,
        "diagnostic_evaluation_units": diagnostic_units,
        "evaluation_units": candidate_units + diagnostic_units,
        "model_calls": model_calls,
    }


def _model_call_attempts(run_dir: Path) -> int:
    ledger_path = run_dir / ATTEMPT_LEDGER_FILENAME
    if ledger_path.is_file():
        return AttemptLedger(run_dir).summary().used_model_calls
    controller = len(list((run_dir / "rounds").glob("round_*/controller_attempt.json")))
    return controller + sum(
        _node_model_call_attempts(node_root)
        for node_root in (run_dir / "nodes").glob("node_*")
    )


def _model_token_usage(run_dir: Path) -> int:
    ledger_path = run_dir / ATTEMPT_LEDGER_FILENAME
    if ledger_path.is_file():
        return AttemptLedger(run_dir).summary().used_model_tokens
    return sum(
        _usage_token_count(_read_usage(path))
        for path in _model_usage_records(run_dir)
    )


def _model_usage_records(run_dir: Path) -> list[Path]:
    records = list((run_dir / "rounds").glob("round_*/controller_attempt.json"))
    for node_root in (run_dir / "nodes").glob("node_*"):
        hypothesis = node_root / "hypothesis_response.json"
        implementation = node_root / "agent_response.json"
        records.extend(path for path in (hypothesis, implementation) if path.is_file())
        error = node_root / "generation_error.json"
        if not error.is_file():
            continue
        data = json.loads(error.read_text(encoding="utf-8"))
        stage = data.get("stage")
        if (
            (stage == "hypothesis" and not hypothesis.is_file())
            or (stage == "implementation" and not implementation.is_file())
            or (stage is None and not implementation.is_file())
        ):
            records.append(error)
    return records


def _model_token_usage_complete(run_dir: Path) -> bool:
    ledger_path = run_dir / ATTEMPT_LEDGER_FILENAME
    if ledger_path.is_file():
        terminal_records = [
            record["payload"]
            for record in AttemptLedger(run_dir).records()
            if record["event_type"] == "attempt_terminal"
        ]
        return bool(terminal_records) and all(
            record.get("usage_status") in {"reported", "reported_zero"}
            for record in terminal_records
        )
    records = _model_usage_records(run_dir)
    return bool(records) and all(
        _usage_token_count(_read_usage(path)) > 0 for path in records
    )


def _model_usage_fingerprint(run_dir: Path) -> str:
    ledger_path = run_dir / ATTEMPT_LEDGER_FILENAME
    if ledger_path.is_file():
        return _file_sha256(ledger_path)
    return _text_sha256(
        json.dumps(
            {
                path.relative_to(run_dir).as_posix(): _file_sha256(path)
                for path in sorted(_model_usage_records(run_dir))
            },
            sort_keys=True,
        )
    )


def _read_usage(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    usage = data.get("usage")
    return usage if isinstance(usage, dict) else {}


def _usage_token_count(usage: dict) -> int:
    for key in ("total_tokens", "total_token_count"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    for first, second in (
        ("input_tokens", "output_tokens"),
        ("prompt_tokens", "completion_tokens"),
    ):
        values = usage.get(first), usage.get(second)
        if all(isinstance(value, (int, float)) for value in values):
            return max(0, int(values[0])) + max(0, int(values[1]))
    return 0


def _model_call_limit(config: PlannerEngineConfig) -> int:
    if config.protocol_mode == "v4":
        return config.resolved_protocol_model_call_budget
    return config.resolved_model_call_budget


def _reserved_model_calls(config: PlannerEngineConfig, logical_calls: int) -> int:
    return logical_calls * 3 if config.protocol_mode == "v4" else logical_calls


def _wall_clock_elapsed(run_dir: Path) -> float:
    path = run_dir / RESOURCE_CLOCK_FILENAME
    if not path.is_file():
        return 0.0
    started = float(json.loads(path.read_text(encoding="utf-8"))["started_at_epoch_sec"])
    return max(0.0, time.time() - started)


def _start_resource_clock(run_dir: Path) -> None:
    path = run_dir / RESOURCE_CLOCK_FILENAME
    if path.is_file():
        return
    _write_json(path, {"started_at_epoch_sec": time.time()})


def _resource_budget_gate(
    run_dir: Path,
    config: PlannerEngineConfig,
) -> dict | None:
    if config.model_token_budget is not None and not _model_token_usage_complete(run_dir):
        return {
            "status": "blocked",
            "reason": "model_token_usage_missing",
            "used": _model_token_usage(run_dir),
            "limit": config.model_token_budget,
        }
    checks = (
        (
            "model_call_budget_exceeded",
            _model_call_attempts(run_dir),
            _model_call_limit(config),
        ),
        (
            "model_token_budget_exceeded",
            _model_token_usage(run_dir),
            config.model_token_budget,
        ),
        (
            "wall_clock_budget_exceeded",
            _wall_clock_elapsed(run_dir),
            config.wall_clock_budget_sec,
        ),
    )
    for reason, used, limit in checks:
        if limit is not None and used > limit:
            return {
                "status": "blocked",
                "reason": reason,
                "used": used,
                "limit": limit,
            }
    return None


def _node_model_call_attempts(node_root: Path) -> int:
    hypothesis = node_root / "hypothesis_response.json"
    implementation = node_root / "agent_response.json"
    attempts = int(hypothesis.is_file()) + int(implementation.is_file())
    error_path = node_root / "generation_error.json"
    if not error_path.is_file():
        return attempts
    stage = json.loads(error_path.read_text(encoding="utf-8")).get("stage")
    if stage == "hypothesis" and not hypothesis.is_file():
        attempts += 1
    elif stage == "implementation" and not implementation.is_file():
        attempts += 1
    return attempts


def _evidence_candidate_ids(evidence: list[EvidenceRecord]) -> set[str]:
    return {
        item.candidate_id
        for item in evidence
        if item.status in {"pending", "supported"}
        and item.candidate_id != BASELINE_CONTROL_ID
    }


def _baseline_observations(state: SearchState) -> dict:
    record = next(
        (
            item
            for item in reversed(state.evaluations)
            if item.candidate_id == BASELINE_CONTROL_ID
            and item.split == "train"
            and item.valid
        ),
        None,
    )
    if record is None:
        return {}
    from agent_ahd.core.research_records import Observation

    return {
        "baseline:profile": Observation(
            observation_id="baseline:profile",
            kind="baseline_control",
            statement=(
                f"The locked baseline has mean training {record.metric_name} {record.mean_score:.12g} ({record.metric_unit}) "
                f"under seeds {list(record.seeds)}."
            ),
            values={
                "candidate_id": BASELINE_CONTROL_ID,
                "mean_score": record.mean_score,
                "metric_name": record.metric_name, "metric_unit": record.metric_unit,
                **({"mean_gap": record.mean_score} if record.metric_name == "optimality_gap" else {}),
                "seeds": list(record.seeds),
            },
        )
    }


def _diagnostic_evaluation_observations(
    state: SearchState,
    *,
    candidate_ids: set[str],
    adapter: ProblemAdapter,
    config: PlannerEngineConfig,
) -> dict:
    from agent_ahd.core.research_records import Observation

    canonical = {
        "train": _canonical_instance_ids(
            state,
            adapter,
            config,
            split="train",
        )
    }
    observations = {}
    for record in state.evaluations:
        if record.candidate_id not in candidate_ids or record.split != "train":
            continue
        represented_by_archive = (
            record.valid
            and record.instance_ids == canonical.get(record.split)
            and record.seeds == config.base_seeds
        )
        if represented_by_archive:
            continue
        observation_id = f"evaluation:{record.evaluation_id}"
        if record.valid:
            per_instance = {
                instance_id: float(
                    sum(row[column] for row in record.score_matrix)
                    / len(record.score_matrix)
                )
                for column, instance_id in enumerate(record.instance_ids)
            }
            statement = (
                f"{record.candidate_id} was retested on {record.split} with seeds "
                f"{list(record.seeds)} and mean {record.metric_name} {record.mean_score:.12g} ({record.metric_unit})."
            )
            values = {
                "candidate_id": record.candidate_id,
                "split": record.split,
                "seeds": list(record.seeds),
                "per_instance_gap" if record.metric_name == "optimality_gap" else "per_instance_score": per_instance,
                "mean_score": record.mean_score,
                "metric_name": record.metric_name, "metric_unit": record.metric_unit,
                **({"mean_gap": record.mean_score} if record.metric_name == "optimality_gap" else {}),
                "runtime_sec": record.total_runtime_sec,
            }
        else:
            statement = (
                f"Evaluation {record.evaluation_id} for {record.candidate_id} failed "
                f"with {record.error_type}: {record.error_message}."
            )
            values = {
                "candidate_id": record.candidate_id,
                "split": record.split,
                "error_type": record.error_type,
                "error_message": record.error_message,
            }
        observations[observation_id] = Observation(
            observation_id=observation_id,
            kind="evaluation_diagnostic",
            statement=statement,
            values=values,
        )
    return observations


def _completed_retests(
    state: SearchState,
    adapter: ProblemAdapter,
    config: PlannerEngineConfig,
) -> set[tuple]:
    completed = set()
    manifests = {
        "train": _canonical_instance_ids(
            state,
            adapter,
            config,
            split="train",
        )
    }
    for record in state.evaluations:
        if not record.valid or record.split != "train":
            continue
        key = (
            record.candidate_id,
            record.split,
            tuple(sorted(record.seeds)),
            tuple(sorted(record.instance_ids)),
        )
        completed.add(key)
        full_ids = manifests.get(record.split, ())
        if (
            len(record.instance_ids) == len(full_ids)
            and set(record.instance_ids) == set(full_ids)
        ):
            completed.add(
                (
                    record.candidate_id,
                    record.split,
                    tuple(sorted(record.seeds)),
                    (),
                )
            )
    return completed


def _canonical_instance_ids(
    state: SearchState,
    adapter: ProblemAdapter,
    config: PlannerEngineConfig,
    *,
    split: str,
) -> tuple[str, ...]:
    if split == "train":
        baseline = next(
            (
                record
                for record in reversed(state.evaluations)
                if record.candidate_id == BASELINE_CONTROL_ID
                and record.split == "train"
                and record.valid
                and record.seeds == config.base_seeds
            ),
            None,
        )
        if baseline is not None:
            return baseline.instance_ids
        manifest = adapter.train_manifest
    elif split == "validation":
        manifest = adapter.validation_manifest
    else:
        raise ValueError(f"unsupported archive split: {split}")
    return manifest.instance_ids[: config.n_instances]


def _node_result(node: SolverNode) -> dict:
    return {
        "node_id": node.node_id,
        "score": node.score,
        "metric_name": node.metric_name,
        "metric_unit": node.metric_unit,
        **({"gap_unit": node.gap_unit, "instance_gaps": node.instance_gaps}
           if node.metric_name == "optimality_gap" else {"instance_scores": node.instance_scores}),
        "runtime_sec": node.runtime_sec,
        "status": node.status,
    }


def _copy_python_sources(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=False)
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        destination = target_root / path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _discard_generated_tree(
    path: Path,
    *,
    parent: Path,
    name_prefix: str,
) -> None:
    candidate = path.resolve()
    expected_parent = parent.resolve()
    if candidate.parent != expected_parent or not candidate.name.startswith(name_prefix):
        raise RuntimeError(f"refusing to remove unexpected generated directory: {path}")
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"generated directory has an unsafe type: {path}")
    shutil.rmtree(path)


def _discard_generated_file(
    path: Path,
    *,
    parent: Path,
    name_prefix: str,
) -> None:
    candidate = path.resolve()
    expected_parent = parent.resolve()
    if candidate.parent != expected_parent or not candidate.name.startswith(name_prefix):
        raise RuntimeError(f"refusing to remove unexpected generated file: {path}")
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"generated file has an unsafe type: {path}")
    path.unlink()


def _discard_partial_candidate_preparation(
    *,
    nodes_root: Path,
    node_id: str,
    snapshot_root: Path,
    task_path: Path,
) -> None:
    for path in nodes_root.glob(f".prepare-{node_id}-*"):
        _discard_generated_tree(
            path,
            parent=nodes_root,
            name_prefix=f".prepare-{node_id}-",
        )
    if snapshot_root.parent.exists():
        for path in snapshot_root.parent.glob(f".prepare-{node_id}-*"):
            _discard_generated_tree(
                path,
                parent=snapshot_root.parent,
                name_prefix=f".prepare-{node_id}-",
            )
    if task_path.parent.exists():
        for path in task_path.parent.glob(f".{node_id}.*.tmp"):
            _discard_generated_file(
                path,
                parent=task_path.parent,
                name_prefix=f".{node_id}.",
            )
    if snapshot_root.exists():
        if snapshot_root.name != node_id:
            raise RuntimeError("candidate snapshot recovery target changed")
        _discard_generated_tree(
            snapshot_root,
            parent=snapshot_root.parent,
            name_prefix=node_id,
        )
    if task_path.exists():
        if task_path.name != f"{node_id}.md":
            raise RuntimeError("candidate task recovery target changed")
        task_path.unlink()


def _write_candidate_artifacts(
    workspace: CandidateWorkspace,
    artifact: CandidateArtifact,
) -> None:
    (workspace.root / "candidate.diff").write_text(artifact.diff, encoding="utf-8")
    _write_json(
        workspace.root / "candidate_manifest.json",
        {
            "candidate_id": artifact.candidate_id,
            "baseline_sha256": artifact.baseline_sha256,
            "comparison_sha256": artifact.comparison_sha256,
            "comparison_label": artifact.comparison_label,
            "source_sha256": artifact.source_sha256,
            "file_sha256": artifact.file_sha256,
            "changed": artifact.changed,
        },
    )


def _snapshot_record(snapshot: ArchiveSnapshot) -> dict:
    return {
        "profiles": {
            candidate_id: asdict(profile)
            for candidate_id, profile in snapshot.profiles.items()
        },
        "mean_elites": list(snapshot.mean_elites),
        "cvar_elites": list(snapshot.cvar_elites),
        "worst_elites": list(snapshot.worst_elites),
        "structural_elites": list(snapshot.structural_elites),
        "candidate_clusters": {
            key: list(value) for key, value in snapshot.candidate_clusters.items()
        },
        "failure_clusters": {
            key: asdict(value) for key, value in snapshot.failure_clusters.items()
        },
        "observations": {
            key: asdict(value) for key, value in snapshot.observations.items()
        },
    }


def _structural_signatures(state: SearchState, run_dir: Path) -> dict[str, str]:
    signatures = {}
    for node in state.nodes:
        if node.valid is not True:
            continue
        source_path = run_dir / node.solver_relpath
        if source_path.is_file():
            signatures[node.node_id] = source_structure_fingerprint(source_path)
    return signatures


def _controller_snapshot(
    snapshot: ArchiveSnapshot,
    config: PlannerEngineConfig,
) -> ArchiveSnapshot:
    if config.planner_variant != "without_risk_archive":
        return snapshot
    return restrict_archive_snapshot_to_mean(snapshot)


def _controller_evidence(
    evidence: list[EvidenceRecord],
    snapshot: ArchiveSnapshot,
    config: PlannerEngineConfig,
) -> list[EvidenceRecord]:
    if config.planner_variant == "without_evidence":
        return []
    if config.planner_variant != "without_risk_archive":
        return evidence
    visible = snapshot.visible_candidate_ids
    return [item for item in evidence if item.candidate_id in visible]


def _planner_actions(config: PlannerEngineConfig) -> set[SearchAction]:
    actions = set(SearchAction)
    if config.planner_variant == "without_evidence":
        actions -= {SearchAction.ABLATE, SearchAction.CROSS}
    elif config.planner_variant == "without_controlled_ablation":
        actions.remove(SearchAction.ABLATE)
    return actions


def _apply_framework_off_contract(context: dict) -> dict:
    """Expose scalar fitness and real parent/action choices, but no ON mechanisms."""
    catalog = DecisionCatalog.from_dict(context["decision_catalog"])
    visible_profiles = context.get("visible_profiles", {}).get("train", {})
    observations = context.get("observations", {})
    restricted_observations = {
        observation_id: json.loads(json.dumps(record))
        for observation_id, record in sorted(observations.items())
        if record.get("kind") in {"problem", "baseline_control"}
    }
    for candidate_id, profile in sorted(visible_profiles.items()):
        mean_gap = profile.get("mean_gap")
        if isinstance(mean_gap, bool) or not isinstance(mean_gap, (int, float)):
            raise RuntimeError(
                f"framework off mode requires scalar mean gap for {candidate_id}"
            )
        restricted_observations[f"scalar:{candidate_id}:mean_gap"] = {
            "kind": "candidate_profile",
            "statement": (
                f"Candidate {candidate_id} has scalar mean training gap "
                f"{float(mean_gap):.12g}."
            ),
            "values": {
                "candidate_id": candidate_id,
                "mean_gap": float(mean_gap),
            },
        }

    baseline_context = tuple(sorted(
        observation_id
        for observation_id, record in restricted_observations.items()
        if record.get("kind") in {"problem", "baseline_control"}
    ))
    options = []
    for option in catalog.candidate_options:
        decision = option.decision
        if decision.action == SearchAction.DIVERSIFY:
            options.append(replace(
                option,
                decision=replace(
                    decision,
                    observation_ids=baseline_context,
                    generation_context_ids=baseline_context,
                    scope=ScopeSpec("full_solver"),
                ),
            ))
            continue
        if decision.action != SearchAction.EXPLOIT:
            continue
        candidate_id = decision.primary_parent_id
        if candidate_id not in visible_profiles:
            continue
        if decision.scope != ScopeSpec("full_solver"):
            continue
        scalar_id = f"scalar:{candidate_id}:mean_gap"
        options.append(replace(
            option,
            decision=replace(
                decision,
                observation_ids=(scalar_id,),
                generation_context_ids=(scalar_id,),
            ),
        ))
    if not options:
        raise RuntimeError("framework off contract produced no candidate options")
    actions = tuple(
        action
        for action in (SearchAction.EXPLOIT, SearchAction.DIVERSIFY)
        if any(option.decision.action == action for option in options)
    )
    restricted_catalog = DecisionCatalog(
        round_index=catalog.round_index,
        available_actions=actions,
        candidate_options=tuple(options),
        diagnostic_options=(),
    )
    restricted = dict(context)
    restricted.update({
        "archives": {},
        "search_progress": {},
        "visible_profiles": {
            "train": {
                candidate_id: {"mean_gap": float(profile["mean_gap"])}
                for candidate_id, profile in sorted(visible_profiles.items())
            },
            "validation": {},
        },
        "candidate_clusters": {},
        "candidate_scopes": {
            candidate_id: {"kind": "full_solver", "symbols": []}
            for candidate_id in sorted(visible_profiles)
        },
        "scope_boundary_evidence": [],
        "failure_clusters": {},
        "cluster_contracts": {},
        "observations": restricted_observations,
        "retest_options": {"diagnostic_seeds": [], "split_instance_ids": {}},
        "actions": [action.value for action in actions],
        "action_contracts": {
            "EXPLOIT": "Choose a parent using only its scalar mean gap.",
            "DIVERSIFY": "Create an independent design from the locked baseline.",
        },
        "generation_context_contracts": {
            key: value
            for key, value in context.get("generation_context_contracts", {}).items()
            if key in {"EXPLOIT", "DIVERSIFY"}
        },
        "generation_context_link_contracts": {
            key: value
            for key, value in context.get(
                "generation_context_link_contracts", {}
            ).items()
            if key in {"EXPLOIT", "DIVERSIFY"}
        },
        "target_cluster_contracts": {
            key: value
            for key, value in context.get("target_cluster_contracts", {}).items()
            if key in {"EXPLOIT", "DIVERSIFY"}
        },
        "decision_catalog": restricted_catalog.to_dict(),
        "output_schema": controller_selection_schema(
            restricted_catalog,
            candidate_count=int(context["maximum_candidate_decisions"]),
        ),
        "framework_ablation_contract": {
            "mode": "off",
            "controller_information": "scalar_mean_gap_only",
            "allowed_actions": [action.value for action in actions],
            "dynamic_scope": False,
            "risk_archive": False,
            "mechanism_evidence": False,
        },
    })
    return restricted


def _inheritance_experiment_visible_ids(
    visible_candidate_ids: set[str],
    config: PlannerEngineConfig,
) -> set[str]:
    visible = set(visible_candidate_ids)
    if config.inheritance_experiment_arm is None:
        return visible
    if config.inheritance_donor_id is not None:
        visible.add(config.inheritance_donor_id)
    if (
        config.inheritance_experiment_arm in {"mechanism_only", "agent_choice"}
        and config.inheritance_scaffold_id is not None
    ):
        visible.add(config.inheritance_scaffold_id)
    return visible


def _apply_inheritance_experiment_contract(
    context: dict,
    config: PlannerEngineConfig,
) -> dict:
    """Restrict controller choices so inheritance is the only arm-level variable."""
    arm = config.inheritance_experiment_arm
    if arm is None:
        return context

    donor_id = config.inheritance_donor_id
    scaffold_id = config.inheritance_scaffold_id
    catalog = DecisionCatalog.from_dict(context["decision_catalog"])
    def allowed(option) -> bool:
        decision = option.decision
        local = (
            decision.inheritance_channel == InheritanceChannel.LOCAL_SOURCE
            and decision.primary_parent_id == donor_id
        )
        mechanism = (
            decision.inheritance_channel == InheritanceChannel.MECHANISM_ONLY
            and decision.primary_parent_id == scaffold_id
            and decision.secondary_parent_id == donor_id
        )
        full = (
            decision.inheritance_channel == InheritanceChannel.FULL_SOURCE
            and decision.primary_parent_id == donor_id
        )
        if arm == "full_source":
            return full
        if arm == "local_source":
            return local
        if arm == "mechanism_only":
            return mechanism
        return local or mechanism

    candidate_options = tuple(
        option for option in catalog.candidate_options if allowed(option)
    )
    if not candidate_options:
        raise RuntimeError(
            "inheritance experiment contract produced no candidate options: "
            f"arm={arm}, donor={donor_id}, scaffold={scaffold_id}"
        )
    actions = tuple(dict.fromkeys(option.decision.action for option in candidate_options))
    filtered = DecisionCatalog(
        round_index=catalog.round_index,
        available_actions=actions,
        candidate_options=candidate_options,
        diagnostic_options=(),
    )
    schema = controller_selection_schema(
        filtered,
        candidate_count=config.batch_size,
    )
    selection_properties = schema["properties"]
    selection_properties["candidate_option_ids"]["minItems"] = config.batch_size
    selection_properties["candidate_option_ids"]["maxItems"] = config.batch_size
    selection_properties["diagnostic_option_ids"]["maxItems"] = 0
    selection_properties["stop"] = {"type": "boolean", "const": False}
    selection_properties["stop_reason"] = {"type": "null"}

    restricted = dict(context)
    restricted["actions"] = [action.value for action in actions]
    restricted["decision_catalog"] = filtered.to_dict()
    restricted["output_schema"] = schema
    restricted["inheritance_experiment_contract"] = {
        "arm": arm,
        "donor_candidate_id": donor_id,
        "scaffold_candidate_id": scaffold_id,
        "required_candidate_count": config.batch_size,
        "allowed_inheritance_channels": sorted(
            {
                option.decision.inheritance_channel.value
                for option in candidate_options
            }
        ),
    }
    return restricted


def _disable_controller_stop(context: dict) -> dict:
    contracted = dict(context)
    schema = json.loads(json.dumps(context["output_schema"]))
    schema["properties"]["stop"] = {"type": "boolean", "const": False}
    schema["properties"]["stop_reason"] = {"type": "null"}
    contracted["output_schema"] = schema
    return contracted


def _apply_fixed_round_controller_contract(
    context: dict,
    *,
    batch_size: int,
) -> dict:
    contracted = _disable_controller_stop(context)
    schema = json.loads(json.dumps(contracted["output_schema"]))
    candidate_ids = schema["properties"]["candidate_option_ids"]
    candidate_ids["minItems"] = batch_size
    candidate_ids["maxItems"] = batch_size
    schema["properties"]["diagnostic_option_ids"]["maxItems"] = 0
    contracted["output_schema"] = schema
    return contracted


def _apply_scope_experiment_contract(
    context: dict,
    config: PlannerEngineConfig,
) -> dict:
    """Restrict choices so modification-scope policy is the only arm variable."""
    arm = config.scope_experiment_arm
    if arm is None:
        return context

    initial_scope = ScopeSpec(
        "symbol_set",
        config.scope_experiment_initial_symbols,
    )
    catalog = DecisionCatalog.from_dict(context["decision_catalog"])
    candidate_scopes = {
        candidate_id: ScopeSpec.from_dict(scope)
        for candidate_id, scope in context.get("candidate_scopes", {}).items()
    }
    preserving_actions = {SearchAction.EXPLOIT, SearchAction.REPAIR}
    dynamic_actions = preserving_actions | {
        SearchAction.EXPAND_SCOPE,
        SearchAction.NARROW_SCOPE,
    }

    def allowed(option) -> bool:
        decision = option.decision
        parent_id = decision.primary_parent_id
        if parent_id in {None, BASELINE_CONTROL_ID}:
            return False
        parent_scope = candidate_scopes.get(parent_id)
        if arm == "S1_fixed_single":
            return (
                decision.action in preserving_actions
                and decision.scope == initial_scope
                and parent_scope == initial_scope
            )
        if arm == "S2_fixed_full":
            return (
                decision.action in preserving_actions
                and decision.scope == ScopeSpec("full_solver")
                and parent_scope == ScopeSpec("full_solver")
            )
        if parent_scope is None or decision.action not in dynamic_actions:
            return False
        if decision.action in preserving_actions:
            return decision.scope == parent_scope
        if decision.action == SearchAction.EXPAND_SCOPE:
            if parent_scope.kind != "symbol_set":
                return False
            return (
                decision.scope.kind == "full_solver"
                or (
                    decision.scope.kind == "symbol_set"
                    and set(decision.scope.symbols) > set(parent_scope.symbols)
                )
            )
        if decision.scope.kind != "symbol_set":
            return False
        return (
            parent_scope.kind == "full_solver"
            or set(decision.scope.symbols) < set(parent_scope.symbols)
        )

    candidate_options = tuple(
        option for option in catalog.candidate_options if allowed(option)
    )
    if not candidate_options:
        raise RuntimeError(
            "scope experiment contract produced no candidate options: "
            f"arm={arm}, parent={config.scope_experiment_parent_id}, "
            f"initial_symbols={config.scope_experiment_initial_symbols}"
        )
    actions = tuple(dict.fromkeys(
        option.decision.action for option in candidate_options
    ))
    filtered = DecisionCatalog(
        round_index=catalog.round_index,
        available_actions=actions,
        candidate_options=candidate_options,
        diagnostic_options=(),
    )
    schema = controller_selection_schema(
        filtered,
        candidate_count=config.batch_size,
    )
    properties = schema["properties"]
    properties["candidate_option_ids"]["minItems"] = config.batch_size
    properties["candidate_option_ids"]["maxItems"] = config.batch_size
    properties["diagnostic_option_ids"]["maxItems"] = 0
    properties["stop"] = {"type": "boolean", "const": False}
    properties["stop_reason"] = {"type": "null"}

    restricted = dict(context)
    restricted["actions"] = [action.value for action in actions]
    restricted["decision_catalog"] = filtered.to_dict()
    restricted["output_schema"] = schema
    restricted["scope_experiment_contract"] = {
        "arm": arm,
        "parent_candidate_id": config.scope_experiment_parent_id,
        "initial_symbols": list(config.scope_experiment_initial_symbols),
        "required_candidate_count": config.batch_size,
        "allowed_actions": [action.value for action in actions],
    }
    return restricted


def _cluster_gap_context(
    snapshot: ArchiveSnapshot,
    instance_ids: tuple[str, ...],
    parent_id: str | None,
) -> dict:
    metrics = {(p.metric_name, p.metric_unit) for p in snapshot.profiles.values()}
    if len(metrics) > 1:
        raise ValueError("cluster context cannot mix metric identities")
    metric_name, metric_unit = next(iter(metrics), ("optimality_gap", "percent"))
    suffix = "gap" if metric_name == "optimality_gap" else "score"
    profile_maps = {
        candidate_id: dict(zip(profile.instance_ids, profile.per_instance_mean))
        for candidate_id, profile in snapshot.profiles.items()
    }
    parent_map = profile_maps.get(parent_id, {})
    return {
        "metric_name": metric_name,
        "metric_unit": metric_unit,
        "parent_per_instance_"+suffix: {
            instance_id: parent_map[instance_id]
            for instance_id in instance_ids
            if instance_id in parent_map
        },
        "archive_best_per_instance_"+suffix: {
            instance_id: min(
                values[instance_id]
                for values in profile_maps.values()
                if instance_id in values
            )
            for instance_id in instance_ids
            if any(instance_id in values for values in profile_maps.values())
        },
    }


def _manifest_record(manifest) -> dict:
    source_hashes = {}
    if manifest.source_path is not None and manifest.source_path.is_dir():
        source_hashes = {
            instance_id: _file_sha256(manifest.source_path / instance_id)
            for instance_id in manifest.instance_ids
        }
    return {
        "split": manifest.split,
        "source_sha256": (
            _file_sha256(manifest.source_path)
            if manifest.source_path is not None and manifest.source_path.is_file()
            else None
        ),
        "source_hashes": source_hashes,
        "instance_ids": list(manifest.instance_ids),
        "family_by_instance": manifest.family_by_instance,
        "exposed_to_search": manifest.exposed_to_search,
    }


def _search_dataset_protocol(
    protocol: dict,
    *,
    data_root: str | Path | None = None,
) -> dict:
    exposed = json.loads(json.dumps(protocol))
    resolved_data_root = (
        Path(data_root).resolve()
        if data_root is not None
        else Path(__file__).resolve().parents[2]
    )
    exposed["data_root"] = str(resolved_data_root)
    for split in ("train", "validation"):
        if exposed[split]["source_path"] is None:
            if split == "validation" and exposed.get("training_only") and not exposed[split]["instance_ids"]:
                continue
            raise ValueError("missing dataset source path")
        value = Path(str(exposed[split]["source_path"]))
        resolved = value.resolve() if value.is_absolute() else (resolved_data_root / value).resolve()
        if resolved.is_relative_to(resolved_data_root):
            exposed[split]["source_path"] = resolved.relative_to(resolved_data_root).as_posix()
    blind = exposed.get("blind")
    if not isinstance(blind, dict):
        raise ValueError("dataset protocol requires blind commitment metadata")
    commitment = blind.get("commitment")
    if not isinstance(commitment, str) or not commitment.strip():
        raise ValueError("new runs require an opaque blind commitment")
    exposed["blind"] = {"commitment": commitment}
    return exposed


def _protocol_artifact_hashes(adapter: ProblemAdapter) -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    paths = []
    for package_name in ("agents", "core", "evaluators", "problems"):
        paths.extend(
            (package_root / package_name).rglob("*.py")
        )
    paths.extend([
        package_root / "problems" / "base.py",
        *sorted((package_root / "experiments").glob("*.py")),
        package_root / "requirements.runtime.lock",
        adapter.baseline_path,
        adapter.spec.root_dir / "adapter.py",
        adapter.spec.root_dir / "get_instance.py",
    ])
    if adapter.train_manifest.source_path is not None:
        paths.append(adapter.train_manifest.source_path)
    if (
        adapter.validation_manifest.source_path is not None
        and adapter.validation_manifest.source_path.is_dir()
    ):
        paths.extend(
            adapter.validation_manifest.source_path / instance_id
            for instance_id in adapter.validation_manifest.instance_ids
        )
    elif adapter.validation_manifest.source_path is not None:
        paths.append(adapter.validation_manifest.source_path)
    hashes = {}
    for path in sorted({item.resolve() for item in paths if item.is_file()}):
        try:
            name = "agent_ahd/" + path.relative_to(package_root).as_posix()
        except ValueError:
            name = str(path)
        hashes[name] = _file_sha256(path)
    return hashes


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in GENERATED_ARTIFACT_NAMES
        and "__pycache__" not in path.parts
    }


def _source_tree_contains_symlink(root: Path) -> bool:
    return root.is_symlink() or any(path.is_symlink() for path in root.rglob("*"))


def _verify_node_sources(run_dir: Path, state: SearchState) -> None:
    nodes_root = (run_dir / "nodes").resolve()
    for node in state.nodes:
        expected = node.solver_file_sha256
        if expected is None:
            if node.valid is True:
                raise RuntimeError(
                    f"valid candidate {node.node_id} has no locked source hash"
                )
            continue
        workspace_root = (run_dir / node.solver_relpath).resolve().parent
        if not workspace_root.is_relative_to(nodes_root):
            raise RuntimeError(f"candidate path escapes run directory: {node.node_id}")
        if not workspace_root.is_dir():
            raise RuntimeError(f"candidate source is missing: {node.node_id}")
        if _candidate_file_hashes(workspace_root) != expected:
            raise RuntimeError(f"candidate source changed: {node.node_id}")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_agent_response_record(
    path: Path,
    *,
    model: str,
    logical_call_id: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: dict,
) -> AgentResponse:
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model": model,
        "logical_call_id": logical_call_id,
        "system_prompt_sha256": _text_sha256(system_prompt),
        "user_prompt_sha256": _text_sha256(user_prompt),
        "output_schema_sha256": _text_sha256(
            json.dumps(output_schema, sort_keys=True)
        ),
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise RuntimeError(f"locked agent response identity changed: {path}")
    payload = record.get("payload")
    raw_content = record.get("raw_content")
    usage = record.get("usage")
    if (
        not isinstance(payload, dict)
        or not isinstance(raw_content, str)
        or not isinstance(usage, dict)
    ):
        raise RuntimeError(f"locked agent response is incomplete: {path}")
    return AgentResponse(payload, raw_content, usage)


def _agent_attempt_identity(
    *,
    model: str,
    logical_call_id: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: dict,
) -> dict[str, object]:
    return {
        "model": model,
        "logical_call_id": logical_call_id,
        "system_prompt_sha256": _text_sha256(system_prompt),
        "user_prompt_sha256": _text_sha256(user_prompt),
        "output_schema_sha256": _text_sha256(
            json.dumps(output_schema, sort_keys=True)
        ),
    }


def _read_agent_attempt(path: Path, identity: dict[str, object]) -> dict | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if set(record) != {
        "error_message",
        "error_type",
        "failure_stage",
        "logical_call_id",
        "model",
        "output_schema_sha256",
        "response_sha256",
        "status",
        "system_prompt_sha256",
        "user_prompt_sha256",
        "version",
    }:
        raise RuntimeError(f"agent attempt fields are invalid: {path}")
    if record.get("version") != 1 or any(
        record.get(field) != value for field, value in identity.items()
    ):
        raise RuntimeError(f"agent attempt identity changed: {path}")
    return record


def _begin_agent_attempt(
    path: Path,
    *,
    identity: dict[str, object],
    allow_recovery: bool,
    role: str,
) -> None:
    existing = _read_agent_attempt(path, identity)
    if existing is not None:
        status = existing.get("status")
        if status == "started" and not allow_recovery:
            raise TransactionAmbiguityError(
                f"ambiguous {role} attempt; direct backend replay is forbidden"
            )
        if status == "failed":
            failure_stage = existing.get("failure_stage")
            if failure_stage == "orphan":
                raise TransactionAmbiguityError(
                    f"{role} attempt previously failed with an orphaned terminal"
                )
            if not (allow_recovery and failure_stage == "network"):
                raise RuntimeError(f"{role} attempt previously failed")
        if status not in {"started", "failed"}:
            raise RuntimeError(f"{role} attempt has no durable response")
    _write_json(path, {
        "version": 1,
        "status": "started",
        **identity,
        "response_sha256": None,
        "failure_stage": None,
        "error_type": None,
        "error_message": None,
    })


def _finish_agent_attempt(
    path: Path,
    *,
    identity: dict[str, object],
    response_path: Path,
) -> None:
    if not response_path.is_file():
        raise RuntimeError("agent attempt cannot finish without a durable response")
    existing = _read_agent_attempt(path, identity)
    response_sha256 = _file_sha256(response_path)
    if existing is not None and existing.get("status") == "returned":
        if existing.get("response_sha256") != response_sha256:
            raise RuntimeError("agent attempt response changed")
        return
    try:
        _write_json(path, {
            "version": 1,
            "status": "returned",
            **identity,
            "response_sha256": response_sha256,
            "failure_stage": None,
            "error_type": None,
            "error_message": None,
        })
    except OSError as exc:
        raise TransactionAmbiguityError(
            "agent response is durable but attempt finalization failed"
        ) from exc


def _fail_agent_attempt(
    path: Path,
    *,
    identity: dict[str, object],
    exc: Exception,
) -> None:
    _read_agent_attempt(path, identity)
    _write_json(path, {
        "version": 1,
        "status": "failed",
        **identity,
        "response_sha256": None,
        "failure_stage": (
            exc.stage if isinstance(exc, AgentResponseError) else "request"
        ),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    })


def _verified_candidate_commit(
    control_workspace: CandidateWorkspace,
) -> tuple[dict, CandidateWorkspace] | None:
    commit_path = control_workspace.root / "candidate_commit.json"
    if not commit_path.is_file():
        return None
    response_path = control_workspace.root / "agent_response.json"
    if not response_path.is_file():
        raise RuntimeError("candidate commit has no durable agent response")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if set(commit) != {
        "candidate_id",
        "response_sha256",
        "source_files",
        "source_fingerprint",
        "source_root_relpath",
        "version",
    }:
        raise RuntimeError("candidate commit fields are invalid")
    if (
        commit["version"] != 1
        or commit["candidate_id"] != control_workspace.candidate_id
    ):
        raise RuntimeError("candidate commit identity changed")
    source_root_relpath = commit.get("source_root_relpath")
    if not isinstance(source_root_relpath, str) or not source_root_relpath:
        raise RuntimeError("candidate commit source pointer is invalid")
    versions_path = control_workspace.root / "versions"
    lexical_source_root = control_workspace.root / source_root_relpath
    if (
        control_workspace.root.is_symlink()
        or versions_path.is_symlink()
        or lexical_source_root.is_symlink()
    ):
        raise RuntimeError("candidate commit source pointer uses a symlink")
    source_root = lexical_source_root.resolve()
    versions_root = versions_path.resolve()
    if versions_root.parent != control_workspace.root.resolve():
        raise RuntimeError("candidate version storage escapes workspace")
    if source_root.parent != versions_root or not source_root.is_dir():
        raise RuntimeError("candidate commit source pointer escapes version storage")
    if _source_tree_contains_symlink(source_root):
        raise RuntimeError("candidate committed source contains a symlink")
    source_files = commit.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise RuntimeError("candidate commit source manifest is empty")
    actual_files = _candidate_file_hashes(source_root)
    if source_files != actual_files:
        raise RuntimeError("candidate committed source changed")
    if commit["response_sha256"] != _file_sha256(response_path):
        raise RuntimeError("candidate committed response changed")
    fingerprint = _text_sha256(json.dumps(source_files, sort_keys=True))
    if commit["source_fingerprint"] != fingerprint:
        raise RuntimeError("candidate commit source fingerprint changed")
    if source_root.name != fingerprint:
        raise RuntimeError("candidate commit source version name changed")
    workspace = CandidateWorkspace(
        candidate_id=control_workspace.candidate_id,
        root=source_root,
        solver_path=source_root / "solver.py",
    )
    if not workspace.solver_path.is_file():
        raise RuntimeError("candidate committed solver is missing")
    return commit, workspace


def _write_candidate_commit(
    control_workspace: CandidateWorkspace,
    *,
    source_root: Path,
    source_files: dict[str, str],
) -> dict:
    response_path = control_workspace.root / "agent_response.json"
    source = source_root.resolve()
    control = control_workspace.root.resolve()
    versions_path = control_workspace.root / "versions"
    if (
        control_workspace.root.is_symlink()
        or versions_path.is_symlink()
        or source_root.is_symlink()
    ):
        raise RuntimeError("candidate version storage uses a symlink")
    versions = versions_path.resolve()
    if versions.parent != control:
        raise RuntimeError("candidate version storage escapes workspace")
    if source.parent != versions:
        raise RuntimeError("candidate version is outside version storage")
    if _source_tree_contains_symlink(source):
        raise RuntimeError("candidate version contains a symlink")
    actual_files = _candidate_file_hashes(source)
    if source_files != actual_files:
        raise RuntimeError("candidate version source manifest changed")
    source_fingerprint = _text_sha256(json.dumps(source_files, sort_keys=True))
    if source.name != source_fingerprint:
        raise RuntimeError("candidate version directory is not content addressed")
    commit = {
        "version": 1,
        "candidate_id": control_workspace.candidate_id,
        "response_sha256": _file_sha256(response_path),
        "source_root_relpath": source.relative_to(control).as_posix(),
        "source_files": source_files,
        "source_fingerprint": source_fingerprint,
    }
    _write_idempotent_json(
        control / "candidate_commit.json",
        commit,
        label="candidate commit",
    )
    return commit


def _evaluation_task_batch_fingerprint(
    tasks: list[CandidateTask],
    *,
    execution_contract: dict[str, object] | None = None,
) -> str:
    records = []
    for task in tasks:
        records.append({
            "candidate_id": task.candidate_id,
            "candidate_path": str(task.candidate_path.resolve()),
            "candidate_sha256": _file_sha256(task.candidate_path),
            "evaluator_module": task.evaluator_module,
            "evaluator_function": task.evaluator_function,
            "budget": asdict(task.budget),
            "gap_unit": task.gap_unit,
            "metric_name": task.metric_name,
            "metric_unit": task.metric_unit,
            "contract_id": task.contract_id,
            "split": task.split,
            "instance_ids": list(task.instance_ids),
            "data_path": (
                str(task.data_path.resolve()) if task.data_path is not None else None
            ),
            "collect_diagnostics": task.collect_diagnostics,
        })
    payload = {
        "candidate_execution": (
            candidate_execution_contract()
            if execution_contract is None
            else execution_contract
        ),
        "tasks": records,
    }
    return _text_sha256(json.dumps(payload, sort_keys=True))


def _write_evaluation_receipt(
    path: Path,
    *,
    task_fingerprint: str,
    results: list[EvaluationResult],
) -> None:
    _write_json(path, {
        "version": 1,
        "task_fingerprint": task_fingerprint,
        "results": [result.to_record() for result in results],
    })


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_idempotent_json(path: Path, data: object, *, label: str) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != data:
            raise RuntimeError(f"{label} record changed during resume")
        return
    _write_json(path, data)
