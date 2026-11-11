from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, median, pstdev

import numpy as np

from agent_ahd.core.research_records import EvaluationRecord, Observation


@dataclass(frozen=True)
class BehaviorProfile:
    candidate_id: str
    seeds: tuple[int, ...]
    seed_mean_scores: tuple[float, ...]
    seed_std_score: float
    instance_ids: tuple[str, ...]
    instance_families: tuple[str, ...]
    per_instance_mean: tuple[float, ...]
    mean_score: float
    median_score: float
    instance_std_score: float
    worst_score: float
    top_k_mean: float
    cvar: float
    total_runtime_sec: float
    metric_name: str = "optimality_gap"
    metric_unit: str = "percent"

    def _gap_alias(self, name):
        if self.metric_name != "optimality_gap":
            raise ValueError("raw objective profiles do not expose gap aliases")
        return getattr(self, name)

    @property
    def seed_mean_gaps(self):
        return self._gap_alias("seed_mean_scores")

    @property
    def seed_std_gap(self):
        return self._gap_alias("seed_std_score")

    @property
    def mean_gap(self):
        return self._gap_alias("mean_score")

    @property
    def median_gap(self):
        return self._gap_alias("median_score")

    @property
    def instance_std_gap(self):
        return self._gap_alias("instance_std_score")

    @property
    def worst_gap(self):
        return self._gap_alias("worst_score")


@dataclass(frozen=True)
class FailureCluster:
    cluster_id: str
    instance_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    failing_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveSnapshot:
    profiles: dict[str, BehaviorProfile]
    mean_elites: tuple[str, ...]
    cvar_elites: tuple[str, ...]
    worst_elites: tuple[str, ...]
    structural_elites: tuple[str, ...]
    candidate_clusters: dict[str, tuple[str, ...]]
    failure_clusters: dict[str, FailureCluster]
    observations: dict[str, Observation]

    @property
    def visible_candidate_ids(self) -> set[str]:
        return (
            set(self.mean_elites)
            | set(self.cvar_elites)
            | set(self.worst_elites)
            | set(self.structural_elites)
        )


def restrict_archive_snapshot_to_mean(
    snapshot: ArchiveSnapshot,
) -> ArchiveSnapshot:
    profiles = {
        candidate_id: snapshot.profiles[candidate_id]
        for candidate_id in snapshot.mean_elites
        if candidate_id in snapshot.profiles
    }
    candidate_clusters = _build_candidate_clusters(profiles)
    failure_clusters = _build_failure_clusters(profiles)
    observations = _build_observations(
        profiles,
        failure_clusters,
        candidate_clusters=candidate_clusters,
        mean_elites=tuple(profiles),
        cvar_elites=(),
        worst_elites=(),
        structural_elites=(),
    )
    return ArchiveSnapshot(
        profiles=profiles,
        mean_elites=tuple(profiles),
        cvar_elites=(),
        worst_elites=(),
        structural_elites=(),
        candidate_clusters=candidate_clusters,
        failure_clusters=failure_clusters,
        observations=observations,
    )


def build_archive_snapshot(
    evaluations: list[EvaluationRecord],
    *,
    split: str = "train",
    view_size: int = 5,
    tail_fraction: float = 0.1,
    excluded_candidate_ids: set[str] | None = None,
    canonical_instance_ids: tuple[str, ...] | None = None,
    instance_families: dict[str, str] | None = None,
    structural_signatures: dict[str, str] | None = None,
) -> ArchiveSnapshot:
    if view_size <= 0:
        raise ValueError("view_size must be positive")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")

    metrics = {(r.metric_name, r.metric_unit) for r in evaluations if r.valid and r.split == split}
    if len(metrics) > 1:
        raise ValueError("archive cannot mix metric identities")
    excluded_candidate_ids = excluded_candidate_ids or set()
    profiles = _build_profiles(
        [
            record
            for record in evaluations
            if record.candidate_id not in excluded_candidate_ids
        ],
        split=split,
        tail_fraction=tail_fraction,
        canonical_instance_ids=canonical_instance_ids,
        instance_families=instance_families or {},
    )
    ordered_mean = sorted(
        profiles.values(), key=lambda item: (item.mean_score, item.candidate_id)
    )
    ordered_cvar = sorted(
        profiles.values(),
        key=lambda item: (item.cvar, item.worst_score, item.mean_score, item.candidate_id),
    )
    ordered_worst = sorted(
        profiles.values(),
        key=lambda item: (item.worst_score, item.mean_score, item.candidate_id),
    )
    candidate_clusters = _build_candidate_clusters(profiles)
    clusters = _build_failure_clusters(profiles)
    mean_elites = tuple(item.candidate_id for item in ordered_mean[:view_size])
    cvar_elites = tuple(item.candidate_id for item in ordered_cvar[:view_size])
    worst_elites = tuple(item.candidate_id for item in ordered_worst[:view_size])
    structural_elites = _structural_elites(
        ordered_mean,
        structural_signatures or {},
        view_size=view_size,
    )
    observations = _build_observations(
        profiles,
        clusters,
        mean_elites=mean_elites,
        cvar_elites=cvar_elites,
        worst_elites=worst_elites,
        structural_elites=structural_elites,
        candidate_clusters=candidate_clusters,
    )
    return ArchiveSnapshot(
        profiles=profiles,
        mean_elites=mean_elites,
        cvar_elites=cvar_elites,
        worst_elites=worst_elites,
        structural_elites=structural_elites,
        candidate_clusters=candidate_clusters,
        failure_clusters=clusters,
        observations=observations,
    )


def _build_profiles(
    evaluations: list[EvaluationRecord],
    *,
    split: str,
    tail_fraction: float,
    canonical_instance_ids: tuple[str, ...] | None,
    instance_families: dict[str, str],
) -> dict[str, BehaviorProfile]:
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in evaluations:
        if record.valid and record.split == split:
            grouped.setdefault(record.candidate_id, []).append(record)

    profiles = {}
    for candidate_id, records in grouped.items():
        records = [
            record
            for record in records
            if canonical_instance_ids is None
            or record.instance_ids == canonical_instance_ids
        ]
        if not records:
            continue
        instance_ids = canonical_instance_ids or records[0].instance_ids
        seed_rows: dict[int, tuple[float, ...]] = {}
        seed_runtime: dict[int, float] = {}
        for record in records:
            if record.instance_ids != instance_ids:
                raise ValueError(f"instance mismatch for candidate {candidate_id}")
            for row_index, (seed, row) in enumerate(
                zip(record.seeds, record.score_matrix)
            ):
                seed_rows[seed] = row
                seed_runtime[seed] = (
                    float(sum(record.runtime_matrix[row_index]))
                    if record.runtime_matrix
                    else record.total_runtime_sec / len(record.seeds)
                )
        seeds = tuple(sorted(seed_rows))
        matrix = np.asarray([seed_rows[seed] for seed in seeds], dtype=float)
        per_instance = tuple(float(value) for value in matrix.mean(axis=0))
        seed_mean_scores = tuple(float(value) for value in matrix.mean(axis=1))
        tail_count = max(1, math.ceil(len(per_instance) * tail_fraction))
        ordered_tail = sorted(per_instance, reverse=True)
        top_k = ordered_tail[: min(5, len(ordered_tail))]
        tail = ordered_tail[:tail_count]
        profiles[candidate_id] = BehaviorProfile(
            candidate_id=candidate_id,
            metric_name=records[0].metric_name,
            metric_unit=records[0].metric_unit,
            seeds=seeds,
            seed_mean_scores=seed_mean_scores,
            seed_std_score=(
                float(pstdev(seed_mean_scores)) if len(seed_mean_scores) > 1 else 0.0
            ),
            instance_ids=instance_ids,
            instance_families=tuple(
                instance_families.get(instance_id, "unknown")
                for instance_id in instance_ids
            ),
            per_instance_mean=per_instance,
            mean_score=float(matrix.mean()),
            median_score=float(median(per_instance)),
            instance_std_score=(
                float(pstdev(per_instance)) if len(per_instance) > 1 else 0.0
            ),
            worst_score=max(per_instance),
            top_k_mean=float(fmean(top_k)),
            cvar=float(fmean(tail)),
            total_runtime_sec=float(sum(seed_runtime.values())),
        )
    return profiles


def _structural_elites(
    ordered_mean: list[BehaviorProfile],
    signatures: dict[str, str],
    *,
    view_size: int,
) -> tuple[str, ...]:
    selected = []
    seen = set()
    for profile in ordered_mean:
        signature = signatures.get(profile.candidate_id)
        if signature is None or signature in seen:
            continue
        seen.add(signature)
        selected.append(profile.candidate_id)
        if len(selected) == view_size:
            break
    return tuple(selected)


def _build_failure_clusters(
    profiles: dict[str, BehaviorProfile],
) -> dict[str, FailureCluster]:
    if not profiles:
        return {}
    identifiers = sorted(profiles)
    instance_ids = profiles[identifiers[0]].instance_ids
    if any(profiles[item].instance_ids != instance_ids for item in identifiers):
        raise ValueError("candidate profiles use different instance sets")
    matrix = np.asarray(
        [profiles[candidate_id].per_instance_mean for candidate_id in identifiers],
        dtype=float,
    )
    instance_groups = _cluster_labeled_rows(
        matrix.T,
        list(instance_ids),
        max_clusters=8,
    )
    clusters = {}
    instance_position = {instance_id: index for index, instance_id in enumerate(instance_ids)}
    family_by_instance = dict(
        zip(instance_ids, profiles[identifiers[0]].instance_families)
    )
    for index, members in enumerate(instance_groups):
        columns = [instance_position[instance_id] for instance_id in members]
        candidate_means = matrix[:, columns].mean(axis=1)
        threshold = float(np.quantile(candidate_means, 0.75))
        signature = tuple(
            identifiers[position]
            for position, value in enumerate(candidate_means)
            if value >= threshold and value > 0
        )
        cluster_id = f"failure_{index:03d}"
        clusters[cluster_id] = FailureCluster(
            cluster_id=cluster_id,
            instance_ids=tuple(members),
            family_ids=tuple(sorted({family_by_instance[item] for item in members})),
            failing_candidate_ids=signature,
        )
    return clusters


def _build_candidate_clusters(
    profiles: dict[str, BehaviorProfile],
) -> dict[str, tuple[str, ...]]:
    if not profiles:
        return {}
    identifiers = sorted(profiles)
    matrix = np.asarray(
        [profiles[candidate_id].per_instance_mean for candidate_id in identifiers],
        dtype=float,
    )
    groups = _cluster_labeled_rows(matrix, identifiers, max_clusters=10)
    return {
        f"candidate_{index:03d}": tuple(group)
        for index, group in enumerate(groups)
    }


def _cluster_labeled_rows(
    matrix: np.ndarray,
    labels: list[str],
    *,
    max_clusters: int,
) -> list[list[str]]:
    row_count = len(labels)
    if row_count == 0:
        return []
    cluster_count = max(1, min(max_clusters, round(math.sqrt(row_count))))
    if cluster_count == 1:
        return [sorted(labels)]

    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    scale = np.maximum(np.maximum(mad, matrix.std(axis=0)), 1e-9)
    normalized = (matrix - center) / scale
    first = int(np.argmax(np.linalg.norm(normalized, axis=1)))
    centroid_indices = [first]
    nearest = np.linalg.norm(normalized - normalized[first], axis=1)
    nearest[first] = -1.0
    while len(centroid_indices) < cluster_count:
        selected = int(np.argmax(nearest))
        centroid_indices.append(selected)
        nearest = np.minimum(nearest, np.linalg.norm(normalized - normalized[selected], axis=1))
        nearest[centroid_indices] = -1.0
    centroids = normalized[centroid_indices].copy()

    assignments = np.zeros(row_count, dtype=int)
    for _ in range(25):
        # Stream one centroid at a time: O(rows * features), not
        # O(rows * clusters * features) temporary space. Strict < retains
        # NumPy argmin's first-centroid tie rule and subtraction arithmetic.
        best_distance = np.full(row_count, np.inf)
        updated = np.zeros(row_count, dtype=int)
        for cluster_index, centroid in enumerate(centroids):
            distance = np.linalg.norm(normalized - centroid, axis=1)
            better = distance < best_distance
            updated[better] = cluster_index
            best_distance[better] = distance[better]
        if np.array_equal(updated, assignments) and _ > 0:
            break
        assignments = updated
        for cluster_index in range(cluster_count):
            members = normalized[assignments == cluster_index]
            if len(members):
                centroids[cluster_index] = members.mean(axis=0)

    groups = [
        sorted(labels[index] for index in range(row_count) if assignments[index] == cluster)
        for cluster in range(cluster_count)
    ]
    return sorted((group for group in groups if group), key=lambda group: group[0])


def _build_observations(
    profiles: dict[str, BehaviorProfile],
    clusters: dict[str, FailureCluster],
    *,
    candidate_clusters: dict[str, tuple[str, ...]],
    mean_elites: tuple[str, ...],
    cvar_elites: tuple[str, ...],
    worst_elites: tuple[str, ...],
    structural_elites: tuple[str, ...],
) -> dict[str, Observation]:
    observations = {
        "baseline_available": Observation(
            observation_id="baseline_available",
            kind="problem",
            statement="The problem adapter provides a complete baseline solver.",
        )
    }
    memberships = {
        "mean": set(mean_elites),
        "cvar10": set(cvar_elites),
        "worst": set(worst_elites),
        "structure": set(structural_elites),
    }
    for candidate_id, profile in sorted(profiles.items()):
        observation_id = f"candidate:{candidate_id}:profile"
        observations[observation_id] = Observation(
            observation_id=observation_id,
            kind="candidate_profile",
            statement=(
                f"{candidate_id} has mean {profile.metric_name} {profile.mean_score:.12g}, "
                f"tail {profile.cvar:.12g}, and worst {profile.worst_score:.12g} ({profile.metric_unit})."
            ),
            values={
                "candidate_id": candidate_id,
                "metric_name": profile.metric_name,
                "metric_unit": profile.metric_unit,
                "mean_score": profile.mean_score,
                **({"mean_gap": profile.mean_score, "worst_gap": profile.worst_score}
                   if profile.metric_name == "optimality_gap" else {}),
                "cvar": profile.cvar,
                "worst_score": profile.worst_score,
                "archives": [
                    name for name, members in memberships.items() if candidate_id in members
                ],
            },
        )
    for cluster_id, cluster in sorted(clusters.items()):
        observation_id = f"cluster:{cluster_id}"
        observations[observation_id] = Observation(
            observation_id=observation_id,
            kind="failure_cluster",
            statement=(
                f"{cluster_id} contains {len(cluster.instance_ids)} instances and "
                f"is difficult for {', '.join(cluster.failing_candidate_ids) or 'no candidate'}."
            ),
            values={
                "cluster_id": cluster_id,
                "instance_ids": list(cluster.instance_ids),
                "failing_candidate_ids": list(cluster.failing_candidate_ids),
            },
        )
    for cluster_id, members in sorted(candidate_clusters.items()):
        observation_id = f"cluster:{cluster_id}"
        observations[observation_id] = Observation(
            observation_id=observation_id,
            kind="candidate_cluster",
            statement=f"{cluster_id} contains behaviorally similar candidates: {', '.join(members)}.",
            values={"cluster_id": cluster_id, "candidate_ids": list(members)},
        )
    return observations
