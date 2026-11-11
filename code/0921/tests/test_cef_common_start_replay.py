"""Pure common-start manifest and paired-summary semantics; no solver/API calls."""
import copy
import unittest

try:
    from cef import replay
except ImportError:
    replay = None
from cef.patch import digest
from cef.research import ResearchTools
from cef.store import canonical


def rows(count=10):
    return [dict(id=f"train_{i:02d}", split="train", instance_hash=f"instance_{i}") for i in range(count)]


def snapshot(row, phase="early"):
    solution = [0, 2, 1]
    return dict(snapshot_id=f"{row['id']}:{phase}", source_candidate_id="parent:PAB",
                source_hash="source-a", instance_id=row["id"], instance_hash=row["instance_hash"],
                source_seed=2025, elapsed_s=0.1 if phase == "early" else 2.5,
                solution=solution, solution_hash=digest(canonical(solution)), objective=100.,
                phase=phase, stagnation_s=0. if phase == "early" else 1.)


def records(states, seeds, reference=90., candidate=80.):
    return [dict(snapshot_id=s["snapshot_id"], restart_seed=seed, role=role, valid=True,
                 start_objective=s["objective"], objective=objective, elapsed=1.,
                 wallclock_s=1.2, cpu_seconds=0.8)
            for s in states for seed in seeds
            for role, objective in (("reference", reference), ("candidate", candidate))]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(replay, "common-start replay module is not implemented")
        self.rows = rows()
        self.snapshots = [snapshot(r, phase) for r in self.rows for phase in ("early", "late")]
        self.states = [snapshot(r) for r in self.rows[:2]]
        self.seeds = [301, 302]

    def test_frozen_policy_and_deterministic_balanced_unique_selection(self):
        self.assertEqual(replay.RESEARCH_VERSION, "common_start_v1")
        self.assertEqual(replay.TRACE_POLICY["max_snapshots"], 6)
        self.assertTrue(replay.TRACE_POLICY["fractions"])
        selected = replay.select_states(self.snapshots, self.rows, "source-a", 2025, 8)
        reversed_selection = replay.select_states(list(reversed(self.snapshots)), list(reversed(self.rows)), "source-a", 2025, 8)
        self.assertEqual(selected, reversed_selection)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({s["instance_id"] for s in selected}), 8)
        self.assertEqual(sum(s["phase"] == "early" for s in selected), 4)
        self.assertEqual(sum(s["phase"] == "late" for s in selected), 4)
        self.assertNotEqual(selected, replay.select_states(self.snapshots, self.rows, "source-a", 2026, 8))
        selected[0]["solution"][0] = 999
        self.assertNotIn(999, self.snapshots[0]["solution"])

    def test_missing_late_and_small_pool_are_not_padded(self):
        early_only = [s for s in self.snapshots if s["phase"] == "early"]
        self.assertEqual(len(replay.select_states(early_only, self.rows, "source-a", 2025, 8)), 4)
        small_rows = self.rows[:3]
        selected = replay.select_states(self.snapshots[:6], small_rows, "source-a", 2025, 8)
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(len(selected), len({s["instance_id"] for s in selected}))

    def test_other_source_is_not_selected(self):
        other = copy.deepcopy(self.snapshots[0]); other["source_hash"] = "source-b"
        self.assertEqual(replay.select_states([other], self.rows, "source-a", 2025, 8), [])

    def test_small_pool_keeps_phase_balance_and_reserves_scarce_late_instances(self):
        selected = replay.select_states(self.snapshots[:6], self.rows[:3], "source-a", 2025, 8)
        self.assertEqual([s["phase"] for s in selected].count("early"), 2)
        self.assertEqual([s["phase"] for s in selected].count("late"), 1)
        scarce = [snapshot(self.rows[0]), snapshot(self.rows[1]), snapshot(self.rows[0], "late")]
        selected = replay.select_states(scarce, self.rows[:2], "source-a", 0, 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual([s["phase"] for s in selected], ["early", "late"])

    def test_invalid_provenance_hash_solution_and_test_split_are_rejected(self):
        changes = [dict(instance_id="test_01"), dict(instance_hash="wrong"),
                   dict(solution_hash="wrong"), dict(solution=[0, 0, 1]),
                   dict(objective=float("nan")), dict(objective=0.),
                   dict(source_candidate_id=""), dict(source_seed=None),
                   dict(elapsed_s=-1.), dict(phase="unknown")]
        for change in changes:
            with self.subTest(change=change):
                item = {**self.snapshots[0], **change}
                with self.assertRaises(ValueError):
                    replay.select_states([item], self.rows, "source-a", 2025, 8)
        bad_rows = [dict(self.rows[0], split="test")]
        with self.assertRaises(ValueError):
            replay.select_states([self.snapshots[0]], bad_rows, "source-a", 2025, 8)

    def test_inline_instance_dimension_is_checked(self):
        row = dict(self.rows[0], instance={"distance_matrix": [[0, 1], [1, 0]], "reference": 2})
        row["instance_hash"] = digest(canonical({"distance_matrix": [[0, 1], [1, 0]]}))
        item = dict(self.snapshots[0], instance_hash=row["instance_hash"])
        with self.assertRaises(ValueError):
            replay.select_states([item], [row], "source-a", 2025, 1)

    def test_count_and_duplicate_manifest_ids_rejected(self):
        for count in (0, 9, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                replay.select_states(self.snapshots, self.rows, "source-a", 2025, count)
        with self.assertRaises(ValueError):
            replay.select_states(self.snapshots + [self.snapshots[0]], self.rows, "source-a", 2025, 8)

    def test_cache_binds_every_execution_input_and_version(self):
        args = dict(reference_hash="ref", candidate_hash="cand", states=self.states,
                    seeds=self.seeds, seconds=3., implementation={"worker": "v1"},
                    workers=4, environment={"cpu": "frozen"})
        key = replay.cache_key(**args)
        self.assertEqual(len(key), 64)
        self.assertEqual(key, replay.cache_key(**copy.deepcopy(args)))
        changes = dict(reference_hash="other", candidate_hash="other", seeds=[301, 303],
                       seconds=2., implementation={"worker": "v2"}, workers=2,
                       environment={"cpu": "changed"})
        for field, value in changes.items():
            with self.subTest(field=field):
                self.assertNotEqual(key, replay.cache_key(**{**args, field: value}))
        state = copy.deepcopy(self.states); state[0]["solution_hash"] = "changed"
        self.assertNotEqual(key, replay.cache_key(**{**args, "states": state}))
        original = replay.RESEARCH_VERSION
        try:
            replay.RESEARCH_VERSION = "different"
            self.assertNotEqual(key, replay.cache_key(**args))
        finally:
            replay.RESEARCH_VERSION = original

    def test_gains_positive_support_and_usage(self):
        summary = replay.summarize(records(self.states, self.seeds), self.states, self.seeds)
        self.assertEqual(summary["status"], "supported")
        self.assertAlmostEqual(summary["mean_delta"], .1)
        self.assertEqual(summary["positive_states"], 2)
        self.assertEqual(summary["valid_runs"], 8)
        self.assertEqual(summary["total_runs"], summary["expected_runs"])
        self.assertAlmostEqual(summary["usage"]["elapsed"], 8.)
        self.assertAlmostEqual(summary["usage"]["wallclock_s"], 9.6)
        self.assertAlmostEqual(summary["usage"]["cpu_seconds"], 6.4)
        self.assertFalse(summary["statistical_claim"])

    def test_agent_report_names_the_measured_comparison_and_ratio_scale(self):
        summary = replay.summarize(records(self.states, self.seeds), self.states, self.seeds)
        report = replay.agent_report(summary, "reference:PAB", "candidate:PAB",
            "Does the candidate improve the full solver?",
            dict(states=2,restart_seeds=[301,302],time_limit_s=3.))
        self.assertEqual(report["research_purpose"], "Does the candidate improve the full solver?")
        self.assertEqual(report["candidate_better_states"], 2)
        self.assertEqual(report["reference_better_states"], 0)
        self.assertEqual(report["tied_states"], 0)
        self.assertAlmostEqual(report["mean_candidate_minus_reference_gain_ratio"], .1)
        self.assertAlmostEqual(report["mean_candidate_minus_reference_gain_percent_of_start"], 10.)
        self.assertIn("percentage of the frozen starting objective", report["metric_definition"])
        self.assertIn("candidate:PAB", report["comparison"])
        self.assertIn("Same frozen training starts", report["conditions"])
        self.assertIn("does not measure TSPLIB gap", report["limits"])
        self.assertIn("2 of 2", report["report"])

    def test_agent_report_does_not_turn_mixed_ties_into_a_binary_verdict(self):
        summary = replay.summarize(records(self.states, self.seeds, 90., 90.), self.states, self.seeds)
        report = replay.agent_report(summary, "reference:PAB", "candidate:PAB", "Why did this regress?")
        self.assertEqual(report["direction"], "mixed_or_tied")
        self.assertEqual(report["tied_states"], 2)
        self.assertIn("no directional conclusion", report["report"])

    def test_agent_report_keeps_negative_mean_as_reference_favored_when_majority(self):
        summary = replay.summarize(records(self.states, self.seeds, 80., 90.), self.states, self.seeds)
        report = replay.agent_report(summary, "reference:PAB", "candidate:PAB", "Compare them")
        self.assertEqual(report["direction"], "reference_favored")
        self.assertAlmostEqual(report["mean_candidate_minus_reference_gain_percent_of_start"], -10.)
        self.assertIn("reference gained more", report["report"])

    def test_planner_brief_uses_readable_report_and_drops_ambiguous_status(self):
        summary = replay.summarize(records(self.states, self.seeds, 80., 90.), self.states, self.seeds)
        raw = dict(summary, event_id="probe:test", reference_candidate_id="reference:PAB",
            candidate_id="candidate:PAB", question="Investigate continuation quality.",
            unit="ratio", experiment=dict(method="paired_common_start", time_limit_s=3))
        view = ResearchTools.brief(raw)
        self.assertNotIn("status", view)
        self.assertNotIn("question", view)
        self.assertEqual(view["research_purpose"], raw["question"])
        self.assertEqual(view["direction"], "reference_favored")
        self.assertEqual(view["event_id"], "probe:test")
        self.assertNotIn("run_results", view)
        self.assertEqual(ResearchTools.brief(view), view)

    def test_integration_rule_requires_five_wins_and_positive_mean(self):
        base = dict(event_id="probe:test", candidate_id="candidate:PAB",
            mean_candidate_minus_reference_gain_ratio=.01,
            candidate_better_states=4, execution=dict(completed=True))
        self.assertFalse(ResearchTools.integration_option(base)["available"])
        self.assertFalse(ResearchTools.integration_option(dict(base,candidate_better_states=5,
            mean_candidate_minus_reference_gain_ratio=-.01))["available"])
        self.assertTrue(ResearchTools.integration_option(dict(base,candidate_better_states=5))["available"])
        self.assertFalse(ResearchTools.integration_option(dict(base,candidate_better_states=8),used=True)["available"])

    def test_swapping_roles_reverses_delta(self):
        result = replay.summarize(records(self.states, self.seeds, 80., 90.), self.states, self.seeds)
        self.assertEqual(result["status"], "not_supported")
        self.assertAlmostEqual(result["mean_delta"], -.1)
        self.assertEqual(result["negative_states"], 2)

    def test_negative_gain_is_preserved(self):
        result = replay.summarize(records(self.states, self.seeds, 100., 110.), self.states, self.seeds)
        self.assertAlmostEqual(result["state_results"][0]["candidate_gain_ratio"], -.1)
        self.assertAlmostEqual(result["mean_delta"], -.1)

    def test_ties_and_positive_mean_without_majority_are_not_supported(self):
        runs = records(self.states, self.seeds, 100., 100.)
        tied = replay.summarize(runs, self.states, self.seeds)
        self.assertEqual(tied["status"], "not_supported")
        self.assertEqual(tied["tied_states"], 2)
        for run in runs:
            if run["role"] == "candidate":
                run["objective"] = 70. if run["snapshot_id"] == self.states[0]["snapshot_id"] else 110.
        result = replay.summarize(runs, self.states, self.seeds)
        self.assertGreater(result["mean_delta"], 0.)
        self.assertEqual(result["status"], "not_supported")

    def test_incomplete_invalid_duplicate_unknown_and_mismatched_start_are_inconclusive(self):
        good = records(self.states, self.seeds)
        cases = [good[:-1], good + [good[0]], good + [dict(good[0], snapshot_id="unknown")],
                 [dict(good[0], valid=False, objective=None)] + good[1:],
                 [dict(good[0], objective=None)] + good[1:],
                 [dict(good[0], start_objective=200.)] + good[1:]]
        for runs in cases:
            with self.subTest(runs=runs):
                result = replay.summarize(runs, self.states, self.seeds)
                self.assertEqual(result["status"], "inconclusive")
                self.assertIsNone(result["mean_delta"])
                self.assertTrue(result["errors"])
                self.assertEqual(result["total_runs"], len(runs))
                self.assertEqual(result["expected_runs"], 8)

    def test_empty_batch_is_inconclusive_and_unknown_usage_is_not_zero(self):
        self.assertEqual(replay.summarize([], [], self.seeds)["status"], "inconclusive")
        runs = records(self.states, self.seeds)
        del runs[0]["cpu_seconds"]
        result = replay.summarize(runs, self.states, self.seeds)
        self.assertIsNone(result["usage"]["cpu_seconds"])
        self.assertEqual(result["usage_unknown"]["cpu_seconds"], 1)


if __name__ == "__main__":
    unittest.main()
