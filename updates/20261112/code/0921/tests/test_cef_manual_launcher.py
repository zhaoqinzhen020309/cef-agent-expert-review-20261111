"""Launcher checks never dispatch a real provider or run a solver."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class ManualLauncherTests(unittest.TestCase):
    def launcher(self):
        # Explicit legacy full-contrast fixture; runCEF.py uses joint-only generation.
        self.settings = dict(model='deepseek-flash',base_url='https://api.deepseek.com',
                             seeds=[2025],iterations=20,workers=4,output_dir=ROOT/'experiments/legacy_cef_fixture')
        return importlib.import_module("scripts.cef_launcher")

    def test_configuration_keeps_data_and_sets_training_and_test_seeds(self):
        launcher = self.launcher()
        with tempfile.TemporaryDirectory() as tmp:
            config, test = launcher.build_config(2027, Path(tmp) / "fresh", self.settings)
        self.assertEqual(len(config["data_manifest"]["train"]), 64)
        self.assertEqual(len(test["instances"]), 29)
        self.assertEqual(set(config["evaluation_seed_manifest"].values()), {2027})
        self.assertEqual(set(test["evaluation_seeds"].values()), {2027})
        self.assertNotIn("seed", config["model_sampling"])
        self.assertEqual(config["model_seed_support"], "unsupported")
        self.assertTrue(config["implementation_source_manifest"])
        from cef.config import validate
        self.assertEqual(validate(config)["task"], "tsp")

    def test_check_does_not_create_output_or_dispatch_commands(self):
        launcher = self.launcher()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not_created"
            settings = dict(self.settings, output_dir=output)
            with patch.object(
                launcher, "run_command", side_effect=AssertionError("No dispatch in --check")
            ):
                self.assertEqual(launcher.main(settings, ["--check"]), 0)
            self.assertFalse(output.exists())

    def test_failed_search_does_not_start_final_test(self):
        launcher = self.launcher()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "trial"
            config, test = launcher.build_config(2025, folder, self.settings)
            actions = []

            def dispatch(*args):
                actions.append(args[0])
                self.assertEqual(args[0], "run")
                search = Path(config["run_dir"])
                search.mkdir()
                (search / "result.json").write_text(
                    json.dumps({"status": "failed", "reason": "provider response incomplete"}),
                    encoding="utf-8",
                )

            with patch.object(launcher, "matching_receipt", return_value=True), patch.object(
                launcher, "run_command", side_effect=dispatch
            ):
                with self.assertRaisesRegex(RuntimeError, "provider response incomplete"):
                    launcher.run_one(config, test, folder)
            self.assertEqual(actions, ["run"])
            self.assertFalse((folder / "test").exists())

    def test_search_finishes_before_test_and_records_summary(self):
        launcher = self.launcher()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "trial"
            config, test = launcher.build_config(2026, folder, self.settings)
            actions = []

            def dispatch(*args):
                actions.append(args[0])
                if args[0] == "run":
                    search = Path(config["run_dir"])
                    search.mkdir()
                    (search / "result.json").write_text(json.dumps({
                        "status": "finished", "reason": "configured iterations completed",
                        "completed_iterations": config["max_iterations"],
                        "best": {"candidate_id": "step_1:PAB", "fitness": 0.1},
                        "resource_usage": {"model_requests": 3, "model_fee": 0.03},
                    }), encoding="utf-8")
                elif args[0] == "test":
                    self.assertTrue((folder / "search/result.json").exists())
                    output = Path(args[args.index("--output") + 1])
                    output.mkdir()
                    (output / "summary.json").write_text(json.dumps({
                        "count": 29, "valid": True, "mean_gap_pct": 0.2,
                        "search_unchanged": True,
                        "evaluation_metric": "tsplib_unrounded_original_reference_v1",
                    }), encoding="utf-8")
                else:
                    self.fail("Unexpected command: " + args[0])

            with patch.object(launcher, "matching_receipt", return_value=True), patch.object(
                launcher, "run_command", side_effect=dispatch
            ):
                launcher.run_one(config, test, folder)
            self.assertEqual(actions, ["run", "test"])
            recorded = json.loads((folder / "launch_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["solver_seed"], 2026)
            self.assertEqual(recorded["test"]["mean_gap_pct"], 0.2)
            self.assertEqual(recorded['evaluation_metric'], 'tsplib_unrounded_original_reference_v1')

    def test_failed_preflight_is_preserved_without_search_or_retry(self):
        launcher = self.launcher()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "trial"
            config, test = launcher.build_config(2025, folder, self.settings)
            actions = []

            def dispatch(*args):
                actions.append(args[0])
                preflight = folder / "preflight"
                preflight.mkdir()
                (preflight / "receipt.json").write_text(
                    json.dumps({"passed": False, "error": "incomplete response"}), encoding="utf-8"
                )

            with patch.object(launcher, "matching_receipt", return_value=False), patch.object(
                launcher, "run_command", side_effect=dispatch
            ):
                with self.assertRaisesRegex(RuntimeError, "Preflight failed"):
                    launcher.run_one(config, test, folder)
            self.assertEqual(actions, ["preflight"])
            self.assertTrue((folder / "preflight/receipt.json").exists())
            self.assertFalse((folder / "search").exists())


if __name__ == "__main__":
    unittest.main()
