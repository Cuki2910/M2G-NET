"""Synthetic-only checks; no fits, production datasets, or held-out predictions.

Run: .venv/bin/python -m unittest tests.test_v3_diagnostic_runner -v
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_v3_diagnostics as runner


class DiagnosticRunnerTests(unittest.TestCase):
    def fixture(self):
        references, arrays = {}, {}
        for split, offset in (("train", 0), ("validation", 4)):
            y = np.tile([[0], [1], [0], [1]], (1, len(runner.cfg.TASK_NAMES)))
            references[split] = dict(row_ids=np.arange(4) + offset, targets=y,
                                     probabilities=np.zeros_like(y), masks=np.ones_like(y))
            for key, value in references[split].items():
                arrays[f"{split}_{key}"] = y * 0.6 + 0.2 if key == "probabilities" else value.copy()
        _, metrics = runner.metric_rows(arrays)
        telemetry = {"history": [{"epoch": 1, "training_metrics_dropout": metrics["train"],
                                   "validation_metrics": metrics["validation"]}],
                     "best": {"epoch": 1, "train_metrics": metrics["train"],
                              "validation_metrics": metrics["validation"]}}
        return references, arrays, telemetry

    def test_atomic_cache_identity_hash_and_fresh_rows(self):
        references, arrays, telemetry = self.fixture()
        identity = {"synthetic": True, "member": 0}
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "tests") as temporary:
            directory = Path(temporary) / "member-0"
            runner.write_cache(directory, identity, arrays, telemetry, references)
            loaded, history = runner.read_cache(directory, identity, references)
            np.testing.assert_array_equal(loaded["train_targets"], arrays["train_targets"])
            self.assertEqual(history["best"]["epoch"], 1)
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.read_cache(directory, dict(identity, member=1), references)
            changed = deepcopy(references)
            changed["train"]["targets"][0, 0] = 1
            with self.assertRaisesRegex(ValueError, "alignment"):
                runner.read_cache(directory, identity, changed)
            for artifact in runner.ARTIFACTS:
                path = directory / artifact
                original = path.read_bytes()
                path.write_bytes(original + b"tamper")
                with self.assertRaisesRegex(ValueError, "changed"):
                    runner.read_cache(directory, identity, references)
                path.write_bytes(original)
            manifest = json.loads((directory / "complete.json").read_text())
            self.assertEqual(set(manifest["artifact_sha256"]), set(runner.ARTIFACTS))

    def test_mean_raw_probabilities_not_mean_metrics(self):
        references, first, _ = self.fixture()
        second = deepcopy(first)
        for split in runner.SPLITS:
            second[f"{split}_probabilities"] = 1 - first[f"{split}_probabilities"]
        ensemble = runner.ensemble_arrays([first, second], references)
        rows, _ = runner.metric_rows(ensemble)
        for split in runner.SPLITS:
            np.testing.assert_allclose(ensemble[f"{split}_probabilities"], 0.5)
        for row in rows:
            self.assertAlmostEqual(row["train_auc"], 0.5)
            self.assertAlmostEqual(row["train_brier"], 0.25)
            self.assertAlmostEqual(row["auc_gap"], 0)
        first_rows, _ = runner.metric_rows(first)
        second_rows, _ = runner.metric_rows(second)
        self.assertNotAlmostEqual(rows[0]["train_brier"],
                                  (first_rows[0]["train_brier"] + second_rows[0]["train_brier"]) / 2)
        with self.assertRaisesRegex(ValueError, "empty"):
            runner.ensemble_arrays([], references)
        second["train_row_ids"] = second["train_row_ids"][::-1]
        with self.assertRaisesRegex(ValueError, "alignment"):
            runner.ensemble_arrays([first, second], references)

    def test_telemetry_checkpoint_selection_and_dropout_label(self):
        _, _, telemetry = self.fixture()
        later = deepcopy(telemetry["history"][0])
        later["epoch"] = 2
        telemetry["history"].append(later)
        runner.validate_telemetry(telemetry)  # First maximum wins, not latest tie.
        telemetry["best"]["epoch"] = 2
        with self.assertRaisesRegex(ValueError, "first maximum"):
            runner.validate_telemetry(telemetry)
        telemetry["best"]["epoch"] = 1
        del telemetry["history"][0]["training_metrics_dropout"]
        with self.assertRaises(KeyError):
            runner.validate_telemetry(telemetry)

    def test_group_means_average_metrics_over_training_seeds(self):
        import pandas as pd

        records = []
        columns = ("train_auc", "validation_auc", "auc_gap", "train_brier", "validation_brier", "brier_gap")
        for training, value in ((42, 0.2), (101, 0.6)):
            for member in range(5):
                records.append(dict(mechanism="additive", generator_seed=1701,
                                    training_seed=training, estimator="member", member=member,
                                    task="macro", **dict.fromkeys(columns, value)))
            records.append(dict(mechanism="additive", generator_seed=1701,
                                training_seed=training, estimator="ensemble", member="",
                                task="macro", **dict.fromkeys(columns, value / 2)))
        generators, summary = runner.group_metrics(pd.DataFrame(records))
        self.assertAlmostEqual(generators.query("estimator == 'member'").iloc[0].train_auc, 0.4)
        self.assertAlmostEqual(generators.query("estimator == 'ensemble'").iloc[0].train_auc, 0.2)
        self.assertTrue(all(row["n_generators"] == 1 and row["mcse"] is None for row in summary))

    def test_score_publishes_immutable_hashed_synthetic_report(self):
        import pandas as pd

        references, arrays, telemetry = self.fixture()
        job = ("additive", 1701, 42)
        with tempfile.TemporaryDirectory(dir=runner.ROOT / "tests") as temporary:
            root = Path(temporary)
            data = root / "synthetic.csv"
            data.write_text("synthetic only")
            identity = dict(job=[*job, "legacy"], sources={}, dataset_sha256=runner.sha256(data),
                            runtime={"synthetic": "1"}, python="synthetic")
            for index in range(runner.cfg.V3_ENSEMBLE_SIZE):
                runner.write_cache(root / "unit" / f"member-{index}",
                                   runner.member_identity(identity, index), arrays, telemetry, references)
            with (patch.object(runner, "diagnostic_identity", return_value=identity),
                  patch.object(runner, "dataset_path", return_value=data),
                  patch.object(runner, "validation_frames", return_value=("train", "validation")),
                  patch.object(runner, "reference_arrays", side_effect=lambda split: references[split]),
                  patch.object(runner, "unit_path", return_value=root / "unit"),
                  patch.object(runner.cfg, "V3_DIAGNOSTIC_REPORT_ROOT", str(root / "reports")),
                  patch("builtins.print")):
                report = runner.score([job])
                second_report = runner.score([job])
            self.assertTrue(report.name.isnumeric())
            self.assertNotEqual(report, second_report)
            units = pd.read_csv(report / "unit-metrics.csv")
            self.assertEqual(len(units), 6 * (len(runner.cfg.TASK_NAMES) + 1))
            curves = pd.read_csv(report / "epoch-curves.csv")
            self.assertIn("training_auc_dropout", curves)
            self.assertNotIn("train_auc", curves)
            provenance = json.loads((report / "provenance.json").read_text())
            for name, digest in provenance["report_sha256"].items():
                self.assertEqual(runner.sha256(report / name), digest)
            self.assertTrue(all(runner.sha256(runner.project_path(path)) == digest
                                for path, digest in provenance["input_sha256"].items()))

    def test_cli_defaults_to_full_score_without_calling_fit(self):
        with patch.object(runner, "score") as score, patch.object(runner, "run_fits") as fits:
            runner.main([])
            fits.assert_not_called()
            self.assertEqual(len(score.call_args.args[0]),
                             len(runner.cfg.V3_DEV_MECHANISMS) * len(runner.cfg.V3_DEV_GENERATOR_SEEDS)
                             * len(runner.cfg.V3_DEV_TRAINING_SEEDS))
            self.assertEqual(len(score.call_args.args[0]) * runner.cfg.V3_ENSEMBLE_SIZE, 300)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.main(["--generator-seeds", "1711"])


if __name__ == "__main__":
    unittest.main()
