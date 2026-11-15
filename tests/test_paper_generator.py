"""Small-seed checks; never access reserved final generator seeds."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_paper_simulation import generate, write_dataset, validate, sha256
import config as cfg
from src.data_pipeline import prepare_labels, encode_splits, extract_views


class GeneratorTest(unittest.TestCase):
    def test_equations_pairing_and_schema(self):
        additive, _ = generate(11, "additive", 120)
        interaction, interaction_oracle = generate(11, "interaction", 120)
        partial, partial_oracle = generate(11, "partial", 120)
        feature_columns = [column for columns in cfg.INDIVIDUAL_VIEW_COLS.values() for column in columns]
        feature_columns += [column for columns in cfg.CONTEXTUAL_VIEW_COLS.values() for column in columns]
        feature_columns += [cfg.SITE_ID_COL, "red_phase_encounter", "made_turn"]
        pd.testing.assert_frame_equal(additive[feature_columns], partial[feature_columns])
        pd.testing.assert_frame_equal(interaction[feature_columns], partial[feature_columns])
        for task in cfg.TASK_NAMES:
            for suffix in ("_latent", "_probability", "_eligible"):
                pd.testing.assert_series_equal(interaction_oracle[task + suffix], partial_oracle[task + suffix])
        changed, _ = generate(12, "partial", 120)
        self.assertFalse(partial.equals(changed))
        prepare_labels(partial, require_masks=True)
        clean = partial.copy()
        for column in clean.columns:
            if column not in feature_columns:
                clean[column] = 0
        encoded, _, _, _, vocab = encode_splits(partial.iloc[:72], partial.iloc[72:96], partial.iloc[96:])
        cleaned, _, _, _, clean_vocab = encode_splits(clean.iloc[:72], clean.iloc[72:96], clean.iloc[96:])
        encoded, cleaned = extract_views(encoded, vocab), extract_views(cleaned, clean_vocab)
        for view in encoded:
            np.testing.assert_array_equal(encoded[view], cleaned[view])
        bad = partial.copy()
        task = cfg.TASK_NAMES[0]
        bad.loc[bad[cfg.TASK_MASK_COLS[task]].eq(0), task] = 1
        with self.assertRaises(ValueError):
            validate(bad, partial_oracle, "partial")

    def test_determinism_cli_and_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first", Path(directory) / "second"
            write_dataset(11, "partial", 120, first)
            write_dataset(11, "partial", 120, second)
            for filename in ("data.csv", "oracle.csv"):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
            before = sha256(first / "data.csv")
            with self.assertRaises(FileExistsError):
                write_dataset(11, "partial", 120, first)
            subprocess.run([sys.executable, "scripts/check_data.py", "--data", str(first / "data.csv"),
                            "--require-masks"], check=True, capture_output=True)
            self.assertEqual(before, sha256(first / "data.csv"))
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("simulated disk failure")):
                with self.assertRaises(OSError):
                    write_dataset(11, "partial", 120, Path(directory) / "failed")
            self.assertFalse((Path(directory) / "failed/manifest.json").exists())

    def test_reject_invalid_arguments(self):
        for seed, mechanism, rows in ((-1, "partial", 120), (11, "invalid", 120),
                                      (11, "additive", 0), (11, "partial", 121)):
            with self.assertRaises(ValueError):
                generate(seed, mechanism, rows)

    def test_red_light_equation(self):
        data, oracle = generate(11, "partial", 120)
        for index, row in data.iterrows():
            if not row.red_light_running_eligible:
                continue
            rider = {"Delivery": 1, "RideHailing": 0, "Private": -1}[row.rider_type]
            age = {"Young": 1, "Middle": 0, "Older": -1}[row.age_group]
            traffic = {"Light": -1, "Moderate": 0, "Heavy": 1}[row.traffic_condition]
            gender = 1 if row.gender == "Male" else -1
            police = 1 if row.police_presence == "Yes" else -1
            weather = 1 if row.weather == "Rain" else -1
            peak = 1 if row.time_slot == "Peak" else -1
            weekend = 1 if row.weekend == "Yes" else -1
            cross = 1 if row.intersection_type == "Cross" else -1
            logit = (-.50 + .35*rider + .25*age + .10*gender + .35*traffic - .35*police +
                     .10*weather + .20*peak + .10*weekend + .10*(row.num_lanes-3) + .10*cross +
                     .25*np.sin(2*np.pi*(row.intersection_id+1)/13) +
                     1.20*rider*traffic - .70*age*police + .50*rider*weather)
            self.assertAlmostEqual(oracle.loc[index, "red_light_running_probability"], 1/(1+np.exp(-logit)))
            expected = .75+.20/(1+np.exp(-(-.40+.50*rider-.40*traffic+.30*peak+.20*gender)))
            self.assertAlmostEqual(oracle.loc[index, "red_light_running_propensity"], expected)


if __name__ == "__main__":
    unittest.main()
