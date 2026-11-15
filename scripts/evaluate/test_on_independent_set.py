"""Evaluate a frozen checkpoint; the historical out-of-site CSV is development data."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch
import config as cfg
from scripts.train import collect_predictions
from src.checkpoint import load_model_bundle, dataset_sha256
from src.data_pipeline import RiderDataset, encode_with_preprocessing
from src.calibration import prediction_metrics
from src.metrics import print_results


def test_on_independent_set(checkpoint=cfg.CHECKPOINT_PATH, data="data/test/independent_test_set.csv"):
    bundle = load_model_bundle(checkpoint)
    raw = pd.read_csv(data)
    sites = set(raw[cfg.SITE_ID_COL])
    if sites & set(bundle["vocab"]["site_mapping"]):
        raise ValueError("Out-of-site evaluation requires site IDs disjoint from training.")
    encoded = encode_with_preprocessing(raw, bundle["encoders"], bundle["vocab"]["site_mapping"], bundle["label_policy"])
    loader = torch.utils.data.DataLoader(RiderDataset(encoded, bundle["vocab"]), batch_size=cfg.BATCH_SIZE)
    targets, probabilities, masks = collect_predictions(bundle["model"], loader, use_site_intercept=False)
    reports = prediction_metrics(targets, probabilities, masks, bundle["calibrator"], bundle["thresholds"])
    print("Development out-of-site evaluation; not proof of untouched external validation.")
    for rule, metrics in reports.items():
        print_results(metrics, title=rule)
    print(f"Data SHA256: {dataset_sha256(data)}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=cfg.CHECKPOINT_PATH)
    parser.add_argument("--data", default="data/test/independent_test_set.csv")
    args = parser.parse_args()
    return test_on_independent_set(args.checkpoint, args.data)


if __name__ == "__main__":
    main()
