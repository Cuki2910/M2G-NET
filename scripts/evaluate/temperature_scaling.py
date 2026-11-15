"""Fit temperature, then thresholds, using validation only; save both together."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import config as cfg
from scripts.train import collect_predictions
from src.checkpoint import load_model_bundle
from src.data_pipeline import get_loaders
from src.calibration import fit_validation_rule, prediction_metrics
from src.metrics import print_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=cfg.CHECKPOINT_PATH)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    bundle = load_model_bundle(args.checkpoint)
    _, validation, test = get_loaders(bundle["train_df"], bundle["val_df"], bundle["test_df"], bundle["vocab"])
    targets, probabilities, masks = collect_predictions(
        bundle["model"], validation, bundle["validation_use_site_intercept"])
    scaler, thresholds = fit_validation_rule(targets, probabilities, masks)
    targets, probabilities, masks = collect_predictions(
        bundle["model"], test, bundle["test_use_site_intercept"])
    reports = prediction_metrics(targets, probabilities, masks, scaler, thresholds)
    for rule, metrics in reports.items():
        print_results(metrics, title=rule)
    if not args.no_save:
        checkpoint = dict(bundle["checkpoint"])
        checkpoint.update(temperature=scaler.temperature, thresholds=thresholds)
        torch.save(checkpoint, args.checkpoint)
    print(json.dumps({"temperature": scaler.temperature, "thresholds": thresholds}, indent=2))
    return reports


if __name__ == "__main__":
    main()
