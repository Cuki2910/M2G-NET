"""
Tune per-task decision thresholds on the validation split and persist them to
the checkpoint. This does not retrain the model; it only changes probability ->
class decision cutoffs used by reporting scripts.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as cfg
from src.checkpoint import load_model_bundle
from src.data_pipeline import RiderDataset, encode_with_preprocessing
from src.metrics import compute_all_metrics, print_results, tune_task_thresholds


def collect_predictions(model, df, vocab, use_site_intercept=True):
    loader = DataLoader(RiderDataset(df, vocab), batch_size=cfg.BATCH_SIZE, shuffle=False)
    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for views, targets, masks in loader:
            preds, _, _ = model(views, use_site_intercept=use_site_intercept)
            all_probs.append(torch.stack(preds, dim=1).numpy())
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())
    return np.vstack(all_targets), np.vstack(all_probs), np.vstack(all_masks)


def report_split(name, y, p, m, thresholds):
    default_metrics = compute_all_metrics(y, p, m)
    tuned_metrics = compute_all_metrics(y, p, m, thresholds=thresholds)
    print_results(default_metrics, title=f"{name} Results (threshold=0.5)")
    print_results(tuned_metrics, title=f"{name} Results (validation-tuned thresholds)")
    return {"default": default_metrics, "tuned": tuned_metrics}


def metrics_summary(metrics):
    return {
        "macro_roc_auc": float(metrics["macro"]["roc_auc"]),
        "macro_pr_auc": float(metrics["macro"]["pr_auc"]),
        "macro_f1": float(metrics["macro"]["f1"]),
        "macro_balanced_acc": float(metrics["macro"]["balanced_acc"]),
        "macro_brier": float(metrics["macro"]["brier"]),
        "macro_ece": float(metrics["macro"]["ece"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["f1", "balanced_acc"], default="f1",
                        help="Validation metric used to choose thresholds.")
    parser.add_argument("--checkpoint", default=cfg.CHECKPOINT_PATH,
                        help="Checkpoint to update.")
    parser.add_argument("--no-save", action="store_true",
                        help="Report tuned thresholds without writing them to checkpoint.")
    parser.add_argument("--independent", default="data/test/independent_test_set.csv",
                        help="Optional independent CSV to report after tuning.")
    args = parser.parse_args()

    bundle = load_model_bundle(checkpoint_path=args.checkpoint)
    model = bundle["model"]
    vocab = bundle["vocab"]

    val_y, val_p, val_m = collect_predictions(model, bundle["val_df"], vocab)
    thresholds = tune_task_thresholds(val_y, val_p, val_m, metric=args.metric)

    print("\n=== Tuned Thresholds ===")
    for task in cfg.TASK_NAMES:
        print(f"{task:<25} {thresholds[task]:.3f}")

    test_y, test_p, test_m = collect_predictions(model, bundle["test_df"], vocab)
    reports = {
        "thresholds": thresholds,
        "validation_metric": args.metric,
        "test": report_split("Random Split Test", test_y, test_p, test_m, thresholds),
    }

    if args.independent and os.path.exists(args.independent):
        raw_independent = pd.read_csv(args.independent)
        independent_df = encode_with_preprocessing(
            raw_independent,
            bundle["encoders"],
            vocab["site_mapping"],
        )
        ind_y, ind_p, ind_m = collect_predictions(
            model, independent_df, vocab, use_site_intercept=False)
        reports["independent"] = report_split(
            "Independent Test", ind_y, ind_p, ind_m, thresholds)

    os.makedirs("outputs", exist_ok=True)
    serializable = {
        "thresholds": thresholds,
        "validation_metric": args.metric,
        "test": {
            key: metrics_summary(value)
            for key, value in reports["test"].items()
        },
    }
    if "independent" in reports:
        serializable["independent"] = {
            key: metrics_summary(value)
            for key, value in reports["independent"].items()
        }
    with open("outputs/threshold_tuning.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    if not args.no_save:
        ckpt = dict(bundle["checkpoint"])
        ckpt["thresholds"] = {task: float(thresholds[task]) for task in cfg.TASK_NAMES}
        torch.save(ckpt, args.checkpoint)
        print(f"\nSaved thresholds to {args.checkpoint}")
    else:
        print("\nDid not modify checkpoint (--no-save).")

    print("Saved report to outputs/threshold_tuning.json")


if __name__ == "__main__":
    main()
