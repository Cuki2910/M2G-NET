import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as cfg
from src.checkpoint import load_model_bundle
from src.data_pipeline import RiderDataset
from src.metrics import compute_all_metrics


os.makedirs("outputs", exist_ok=True)


def collect_predictions(model, loader):
    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for views, targets, masks in loader:
            preds, _, _ = model(views)
            all_probs.append(torch.stack(preds, dim=1).numpy())
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())
    return np.vstack(all_targets), np.vstack(all_probs), np.vstack(all_masks)


def subgroup_rows(df, y, p, m, column, encoders, thresholds=None):
    rows = []
    values = sorted(df[column].dropna().unique())
    for value in values:
        mask = (df[column].to_numpy() == value)
        if mask.sum() == 0:
            continue

        label = value
        if column in encoders:
            label = encoders[column].inverse_transform([int(value)])[0]

        metrics = compute_all_metrics(y[mask], p[mask], m[mask], thresholds=thresholds)
        for task in cfg.TASK_NAMES + ["macro"]:
            row = {
                "group_variable": column,
                "group_value": label,
                "task": task,
                "n": int(mask.sum()),
            }
            row.update(metrics[task])
            rows.append(row)
    return rows


def main():
    bundle = load_model_bundle()
    test_df = bundle["test_df"]
    encoders = bundle["encoders"]
    vocab = bundle["vocab"]
    thresholds = bundle["thresholds"]
    test_loader = DataLoader(RiderDataset(test_df, vocab), batch_size=cfg.BATCH_SIZE, shuffle=False)
    model = bundle["model"]

    y, p, m = collect_predictions(model, test_loader)

    rows = []
    for column in ["gender", "age_group"]:
        if column in test_df.columns:
            rows.extend(subgroup_rows(test_df, y, p, m, column, encoders, thresholds=thresholds))

    result = pd.DataFrame(rows)
    out = "outputs/fairness_calibration_by_group.csv"
    result.to_csv(out, index=False)

    overall = compute_all_metrics(y, p, m, thresholds=thresholds)
    overall_rows = []
    for task in cfg.TASK_NAMES + ["macro"]:
        row = {"task": task}
        row.update(overall[task])
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows)
    overall_out = "outputs/calibration_overall.csv"
    overall_df.to_csv(overall_out, index=False)

    print("\n=== Overall Calibration / Performance ===")
    cols = ["task", "roc_auc", "pr_auc", "f1", "balanced_acc", "brier", "ece", "sensitivity", "specificity"]
    print(overall_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {overall_out}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
