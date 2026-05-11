"""
Sweep gate prior smoothing strength across a few seeds.

The sweep retrains M2G-Net for each lambda/seed pair, stores temporary
checkpoints under outputs/gate_prior_sweep/checkpoints, and writes aggregate
metrics to outputs/gate_prior_sweep/results.csv and summary.json.
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
from scripts.train import train
from src.checkpoint import load_model_bundle
from src.data_pipeline import (
    RiderDataset,
    encode_with_preprocessing,
    load_data,
)
from src.metrics import compute_all_metrics


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def summarize_metrics(prefix, metrics):
    return {
        f"{prefix}_roc_auc": float(metrics["macro"]["roc_auc"]),
        f"{prefix}_pr_auc": float(metrics["macro"]["pr_auc"]),
        f"{prefix}_f1": float(metrics["macro"]["f1"]),
        f"{prefix}_balanced_acc": float(metrics["macro"]["balanced_acc"]),
        f"{prefix}_brier": float(metrics["macro"]["brier"]),
        f"{prefix}_ece": float(metrics["macro"]["ece"]),
    }


def evaluate_independent(checkpoint_path, independent_path):
    if not independent_path or not os.path.exists(independent_path):
        return {}

    bundle = load_model_bundle(checkpoint_path=checkpoint_path)
    raw_df = pd.read_csv(independent_path)
    test_df = encode_with_preprocessing(
        raw_df,
        bundle["encoders"],
        bundle["vocab"]["site_mapping"],
    )
    y, p, m = collect_predictions(
        bundle["model"], test_df, bundle["vocab"], use_site_intercept=False)
    thresholds = bundle["thresholds"]
    default = compute_all_metrics(y, p, m)
    tuned = compute_all_metrics(y, p, m, thresholds=thresholds)
    return {
        **summarize_metrics("ind_default", default),
        **summarize_metrics("ind_tuned", tuned),
    }


def aggregate(rows):
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in {"lambda", "seed", "checkpoint"}]
    grouped = df.groupby("lambda", as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [
        col if stat == "" else f"{col}_{stat}"
        for col, stat in grouped.columns.to_flat_index()
    ]
    return grouped.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2],
                        help="Gate prior smoothing strengths to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 101],
                        help="Random seeds used for train/val/test splits and training.")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--independent", default="data/test/independent_test_set.csv")
    parser.add_argument("--output-dir", default="outputs/gate_prior_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    old_lambda = cfg.GATE_PRIOR_WEIGHT
    old_checkpoint = cfg.CHECKPOINT_PATH
    rows = []
    try:
        for lambda_value in args.lambdas:
            for seed in args.seeds:
                print(f"\n=== lambda={lambda_value:.3f} | seed={seed} ===")
                set_random_seed(seed)
                cfg.GATE_PRIOR_WEIGHT = float(lambda_value)
                cfg.CHECKPOINT_PATH = os.path.join(
                    checkpoint_dir, f"gfnet_lambda_{lambda_value:g}_seed_{seed}.pt")

                train_df, val_df, test_df, encoders, vocab = load_data(seed=seed)
                _, _, test_metrics, _ = train(
                    vocab, train_df, val_df, test_df,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    encoders=encoders,
                    split_seed=seed,
                )

                row = {
                    "lambda": float(lambda_value),
                    "seed": int(seed),
                    "checkpoint": cfg.CHECKPOINT_PATH,
                    **summarize_metrics("test_tuned", test_metrics),
                    **evaluate_independent(cfg.CHECKPOINT_PATH, args.independent),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    os.path.join(args.output_dir, "results.csv"), index=False)
    finally:
        cfg.GATE_PRIOR_WEIGHT = old_lambda
        cfg.CHECKPOINT_PATH = old_checkpoint

    summary = aggregate(rows)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    best_metric = "ind_tuned_roc_auc_mean" if "ind_tuned_roc_auc_mean" in summary.columns else "test_tuned_roc_auc_mean"
    best_row = summary.loc[summary[best_metric].idxmax()].to_dict()
    manifest = {
        "best_metric": best_metric,
        "best": best_row,
        "n_runs": len(rows),
        "lambdas": args.lambdas,
        "seeds": args.seeds,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Gate Prior Sweep Summary ===")
    print(summary.to_string(index=False))
    print(f"\nBest by {best_metric}: lambda={best_row['lambda']}")
    print(f"Saved: {os.path.join(args.output_dir, 'results.csv')}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
