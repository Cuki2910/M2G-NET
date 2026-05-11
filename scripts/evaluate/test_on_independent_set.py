import sys
sys.path.insert(0, "..")

"""
Test model on independent test set
Evaluates M2G-Net v2 on the configured independent test CSV.
When the repository uses synthetic CSVs, this should be interpreted as a
synthetic out-of-site test rather than external real-world validation.
"""

import sys, os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import torch

import config as cfg
from src.checkpoint import load_model_bundle
from src.data_pipeline import RiderDataset, encode_with_preprocessing
from src.metrics import compute_all_metrics, print_results
from torch.utils.data import DataLoader


def load_independent_test_set():
    """Load the independent test set."""
    test_path = "data/test/independent_test_set.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Independent test set not found at {test_path}")

    df = pd.read_csv(test_path)
    print(f"Loaded independent test set: {len(df)} samples")
    print(f"Intersection IDs: {df['intersection_id'].min()} - {df['intersection_id'].max()}")
    return df


def report_site_coverage(test_df, train_vocab):
    """
    Report independent-site coverage using the train-time site mapping.
    """
    test_sites = set(test_df['intersection_id'].unique())
    train_sites = set(train_vocab.get("site_mapping", {}).keys())
    unseen_sites = test_sites - train_sites

    print(f"\n[Site Analysis]")
    if train_sites:
        print(f"Training sites: {min(train_sites)}-{max(train_sites)} ({len(train_sites)} sites)")
    else:
        print("Training site mapping unavailable in vocab")
    print(f"Test sites: {min(test_sites)}-{max(test_sites)} ({len(test_sites)} sites)")
    print(f"Unseen sites: {len(unseen_sites)} sites")

    if unseen_sites:
        print("[INFO] Unseen sites use the reserved unknown-site id; evaluation disables site intercept.")


def test_on_independent_set():
    """Main testing function."""
    print("="*70)
    print("Testing M2G-Net v2 on Configured Test Set")
    print("="*70)

    # Load model and train-time preprocessing from checkpoint metadata.
    bundle = load_model_bundle()
    encoders = bundle["encoders"]
    vocab = bundle["vocab"]
    model = bundle["model"]
    thresholds = bundle["thresholds"]

    print(f"Loaded model from: {cfg.CHECKPOINT_PATH}")
    train_sites = sorted(vocab.get("site_mapping", {}).keys())
    if train_sites:
        print(f"Model trained on sites: {train_sites[0]}-{train_sites[-1]}")

    # Load independent test set
    test_df = load_independent_test_set()
    report_site_coverage(test_df, vocab)

    # Encode test data using checkpoint preprocessing.
    test_df_encoded = encode_with_preprocessing(test_df, encoders, vocab["site_mapping"])

    # Create dataset and loader
    test_dataset = RiderDataset(test_df_encoded, vocab)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Evaluate with site intercept OFF (since all sites are unseen)
    print("\n[Evaluation Mode: Site intercept OFF for held-out/synthetic site IDs]")
    all_probs, all_targets, all_masks = [], [], []

    with torch.no_grad():
        for views, targets, masks in test_loader:
            # use_site_intercept=False for unseen sites
            preds, _, _ = model(views, use_site_intercept=False)
            probs = torch.stack(preds, dim=1).numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_masks = np.vstack(all_masks)

    # Compute metrics
    metrics_default = compute_all_metrics(all_targets, all_probs, all_masks)
    print_results(metrics_default, title="Independent Test Set Results (threshold=0.5)")

    metrics = compute_all_metrics(all_targets, all_probs, all_masks, thresholds=thresholds)
    print_results(metrics, title="Independent Test Set Results (validation-tuned thresholds)")

    # Compare with original test set performance
    print("\n" + "="*70)
    print("Comparison with Original Test Set")
    print("="*70)
    print(f"{'Metric':<25} {'Original Test':>15} {'Independent Test':>18} {'Difference':>12}")
    print("-"*70)

    # Original test results (from tracking.md)
    original = {
        "macro": 0.7107,
        "red_light_running": 0.6784,
        "no_turn_signal": 0.6199,
        "helmet_nonuse": 0.7886,
        "mobile_phone_use": 0.7558,
    }

    for task in ["macro"] + cfg.TASK_NAMES:
        orig_auc = original.get(task, 0.0)
        new_auc = metrics[task]["roc_auc"]
        diff = new_auc - orig_auc
        sign = "+" if diff >= 0 else ""
        print(f"{task:<25} {orig_auc:>15.4f} {new_auc:>18.4f} {sign}{diff:>11.4f}")

    print("\n[Interpretation]")
    print("- With the default synthetic CSVs, this is a synthetic out-of-site test.")
    print("- Use `scripts/evaluate/repeated_runs_significance.py` to check if differences are statistically significant.")
    print("- Use real held-out intersections before making real-world generalization claims.")

    return metrics


if __name__ == "__main__":
    metrics = test_on_independent_set()
