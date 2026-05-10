import sys
sys.path.insert(0, "..")

"""
Test model on independent test set
Evaluates TG-MVMT-GFNet v2 on the configured independent test CSV.
When the repository uses synthetic CSVs, this should be interpreted as a
synthetic out-of-site test rather than external real-world validation.
"""

import sys, os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import torch

import config as cfg
from src.data_pipeline import RiderDataset
from src.model import TGMVMTGFNetV2
from src.loss import UncertaintyWeightedLoss
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


def build_vocab_for_test(test_df, train_vocab):
    """
    Build vocab for test set, handling unseen intersections.
    For unseen site_ids, we'll map them to the last training site (31).
    """
    vocab = train_vocab.copy()

    # Check for unseen intersections
    test_sites = set(test_df['intersection_id'].unique())
    train_sites = set(range(1, 32))  # Training had sites 1-31
    unseen_sites = test_sites - train_sites

    print(f"\n[Site Analysis]")
    print(f"Training sites: 1-31 ({len(train_sites)} sites)")
    print(f"Test sites: {min(test_sites)}-{max(test_sites)} ({len(test_sites)} sites)")
    print(f"Unseen sites: {len(unseen_sites)} sites")

    if unseen_sites:
        print(f"[WARNING] Mapping unseen sites to site 31 (will use site_obs only)")

    return vocab


def encode_test_data(test_df, train_encoders):
    """
    Encode test data using training encoders.
    Map unseen intersection_ids to 30 (last training site index, 0-indexed).
    """
    df = test_df.copy()

    # Use training encoders
    for col, encoder in train_encoders.items():
        if col == 'intersection_id':
            continue  # Handle separately
        df[col] = encoder.transform(df[col])

    # Handle intersection_id: map unseen (32-50) to 30 (last training site, 0-indexed)
    # Training sites: 1-31 → encoded as 0-30
    df['intersection_id'] = df['intersection_id'].apply(
        lambda x: 30 if x > 31 else (x - 1)  # Map 32-50 → 30, map 1-31 → 0-30
    )

    return df


def test_on_independent_set():
    """Main testing function."""
    print("="*70)
    print("Testing TG-MVMT-GFNet v2 on Configured Test Set")
    print("="*70)

    # Load model and training data info
    from src.data_pipeline import load_data
    train_df, val_df, test_df_orig, encoders, vocab = load_data()

    model = TGMVMTGFNetV2(vocab)
    loss_fn = UncertaintyWeightedLoss()

    ckpt = torch.load(cfg.CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    loss_fn.load_state_dict(ckpt["loss_state"])
    model.eval()

    print(f"Loaded model from: {cfg.CHECKPOINT_PATH}")
    print(f"Model trained on sites: 1-31")

    # Load independent test set
    test_df = load_independent_test_set()
    test_vocab = build_vocab_for_test(test_df, vocab)

    # Encode test data using training encoders
    test_df_encoded = encode_test_data(test_df, encoders)

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
    metrics = compute_all_metrics(all_targets, all_probs, all_masks)
    print_results(metrics, title="Independent Test Set Results")

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
