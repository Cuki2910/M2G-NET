"""
Phase 3: Leave-Site-Out (Leave-Intersection-Out) Validation.

Wraps the existing evaluate_leave_intersection_out() from scripts/train.py.
Trains a fresh model per fold (31 folds), reports per-site and macro AUC.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from scripts.train import evaluate_leave_intersection_out

os.makedirs("outputs", exist_ok=True)


def _plot_per_site_auc(fold_aucs, mean_auc, std_auc, save_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(fold_aucs))
    bars = ax.bar(x, fold_aucs, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(mean_auc, color="red", linewidth=2, linestyle="--", label=f"Mean AUC = {mean_auc:.4f}")
    ax.axhline(mean_auc - std_auc, color="orange", linewidth=1, linestyle=":", label=f"-1 SD = {mean_auc - std_auc:.4f}")
    ax.axhline(0.65, color="gray", linewidth=1, linestyle="-.", alpha=0.7, label="Threshold = 0.65")

    ax.set_xlabel("Fold (held-out site index)", fontsize=11)
    ax.set_ylabel("Macro ROC-AUC", fontsize=11)
    ax.set_title("Leave-Site-Out Validation: Per-Fold Macro ROC-AUC", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("Loading raw data...")
    raw_df = pd.read_csv(cfg.DATA_PATH)
    n_sites = raw_df[cfg.SITE_ID_COL].nunique()
    print(f"Total sites in training data: {n_sites}")
    print(f"Starting leave-site-out validation ({n_sites} folds)...")
    print("Note: each fold trains a fresh model with early stopping (max 50 epochs).\n")

    results = evaluate_leave_intersection_out(raw_df, max_epochs=50, patience=10, min_test_size=5)

    mean_auc = results["macro_roc_auc_mean"]
    std_auc  = results["macro_roc_auc_std"]
    n_folds  = results["n_folds"]
    fold_aucs = results["fold_aucs"]

    print("\n" + "=" * 60)
    print("  LEAVE-SITE-OUT VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Folds completed : {n_folds}")
    print(f"  Mean AUC        : {mean_auc:.4f}")
    print(f"  Std AUC         : {std_auc:.4f}")
    print(f"  Min AUC (worst) : {min(fold_aucs):.4f}")
    print(f"  Max AUC (best)  : {max(fold_aucs):.4f}")
    print("=" * 60)

    assessment = "Good" if mean_auc >= 0.70 else "Acceptable" if mean_auc >= 0.65 else "Poor"
    print(f"\nAssessment: {assessment}")
    print(f"Comparison: random-split test AUC = 0.7115")
    print(f"Leave-site-out mean AUC = {mean_auc:.4f} (gap = {0.7115 - mean_auc:+.4f})")

    csv_path = "outputs/leave_site_out_results.csv"
    pd.DataFrame({"fold": range(n_folds), "macro_roc_auc": fold_aucs}).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    json_path = "outputs/leave_site_out_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    _plot_per_site_auc(fold_aucs, mean_auc, std_auc, "outputs/leave_site_out.png")

    print("\nDone: Leave-site-out validation complete.")
    return results


if __name__ == "__main__":
    main()
