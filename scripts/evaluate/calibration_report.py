"""
Calibration Analysis and Expected Calibration Error (ECE) Report
Generates calibration plots and detailed ECE analysis for TG-MVMT-GFNet v2.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import config as cfg
from src.checkpoint import load_model_bundle
from src.data_pipeline import get_loaders
from src.metrics import expected_calibration_error

os.makedirs("outputs", exist_ok=True)


def compute_calibration_curve(y_true, y_prob, n_bins=10):
    """
    Compute calibration curve: bin predictions and compute accuracy vs confidence.
    Returns: bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob < hi)

        if np.sum(in_bin) == 0:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)
        else:
            bin_accuracies.append(float(np.mean(y_true[in_bin])))
            bin_confidences.append(float(np.mean(y_prob[in_bin])))
            bin_counts.append(int(np.sum(in_bin)))

    return bin_centers, np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


def plot_calibration_curves(all_targets, all_probs, all_masks):
    """
    Plot calibration curves for all tasks in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for k, task in enumerate(cfg.TASK_NAMES):
        ax = axes[k]
        observed = all_masks[:, k].astype(bool)
        y_true = all_targets[observed, k]
        y_prob = all_probs[observed, k]

        # Compute calibration curve
        bin_centers, bin_accs, bin_confs, bin_counts = compute_calibration_curve(y_true, y_prob, n_bins=10)

        # Plot
        valid = ~np.isnan(bin_accs)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label="Perfect Calibration")
        ax.plot(bin_confs[valid], bin_accs[valid], 'o-', linewidth=2, markersize=8,
                color="steelblue", label="Model Calibration")

        # Add bin counts as annotations
        for i in range(len(bin_centers)):
            if valid[i] and bin_counts[i] > 0:
                ax.text(bin_confs[i], bin_accs[i] + 0.03, f"n={bin_counts[i]}",
                       ha="center", fontsize=7, color="gray")

        # Compute ECE
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Probability (Confidence)", fontsize=10)
        ax.set_ylabel("Observed Frequency (Accuracy)", fontsize=10)
        ax.set_title(f"{task.replace('_', ' ').title()}\nECE = {ece:.4f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    plt.suptitle("Calibration Curves for TG-MVMT-GFNet v2", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "outputs/calibration_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_reliability_diagram(all_targets, all_probs, all_masks):
    """
    Plot reliability diagram (calibration histogram) for all tasks.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for k, task in enumerate(cfg.TASK_NAMES):
        ax = axes[k]
        observed = all_masks[:, k].astype(bool)
        y_true = all_targets[observed, k]
        y_prob = all_probs[observed, k]

        # Compute calibration curve
        bin_centers, bin_accs, bin_confs, bin_counts = compute_calibration_curve(y_true, y_prob, n_bins=10)

        # Plot bars
        valid = ~np.isnan(bin_accs)
        width = 0.08
        colors = ["green" if abs(bin_accs[i] - bin_confs[i]) < 0.1 else "orange" if abs(bin_accs[i] - bin_confs[i]) < 0.2 else "red"
                  for i in range(len(bin_centers))]

        for i in range(len(bin_centers)):
            if valid[i]:
                ax.bar(bin_confs[i], bin_accs[i], width=width, color=colors[i], alpha=0.7, edgecolor="black")
                # Add gap indicator
                gap = bin_accs[i] - bin_confs[i]
                if abs(gap) > 0.05:
                    ax.plot([bin_confs[i], bin_confs[i]], [bin_confs[i], bin_accs[i]],
                           'k-', linewidth=2, alpha=0.5)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label="Perfect Calibration")

        # Compute ECE
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Probability", fontsize=10)
        ax.set_ylabel("Observed Frequency", fontsize=10)
        ax.set_title(f"{task.replace('_', ' ').title()}\nECE = {ece:.4f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    plt.suptitle("Reliability Diagrams for TG-MVMT-GFNet v2\n(Green = well-calibrated, Orange = moderate, Red = poor)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "outputs/reliability_diagrams.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def generate_ece_report(all_targets, all_probs, all_masks):
    """
    Generate detailed ECE report with interpretation.
    """
    print("\n" + "="*80)
    print("  EXPECTED CALIBRATION ERROR (ECE) REPORT")
    print("="*80)

    ece_values = []
    for k, task in enumerate(cfg.TASK_NAMES):
        observed = all_masks[:, k].astype(bool)
        y_true = all_targets[observed, k]
        y_prob = all_probs[observed, k]

        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        ece_values.append(ece)

        # Interpretation
        if ece < 0.05:
            interpretation = "Excellent (well-calibrated)"
        elif ece < 0.10:
            interpretation = "Good (acceptable calibration)"
        elif ece < 0.15:
            interpretation = "Moderate (some miscalibration)"
        else:
            interpretation = "Poor (significant miscalibration)"

        print(f"\n{task.replace('_', ' ').title()}")
        print(f"  ECE: {ece:.4f} - {interpretation}")

        # Compute per-bin details
        bin_centers, bin_accs, bin_confs, bin_counts = compute_calibration_curve(y_true, y_prob, n_bins=10)
        print(f"  Per-bin calibration:")
        print(f"    {'Bin':<10} {'Confidence':>12} {'Accuracy':>12} {'Gap':>10} {'Count':>8}")
        print(f"    {'-'*60}")
        for i in range(len(bin_centers)):
            if not np.isnan(bin_accs[i]):
                gap = bin_accs[i] - bin_confs[i]
                print(f"    {f'[{bin_centers[i]-.05:.2f}, {bin_centers[i]+.05:.2f})':<10} "
                      f"{bin_confs[i]:>12.4f} {bin_accs[i]:>12.4f} {gap:>+10.4f} {bin_counts[i]:>8}")

    # Macro average
    macro_ece = np.mean(ece_values)
    print(f"\n{'='*80}")
    print(f"Macro Average ECE: {macro_ece:.4f}")

    if macro_ece < 0.05:
        print("Overall Assessment: EXCELLENT - Model probabilities are well-calibrated")
    elif macro_ece < 0.10:
        print("Overall Assessment: GOOD - Model probabilities are acceptably calibrated")
    elif macro_ece < 0.15:
        print("Overall Assessment: MODERATE - Some miscalibration present")
    else:
        print("Overall Assessment: POOR - Significant miscalibration, consider recalibration")

    print("="*80)

    return ece_values, macro_ece


def compare_with_baselines():
    """
    Compare ECE with baseline models (from previous runs).
    """
    print("\n" + "="*80)
    print("  ECE COMPARISON WITH BASELINES")
    print("="*80)

    # These values are from the baseline runs
    baseline_eces = {
        "Decision Tree": 0.0172,
        "Logistic Regression": 0.0113,
        "Random Forest": 0.0693,
        "XGBoost": 0.0818,
        "LightGBM": 0.0221,
        "Early-Fusion MLP": 0.0151,
        "Single-Task MLP": 0.0215,
    }

    print(f"\n{'Model':<25} {'Macro ECE':>12} {'Calibration Quality':>25}")
    print("-"*80)

    for model, ece in sorted(baseline_eces.items(), key=lambda x: x[1]):
        quality = "Excellent" if ece < 0.05 else "Good" if ece < 0.10 else "Moderate" if ece < 0.15 else "Poor"
        print(f"{model:<25} {ece:>12.4f} {quality:>25}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Load model and data
    bundle = load_model_bundle()
    train_df, val_df, test_df = bundle["train_df"], bundle["val_df"], bundle["test_df"]
    vocab = bundle["vocab"]
    _, _, test_loader = get_loaders(train_df, val_df, test_df, vocab)
    model = bundle["model"]

    # Collect predictions
    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for views, targets, masks in test_loader:
            preds, _, _ = model(views)
            probs = torch.stack(preds, dim=1).numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_masks = np.vstack(all_masks)

    # Generate ECE report
    ece_values, macro_ece = generate_ece_report(all_targets, all_probs, all_masks)

    # Plot calibration curves
    plot_calibration_curves(all_targets, all_probs, all_masks)
    plot_reliability_diagram(all_targets, all_probs, all_masks)

    # Compare with baselines
    compare_with_baselines()

    print(f"\nTG-MVMT-GFNet v2 Macro ECE: {macro_ece:.4f}")
    print("\n✓ Calibration analysis complete.")
