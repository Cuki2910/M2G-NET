"""
Phase 1 (Calibration Fix): Post-hoc temperature scaling.

Fits a single temperature T on the validation set, then reports ECE
before and after scaling on the test set. Saves T to outputs/.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from src.checkpoint import load_model_bundle
from src.data_pipeline import get_loaders
from src.metrics import expected_calibration_error, safe_nanmean
from src.calibration import TemperatureScaler

os.makedirs("outputs", exist_ok=True)


def _collect_predictions(model, loader):
    """Return (probs, targets, masks) as numpy arrays (N, num_tasks)."""
    all_probs, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for views, targets, masks in loader:
            preds, _, _ = model(views)
            probs = torch.stack(preds, dim=1).numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())
    return (
        np.vstack(all_probs),
        np.vstack(all_targets),
        np.vstack(all_masks),
    )


def _compute_ece_per_task(probs, targets, masks):
    """Return list of per-task ECE and macro average."""
    ece_list = []
    for k, task in enumerate(cfg.TASK_NAMES):
        observed = masks[:, k].astype(bool)
        ece = expected_calibration_error(targets[observed, k], probs[observed, k])
        ece_list.append(ece)
    return ece_list, safe_nanmean(ece_list)


def _plot_comparison(before_eces, after_eces, T, save_path):
    tasks = [t.replace("_", " ").title() for t in cfg.TASK_NAMES]
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_before = ax.bar(x - width / 2, before_eces, width, label="Before scaling", color="salmon", alpha=0.85)
    bars_after = ax.bar(x + width / 2, after_eces, width, label=f"After scaling (T={T:.3f})", color="steelblue", alpha=0.85)

    ax.axhline(0.05, color="green", linestyle="--", linewidth=1, label="ECE=0.05 (Excellent)")
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=1, label="ECE=0.10 (Good)")

    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("ECE (lower is better)", fontsize=11)
    ax.set_title("Calibration: Before vs After Temperature Scaling", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars_before:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_after:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("Loading model bundle...")
    bundle = load_model_bundle()
    model = bundle["model"]
    train_df, val_df, test_df = bundle["train_df"], bundle["val_df"], bundle["test_df"]
    vocab = bundle["vocab"]

    _, val_loader, test_loader = get_loaders(train_df, val_df, test_df, vocab)

    # ── Collect predictions ─────────────────────────────────────────────────────
    print("Collecting validation predictions for fitting T...")
    val_probs, val_targets, val_masks = _collect_predictions(model, val_loader)

    print("Collecting test predictions for evaluation...")
    test_probs, test_targets, test_masks = _collect_predictions(model, test_loader)

    # ── ECE before scaling ──────────────────────────────────────────────────────
    before_eces, before_macro = _compute_ece_per_task(test_probs, test_targets, test_masks)

    # ── Fit temperature on validation set ──────────────────────────────────────
    # Pre-filter using masks so unobserved tasks don't contaminate the fit
    val_obs_mask = val_masks.astype(bool)
    val_probs_fit = np.where(val_obs_mask, val_probs, np.nan)
    val_targets_fit = np.where(val_obs_mask, val_targets, np.nan)

    scaler = TemperatureScaler().fit(val_probs_fit, val_targets_fit)
    print(f"\nLearned temperature T = {scaler.temperature:.4f}")

    # ── Apply scaling and re-evaluate ───────────────────────────────────────────
    test_probs_scaled = scaler.transform(test_probs)
    after_eces, after_macro = _compute_ece_per_task(test_probs_scaled, test_targets, test_masks)

    # ── Report ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CALIBRATION: BEFORE vs AFTER TEMPERATURE SCALING")
    print("=" * 65)
    print(f"{'Task':<28} {'ECE Before':>12} {'ECE After':>12} {'Delta ECE':>12}")
    print("-" * 65)
    for task, eb, ea in zip(cfg.TASK_NAMES, before_eces, after_eces):
        delta = ea - eb
        print(f"{task:<28} {eb:>12.4f} {ea:>12.4f} {delta:>+12.4f}")
    print("-" * 65)
    print(f"{'Macro Average':<28} {before_macro:>12.4f} {after_macro:>12.4f} {after_macro - before_macro:>+12.4f}")
    print("=" * 65)

    quality_before = "Excellent" if before_macro < 0.05 else "Good" if before_macro < 0.10 else "Moderate" if before_macro < 0.15 else "Poor"
    quality_after  = "Excellent" if after_macro  < 0.05 else "Good" if after_macro  < 0.10 else "Moderate" if after_macro  < 0.15 else "Poor"
    print(f"\nBefore: {quality_before}  |  After: {quality_after}")
    print(f"Temperature T = {scaler.temperature:.4f}  (T>1 means model was overconfident)")

    # ── Save outputs ────────────────────────────────────────────────────────────
    t_path = "outputs/temperature_T.json"
    scaler.save(t_path)
    print(f"\nSaved: {t_path}")

    results = {
        "temperature": scaler.temperature,
        "ece_before": {task: float(e) for task, e in zip(cfg.TASK_NAMES, before_eces)},
        "ece_after": {task: float(e) for task, e in zip(cfg.TASK_NAMES, after_eces)},
        "macro_ece_before": float(before_macro),
        "macro_ece_after": float(after_macro),
    }
    results_path = "outputs/temperature_scaling_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    _plot_comparison(before_eces, after_eces, scaler.temperature, "outputs/calibration_comparison.png")

    print("\nDone: Temperature scaling complete.")
    return results


if __name__ == "__main__":
    main()
