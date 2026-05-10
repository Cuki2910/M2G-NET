"""
Phase 7: Metrics
ROC-AUC, PR-AUC, F1, Balanced Accuracy, and MTL Transfer Ratio.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix,
)
import config as cfg


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob < hi)
        if not np.any(in_bin):
            continue
        conf = float(np.mean(y_prob[in_bin]))
        acc = float(np.mean(y_true[in_bin]))
        ece += (np.sum(in_bin) / len(y_true)) * abs(acc - conf)
    return ece


def compute_task_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {}
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics["pr_auc"] = float("nan")
    metrics["f1"]           = f1_score(y_true, y_pred, zero_division=0)
    metrics["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    try:
        metrics["brier"] = brier_score_loss(y_true, y_prob)
    except Exception:
        metrics["brier"] = float("nan")
    try:
        metrics["ece"] = expected_calibration_error(y_true, y_prob)
    except Exception:
        metrics["ece"] = float("nan")

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    except Exception:
        metrics["sensitivity"] = float("nan")
        metrics["specificity"] = float("nan")
    return metrics


def compute_all_metrics(all_targets, all_probs, all_masks=None):
    """
    all_targets: (N, num_tasks) numpy array
    all_probs:   (N, num_tasks) numpy array
    all_masks:   optional (N, num_tasks) numpy array, 1 means observed label
    Returns dict: {task_name: {metric: value}, "macro": {...}}
    """
    if all_masks is None:
        all_masks = np.ones_like(all_targets, dtype=float)

    results = {}
    for k, task in enumerate(cfg.TASK_NAMES):
        observed = all_masks[:, k].astype(bool)
        if observed.sum() == 0:
            results[task] = {
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "f1": float("nan"),
                "balanced_acc": float("nan"),
                "brier": float("nan"),
                "ece": float("nan"),
                "sensitivity": float("nan"),
                "specificity": float("nan"),
            }
        else:
            results[task] = compute_task_metrics(all_targets[observed, k], all_probs[observed, k])

    # Macro average
    results["macro"] = {
        m: np.nanmean([results[t][m] for t in cfg.TASK_NAMES])
        for m in ["roc_auc", "pr_auc", "f1", "balanced_acc", "brier", "ece", "sensitivity", "specificity"]
    }
    return results


def mtl_transfer_ratio(mtl_results, single_results):
    """
    mtl_results, single_results: dicts from compute_all_metrics
    Returns {task: ratio} where ratio = mtl_auc / single_auc
    """
    return {
        task: mtl_results[task]["roc_auc"] / (single_results[task]["roc_auc"] + 1e-9)
        for task in cfg.TASK_NAMES
    }


def print_results(results, title="Results"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    header = (
        f"{'Task':<25} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>7} "
        f"{'Bal.Acc':>9} {'Brier':>8} {'ECE':>8}"
    )
    print(header)
    print("-" * 78)
    for task in cfg.TASK_NAMES + ["macro"]:
        m = results[task]
        print(f"{task:<25} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
              f"{m['f1']:>7.4f} {m['balanced_acc']:>9.4f} "
              f"{m['brier']:>8.4f} {m['ece']:>8.4f}")
