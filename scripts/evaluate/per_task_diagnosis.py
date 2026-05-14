"""
Per-task diagnosis: compare M2G-Net v2 vs CatBoost task-by-task.
Runs N seeds, reports per-task ROC-AUC / PR-AUC / F1.
Identifies tasks where M2G leads or lags, and measures seed stability.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import config as cfg
from src.data_pipeline import load_data, RiderDataset
from scripts.train import fit_model_for_split
from baselines.run_all_baselines import get_flat_features, get_targets, get_task_masks

OUTPUT_JSON = "outputs/per_task_diagnosis.json"
OUTPUT_MD   = "docs/PER_TASK_DIAGNOSIS.md"
SEEDS       = [42, 101, 2023, 777, 999]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_m2g_probs(model, loader):
    all_probs, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for views, targets, _ in loader:
            preds, _, _ = model(views)
            all_probs.append(torch.stack(preds, dim=1).numpy())
            all_targets.append(targets.numpy())
    return np.vstack(all_targets), np.vstack(all_probs)


def collect_catboost_probs(train_df, val_df, test_df, seed):
    trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    X_train = get_flat_features(trainval_df)
    X_test  = get_flat_features(test_df)
    Y_train = get_targets(trainval_df)
    Y_test  = get_targets(test_df)
    M_train = get_task_masks(trainval_df)

    all_probs = np.zeros((len(X_test), cfg.NUM_TASKS))
    for k in range(cfg.NUM_TASKS):
        observed = M_train[:, k].astype(bool)
        clf = CatBoostClassifier(
            iterations=100, learning_rate=0.05, depth=6,
            loss_function="Logloss", random_seed=seed,
            verbose=False, allow_writing_files=False,
        )
        clf.fit(X_train[observed], Y_train[observed, k])
        all_probs[:, k] = clf.predict_proba(X_test)[:, 1]
    return Y_test, all_probs


def per_task_metrics(y_true, y_probs):
    result = {}
    for k, task in enumerate(cfg.TASK_NAMES):
        yt = y_true[:, k]
        yp = y_probs[:, k]
        try:
            roc = roc_auc_score(yt, yp)
        except Exception:
            roc = float("nan")
        try:
            pr = average_precision_score(yt, yp)
        except Exception:
            pr = float("nan")
        f1 = f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
        result[task] = {"roc_auc": roc, "pr_auc": pr, "f1": f1}
    return result


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    task_scores = {
        task: {"m2g": {"roc_auc": [], "pr_auc": [], "f1": []},
               "catboost": {"roc_auc": [], "pr_auc": [], "f1": []}}
        for task in cfg.TASK_NAMES
    }
    macro_scores = {"m2g": [], "catboost": []}

    for i, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({i+1}/{len(SEEDS)}) ---")
        set_seed(seed)
        train_df, val_df, test_df, encoders, vocab = load_data(seed=seed)

        print("  Training M2G-Net...")
        model, _, _, _ = fit_model_for_split(vocab, train_df, val_df, encoders=encoders)
        test_loader = torch.utils.data.DataLoader(
            RiderDataset(test_df, vocab), batch_size=cfg.BATCH_SIZE, shuffle=False,
        )
        m2g_targets, m2g_probs = collect_m2g_probs(model, test_loader)
        m2g_m = per_task_metrics(m2g_targets, m2g_probs)

        print("  Training CatBoost...")
        cb_targets, cb_probs = collect_catboost_probs(train_df, val_df, test_df, seed)
        cb_m = per_task_metrics(cb_targets, cb_probs)

        m2g_macro = float(np.nanmean([m2g_m[t]["roc_auc"] for t in cfg.TASK_NAMES]))
        cb_macro  = float(np.nanmean([cb_m[t]["roc_auc"]  for t in cfg.TASK_NAMES]))
        macro_scores["m2g"].append(m2g_macro)
        macro_scores["catboost"].append(cb_macro)

        for task in cfg.TASK_NAMES:
            for metric in ["roc_auc", "pr_auc", "f1"]:
                task_scores[task]["m2g"][metric].append(m2g_m[task][metric])
                task_scores[task]["catboost"][metric].append(cb_m[task][metric])

        print(f"  Macro: M2G={m2g_macro:.4f} | CB={cb_macro:.4f}")
        print("  Per-task ROC-AUC (M2G vs CatBoost):")
        for task in cfg.TASK_NAMES:
            m = m2g_m[task]["roc_auc"]
            c = cb_m[task]["roc_auc"]
            winner = "M2G" if m > c + 0.001 else ("CB" if c > m + 0.001 else "TIE")
            print(f"    {task:<28}: M2G={m:.4f}  CB={c:.4f}  [{winner}]")

    # Aggregate summary
    print("\n\n=== PER-TASK DIAGNOSIS SUMMARY (mean over 5 seeds) ===")
    print(f"\n{'Task':<30} {'M2G AUC':>10} {'CB AUC':>10} {'Delta':>8} {'Winner':>10}")
    print("-" * 72)

    summary = {}
    for task in cfg.TASK_NAMES:
        ts = task_scores[task]
        summary[task] = {}
        for model_key in ["m2g", "catboost"]:
            summary[task][model_key] = {}
            for metric in ["roc_auc", "pr_auc", "f1"]:
                vals = [v for v in ts[model_key][metric] if np.isfinite(v)]
                summary[task][model_key][metric] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
        m_mean = summary[task]["m2g"]["roc_auc"]["mean"]
        c_mean = summary[task]["catboost"]["roc_auc"]["mean"]
        delta  = m_mean - c_mean
        winner = "M2G" if delta > 0.001 else ("CatBoost" if delta < -0.001 else "TIE")
        print(f"{task:<30} {m_mean:>10.4f} {c_mean:>10.4f} {delta:>+8.4f} {winner:>10}")

    print("\n  Macro (mean across tasks and seeds):")
    m2g_overall  = float(np.mean(macro_scores["m2g"]))
    cb_overall   = float(np.mean(macro_scores["catboost"]))
    print(f"    M2G-Net: {m2g_overall:.4f}   CatBoost: {cb_overall:.4f}   Delta: {m2g_overall - cb_overall:+.4f}")

    results = {
        "seeds": SEEDS,
        "task_scores": task_scores,
        "summary": summary,
        "macro": {"m2g": macro_scores["m2g"], "catboost": macro_scores["catboost"]},
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    _write_markdown(summary, macro_scores)
    print(f"Saved: {OUTPUT_MD}")


def _write_markdown(summary, macro_scores):
    m2g_macro_mean = float(np.mean(macro_scores["m2g"]))
    cb_macro_mean  = float(np.mean(macro_scores["catboost"]))

    lines = [
        "# Per-Task Diagnosis: M2G-Net v2 vs CatBoost",
        "",
        "**Data status:** synthetic proof-of-concept.",
        f"**Seeds:** {SEEDS}",
        "",
        "## Macro ROC-AUC (mean over 5 seeds)",
        "",
        f"| Model | Mean macro ROC-AUC |",
        "|---|---:|",
        f"| M2G-Net v2 | {m2g_macro_mean:.4f} |",
        f"| CatBoost   | {cb_macro_mean:.4f} |",
        "",
        "## ROC-AUC per task (mean +/- std, 5 seeds)",
        "",
        "| Task | M2G-Net | CatBoost | Delta | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for task in cfg.TASK_NAMES:
        m = summary[task]["m2g"]["roc_auc"]
        c = summary[task]["catboost"]["roc_auc"]
        delta  = m["mean"] - c["mean"]
        winner = "M2G" if delta > 0.001 else ("CatBoost" if delta < -0.001 else "TIE")
        display = cfg.TASK_DISPLAY_NAMES.get(task, task)
        lines.append(
            f"| {display} | {m['mean']:.4f} +/- {m['std']:.4f} | "
            f"{c['mean']:.4f} +/- {c['std']:.4f} | {delta:+.4f} | {winner} |"
        )

    lines.extend([
        "",
        "## PR-AUC per task (mean, 5 seeds)",
        "",
        "| Task | M2G-Net | CatBoost | Delta |",
        "|---|---:|---:|---:|",
    ])
    for task in cfg.TASK_NAMES:
        m = summary[task]["m2g"]["pr_auc"]
        c = summary[task]["catboost"]["pr_auc"]
        delta = m["mean"] - c["mean"]
        display = cfg.TASK_DISPLAY_NAMES.get(task, task)
        lines.append(
            f"| {display} | {m['mean']:.4f} | {c['mean']:.4f} | {delta:+.4f} |"
        )

    lines.extend([
        "",
        "## F1 per task (threshold=0.5, mean, 5 seeds)",
        "",
        "| Task | M2G-Net | CatBoost | Delta |",
        "|---|---:|---:|---:|",
    ])
    for task in cfg.TASK_NAMES:
        m = summary[task]["m2g"]["f1"]
        c = summary[task]["catboost"]["f1"]
        delta = m["mean"] - c["mean"]
        display = cfg.TASK_DISPLAY_NAMES.get(task, task)
        lines.append(
            f"| {display} | {m['mean']:.4f} | {c['mean']:.4f} | {delta:+.4f} |"
        )

    lines.extend([
        "",
        "## Stability (ROC-AUC std dev across seeds)",
        "",
        "| Task | M2G std | CatBoost std | More stable |",
        "|---|---:|---:|---|",
    ])
    for task in cfg.TASK_NAMES:
        ms = summary[task]["m2g"]["roc_auc"]["std"]
        cs = summary[task]["catboost"]["roc_auc"]["std"]
        more_stable = "M2G" if ms < cs - 0.001 else ("CatBoost" if cs < ms - 0.001 else "Similar")
        display = cfg.TASK_DISPLAY_NAMES.get(task, task)
        lines.append(f"| {display} | {ms:.4f} | {cs:.4f} | {more_stable} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- M2G-Net trained with `fit_model_for_split` (same early stopping as main training).",
        "- CatBoost uses 100 iterations, lr=0.05, depth=6, Logloss.",
        "- Delta = M2G mean - CatBoost mean; positive = M2G leads.",
        "- Winner threshold: |delta| > 0.001 (within threshold = TIE).",
    ])

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
