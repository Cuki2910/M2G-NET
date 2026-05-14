"""
Repeated runs and statistical significance testing.

Runs M2G-Net and strong tabular baselines over matched random seeds, then writes
per-run scores, confidence intervals, and paired tests.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import warnings
import numpy as np
import torch
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

import config as cfg
from src.data_pipeline import load_data
from scripts.train import train
from baselines.run_all_baselines import run_sklearn_baseline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

OUTPUT_JSON = "outputs/repeated_runs_strong_baselines.json"
OUTPUT_CSV = "outputs/repeated_runs_strong_baselines.csv"
OUTPUT_MD = "docs/STRONG_BASELINE_REPEATED_RUNS_RESULTS.md"

BASELINES = {
    "xgboost": (
        XGBClassifier,
        lambda seed: {
            "n_estimators": 100,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": seed,
            "verbosity": 0,
        },
        "XGBoost",
    ),
    "lightgbm": (
        LGBMClassifier,
        lambda seed: {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": seed,
            "verbose": -1,
        },
        "LightGBM",
    ),
    "random_forest": (
        RandomForestClassifier,
        lambda seed: {
            "n_estimators": 100,
            "random_state": seed,
        },
        "Random Forest",
    ),
    "catboost": (
        CatBoostClassifier,
        lambda seed: {
            "iterations": 100,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        },
        "CatBoost",
    ),
}

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        m = float(np.mean(a)) if n else float("nan")
        return m, m, m
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def paired_test(reference, candidate):
    if len(reference) < 2 or len(candidate) < 2:
        return {"t_statistic": float("nan"), "p_value": float("nan")}
    t_stat, p_value = stats.ttest_rel(reference, candidate)
    return {"t_statistic": float(t_stat), "p_value": float(p_value)}


def summarize_scores(scores_by_model):
    summary = {}
    reference = scores_by_model["m2g_net"]
    for model_name, scores in scores_by_model.items():
        mean, lower, upper = compute_ci(scores)
        row = {
            "mean_macro_roc_auc": float(mean),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "scores": [float(score) for score in scores],
        }
        if model_name != "m2g_net":
            test = paired_test(reference, scores)
            row.update(test)
            row["mean_delta_vs_m2g"] = float(np.mean(reference) - np.mean(scores))
        summary[model_name] = row
    return summary


def write_outputs(run_rows, summary, seeds):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    os.makedirs("docs/results", exist_ok=True)

    pd.DataFrame(run_rows).to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seeds,
                "metric": "macro_roc_auc",
                "summary": summary,
                "runs": run_rows,
            },
            f,
            indent=2,
        )

    lines = [
        "# Strong Baseline Repeated Runs Results",
        "",
        "**Updated:** 2026-05-14",
        "**Metric:** Macro ROC-AUC over 5 matched seeds.",
        "**Data status:** synthetic proof-of-concept unless replaced by real observational data.",
        "",
        "## Summary",
        "",
        "| Model | Mean macro ROC-AUC | 95% CI | Delta vs M2G | Paired p-value |",
        "|---|---:|---:|---:|---:|",
    ]
    display_names = {
        "m2g_net": "M2G-Net v2",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "random_forest": "Random Forest",
        "catboost": "CatBoost",
    }
    for model_name in ["m2g_net", "xgboost", "lightgbm", "random_forest", "catboost"]:
        row = summary[model_name]
        delta = "ref" if model_name == "m2g_net" else f"{row['mean_delta_vs_m2g']:+.4f}"
        p_value = "ref" if model_name == "m2g_net" else f"{row['p_value']:.4g}"
        lines.append(
            f"| {display_names[model_name]} | {row['mean_macro_roc_auc']:.4f} | "
            f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] | {delta} | {p_value} |"
        )

    lines.extend([
        "",
        "## Per-Seed Scores",
        "",
        "| Seed | M2G-Net | XGBoost | LightGBM | Random Forest | CatBoost |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for run in run_rows:
        lines.append(
            f"| {run['seed']} | {run['m2g_net']:.4f} | {run['xgboost']:.4f} | "
            f"{run['lightgbm']:.4f} | {run['random_forest']:.4f} | {run['catboost']:.4f} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Each seed uses the same train/validation/test split for M2G-Net and all baselines.",
        "- Baselines use the current fixed hyperparameters in `scripts/evaluate/repeated_runs_significance.py`; these are not yet tuned or nested-CV optimized.",
        "- Paired p-values compare M2G-Net against each baseline over matched seeds.",
    ])
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    n_runs = 5
    seeds = [42, 101, 2023, 777, 999]
    
    scores_by_model = {"m2g_net": []}
    for baseline_name in BASELINES:
        scores_by_model[baseline_name] = []
    run_rows = []
    
    print(f"Starting repeated runs ({n_runs} runs)...")
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{n_runs} (Seed: {seed}) ---")
        set_random_seed(seed)
        train_df, val_df, test_df, encoders, vocab = load_data(seed=seed)
        trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        
        # 1. Train GFNet
        print("Training M2G-Net...")
        # Train using the default epochs and patience
        model, loss_fn, test_metrics, history = train(
            vocab, train_df, val_df, test_df, 
            max_epochs=cfg.MAX_EPOCHS, 
            patience=cfg.EARLY_STOPPING_PATIENCE,
            encoders=encoders,
            split_seed=seed,
        )
        gfnet_auc = test_metrics["macro"]["roc_auc"]
        scores_by_model["m2g_net"].append(gfnet_auc)
        run_row = {"seed": seed, "m2g_net": float(gfnet_auc)}
        
        # 2. Train strong tabular baselines on the same split.
        for baseline_name, (clf_class, kwargs_factory, display_name) in BASELINES.items():
            print(f"Training Baseline ({display_name})...")
            res_baseline = run_sklearn_baseline(
                clf_class, kwargs_factory(seed),
                trainval_df, test_df, display_name
            )
            baseline_auc = res_baseline["macro"]["roc_auc"]
            scores_by_model[baseline_name].append(baseline_auc)
            run_row[baseline_name] = float(baseline_auc)
        
        run_rows.append(run_row)
        print(
            f"Run {i+1} Results -> "
            + ", ".join(f"{name}: {value:.4f}" for name, value in run_row.items() if name != "seed")
        )
        
    print("\n\n=== REPEATED RUNS SUMMARY ===")
    summary = summarize_scores(scores_by_model)
    for model_name, row in summary.items():
        print(
            f"{model_name:<14}: {row['mean_macro_roc_auc']:.4f} "
            f"(95% CI: [{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}])"
        )
        if model_name != "m2g_net":
            print(
                f"  Paired vs M2G -> t={row['t_statistic']:.4f}, "
                f"p={row['p_value']:.4e}, delta={row['mean_delta_vs_m2g']:+.4f}"
            )
    write_outputs(run_rows, summary, seeds)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_MD}")

if __name__ == "__main__":
    main()
