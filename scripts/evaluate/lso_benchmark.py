"""
Leave-Site-Out Benchmark: M2G-Net v2 vs CatBoost / LightGBM / XGBoost / RF.

For every held-out site, trains a fresh model per method on the remaining sites,
evaluates on the held-out site, then reports fold-level AUCs and paired tests.
This is the "scientific" test: random-split AUC measures memorisation; LSO AUC
measures whether representations generalise to unseen sites/intersections.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import config as cfg
from src.data_pipeline import encode_splits, safe_train_val_split, RiderDataset
from scripts.train import fit_model_for_split
from baselines.run_all_baselines import get_flat_features, get_targets, get_task_masks

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_JSON = "outputs/lso_benchmark.json"
OUTPUT_CSV  = "outputs/lso_benchmark.csv"
OUTPUT_PNG  = "outputs/lso_benchmark.png"
OUTPUT_MD   = "docs/LSO_BENCHMARK.md"

MIN_TEST_SIZE   = 5
LSO_MAX_EPOCHS  = 50
LSO_PATIENCE    = 10

BASELINE_CFGS = {}

if True:  # XGBoost always available
    BASELINE_CFGS["xgboost"] = {
        "cls":    XGBClassifier,
        "kwargs": {
            "n_estimators": 100,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": cfg.RANDOM_SEED,
            "verbosity": 0,
        },
        "display": "XGBoost",
    }

BASELINE_CFGS["random_forest"] = {
    "cls":    RandomForestClassifier,
    "kwargs": {"n_estimators": 100, "random_state": cfg.RANDOM_SEED},
    "display": "Random Forest",
}

if LIGHTGBM_AVAILABLE:
    BASELINE_CFGS["lightgbm"] = {
        "cls":    lgb.LGBMClassifier,
        "kwargs": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": cfg.RANDOM_SEED,
            "verbose": -1,
        },
        "display": "LightGBM",
    }

if CATBOOST_AVAILABLE:
    BASELINE_CFGS["catboost"] = {
        "cls":    CatBoostClassifier,
        "kwargs": {
            "iterations": 100,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "Logloss",
            "random_seed": cfg.RANDOM_SEED,
            "verbose": False,
            "allow_writing_files": False,
        },
        "display": "CatBoost",
    }


def collect_m2g_probs(model, test_df, vocab):
    loader = torch.utils.data.DataLoader(
        RiderDataset(test_df, vocab), batch_size=cfg.BATCH_SIZE, shuffle=False,
    )
    all_probs, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for views, targets, _ in loader:
            preds, _, _ = model(views, use_site_intercept=False)
            all_probs.append(torch.stack(preds, dim=1).numpy())
            all_targets.append(targets.numpy())
    return np.vstack(all_targets), np.vstack(all_probs)


def macro_auc(y_true, y_probs):
    aucs = []
    for k in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:, k], y_probs[:, k]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


def run_baseline_fold(clf_cls, clf_kwargs, train_df, val_df, test_df):
    trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    X_tr = get_flat_features(trainval_df)
    X_te = get_flat_features(test_df)
    Y_tr = get_targets(trainval_df)
    Y_te = get_targets(test_df)
    M_tr = get_task_masks(trainval_df)

    all_probs = np.zeros((len(X_te), cfg.NUM_TASKS))
    for k in range(cfg.NUM_TASKS):
        observed = M_tr[:, k].astype(bool)
        clf = clf_cls(**clf_kwargs)
        clf.fit(X_tr[observed], Y_tr[observed, k])
        if hasattr(clf, "predict_proba"):
            all_probs[:, k] = clf.predict_proba(X_te)[:, 1]
        else:
            all_probs[:, k] = clf.predict(X_te)
    return Y_te, all_probs


def paired_stats(ref_aucs, cmp_aucs):
    """Returns mean_delta (ref - cmp), t-stat, p-value, wilcoxon p-value."""
    ref = np.array(ref_aucs)
    cmp = np.array(cmp_aucs)
    delta = float(np.mean(ref - cmp))
    try:
        t, p_t = stats.ttest_rel(ref, cmp)
    except Exception:
        t, p_t = float("nan"), float("nan")
    try:
        _, p_w = stats.wilcoxon(ref - cmp)
    except Exception:
        p_w = float("nan")
    return delta, float(t), float(p_t), float(p_w)


def plot_lso_benchmark(fold_results, model_keys, display_names, mean_aucs, save_path):
    n_folds  = len(fold_results)
    x        = np.arange(n_folds)
    colors   = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]

    fig, ax = plt.subplots(figsize=(max(14, n_folds * 0.45), 5))
    width = 0.8 / len(model_keys)
    offsets = np.linspace(-(len(model_keys)-1)/2, (len(model_keys)-1)/2, len(model_keys)) * width

    for i, (key, offset) in enumerate(zip(model_keys, offsets)):
        aucs = [r[key] for r in fold_results]
        color = colors[i % len(colors)]
        ax.bar(x + offset, aucs, width=width * 0.9, color=color, alpha=0.75,
               label=f"{display_names[key]} (mean={mean_aucs[key]:.4f})",
               edgecolor="black", linewidth=0.3)

    ax.axhline(0.65, color="gray", linestyle="-.", linewidth=1, alpha=0.6, label="Threshold=0.65")
    ax.set_xlabel("Fold (held-out site index)", fontsize=11)
    ax.set_ylabel("Macro ROC-AUC", fontsize=11)
    ax.set_title("Leave-Site-Out Benchmark: Per-Fold Macro ROC-AUC", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_folds)], fontsize=7)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def write_markdown(summary, fold_results, model_keys, display_names):
    m2g_mean = summary["m2g"]["mean"]
    lines = [
        "# Leave-Site-Out Benchmark: M2G-Net v2 vs Baselines",
        "",
        "**Data status:** synthetic proof-of-concept.",
        "**Generalization test:** each fold trains on all sites except one, evaluates on the held-out site.",
        "**Statistical tests:** paired t-test and Wilcoxon signed-rank over fold AUCs.",
        "",
        "## Summary",
        "",
        "| Model | Mean LSO AUC | Std | Delta vs M2G | t-stat | t p-value | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.append(
        f"| M2G-Net v2 | {m2g_mean:.4f} | {summary['m2g']['std']:.4f} | ref | ref | ref | ref |"
    )
    for key in model_keys:
        if key == "m2g":
            continue
        s = summary[key]
        lines.append(
            f"| {display_names[key]} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['delta_vs_m2g']:+.4f} | {s['t_stat']:.3f} | {s['p_t']:.4g} | {s['p_w']:.4g} |"
        )

    lines.extend([
        "",
        "## Per-Fold AUC",
        "",
        "| Fold | " + " | ".join(display_names[k] for k in model_keys) + " |",
        "|---:|" + "---:|" * len(model_keys),
    ])
    for i, r in enumerate(fold_results):
        row = f"| {i} | " + " | ".join(f"{r[k]:.4f}" for k in model_keys) + " |"
        lines.append(row)

    lines.extend([
        "",
        "## Notes",
        "",
        "- M2G uses `use_site_intercept=False` at test time (site is truly unseen).",
        "- Baselines trained on encoded trainval features; no early stopping.",
        "- Delta > 0 means M2G outperforms the baseline on average across LSO folds.",
        "- p < 0.05 on either paired test indicates the difference is statistically significant.",
    ])

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {OUTPUT_MD}")


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("Loading raw data...")
    raw_df = pd.read_csv(cfg.DATA_PATH)
    sites  = sorted(raw_df[cfg.SITE_ID_COL].unique())
    n_sites = len(sites)
    print(f"Sites: {n_sites}  —  running {n_sites} LSO folds")
    print(f"Baselines: {list(BASELINE_CFGS.keys())}\n")

    model_keys    = ["m2g"] + list(BASELINE_CFGS.keys())
    display_names = {"m2g": "M2G-Net v2"}
    display_names.update({k: v["display"] for k, v in BASELINE_CFGS.items()})

    fold_results = []
    fold_aucs    = {k: [] for k in model_keys}

    for fold_idx, site_id in enumerate(sites):
        test_raw = raw_df[raw_df[cfg.SITE_ID_COL] == site_id].reset_index(drop=True)
        if len(test_raw) < MIN_TEST_SIZE:
            print(f"  [skip] site {site_id}: only {len(test_raw)} samples")
            continue

        trainval_raw = raw_df[raw_df[cfg.SITE_ID_COL] != site_id].reset_index(drop=True)
        train_raw, val_raw = safe_train_val_split(trainval_raw)
        train_enc, val_enc, test_enc, encoders, vocab = encode_splits(
            train_raw.reset_index(drop=True),
            val_raw.reset_index(drop=True),
            test_raw,
        )

        print(f"\n[Fold {fold_idx+1}/{n_sites}] site={site_id}  "
              f"train={len(train_enc)}  val={len(val_enc)}  test={len(test_enc)}")

        row = {}

        # M2G
        try:
            model, _, _, _ = fit_model_for_split(
                vocab, train_enc, val_enc, encoders=encoders,
                max_epochs=LSO_MAX_EPOCHS, patience=LSO_PATIENCE,
            )
            y_true, y_probs = collect_m2g_probs(model, test_enc, vocab)
            m2g_auc = macro_auc(y_true, y_probs)
        except Exception as e:
            print(f"    [M2G error] {e}")
            m2g_auc = float("nan")
        row["m2g"] = m2g_auc
        print(f"    M2G: {m2g_auc:.4f}")

        # Baselines
        for key, bcfg in BASELINE_CFGS.items():
            try:
                y_te, y_pr = run_baseline_fold(
                    bcfg["cls"], bcfg["kwargs"],
                    train_enc, val_enc, test_enc,
                )
                auc = macro_auc(y_te, y_pr)
            except Exception as e:
                print(f"    [{key} error] {e}")
                auc = float("nan")
            row[key] = auc
            print(f"    {bcfg['display']}: {auc:.4f}")

        fold_results.append(row)
        for k in model_keys:
            if np.isfinite(row.get(k, float("nan"))):
                fold_aucs[k].append(row[k])

    n_folds = len(fold_results)
    print(f"\n\nCompleted {n_folds} folds.\n")

    # Summary statistics
    summary = {}
    m2g_aucs = fold_aucs["m2g"]
    for key in model_keys:
        aucs = fold_aucs[key]
        mean = float(np.mean(aucs)) if aucs else float("nan")
        std  = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        s = {"mean": mean, "std": std, "n_folds": len(aucs), "fold_aucs": aucs}
        if key != "m2g" and len(m2g_aucs) >= 2 and len(aucs) >= 2:
            # align folds (take min length in case of errors)
            n = min(len(m2g_aucs), len(aucs))
            delta, t_stat, p_t, p_w = paired_stats(m2g_aucs[:n], aucs[:n])
            s.update({"delta_vs_m2g": delta, "t_stat": t_stat, "p_t": p_t, "p_w": p_w})
        summary[key] = s

    # Console report
    print(f"{'Model':<20} {'Mean LSO AUC':>14} {'Std':>8} {'Delta vs M2G':>14} {'t p-val':>10}")
    print("-" * 72)
    for key in model_keys:
        s = summary[key]
        delta_str = "ref" if key == "m2g" else f"{s.get('delta_vs_m2g', float('nan')):+.4f}"
        p_str     = "ref" if key == "m2g" else f"{s.get('p_t', float('nan')):.4g}"
        print(f"{display_names[key]:<20} {s['mean']:>14.4f} {s['std']:>8.4f} {delta_str:>14} {p_str:>10}")

    # Save outputs
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "n_folds": n_folds,
            "model_keys": model_keys,
            "display_names": display_names,
            "fold_results": fold_results,
            "summary": summary,
        }, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    mean_aucs = {k: summary[k]["mean"] for k in model_keys}
    plot_lso_benchmark(fold_results, model_keys, display_names, mean_aucs, OUTPUT_PNG)

    # CSV: one row per fold
    csv_data = {f"fold": list(range(n_folds))}
    for k in model_keys:
        csv_data[display_names[k]] = [r.get(k, float("nan")) for r in fold_results]
    pd.DataFrame(csv_data).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    write_markdown(summary, fold_results, model_keys, display_names)


if __name__ == "__main__":
    main()
