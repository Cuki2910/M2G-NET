"""
Baselines: Decision Tree, Logistic Regression, Random Forest, XGBoost,
Early-Fusion MLP, Late Fusion, Single-Task MLP.
Run: python baselines/run_all_baselines.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from src.data_pipeline import load_data
from src.metrics import compute_all_metrics, print_results, mtl_transfer_ratio


# ── Feature preparation ────────────────────────────────────────────────────────

def get_flat_features(df):
    """Return numpy array of all feature columns (excluding targets and site_id)."""
    feature_cols = (
        list(cfg.INDIVIDUAL_VIEW_COLS["rider_role"]) +
        list(cfg.INDIVIDUAL_VIEW_COLS["rider_traits"]) +
        list(cfg.CONTEXTUAL_VIEW_COLS["road_context"]) +
        list(cfg.CONTEXTUAL_VIEW_COLS["environment"]) +
        list(cfg.CONTEXTUAL_VIEW_COLS["site_obs"]) +
        [cfg.SITE_ID_COL]
    )
    return df[feature_cols].values.astype(float)


def get_targets(df):
    return df[cfg.TARGET_COLS].values


def get_task_masks(df):
    masks = []
    for task in cfg.TASK_NAMES:
        col = cfg.TASK_MASK_COLS.get(task)
        if col and col in df.columns:
            masks.append(df[col].astype(float).values)
        else:
            masks.append(np.ones(len(df), dtype=float))
    return np.stack(masks, axis=1)


# ── Sklearn baselines ──────────────────────────────────────────────────────────

def run_sklearn_baseline(clf_class, clf_kwargs, train_df, test_df, name):
    X_train = get_flat_features(train_df)
    X_test  = get_flat_features(test_df)
    Y_train = get_targets(train_df)
    Y_test  = get_targets(test_df)
    M_train = get_task_masks(train_df)
    M_test  = get_task_masks(test_df)

    all_probs = np.zeros((len(X_test), cfg.NUM_TASKS))
    for k, task in enumerate(cfg.TASK_NAMES):
        observed = M_train[:, k].astype(bool)
        clf = clf_class(**clf_kwargs)
        clf.fit(X_train[observed], Y_train[observed, k])
        if hasattr(clf, "predict_proba"):
            all_probs[:, k] = clf.predict_proba(X_test)[:, 1]
        else:
            all_probs[:, k] = clf.predict(X_test)

    metrics = compute_all_metrics(Y_test, all_probs, M_test)
    print_results(metrics, title=name)
    return metrics


# ── Early-Fusion MLP ──────────────────────────────────────────────────────────

class EarlyFusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(32, 1), nn.Sigmoid()) for _ in range(cfg.NUM_TASKS)
        ])

    def forward(self, x):
        h = self.shared(x)
        return [head(h).squeeze(-1) for head in self.heads]


def run_early_fusion_mlp(train_df, test_df, epochs=50):
    X_tr = torch.tensor(get_flat_features(train_df), dtype=torch.float32)
    Y_tr = torch.tensor(get_targets(train_df), dtype=torch.float32)
    X_te = torch.tensor(get_flat_features(test_df),  dtype=torch.float32)
    Y_te = get_targets(test_df)
    M_tr = torch.tensor(get_task_masks(train_df), dtype=torch.float32)
    M_te = get_task_masks(test_df)

    model = EarlyFusionMLP(X_tr.shape[1])
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    ds    = TensorDataset(X_tr, Y_tr, M_tr)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb, mb in loader:
            preds = model(xb)
            loss = preds[0].sum() * 0.0
            for k in range(cfg.NUM_TASKS):
                per_sample = nn.functional.binary_cross_entropy(preds[k], yb[:, k], reduction="none")
                denom = mb[:, k].sum()
                if denom.item() > 0:
                    loss = loss + (per_sample * mb[:, k]).sum() / denom
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.stack(model(X_te), dim=1).numpy()
    metrics = compute_all_metrics(Y_te, probs, M_te)
    print_results(metrics, "Early-Fusion MLP")
    return metrics


# ── Single-Task MLP ───────────────────────────────────────────────────────────

def run_single_task_mlp(train_df, test_df, epochs=50):
    X_tr = torch.tensor(get_flat_features(train_df), dtype=torch.float32)
    Y_tr = torch.tensor(get_targets(train_df), dtype=torch.float32)
    X_te = torch.tensor(get_flat_features(test_df), dtype=torch.float32)
    Y_te = get_targets(test_df)
    M_tr = get_task_masks(train_df)
    M_te = get_task_masks(test_df)

    all_probs = np.zeros((len(X_te), cfg.NUM_TASKS))
    for k, task in enumerate(cfg.TASK_NAMES):
        model = nn.Sequential(
            nn.Linear(X_tr.shape[1], 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        opt = optim.Adam(model.parameters(), lr=1e-3)
        observed = M_tr[:, k].astype(bool)
        ds  = TensorDataset(X_tr[observed], Y_tr[observed, k])
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                p = model(xb).squeeze(-1)
                loss = nn.functional.binary_cross_entropy(p, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            all_probs[:, k] = model(X_te).squeeze(-1).numpy()

    metrics = compute_all_metrics(Y_te, all_probs, M_te)
    print_results(metrics, "Single-Task MLP (4 models)")
    return metrics


# ── Main runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_df, val_df, test_df, encoders, vocab = load_data()
    # Combine train + val for baselines (they don't need a val split)
    trainval_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    results = {}

    results["decision_tree"] = run_sklearn_baseline(
        DecisionTreeClassifier, {"max_depth": 5, "random_state": cfg.RANDOM_SEED},
        trainval_df, test_df, "Decision Tree")

    results["logistic_regression"] = run_sklearn_baseline(
        LogisticRegression, {"max_iter": 1000, "random_state": cfg.RANDOM_SEED},
        trainval_df, test_df, "Logistic Regression")

    results["random_forest"] = run_sklearn_baseline(
        RandomForestClassifier, {"n_estimators": 100, "random_state": cfg.RANDOM_SEED},
        trainval_df, test_df, "Random Forest")

    results["xgboost"] = run_sklearn_baseline(
        xgb.XGBClassifier,
        {"n_estimators": 100, "use_label_encoder": False,
         "eval_metric": "logloss", "random_state": cfg.RANDOM_SEED, "verbosity": 0},
        trainval_df, test_df, "XGBoost")

    if LIGHTGBM_AVAILABLE:
        results["lightgbm"] = run_sklearn_baseline(
            lgb.LGBMClassifier,
            {"n_estimators": 100, "learning_rate": 0.05,
             "random_state": cfg.RANDOM_SEED, "verbose": -1},
            trainval_df, test_df, "LightGBM")
    else:
        print("\n[WARN] lightgbm is not installed; skipping LightGBM baseline.")

    results["early_fusion_mlp"] = run_early_fusion_mlp(trainval_df, test_df)
    results["single_task_mlp"]  = run_single_task_mlp(trainval_df, test_df)

    # Summary table
    print("\n\n=== BASELINE SUMMARY (Macro ROC-AUC) ===")
    for name, res in results.items():
        print(f"  {name:<25}: {res['macro']['roc_auc']:.4f}")
