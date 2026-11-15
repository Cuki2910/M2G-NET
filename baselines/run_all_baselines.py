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
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from src.data_pipeline import load_data, extract_task_masks, extract_targets


# ── Feature preparation ────────────────────────────────────────────────────────

def get_flat_features(df):
    """Return encoded categorical predictors, including the site ID."""
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
    return extract_targets(df).numpy()


def get_task_masks(df):
    return extract_task_masks(df).numpy()

def fit_tuned_baseline(name, train_df, val_df, seed=cfg.RANDOM_SEED, quick=False):
    """Choose each task's estimator using validation only; never refit on validation."""
    from sklearn.metrics import roc_auc_score
    factories = {
        "logistic_regression": (LogisticRegression, [{"C": value, "max_iter": 500} for value in [0.1, 1.0, 10.0]]),
        "decision_tree": (DecisionTreeClassifier, [{"max_depth": value, "min_samples_leaf": 10} for value in [3, 6, None]]),
        "random_forest": (RandomForestClassifier, [{"n_estimators": 150, "max_depth": value, "n_jobs": 1} for value in [5, 10, None]]),
    }
    if name == "xgboost":
        import xgboost as xgb
        factories[name] = (xgb.XGBClassifier, [{"n_estimators": 150, "max_depth": value, "learning_rate": 0.05, "n_jobs": 1, "eval_metric": "logloss"} for value in [2, 4, 6]])
    if name == "lightgbm":
        import lightgbm as lgb
        factories["lightgbm"] = (lgb.LGBMClassifier, [{"n_estimators": 150, "num_leaves": value, "learning_rate": 0.05, "verbosity": -1, "n_jobs": 1} for value in [7, 15, 31]])
    if name == "catboost":
        from catboost import CatBoostClassifier
        factories["catboost"] = (CatBoostClassifier, [{"iterations": 150, "depth": value, "learning_rate": 0.05, "verbose": False, "allow_writing_files": False, "thread_count": 1} for value in [4, 6, 8]])
    if name not in factories:
        raise ValueError(f"Unknown or unavailable baseline: {name}")
    classifier, candidates = factories[name]
    if quick:
        candidates = [dict(candidates[0])]
        for key in ("iterations", "n_estimators"):
            if key in candidates[0]:
                candidates[0][key] = 10
    train_x, val_x = [get_flat_features(frame).astype(str) for frame in (train_df, val_df)]
    train_y, val_y = [extract_targets(frame).numpy() for frame in (train_df, val_df)]
    train_masks, val_masks = [get_task_masks(frame).astype(bool) for frame in (train_df, val_df)]
    models, selections = [], []
    for task_idx, task in enumerate(cfg.TASK_NAMES):
        observed, valid = train_masks[:, task_idx], val_masks[:, task_idx]
        if not observed.any() or len(np.unique(val_y[valid, task_idx])) < 2:
            raise ValueError(f"{task}: training needs observed labels; validation needs both classes for tuning.")
        scores, best_model, best_score, best_params = [], None, -np.inf, None
        task_candidates = candidates if len(np.unique(train_y[observed, task_idx])) == 2 else [{}]
        for params in task_candidates:
            if len(np.unique(train_y[observed, task_idx])) < 2:
                estimator = DummyClassifier(strategy="prior")
            elif name == "catboost":
                estimator = classifier(**params, random_seed=seed, cat_features=list(range(train_x.shape[1])))
            else:
                estimator = make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                          classifier(**params, random_state=seed))
            estimator.fit(train_x[observed], train_y[observed, task_idx])
            probabilities = positive_probability(estimator, val_x[valid])
            score = float(roc_auc_score(val_y[valid, task_idx], probabilities))
            scores.append({"params": params, "validation_roc_auc": score})
            if score > best_score:
                best_model, best_score, best_params = estimator, score, params
        models.append(best_model)
        selections.append({"task": task, "selected": best_params, "candidates": scores})
    return models, selections

def positive_probability(estimator, features):
    probabilities = estimator.predict_proba(features)
    classes = np.asarray(estimator.classes_)
    return probabilities[:, int(np.flatnonzero(classes == 1)[0])] if 1 in classes else np.zeros(len(features))

def predict_baseline(models, frame):
    features = get_flat_features(frame).astype(str)
    return np.column_stack([positive_probability(model, features) for model in models])

def fit_validation_mlp(train_df, val_df, seed=42, max_epochs=150, patience=20, single_task=False):
    from src.metrics import safe_nanmean
    from sklearn.metrics import roc_auc_score
    import copy
    torch.manual_seed(seed)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_x = torch.tensor(encoder.fit_transform(get_flat_features(train_df).astype(str)), dtype=torch.float32)
    val_x = torch.tensor(encoder.transform(get_flat_features(val_df).astype(str)), dtype=torch.float32)
    train_y = extract_targets(train_df)
    masks = extract_task_masks(train_df)
    val_y, val_masks = extract_targets(val_df).numpy(), get_task_masks(val_df).astype(bool)
    if single_task:
        model = nn.ModuleList([nn.Sequential(nn.Linear(train_x.shape[1], 32), nn.ReLU(),
                                            nn.Linear(32, 1), nn.Sigmoid()) for _ in cfg.TASK_NAMES])
    else:
        model = EarlyFusionMLP(train_x.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    loader = DataLoader(TensorDataset(train_x, train_y, masks), batch_size=cfg.BATCH_SIZE, shuffle=True)
    best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for features, targets, observed in loader:
            predictions = [head(features).squeeze(-1) for head in model] if single_task else model(features)
            loss = predictions[0].sum() * 0
            for task_idx in range(cfg.NUM_TASKS):
                valid = observed[:, task_idx].bool()
                if valid.any():
                    loss = loss + nn.functional.binary_cross_entropy(predictions[task_idx][valid], targets[valid, task_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = [head(val_x).squeeze(-1) for head in model] if single_task else model(val_x)
        scores = [roc_auc_score(val_y[val_masks[:, idx], idx], pred.numpy()[val_masks[:, idx]])
                  for idx, pred in enumerate(predictions) if len(np.unique(val_y[val_masks[:, idx], idx])) == 2]
        score = safe_nanmean(scores)
        if np.isfinite(score) and score > best_score:
            best_score, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise ValueError("MLP validation needs at least one task with both classes.")
    model.load_state_dict(best_state)
    model.eval()
    return model, encoder, {"validation_roc_auc": best_score, "best_epoch": best_epoch, "single_task": single_task}

def predict_mlp(model, encoder, frame, single_task=False):
    features = torch.tensor(encoder.transform(get_flat_features(frame).astype(str)), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = [head(features).squeeze(-1) for head in model] if single_task else model(features)
    return torch.stack(predictions, dim=1).numpy()


# ── Sklearn baselines ──────────────────────────────────────────────────────────

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


if __name__ == "__main__":
    from scripts.evaluate.paper_pipeline import main
    main(["--models", "logistic_regression", "decision_tree", "random_forest", "xgboost",
          "lightgbm", "catboost", "early_fusion_mlp", "single_task_mlp", *sys.argv[1:]])
