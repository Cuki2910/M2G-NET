"""
Phase 1: Data Pipeline
Loads the synthetic dataset, encodes categoricals, splits views (Individual / Contextual),
creates stratified train/val/test splits and a Leave-Intersection-Out split,
then wraps everything in a PyTorch Dataset + DataLoader.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg

UNKNOWN_CATEGORY = "__UNK__"


def _configured_categorical_cols(df):
    """Categorical feature columns that feed the model, excluding site ids."""
    cols = []
    for view_cols in cfg.INDIVIDUAL_VIEW_COLS.values():
        cols.extend(view_cols)
    for view_cols in cfg.CONTEXTUAL_VIEW_COLS.values():
        cols.extend(view_cols)
    return [c for c in cols if c in df.columns and df[c].dtype == object]


# ── Encoding helpers ──────────────────────────────────────────────────────────

def build_encoders(df):
    """Fit feature encoders on training data only, with an explicit unknown bucket."""
    cat_cols = _configured_categorical_cols(df)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        values = pd.Series(df[col].dropna().unique())
        le.fit(pd.concat([values, pd.Series([UNKNOWN_CATEGORY])], ignore_index=True))
        encoders[col] = le
    return encoders


def apply_encoders(df, encoders):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            values = df[col].where(df[col].isin(known), UNKNOWN_CATEGORY)
            df[col] = le.transform(values)
    return df


def vocab_sizes(df, encoders):
    """Return {col: num_unique_classes} for every encoded column."""
    return {col: len(le.classes_) for col, le in encoders.items()}


def encoders_to_state(encoders):
    """Serialize LabelEncoder classes using only weights-only-safe primitives."""
    return {
        col: [str(value) for value in le.classes_.tolist()]
        for col, le in encoders.items()
    }


def encoders_from_state(encoder_state):
    """Rebuild LabelEncoder instances from checkpoint metadata."""
    encoders = {}
    for col, classes in encoder_state.items():
        le = LabelEncoder()
        le.classes_ = np.array(classes, dtype=object)
        encoders[col] = le
    return encoders


def build_site_mapping(df):
    """Map train-time site ids to contiguous embedding ids; reserve the last id for unknown sites."""
    train_sites = sorted(df[cfg.SITE_ID_COL].dropna().unique())
    return {int(site_id): int(idx) for idx, site_id in enumerate(train_sites)}


def apply_site_mapping(df, site_mapping):
    df = df.copy()
    unknown_site = len(site_mapping)
    df[cfg.SITE_ID_COL] = df[cfg.SITE_ID_COL].map(site_mapping).fillna(unknown_site).astype(int)
    return df


def encode_splits(train_df, val_df, test_df):
    """Fit preprocessing on train only, then transform all splits."""
    encoders = build_encoders(train_df)
    site_mapping = build_site_mapping(train_df)
    return encode_splits_with_preprocessing(train_df, val_df, test_df, encoders, site_mapping)


def encode_with_preprocessing(df, encoders, site_mapping):
    return apply_site_mapping(apply_encoders(df, encoders), site_mapping)


def build_vocab(encoders, site_mapping):
    vocab = vocab_sizes(None, encoders)
    vocab["num_sites"] = len(site_mapping) + 1
    vocab["site_mapping"] = site_mapping
    vocab["unknown_site_id"] = len(site_mapping)
    return vocab


def encode_splits_with_preprocessing(train_df, val_df, test_df, encoders, site_mapping):
    """Transform splits with checkpoint/train-time preprocessing."""
    train_enc = encode_with_preprocessing(train_df, encoders, site_mapping)
    val_enc = encode_with_preprocessing(val_df, encoders, site_mapping)
    test_enc = encode_with_preprocessing(test_df, encoders, site_mapping)
    vocab = build_vocab(encoders, site_mapping)

    return train_enc, val_enc, test_enc, encoders, vocab


# ── View extraction helpers ───────────────────────────────────────────────────

def extract_views(df, vocab):
    """
    Extract tensors for each view from a (already encoded) DataFrame.
    Returns dict: view_name → LongTensor (N, num_features_in_view)
    """
    views = {}

    for view_name, cols in cfg.INDIVIDUAL_VIEW_COLS.items():
        views[view_name] = torch.tensor(df[cols].values, dtype=torch.long)

    for view_name, cols in cfg.CONTEXTUAL_VIEW_COLS.items():
        if view_name == "site_obs":
            views["site_obs"] = torch.tensor(df[cols].values, dtype=torch.long)
        else:
            views[view_name] = torch.tensor(df[cols].values, dtype=torch.long)

    views["site_id"] = torch.tensor(df[cfg.SITE_ID_COL].values, dtype=torch.long)

    return views


def extract_targets(df):
    return torch.tensor(df[cfg.TARGET_COLS].values, dtype=torch.float32)


def extract_task_masks(df):
    """
    Return an (N, K) mask tensor where 1 means the task label is observed and
    applicable. Missing mask columns default to 1 for backward compatibility.
    """
    mask_cols = []
    for task in cfg.TASK_NAMES:
        col = cfg.TASK_MASK_COLS.get(task)
        if col and col in df.columns:
            mask_cols.append(df[col].astype(float).values)
        else:
            mask_cols.append(np.ones(len(df), dtype=float))
    masks = np.stack(mask_cols, axis=1)
    return torch.tensor(masks, dtype=torch.float32)


def compute_pos_weights(df):
    """
    Return per-task negative/positive ratios on observed labels.
    Rare positive classes receive larger weights during BCE/focal loss.
    """
    weights = []
    for task in cfg.TASK_NAMES:
        mask_col = cfg.TASK_MASK_COLS.get(task)
        observed = df[mask_col].astype(bool).to_numpy() if mask_col in df.columns else np.ones(len(df), dtype=bool)
        labels = df.loc[observed, task].astype(float)
        positives = float(labels.sum())
        negatives = float(len(labels) - positives)
        if positives <= 0:
            weights.append(1.0)
        else:
            weights.append(max(1.0, negatives / positives))
    return np.array(weights, dtype=np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class RiderDataset(Dataset):
    """
    Each item: (views_dict, targets_tensor, task_mask_tensor)
    views_dict keys: rider_role, rider_traits, road_context, environment,
                     site_obs, site_id
    targets_tensor shape: (4,)
    """

    def __init__(self, df, vocab):
        self.views = extract_views(df, vocab)
        self.targets = extract_targets(df)
        self.masks = extract_task_masks(df)
        self.n = len(df)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.views.items()}
        return item, self.targets[idx], self.masks[idx]


# ── Split helpers ─────────────────────────────────────────────────────────────

def stratified_split(df, test_size=cfg.TEST_SIZE, val_size=cfg.VAL_SIZE,
                     seed=cfg.RANDOM_SEED):
    """
    Stratified split on the multi-task label signature when possible.
    Returns train_df, val_df, test_df.
    """
    def stratify_key(frame):
        multi = frame[cfg.TARGET_COLS].astype(str).agg("_".join, axis=1)
        if multi.value_counts().min() >= 2:
            return multi
        return frame[cfg.TARGET_COLS[0]]

    strat_col = stratify_key(df)
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=strat_col, random_state=seed)
    val_rel = val_size / (1 - test_size)
    train_strat_col = stratify_key(train_df)
    train_df, val_df = train_test_split(
        train_df, test_size=val_rel,
        stratify=train_strat_col, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def safe_train_val_split(df, val_size=cfg.VAL_SIZE, seed=cfg.RANDOM_SEED):
    """Split train/val with stratification when possible, fallback when a fold is degenerate."""
    stratify = df[cfg.TARGET_COLS[0]]
    try:
        train_df, val_df = train_test_split(
            df, test_size=val_size, stratify=stratify, random_state=seed)
    except ValueError:
        train_df, val_df = train_test_split(
            df, test_size=val_size, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(seed=cfg.RANDOM_SEED, data_path=cfg.DATA_PATH):
    """
    Load CSV, split first, then fit preprocessing on train only.
    Returns (train_df, val_df, test_df, encoders, vocab).
    """
    df = pd.read_csv(data_path)
    train_df, val_df, test_df = stratified_split(df, seed=seed)
    return encode_splits(train_df, val_df, test_df)


def load_raw_splits(seed=cfg.RANDOM_SEED, data_path=cfg.DATA_PATH):
    df = pd.read_csv(data_path)
    return stratified_split(df, seed=seed)


def get_loaders(train_df, val_df, test_df, vocab,
                batch_size=cfg.BATCH_SIZE, num_workers=0):
    train_ds = RiderDataset(train_df, vocab)
    val_ds   = RiderDataset(val_df,   vocab)
    test_ds  = RiderDataset(test_df,  vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_df, val_df, test_df, encoders, vocab = load_data()
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Vocab sample:", {k: v for k, v in list(vocab.items())[:5]})
    train_loader, val_loader, test_loader = get_loaders(train_df, val_df, test_df, vocab)
    views, targets, masks = next(iter(train_loader))
    print("Batch views:", {k: v.shape for k, v in views.items()})
    print("Batch targets:", targets.shape)
    print("Batch masks:", masks.shape)
