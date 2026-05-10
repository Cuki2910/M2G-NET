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


# ── Encoding helpers ──────────────────────────────────────────────────────────

def build_encoders(df):
    """Fit a LabelEncoder for every categorical column. Returns dict of encoders."""
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le
    return encoders


def apply_encoders(df, encoders):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    return df


def vocab_sizes(df, encoders):
    """Return {col: num_unique_classes} for every encoded column."""
    return {col: len(le.classes_) for col, le in encoders.items()}


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
    Stratified split on the first target column (red_light_running).
    Returns train_df, val_df, test_df.
    """
    strat_col = df[cfg.TARGET_COLS[0]]
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=strat_col, random_state=seed)
    val_rel = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_rel,
        stratify=train_df[cfg.TARGET_COLS[0]], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def leave_intersection_out_split(df, test_site_id, seed=cfg.RANDOM_SEED):
    """
    Hold out one intersection as test set; rest is train.
    """
    test_df  = df[df[cfg.SITE_ID_COL] == test_site_id].reset_index(drop=True)
    train_df = df[df[cfg.SITE_ID_COL] != test_site_id].reset_index(drop=True)
    return train_df, test_df


# ── Public API ────────────────────────────────────────────────────────────────

def load_data():
    """
    Load CSV, encode categoricals, return (train_df, val_df, test_df, encoders, vocab).
    """
    df = pd.read_csv(cfg.DATA_PATH)

    # Fit encoders on full dataset so all classes are seen
    encoders = build_encoders(df)
    df_enc   = apply_encoders(df, encoders)

    # Shift intersection_id to 0-indexed
    df_enc[cfg.SITE_ID_COL] = df_enc[cfg.SITE_ID_COL] - df_enc[cfg.SITE_ID_COL].min()

    vocab = vocab_sizes(df_enc, encoders)
    vocab["num_sites"] = df_enc[cfg.SITE_ID_COL].nunique()

    train_df, val_df, test_df = stratified_split(df_enc)
    return train_df, val_df, test_df, encoders, vocab


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
