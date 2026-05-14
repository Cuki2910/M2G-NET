"""
Checkpoint helpers for reproducible evaluation.

The checkpoint stores both model weights and the preprocessing contract used to
produce the encoded train/val/test splits. Evaluation scripts should load this
bundle instead of rebuilding encoders from the current working tree.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from src.data_pipeline import (
    encode_splits_with_preprocessing,
    encoders_from_state,
    encoders_to_state,
    load_data,
    load_raw_splits,
)
from src.loss import UncertaintyWeightedLoss
from src.model import TGMVMTGFNetV2


MODEL_CONFIG_KEYS = {
    "view_dim": "VIEW_DIM",
    "interaction_dim": "INTERACTION_DIM",
    "num_tasks": "NUM_TASKS",
    "num_views": "NUM_VIEWS",
    "num_gate_inputs": "NUM_GATE_INPUTS",
    "temperature_init": "TEMPERATURE_INIT",
    "temperature_final": "TEMPERATURE_FINAL",
    "gate_prior_weight": "GATE_PRIOR_WEIGHT",
    "dropout_rate": "DROPOUT_RATE",
}


def current_model_config():
    """Return the config values needed to reproduce checkpoint inference."""
    return {
        key: (
            int(getattr(cfg, cfg_name))
            if isinstance(getattr(cfg, cfg_name), int)
            else float(getattr(cfg, cfg_name))
        )
        for key, cfg_name in MODEL_CONFIG_KEYS.items()
    }


def _site_mapping_for_checkpoint(site_mapping):
    return {str(int(site_id)): int(encoded_id) for site_id, encoded_id in site_mapping.items()}


def _site_mapping_from_checkpoint(site_mapping):
    return {int(site_id): int(encoded_id) for site_id, encoded_id in site_mapping.items()}


def make_checkpoint_payload(epoch, model, loss_fn, val_auc, vocab, encoders=None,
                            thresholds=None,
                            split_seed=cfg.RANDOM_SEED, data_path=cfg.DATA_PATH):
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "loss_state": loss_fn.state_dict(),
        "val_auc": float(val_auc),
        "vocab": vocab,
        "split_seed": int(split_seed),
        "data_path": str(data_path),
        "target_cols": list(cfg.TARGET_COLS),
        "site_id_col": cfg.SITE_ID_COL,
        "model_config": current_model_config(),
    }
    if thresholds is not None:
        payload["thresholds"] = {
            task: float(thresholds.get(task, 0.5))
            for task in cfg.TASK_NAMES
        }
    if encoders is not None:
        payload["preprocessing"] = {
            "encoder_classes": encoders_to_state(encoders),
            "site_mapping": _site_mapping_for_checkpoint(vocab["site_mapping"]),
            "unknown_site_id": int(vocab["unknown_site_id"]),
            "num_sites": int(vocab["num_sites"]),
        }
    return payload


def load_checkpoint(checkpoint_path=cfg.CHECKPOINT_PATH, map_location="cpu"):
    return torch.load(checkpoint_path, map_location=map_location, weights_only=True)


def _validate_architecture_config(saved_config):
    """Fail fast when a checkpoint needs a different model architecture."""
    if not saved_config:
        return []

    current = current_model_config()
    architecture_keys = [
        "view_dim",
        "interaction_dim",
        "num_tasks",
        "num_views",
        "num_gate_inputs",
        "dropout_rate",
    ]
    mismatches = []
    for key in architecture_keys:
        if key in saved_config and saved_config[key] != current[key]:
            mismatches.append((key, saved_config[key], current[key]))
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={saved}, current={current_value}"
            for key, saved, current_value in mismatches
        )
        raise ValueError(
            "Checkpoint model architecture does not match current config. "
            f"Mismatches: {details}."
        )
    return mismatches


def apply_checkpoint_model_config(model, ckpt):
    """
    Apply non-state-dict model settings stored in a checkpoint.

    Older checkpoints do not contain model_config; in that case the current
    config is used for backward compatibility.
    """
    saved_config = ckpt.get("model_config") or {}
    _validate_architecture_config(saved_config)

    prior_weight = saved_config.get("gate_prior_weight")
    if prior_weight is not None:
        for gate in model.fusion.gates:
            gate.prior_weight = float(prior_weight)
    return model


def splits_from_checkpoint(ckpt, data_path=None):
    data_path = data_path or ckpt.get("data_path", cfg.DATA_PATH)
    split_seed = int(ckpt.get("split_seed", cfg.RANDOM_SEED))
    raw_train, raw_val, raw_test = load_raw_splits(seed=split_seed, data_path=data_path)

    preprocessing = ckpt.get("preprocessing")
    if preprocessing is None:
        train_df, val_df, test_df, encoders, vocab = load_data(seed=split_seed, data_path=data_path)
        return raw_train, raw_val, raw_test, train_df, val_df, test_df, encoders, vocab

    encoders = encoders_from_state(preprocessing["encoder_classes"])
    site_mapping = _site_mapping_from_checkpoint(preprocessing["site_mapping"])
    train_df, val_df, test_df, encoders, vocab = encode_splits_with_preprocessing(
        raw_train, raw_val, raw_test, encoders, site_mapping)
    return raw_train, raw_val, raw_test, train_df, val_df, test_df, encoders, vocab


def load_model_bundle(checkpoint_path=cfg.CHECKPOINT_PATH, data_path=None, map_location="cpu"):
    ckpt = load_checkpoint(checkpoint_path, map_location=map_location)
    raw_train, raw_val, raw_test, train_df, val_df, test_df, encoders, vocab = splits_from_checkpoint(
        ckpt, data_path=data_path)

    model = TGMVMTGFNetV2(vocab)
    loss_fn = UncertaintyWeightedLoss()
    apply_checkpoint_model_config(model, ckpt)
    model.load_state_dict(ckpt["model_state"])
    loss_fn.load_state_dict(ckpt["loss_state"])
    model.eval()

    return {
        "checkpoint": ckpt,
        "model": model,
        "loss_fn": loss_fn,
        "vocab": vocab,
        "encoders": encoders,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "raw_train_df": raw_train,
        "raw_val_df": raw_val,
        "raw_test_df": raw_test,
        "thresholds": ckpt.get("thresholds"),
    }
