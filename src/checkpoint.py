"""
Checkpoint helpers for reproducible evaluation.

The checkpoint stores both model weights and the preprocessing contract used to
produce the encoded train/val/test splits. Evaluation scripts should load this
bundle instead of rebuilding encoders from the current working tree.
"""

import os
import sys
import hashlib
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as cfg
from src.data_pipeline import (
    encode_splits_with_preprocessing,
    encoders_from_state,
    encoders_to_state,
    load_raw_splits,
    prepare_labels,
)
from src.loss import UncertaintyWeightedLoss
from src.calibration import TemperatureScaler
from src.model import TGMVMTGFNetV2


MODEL_CONFIG_KEYS = {
    "embedding_dim": "EMBEDDING_DIM",
    "hidden_dim": "HIDDEN_DIM",
    "view_dim": "VIEW_DIM",
    "interaction_dim": "INTERACTION_DIM",
    "num_tasks": "NUM_TASKS",
    "num_views": "NUM_VIEWS",
    "num_gate_inputs": "NUM_GATE_INPUTS",
    "temperature_init": "TEMPERATURE_INIT",
    "temperature_final": "TEMPERATURE_FINAL",
    "gate_prior_weight": "GATE_PRIOR_WEIGHT",
    "dropout_rate": "DROPOUT_RATE",
    "gate_hidden_dim": "V3_GATE_HIDDEN_DIM",
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


def dataset_sha256(data_path):
    digest = hashlib.sha256()
    with open(data_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def make_checkpoint_payload(epoch, model, loss_fn, val_auc, vocab, encoders=None,
                            thresholds=None,
                            split_seed=cfg.RANDOM_SEED, data_path=cfg.DATA_PATH,
                            temperature=1.0, split_rows=None, label_policy=cfg.LABEL_POLICY):
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
        "ablation": getattr(model, "ablation", "full"),
        "gate_variant": getattr(model, "gate_variant", "legacy"),
        "interaction_skip": getattr(model, "interaction_skip", False),
        "temperature": float(temperature),
        "label_policy": label_policy,
        "loss_config": {"use_focal": loss_fn.use_focal,
                        "use_pos_weight": loss_fn.use_pos_weight,
                        "use_uncertainty": loss_fn.use_uncertainty},
    }
    if os.path.isfile(data_path):
        payload["dataset_sha256"] = dataset_sha256(data_path)
    payload["feature_schema"] = {
        "individual": cfg.INDIVIDUAL_VIEW_COLS, "contextual": cfg.CONTEXTUAL_VIEW_COLS,
        "mask_columns": cfg.TASK_MASK_COLS,
    }
    if split_rows is not None:
        payload["split_rows"] = {key: [int(value) for value in values] for key, values in split_rows.items()}
    if model.fusion.gates:
        payload["model_config"]["gate_prior_weight"] = model.fusion.gates[0].prior_weight
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
        "embedding_dim",
        "hidden_dim",
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
    variant = ckpt.get("gate_variant", "legacy")
    if getattr(model, "gate_variant", "legacy") != variant:
        raise ValueError("Checkpoint gate variant differs from the constructed model.")
    if variant in {"nonlinear", "nonlinear_task_blend"} and saved_config.get(
            "gate_hidden_dim", cfg.V3_GATE_HIDDEN_DIM) != cfg.V3_GATE_HIDDEN_DIM:
        raise ValueError("Checkpoint nonlinear gate hidden dimension differs from current config.")

    prior_weight = saved_config.get("gate_prior_weight")
    if prior_weight is not None:
        for gate in model.fusion.gates:
            gate.prior_weight = float(prior_weight)
    return model


def splits_from_checkpoint(ckpt, data_path=None):
    data_path = data_path or ckpt.get("data_path", cfg.DATA_PATH)
    split_seed = int(ckpt.get("split_seed", cfg.RANDOM_SEED))
    expected_hash = ckpt.get("dataset_sha256")
    if expected_hash and dataset_sha256(data_path) != expected_hash:
        raise ValueError("Dataset hash differs from checkpoint; evaluation would not reproduce its training split.")
    policy = ckpt.get("label_policy", cfg.LABEL_POLICY)
    split_rows = ckpt.get("split_rows")
    if split_rows:
        raw = pd.read_csv(data_path)
        raw["_row_id"] = np.arange(len(raw))
        raw = prepare_labels(raw, policy)
        membership = [split_rows[key] for key in ("train", "validation", "test")]
        flat = [row for rows in membership for row in rows]
        if len(flat) != len(set(flat)) or any(row < 0 or row >= len(raw) for row in flat):
            raise ValueError("Checkpoint split rows overlap or fall outside the dataset.")
        raw_train, raw_val, raw_test = [raw.iloc[rows].reset_index(drop=True) for rows in membership]
    else:
        warnings.warn("Legacy checkpoint lacks exact split membership; reconstructing from seed.", RuntimeWarning)
        raw_train, raw_val, raw_test = load_raw_splits(seed=split_seed, data_path=data_path, label_policy=policy)

    preprocessing = ckpt.get("preprocessing")
    if preprocessing is None:
        from src.data_pipeline import encode_splits
        train_df, val_df, test_df, encoders, vocab = encode_splits(raw_train, raw_val, raw_test)
        return raw_train, raw_val, raw_test, train_df, val_df, test_df, encoders, vocab

    encoders = encoders_from_state(preprocessing["encoder_classes"])
    site_mapping = _site_mapping_from_checkpoint(preprocessing["site_mapping"])
    train_df, val_df, test_df, encoders, vocab = encode_splits_with_preprocessing(
        raw_train, raw_val, raw_test, encoders, site_mapping)
    return raw_train, raw_val, raw_test, train_df, val_df, test_df, encoders, vocab


def load_model_bundle(checkpoint_path=cfg.CHECKPOINT_PATH, data_path=None, map_location="cpu"):
    ckpt = load_checkpoint(checkpoint_path, map_location=map_location)
    current_schema = {"individual": cfg.INDIVIDUAL_VIEW_COLS, "contextual": cfg.CONTEXTUAL_VIEW_COLS,
                      "mask_columns": cfg.TASK_MASK_COLS}
    if ckpt.get("target_cols", cfg.TARGET_COLS) != cfg.TARGET_COLS or ckpt.get("feature_schema", current_schema) != current_schema:
        raise ValueError("Checkpoint target/feature schema differs from current config.")
    raw_train, raw_val, raw_test, train_df, val_df, test_df, encoders, vocab = splits_from_checkpoint(
        ckpt, data_path=data_path)

    model = TGMVMTGFNetV2(vocab, ablation=ckpt.get("ablation", "full"),
                          gate_variant=ckpt.get("gate_variant", "legacy"),
                          interaction_skip=ckpt.get("interaction_skip", False))
    loss_fn = UncertaintyWeightedLoss(**ckpt.get("loss_config", {}))
    apply_checkpoint_model_config(model, ckpt)
    model.load_state_dict(ckpt["model_state"])
    loss_fn.load_state_dict(ckpt["loss_state"])
    model.eval()
    calibrator = TemperatureScaler()
    calibrator.temperature = ckpt.get("temperature", 1.0)
    calibrator.transform(np.empty((0, cfg.NUM_TASKS)))

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
        "temperature": ckpt.get("temperature", 1.0),
        "calibrator": calibrator,
        "label_policy": ckpt.get("label_policy", cfg.LABEL_POLICY),
        "validation_use_site_intercept": ckpt.get("validation_use_site_intercept", True),
        "test_use_site_intercept": ckpt.get("test_use_site_intercept", True),
    }
