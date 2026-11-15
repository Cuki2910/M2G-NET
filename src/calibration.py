"""
Post-hoc temperature scaling for probability calibration.

Works on models that bake nn.Sigmoid() into the forward pass by recovering
pre-sigmoid logits via the inverse sigmoid (logit) transform, scaling by T,
then re-applying sigmoid.
"""

import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit  # sigmoid, inverse-sigmoid


_T_MIN = 0.01
_T_MAX = 10.0
_PROB_CLIP = 1e-7  # keep logit transform numerically stable


class TemperatureScaler:
    """
    Fits a single per-model temperature T on a validation set.
    T > 1 softens probabilities (reduces overconfidence).
    T < 1 sharpens probabilities.
    """

    def __init__(self):
        self.temperature: float = 1.0

    def _nll(self, T: float, logits: np.ndarray, targets: np.ndarray) -> float:
        """Binary cross-entropy of sigmoid(logits / T) vs targets."""
        scaled = expit(logits / T)
        scaled = np.clip(scaled, _PROB_CLIP, 1 - _PROB_CLIP)
        return -float(np.mean(
            targets * np.log(scaled) + (1 - targets) * np.log(1 - scaled)
        ))

    def fit(self, probs: np.ndarray, targets: np.ndarray) -> "TemperatureScaler":
        """
        Learn temperature from validation-set probabilities and binary targets.

        Args:
            probs:   (N, num_tasks) or (N,) float array of model probabilities in [0,1]
            targets: same shape as probs, binary {0, 1}
        """
        probs = np.asarray(probs, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if probs.shape != targets.shape:
            raise ValueError("Calibration probabilities and targets must have the same shape.")
        if np.isinf(targets).any():
            raise ValueError("Calibration labels must be binary or NaN, not infinite.")
        if (np.isfinite(targets) & ~np.isfinite(probs)).any():
            raise ValueError("Observed calibration probabilities must be finite.")

        flat_probs = probs.ravel()
        flat_targets = targets.ravel()

        # Drop NaN targets (unobserved tasks)
        valid = np.isfinite(flat_targets) & np.isfinite(flat_probs)
        flat_probs = flat_probs[valid]
        flat_targets = flat_targets[valid]
        if not len(flat_targets):
            raise ValueError("Calibration needs at least one observed validation label.")
        if not np.isin(flat_targets, [0, 1]).all() or not ((flat_probs >= 0) & (flat_probs <= 1)).all():
            raise ValueError("Calibration requires binary labels and probabilities in [0, 1].")

        # Recover pre-sigmoid logits
        flat_probs = np.clip(flat_probs, _PROB_CLIP, 1 - _PROB_CLIP)
        logits = logit(flat_probs)

        result = minimize_scalar(
            self._nll,
            args=(logits, flat_targets),
            bounds=(_T_MIN, _T_MAX),
            method="bounded",
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("Temperature optimization failed.")
        self.temperature = float(result.x) if result.fun <= self._nll(1.0, logits, flat_targets) else 1.0
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probability array."""
        probs = np.asarray(probs, dtype=float)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")
        if not np.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
            raise ValueError("Predicted probabilities must be finite and in [0, 1].")
        clipped = np.clip(probs, _PROB_CLIP, 1 - _PROB_CLIP)
        scaled_logits = logit(clipped) / self.temperature
        return expit(scaled_logits)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        with open(path) as f:
            data = json.load(f)
        scaler = cls()
        scaler.temperature = float(data["temperature"])
        return scaler

def fit_validation_rule(targets, probs, masks):
    from src.metrics import tune_task_thresholds
    observed = np.asarray(masks, dtype=bool)
    scaler = TemperatureScaler().fit(
        np.where(observed, probs, np.nan), np.where(observed, targets, np.nan))
    thresholds = tune_task_thresholds(targets, scaler.transform(probs), masks)
    return scaler, thresholds

def prediction_metrics(targets, probs, masks, scaler, thresholds):
    from src.metrics import compute_all_metrics
    return {
        "raw_default": compute_all_metrics(targets, probs, masks),
        "calibrated_default": compute_all_metrics(targets, scaler.transform(probs), masks),
        "calibrated_tuned": compute_all_metrics(targets, scaler.transform(probs), masks, thresholds),
    }
