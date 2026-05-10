# Experiment Tracking Notes

This file records current experiment notes for the repository. The default
data are synthetic, so all metrics below are descriptive proof-of-concept
results unless real observational data are substituted.

## Data

| Split/report | Source |
|---|---|
| Training/evaluation | `data/raw/synthetic_rider_data.csv` |
| Out-of-site test | `data/test/independent_test_set.csv` |

## Current Metrics

Best checkpoint after retraining:

| Field | Value |
|---|---:|
| Best epoch | 36 |
| Best validation macro ROC-AUC | 0.7159 |

| Evaluation | Macro ROC-AUC | Interpretation |
|---|---:|---|
| Random split test | 0.7142 | synthetic random split |
| Leave-intersection-out CV | 0.7413 +/- 0.0176 | synthetic site holdout |
| Synthetic out-of-site test | 0.7258 | synthetic sites 32-50 |

Per-task ROC-AUC on the synthetic out-of-site test:

| Task | ROC-AUC |
|---|---:|
| red_light_running | 0.7087 |
| no_turn_signal | 0.6389 |
| helmet_nonuse | 0.8015 |
| mobile_phone_use | 0.7542 |

## Task Uncertainty

The implementation now reports learned task uncertainty as `tau_k`, with
`eta_k = log(tau_k^2)` in the loss.

| Task | tau_k |
|---|---:|
| red_light_running | 0.5498 |
| no_turn_signal | 0.7627 |
| helmet_nonuse | 0.4483 |
| mobile_phone_use | 0.6486 |

After retraining with interaction included in the gated weighted sum:

| Task | tau_k |
|---|---:|
| red_light_running | 0.5482 |
| no_turn_signal | 0.7632 |
| helmet_nonuse | 0.4505 |
| mobile_phone_use | 0.6506 |

Alpha blend after retraining: `0.5496`.

## Interpretation Notes

Gate weights should be described as relative reliance on learned view
representations. They are not causal feature importance unless supported by
ablation, Integrated Gradients, and stability analyses.

## Remaining Work Before Strong Paper Claims

1. Replace or validate synthetic data with real observational data.
2. Add task-observation masks for partial labels in real data.
3. Run repeated seeds and confidence intervals.
4. Tune tabular baselines under the same protocol.
5. Add significance tests and fairness/calibration analyses.
