# Synthetic Out-of-Site Test Results

This file summarizes the current evaluation on
`data/test/independent_test_set.csv`. In the default repository state, this CSV
is synthetic. The results should therefore be interpreted as synthetic
out-of-site performance under the assumed data-generating process, not as
external validation on real intersections.

## Test Setup

| Item | Value |
|---|---:|
| Training data | `data/raw/synthetic_rider_data.csv` |
| Test data | `data/test/independent_test_set.csv` |
| Test samples | 2,000 |
| Site IDs | 32-50 |
| Site-specific embedding at test time | disabled |

For unseen sites, the model sets the site-specific embedding to zero and uses
observed site covariates only.

## Current Reported Results

| Task | ROC-AUC | PR-AUC | F1 | Balanced Acc. | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Red-light running | 0.7087 | 0.2055 | 0.0000 | 0.5000 | 0.0883 | 0.0099 |
| Turn-signal non-use | 0.6389 | 0.3799 | 0.0000 | 0.5000 | 0.1961 | 0.0261 |
| Helmet non-use | 0.8015 | 0.1999 | 0.0000 | 0.5000 | 0.0611 | 0.0159 |
| Mobile phone visibility/use | 0.7542 | 0.3440 | 0.0000 | 0.5000 | 0.1299 | 0.0190 |
| Macro average | 0.7258 | 0.2823 | 0.0000 | 0.5000 | 0.1189 | 0.0177 |

These values are from the retrained checkpoint after the fusion implementation
was aligned with the mathematical formulation.

The no-turn-signal task is the weakest task and should be reported separately,
not hidden behind macro-AUC. For real data, this task should use a mask so that
only riders for whom turn-signal behavior is observable are included.

F1 and balanced accuracy use the default threshold \(0.5\). Before paper
submission, select thresholds on the validation set or report threshold-free
metrics as primary outcomes.

## Reporting Guidance

Use cautious wording:

```text
The model was evaluated on a synthetic out-of-site test set generated under
the same assumed data-generating process.
```

Avoid stronger wording unless real held-out data are used:

```text
The model generalizes to unseen real intersections.
```

## Needed Before Strong Empirical Claims

1. Real observational held-out intersections or external validation data.
2. Task-specific label masks for partial-observation outcomes.
3. Repeated runs with confidence intervals.
4. Statistical comparison with tuned baselines.
5. Fairness and calibration checks for gender and age-group features.
