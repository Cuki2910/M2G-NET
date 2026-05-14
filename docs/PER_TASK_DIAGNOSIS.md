# Per-Task Diagnosis: M2G-Net v2 vs CatBoost

**Data status:** synthetic proof-of-concept.
**Seeds:** [42, 101, 2023, 777, 999]

## Macro ROC-AUC (mean over 5 seeds)

| Model | Mean macro ROC-AUC |
|---|---:|
| M2G-Net v2 | 0.7303 |
| CatBoost   | 0.7310 |

## ROC-AUC per task (mean +/- std, 5 seeds)

| Task | M2G-Net | CatBoost | Delta | Winner |
|---|---:|---:|---:|---|
| Red-light running | 0.7134 +/- 0.0228 | 0.7137 +/- 0.0191 | -0.0004 | TIE |
| Turn-signal non-use | 0.6366 +/- 0.0105 | 0.6348 +/- 0.0113 | +0.0018 | M2G |
| Helmet non-use | 0.8196 +/- 0.0190 | 0.8214 +/- 0.0198 | -0.0018 | CatBoost |
| Mobile phone visibility/use | 0.7517 +/- 0.0185 | 0.7540 +/- 0.0188 | -0.0023 | CatBoost |

## PR-AUC per task (mean, 5 seeds)

| Task | M2G-Net | CatBoost | Delta |
|---|---:|---:|---:|
| Red-light running | 0.1995 | 0.2149 | -0.0154 |
| Turn-signal non-use | 0.4030 | 0.4018 | +0.0012 |
| Helmet non-use | 0.2161 | 0.2195 | -0.0034 |
| Mobile phone visibility/use | 0.3649 | 0.3695 | -0.0046 |

## F1 per task (threshold=0.5, mean, 5 seeds)

| Task | M2G-Net | CatBoost | Delta |
|---|---:|---:|---:|
| Red-light running | 0.0000 | 0.0000 | +0.0000 |
| Turn-signal non-use | 0.0000 | 0.0281 | -0.0281 |
| Helmet non-use | 0.0000 | 0.0000 | +0.0000 |
| Mobile phone visibility/use | 0.0000 | 0.0011 | -0.0011 |

## Stability (ROC-AUC std dev across seeds)

| Task | M2G std | CatBoost std | More stable |
|---|---:|---:|---|
| Red-light running | 0.0228 | 0.0191 | CatBoost |
| Turn-signal non-use | 0.0105 | 0.0113 | Similar |
| Helmet non-use | 0.0190 | 0.0198 | Similar |
| Mobile phone visibility/use | 0.0185 | 0.0188 | Similar |

## Notes

- M2G-Net trained with `fit_model_for_split` (same early stopping as main training).
- CatBoost uses 100 iterations, lr=0.05, depth=6, Logloss.
- Delta = M2G mean - CatBoost mean; positive = M2G leads.
- Winner threshold: |delta| > 0.001 (within threshold = TIE).
