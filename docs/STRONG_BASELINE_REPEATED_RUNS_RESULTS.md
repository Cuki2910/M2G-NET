# Strong Baseline Repeated Runs Results

**Updated:** 2026-05-14

**Metric:** Macro ROC-AUC over 5 matched seeds.

**Data status:** synthetic proof-of-concept unless replaced by real observational data.

## Summary

| Model | Mean macro ROC-AUC | 95% CI | Delta vs M2G | Paired p-value |
|---|---:|---:|---:|---:|
| M2G-Net v2 | 0.7303 | [0.7141, 0.7464] | ref | ref |
| XGBoost | 0.6796 | [0.6644, 0.6947] | +0.0507 | 0.001255 |
| LightGBM | 0.7182 | [0.7015, 0.7349] | +0.0121 | 0.00711 |
| Random Forest | 0.6706 | [0.6603, 0.6809] | +0.0597 | 0.0003337 |
| CatBoost | 0.7310 | [0.7157, 0.7463] | -0.0007 | 0.6838 |

## Per-Seed Scores

| Seed | M2G-Net | XGBoost | LightGBM | Random Forest | CatBoost |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.7112 | 0.6657 | 0.6996 | 0.6588 | 0.7121 |
| 101 | 0.7334 | 0.6742 | 0.7193 | 0.6700 | 0.7350 |
| 2023 | 0.7329 | 0.6985 | 0.7298 | 0.6815 | 0.7387 |
| 777 | 0.7268 | 0.6826 | 0.7105 | 0.6743 | 0.7259 |
| 999 | 0.7471 | 0.6770 | 0.7317 | 0.6685 | 0.7432 |

## Notes

- Each seed uses the same train/validation/test split for M2G-Net and all baselines.
- Baselines use the current fixed hyperparameters in `scripts/evaluate/repeated_runs_significance.py`; these are not yet tuned or nested-CV optimized.
- Paired p-values compare M2G-Net against each baseline over matched seeds.
- CatBoost slightly exceeds M2G-Net on mean ROC-AUC by 0.0007, but the paired difference is not statistically significant (`p = 0.6838`).
