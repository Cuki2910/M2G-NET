# Current Technical Metrics

**Updated:** 2026-05-14  
**Data status:** synthetic proof-of-concept unless replaced by real observational data.

This file is the tracked source for headline metrics. Large logs and plots remain
under ignored `outputs/`.

## After Retrain

| Metric | Value |
|---|---:|
| Random-split macro ROC-AUC | 0.7095 |
| Random-split macro PR-AUC | 0.2894 |
| Random-split tuned macro F1 | 0.3852 |
| Random-split tuned balanced accuracy | 0.6673 |
| Macro ECE before temperature scaling | 0.1464 |
| Macro ECE after temperature scaling | 0.0135 |
| Temperature | 0.3780 |
| Synthetic independent macro ROC-AUC | 0.7248 |
| Synthetic independent tuned macro F1 | 0.3848 |
| Synthetic independent tuned balanced accuracy | 0.6806 |

## Repeated Runs

| Model | Mean macro ROC-AUC | 95% CI |
|---|---:|---:|
| TG-MVMT-GFNet v2 | 0.7303 | [0.7141, 0.7464] |
| CatBoost | 0.7310 | [0.7157, 0.7463] |
| LightGBM | 0.7182 | [0.7015, 0.7349] |
| XGBoost | 0.6796 | [0.6644, 0.6947] |
| Random Forest | 0.6706 | [0.6603, 0.6809] |

Paired tests versus TG-MVMT-GFNet v2:

| Baseline | Mean delta M2G - baseline | t-statistic | p-value |
|---|---:|---:|---:|
| XGBoost | +0.0507 | 8.1132 | 0.001255 |
| LightGBM | +0.0121 | 5.0741 | 0.007110 |
| Random Forest | +0.0597 | 11.4352 | 0.0003337 |
| CatBoost | -0.0007 | -0.4383 | 0.6838 |

Full repeated-run table: `docs/STRONG_BASELINE_REPEATED_RUNS_RESULTS.md`.

## Single-Split Baselines

| Model | Macro ROC-AUC |
|---|---:|
| TG-MVMT-GFNet v2 | 0.7095 |
| CatBoost | 0.7121 |
| Early-Fusion MLP | 0.7085 |
| Decision Tree | 0.7069 |
| Single-Task MLP | 0.7013 |
| LightGBM | 0.6996 |
| XGBoost | 0.6657 |
| Random Forest | 0.6588 |
| Logistic Regression | 0.6461 |

## Interpretation Limits

- These metrics support methodological/prototype feasibility, not real-world external validation.
- Gate weights should be reported as supplementary attention diagnostics, not causal effects or stable feature importance.
- For probability reliability, report calibrated ECE after temperature scaling alongside uncalibrated ECE.
