# Baseline Comparison Results — M2G-NET v2

**Date**: 2026-05-14
**Checkpoint**: `checkpoints/best_model.pt`
**Data**: `data/raw/synthetic_rider_data.csv` (9,164 samples, 4 tasks, synthetic proof-of-concept)

---

## 1. Discrimination (Macro ROC-AUC)

### Single-run comparison

| Model | Macro ROC-AUC | vs. M2G-NET |
|-------|:------------:|:-----------:|
| **M2G-NET v2 (full)** | **0.7343** | — |
| Decision Tree | 0.7069 | -0.0274 |
| Early-Fusion MLP | 0.7056 | -0.0287 |
| LightGBM | 0.6996 | -0.0347 |
| Single-Task MLP | 0.7009 | -0.0334 |
| XGBoost | 0.6657 | -0.0686 |
| Random Forest | 0.6588 | -0.0755 |
| Logistic Regression | 0.6461 | -0.0882 |

### Statistical significance (5 runs × random seeds)

| Model | Mean ROC-AUC | 95% CI |
|-------|:------------:|:------:|
| **M2G-NET v2** | **0.7304** | [0.7148, 0.7459] |
| XGBoost | 0.6796 | [0.6644, 0.6947] |

**Paired t-test (M2G-NET vs. XGBoost)**: t = 8.294, p = 0.00115 → **statistically significant** (p < 0.05)

M2G-NET consistently outperforms XGBoost across all 5 seeds with non-overlapping confidence intervals.

---

## 2. Calibration (Macro ECE — lower is better)

| Model | Macro ECE | Quality |
|-------|:---------:|:-------:|
| Logistic Regression | 0.0113 | Excellent |
| Early-Fusion MLP | 0.0143 | Excellent |
| Decision Tree | 0.0172 | Excellent |
| Single-Task MLP | 0.0195 | Excellent |
| LightGBM | 0.0221 | Excellent |
| Random Forest | 0.0693 | Good |
| XGBoost | 0.0818 | Good |
| **M2G-NET v2** | **0.1570** | **Poor** |

**Finding**: M2G-NET leads on discrimination (ROC-AUC) but has significantly worse calibration than all baselines. Temperature scaling is needed before deployment.

Per-task ECE detail:

| Task | ECE | Assessment |
|------|:---:|:----------:|
| Red Light Running | 0.1914 | Poor |
| Helmet Non-use | 0.1650 | Poor |
| Mobile Phone Use | 0.1474 | Moderate |
| No Turn Signal | 0.1240 | Moderate |

---

## 3. View Importance (Ablation)

| View removed | AUC without view | AUC drop | Importance |
|-------------|:----------------:|:--------:|:----------:|
| `rider_role` | 0.6065 | -0.1278 | Critical |
| `road_context` | 0.6810 | -0.0533 | High |
| `environment` | 0.7048 | -0.0295 | Moderate |
| `rider_traits` | 0.7338 | -0.0005 | Negligible |
| `site` | 0.7343 | 0.0000 | None (synthetic data) |

**Finding**: `rider_role` is by far the most important view. `site` embedding contributes nothing on synthetic data — expected, as all samples share the same site distribution.

---

## 4. Cross-Site Generalization (Leave-Site-Out)

| Metric | Value |
|--------|:-----:|
| **Mean AUC (G̅_LSO)** | **0.7274** |
| Std Dev (s_G) | 0.0171 |
| Folds completed | 31 |
| Min AUC (worst site) | 0.6989 |
| Max AUC (best site) | 0.7643 |
| **Gap (Δ_site)** | **-0.0159** |

**Finding**: LSO mean AUC (0.7274) is **higher** than random-split test AUC (0.7115) by 0.0159 — negative gap means the model generalizes **better** to unseen sites than to random held-out samples from seen sites. This is unexpected and suggests the synthetic data's site structure may not reflect real cross-site distribution shift. With real observational data, a positive gap (LSO < random-split) is more typical.

---

## 5. Key Findings

1. **M2G-NET discrimination is best overall**: macro ROC-AUC 0.7343 (single run) / 0.7304 ± CI (5 runs), statistically significantly better than XGBoost (p = 0.00115).
2. **Calibration is the main weakness**: ECE 0.1570 vs. 0.0113–0.0818 for baselines. Confidence scores are systematically overconfident (predicted probabilities exceed true frequencies across all tasks).
3. **Multi-view architecture earns its complexity**: removing `rider_role` alone drops AUC by 0.1278 — the view-specific encoding contributes real signal, not just parameter inflation.
4. **`rider_traits` and `site` need review**: near-zero contribution on synthetic data may indicate feature redundancy or synthetic data limitations; should be re-evaluated with real observational data.

---

## 6. Recommended Next Steps

1. Run `scripts/evaluate/temperature_scaling.py` — fit post-hoc temperature to improve ECE before any deployment claim.
2. Re-examine `rider_traits` feature set — negligible contribution warrants feature importance analysis via `scripts/explain/ig_explain.py`.
3. When real data is available: re-run this full comparison — site embedding and rider_traits contributions may change substantially.

---

## Config Snapshot

| Parameter | Value |
|-----------|-------|
| Checkpoint | `best_model.pt` |
| Tasks (K) | 4 |
| Views (V) | 5 |
| Gate type | sparsemax + prior smoothing |
| Data | synthetic (9,164 samples) |
| Significance runs | 5 (seeds: 42, 101, 2023, 777, 999) |
