# Paper Experiment Plan

This checklist captures the minimum evidence needed before positioning
M2G-Net as a strong Q1/A* empirical paper.

## Core Contribution Statement

Use restrained wording:

```text
We propose an integrated theory-guided multi-view multi-task framework
tailored to partially observed motorcycle risky-behavior outcomes.
```

Focus on three contributions:

1. Theory-guided multi-view representation of rider, contextual, environmental,
   and site-aware factors.
2. Task-specific gated fusion over five views plus the cross-level interaction
   representation.
3. Masked multi-task learning for partially observed risky-behavior outcomes.

## Figure Requirements

The model figure should explicitly include:

- `Individual x Context Interaction` between the multi-view encoders and gated
  fusion.
- Six gate inputs: rider role, rider traits, road context, environment,
  site-aware context, and interaction.
- `Task-specific observation masks` entering the loss.
- `Site-Aware Context View`, described as observed site features plus a
  regularized site-specific embedding.
- `View reliance` or `view contribution analysis`, not causal view importance.

## Required Tables

1. Dataset/task observation table: total N, observed N per task, positive rate.
2. Label correlation/co-occurrence table.
3. Model configuration table.
4. Baseline comparison table with mean, SD or CI, and p-value.
5. Per-task performance table with ROC-AUC, PR-AUC, F1, balanced accuracy,
   sensitivity, specificity, Brier score, and ECE.
6. Ablation table covering gated fusion, cross-level interaction, site-aware
   view, masks, uncertainty weighting, early-fusion only, and shared gate.
7. Gate/ablation/Integrated Gradients consistency table.
8. Fairness and calibration table by gender and age group.

## Scripts Added For Paper Tables

```bash
python scripts/evaluate/dataset_task_observation.py
python scripts/evaluate/fairness_calibration.py
```

These produce:

- `outputs/dataset_task_observation.csv`
- `outputs/label_correlation.csv`
- `outputs/label_cooccurrence.csv`
- `outputs/calibration_overall.csv`
- `outputs/fairness_calibration_by_group.csv`

## Baseline Requirements

Baselines should use the same train/validation/test protocol and task masks:

- Logistic regression.
- CART/decision tree or CHAID-style tree.
- Random forest.
- Tuned XGBoost.
- Tuned LightGBM.
- CatBoost if categorical handling is central.
- Early-fusion MLP.
- Shared-bottom multi-task MLP.
- Single-task MLPs.
- Site-aware or hierarchical logistic baseline if site effects are emphasized.

## Interpretation Requirements

Report interpretability in three layers:

1. Gate reliance: relative reliance on learned view representations.
2. Ablation importance: performance drop after removing each view/module.
3. Integrated Gradients: feature-level attribution.

Avoid causal language unless supported by a causal design.
