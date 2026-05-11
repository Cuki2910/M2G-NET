# M2G-Net v2

Incorporating Behavioral Theory into Deep Learning: A Multi-View Multi-Task Gated Fusion Network (M2G-Net) for motorcycle risky behavior prediction.

Guided by transportation behavioral models (e.g., Theory of Planned Behavior), this architecture models the complex interaction between rider characteristics and environmental context. This research provides an interpretable tool capable of handling missing observation labels, aiming to support policymakers in designing safer intersections and optimizing traffic enforcement.

This repository contains a research implementation for training, evaluating, comparing baselines, and generating explanation analyses for multi-task behavioral prediction.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py
python baselines/run_all_baselines.py
python scripts/evaluate/ablation.py
python scripts/evaluate/test_on_independent_set.py
python scripts/evaluate/repeated_runs_significance.py
python scripts/evaluate/interpret.py              # Gate weights analysis
python scripts/explain/ig_explain.py              # Integrated Gradients attribution
python scripts/explain/visualize.py
```

## Architecture

M2G-Net uses:
1. **Five view-specific encoders:** Rider role, rider traits, road context, environment, and site-aware context.
2. **Cross-level interaction:** Blends individual and contextual representations.
3. **Gated fusion:** Task-specific gated fusion with temperature annealing.
4. **Masked multi-task loss:** Handles partially observed risky-behavior labels.

## Repository Structure

```text
src/
  data_pipeline.py     data loading, encoding, task masks
  views.py             view-specific and site-aware encoders
  interaction.py       cross-level interaction module
  fusion.py            task-specific gated fusion
  loss.py              masked uncertainty-weighted loss
  metrics.py           masked metrics
  model.py             M2G-Net v2 model

scripts/
  train.py             Main training pipeline
  evaluate/            Test sets, ablations, fairness, significance tests
  explain/             Integrated Gradients & visualizations
  search/              Hyperparameter search

baselines/             Tabular and MLP baselines
docs/                  Mathematical formulations and architecture visual
```

## Key Documentation

- **Architecture:** `docs/ARCHITECTURE_VISUALIZATION.md`
- **Math Formulas:** `docs/MATHEMATICAL_FORMULAS.md`
- **Results:** `docs/INDEPENDENT_TEST_RESULTS.md`
