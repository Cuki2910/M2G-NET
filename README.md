# TG-MVMT-GFNet v2

Theory-Guided Multi-View Multi-Task Gated Fusion Network for motorcycle risky
behavior prediction.

This repository contains a research implementation for training, evaluating,
comparing baselines, and generating explanation analyses. The default data
files are synthetic proof-of-concept CSVs; results from them should not be
presented as real-world external validation.

## Current Scope

The project is suitable for method development and reproducibility testing.
For a Q1/A* empirical submission, replace or validate the synthetic data with
real observational data, report repeated-run uncertainty, and run statistical
significance tests against tuned baselines.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py
python baselines/run_all_baselines.py
python scripts/evaluate/ablation.py
python scripts/evaluate/test_on_independent_set.py
python scripts/evaluate/dataset_task_observation.py
python scripts/evaluate/fairness_calibration.py
python scripts/explain/visualize.py
python scripts/explain/ig_explain.py
```

## Data

Default paths in `config.py`:

| File | Role |
|---|---|
| `data/raw/synthetic_rider_data.csv` | synthetic training/evaluation data |
| `data/test/independent_test_set.csv` | synthetic out-of-site test data |

The current independent test should be described as a synthetic out-of-site
test under the same assumed data-generating process. Do not describe it as
external validation on real unseen intersections unless the CSV is replaced by
real held-out observations.

## Partial-Observation Labels

The code supports task-specific missing-label masks for outcomes that are only
observable on a subset of riders, such as red-light running and turn-signal
usage.

Optional mask columns:

| Task | Mask column |
|---|---|
| `red_light_running` | `red_light_running_observed` |
| `no_turn_signal` | `no_turn_signal_observed` |
| `helmet_nonuse` | `helmet_nonuse_observed` |
| `mobile_phone_use` | `mobile_phone_use_observed` |

If these columns are absent, the pipeline defaults to all labels observed for
backward compatibility with the current synthetic datasets.

## Architecture

TG-MVMT-GFNet uses:

1. Five view-specific encoders: rider role, rider traits, road context,
   environment, and site-aware context.
2. Cross-level interaction between individual and contextual representations.
3. Task-specific gated fusion with temperature annealing.
4. Residual blending with an early-fusion representation.
5. Four binary task heads.
6. Masked uncertainty-weighted multi-task loss.

## Method Contributions

The intended methodological contribution is an integrated framework, not a
claim that every component is individually new:

1. Theory-guided multi-view representation for motorcycle risky-behavior
   prediction, separating rider role, rider traits, road context, environment,
   and site-aware context.
2. Task-specific gated fusion, allowing each behavior outcome to learn a
   different reliance pattern over the view representations and cross-level
   interaction.
3. Masked multi-task learning for partially observed risky-behavior labels,
   such as red-light running and turn-signal non-use.

The site-aware view uses observed site covariates plus a regularized
site-specific embedding. For unseen sites, the site-specific embedding is
zeroed out and the model relies on observed site features.

## Implemented Configuration

The implemented hyperparameters are defined in `config.py`:

| Parameter | Value |
|---|---:|
| `EMBEDDING_DIM` | 8 |
| `HIDDEN_DIM` | 32 |
| `VIEW_DIM` | 16 |
| `INTERACTION_DIM` | 16 |
| `DROPOUT_RATE` | 0.30 |
| `TEMPERATURE_INIT` | 2.0 |
| `TEMPERATURE_FINAL` | 1.0 |
| `GATE_PRIOR_WEIGHT` | 1.0 |
| `LEARNING_RATE` | 1e-3 |
| `WEIGHT_DECAY` | 1e-4 |
| `BATCH_SIZE` | 64 |
| `MAX_EPOCHS` | 150 |
| `USE_FOCAL_LOSS` | False |

## Repository Structure

```text
src/
  data_pipeline.py     data loading, encoding, task masks
  views.py             view-specific and site-aware encoders
  interaction.py       cross-level interaction module
  fusion.py            task-specific gated fusion
  loss.py              masked uncertainty-weighted loss
  metrics.py           masked metrics
  model.py             TG-MVMT-GFNet v2 model

scripts/
  train.py
  evaluate/
  explain/
  search/

baselines/
  run_all_baselines.py

docs/
  MATHEMATICAL_FORMULAS.md
  ARCHITECTURE_VISUALIZATION.md
```

## Interpreting Results

Gate weights indicate the model's relative reliance on learned view
representations. They should not be treated as causal feature importance
without supporting evidence from ablation, Integrated Gradients, and stability
checks across folds and random seeds.

## Recommended Next Steps Before Paper Submission

1. Replace or validate the synthetic data with real observational data.
2. Use task masks for partially observed labels.
3. ~~Report repeated runs with confidence intervals.~~ *(Completed: `scripts/evaluate/repeated_runs_significance.py`)*
4. ~~Add statistical tests against tuned tabular baselines.~~ *(Completed: `scripts/evaluate/repeated_runs_significance.py`)*
5. Add fairness checks for gender and age-group features.
6. Keep paper claims scoped to the evidence: synthetic proof-of-concept,
   synthetic out-of-site test, or real-world validation as appropriate.

## Key Documentation

| File | Purpose |
|---|---|
| `docs/MATHEMATICAL_FORMULAS.md` | final mathematical formulation |
| `docs/ARCHITECTURE_VISUALIZATION.md` | implemented architecture reference |
| `docs/INDEPENDENT_TEST_RESULTS.md` | current synthetic test report |
| `docs/tracking.md` | experiment notes |
| `outputs/tg_mvmt_gfnet_architecture_v2.svg` | paper-facing architecture figure |
| `outputs/dataset_task_observation.csv` | task observation and positive-rate table |
| `outputs/label_correlation.csv` | label correlation table |
| `outputs/fairness_calibration_by_group.csv` | gender/age subgroup metrics |

## Citation Note

The uncertainty-weighted multi-task loss follows Kendall et al. (2018). See
`docs/MATHEMATICAL_FORMULAS.md` for the BibTeX entry.
