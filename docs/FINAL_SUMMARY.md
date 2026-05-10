# TG-MVMT-GFNet v2 Summary

This repository implements TG-MVMT-GFNet v2 for method development and
reproducibility testing. The default data files are synthetic, so current
results should be interpreted as proof-of-concept evidence rather than
real-world external validation.

## Current Implementation

| Component | Status |
|---|---|
| Multi-view encoders | implemented |
| Cross-level interaction | implemented |
| Task-specific gated fusion | implemented |
| Residual early-fusion blend | implemented |
| Masked multi-task loss | implemented |
| Baseline scripts | available |
| Ablation and explanation scripts | available |

## Current Reported Metrics

These numbers come from the existing synthetic CSVs and saved experiment
artifacts.

| Evaluation | Macro ROC-AUC | Notes |
|---|---:|---|
| Random split test | 0.7142 | synthetic data |
| Leave-intersection-out CV | 0.7413 +/- 0.0176 | synthetic site holdout |
| Synthetic out-of-site test | 0.7258 | sites 32-50, synthetic |

Per-task ROC-AUC on the synthetic out-of-site test:

| Task | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| Red-light running | 0.7087 | 0.2055 | 0.0883 | 0.0099 |
| Turn-signal non-use | 0.6389 | 0.3799 | 0.1961 | 0.0261 |
| Helmet non-use | 0.8015 | 0.1999 | 0.0611 | 0.0159 |
| Mobile phone visibility/use | 0.7542 | 0.3440 | 0.1299 | 0.0190 |

## Main Limitations Before Paper Submission

1. The default data are synthetic.
2. Red-light and turn-signal outcomes require task-specific observation masks
   on real data.
3. Strong empirical claims require real held-out intersections or external
   validation.
4. Baselines should be tuned under the same protocol and compared with
   confidence intervals and significance tests.
5. Fairness and calibration should be reported for gender and age-group
   features.

## Recommended Paper Positioning

The safest current framing is:

```text
A theory-guided multi-view multi-task neural architecture for jointly modeling
partially observed risky riding behaviors, evaluated as a synthetic
proof-of-concept and prepared for validation on real observational data.
```

With real data and masked labels, the contribution can be strengthened toward:

```text
A theory-guided multi-view multi-task architecture that jointly models
partially observed risky riding behaviors while preserving site-level
generalization and task-specific view reliance.
```
