# Cleanup Notes

This file records repository-maintenance changes. It is not a paper-readiness
assessment and should not be included as academic supplementary material.

## Scope

The cleanup organized scripts, documentation, outputs, and checkpoints into a
clearer repository structure. It does not imply that the model is ready for
deployment or paper submission.

## Current Repository Status

The repository now supports:

- model training;
- baseline comparison;
- synthetic out-of-site evaluation;
- ablation experiments;
- gate-weight and Integrated Gradients analyses;
- case-study generation;
- masked multi-task loss for partially observed labels.

## Important Caveats

- The default training and test CSVs are synthetic.
- Strong real-world generalization claims require real observational
  validation data.
- Partially observed outcomes require task-specific masks.
- Baselines should be tuned and compared with confidence intervals and
  statistical tests before paper submission.

See `README.md`, `docs/MATHEMATICAL_FORMULAS.md`, and
`docs/ARCHITECTURE_VISUALIZATION.md` for the current technical references.
