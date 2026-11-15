# Weight-decay comparison — completed DEVELOPMENT round

600/600 member fits and scoring completed (exit0). Two Adam decays, legacy M5, three synthetic mechanisms, ten generators1701–1710, two training seeds42/101, five members. Validation only. No LOCKED TEST evaluation.

**Neither configuration passes the full development screen. Stronger decay is not adopted; retain1e-4.**

## Mean AUC improvement vs old m2g_net

| Variant | Mechanism | Mean ± MCSE | Min–max | Positive/10 |
|---|---|---:|---:|---:|
| wd_base | additive | +0.005149 ± 0.001546 | -0.000195 to +0.017804 | 9/10 |
| wd_base | interaction | +0.003909 ± 0.001617 | -0.003385 to +0.011976 | 8/10 |
| wd_base | partial | +0.003219 ± 0.000601 | -0.000534 to +0.005272 | 9/10 |
| wd_strong | additive | +0.001833 ± 0.001043 | -0.001551 to +0.009548 | 7/10 |
| wd_strong | interaction | +0.002634 ± 0.001371 | -0.005778 to +0.007756 | 7/10 |
| wd_strong | partial | +0.003044 ± 0.001239 | -0.003560 to +0.009795 | 8/10 |

## Stronger decay vs control (paired generator deltas)

Positive favors stronger decay: AUC strong−base; Brier base−strong.

| Mechanism | Metric | Mean ± MCSE | Min–max | Positive/10 |
|---|---|---:|---:|---:|
| additive | auc_gain | -0.003316 ± 0.000817 | -0.008256 to +0.000549 | 1/10 |
| additive | brier_gain | -0.000605 ± 0.000206 | -0.001686 to +0.000124 | 2/10 |
| interaction | auc_gain | -0.001274 ± 0.001081 | -0.006481 to +0.002858 | 4/10 |
| interaction | brier_gain | -0.000159 ± 0.000389 | -0.002133 to +0.001306 | 5/10 |
| partial | auc_gain | -0.000175 ± 0.001248 | -0.006592 to +0.006776 | 5/10 |
| partial | brier_gain | -0.000024 ± 0.000407 | -0.001950 to +0.002190 | 4/10 |

## Full-screen outcomes

Mean>.005 and >=9/10 positive for dev interaction/partial vs every opponent, unchanged final locked rule18/20. All additive mean noninferiority and aggregate Brier conditions pass for both configs. Neither passes all AUC checks. No p-values; no claim about held-out success.

### wd_base failed checks
- interaction vs m2g_net: auc_superiority
- interaction vs shared_gate: auc_superiority
- partial vs m2g_net: auc_superiority
- partial vs shared_gate: auc_superiority
- partial vs no_interaction: auc_superiority
- partial vs early_fusion: auc_superiority

### wd_strong failed checks
- interaction vs m2g_net: auc_superiority
- interaction vs shared_gate: auc_superiority
- interaction vs no_interaction: auc_superiority
- interaction vs early_fusion: auc_superiority
- partial vs m2g_net: auc_superiority
- partial vs shared_gate: auc_superiority
- partial vs no_interaction: auc_superiority
- partial vs early_fusion: auc_superiority

## Verification and scope

- Original control reproducibility: 300/300 member validation arrays compared against prior diagnostic M5, maximum probability difference=0.0; IDs/labels/masks identical.
- Preflight82 tests/78subtests passed;30datasets and480opponent validation contracts checked; independent optimizer/cache/scoring review approved. flake8 absent; not claimed pass.
- Applies existing Adam weight decay to model AND learned loss parameters, not AdamW or gate-only penalty. No config/source edits during fits.
- Secondary per-task validation metrics: WEIGHT-DECAY-PER-TASK.csv. No post-hoc task-specific model choice.
- Full eight-opponent summaries, unit scores, criteria and provenance: `docs/paper/extension-v3/gate-search/score-1794734988511532000`.
- Direct comparison per generator: WEIGHT-DECAY-PAIRED.csv.
- Conclusion: the specific stronger-decay hypothesis is not supported. This does not prove all regularization changes futile or prove irreducible noise.
- No further round started; no new freeze, commit, push, submission, or test-data access. Existing FREEZE remains stale after source changes.
