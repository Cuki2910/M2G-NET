# Weight-decay comparison — declared before fits

User said 'bắt đầu' after the fixed diagnostic recommendation. Bounded comparison only, not open-ended tuning.

## Registered configurations

- `wd_base`: original M5, Adam weight_decay=1e-4.
- `wd_strong`: same model, Adam weight_decay=1e-3.

Both values are registered centrally as config.V3_DECAY_VARIANTS. Global cfg.WEIGHT_DECAY remains unchanged at1e-4. Explicit fit keyword selects the value; no in-process config mutation. Applies to the existing Adam parameter list (model AND learned loss parameters), not AdamW, not a gate-only penalty. No changes to architecture, dropout .30, learning rate, batches, gate prior, uncertainty-weighted BCE, five members, or selection rule. No new mathematical model/loss formula.

## Scope and verification

1. Preflight full suite and all DEVELOPMENT data/480 opponent validation contracts; independent code review of optimizer dispatch/cache key.
2. Fresh two-configuration caches: all3 mechanisms × generator seeds1701–1710 × training seeds42/101 × five member seeds =600 fits (300 per configuration). Old caches remain historical, not silently reused after source/config changes. Shared robust gate-dev runner uses named decay configs mapped to legacy gate and explicit decay; thin wrapper fixes full scope.
3. Select checkpoints using validation macro AUC only, original patience/budget. No test/locked/oracle predictions. Four isolated subprocess workers. Source/config fixed for entire run.
4. Score with same observed validation labels and exact8opponent pipeline; training-seed metrics averaged per generator; summaries mean±MCSE/min–max/positive sets. AUC positives >=9/10 and mean>.005 for interaction/partial against every opponent; additive mean not below old/catboost; aggregate across3mechanisms Brier not worse than early_fusion_mlp/catboost. These are dev-screen conditions, not final held-out criteria.
5. Report paired wd_strong−wd_base AUC and Brier, per-task scores as secondary diagnosis, no post-hoc task-specific selection. If no full-screen pass, report honestly without changing thresholds or picking favorable datasets. If passes, still requires review, new freeze, explicit user confirmation, then exactly one locked evaluation.

## Motivation and limitations

DIAGNOSTIC-RESULTS.md showed eval-mode train-validation gaps, particularly partial red-light running. Stronger decay is a hypothesis, not proven cure. Gaps do not prove under/overfitting mechanisms; repeated dev selection can be optimistic. Only two fixed values, retain all negative findings. No extra generator data, pooling, latent probabilities, altered baselines, p-values, or publication claims.

## Required re-evaluations

config/source changes make prior FREEZE stale; do not edit frozen v2 run-plan/results. Re-run full current control/candidate dev comparisons and tests. If adopted, any future new-variant ablation/calibration/site-out results require matched reruns under a separately authorized plan; old published results cannot be re-labelled as strong-decay results.

Status: implementation/preflight; no weight-decay comparison fit started at declaration.
