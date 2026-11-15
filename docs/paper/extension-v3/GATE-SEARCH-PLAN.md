# Gate development round — declaration before fitting

User authorized gate changes, parallel subagents, and continued DEVELOPMENT search. No guarantee of success; no changing opponents, metrics, datasets, or thresholds after scores. LOCKED TEST 1711–1730 remains forbidden until a new freeze and explicit confirmation. No commit/submission.

## Registered directions

1. `nonlinear`: replace per-task affine gate logits by Linear → GELU → Linear (hidden size from config); preserve sparsemax, temperature and uniform prior.
2. `task_blend`: preserve affine gate; replace shared gate/early blend scalar by one scalar per task.
3. `nonlinear_task_blend`: combine those two changes.

Control: `legacy` = existing M5 BCE probability ensemble; skip off. Default old models stay legacy. All settings in config.py. No hand-designed generator-specific products, no latent probabilities/oracle used for selection.

## Execution and checks

1. Data/full-test/math preflight → all checks must pass before fits. Two document failures discovered before this round require repair; user confirmed single-author identity, not a waiver.
2. Implement optional variants → unit numeric/shape/gradient checks, legacy initialization/RNG/output equivalence, checkpoint reload and independent math review.
3. Development fit → generator seeds 1701–1710, validation only, training seeds 42 and 101, 5 initialization members. Small explicitly marked exploratory subsets allowed for debugging; never count them as full screening. Holdout discarded, not predicted/scored. Fit each variant under identical splits; isolated subprocesses, source/config/data-identified atomic caches. Do not mutate config during fits.
4. Compare with all 8 declared opponents using their validation predictions, matching row IDs, labels, and masks. Average the two training seeds per generated set before paired summaries. Report mean ± MCSE, min/max, positive count, no p-values. DEV screen mirrors locked mean > .005 and >=90% positives for both interaction/partial vs every opponent; additive no mean loss vs catboost/old and aggregate Brier no worse than catboost/early_fusion_mlp. Dev results are adaptive selection evidence, not final success.
5. If no variant passes, retain complete negative results; declare any next search round before its fits. If dev screen passes, independently review, freeze selected configuration plus runner/source hashes, STOP for confirmation. Never repeatedly evaluate LOCKED TEST until it passes.

## Corrections to prior reports

The earlier structure-aware LR is not an upper bound on a neural model. Its equality to M5 does not prove zero architectural headroom; claims of a numerical noise floor were unsupported. Also, the first run_v3_dev.py invokes the full paper pipeline which produces DEVELOPMENT test outputs even though its summary uses validation only. This exposure must be disclosed; no new runner may repeat it. Earlier variants cache lacked source/config/data identity and row-alignment checks. Use a fresh gate-search cache, never silently reuse those outputs.

## Re-runs required by config/source changes

Re-run the complete gate-search dev control/candidates and gate/checkpoint/ensemble tests before choosing a model. The old frozen v2 publication scripts detect changed source hashes: do not rewrite their run-plan or results. Ablation, gate stability, calibration, independent/site-out evaluations would need separate matched re-runs for any adopted new architecture; old published numbers are not new-variant evidence.

## Status

Implementation and independent reviews in progress. Full training blocked pending repair of the two manuscript packaging tests. Existing FREEZE.json is now stale during development; do not execute locked runner against it.
