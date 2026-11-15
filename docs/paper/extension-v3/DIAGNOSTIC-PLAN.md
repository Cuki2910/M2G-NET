# Fixed M5 diagnostic round — declared before fitting

User approved the bounded diagnostic recommendation in FEASIBILITY-NOTES.md. This round is NOT a new hyperparameter search and does not promise target attainment.

## Scope

- Existing synthetic DEVELOPMENT generator seeds 1701–1710, all three mechanisms, fixed original split seed 42, training seeds 42/101, five M5 legacy members per seed = 300 fits.
- Architecture, BCE loss, gate prior, optimizer, training budget, patience, and early-stopping criterion unchanged. No new features, no oracle probabilities/latent labels, no DEVELOPMENT test predictions, no LOCKED TEST generation/access.
- Config adds only diagnostic output/report paths; all existing source hashes/caches from gate search retained as historical. Do not relabel stale FREEZE as current.

## Measurements

1. Each executed epoch: retain already-computed raw per-task training (dropout, mixed within-epoch weights) metrics and validation metrics. Those in-training training metrics are descriptive, NOT directly comparable with eval-mode validation for a generalization gap.
2. At validation-selected best checkpoint only: compute eval-mode training and validation raw-probability AUC/Brier per task and macro on observed/applicable labels. Training eval loader is nonshuffled. Preserve PyTorch and NumPy RNG around extra evaluation so telemetry does not change member initialization/sampling/checkpoint selection.
3. Save train/validation row identifiers, masks, labels, probabilities, epoch history, configuration/source/data/runtime identity and artifact hashes. Atomic member completion and per-unit process locking. Five-member ensemble evaluated by averaging probabilities; then metrics averaged over the two training seeds within each generator. Retain all 30 datasets and all tasks; no silent omissions.
4. Report eval-mode train–validation gaps, selected epochs, finite metric coverage, per-task ensemble metrics, and whether validation curves show continued learning or deterioration. No absolute gap threshold automatically labels underfit/overfit; gaps are diagnostic evidence, not proof of error decomposition.

## Verification before actual M5 fitting

Run full suite and data checks; telemetry-on/off must preserve exact selected state, epoch, predictions and post-fit RNG in a synthetic runnable regression. Existing gate prior, scoring, cache provenance and checkpoint tests remain required. Independent review of runner must confirm no extra forward passes inside training and no test/locked reads.

## Decision boundary

After the complete diagnostic table, declare a separate SMALL candidate grid motivated by the findings. Do not change architecture/regularization or tune epochs during this diagnostic run. Existing success criteria and all eight final comparators unchanged. Any later dev pass is selection evidence only; still requires explicit confirmed one-time LOCKED TEST.

## Expected cost

300 member fits, four isolated workers, plus final train evaluation; duration unverified until measured. No massive sweep or paid model upgrade authorized/required by this step.
