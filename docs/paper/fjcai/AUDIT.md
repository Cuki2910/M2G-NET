# FJCAI 2027 package audit (2026-11-15)

Build: tectonic (XeTeX), `main.tex` -> `main.pdf`, 6 pages (rule: 6-8), no undefined refs/cites, 11/11 bibitems cited, IEEEtran.cls unchanged.
Numbers: Tables I-IV and quoted deltas match `../paired-summary.csv`, `../results-summary.csv`, `../tables-supplement.md` (S1-S3) and `../number-audit.md`.
Scope: synthetic data only; negative results retained; no significance or real-world claims.

Not verified / human-only:
- Double-blind: author block is filled; blank it if CMT/CFP requires anonymity (unverified).
- Originality / not submitted elsewhere: needs author confirmation.
- Cutoff time/timezone for 15/11/2026 unverified.
- Submission via CMT (https://cmt3.research.microsoft.com/FJCAI2027) is done by the author; nothing was submitted.
- Build uses system Times New Roman via fontspec (XeTeX); with pdflatex the file falls back to `vietnam` package (untested).
- `PAPER_TASK_STATE.json` P29 not modified.

## Citation pass (cite-after-writing, 2026-11-15)
Added 2 refs after search + abstract check (abstract only, full text not read): Adam (Kingma & Ba, arXiv:1412.6980) for the optimizer; CatBoost (Prokhorenkova et al., arXiv:1706.09516) for the classifier. Bibliography now 13 entries, all cited; 6 pages, no undefined refs.
No source found/inserted for: Brier score, ROC-AUC/AP, dropout (only Hinton 2012 co-adaptation preprint surfaced, not the dropout JMLR paper; not inserted), logistic regression. Existing refs are from the earlier audited bibliography; not re-verified in this pass.
Backup of previous tex: /tmp/main.tex.before_cite

## Recent-reference pass + final render check (2026-11-15)

Added 7 references (abstract-level verification via OpenAlex/Crossref; full text NOT read):
- [14] Ding et al., CVPR 2023, pp. 7756-7765 (task interference, shared/task-specific branches)
- [15] Zhang et al., AAAI 2025, vol. 39 no. 21, pp. 22389-22397 (MMoE polarization)
- [16] Nishi et al., CVPR 2024, pp. 16152-16162 (partially labeled MTL)
- [17] Xiao & Duan, Sci. Rep. 15(1), Art. 21978, 2025 (crash-severity multi-task)
- [18] McElfresh et al., NeurIPS 2023, vol. 36, pp. 76336-76369 (tabular NN vs trees)
- [19] Siepe et al., Psychol. Methods 2024, doi:10.1037/met0000695 (no volume/pages in Crossref)
- [20] Hardwicke & Wagenmakers, Nat. Hum. Behav. 7(1):15-26, 2023

Venue tiers (Q1/Q2, CORE A*/A/B): NOT verified against SCImago/CORE. Claim "top venue" is unverified.
CVPR/NeurIPS/AAAI are widely regarded as top ML venues, but this was not checked here.
Sci. Rep. is probably not "top"; treat as the weakest ref.
Foundational refs [3]-[11] are old on purpose (original methods); not replaced.
Removed the sentence "mechanisms were specified before the frozen runs" (not cross-checked); kept only "no public registration".

Render: tectonic (XeTeX), 6 pages (limit 6-8), 20/20 cites matched to 20 bibitems, no undefined refs.
All 6 pages inspected visually (header, tables III-IV, Figs 1-3, bibliography). Remaining warnings: only Underfull hbox + absolute font path (Times New Roman).

Human-only blockers before CMT upload:
1. Anonymity: author block is filled but the CFP says blind review; decide whether to anonymize.
2. Originality confirmation (not submitted elsewhere).
3. Cutoff time/timezone of 15/11/2026 unverified.
4. Submission itself (CMT) not performed; PAPER_TASK_STATE untouched.

## Reframing pass (2026-11-15, after reviewer-style read)
Problem found: abstract/Results/Discussion presented only negative or neutral results and hid the strongest consistent result.
Changes (backup: /tmp/main.tex.before_framing):
- Abstract: added that the full model was ahead of single_task_mlp and logistic regression on every dataset (interaction, partial) and of CatBoost on every interaction dataset; kept "no consistent advantage vs own ablations".
- Results: new paragraph "Where the full model was consistently ahead" (Table IV numbers).
- Discussion "Answer": rewritten as two-part answer (ahead of independent baselines; not of ablations).
Numbers re-checked against docs/paper/paired-summary.csv (macro rows): single_task_mlp interaction +0.0133 [+0.0101,+0.0174], partial +0.0114 [+0.0081,+0.0158]; logistic +0.0848/+0.0780; CatBoost interaction +0.0069 [+0.0008,+0.0114], additive +0.0044 [+0.0020,+0.0069], partial range [-0.0011,+0.0081] (not claimed as consistent); early_fusion_mlp interaction +0.0056 [+0.0030,+0.0096].
Claim limits kept: sign consistency over 3 datasets is not significance; comparators differ in capacity/search budget; no claim of which module causes the gain.
Build: 7 pages (within 6-8), 20/20 cites, no undefined refs. Anonymity/originality/deadline-cutoff blockers unchanged.

## 10-replicate extension pass (post hoc)
- Added generator seeds 1704–1710 (7 per mechanism) with the frozen commands/hyperparameters via `scripts/run_extension.py` (350 new fits, 0 failures). Total 500 fits, 30 datasets.
- Decision to extend was made AFTER the first three replicates were examined; the paper discloses this (Simulation Design, Results "Effect of the post-hoc extension", Limits, Conclusion, abstract). No significance claims.
- Frozen artifacts (docs/paper/results*.csv, paired-*.csv, outputs/paper-final, run-plan.json) untouched; the original 150 rows reproduce exactly inside docs/paper/extension/results-all.csv (max AUC diff < 1e-12). Hashes: docs/paper/extension/extension-manifest.json.
- Tables I, III, IV and Fig. 3 regenerated from docs/paper/extension/{results-summary-10,paired-summary-10,paired-deltas-10}.csv (fig script: scripts/figures/fig3_paired_auc_10.py). Table II (params/runtime) still the frozen 150-fit values; extension timings are contended (4 parallel workers) and not reported as comparable.
- Claims removed because they no longer hold: "highest macro AUC in the interaction setting" (now tied with shared_gate at 0.740), "ahead of CatBoost on every interaction dataset" (9/10, min −0.0019).
- Not independently recomputed for the extension (same scorer reused); text numbers were typed from the CSVs by hand, not by scripts/audit_paper_numbers.py (not re-run).
- Build: tectonic, 7 pages, 20 refs, no undefined citations; pages 4–7 visually inspected. Backup of previous version: /tmp/main.tex.before_ext10 (not persistent).
- Still human-only: anonymity of author block, originality confirmation, deadline cutoff/timezone (15/11/2026), venue-rank verification of references, CMT submission.
- Title changed to "M2G-NET: Auditing Gated Multi-Task Learning under Outcome Eligibility and Partial Labels" (rebuilt, 7 pages, no undefined citations).
- Title changed again to "M2G-NET Under Eligibility and Partial Labels: A Synthetic Audit" (rebuilt, page count checked).

## External-model-only manuscript scope (user requested)
- Removed same-family variant and complete-case comparisons from the FJCAI text, all four tables and Fig. 3. Retained M2G-NET and four external comparators, including all their unfavorable additive and Brier results.
- This is a post-result reporting-scope change, not a new preregistered comparison. The 300 fits shown are the five-procedure subset of the existing 500-fit study; no model was retrained and no frozen result was modified.
- The separate v3 LOCKED TEST protocol and its failed overall superiority verdict are unchanged. This manuscript still reports the earlier model and ten-replicate results, not v3 or the locked seeds.
- Figure source: `scripts/figures/fig3_paired_auc_10.py`; compact inputs: `docs/paper/extension/{paired-deltas-10,paired-summary-10,results-summary-10}.csv`. The generator asserts ten dataset-level values for every retained comparator and mechanism.
- Removed the obsolete claim that no public repository exists. Only manuscript assets, compact supporting tables and figure-generation files are included in this commit; unrelated working-tree changes and raw logs remain local.
- Verification: 82 tests and 78 subtests passed. Final PDF rebuilt as 6 pages with no undefined citations/references; all 20 bibliography entries are cited. Figure input coverage verified as ten datasets times two training seeds for every retained comparator and mechanism. Nonfatal font/box warnings remain. Earlier visual checks in this log describe earlier revisions, not a fresh visual inspection of every page of this revision.

## Final-model rewrite (2026-11-15)
- Manuscript rewritten for the final configuration (named only "M2G-NET": BCE, five-member probability-averaged ensemble, gate prior 0.1) against four external comparators on 20 independent generator seeds per mechanism (1711-1730, two training seeds each; 600 fits).
- Source: `docs/paper/extension-v3/per-set-locked.csv` and `results-locked.csv` (label `m2g_net_v3` in the files). Tables I-III, the abstract/results/discussion numbers and Fig. 3 were recomputed from these CSVs (scripts: `scripts/figures/fig3_paired_auc_locked.py`; table rows generated from the CSVs, not typed). Prose claims (per-task rankings, Brier gaps, positive counts, partial-label eligibility/observation ranges, minimum class counts) checked against the same data.
- Removed: focal-loss text and Lin2017focal bibitem (final model uses BCE), the post-hoc-extension passages, Table "Computational comparison" (no comparable parameter/runtime measurements exist for the ensemble run; the text states they are not reported).
- Added caveat: M2G-NET is a five-member ensemble, comparators are single fits (no matched-capacity or ensembled-comparator run).
- Build: tectonic, 6 pages, 19/19 bibitems cited, no undefined refs; pages 4-5 inspected visually, pages 1-3 and 6 not.
- Not verified: Method details for preprocessing/schedules unchanged from the earlier audited text and not re-checked line by line; comparator hyperparameter grids unchanged from earlier text; ensemble members are not refitted (read from code, not run). No AI-detector run performed. Nothing committed or pushed in this pass.
- Second full pass (all 6 pages viewed; Method claims re-checked in `scripts/train.py`, `scripts/evaluate/paper_pipeline.py`, `baselines/run_all_baselines.py`: BCE without class weights, member seed s+1000m, cosine schedule, clip 1.0, C and depth grids; every Results sentence re-checked against the CSV-derived tables). Wording on development data narrowed to "validation splits only" per FREEZE.json.
- Open point for the author: development seeds 1701-1710 had earlier test results reported for a previous model version; configuration choices for the final model used validation splits only, but the author should decide whether to state that history.

## Comparator seed-average supplement (2026-11-15, post hoc)
- Question: is the ensemble-vs-single-fit confound decisive? Checked without refits by averaging the stored test probabilities of training seeds 42 and 101 for each comparator (`scripts/figures/comparator_seed_ensemble.py` -> `docs/paper/extension-v3/comparator-seed-ensemble.csv`, 300 rows). The script recomputes the reported single-fit macro AUC from `predictions.csv` + oracle and matches `per-set-locked.csv` to 1.1e-16 before using the ensemble.
- Result: seed averaging raised mean comparator AUC by <= 0.0031; M2G-NET paired differences kept their signs. Interaction/partial positive on 20/20 datasets for all comparators except CatBoost on partial (19/20); additive -0.0038/-0.0037/-0.0059 (MLPs, logistic), +0.0051 CatBoost (16/20).
- Added as a labelled post hoc paragraph in Results; Limits sentence updated. Limitation unchanged in kind: comparators are 2-member, M2G-NET is 5-member. Raw-probability averaging used (comparator `probability` column is calibrated for some models; M2G-NET reported values use the same scorer on `raw_probability`).
- Build: 6 pages, no undefined refs. Not committed.

## Reviewer-driven revision (2026-11-15, after meta-review and user checklist)
- Results opens with a reading guide (ensemble vs single fits, generator aligned with interaction block, main-effects-only logistic). Subsection renamed "Capacity and compute mismatch".
- Validation metric stated: macro AUC over observed labels (M=1) only; oracle never used for selection (code: scripts/train.py, src/metrics.py).
- Split: `stratified_split` (src/data_pipeline.py) stratifies on the 4 observed-label pattern (ineligible/unobserved coded 0 by `prepare_labels`), fallback to first task's label if any pattern < 2 rows; applied to the full predictor export, so test rows' observed labels are used for stratification (not given to the model). Oracle labels not used. Text now says so.
- Table III caption points to 95% t-intervals (text + `docs/paper/extension-v3/paired-auc-ci.csv`, script `scripts/figures/paired_auc_ci.py`); sentence on 2*0.5^20 ~ 2e-6 is descriptive only; no p-value table.
- Contributions rewritten (design / controlled comparison / conditions of lead and lag). MMoE sentence rewritten; the "our check verifies that abstract" sentence removed. Abstract states that partial data reuse the interaction law.
- Discussion: logistic misspecification, MAR-only (<=25% missing, observation depends on rider type, traffic, time slot, gender), Scope (supported / not supported), broader impact. A first draft of broader impact wrongly said observation depends on police presence; corrected against scripts/generate_paper_simulation.py.
- 6 refs added (Lakshminarayanan 2017, Grinsztajn 2022, Gorishniy 2021, Guo & Berkhahn 2016, Rendle 2010, Little & Rubin 2019); 25/25 bibitems cited. Little & Rubin content and Guo & Berkhahn venue not verified; abstracts only for the arXiv papers.
- Data/code availability narrowed to what is actually provided. WARNING: scripts/generate_paper_simulation.py, scripts/summarize_paper_results.py, scripts/evaluate/paper_pipeline.py, scripts/run_extension.py, docs/paper/generator-spec.md and docs/paper/extension-v3/*.csv are currently UNTRACKED in git; they must be committed for the availability statement to be true.
- Build: tectonic, 7 pages (CFP limit 6-8 per earlier audit). Not run: single-vs-ensemble M2G-NET ablation; nothing committed.
