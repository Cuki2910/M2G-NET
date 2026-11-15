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
