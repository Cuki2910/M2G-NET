# Feasibility assessment after gate round

## Questions and scope

1. Do measured dev results support possible success under the unchanged protocol?
2. Would another development round likely be useful, and what should it diagnose?
3. Does a stronger AI coding/research assistant change expected outcomes?
4. Does a larger/different predictive network necessarily help?

No new fit, data generation, or locked evaluation for this assessment. Sources: full local gate report/summary and fetched official methodology pages. Web queries covered model selection optimism, training/validation learning curves and coding-agent evaluation. No model-family ranking or probability of success is justified by available evidence.

## Local measured evidence

Source: [GATE-SEARCH-RESULTS.md](GATE-SEARCH-RESULTS.md) and gate-search/score-1794730441543881000/summary.csv.

- Completed 1200 member fits = 4 gate modes × 3 mechanisms × 10 generator seeds × 2 training seeds × 5 members. This is four gate configurations, NOT 1200 distinct architectural hypotheses.
- All modes fail the complete dev screen. Old M5/legacy remains best on mean AUC in interaction/partial among these modes, not necessarily best outside this search space.
- Legacy vs old m2g_net: interaction mean +0.003909, MCSE .001617, positive 8/10; partial mean +0.003219, MCSE .000601, positive 9/10. Reaching > .005 would require more than an extra .001091/.001781 mean on the same dev comparisons. These arithmetic shortfalls are not predictions for locked data.
- Other failures also matter: interaction vs shared_gate; partial vs shared_gate, no_interaction and early_fusion. A change must pass every opponent, both mechanisms, additive and Brier conditions jointly.
- Legacy aggregate Brier improves on early_fusion_mlp and catboost on dev. Calibration is not the current failed requirement; further Brier gains alone cannot satisfy AUC superiority.
- No training/validation learning-curve diagnosis in this round establishes whether remaining error is predominantly capacity, regularization, optimization or irreducible noise.
- Structure-aware LR from earlier diagnostics is not a mathematical upper bound; the prior claim of no headroom was unsupported. Neither a reliable pass probability nor impossibility follows from the current runs.

## Official external evidence, fully retrieved

### Selection optimism
URL: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
Full fetched text: outputs/v3-feasibility-sources/selection.txt.
Choosing hyperparameters using the same validation/CV score makes that score optimistic for generalization; independent outer evaluation assesses the complete selection procedure. Repeated dev trials do not automatically contaminate an untouched final holdout, but dev success is not sufficient evidence of holdout success. This supports maintaining the locked protocol, not changing data/seeds/thresholds.

### Model complexity and diagnosis
URL: https://scikit-learn.org/stable/modules/learning_curve.html
Full fetched text: outputs/v3-feasibility-sources/capacity.txt.
Training and validation scores/learning curves help distinguish underfitting from overfitting. More model capacity can improve fit or worsen generalization; complexity alone is not a universal remedy. Apply diagnostics inside the already allowed DEVELOPMENT train/validation partitions; do not inspect locked data or enlarge training data beyond the registered comparison protocol.

### Coding assistant versus predictive model
URL: https://www.anthropic.com/research/building-effective-agents
Full fetched text: outputs/v3-feasibility-sources/agents.txt.
The engineering guidance recommends simple components, tool-grounded feedback, measurable evaluation, and complexity only when it demonstrably improves outcomes. Coding tests verify implementation behavior, not scientific novelty or guaranteed improvement of the trained predictive model. This source does NOT compare the currently available assistant models on this repository. Any claim a specific assistant will obtain the target AUC is unverified.

## Assessment and recommendation (inference, not measured forecast)

Success remains possible but has not been demonstrated; confidence should not be described as high. Another blind gate/size tweak is not well supported by the negative controlled gate results. One bounded, predeclared diagnostic round is more defensible than open-ended try-until-dev-passes: retain legacy and evaluate training/validation gap and per-task performance on allowed development partitions, then test a small fixed set of regularization/optimization changes motivated by that diagnosis. Width/feature interaction changes are candidates only if diagnostics motivate them; their benefit is unverified.

A stronger assistant may offer better hypotheses or catch methodological errors, but no local comparison measures its effect; buying a more expensive assistant is not established as necessary. A larger predictive network is likewise not established as necessary. The recommendation is to improve experimental information before spending on either. An optional stronger independent reviewer could be used for hypothesis critique, while routine runner execution remains cheaper mechanical work.

Do not promise that one extra round will pass. Keep all failed trials; no test-based retraining. Freeze only a single selected candidate after full review, request explicit permission, then evaluate locked seeds once. Report 'chưa vượt trội' if any registered final condition fails. Current assessment changes neither the criterion nor the pending goal, and starts no new experiment.
