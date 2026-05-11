# Rewrite Templates for Common Issues

This file provides before/after templates for the most common writing issues in transportation and green transition research.

---

## Template 1: Causal Language → Associational Language

### Pattern: Observational data with causal claims

**Before:**
```
Our analysis proves that bike lanes cause a 23% reduction in car usage.
```

**After:**
```
Our analysis shows that bike lane installation is associated with a 23% reduction in car usage. However, reverse causality (cities with declining car usage may be more likely to invest in bike infrastructure) and confounding factors (income, urban density, political climate) limit causal interpretation. A natural experiment or instrumental variable approach would be needed to establish causality.
```

**Key changes:**
- "proves" → "shows"
- "cause" → "is associated with"
- Added limitations paragraph
- Suggested what would be needed for causal claims

---

## Template 2: Generalization Overclaim → Scoped Claim

### Pattern: Single-context study with broad claims

**Before:**
```
Our traffic prediction model generalizes to all cities and achieves 0.85 accuracy.
```

**After:**
```
Our traffic prediction model achieves 0.85 accuracy in Amsterdam, a dense, bike-friendly city with extensive sensor coverage. Generalization to car-dependent cities, cities with sparse data, or cities with different traffic patterns (e.g., developing countries) requires validation. We identify three boundary conditions that may limit transferability: (1) minimum sensor density, (2) modal split characteristics, and (3) road network topology.
```

**Key changes:**
- Specified context (Amsterdam)
- Listed characteristics that may affect generalization
- Explicitly stated what needs validation
- Identified boundary conditions

---

## Template 3: Synthetic Data → Real Validation Needed

### Pattern: Synthetic-only validation claiming real-world applicability

**Before:**
```
We validate our model on synthetic data and demonstrate strong performance (0.73 AUC), showing it is ready for real-world deployment.
```

**After:**
```
We validate our model on synthetic data generated under assumed behavioral mechanisms and achieve 0.73 AUC. This proof-of-concept demonstrates feasibility, but real-world validation on held-out observational data is required before deployment. Key validation steps include: (1) testing on real intersections not seen during training, (2) assessing performance across diverse contexts (urban/rural, different countries), and (3) evaluating robustness to distribution shift.
```

**Key changes:**
- Clarified "synthetic data" and its assumptions
- Changed "ready for deployment" to "proof-of-concept"
- Listed specific validation steps needed
- Acknowledged distribution shift risk

---

## Template 4: Vague Methodology → Specific Details

### Pattern: "We processed the data appropriately"

**Before:**
```
We collected data from multiple sources, processed it appropriately, and trained our model using standard techniques.
```

**After:**
```
We collected data from three sources: (1) city traffic sensors (N=50 intersections, 2020-2023), (2) weather API (hourly data), and (3) census demographics (2020). We handled missing data (8% of sensor readings) using forward-fill for short gaps (<1 hour) and excluded longer gaps. We normalized continuous features using training set statistics (μ, σ) and one-hot encoded categorical variables. We split data temporally (train: 2020-2022, validation: Jan-Jun 2023, test: Jul-Dec 2023) to prevent leakage. We trained a 3-layer MLP (hidden dims: 128, 64, 32) using Adam optimizer (lr=0.001, weight decay=1e-4) with early stopping (patience=20 epochs on validation loss).
```

**Key changes:**
- Specified data sources with details
- Quantified missing data and handling method
- Described preprocessing steps
- Detailed train/val/test split strategy
- Listed model architecture and hyperparameters

---

## Template 5: No Baseline → Proper Comparison

### Pattern: Reporting model performance in isolation

**Before:**
```
Our deep learning model achieves 0.85 ROC-AUC on the test set.
```

**After:**
```
Our deep learning model achieves 0.85 ROC-AUC on the test set, outperforming logistic regression (0.78), random forest (0.81), and tuned XGBoost (0.83). The improvement over XGBoost is statistically significant (p=0.03, paired t-test over 10 runs with different random seeds). However, the model requires 3× more training time and 5× more parameters than XGBoost, suggesting a tradeoff between performance and complexity.
```

**Key changes:**
- Added multiple baseline comparisons
- Reported statistical significance
- Quantified computational cost tradeoff
- Provided context for interpreting the improvement

---

## Template 6: Point Estimate → Uncertainty Quantification

### Pattern: Reporting results without confidence intervals

**Before:**
```
Model A achieves 0.85 accuracy and Model B achieves 0.83 accuracy, so Model A is better.
```

**After:**
```
Model A achieves 0.85 ± 0.03 accuracy (mean ± std over 10 runs) and Model B achieves 0.83 ± 0.04 accuracy. The difference is not statistically significant (p=0.18, Wilcoxon signed-rank test), suggesting both models perform comparably within measurement uncertainty. We recommend Model B for deployment due to its lower computational cost (2× faster inference) and simpler architecture (easier to maintain).
```

**Key changes:**
- Added confidence intervals
- Reported statistical test
- Acknowledged non-significance
- Provided practical recommendation considering tradeoffs

---

## Template 7: Weak Abstract → Strong Abstract

### Pattern: Abstract that doesn't convey contribution

**Before:**
```
Traffic safety is important. We propose a new model for predicting risky behaviors. Our model uses multiple data sources and achieves good performance. This work contributes to traffic safety research.
```

**After:**
```
Motorcycle risky behaviors at intersections cause 30% of urban traffic fatalities, yet existing prediction models treat all riders identically. We propose M2G-Net, a theory-guided multi-view multi-task network that decomposes risk factors into individual-level (rider role, traits) and contextual-level (road, environment, site) representations, learning their interactions explicitly. On 9,164 observations from 31 intersections, our model achieves 0.73 macro ROC-AUC across four behaviors, comparable to XGBoost (0.67) but with interpretable view-level attributions. Rider role contributes most to helmet non-use (35% gate weight), while road context dominates red-light running (28%), suggesting behavior-specific intervention strategies. Limitations include synthetic validation data and modest improvement over baselines, requiring real-world validation before deployment.
```

**Key changes:**
- Opened with problem + gap
- Stated specific contribution (theory-guided decomposition)
- Reported concrete results with context
- Highlighted key insight (behavior-specific patterns)
- Acknowledged limitations upfront

---

## Template 8: Missing Limitations → Honest Discussion

### Pattern: No limitations section or vague limitations

**Before:**
```
Our study has some limitations that should be addressed in future work.
```

**After:**
```
Our study has four main limitations. First, validation is currently on synthetic data generated under assumed behavioral mechanisms; real-world validation on held-out intersections is needed to assess external validity. Second, the dataset is small (31 sites) and geographically limited (one city), limiting generalization to diverse contexts. Third, class imbalance is severe for some behaviors (helmet non-use: 7.4%), and despite using uncertainty weighting, minority-class performance remains weak. Fourth, our interpretability analysis (gate weights + Integrated Gradients) shows inconsistencies for some views (e.g., site context), suggesting the learned representations may not fully align with theoretical constructs. These limitations do not invalidate the methodological contribution but constrain the strength of empirical claims.
```

**Key changes:**
- Listed specific, numbered limitations
- Explained impact of each limitation
- Acknowledged what this means for claims
- Balanced honesty with contribution framing

---

## Template 9: Weak Policy Recommendation → Evidence-Based Recommendation

### Pattern: Policy claims not supported by analysis

**Before:**
```
Our findings show that bike lanes reduce emissions. Therefore, all cities should invest heavily in bike infrastructure.
```

**After:**
```
Our findings suggest that bike lane installation is associated with reduced car usage in dense, bike-friendly cities (23% reduction, 95% CI: 18-28%). However, this association may not hold in car-dependent cities with different urban form, climate, or cultural norms. Before large-scale investment, we recommend: (1) pilot programs in 3-5 diverse cities to assess context-specific effects, (2) cost-benefit analysis comparing bike infrastructure to alternative interventions (e.g., transit, congestion pricing), and (3) equity analysis to ensure benefits reach low-income and mobility-limited populations. Cities with existing cycling culture and dense urban form are most likely to see benefits.
```

**Key changes:**
- Scoped recommendation to appropriate contexts
- Suggested pilot testing before scaling
- Called for cost-benefit and equity analysis
- Identified boundary conditions for success

---

## Template 10: Greenwashing → Honest Lifecycle Assessment

### Pattern: Claiming environmental benefits without full accounting

**Before:**
```
Electric vehicles are zero-emission and solve the climate crisis.
```

**After:**
```
Electric vehicles have zero tailpipe emissions, but lifecycle emissions depend on grid carbon intensity and battery production. In regions with coal-heavy grids (e.g., >600 gCO2/kWh), EVs may have higher lifecycle emissions than efficient hybrid vehicles for the first 100,000 km. In clean-grid regions (<200 gCO2/kWh), EVs offer 50-70% lifecycle emission reductions. Additionally, EVs do not address congestion, sprawl, or equity issues in car-dependent systems. A comprehensive climate strategy requires combining EVs with transit investment, land-use reform, and mode shift incentives.
```

**Key changes:**
- Clarified "zero tailpipe" vs "zero lifecycle"
- Quantified context-dependence (grid intensity)
- Acknowledged non-climate impacts (congestion, equity)
- Positioned EVs as part of solution, not silver bullet

---

## Quick Reference: Language Substitutions

| Avoid | Use Instead |
|-------|-------------|
| "proves" | "suggests", "provides evidence for" |
| "causes" (observational) | "is associated with", "correlates with" |
| "all cities" | "cities with characteristics X, Y, Z" |
| "ready for deployment" | "shows promise; real-world validation needed" |
| "our model works" | "our model achieves X performance in context Y" |
| "zero-emission" | "zero tailpipe emission; lifecycle emissions depend on..." |
| "solves the problem" | "addresses one aspect of the problem" |
| "significantly better" | "statistically significantly better (p=X)" |
| "we processed the data" | "we [specific preprocessing steps]" |
| "good performance" | "0.85 ROC-AUC (95% CI: 0.82-0.88)" |

---

## Using These Templates

1. **Identify the pattern** in your writing that matches a template
2. **Adapt the template** to your specific context (don't copy verbatim)
3. **Preserve your voice** while improving precision
4. **Check if the rewrite** addresses the core issue (causal language, generalization, etc.)

Remember: The goal is not to weaken your claims unnecessarily, but to **align claims with evidence**.
