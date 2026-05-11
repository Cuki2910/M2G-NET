# Common Flaws in Transportation & Green Transition Research

This reference provides a curated list of the most frequent issues found in transportation and sustainability research, organized by severity and domain.

## Table of Contents

1. [Causal Inference Errors](#causal-inference-errors)
2. [Generalization Overclaims](#generalization-overclaims)
3. [Data Quality Issues](#data-quality-issues)
4. [Methodology Red Flags](#methodology-red-flags)
5. [Green Transition Specific](#green-transition-specific)
6. [Traffic Safety Specific](#traffic-safety-specific)
7. [Machine Learning Pitfalls](#machine-learning-pitfalls)

---

## Causal Inference Errors

### 1. Correlation → Causation Leap

**Pattern**: "X causes Y" when only observational correlation is shown.

**Example**:
- ❌ "Bike lanes cause reduced car usage"
- ✅ "Bike lanes are associated with reduced car usage, though reverse causality (cities with less car usage build more bike lanes) cannot be ruled out"

**Fix**: Use causal language only with:
- Randomized controlled trials
- Natural experiments with valid identification strategy
- Instrumental variables with strong first stage
- Regression discontinuity with valid cutoff
- Difference-in-differences with parallel trends

### 2. Confounding Ignored

**Pattern**: Claiming effect of X on Y without controlling for Z that affects both.

**Example**:
- ❌ "Electric vehicle adoption reduces emissions" (ignoring that early adopters are wealthier, live in cleaner-grid regions)
- ✅ "After controlling for income, grid carbon intensity, and urban density, EV adoption is associated with X% emission reduction"

**Common confounders in transportation**:
- Income/SES
- Urban vs rural
- Weather/climate
- Policy environment
- Infrastructure quality
- Cultural norms

### 3. Reverse Causality

**Pattern**: Assuming X → Y when Y → X is equally plausible.

**Example**:
- ❌ "Public transit investment increases ridership"
- ⚠️ Could be: Cities with high ridership demand invest in transit

**Fix**: Acknowledge bidirectionality or use lagged variables, instrumental variables, or natural experiments.

---

## Generalization Overclaims

### 1. Single-City Syndrome

**Pattern**: "Our model works" based on data from one city/country.

**Example**:
- ❌ "This traffic prediction model generalizes to all cities" (tested only in Amsterdam)
- ✅ "This model performs well in Amsterdam (dense, bike-friendly, flat). Generalization to car-dependent, hilly cities requires validation."

**Fix**: Either:
- Test on multiple diverse contexts
- Explicitly scope claims to similar contexts
- Discuss boundary conditions

### 2. Synthetic → Real Leap

**Pattern**: Claiming real-world applicability from synthetic data only.

**Example**:
- ❌ "Our model predicts risky behavior in real intersections" (tested only on synthetic data)
- ✅ "Our model shows promise on synthetic data. Real-world validation on held-out intersections is needed before deployment."

**Fix**: Always validate on real data before claiming real-world applicability.

### 3. Lab → Field Gap

**Pattern**: Assuming controlled experiments transfer to messy reality.

**Example**:
- ❌ "Our autonomous vehicle is safe" (tested in closed track)
- ✅ "Our AV performs well in controlled conditions. Real-world deployment requires testing in rain, construction zones, and adversarial scenarios."

---

## Data Quality Issues

### 1. Missing Data Handling Not Reported

**Pattern**: No mention of how missing data was handled.

**Red flags**:
- "We removed incomplete records" (how many? biased sample?)
- "We imputed missing values" (how? sensitivity analysis?)

**Fix**: Report:
- % missing per variable
- Missingness mechanism (MCAR/MAR/MNAR)
- Imputation method + sensitivity analysis

### 2. Class Imbalance Ignored

**Pattern**: Reporting accuracy on imbalanced data without addressing it.

**Example**:
- ❌ "Our model achieves 95% accuracy" (when 95% of samples are negative class)
- ✅ "Despite class imbalance (95% negative), our model achieves 0.85 ROC-AUC and 0.72 PR-AUC"

**Fix**: Report:
- Class distribution
- Imbalance-robust metrics (ROC-AUC, PR-AUC, F1, balanced accuracy)
- Handling strategy (resampling, class weights, focal loss)

### 3. Temporal Leakage

**Pattern**: Using future information to predict the past.

**Example**:
- ❌ Training on 2020-2023 data, testing on 2022 data
- ❌ Using "total trip duration" to predict "arrival time"

**Fix**: Strict temporal split, only use information available at prediction time.

---

## Methodology Red Flags

### 1. No Baseline Comparison

**Pattern**: Reporting model performance without comparing to simpler alternatives.

**Example**:
- ❌ "Our deep learning model achieves 0.75 AUC"
- ✅ "Our model (0.75 AUC) outperforms logistic regression (0.68) and XGBoost (0.72)"

**Fix**: Always compare to:
- Naive baseline (majority class, mean prediction)
- Simple baseline (logistic regression, decision tree)
- Strong baseline (tuned XGBoost/LightGBM)
- Prior SOTA if exists

### 2. Hyperparameter Tuning on Test Set

**Pattern**: Choosing hyperparameters based on test set performance.

**Example**:
- ❌ "We tried learning rates 0.001, 0.01, 0.1 and chose 0.01 because it gave best test accuracy"

**Fix**: Use proper train/validation/test split:
- Train: fit model
- Validation: tune hyperparameters
- Test: report final performance (touch only once)

### 3. No Uncertainty Quantification

**Pattern**: Reporting point estimates without confidence intervals or significance tests.

**Example**:
- ❌ "Model A: 0.75 AUC, Model B: 0.73 AUC → A is better"
- ✅ "Model A: 0.75±0.03, Model B: 0.73±0.04 (p=0.12, not significant)"

**Fix**: Report:
- Confidence intervals (bootstrap or repeated runs)
- Statistical significance tests
- Effect sizes

---

## Green Transition Specific

### 1. Lifecycle Emissions Ignored

**Pattern**: Claiming "green" based on operational emissions only.

**Example**:
- ❌ "Electric buses are zero-emission"
- ✅ "Electric buses have zero tailpipe emissions. Lifecycle emissions depend on grid carbon intensity and battery production."

**Fix**: Include:
- Manufacturing emissions
- Grid carbon intensity
- End-of-life disposal
- Infrastructure construction

### 2. Rebound Effects Ignored

**Pattern**: Assuming efficiency gains = net reduction.

**Example**:
- ❌ "Fuel-efficient cars reduce emissions by 30%"
- ⚠️ Rebound: People drive more when fuel is cheaper

**Fix**: Account for:
- Direct rebound (more usage of efficient technology)
- Indirect rebound (savings spent on other carbon-intensive activities)

### 3. Equity Blindness

**Pattern**: Ignoring distributional impacts of green policies.

**Example**:
- ❌ "Carbon tax reduces emissions"
- ⚠️ Regressive: hurts low-income households more

**Fix**: Analyze:
- Who benefits? Who pays?
- Spatial equity (urban vs rural)
- Temporal equity (current vs future generations)

---

## Traffic Safety Specific

### 1. Surrogate Measures Without Validation

**Pattern**: Using proxy metrics (speed, near-misses) without linking to actual crashes.

**Example**:
- ❌ "Our intervention reduces speeding by 20%, improving safety"
- ⚠️ Does reduced speeding actually reduce crashes in this context?

**Fix**: Either:
- Validate surrogate against crash data
- Use actual crash/injury data
- Acknowledge limitation

### 2. Exposure Not Controlled

**Pattern**: Comparing crash rates without accounting for exposure (miles driven, time at risk).

**Example**:
- ❌ "Motorcyclists have more crashes than car drivers"
- ⚠️ Per trip? Per mile? Per hour?

**Fix**: Report rates per:
- Vehicle-miles traveled (VMT)
- Trips
- Hours of exposure

### 3. Severity Ignored

**Pattern**: Treating all crashes equally.

**Example**:
- ❌ "Intervention reduced crashes by 10%"
- ⚠️ Fatal crashes? Minor fender-benders?

**Fix**: Stratify by severity:
- Fatal
- Serious injury (ISS > 15)
- Minor injury
- Property damage only

---

## Machine Learning Pitfalls

### 1. Train-Test Contamination

**Pattern**: Information leaking from test set into training.

**Examples**:
- Normalizing on full dataset before split
- Feature selection on full dataset
- Imputation using test set statistics

**Fix**: All preprocessing must use only training data statistics.

### 2. Overfitting to Small Data

**Pattern**: Complex model on small dataset.

**Red flags**:
- Parameters > samples
- Perfect training accuracy
- No regularization
- No cross-validation

**Fix**:
- Use simpler models
- Regularization (L1/L2, dropout)
- Cross-validation
- Report train vs validation curves

### 3. Interpretability Theater

**Pattern**: Claiming interpretability without rigorous analysis.

**Example**:
- ❌ "Our model is interpretable because it uses attention"
- ⚠️ Attention weights ≠ feature importance

**Fix**: Use validated methods:
- SHAP values
- Integrated Gradients
- Ablation studies
- Consistency checks (attention vs gradient vs ablation)

---

## Quick Checklist

Before claiming your work is ready:

- [ ] Causal language only with causal design
- [ ] Confounders addressed or acknowledged
- [ ] Generalization scope explicitly stated
- [ ] Missing data handling reported
- [ ] Class imbalance addressed
- [ ] Baseline comparisons included
- [ ] Hyperparameters tuned on validation set only
- [ ] Uncertainty quantified (CI, p-values)
- [ ] Lifecycle thinking (for green claims)
- [ ] Equity impacts considered
- [ ] Exposure controlled (for safety claims)
- [ ] Interpretability validated (for ML claims)

---

## When in Doubt

Ask yourself:
1. **Would this claim survive a hostile reviewer?**
2. **Can I defend this with the data I have?**
3. **What's the weakest link in my argument?**
4. **What would I ask if I were reviewing this?**

If the answer makes you uncomfortable, revise before submission.
