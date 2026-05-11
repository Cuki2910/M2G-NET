# Gate Weights vs. Causal Analysis: Critical Distinction

**Status:** Active guidance document  
**Purpose:** Clarify what gate weights can and cannot tell us about causal relationships

---

## Executive Summary

**Gate weights (α_k) are attention mechanisms, NOT causal effects.**

They tell us:
- ✓ Which views the model attends to for each task
- ✓ Relative importance in the model's prediction strategy
- ✓ Task-specific differences in information usage

They do NOT tell us:
- ✗ Whether a view causes the outcome
- ✗ What would happen if we intervened on a view
- ✗ The true causal structure of the data-generating process

---

## 1. What Are Gate Weights?

Gate weights are learned attention coefficients that determine how much each view representation contributes to the final prediction for a specific task.

### Mathematical Definition

For task k, the gate weights are:

```
α_k = softmax(Gate_k(concat(h_role, h_trait, h_road, h_env, h_site, h_inter)) / T)
```

After regularization:
```
α_k = (α_k_raw + λ · uniform_prior) / (1 + λ)
```

The fused representation is:
```
z_k_gated = Σ_v α_k_v · h_v
```

### What This Means

- α_k_v is high → model pays more attention to view v when predicting task k
- α_k_v is low → model relies less on view v for task k
- These are **learned weights optimized for prediction accuracy**, not causal discovery

---

## 2. Why Gate Weights ≠ Causal Effects

### 2.1. Confounding

Gate weights reflect **predictive association**, which can be driven by:
- Direct causal effects
- Confounding (common causes)
- Collider bias
- Spurious correlations

**Example:**
```
Suppose: Weather → Rider Role → Risky Behavior
         Weather → Risky Behavior (direct)
```

If the model learns high gate weight for Rider Role, it could be because:
1. Rider Role directly causes risky behavior (causal)
2. Rider Role is correlated with weather, which causes risky behavior (confounding)
3. The model finds Rider Role easier to encode than weather (representation artifact)

**We cannot distinguish these scenarios from gate weights alone.**

### 2.2. Mediation

Gate weights do not reveal mediation paths.

**Example:**
```
Police Presence → Perceived Risk → Risky Behavior
```

If gate weight for Police Presence is low, it could mean:
1. Police presence has no effect (true null)
2. Police presence has a strong effect, but it's fully mediated through other views already captured by the model
3. The encoder for Police Presence is weak

### 2.3. Collider Bias

Conditioning on a collider can create spurious associations.

**Example:**
```
Rider Role → Intersection Selection ← Traffic Density
```

If we condition on intersection (by including site context), we may induce a spurious association between Rider Role and Traffic Density, which the gate weights will reflect.

### 2.4. Representation Quality

Gate weights depend on encoder quality, not just true importance.

**Example:**
- If the Rider Role encoder is well-trained and captures rich information, α_role will be high
- If the Environment encoder is poorly initialized or undertrained, α_env will be low **even if environment is causally important**

---

## 3. What Gate Weights Actually Measure

Gate weights measure **model-based predictive contribution under the learned representation**.

This is useful for:
- Understanding the model's decision-making strategy
- Debugging model behavior
- Identifying which views the model finds informative
- Comparing task-specific attention patterns

This is NOT sufficient for:
- Claiming causal effects
- Designing interventions
- Making policy recommendations without additional evidence
- Replacing causal inference methods

---

## 4. Integrated Gradients: A Complementary Tool

Integrated Gradients (IG) provide a different perspective on feature importance.

### What IG Measures

IG computes the path integral of gradients from a baseline (zero vector) to the actual input:

```
IG(view_k, task_j) = (x_k - x_baseline) · ∫₀¹ (∂F_j / ∂x_k) dλ
```

### IG vs. Gate Weights

| Aspect | Gate Weights | Integrated Gradients |
|--------|--------------|---------------------|
| Scope | Per-sample attention | Population-level attribution |
| What it measures | Learned attention coefficients | Gradient-based feature importance |
| Interpretation | "Model attends to this view" | "This view drives predictions" |
| Causal claim | No | No |

**Key insight:** If both gate weights AND IG agree that a view is important, we have stronger evidence of **predictive importance** (but still not causality).

---

## 5. Triple Validation: Gate + IG + Ablation

To make robust claims about view importance, we use three complementary methods:

### 5.1. Gate Weights (Attention)
- Per-sample: which views does the model attend to?
- Population average: which views get attention on average?

### 5.2. Integrated Gradients (Attribution)
- Population-level: which views drive predictions via gradients?

### 5.3. Ablation Study (Counterfactual)
- Remove view v, retrain model, measure performance drop
- If performance drops significantly → view v is important for prediction

**Triple validation claim:**
> If gate weights, IG, and ablation all indicate that view v is important for task k, we have strong evidence that view v is **predictively important** for task k in this model and dataset.

**This is still not a causal claim.**

---

## 6. How to Report Gate Weights Correctly

### ✓ Correct Phrasing

- "The model assigns higher gate weight to Rider Role for helmet non-use"
- "Gate weights suggest the model relies more on Road Context for red-light running"
- "Task-specific gates reveal different attention patterns across behaviors"
- "Rider Role receives the highest average gate weight for this task"
- "The model's predictive strategy prioritizes Environmental Context"

### ✗ Incorrect Phrasing (Causal Overclaims)

- ~~"Rider Role causes helmet non-use"~~
- ~~"Road Context is the most important factor for red-light running"~~
- ~~"Environmental Context has the strongest effect"~~
- ~~"This proves that Rider Role drives risky behavior"~~
- ~~"Gate weights show the causal mechanism"~~

### Recommended Template

> "Gate weights indicate that the model attends most strongly to [View X] when predicting [Task Y]. This suggests [View X] is predictively informative for [Task Y] in the learned representation. However, gate weights reflect attention mechanisms, not causal effects. To support causal claims, additional evidence from [domain theory / experimental data / causal inference methods] would be required."

---

## 7. When Can We Make Stronger Claims?

Gate weights can support **stronger (but still not causal)** claims when combined with:

### 7.1. Theory Alignment
If gate weights align with established theory, this provides convergent evidence.

**Example:**
- Situational Crime Prevention predicts that Road Context should be important for red-light running
- Gate weights show high α_road for red-light running
- **Claim:** "Gate weights are consistent with Situational Crime Prevention theory"

### 7.2. Ablation Consistency
If ablation shows large performance drop when removing a view with high gate weight:

**Claim:** "View X is both attended to (gate weights) and necessary for prediction (ablation)"

### 7.3. IG Consistency
If IG and gate weights agree:

**Claim:** "View X is important both in terms of attention (gates) and gradient-based attribution (IG)"

### 7.4. Cross-Validation Across Sites
If gate weights remain stable in leave-site-out validation:

**Claim:** "View X's importance generalizes across sites"

---

## 8. What Would Be Needed for Causal Claims?

To make causal claims, we would need:

### 8.1. Randomized Controlled Trials
- Randomly assign riders to different conditions
- Measure outcomes
- Compare treatment vs. control

### 8.2. Natural Experiments
- Exploit quasi-random variation (e.g., policy changes, weather shocks)
- Use difference-in-differences, regression discontinuity, instrumental variables

### 8.3. Causal Inference Methods on Observational Data
- Propensity score matching
- Inverse probability weighting
- Causal graphical models with testable implications
- Sensitivity analysis for unmeasured confounding

### 8.4. Longitudinal Data
- Repeated observations over time
- Panel data methods
- Granger causality tests (with strong assumptions)

**Our model does none of these.** It is a predictive model trained on cross-sectional observational data.

---

## 9. Reporting Checklist

Before writing any claim about gate weights, ask:

- [ ] Am I claiming attention or causation?
- [ ] Have I used causal language (causes, leads to, drives, effects)?
- [ ] Have I stated the limitation that gate weights are not causal?
- [ ] Have I provided convergent evidence (theory, ablation, IG)?
- [ ] Would a reviewer flag this as overclaiming?

---

## 10. Example: Correct Interpretation

### Scenario
- Task: Helmet non-use
- Gate weights: α_role = 0.45, α_road = 0.25, α_env = 0.15, α_trait = 0.10, α_site = 0.05
- Ablation: Removing Rider Role drops AUC by 0.12
- IG: Rider Role has highest attribution score
- Theory: Occupational Risk Theory predicts rider role matters

### Correct Interpretation

> "For helmet non-use prediction, the model assigns the highest gate weight to Rider Role (α = 0.45), followed by Road Context (α = 0.25). This attention pattern is consistent across three validation methods: gate weights, Integrated Gradients attribution, and ablation study (AUC drop = 0.12 when Rider Role is removed). These findings align with Occupational Risk Theory, which posits that occupational roles shape risk exposure and safety behavior norms. However, gate weights reflect the model's learned attention mechanism, not causal effects. Causal claims would require experimental or quasi-experimental designs to rule out confounding and establish counterfactual relationships."

---

## 11. Implications for the Paper

### In the Methods Section
- Define gate weights as attention mechanisms
- State explicitly that they do not imply causation
- Describe triple validation (gate + IG + ablation)

### In the Results Section
- Report gate weights as "attention patterns" or "predictive importance"
- Avoid causal language
- Highlight convergence across methods when present

### In the Discussion Section
- Interpret gate weights in light of theory
- Acknowledge limitations
- Suggest future work with causal designs

### In the Limitations Section
- State clearly: "Gate weights are attention mechanisms and do not establish causal relationships"
- Recommend causal inference methods for future research

---

## 12. Summary Table

| Question | Gate Weights Answer | What's Needed for Causal Answer |
|----------|---------------------|--------------------------------|
| Which view does the model attend to? | ✓ Yes | N/A |
| Which view is predictively important? | ✓ Yes (with ablation) | N/A |
| Which view causes the outcome? | ✗ No | RCT, natural experiment, causal inference |
| What happens if we intervene on a view? | ✗ No | Experimental manipulation |
| What is the causal mechanism? | ✗ No | Mediation analysis, causal graphs |
| Which view should we target in policy? | ✗ No (alone) | Causal evidence + cost-benefit analysis |

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses. *CVPR*.

---

## Changelog

- 2026-05-12: Initial version — clarify gate weights vs. causality distinction
