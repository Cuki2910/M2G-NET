# Gate Stability Analysis Results - CRITICAL FINDING

**Date:** 2026-05-12  
**Status:** ⚠️ **UNSTABLE GATE WEIGHTS DETECTED**

---

## Executive Summary

**CRITICAL FINDING:** Gate weights show **HIGH INSTABILITY** across random seeds, with most Coefficient of Variation (CV) values > 0.6. This indicates that gate weights converge to different attention configurations depending on initialization, raising concerns about their reliability for interpretation.

---

## Results

### Mean Gate Weights (averaged across 10 seeds)

| View | Red-Light | No Turn Signal | Helmet | Mobile Phone |
|------|-----------|----------------|--------|--------------|
| Rider Role | 0.066 | 0.229 | 0.208 | 0.438 |
| Rider Traits | 0.116 | 0.161 | 0.124 | 0.108 |
| Road Context | 0.449 | 0.240 | 0.170 | 0.177 |
| Environment | 0.254 | 0.111 | 0.177 | 0.135 |
| Site | 0.081 | 0.140 | 0.136 | 0.079 |

### Standard Deviation (across 10 seeds)

| View | Red-Light | No Turn Signal | Helmet | Mobile Phone |
|------|-----------|----------------|--------|--------------|
| Rider Role | 0.061 | 0.167 | 0.282 | 0.304 |
| Rider Traits | 0.145 | 0.128 | 0.169 | 0.116 |
| Road Context | 0.287 | 0.173 | 0.183 | 0.131 |
| Environment | 0.054 | 0.095 | 0.239 | 0.152 |
| Site | 0.125 | 0.148 | 0.155 | 0.082 |

### Coefficient of Variation (CV = std/mean)

| View | Red-Light | No Turn Signal | Helmet | Mobile Phone |
|------|-----------|----------------|--------|--------------|
| Rider Role | **0.925** | **0.727** | **1.358** | **0.693** |
| Rider Traits | **1.252** | **0.792** | **1.368** | **1.080** |
| Road Context | **0.638** | **0.719** | **1.081** | **0.737** |
| Environment | 0.211 | **0.854** | **1.346** | **1.131** |
| Site | **1.542** | **1.060** | **1.140** | **1.035** |

**Stability Classification:**
- **Stable (CV < 0.2):** 0/20 (0.0%)
- **Moderate (0.2 ≤ CV < 0.5):** 1/20 (5.0%) - Environment for Red-Light only
- **Unstable (CV ≥ 0.5):** 19/20 (95.0%)

---

## Interpretation

### 1. What This Means

**Gate weights are highly sensitive to initialization and training dynamics.** Different random seeds lead to substantially different attention patterns, even though final predictive performance is similar (Test AUC range: 0.7107 - 0.7443).

### 2. Why This Happens

**Multiple local optima:** The gated fusion mechanism can converge to different attention configurations that achieve similar predictive performance. This is a known issue with attention mechanisms in neural networks.

**Possible causes:**
- Gate temperature allows multiple near-optimal sparsemax/mixed-prior solutions
- Regularization (lambda prior) may not be strong enough
- View representations may be partially redundant
- Multi-task loss creates complex optimization landscape

### 3. Implications for Interpretation

⚠️ **CRITICAL:** Gate weights should NOT be interpreted as stable, reliable indicators of view importance.

**What we CAN say:**
- ✓ Gate weights show that the model uses attention mechanisms
- ✓ Different tasks attend to different views (task-specific gates work)
- ✓ On average across seeds, some patterns emerge (e.g., Road Context for Red-Light)

**What we CANNOT say:**
- ✗ "View X is the most important for Task Y" (unstable across seeds)
- ✗ "Gate weights reveal the true importance ranking" (configuration-dependent)
- ✗ "These attention patterns are robust" (CV > 0.6 for most)

### 4. Comparison with Other Methods

**This is why we need triple validation:**

| Method | Stability | Interpretation |
|--------|-----------|----------------|
| **Gate Weights** | **Unstable (CV > 0.6)** | **Attention patterns vary by seed** |
| **Integrated Gradients** | Unknown (not tested) | Population-level attribution |
| **Ablation Study** | Stable (deterministic) | Performance drop is consistent |

**Ablation is the most reliable method** because it directly measures performance impact, not learned attention weights.

---

## Revised Interpretation Strategy

### Before (Naive Interpretation)
> "Gate weights show that Rider Role is most important for Mobile Phone Use (α = 0.438)."

### After (Stability-Aware Interpretation)
> "Ablation study shows that removing Rider Role causes the largest performance drop (ΔAUC = 0.1182), indicating it is the most critical view. Gate weights vary substantially across random seeds (CV = 0.693 for Mobile Phone Use), suggesting attention patterns are sensitive to initialization. However, averaged across 10 seeds, Rider Role receives the highest mean gate weight for Mobile Phone Use (0.438 ± 0.304), consistent with the ablation finding."

---

## Recommendations for Paper

### 1. Acknowledge Instability

**In Methods Section:**
> "Gate weights were analyzed across 10 random seeds to assess stability. We report mean ± standard deviation and coefficient of variation (CV) to quantify variability."

**In Results Section:**
> "Gate weights showed high variability across random seeds (mean CV = 0.95), indicating sensitivity to initialization. We therefore prioritize ablation-based importance measures, which are deterministic and directly measure performance impact."

### 2. Prioritize Ablation Over Gates

**Reporting Order:**
1. **Ablation Study** (most reliable)
2. **Integrated Gradients** (population-level)
3. **Gate Weights** (averaged across seeds, with caveats)

### 3. Use Gates for Task-Specific Patterns Only

Gate weights are still useful for showing that **different tasks attend to different views**, even if the exact weights are unstable.

**Example:**
> "Task-specific gates reveal different attention patterns: Red-Light Running attends more to Road Context (mean α = 0.449), while Mobile Phone Use attends more to Rider Role (mean α = 0.438). However, these patterns show high variability across seeds (CV > 0.6), suggesting multiple attention configurations achieve similar performance."

### 4. Recommend Future Work

> "Future work should explore methods to stabilize gate weights, such as:
> - Stronger regularization (higher lambda prior)
> - Ensemble averaging across multiple seeds
> - Deterministic attention mechanisms (e.g., learned fixed weights)
> - Bayesian approaches to quantify uncertainty in attention patterns"

---

## Impact on Previous Claims

### Claims That Still Hold
✓ TG-MVMT-GFNet v2 outperforms baselines (performance is stable)  
✓ Rider Role is critical (ablation drop = 0.1182, consistent)  
✓ Multi-task learning provides positive transfer (MTL ratios stable)  
✓ Task-specific gates learn different patterns (qualitative finding)  

### Claims That Need Revision
⚠️ "Gate weights show that View X is most important" → Add "averaged across seeds" and report CV  
⚠️ "Gate weights are consistent with theory" → Weaken to "on average, patterns are consistent"  
⚠️ "Triple validation confirms importance" → Emphasize ablation, de-emphasize gates  

---

## Comparison with Literature

**This finding is consistent with prior work on attention instability:**

- Jain & Wallace (2019): "Attention is not Explanation" - attention weights are not faithful explanations
- Serrano & Smith (2019): "Is Attention Interpretable?" - attention can be arbitrary
- Wiegreffe & Pinter (2019): "Attention is not not Explanation" - attention can be useful but needs validation

**Our contribution:** We provide empirical evidence of gate weight instability in multi-view multi-task fusion, and recommend ablation-based validation.

---

## Updated Conclusion

**Gate weights are useful for showing that task-specific attention mechanisms work, but they are NOT reliable indicators of view importance due to high instability across random seeds (CV > 0.6 for 95% of task-view pairs).**

**Ablation study is the gold standard for measuring view importance** because it directly measures performance impact and is deterministic.

**For the paper:** Report gate weights as supplementary evidence, averaged across seeds with error bars, but prioritize ablation results for importance claims.

---

## Action Items

1. ✅ Update COMPLETION_REPORT.md with gate stability findings
2. ✅ Revise interpretation guidelines in GATE_WEIGHTS_VS_CAUSALITY.md
3. ✅ Update ABLATION_AND_BASELINE_COMPARISON.md to emphasize ablation over gates
4. 📝 Draft manuscript section on gate stability
5. 📊 Create visualization showing gate weight variability (if plots were generated)

---

## References

- Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *NAACL*.
- Serrano, S., & Smith, N. A. (2019). Is Attention Interpretable? *ACL*.
- Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation. *EMNLP*.

---

**Prepared by:** Claude (Kiro)  
**Date:** 2026-05-12  
**Status:** ⚠️ Critical finding - Gate weights unstable, prioritize ablation
