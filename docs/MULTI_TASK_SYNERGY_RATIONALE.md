# Multi-Task Synergy Rationale

**Purpose:** Justify why red-light running, no turn signal, helmet non-use, and mobile phone use should be modeled jointly in a multi-task learning framework.

---

## Executive Summary

**Core Argument:** These four risky behaviors are not independent outcomes. They share latent risk-taking traits, common contextual triggers, and correlated decision-making processes. Multi-task learning allows the model to learn shared representations of these underlying factors, improving prediction and revealing cross-behavior patterns that single-task models would miss.

---

## 1. Theoretical Foundation

### 1.1. Risk Homeostasis Theory (Wilde, 1982)

**Key Principle:** Individuals maintain a target level of subjective risk. When perceived risk decreases in one domain, they may compensate by taking more risk in another.

**Implication for Multi-Task Learning:**
- A rider who perceives low enforcement risk may simultaneously:
  - Run red lights (traffic violation)
  - Not use turn signals (communication violation)
  - Use mobile phone (distraction violation)
- These behaviors are not independent choices but manifestations of a shared risk tolerance level.

**Multi-task advantage:** The model can learn a shared representation of "risk tolerance" that predicts multiple behaviors simultaneously, rather than treating each as an isolated outcome.

---

### 1.2. Theory of Planned Behavior (Ajzen, 1991)

**Key Principle:** Behavior is determined by:
- **Attitude:** Personal evaluation of the behavior
- **Subjective Norm:** Perceived social pressure
- **Perceived Behavioral Control:** Belief in one's ability to perform the behavior

**Implication for Multi-Task Learning:**
- A rider with a "rule-breaking attitude" is likely to violate multiple traffic rules
- A rider in a peer group with lax safety norms may exhibit multiple risky behaviors
- A rider who perceives low enforcement (low behavioral control cost) may engage in multiple violations

**Multi-task advantage:** Shared encoders can capture latent attitude and norm factors that influence all four behaviors, rather than learning separate attitude representations for each.

---

### 1.3. Occupational Risk Theory

**Key Principle:** Occupational roles shape risk exposure and risk-taking behavior through:
- Time pressure (delivery deadlines)
- Economic incentives (more trips = more income)
- Exposure frequency (professional riders spend more time on road)

**Implication for Multi-Task Learning:**
- Food delivery riders under time pressure may:
  - Run red lights to save time
  - Skip turn signals to move faster
  - Use mobile phones to check orders
  - Skip helmet fastening to save seconds
- These behaviors cluster together because they share a common cause: occupational time pressure.

**Multi-task advantage:** The model can learn that "rider role = food delivery" is a shared risk factor across multiple behaviors, improving sample efficiency.

---

### 1.4. Situational Crime Prevention (Clarke, 1980)

**Key Principle:** Crime (including traffic violations) occurs when:
- Opportunity is present
- Guardianship is absent
- Target is attractive

**Implication for Multi-Task Learning:**
- At an intersection with no police presence and dense traffic:
  - Red-light running becomes easier (low guardianship)
  - Not signaling is less noticeable (low detection)
  - Mobile phone use is harder to spot (low enforcement)
- These behaviors co-occur because they share the same situational opportunity structure.

**Multi-task advantage:** Contextual encoders (road context, environment) can learn shared opportunity structures that enable multiple violations simultaneously.

---

## 2. Empirical Evidence for Correlation

### 2.1. Expected Correlation Patterns

If these behaviors are driven by shared latent factors, we expect:

**Positive correlation within individuals:**
- Riders who run red lights are more likely to also not signal
- Riders who don't wear helmets are more likely to use phones while riding

**Shared predictors:**
- Rider role should predict multiple behaviors (occupational pressure)
- Police presence should reduce multiple behaviors (guardianship)
- Traffic density should affect multiple behaviors (opportunity)

### 2.2. What Single-Task Models Miss

**Single-task approach:**
```
Model 1: Rider Role → Red-light running
Model 2: Rider Role → No turn signal
Model 3: Rider Role → Helmet non-use
Model 4: Rider Role → Mobile phone use
```

Each model learns a separate representation of "Rider Role." If the dataset is small, each model has limited samples to learn this representation.

**Multi-task approach:**
```
Shared Encoder: Rider Role → h_role
Task-specific heads: h_role → [Red-light, No signal, Helmet, Phone]
```

The shared encoder learns from all four tasks simultaneously, improving representation quality.

---

## 3. Multi-Task Learning Mechanisms

### 3.1. Shared Representation Learning

**Mechanism:** Early layers learn shared features; late layers specialize.

**Example:**
- Shared encoder learns: "Food delivery rider under time pressure"
- Task-specific gates learn:
  - Red-light task: weight road context heavily (intersection design matters)
  - Helmet task: weight police presence heavily (enforcement matters)
  - Phone task: weight traffic density heavily (detection difficulty matters)

**Benefit:** Sample efficiency — the shared encoder sees 4× more training signal.

---

### 3.2. Regularization Through Multi-Task Constraints

**Mechanism:** Learning multiple tasks simultaneously prevents overfitting to any single task.

**Example:**
- If the model overfits to red-light running by memorizing site-specific patterns, this will hurt performance on other tasks that don't share those patterns.
- Multi-task loss forces the model to learn generalizable features.

**Benefit:** Better generalization to new sites and contexts.

---

### 3.3. Cross-Task Knowledge Transfer

**Mechanism:** Information learned from one task improves another.

**Example:**
- Helmet non-use is rare (class imbalance).
- Red-light running is more common.
- The model learns from red-light data that "rider role + police presence" is important.
- This knowledge transfers to helmet prediction, even with fewer helmet samples.

**Benefit:** Improved performance on rare behaviors.

---

## 4. When Multi-Task Learning Fails: Negative Transfer

### 4.1. What is Negative Transfer?

**Definition:** When learning multiple tasks together hurts performance compared to learning each task separately.

**Causes:**
- Tasks have conflicting optimization objectives
- Tasks require fundamentally different representations
- One task dominates gradient updates, starving others

### 4.2. How We Detect Negative Transfer

**MTL Transfer Ratio:**
```
MTL_Transfer_Ratio_k = MTL_AUC_k / Single_Task_AUC_k
```

- Ratio > 1: Multi-task helps task k (positive transfer)
- Ratio < 1: Multi-task hurts task k (negative transfer)

### 4.3. What Negative Transfer Would Tell Us

**If we find negative transfer for a task:**
- That behavior has a fundamentally different causal mechanism
- It does not share latent factors with other behaviors
- It should be modeled separately

**This is a valuable research finding, not a failure.**

**Example interpretation:**
> "We found negative transfer for helmet non-use (MTL ratio = 0.92), suggesting that helmet behavior is driven by different factors than other traffic violations. While red-light running, no signaling, and phone use share occupational and situational risk factors, helmet use may be more strongly influenced by personal safety attitudes and enforcement history, which do not transfer across tasks."

---

## 5. Why These Four Behaviors Specifically?

### 5.1. Red-Light Running
- **Type:** Active traffic violation
- **Risk:** Collision with cross-traffic
- **Shared factors:** Time pressure, low enforcement, intersection design

### 5.2. No Turn Signal
- **Type:** Communication failure
- **Risk:** Collision due to unpredictability
- **Shared factors:** Time pressure, low enforcement, traffic density

### 5.3. Helmet Non-Use / Not Fastened
- **Type:** Passive safety violation
- **Risk:** Injury severity in crash
- **Shared factors:** Enforcement, rider role norms, perceived invulnerability

### 5.4. Mobile Phone Use
- **Type:** Distraction violation
- **Risk:** Reduced attention, delayed reaction
- **Shared factors:** Occupational demands (checking orders), low enforcement, traffic density

### 5.5. Common Thread

All four behaviors:
- Are observable at intersections
- Are influenced by rider role (occupational pressure)
- Are affected by enforcement (police presence)
- Reflect risk-taking propensity
- Can co-occur in the same observation

**This makes them ideal candidates for multi-task learning.**

---

## 6. Alternative Groupings (Why Not These?)

### 6.1. Why Not Group by Risk Type?

**Alternative:** Group "active violations" (red-light, no signal, phone) separately from "passive violations" (helmet).

**Problem:** This assumes risk type is the primary organizing principle. But empirically, occupational role may be a stronger shared factor across all four.

**Our approach:** Let the model learn whether tasks should share representations (via task-specific gates and MTL transfer ratios).

### 6.2. Why Not Single-Task Models?

**Alternative:** Train four separate models.

**Problem:**
- Wastes data (each model sees only 1/4 of the training signal)
- Cannot discover cross-behavior patterns
- Requires 4× more hyperparameter tuning

**Our approach:** Multi-task with negative transfer detection. If negative transfer occurs, we report it and recommend single-task modeling for that behavior.

---

## 7. Reporting Multi-Task Synergy in the Paper

### 7.1. In the Introduction

> "Risky riding behaviors at intersections are not isolated events but often co-occur within individuals and contexts. Risk Homeostasis Theory suggests that riders maintain a target level of subjective risk, leading to correlated violations across domains. Occupational Risk Theory posits that time pressure and economic incentives drive multiple risky behaviors simultaneously. We therefore model four behaviors jointly—red-light running, no turn signal, helmet non-use, and mobile phone use—to capture shared latent risk factors and improve sample efficiency."

### 7.2. In the Methods

> "We employ multi-task learning to model four risky behaviors simultaneously. This approach is justified by theoretical and empirical evidence that these behaviors share common causes: risk-taking propensity (Risk Homeostasis Theory), occupational pressure (Occupational Risk Theory), and situational opportunity (Situational Crime Prevention). Shared encoders learn representations of these latent factors, while task-specific gates allow each behavior to weight views differently. To detect negative transfer, we compare multi-task performance against single-task baselines using MTL Transfer Ratios."

### 7.3. In the Results

> "Multi-task learning improved performance for three of four behaviors (MTL ratios: red-light = 1.08, no signal = 1.12, phone = 1.05), indicating positive transfer from shared representations. Helmet non-use showed slight negative transfer (ratio = 0.96), suggesting this behavior may be driven by distinct factors. Gate weight analysis revealed that red-light running and no signaling both rely heavily on road context (α = 0.28, 0.17), consistent with shared situational opportunity structures."

### 7.4. In the Discussion

> "The positive transfer observed for most tasks supports our hypothesis that risky behaviors share latent risk-taking traits and contextual triggers. The slight negative transfer for helmet non-use suggests that helmet behavior may be more strongly influenced by personal safety attitudes and long-term enforcement history, which do not transfer across tasks. This finding aligns with prior research showing that helmet use is more habitual and less situationally dependent than active traffic violations."

---

## 8. Checklist for Multi-Task Justification

Before claiming multi-task learning is appropriate, verify:

- [ ] Theoretical rationale: Do theories predict shared causes?
- [ ] Empirical correlation: Do behaviors co-occur in the data?
- [ ] Shared predictors: Do the same features predict multiple behaviors?
- [ ] Sample efficiency: Is the dataset small enough that sharing helps?
- [ ] Negative transfer detection: Have we tested single-task baselines?
- [ ] Reporting: Have we stated the rationale clearly in the paper?

---

## 9. Summary

**Why multi-task learning for these four behaviors?**

1. **Theoretical:** Risk Homeostasis, Theory of Planned Behavior, Occupational Risk Theory, and Situational Crime Prevention all predict shared causes.

2. **Empirical:** Behaviors are expected to correlate within individuals and contexts.

3. **Methodological:** Shared encoders improve sample efficiency and enable cross-task knowledge transfer.

4. **Falsifiable:** We test for negative transfer and report it if found.

5. **Interpretable:** Task-specific gates reveal which behaviors share representations and which do not.

**Bottom line:** Multi-task learning is not just a modeling choice—it is a testable hypothesis about the structure of risky behavior.

---

## References

- Ajzen, I. (1991). The theory of planned behavior. *Organizational Behavior and Human Decision Processes*, 50(2), 179-211.
- Clarke, R. V. (1980). Situational crime prevention: Theory and practice. *British Journal of Criminology*, 20(2), 136-147.
- Wilde, G. J. (1982). The theory of risk homeostasis: Implications for safety and health. *Risk Analysis*, 2(4), 209-225.
- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
- Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*.

---

## Changelog

- 2026-05-12: Initial version — justify multi-task learning for four risky behaviors
