# Competing Methods and Baseline Relevance Research

**Updated:** 2026-05-14  
**Scope:** M2G-Net v2 for motorcycle risky-behavior / traffic-safety tabular prediction.  
**Status:** This repository currently uses synthetic proof-of-concept data unless replaced by
real observational records.

This file lists credible competing methods for M2G-Net v2 and assesses whether
the baselines currently run locally are still widely used. The goal is not to
claim that M2G-Net must beat every method, but to make the comparison set
defensible for a paper, thesis, or technical report.

## Executive Answer

Yes, the models currently run locally are still widely used and methodologically
defensible:

- **Logistic Regression, Decision Tree, Random Forest** are classic and still
  common reference baselines, especially through scikit-learn.
- **XGBoost and LightGBM** are strong tabular gradient-boosting competitors and
  should be treated as the most important predictive baselines.
- **Early-Fusion MLP** is the necessary neural baseline for asking whether
  multi-view gated fusion improves over simple feature concatenation.
- **Single-Task MLP** is necessary for testing whether multi-task learning helps
  or hurts each behavior.

However, the current comparison set should be strengthened before making a
publishable "outperforms competitors" claim. The most useful additions are:
**CatBoost**, **SVM**, **KNN/Naive Bayes as lightweight classic baselines**,
**AdaBoost/Gradient Boosting**, and at least one modern deep-tabular baseline
such as **TabNet** or **FT-Transformer** if runtime allows.

## 1. Local Baselines Currently Implemented

The local runner is `baselines/run_all_baselines.py`. It currently evaluates:

| Local method | Role in comparison | Current relevance |
|---|---|---|
| Decision Tree | Interpretable anchor baseline; close to CART-style traffic-safety analysis. | Still useful, but not usually the strongest modern predictor. Keep it for interpretability and comparability. |
| Logistic Regression | Linear/statistical sanity-check baseline. | Still widely used as a transparent baseline, especially when coefficients and odds-style interpretation matter. |
| Random Forest | Nonlinear tree ensemble baseline. | Very common in road-safety and motorcycle crash-severity studies; often competitive. |
| XGBoost | Strong gradient-boosted tree baseline. | Highly relevant for tabular ML and transport-safety prediction; should be tuned carefully. |
| LightGBM | Efficient gradient-boosted tree baseline. | Highly relevant for larger/high-dimensional tabular data; good complement to XGBoost. |
| Early-Fusion MLP | Neural baseline using concatenated flat features. | Essential to show value beyond "all features -> MLP." |
| Single-Task MLP | One model per task. | Essential to detect negative transfer in multi-task learning. |

Current tracked single-split macro ROC-AUC values are documented in
`docs/results/current_metrics.md`: M2G-Net v2 0.7095, Early-Fusion MLP 0.7085,
Decision Tree 0.7069, Single-Task MLP 0.7013, LightGBM 0.6996, XGBoost 0.6657,
Random Forest 0.6588, Logistic Regression 0.6461.

## 2. Evidence From Transport-Safety Literature

Road-safety machine-learning reviews consistently place this project inside a
well-established comparison landscape:

| Evidence | Implication for M2G-Net baselines |
|---|---|
| A systematic road-safety ML review identifies nearest-neighbor classifiers, decision trees, evolutionary algorithms, support-vector machines, and artificial neural networks as major technique groups, and notes neural networks are heavily used in crash prediction [1]. | The current tree/MLP baselines are aligned, but SVM and KNN are missing if the goal is a broad classic-ML comparison. |
| A 2023 systematic review of crash-prediction ML covers crash occurrence, frequency, and injury severity, and stresses data imbalance and robust evaluation as open needs [2]. | M2G-Net should report class-imbalance-aware metrics such as PR-AUC, F1, balanced accuracy, and calibration, not only ROC-AUC. |
| A two-wheeler crash-severity study uses XGBoost with SHAP for local/global interpretability [3]. | XGBoost is a directly relevant competitor, and SHAP is a useful interpretability comparator against gate weights / Integrated Gradients. |
| A motorcycle injury-severity study compares Random Forest, SVM, MARS, and binary logistic regression, using AUC and confusion matrix; Random Forest performed best in that study [4]. | Random Forest, SVM, and Logistic Regression are credible motorcycle-safety baselines. MARS is optional but can be mentioned as a statistical/nonlinear comparator. |
| A recent motorcycle crash injury-severity review table lists DT, LR, RF, GB, XGBoost, KNN, SVM, Naive Bayes, gradient-boosted trees, and SHAP-based interpretation across recent studies [5]. | The local set is strong but incomplete: add CatBoost/GBM/AdaBoost plus SVM/KNN/NB if aiming for a thorough baseline table. |

## 3. Evidence From General ML / Tabular ML Usage

| Source evidence | Relevance |
|---|---|
| XGBoost's original paper describes it as a scalable tree-boosting system widely used by data scientists to achieve state-of-the-art results on ML challenges [6]. | XGBoost remains a must-have strong baseline for tabular prediction. |
| LightGBM's NeurIPS paper describes GBDT as popular and proposes GOSS/EFB for efficiency and scalability when feature dimension and data size are large [7]. | LightGBM is a strong efficient GBDT baseline, especially for high-dimensional categorical/one-hot-style tabular data. |
| Kaggle's 2022 DS/ML survey reports scikit-learn as the most popular ML framework and notes PyTorch's steady growth; it also discusses boosted-tree dominance on Kaggle tabular tasks [8]. | Local scikit-learn baselines and PyTorch MLP baselines are not obsolete; they match mainstream practice. |
| Python Developers Survey 2024 reports that among respondents who train/generate predictions with ML models, scikit-learn is used by 68%, PyTorch by 66%, and XGBoost by 23% [9]. | Current local stack is widely used: scikit-learn covers LR/DT/RF, PyTorch covers MLP/M2G-Net, XGBoost is still common. |
| TabNet introduces attentive tabular learning with sequential feature selection for interpretability [10]. | A modern deep-tabular baseline can strengthen claims that M2G-Net adds value beyond standard MLPs. |

## 4. Recommended Competitor List for the Paper

### Tier A: Must Keep

These are essential for a defensible comparison.

| Method | Why it matters | Local status |
|---|---|---|
| Decision Tree / CART | Matches classic interpretable safety analysis and is easy to explain to transportation readers. | Implemented |
| Logistic Regression | Transparent statistical baseline; sanity check for whether nonlinear models add value. | Implemented |
| Random Forest | Common, robust nonlinear ensemble in crash and motorcycle severity studies. | Implemented |
| XGBoost | Strong tabular GBDT benchmark; directly used in two-wheeler safety work. | Implemented |
| LightGBM | Efficient GBDT benchmark; commonly compared with XGBoost/RF/CatBoost. | Implemented |
| Early-Fusion MLP | Tests whether view-specific/gated design beats simple neural feature concatenation. | Implemented |
| Single-Task MLP | Tests multi-task transfer and detects task-level negative transfer. | Implemented |

### Tier B: Should Add Before Submission

These would make the comparison set much stronger.

| Method | Why add it | Priority |
|---|---|---|
| CatBoost | Strong GBDT for categorical tabular data; often appears with XGBoost/LightGBM/RF in recent safety ML. | High |
| SVM / SVC | Frequently used in road-safety and motorcycle injury-severity comparisons. | High |
| KNN | Simple non-parametric baseline, common in comparative crash-severity studies. | Medium |
| Naive Bayes | Lightweight probabilistic baseline; useful as a lower-bound classic classifier. | Medium |
| AdaBoost / sklearn GradientBoosting | Classic boosting baseline that bridges Decision Tree and modern GBDT. | Medium |
| SHAP analysis for XGBoost/LightGBM | Gives an interpretability comparator against gate weights and Integrated Gradients. | High |

### Tier C: Optional Advanced Competitors

Add these only if the paper wants to make a stronger ML-methods claim.

| Method | Why add it | Caveat |
|---|---|---|
| TabNet | Interpretable deep-tabular model with attentive feature selection. | Can be sensitive to tuning and dataset size. |
| FT-Transformer / SAINT / TabTransformer | Modern transformer-style tabular baselines. | More compute and tuning; may be excessive for small synthetic data. |
| Shared-bottom MTL MLP | Cleaner multi-task baseline than Early-Fusion MLP if the current Early-Fusion MLP is already multi-head. | Useful to isolate "multi-task" from "multi-view/gated" gains. |
| Hierarchical / mixed-effects logistic model | Strong statistical comparator if site/intersection effects are central. | Needs careful implementation and reporting; not directly comparable to deep representations. |
| Graph Neural Network | Relevant if intersections, roads, or trajectories form an explicit graph. | Not justified unless graph structure is available. |
| Sequence model, e.g. LSTM/GRU/Temporal Transformer | Relevant if behavior is time-series or trajectory-based. | Not justified for purely cross-sectional tabular records. |

## 5. Are the Local Models Widely Used?

Short answer: **yes, mostly**.

| Local model | Popularity assessment | What to say in the paper |
|---|---|---|
| Logistic Regression | Very common baseline in applied prediction and statistical modeling. | "Transparent linear/statistical baseline." |
| Decision Tree | Common interpretable baseline, but usually not the strongest standalone model. | "Interpretable CART-style baseline and replication anchor." |
| Random Forest | Very common in road-safety ML and motorcycle severity modeling. | "Robust nonlinear ensemble baseline." |
| XGBoost | Still a major tabular baseline; also explicitly reported as used by 23% of Python ML respondents in 2024 survey data [9]. | "Strong gradient-boosted tree baseline." |
| LightGBM | Widely used efficient GBDT; especially credible for larger/sparse/high-dimensional tabular problems. | "Efficient gradient-boosted tree baseline." |
| Early-Fusion MLP | Standard neural baseline; PyTorch is heavily used in current ML workflows [9]. | "Neural concatenation baseline." |
| Single-Task MLP | Standard control for multi-task learning. | "Negative-transfer control." |

The weakest current issue is not that the local models are unpopular. The issue
is that the strongest local tree baselines appear only lightly tuned in
`baselines/run_all_baselines.py`. Before making a high-confidence superiority
claim, XGBoost/LightGBM/CatBoost should be tuned with validation or nested
cross-validation, and repeated-run statistics should be reported.

## 6. Suggested Comparison Matrix for a Manuscript

| Comparison question | Required competitor |
|---|---|
| Does M2G-Net beat simple interpretable models? | Decision Tree, Logistic Regression |
| Does it beat strong tabular ML? | Random Forest, XGBoost, LightGBM, CatBoost |
| Does view structure help beyond flat features? | Early-Fusion MLP |
| Does multi-task learning help? | Single-Task MLP and MTL transfer ratio |
| Does gated fusion help? | M2G-Net without gate, or early/late-fusion ablation |
| Does cross-level interaction help? | M2G-Net without interaction module |
| Are explanations consistent? | SHAP for XGBoost/LightGBM plus Integrated Gradients/gate diagnostics |
| Does it generalize to new sites? | Leave-site-out and synthetic/real independent-site tests |

## 7. Recommended Local Implementation Updates

1. Add `CatBoostClassifier` if dependency constraints allow it.
2. Add `SVC(probability=True)` or calibrated linear/RBF SVM.
3. Add `KNeighborsClassifier` and `GaussianNB` as lightweight classic baselines.
4. Add `AdaBoostClassifier` and/or `GradientBoostingClassifier`.
5. Add validation-tuned hyperparameter search for XGBoost, LightGBM, Random
   Forest, and CatBoost.
6. Add SHAP reports for the best GBDT baseline.
7. Report repeated seeds or repeated folds for at least M2G-Net, XGBoost,
   LightGBM, CatBoost, Random Forest, and Early-Fusion MLP.

## 8. Cautious Claim Wording

Recommended wording:

> We compare M2G-Net v2 with interpretable statistical/tree baselines,
> strong gradient-boosted tabular baselines, and neural MLP controls. This
> comparison tests whether theory-guided multi-view gated fusion offers
> complementary predictive and interpretive value beyond standard tabular and
> neural approaches.

Avoid:

> M2G-Net is superior to all existing methods.

That claim would require real data, stronger tuning, more competitor families,
and robust repeated/fold-level statistical tests.

## References

[1] Silva, P. B., Andrade, M., & Ferreira, S. "Machine learning applied to road
safety modeling: A systematic literature review." Journal of Traffic and
Transportation Engineering, 2020. https://www.sciencedirect.com/science/article/pii/S2095756420301410

[2] "Advances, challenges, and future research needs in machine learning-based
crash prediction models: A systematic review." Accident Analysis & Prevention,
2023. https://www.sciencedirect.com/science/article/pii/S0001457523004256

[3] "Investigating two-wheelers risk factors for severe crashes using an
interpretable machine learning approach and SHAP analysis." 2023.
https://www.sciencedirect.com/science/article/pii/S0386111223000353

[4] "Using machine learning techniques for evaluation of motorcycle injury
severity." 2021. https://www.sciencedirect.com/science/article/pii/S0386111220300649

[5] "A Comparative Study of a Series of Supervised Learning Models for
Motorcycle Crash Injury Severity Prediction." Civil Engineering Journal, 2025.
https://civilejournal.org/index.php/cej/article/download/6364/1955

[6] Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD
2016. https://arxiv.org/abs/1603.02754

[7] Ke, G. et al. "LightGBM: A Highly Efficient Gradient Boosting Decision
Tree." NeurIPS 2017.
https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boost

[8] Kaggle. "State of Machine Learning and Data Science Report 2022."
https://storage.googleapis.com/kaggle-media/surveys/Kaggle%20State%20of%20Machine%20Learning%20and%20Data%20Science%20Report%202022.pdf

[9] Python Software Foundation & JetBrains. "Python Developers Survey 2024
Results." https://lp.jetbrains.com/python-developers-survey-2024/

[10] Arik, S. O., & Pfister, T. "TabNet: Attentive Interpretable Tabular
Learning." https://arxiv.org/abs/1908.07442
