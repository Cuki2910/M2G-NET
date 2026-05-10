# Mathematical Formulation: TG-MVMT-GFNet v2

This file is the final mathematical reference for TG-MVMT-GFNet v2. It is
intended to be the single source of truth for the current model formulation.
The repository data files are synthetic proof-of-concept datasets unless
replaced by real observational data.

## Notation

Let
\[
    \mathcal{D}=\{(x^{(n)},y^{(n)})\}_{n=1}^{N}
\]
denote the training set, where
\[
    y^{(n)}=(y_1^{(n)},\ldots,y_K^{(n)})\in\{0,1\}^{K},
    \qquad K=4 .
\]
Each input sample is represented as a multi-view tuple
\[
    x = (x_1,\ldots,x_V), \qquad V=5,
\]
corresponding to rider role, rider traits, road context, environment, and
site-observed views.

Let \(S_{\mathrm{train}}\) be the number of sites observed during training.
The site identifier satisfies
\[
    s\in\{1,\ldots,S_{\mathrm{train}}\}.
\]
For partially observed outcomes, define
\[
    m_k^{(n)}\in\{0,1\},
\]
where \(m_k^{(n)}=1\) only when task \(k\)'s label is observed and applicable
for sample \(n\).

## 1. View-Specific Encoding

Each view is encoded by a view-specific embedding and MLP encoder:
\[
    h_i = f_i(x_i),
    \qquad
    h_i\in\mathbb{R}^{d},
    \quad i=1,\ldots,V .
\]
The individual and contextual representations are
\[
    h_{\mathrm{ind}} = [h_{\mathrm{role}};h_{\mathrm{traits}}],
    \qquad
    h_{\mathrm{ctx}} = [h_{\mathrm{road}};h_{\mathrm{env}};h_{\mathrm{site}}].
\]

## 2. Site-Aware Representation

Let \(s\) denote the site identifier and let \(h_{\mathrm{obs}}\) denote the
encoded observed site-level covariates from the site-observed view. The
site-aware representation uses a regularized site-specific embedding:
\[
    h_{\mathrm{site}}
    =
    \phi\!\left([h_{\mathrm{obs}};h_{\mathrm{rand}}]\right),
    \qquad
    h_{\mathrm{rand}} = E_{\mathrm{site}}(s),
\]
where \(E_{\mathrm{site}}\) is a learnable site embedding trained with weight
decay. This component acts as a site-specific intercept-like representation,
but it should not be interpreted as a full probabilistic mixed-effects model.
For unseen sites at inference time, the site-specific component is set to zero:
\[
    h_{\mathrm{rand}}=\mathbf{0}.
\]
The site projection function is
\[
    \phi:\mathbb{R}^{d_{\mathrm{obs}}+d_{\mathrm{site}}}\rightarrow
    \mathbb{R}^{d}.
\]

## 3. Cross-Level Interaction

The model captures interactions between individual-level and contextual-level
factors using projected additive and multiplicative terms:
\[
    p_{\mathrm{ind}} = W_{\mathrm{ind}}h_{\mathrm{ind}},
    \qquad
    p_{\mathrm{ctx}} = W_{\mathrm{ctx}}h_{\mathrm{ctx}},
\]
with
\[
    W_{\mathrm{ind}}\in\mathbb{R}^{d\times 2d},
    \qquad
    W_{\mathrm{ctx}}\in\mathbb{R}^{d\times 3d}.
\]
\[
    h_{\mathrm{int}}
    =
    \psi\!\left(
        p_{\mathrm{ind}}
        + p_{\mathrm{ctx}}
        + W_{\mathrm{int}}
          \left(p_{\mathrm{ind}}\odot p_{\mathrm{ctx}}\right)
    \right),
    \qquad
    h_{\mathrm{int}}\in\mathbb{R}^{d},
\]
where \(\odot\) denotes element-wise multiplication.
Here
\[
    W_{\mathrm{int}}\in\mathbb{R}^{d\times d},
    \qquad
    \psi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}.
\]

## 4. Task-Specific Gated Fusion

For task \(k\), gated attention is computed over the view representations and
the cross-level interaction representation. Let
\[
    \tilde{h}_j=P_jh_j,\quad j=1,\ldots,V,
    \qquad
    \tilde{h}_{V+1}=P_{\mathrm{int}}h_{\mathrm{int}},
\]
where
\[
    P_j\in\mathbb{R}^{d\times d},
    \qquad
    P_{\mathrm{int}}\in\mathbb{R}^{d\times d}.
\]
All \(\tilde{h}_j\) therefore have the same dimensionality \(d\). Define
\[
    r=[\tilde{h}_1;\ldots;\tilde{h}_{V+1}]\in\mathbb{R}^{(V+1)d}.
\]
The task-specific attention weights are
\[
    \alpha_k
    =
    \operatorname{softmax}
    \left(
        \frac{A_k r + c_k + \pi}{T}
    \right),
\]
where
\[
    A_k\in\mathbb{R}^{(V+1)\times (V+1)d},
    \qquad
    c_k\in\mathbb{R}^{V+1}.
\]
The prior term is
\[
    \pi=\lambda p_0,
    \qquad
    p_0=\frac{1}{V+1}\mathbf{1}_{V+1},
    \qquad
    \pi\in\mathbb{R}^{V+1},
\]
where \(\lambda\ge 0\) controls the strength of the prior and \(T>0\) is the
gate temperature.
The gated representation for task \(k\) is
\[
    z_{\mathrm{gated}}^{(k)}
    =
    \sum_{j=1}^{V+1}\alpha_{k,j}\tilde{h}_j .
\]

## 5. Residual Early-Fusion Blending

The early-fusion baseline representation is
\[
    z_{\mathrm{early}} = g([h_1;\ldots;h_V]).
\]
where
\[
    g:\mathbb{R}^{Vd}\rightarrow\mathbb{R}^{d}.
\]
TG-MVMT-GFNet combines early fusion and task-specific gated fusion using a
learnable scalar blending coefficient:
\[
    z_k
    =
    \operatorname{sigmoid}(a)z_{\mathrm{gated}}^{(k)}
    +
    \left(1-\operatorname{sigmoid}(a)\right)z_{\mathrm{early}},
\]
where \(a\in\mathbb{R}\) is a shared scalar learned jointly with the model
parameters.

## 6. Task Prediction

For each task \(k\), the prediction head produces a binary probability:
\[
    \hat{y}_k
    =
    \operatorname{sigmoid}\!\left(w_k^{\top}z_k+\beta_k\right),
    \qquad k=1,\ldots,K .
\]

## 7. Training Objective

For sample \(n\) and task \(k\), the binary cross-entropy loss is
\[
    \ell_k^{(n)}
    =
    -y_k^{(n)}\log \hat{y}_k^{(n)}
    -
    \left(1-y_k^{(n)}\right)
    \log\left(1-\hat{y}_k^{(n)}\right).
\]
The model is trained using a masked uncertainty-weighted multi-task objective
adapted from Kendall et al.~\cite{kendall2018multi}. For numerical stability,
the model learns the log-variance parameter \(s_k=\log\sigma_k^2\):
\[
    \mathcal{L}
    =
    \sum_{k=1}^{K}
    \left[
        \frac{1}{2}\exp(-s_k)
        \frac{\sum_{n=1}^{N}m_k^{(n)}\ell_k^{(n)}}
             {\sum_{n=1}^{N}m_k^{(n)}+\epsilon}
        +
        \frac{1}{2}s_k
    \right],
\]
where \(\sigma_k>0\) is the task uncertainty associated with task \(k\), and
\(\epsilon\) avoids division by zero when a task has no observed labels in a
mini-batch. The equivalent unnormalized form is
\[
    \mathcal{L}_{\mathrm{sum}}
    =
    \sum_{n=1}^{N}\sum_{k=1}^{K}
    m_k^{(n)}
    \left[
        \frac{1}{2}\exp(-s_k)\ell_k^{(n)}
        +
        \frac{1}{2}s_k
    \right].
\]

## 8. Optimization

The objective is optimized with Adam and weight decay. The learning rate and
gate temperature are scheduled using cosine annealing:
\[
    q(e)
    =
    q_{\min}
    +
    \frac{1}{2}
    \left(q_{\max}-q_{\min}\right)
    \left(1+\cos\frac{\pi e}{E}\right),
\]
where \(e\) is the current epoch and \(E\) is the total number of epochs.

## 9. Evaluation

The primary evaluation metric is macro-averaged ROC-AUC:
\[
    \operatorname{AUC}_{\mathrm{macro}}
    =
    \frac{1}{K}
    \sum_{k=1}^{K}\operatorname{AUC}_k .
\]
Here \(\operatorname{AUC}_k\) denotes the ROC-AUC for binary task \(k\),
computed only over test samples with \(m_k^{(n)}=1\).

## 10. Code Mapping

| Component | Implementation |
|---|---|
| View-specific encoders | `src/views.py` |
| Site-aware encoder | `src/views.py` |
| Cross-level interaction | `src/interaction.py` |
| Gated/residual fusion | `src/fusion.py` |
| Task prediction heads | `src/model.py` |
| Multi-task objective | `src/loss.py` |
| Evaluation metrics | `src/metrics.py` |

## BibTeX

```bibtex
@inproceedings{kendall2018multi,
  title     = {Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
  author    = {Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {7482--7491},
  year      = {2018}
}
```
