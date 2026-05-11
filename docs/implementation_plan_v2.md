# 🏗️ Implementation Plan: M2G-Net v2

**Model:** Theory-Guided Multi-View Multi-Task Gated Fusion Network (v2)
**Source:** [methodology_v2.md](file:///a:/GREEN-X/M2G-Net/m2g_net_methodology_v2.md) | [comparison](file:///a:/GREEN-X/M2G-Net/comparison_v1_vs_v2.md)
**Guidelines:** [CLAUDE.md](file:///a:/GREEN-X/M2G-Net/CLAUDE.md)

---

## Kiến trúc tổng quát v2

```text
Input
  ↓
Individual Views           Contextual Views
[Role] [Traits]       [Road] [Env] [Site*]
     ↓                        ↓
  Encoders                Encoders
     └──── Cross-Level Interaction ────┘
                      ↓
   Regularized Task-Specific Gated Fusion
   + Residual từ Early Fusion (alpha blend)
                      ↓
    Uncertainty-Weighted Multi-Task Output
                      ↓
   Gate Weights + Integrated Gradients
          (View Contribution Report)
```

*Site context = observed infrastructure features + site random intercept (zero-out khi test trên site mới)

---

## Cấu trúc thư mục

```text
M2G-Net/
├── config.py                    # Hyperparameters
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         # Load, split views, DataLoader
│   ├── views.py                 # View-specific encoders
│   ├── interaction.py           # [MỚI] Cross-level interaction module
│   ├── fusion.py                # Regularized task-specific gated fusion + residual
│   ├── model.py                 # Full M2G-Net v2
│   ├── loss.py                  # Uncertainty-weighted multi-task loss
│   └── metrics.py               # ROC-AUC, PR-AUC, F1, Balanced Acc, MTL Ratio
├── baselines/
│   ├── decision_tree.py
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── xgboost_baseline.py
│   ├── early_fusion_mlp.py
│   ├── late_fusion.py
│   └── single_task_mlp.py       # [MỚI] 4 single-task models cho negative transfer detection
├── train.py
├── evaluate.py
├── ablation.py
├── interpret.py                 # [MỚI] Gate weights + Integrated Gradients
└── notebooks/
    ├── 01_eda.ipynb
    ├── 02_baseline_comparison.ipynb
    └── 03_ablation_analysis.ipynb
```

---

## Phase 0: Project Setup

| # | Step | Verify |
|---|------|--------|
| 0.1 | Tạo cấu trúc thư mục | Đủ folders |
| 0.2 | Tạo `requirements.txt` | File tồn tại |
| 0.3 | Tạo `config.py` | Import thành công |

### `requirements.txt`

```text
torch>=2.0
captum                           # [MỚI] Cho Integrated Gradients
numpy
pandas
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
```

### `config.py`

```python
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# --- View Encoder ---
EMBEDDING_DIM = 16               # d: output dim mỗi view encoder
HIDDEN_DIM = 32
DROPOUT_RATE = 0.3

# --- Hierarchical Levels (MỚI v2) ---
INDIVIDUAL_VIEWS = ["rider_role", "rider_traits"]
CONTEXTUAL_VIEWS = ["road_context", "environment", "site"]
ALL_VIEWS = INDIVIDUAL_VIEWS + CONTEXTUAL_VIEWS

# --- Cross-Level Interaction (MỚI v2) ---
INTERACTION_DIM = 16             # d_inter: output dim của cross-level interaction

# --- Gated Fusion (MỚI v2) ---
NUM_GATE_INPUTS = 6              # 5 views + 1 interaction term
GATE_TEMPERATURE_INIT = 2.0      # Temperature khởi tạo cao → anneal xuống 1.0
GATE_PRIOR = "uniform"           # Prior đều để tránh mode collapse

# --- Residual Early Fusion (MỚI v2) ---
RESIDUAL_ALPHA_INIT = 0.5        # Learnable, blend giữa gated và early fusion

# --- Model ---
NUM_VIEWS = 5
NUM_TASKS = 4
TASK_NAMES = ["red_light_running", "no_turn_signal", "helmet_nonuse", "mobile_phone_use"]

# --- Training ---
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20

# --- Loss (MỚI v2: Uncertainty Weighting) ---
USE_UNCERTAINTY_WEIGHTING = True  # Homoscedastic uncertainty (Kendall et al., 2018)
USE_FOCAL_LOSS = False
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
```

---

## Phase 1: Data Pipeline

**File:** `src/data_pipeline.py`

| # | Step | Verify |
|---|------|--------|
| 1.1 | Load raw data | `df.shape` hợp lý |
| 1.2 | Xác định columns thuộc từng view | Print mapping |
| 1.3 | Handle missing values | `isnull().sum()` == 0 |
| 1.4 | Encode categoricals | Kiểm tra dtype |
| 1.5 | Tách 5 views thành dict | Shape mỗi view đúng |
| 1.6 | **[MỚI]** Tách site view thành `site_obs` + `site_id` | 2 tensors riêng |
| 1.7 | Tách targets: 4 binary columns | Chỉ có 0/1 |
| 1.8 | Train/Val/Test split (stratified) | Tỷ lệ class ổn định |
| 1.9 | Leave-Intersection-Out split | Giao lộ test ∉ train |
| 1.10 | PyTorch Dataset & DataLoader | Iterate 1 batch OK |

### View-Column Mapping

```python
VIEW_COLUMNS = {
    # --- Individual Level ---
    "rider_role": ["rider_type"],
    "rider_traits": ["gender", "age_group"],
    # --- Contextual Level ---
    "road_context": ["police_presence", "num_lanes", "traffic_condition"],
    "environment": ["weather", "time_slot"],
    "site": {                                    # [MỚI v2] Tách 2 phần
        "observed": ["num_lanes_site", "intersection_type", "has_signal"],
        "site_id": "intersection_id",            # Cho random intercept
    },
}
```

---

## Phase 2: View-Specific Encoders

**File:** `src/views.py`

| # | Step | Verify |
|---|------|--------|
| 2.1 | `ViewEncoder` base class | Khởi tạo OK |
| 2.2 | `RiderRoleEncoder` | Output: `(batch, d)` |
| 2.3 | `RiderTraitsEncoder` | Output: `(batch, d)` |
| 2.4 | `RoadContextEncoder` | Output: `(batch, d)` |
| 2.5 | `EnvironmentEncoder` | Output: `(batch, d)` |
| 2.6 | **[MỚI]** `SiteAwareEncoder` (observed features + random intercept) | Output: `(batch, d)` |
| 2.7 | Unit test all encoders | All shapes pass |

### SiteAwareEncoder (thay thế SiteEncoder v1)

```python
class SiteAwareEncoder(nn.Module):
    """
    V2: Observed site features + regularized site random intercept.
    Random intercept bị zero-out khi inference trên site mới.
    """
    def __init__(self, num_site_features, num_sites, emb_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        # (a) Observed site features encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(num_site_features, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        # (b) Site random intercept (regularized)
        self.site_intercept = nn.Embedding(num_sites, output_dim)
        # Projection: concat → d
        self.projection = nn.Linear(output_dim * 2, output_dim)

    def forward(self, site_features, site_id, use_intercept=True):
        h_obs = self.obs_encoder(site_features)
        if use_intercept:
            h_rand = self.site_intercept(site_id)
        else:
            h_rand = torch.zeros_like(h_obs)       # Zero-out cho unseen sites
        return self.projection(torch.cat([h_obs, h_rand], dim=-1))
```

---

## Phase 3: Cross-Level Interaction *(MỚI trong v2)*

**File:** `src/interaction.py`

| # | Step | Verify |
|---|------|--------|
| 3.1 | Implement `CrossLevelInteraction` module | Output: `(batch, d_inter)` |
| 3.2 | Test: random h_ind, h_ctx → h_inter shape đúng | Pass |

### Công thức

```text
h_ind = concat(h_role, h_trait) ∈ R^(2d)
h_ctx = concat(h_road, h_env, h_site) ∈ R^(3d)
h_inter = ReLU(W1·h_ind_proj + W2·h_ctx_proj + W3·(h_ind_proj * h_ctx_proj) + b) ∈ R^d_inter
```

```python
class CrossLevelInteraction(nn.Module):
    """Học tương tác giữa Individual level và Contextual level."""

    def __init__(self, ind_dim, ctx_dim, output_dim):
        super().__init__()
        self.proj_ind = nn.Linear(ind_dim, output_dim)
        self.proj_ctx = nn.Linear(ctx_dim, output_dim)
        self.W_inter = nn.Linear(output_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, h_ind, h_ctx):
        p_ind = self.proj_ind(h_ind)              # (batch, d_inter)
        p_ctx = self.proj_ctx(h_ctx)              # (batch, d_inter)
        interaction = p_ind * p_ctx               # Element-wise product
        h_inter = self.activation(p_ind + p_ctx + self.W_inter(interaction))
        return h_inter
```

---

## Phase 4: Regularized Gated Fusion + Residual *(Nâng cấp v2)*

**File:** `src/fusion.py`

| # | Step | Verify |
|---|------|--------|
| 4.1 | Implement `RegularizedTaskGate` (temperature + prior) | `α` sum ≈ 1.0, không collapse |
| 4.2 | Implement `ResidualGatedFusion` (gated + early fusion blend) | Output: `z_t` shape `(batch, d)` |
| 4.3 | Verify alpha blend parameter is learnable | `alpha` in `model.parameters()` |
| 4.4 | Test: temperature annealing schedule | Temperature giảm qua epochs |

### Công thức

```text
H_full = {h_role, h_trait, h_road, h_env, h_site, h_inter}  (6 inputs)

Với mỗi task t:
  logits_t = W_gate_t · concat(H_full)
  α_raw_t = softmax(logits_t / temperature)
  α_t = (α_raw_t + λ · uniform_prior) / (1 + λ)
  z_t_gated = Σ_v α_t_v · h_v

  z_t_early = MLP_early(concat(h_role, h_trait, h_road, h_env, h_site))

  z_t = alpha · z_t_gated + (1 - alpha) · z_t_early     # alpha: learnable
```

```python
class RegularizedTaskGate(nn.Module):
    def __init__(self, num_inputs, input_dim, temperature_init=2.0):
        super().__init__()
        self.gate = nn.Linear(num_inputs * input_dim, num_inputs)
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.register_buffer('prior', torch.ones(num_inputs) / num_inputs)

    def forward(self, view_reps):
        h_concat = torch.cat(view_reps, dim=-1)
        logits = self.gate(h_concat)
        alpha = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        alpha = (alpha + cfg.GATE_PRIOR_WEIGHT * self.prior) / (1.0 + cfg.GATE_PRIOR_WEIGHT)
        return alpha


class ResidualGatedFusion(nn.Module):
    def __init__(self, num_tasks, num_gate_inputs, gate_input_dim, view_dim, total_view_dim):
        super().__init__()
        self.gates = nn.ModuleList([
            RegularizedTaskGate(num_gate_inputs, gate_input_dim)
            for _ in range(num_tasks)
        ])
        self.early_mlp = nn.Sequential(
            nn.Linear(total_view_dim, view_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(view_dim, view_dim),
        )
        # Learnable blend: alpha=1 → pure gated, alpha=0 → pure early fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, view_reps_for_gate, view_reps_5, h_matrix):
        early_input = torch.cat(view_reps_5, dim=-1)
        z_early = self.early_mlp(early_input)
        alpha = torch.sigmoid(self.alpha)

        fused, alphas = [], []
        for gate in self.gates:
            gate_weights = gate(view_reps_for_gate)
            z_gated = (h_matrix * gate_weights.unsqueeze(-1)).sum(dim=1)
            z = alpha * z_gated + (1 - alpha) * z_early
            fused.append(z)
            alphas.append(gate_weights)
        return fused, alphas, alpha
```

---

## Phase 5: Uncertainty-Weighted Loss *(Nâng cấp v2)*

**File:** `src/loss.py`

| # | Step | Verify |
|---|------|--------|
| 5.1 | Implement `UncertaintyWeightedLoss` | Loss giảm qua epochs |
| 5.2 | Verify σ_k parameters are learnable | `log_sigma` in `model.parameters()` |
| 5.3 | Optional: Focal Loss bên trong mỗi L_k | Toggle qua config |

### Công thức (Kendall et al., 2018)

```text
L_total = Σ_k [ (1 / 2σ_k²) · L_k + log(σ_k) ]
```

```python
class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic uncertainty weighting cho multi-task learning."""

    def __init__(self, num_tasks, use_focal=False):
        super().__init__()
        # log(σ²) thay vì σ trực tiếp, để đảm bảo stability
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks))
        self.use_focal = use_focal

    def forward(self, predictions, targets):
        total_loss = 0
        losses = {}
        for i, task_name in enumerate(TASK_NAMES):
            if self.use_focal:
                loss_i = focal_loss(predictions[i], targets[:, i])
            else:
                loss_i = F.binary_cross_entropy(predictions[i], targets[:, i].float())

            precision = torch.exp(-self.log_sigma_sq[i])    # 1/σ²
            total_loss += 0.5 * precision * loss_i + 0.5 * self.log_sigma_sq[i]
            losses[task_name] = loss_i.item()

        # Log σ values for interpretability
        losses["sigma_values"] = torch.exp(0.5 * self.log_sigma_sq).detach().tolist()
        return total_loss, losses
```

> [!IMPORTANT]
> Sau training, giá trị `σ_k` phản ánh độ khó dự báo của từng behavior — đây là kết quả nghiên cứu có giá trị tự thân.

---

## Phase 6: Full Model Assembly

**File:** `src/model.py`

| # | Step | Verify |
|---|------|--------|
| 6.1 | Assemble `TGMVMTGFNetV2` | Forward pass OK |
| 6.2 | Forward trả về `predictions, gate_weights, alpha_blend` | Shapes đúng |
| 6.3 | Count parameters | < 50K params |

```text
TGMVMTGFNetV2:
  Individual Encoders: RiderRoleEncoder, RiderTraitsEncoder
  Contextual Encoders: RoadContextEncoder, EnvironmentEncoder, SiteAwareEncoder
  CrossLevelInteraction(h_ind, h_ctx) → h_inter
  ResidualGatedFusion({h_views + h_inter}) → {z_t}, {α_t}, alpha_blend
  OutputHeads(z_t) → {ŷ_t}
```

---

## Phase 7: Training & Validation

**File:** `train.py`

| # | Step | Verify |
|---|------|--------|
| 7.1 | Training loop với Adam | Loss giảm |
| 7.2 | Validation loop | Val metrics mỗi epoch |
| 7.3 | Early Stopping (patience=20) | Dừng đúng lúc |
| 7.4 | **[MỚI]** Temperature annealing schedule | Temp giảm từ 2.0 → 1.0 |
| 7.5 | **[MỚI]** Log σ_k values mỗi epoch | Giá trị hội tụ |
| 7.6 | **[MỚI]** Log alpha blend value | alpha ∈ (0, 1) |
| 7.7 | Save best checkpoint | File `.pt` |
| 7.8 | Training curves plot | Convergence rõ |
| 7.9 | Stratified K-Fold CV | Mean ± std |
| 7.10 | Leave-Intersection-Out CV (site intercept OFF) | Metrics trên unseen sites |

---

## Phase 8: Baselines *(Mở rộng v2)*

| # | Baseline | File | Ghi chú |
|---|----------|------|---------|
| 8.1 | Decision Tree | `decision_tree.py` | Baseline gốc paper |
| 8.2 | Logistic Regression | `logistic_regression.py` | Linear baseline |
| 8.3 | Random Forest | `random_forest.py` | Tree ensemble |
| 8.4 | XGBoost / LightGBM | `xgboost_baseline.py` | Tabular ML mạnh nhất |
| 8.5 | Early-Fusion MLP | `early_fusion_mlp.py` | Concat → MLP |
| 8.6 | Late Fusion | `late_fusion.py` | Per-view models |
| 8.7 | **[MỚI] Single-Task MLP** | `single_task_mlp.py` | 4 models riêng → MTL Transfer Ratio |

### Reporting Table

```text
Model                   | Red-light  | No Signal  | Helmet    | Phone     | Macro Avg | MTL Ratio
                        | AUC / F1   | AUC / F1   | AUC / F1  | AUC / F1  | AUC / F1  |
------------------------|------------|------------|-----------|-----------|-----------|----------
Decision Tree           | .6758/.0000| .6215/.0556| .7729/.0000| .7576/.0327| .7069/.0221| N/A
Logistic Regression     | .6373/.0000| .6222/.0000| .6431/.0000| .6818/.0000| .6461/.0000| N/A
Random Forest           | .6306/.0711| .5707/.2831| .7436/.0814| .6980/.2121| .6607/.1619| N/A
XGBoost                 | .6402/.0944| .5697/.2618| .7532/.1256| .6996/.2335| .6657/.1788| N/A
LightGBM                | .6758/.0000| .6138/.0754| .7688/.0000| .7401/.0327| .6996/.0270| N/A
Early-Fusion MLP        | .6636/.0000| .6219/.0103| .7856/.0000| .7490/.0000| .7050/.0026| N/A
Late Fusion             | not run    | not run    | not run   | not run   | not run   | N/A
Single-Task MLP         | .6545/.0104| .6131/.1310| .7959/.0000| .7406/.0110| .7010/.0381| 1.00 (ref)
**M2G-Net v2**    | .6748/.0000| .6165/.0000| .7808/.0000| .7526/.0111| .7062/.0028| 1.01 macro
```

Current numbers are from a single synthetic random-split run using
`python baselines/run_all_baselines.py` plus evaluation of
`checkpoints/best_model.pt` for M2G-Net v2. They are proof-of-concept
results only; paper-ready reporting still needs repeated seeds, tuned
baselines, confidence intervals, and significance tests.

Per-task MTL transfer ratios for M2G-Net v2 versus Single-Task MLP:

```text
Task                  MTL / Single-task AUC
Red-light             1.0310
No signal             1.0055
Helmet                0.9810
Phone                 1.0162
Macro                 1.0074
```

### MTL Transfer Ratio (MỚI v2)

```text
MTL_Transfer_Ratio_k = MTL_AUC_k / Single_Task_AUC_k
  > 1: Multi-task giúp task k
  < 1: Negative transfer → báo cáo, không che giấu
```

---

## Phase 9: Ablation Study *(Mở rộng v2)*

### 9A. Ablation theo View

| # | Experiment | Verify |
|---|-----------|--------|
| 9.1 | Full model (all views) | Baseline |
| 9.2 | − Rider Role View | Drop? |
| 9.3 | − Rider Traits View | Drop? |
| 9.4 | − Road/Intersection View | Drop? |
| 9.5 | − Environmental View | Drop? |
| 9.6 | − Site Context View | Drop? |

### 9B. Ablation theo Fusion

| # | Experiment | Verify |
|---|-----------|--------|
| 9.7 | Early fusion only (alpha=0) | vs gated |
| 9.8 | Late fusion only | vs gated |
| 9.9 | Shared gate (1 gate cho mọi task) | vs task-specific |
| 9.10 | Task-specific gates (no residual) | vs full |
| 9.11 | Task-specific gates + residual (full v2) | Best? |

### 9C. Ablation theo Multi-Task

| # | Experiment | Verify |
|---|-----------|--------|
| 9.12 | Single-task models | vs MTL |
| 9.13 | MTL without gates | vs MTL+gates |
| 9.14 | MTL + gates (fixed λ weights) | vs uncertainty |
| 9.15 | MTL + gates + uncertainty weighting (full v2) | Best? |

### 9D. Ablation theo Cross-Level Interaction *(MỚI v2)*

| # | Experiment | Verify |
|---|-----------|--------|
| 9.16 | Full model | Baseline |
| 9.17 | − Cross-level interaction (views độc lập) | Drop? |

### 9E. Ablation theo Site Encoding *(MỚI v2)*

| # | Experiment | Verify |
|---|-----------|--------|
| 9.18 | Full site (obs + random intercept, in-distribution) | Baseline |
| 9.19 | Site features only (out-of-distribution) | Drop bao nhiêu? |
| 9.20 | No site view | Drop? |

---

## Phase 10: Interpretability *(Nâng cấp v2)*

**File:** `interpret.py`

| # | Step | Output |
|---|------|--------|
| 10.1 | Extract gate weights trung bình trên test set | Bar chart per task |
| 10.2 | **[MỚI]** Integrated Gradients (dùng `captum`) | Population-level attribution per view |
| 10.3 | **[MỚI]** So sánh Gate weights vs IG vs Ablation (Triple-validation) | Table 3 methods side-by-side |
| 10.4 | Report learned σ_k values | Bảng: task nào khó nhất? |
| 10.5 | Report learned alpha blend | Gated fusion quan trọng bao nhiêu? |
| 10.6 | Risk profile visualization | Heatmap |
| 10.7 | SHAP cho XGBoost baseline | Cross-check |

### Triple-Validation Table (MỚI v2, current run)

```text
View              | Gate Weight | Integrated Gradients | Ablation Drop | Consistent?
------------------|-------------|----------------------|---------------|------------
Rider Role        | 26.69%      | 26.33%               | +0.1419 AUC   | Yes
Road Context      | 22.71%      | 24.84%               | +0.0390 AUC   | Yes
Environment       | 17.88%      | 24.32%               | +0.0277 AUC   | Partial
Site              | 6.88%       | 10.85%               | -0.0031 AUC   | No
Rider Traits      | 6.68%       | 8.01%                | +0.0011 AUC   | No
Cross-Level Inter | 19.16%      | 5.64%                | +0.0442 AUC   | Yes
```

Current triple-validation values are macro averages across the four risky
behavior tasks on the synthetic random-split test set. Full-model macro
ROC-AUC is `0.7062`; `Ablation Drop = full model AUC - ablated AUC`, so a
positive value means the model gets worse when that component is removed.
Integrated Gradients is approximated with normalized gradient x activation at
the view representation level. `Consistent?` is marked `Yes` when at least two
of the three signals rank the component in the top three.

---

## Checklist tổng thể

```text
Legend: [x] done, [~] partial / proof-of-concept, [ ] not done

Phase 0: Setup
[x] Cấu trúc thư mục
[x] requirements.txt (+ captum)
[x] config.py (+ hierarchical, interaction, uncertainty params)

Phase 1: Data Pipeline
[x] Load & clean data
[x] View-column mapping (individual vs contextual levels)
[x] Site view tách observed + site_id
[x] Train/Val/Test split (stratified)
[x] Leave-Intersection-Out split
[x] PyTorch Dataset & DataLoader

Phase 2: View Encoders
[x] ViewEncoder base + per-view encoders
[x] SiteAwareEncoder (observed + random intercept)
[~] Unit test all encoders

Phase 3: Cross-Level Interaction (MỚI)
[x] CrossLevelInteraction module
[~] Test h_ind × h_ctx → h_inter

Phase 4: Regularized Gated Fusion + Residual (MỚI)
[x] RegularizedTaskGate (temperature + prior)
[x] ResidualGatedFusion (gated + early blend)
[x] Learnable alpha parameter

Phase 5: Uncertainty-Weighted Loss (MỚI)
[x] UncertaintyWeightedLoss (learnable σ_k)
[x] Optional focal loss

Phase 6: Full Model
[x] TGMVMTGFNetV2 assembly
[x] Parameter count < 50K

Phase 7: Training
[x] Training loop + validation
[x] Early stopping
[x] Temperature annealing
[x] Log σ_k + alpha mỗi epoch
[ ] K-Fold CV
[x] Leave-Intersection-Out CV (intercept OFF)

Phase 8: Baselines
[x] Decision Tree
[x] Logistic Regression
[x] Random Forest
[x] XGBoost / LightGBM
[x] Early-Fusion MLP
[ ] Late Fusion
[x] Single-Task MLP (MỚI) → MTL Transfer Ratio

Phase 9: Ablation
[x] View ablation (5 experiments)
[ ] Fusion ablation (5 experiments)
[~] Multi-task ablation (4 experiments)
[~] Cross-level interaction ablation (MỚI)
[x] Site encoding ablation (MỚI)

Phase 10: Interpretability
[x] Gate weights visualization
[~] Integrated Gradients (MỚI)
[x] Triple-validation: Gate vs IG vs Ablation (MỚI)
[x] σ_k analysis (MỚI)
[x] Alpha blend analysis (MỚI)
[x] Risk profile heatmap
[ ] SHAP cross-check
```

---

## Nguyên tắc code (từ CLAUDE.md)

> [!CAUTION]
> 1. **Think Before Coding** — Hỏi trước nếu mapping view-column chưa chắc chắn.
> 2. **Simplicity First** — Encoder nhỏ (2 layers), cross-level interaction dùng element-wise product, không cần attention.
> 3. **Surgical Changes** — Khi thay config, chỉ sửa config.
> 4. **Goal-Driven** — Mỗi phase có verify criteria. Không qua phase mới nếu chưa verify xong.
# Archived Draft Notice

This file is an older planning draft and may contain stale notation or
hyperparameters. Use `docs/MATHEMATICAL_FORMULAS.md`,
`docs/ARCHITECTURE_VISUALIZATION.md`, `config.py`, and the `src/` code as the
current source of truth.
