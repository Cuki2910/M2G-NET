        # M2G-Net v2: Theory-Guided Multi-View Multi-Task Gated Fusion Network

**Mục đích tài liệu:** Trình bày ý tưởng mô hình tối ưu (phiên bản cải thiện) để xin phép thử nghiệm trên dataset quan sát hành vi người đi xe máy trong bài *Not the same: How delivery, ride-hailing, and private riders' roles influence safety behavior*.

**Tên mô hình đề xuất:** **M2G-Net v2**

**Tên đầy đủ:** **Theory-Guided Multi-View Multi-Task Gated Fusion Network (Version 2)**

**Bài toán:** Dự báo và diễn giải các hành vi lái xe rủi ro của người đi xe máy tại giao lộ bằng cách chia dữ liệu quan sát thành các nhóm thông tin có ý nghĩa lý thuyết, tổ chức các nhóm thành hai tầng individual và contextual, học cross-level interaction, rồi kết hợp bằng cơ chế regularized gated fusion với residual baseline và uncertainty-weighted multi-task loss.

---

## 1. Executive Summary

Mô hình đề xuất không nhằm thay thế hoàn toàn phương pháp decision tree trong paper gốc. Thay vào đó, nó được thiết kế như một **methodological extension có kiểm soát**.

Paper gốc sử dụng decision tree để xác định các nhánh điều kiện phân tách hành vi rủi ro giữa các nhóm người lái như **food delivery riders**, **ride-hailing riders** và **private riders**. Decision tree rất mạnh ở khả năng diễn giải vì nó tạo ra các rule trực quan dạng:

```text
Nếu rider type = food delivery
và time slot = noon
và traffic = dense
→ xác suất hành vi rủi ro cao hơn
```

Tuy nhiên, decision tree có đặc điểm là chọn split theo từng node cục bộ. Điều này tạo ra cấu trúc cây không đối xứng: một nhánh có thể tiếp tục chia theo **weather**, nhánh khác lại chia theo **riding opposite way**, nhánh khác nữa chia theo **traffic**. Đây không phải lỗi của decision tree, nhưng nó khiến việc so sánh một cách hệ thống vai trò tương đối của các nhóm thông tin trở nên khó hơn.

Mô hình **M2G-Net v2** được đề xuất để kiểm tra câu hỏi sau:

> Nếu ta tổ chức dataset thành các "view" có ý nghĩa lý thuyết — rider role, rider traits, road/intersection context, environmental context và site context — phân tầng theo individual và contextual levels, học cross-level interaction, rồi fusion bằng regularized task-specific gated fusion với uncertainty-weighted loss, thì mô hình này có cải thiện khả năng dự báo và diễn giải các hành vi rủi ro so với decision tree và các baseline machine learning truyền thống hay không?

Điểm cốt lõi của mô hình là:

```text
Theory-guided views (phân tầng individual vs. contextual)
→ View-specific encoders
→ Cross-level interaction
→ Regularized task-specific gated fusion + residual early fusion
→ Uncertainty-weighted multi-task prediction
→ Risk profile + Integrated Gradients view attribution
```

---

## 2. Motivation: Vì sao cần một mô hình mới?

### 2.1. Decision tree trong paper gốc làm rất tốt điều gì?

Decision tree rất phù hợp cho nghiên cứu traffic safety vì nó có khả năng tạo ra rule dễ hiểu. Nó trả lời tốt các câu hỏi như:

- Nhóm rider nào có rủi ro cao hơn?
- Trong điều kiện nào thì rủi ro tăng?
- Biến nào là split quan trọng ở từng node?
- Có những tổ hợp điều kiện nào tạo ra nhóm rủi ro cao?

Ví dụ logic của decision tree:

```text
Food delivery rider
│
├── Morning / Evening
│   ├── Không đi ngược chiều → risk thấp hơn
│   └── Có đi ngược chiều → risk cao hơn
│
└── Noon
    ├── Traffic dense → risk cao nhất
    └── Traffic normal → risk trung bình
```

Điểm mạnh của decision tree là nó trực quan và phù hợp với policy discussion.

### 2.2. Decision tree có giới hạn gì?

Các giới hạn chính không phải vì decision tree "sai", mà vì bản chất của nó là **path-dependent local splitting**.

#### Giới hạn 1: Cấu trúc nhánh không đồng bộ

Sau khi chia theo rider type, decision tree có thể chọn biến tiếp theo khác nhau cho từng nhánh.

Ví dụ:

```text
Food Delivery → Weather
Ride-hailing  → Riding opposite way
Private       → Weather
```

Điều này hữu ích để phát hiện pattern cục bộ, nhưng gây khó khăn khi muốn so sánh một cách hệ thống:

```text
Weather đóng góp bao nhiêu so với traffic?
Road context quan trọng hơn rider role ở task nào?
Intersection context có còn quan trọng sau khi kiểm soát rider type không?
```

#### Giới hạn 2: Split cứng

Decision tree phân tích theo nhánh:

```text
Nếu A → đi nhánh trái
Nếu không A → đi nhánh phải
```

Trong thực tế, rủi ro có thể được hình thành bởi nhiều nhóm yếu tố cùng lúc, với mức đóng góp mềm hơn:

```text
Rider role: 35%
Road context: 25%
Environment: 15%
Intersection: 20%
Rider traits: 5%
```

#### Giới hạn 3: Khó học representation theo nhóm biến

Decision tree xử lý tất cả feature như một danh sách biến phẳng. Nó không biết rằng các biến có thể thuộc các nhóm lý thuyết khác nhau và có cấu trúc nhân quả phân tầng:

```text
Individual level:   Rider role view, Rider traits view
Contextual level:   Road/intersection context view, Environmental context view, Site context view
```

#### Giới hạn 4: Thường phân tích từng outcome riêng biệt

Nếu mỗi hành vi rủi ro được phân tích bằng một cây riêng, model không tận dụng được khả năng các hành vi này có pattern chung.

Ví dụ:

- Red-light running
- No turn signal
- Helmet non-use
- Mobile phone use

Các hành vi này có thể không độc lập hoàn toàn. Một số nhóm thông tin như rider role hoặc intersection context có thể ảnh hưởng đến nhiều hành vi cùng lúc.

---

## 3. Research Objective

### 3.1. Mục tiêu chính

Đề xuất và kiểm tra một mô hình **theory-guided multi-view multi-task gated fusion** có phân tầng individual–contextual, cross-level interaction, regularized gated fusion và uncertainty-weighted multi-task loss để dự báo nhiều hành vi lái xe rủi ro của người đi xe máy tại giao lộ.

### 3.2. Research Question đề xuất

> **Can a theory-guided multi-view multi-task gated fusion model — with hierarchical individual–contextual encoding, cross-level interaction, and uncertainty-weighted multi-task loss — improve the prediction and interpretation of motorcycle risky riding behaviors compared with decision tree and conventional machine learning baselines?**

Bản tiếng Việt:

> **Liệu mô hình multi-view multi-task gated fusion theo định hướng lý thuyết — có phân tầng individual–contextual, cross-level interaction, và uncertainty-weighted loss — có cải thiện khả năng dự báo và diễn giải các hành vi lái xe rủi ro của người đi xe máy so với decision tree và các baseline machine learning truyền thống hay không?**

### 3.3. Câu hỏi phụ

1. View nào đóng góp nhiều nhất vào từng hành vi rủi ro (theo gate weights và Integrated Gradients)?
2. Rider role còn quan trọng đến mức nào sau khi kiểm soát road context, environment và intersection context?
3. Cross-level interaction giữa individual và contextual views có cải thiện khả năng dự báo so với encoding độc lập không?
4. Task-specific gates có giúp mô hình học cơ chế khác nhau cho từng hành vi rủi ro không?
5. Multi-task learning có giúp dự báo tốt hơn so với single-task models không, hay có negative transfer ở một số task?
6. Site context có cải thiện prediction trong random split, và liệu tác động đó có giữ được trong leave-intersection-out validation không?

---

## 4. Định vị phương pháp: Vì sao gọi là multi-view, không phải multimodal?

Dataset này chủ yếu là dữ liệu quan sát dạng tabular/categorical. Vì vậy, không nên gọi đây là **multimodal fusion** theo nghĩa mạnh như GPS + text + image + sensor.

Cách gọi chính xác hơn là:

> **Theory-guided multi-view fusion**

Lý do:

- Dữ liệu không có nhiều modality vật lý khác nhau.
- Nhưng các biến có thể được chia thành nhiều nhóm thông tin có ý nghĩa lý thuyết.
- Mỗi nhóm phản ánh một cơ chế rủi ro khác nhau.
- Các nhóm này có cấu trúc nhân quả phân tầng: individual-level features và contextual-level features không đồng đẳng nhau.

Ví dụ:

```text
Rider role view
→ phản ánh vai trò xã hội/nghề nghiệp của người lái
→ thuộc Individual Level

Rider traits view
→ phản ánh đặc điểm cá nhân quan sát được
→ thuộc Individual Level

Road/intersection context view
→ phản ánh môi trường giao thông trực tiếp
→ thuộc Contextual Level

Environmental context view
→ phản ánh điều kiện bên ngoài như thời tiết/thời điểm
→ thuộc Contextual Level

Site context view
→ phản ánh đặc điểm cơ sở hạ tầng của từng giao lộ
→ thuộc Contextual Level
```

Nói cách khác:

> Model không giả định dataset là multimodal mạnh. Model chỉ tận dụng cấu trúc multi-view có sẵn và cấu trúc nhân quả phân tầng trong dữ liệu quan sát.

---

## 5. Nền tảng lý thuyết: Vì sao gọi là "Theory-Guided"?

Chữ "Theory-Guided" trong tên mô hình không chỉ là trang trí. Mỗi view được neo vào một framework lý thuyết cụ thể. Điều này biến view decomposition từ việc "chia biến theo nhóm hợp lý" thành một **claim có thể kiểm chứng**.

| View | Theory Framework | Lý do liên kết |
|---|---|---|
| Rider Role | Occupational Risk Theory; Role-Behavior Congruence | Rider role phản ánh động cơ nghề nghiệp, áp lực thời gian và exposure đến rủi ro do vai trò quy định |
| Rider Traits | Theory of Planned Behavior — attitude, subjective norm, perceived behavioral control | Đặc điểm cá nhân quan sát được (giới tính, nhóm tuổi) có thể là proxy cho attitude và norm |
| Road/Intersection Context | Situational Crime Prevention; Environmental Criminology | Môi trường vật lý hình thành cơ hội và rào cản cho hành vi rủi ro |
| Environmental Context | Risk Compensation Theory (Wilde, 1982) | Điều kiện thời tiết/thời điểm ảnh hưởng đến perceived risk, qua đó điều chỉnh hành vi |
| Site Context | Ecological Systems Theory (Bronfenbrenner) | Giao lộ là một microsystem có baseline risk riêng do thiết kế và lịch sử enforcement |

**Ý nghĩa cho interpretation:**

Nếu Integrated Gradients cho thấy Road/Intersection Context đóng góp cao nhất cho red-light running, kết quả này consistent với Situational Crime Prevention. Nếu Rider Role đóng góp cao nhất cho helmet non-use, kết quả này gợi ý Occupational Risk Theory. Bằng cách neo từng view vào lý thuyết, bài toán trở thành kiểm tra empirical của các lý thuyết đó, không chỉ là so sánh model.

---

## 6. Mô hình đề xuất: M2G-Net v2

### 6.1. Tên mô hình

**M2G-Net v2**

Trong đó:

```text
TG    = Theory-Guided
MV    = Multi-View
MT    = Multi-Task
GFNet = Gated Fusion Network
v2    = phiên bản có phân tầng individual–contextual, cross-level interaction,
        regularized gates, residual early fusion, uncertainty-weighted loss,
        và Integrated Gradients attribution
```

### 6.2. Mô tả một câu

> **M2G-Net v2 tổ chức dữ liệu quan sát thành hai tầng lý thuyết — individual views (rider role, rider traits) và contextual views (road/intersection context, environmental context, site context) — học cross-level interaction giữa hai tầng, dùng regularized task-specific gated fusion có residual early fusion để dự báo nhiều hành vi lái xe rủi ro với uncertainty-weighted loss, và phân tích view contribution bằng gate weights kết hợp Integrated Gradients.**

### 6.3. Kiến trúc tổng quát

```text
Input
  ↓
Individual Views          Contextual Views
[Role] [Traits]     [Road] [Env] [Site*]
     ↓                        ↓
  Encoders                Encoders
     └──────── Cross-Level Interaction ─────────┘
                          ↓
       Regularized Task-Specific Gated Fusion
       + Residual từ Early Fusion (alpha blend)
                          ↓
     Uncertainty-Weighted Multi-Task Output
                          ↓
       Gate Weights + Integrated Gradients
              (View Contribution Report)
```

*Site context: observed infrastructure features + site random intercept (không phải pure learned embedding)

Chi tiết luồng:

```text
Input observation: một rider tại một giao lộ

Level 1 — Individual Encoders:
    View 1: Rider Role Encoder      → h_role
    View 2: Rider Traits Encoder    → h_trait
    h_ind = concat(h_role, h_trait)

Level 2 — Contextual Encoders:
    View 3: Road/Intersection Encoder → h_road
    View 4: Environmental Encoder     → h_env
    View 5: Site Context Encoder*     → h_site
    h_ctx = concat(h_road, h_env, h_site)

Level 3 — Cross-Level Interaction:
    h_inter = CrossLevelInteraction(h_ind, h_ctx)

Level 4 — Regularized Task-Specific Gated Fusion:
    For each task t:
        α_raw_t = softmax(Gate_t([h_role, h_trait, h_road, h_env, h_site, h_inter]) / temperature)
        α_t = (α_raw_t + lambda * uniform_prior) / (1 + lambda)
        z_t_gated = Σ_v α_t_v · h_v
        z_t_early = MLP(concat(all views))
        z_t = alpha · z_t_gated + (1 - alpha) · z_t_early

Level 5 — Uncertainty-Weighted Multi-Task Output:
    ŷ_t = Head_t(z_t)
    L_total = Σ_t [(1 / 2σ_t²) · L_t + log(σ_t)]
    σ_t: learnable per-task uncertainty parameter
```

---

## 7. Input Views

### 7.1. View 1: Rider Role View

#### Mục đích

Biểu diễn vai trò nghề nghiệp và xã hội của người lái.

#### Ví dụ biến

```text
- Food delivery rider
- Ride-hailing rider
- Private rider
```

#### Nền tảng lý thuyết

**Occupational Risk Theory** và **Role-Behavior Congruence**: rider role quy định động cơ di chuyển, áp lực thời gian và mức độ tiếp xúc với rủi ro.

- Food delivery riders: áp lực giao hàng nhanh, tần suất tiếp xúc cao.
- Ride-hailing riders: áp lực nhận khách/chuyến, thời gian chờ cạnh tranh.
- Private riders: không bị ràng buộc bởi platform, có thể có pattern hành vi khác.

#### Encoder đề xuất

```text
One-hot / embedding layer
→ Dense layer (d_role units)
→ ReLU → Dropout
→ h_role ∈ R^d
```

**Tầng phân loại: Individual Level**

---

### 7.2. View 2: Rider Traits View

#### Mục đích

Biểu diễn đặc điểm cá nhân quan sát được.

#### Ví dụ biến

```text
- Gender
- Age group
```

#### Nền tảng lý thuyết

**Theory of Planned Behavior (TPB)**: gender và age group có thể là proxy quan sát được cho attitude và subjective norm liên quan đến hành vi giao thông. Cần diễn giải cẩn thận — đây là biến quan sát, không phải giải thích nhân quả trực tiếp.

#### Encoder đề xuất

```text
Categorical variables → embedding / one-hot
→ Dense layer (d_trait units)
→ ReLU → Dropout
→ h_trait ∈ R^d
```

**Tầng phân loại: Individual Level**

---

### 7.3. View 3: Road & Intersection Context View

#### Mục đích

Biểu diễn điều kiện đường và giao lộ tại thời điểm quan sát.

#### Ví dụ biến

```text
- Police presence
- Number of lanes
- Traffic condition
- Road/intersection characteristics
- Signal context, nếu có
```

#### Nền tảng lý thuyết

**Situational Crime Prevention** và **Environmental Criminology**: môi trường vật lý hình thành cơ hội và rào cản hành vi. Police presence tương đương với guardianship; lane complexity tương đương với perceived monitoring difficulty.

Ví dụ:

```text
Không có cảnh sát + giao lộ nhiều làn + traffic dense
→ có thể làm tăng một số hành vi rủi ro
```

#### Encoder đề xuất

```text
Mixed categorical/numeric features → preprocessing
→ Dense layer (d_road units)
→ ReLU → Dropout
→ h_road ∈ R^d
```

**Tầng phân loại: Contextual Level**

---

### 7.4. View 4: Environmental Context View

#### Mục đích

Biểu diễn điều kiện môi trường và thời gian.

#### Ví dụ biến

```text
- Weather
- Time slot (Morning / Noon / Evening)
- Weekday / weekend, nếu có
```

#### Nền tảng lý thuyết

**Risk Compensation Theory (Wilde, 1982)**: điều kiện weather và time slot ảnh hưởng đến *perceived risk*. Khi perceived risk thay đổi, người lái có thể điều chỉnh hành vi để duy trì một mức rủi ro chủ quan "chấp nhận được" — tức là khi trời mưa, người lái có thể lái cẩn thận hơn; khi đường vắng lúc sáng sớm, họ có thể liều hơn.

#### Encoder đề xuất

```text
Categorical/time variables → embedding / one-hot
→ Dense layer (d_env units)
→ ReLU → Dropout
→ h_env ∈ R^d
```

**Tầng phân loại: Contextual Level**

---

### 7.5. View 5: Site Context View *(Cải thiện so với v1)*

#### Mục đích

Biểu diễn đặc điểm cơ sở hạ tầng và baseline risk của từng giao lộ.

#### Nền tảng lý thuyết

**Ecological Systems Theory (Bronfenbrenner)**: giao lộ là một microsystem với lịch sử enforcement, thiết kế vật lý và thói quen di chuyển riêng, tạo ra baseline risk khác nhau.

#### Vì sao không dùng learned embedding thuần?

Với ~4–10 giao lộ trong dataset, một pure learned embedding sẽ:

- Rất dễ overfit do số lượng sites quá ít.
- Hoàn toàn vô dụng trong leave-intersection-out validation, vì model chưa từng thấy site mới.

#### Giải pháp đề xuất: Site-Aware Encoding

Dùng hai thành phần song song:

```text
Site context view
├── (a) Observed site features: số làn, loại giao lộ, có đèn tín hiệu, loại đường, v.v.
│       → Encoder nhỏ → h_site_obs ∈ R^d
│
└── (b) Site random intercept (regularized):
        → Được học trong training, được zero-out hoặc mean-pooled tại inference trên site mới
        → Tương đương với mixed-effects modeling, phù hợp precedent trong traffic safety literature
        → h_site_rand ∈ R^d
        → Chỉ dùng trong in-distribution evaluation

h_site = concat(h_site_obs, h_site_rand)
       → Linear projection → R^d
```

#### Lưu ý báo cáo

Trong ablation và leave-intersection-out evaluation, cần báo cáo riêng hai kịch bản:

```text
Kịch bản 1: h_site = h_site_obs + h_site_rand (in-distribution)
Kịch bản 2: h_site = h_site_obs only (out-of-distribution / leave-intersection-out)
```

Điều này giúp tách biệt rõ: site information nào generalize được và phần nào là site-specific memorization.

**Tầng phân loại: Contextual Level**

---

## 8. View-Specific Encoding

### 8.1. Ký hiệu

```text
x_role   = features của Rider Role View
x_trait  = features của Rider Traits View
x_road   = features của Road/Intersection Context View
x_env    = features của Environmental Context View
x_site   = features của Site Context View (observed + random intercept)
```

Mỗi encoder tạo ra một representation:

```text
h_role  = Encoder_role(x_role)
h_trait = Encoder_trait(x_trait)
h_road  = Encoder_road(x_road)
h_env   = Encoder_env(x_env)
h_site  = Encoder_site(x_site)
```

### 8.2. Vì sao cần encoder riêng?

Nếu gộp tất cả biến vào một vector duy nhất, model sẽ mất cấu trúc lý thuyết của dataset. Encoder riêng giúp:

- Bảo toàn ý nghĩa của từng nhóm biến.
- Cho phép học representation phù hợp với từng loại thông tin.
- Cho phép đo contribution của từng view qua cả gate weights và Integrated Gradients.
- Tạo nền tảng cho ablation study rõ ràng.

### 8.3. Encoder nên nhỏ

Vì dataset không quá lớn, encoder không nên quá phức tạp.

```text
Embedding / one-hot
→ Dense layer
→ ReLU
→ Dropout
→ Dense layer
→ view representation ∈ R^d
```

Với `d` có thể là 8, 16 hoặc 32 tùy dataset size.

---

## 9. Cross-Level Interaction *(Mới trong v2)*

### 9.1. Lý do cần cross-level interaction

Nhiều lý thuyết hành vi gợi ý rằng **tác động của context phụ thuộc vào đặc điểm cá nhân**. Ví dụ:

- Risk Compensation Theory: food delivery rider (individual) phản ứng khác với rain (contextual) so với private rider.
- Situational Crime Prevention: police presence (contextual) có thể tác động mạnh hơn với ride-hailing rider do đặc điểm công việc.

Nếu chỉ encode từng view riêng rồi concat, model không thể học interaction này một cách có cấu trúc. Một flat gate có thể tình cờ học được một phần, nhưng không đảm bảo.

### 9.2. Tổ chức hai tầng

```text
Individual representation:
h_ind = concat(h_role, h_trait) ∈ R^(2d)

Contextual representation:
h_ctx = concat(h_road, h_env, h_site) ∈ R^(3d)
```

### 9.3. Cross-Level Interaction Module

Dùng một bilinear interaction layer nhỏ:

```text
h_inter = Interaction(h_ind, h_ctx)
        = ReLU(W_inter · (h_ind ⊗ h_ctx) + b_inter)
```

Hoặc đơn giản hơn với dataset nhỏ:

```text
h_inter = ReLU(W_1 · h_ind + W_2 · h_ctx + W_3 · (h_ind * h_ctx) + b)
```

Trong đó `h_ind * h_ctx` là element-wise product sau khi project về cùng chiều.

Output:

```text
h_inter ∈ R^d_inter
```

### 9.4. h_inter được dùng như thế nào?

`h_inter` được concat vào tập view representations trước khi đưa vào gated fusion:

```text
H_full = {h_role, h_trait, h_road, h_env, h_site, h_inter}
```

Gate cho từng task sẽ học trọng số trên toàn bộ H_full, bao gồm cả interaction term.

---

## 10. Regularized Task-Specific Gated Fusion *(Cải thiện so với v1)*

### 10.1. Vì sao cần task-specific gates?

Các hành vi rủi ro khác nhau có thể phụ thuộc vào các nhóm thông tin khác nhau.

```text
Red-light running   → có thể phụ thuộc mạnh vào rider role + traffic + intersection
No turn signal      → có thể phụ thuộc vào rider role + road context
Helmet non-use      → có thể phụ thuộc vào rider role + police presence
Mobile phone use    → có thể phụ thuộc vào rider role + weather + traffic + time slot
```

Một fusion gate chung cho tất cả task sẽ bỏ qua sự khác biệt này.

### 10.2. Regularized Gate

**Vấn đề với gate thuần:** softmax weights không ổn định và nhạy cảm với initialization — có thể collapse vào một view ngay từ đầu (mode collapse), làm các view khác không được học.

**Giải pháp: temperature + sparsemax uniform prior mixing**

```python
# Thay vì:
alpha_t = softmax(W_gate_t @ concat(H_full))

# Dùng:
logits_t = W_gate_t @ concat(H_full)
alpha_raw_t = sparsemax(logits_t / temperature)
alpha_t = (1 - lambda_prior) * alpha_raw_t + lambda_prior * uniform_prior

# uniform_prior = vector đều tạo lower bound nhẹ cho mỗi gate input
# temperature: annealed từ cao xuống thấp trong training
```

### 10.3. Representation fused cho từng task

```text
z_t_gated = Σ_v α_t_v · h_v     (v ∈ H_full)
```

### 10.4. Residual từ Early Fusion *(Mới trong v2)*

Thêm residual connection từ early fusion:

```text
z_t_early = MLP_early(concat(h_role, h_trait, h_road, h_env, h_site))

z_t = alpha · z_t_gated + (1 - alpha) · z_t_early
```

Trong đó `alpha` là một learnable scalar (hoặc task-specific learnable scalar).

**Ý nghĩa cho ablation:**

```text
Nếu alpha → 0: model thoái hóa về early-fusion MLP
Nếu alpha → 1: gated fusion có toàn bộ quyền quyết định
Giá trị alpha học được sau training: thông tin về việc gated fusion có thực sự có ích không
```

### 10.5. Ý nghĩa diễn giải gate weights

Nếu với task mobile phone use, model học được:

```text
Rider role:    0.38
Environment:   0.26
Road context:  0.21
Site:          0.10
Rider traits:  0.05
Interaction:   0.00  (không đáng kể)
```

Ta có thể diễn giải rằng rider role và environmental context đóng góp nhiều hơn rider traits. Tuy nhiên, gate weights là **local attention proxy** — cần kết hợp với Integrated Gradients để có attribution toàn diện hơn (xem Section 14).

---

## 11. Uncertainty-Weighted Multi-Task Output *(Cải thiện so với v1)*

### 11.1. Các task cần dự báo

```text
Task 1: Red-light running
Task 2: No turn signal
Task 3: Helmet non-use / helmet not fastened
Task 4: Mobile phone use
```

### 11.2. Vì sao cần uncertainty weighting?

Với fixed task weights (λ1, λ2, λ3, λ4), một task dễ predict hơn sẽ dominate gradient training, ảnh hưởng tiêu cực đến các task khó hơn.

**Giải pháp: Homoscedastic Uncertainty Weighting** (Kendall et al., 2018):

```text
L_total = Σ_k [ (1 / 2σ_k²) · L_k + log(σ_k) ]
```

Trong đó:

```text
σ_k: learnable uncertainty parameter cho task k
L_k: loss của task k (binary cross-entropy)
```

**Cơ chế hoạt động:**

- Nếu σ_k lớn: task k được coi là khó (cao uncertainty), trọng số giảm tự động.
- Nếu σ_k nhỏ: task k được coi là dễ hơn, trọng số tăng tự động.
- log(σ_k) là regularization để tránh model chọn σ_k → ∞.

**Ý nghĩa diễn giải:** Sau training, giá trị σ_k học được phản ánh độ khó dự báo của từng behavior — là kết quả nghiên cứu có giá trị tự thân.

### 11.3. Output heads

Mỗi task có một output head riêng:

```text
ŷ_red_light = Head_red_light(z_red_light)
ŷ_signal    = Head_signal(z_signal)
ŷ_helmet    = Head_helmet(z_helmet)
ŷ_phone     = Head_phone(z_phone)
```

Mỗi output là xác suất:

```text
P(red-light running)
P(no turn signal)
P(helmet non-use)
P(mobile phone use)
```

### 11.4. Class imbalance

Nếu class imbalance mạnh, thêm focal loss hoặc class-weighted BCE bên trong L_k trước khi uncertainty weighting:

```text
L_k = FocalLoss(ŷ_k, y_k)  hoặc  WeightedBCE(ŷ_k, y_k)
```

---

## 12. Output của mô hình

Mô hình tạo ra 4 tầng output.

### 12.1. Output 1: Xác suất từng hành vi rủi ro

```text
Observation ID: 0001
Red-light running: 0.72
No turn signal: 0.65
Helmet non-use: 0.18
Mobile phone use: 0.31
```

### 12.2. Output 2: Risk profile

```text
Overall risk score = weighted average of task probabilities
Risk level: Medium / High / Low
Dominant risk type: No turn signal + Mobile phone use
```

### 12.3. Output 3: Gate weight contribution

Với mỗi task, báo cáo trọng số gate của từng view:

```text
Task: Red-light running
Rider role: 40%
Road context: 30%
Intersection/site: 20%
Environment: 7%
Rider traits: 3%
```

### 12.4. Output 4: Integrated Gradients attribution *(Mới trong v2)*

Xem Section 14 để biết chi tiết. Attribution theo IG cho phép so sánh population-level contribution của từng view, khác với gate weights là per-sample.

### 12.5. Ý nghĩa policy

Decision tree thường cho rule dạng:

```text
Nếu A và B thì rủi ro cao
```

M2G-Net v2 có thể bổ sung:

```text
A, B, C cùng đóng góp vào rủi ro với mức độ khác nhau.
Với mỗi hành vi rủi ro, nhóm thông tin quan trọng nhất có thể khác nhau.
Individual và contextual views tương tác theo cách mà cross-level interaction nắm bắt.
```

---

## 13. Baselines cần so sánh

### 13.1. Baseline 1: Decision Tree

Baseline gần với paper gốc nhất. Mục tiêu: tái tạo logic cây quyết định để có điểm so sánh trực tiếp.

### 13.2. Baseline 2: Logistic Regression

Baseline tuyến tính, dễ diễn giải. Mục tiêu: kiểm tra liệu mô hình phức tạp có thực sự cần thiết hơn tuyến tính không.

### 13.3. Baseline 3: Random Forest

Ensemble của nhiều decision trees. Mục tiêu: kiểm tra performance của tree-based ensemble ổn định hơn.

### 13.4. Baseline 4: XGBoost / LightGBM

Rất mạnh với dữ liệu tabular. **Baseline quan trọng nhất về predictive performance**. Nếu M2G-Net không vượt được XGBoost về accuracy, cần argue rõ giá trị nằm ở view-level attribution và interpretability, không phải predictive superiority.

### 13.5. Baseline 5: Early-Fusion MLP

```text
All features → MLP → outputs
```

Mục tiêu: kiểm tra encoder riêng + gated fusion có tốt hơn concatenate đơn giản không. Đây cũng là giá trị của alpha trong residual blend: nếu z_t_early (alpha=0) là đủ tốt, không cần gating.

### 13.6. Baseline 6: Late Fusion

```text
View-specific models → combine predictions
```

Mục tiêu: kiểm tra representation-level fusion có tốt hơn prediction-level fusion không.

### 13.7. Single-Task Baselines *(Mới trong v2 — Negative Transfer Detection)*

Train 4 mô hình riêng biệt, mỗi mô hình cho một task:

```text
Single-task model for red-light running
Single-task model for no turn signal
Single-task model for helmet non-use
Single-task model for mobile phone use
```

Mục tiêu: phát hiện negative transfer.

**MTL Transfer Ratio:**

```text
MTL_Transfer_Ratio_k = MTL_AUC_k / Single_Task_AUC_k
```

- Nếu ratio > 1: multi-task learning giúp task k.
- Nếu ratio < 1: **negative transfer** — báo cáo và giải thích, không che giấu.

---

## 14. Interpretability Plan *(Cải thiện so với v1)*

### 14.1. Gate weights (local, per-sample)

Sử dụng gate weights để xem view nào được model chú ý cho từng task và từng observation.

Ví dụ population-level average gate weights:

```text
Red-light running:
Rider role: 35% | Road context: 28% | Site: 22% | Environment: 10% | Rider traits: 5%
```

Lưu ý: gate weights là **attention proxy**, không phải causal attribution.

### 14.2. Integrated Gradients (population-level) *(Mới trong v2)*

**Vì sao cần IG ngoài gate weights?**

Gate weights phản ánh attention mechanism của model. Chúng có thể phân bổ cao cho một view ngay cả khi view đó không thực sự drive prediction nhiều — ví dụ khi gate học "attend to view X" nhưng thông tin trong view X lại không informative, gradient sẽ nhỏ.

**Integrated Gradients cho attribution trung thực hơn:**

```text
IG(view_k, task_j) = (x_k - x_k_baseline) · ∫₀¹ (∂F_j / ∂x_k)(x_k_baseline + λ·(x_k - x_k_baseline)) dλ
```

Với baseline = zero vector cho từng view.

**Population-level attribution:**

```text
IG_population(view_k, task_j) = E_x [ |IG(view_k, task_j)| ]
```

**Kết hợp hai phương pháp:**

| Phương pháp | Scope | Câu hỏi trả lời |
|---|---|---|
| Gate weights | Per-sample | View nào model "attend to" cho observation này? |
| Integrated Gradients | Population-level | View nào thực sự drive predictions trên toàn dataset? |

Nếu cả hai cùng chỉ ra một view là quan trọng → kết luận mạnh hơn.

### 14.3. Ablation-based importance

Không chỉ tin vào gate weights hay IG một mình. Cần kiểm tra bằng ablation:

```text
Performance drop khi bỏ từng view
```

Nếu gate/IG và ablation cùng chỉ ra một view quan trọng → **triple-validation**: mạnh nhất có thể với observational data.

### 14.4. Baseline interpretability cho cross-check

Với XGBoost/Random Forest, dùng SHAP hoặc permutation importance để so sánh chéo với view-level attribution của M2G-Net v2.

### 14.5. Policy interpretation

Mô hình nên dẫn đến insight kiểu:

```text
Rider role là yếu tố quan trọng nhất cho helmet non-use
→ consistent với Occupational Risk Theory.

Road/intersection context quan trọng hơn cho red-light running
→ consistent với Situational Crime Prevention.

Cross-level interaction cho thấy tác động của police presence thay đổi theo rider role.

Uncertainty σ_k cho thấy mobile phone use khó predict nhất,
có thể do behavior này nhạy cảm với nhiều cơ chế cùng lúc.
```

---

## 15. Evaluation Plan

### 15.1. Metrics

Vì các risky behaviors có thể mất cân bằng lớp, không nên chỉ dùng accuracy.

```text
ROC-AUC
PR-AUC
F1-score
Balanced accuracy
Precision
Recall
Calibration error, nếu cần
MTL Transfer Ratio (mới)
```

### 15.2. Vì sao PR-AUC quan trọng?

Nếu một hành vi rủi ro hiếm, model có thể đạt accuracy cao bằng cách dự đoán "No" cho hầu hết trường hợp. PR-AUC hữu ích hơn trong tình huống class imbalance.

### 15.3. Validation strategies

#### Strategy 1: Random split

Dùng để có kết quả cơ bản.

#### Strategy 2: Stratified split

Đảm bảo mỗi split giữ tỷ lệ class tương đối ổn định.

#### Strategy 3: Leave-intersection-out validation

Đây là phần quan trọng nhất.

```text
Train trên một số giao lộ
Test trên giao lộ chưa từng thấy
```

Mục tiêu: kiểm tra khả năng generalize sang địa điểm mới. Trong kịch bản này, site random intercept bị zero-out và chỉ dùng observed site features.

Nếu model chỉ tốt ở random split nhưng kém ở leave-intersection-out → có thể đang học thuộc đặc điểm site. Kết quả này tự thân cũng là đóng góp nghiên cứu.

### 15.4. Reporting table đề xuất

```text
Model                    ROC-AUC  PR-AUC  F1   Bal.Acc  MTL_Ratio
Decision Tree            xx       xx      xx   xx       N/A
Logistic Regression      xx       xx      xx   xx       N/A
Random Forest            xx       xx      xx   xx       N/A
XGBoost / LightGBM       xx       xx      xx   xx       N/A
Early-Fusion MLP         xx       xx      xx   xx       N/A
Late Fusion              xx       xx      xx   xx       N/A
Single-Task MLP          xx       xx      xx   xx       1.00 (ref)
M2G-Net v2         xx       xx      xx   xx       xx
```

Bảng nên báo cáo theo từng task, sau đó có macro-average.

---

## 16. Ablation Study

Ablation study là phần quan trọng nhất để chứng minh mô hình không chỉ phức tạp cho vui.

### 16.1. Ablation theo view

```text
Full model
- no Rider Role View
- no Rider Traits View
- no Road/Intersection Context View
- no Environmental Context View
- no Site Context View
```

### 16.2. Ablation theo fusion mechanism

```text
Early fusion only                  (alpha = 0)
Late fusion only
Shared gate for all tasks
Task-specific gates (no residual)
Task-specific gates + residual     (full v2)
```

### 16.3. Ablation theo multi-task learning

```text
Single-task models
Multi-task without gates
Multi-task with task-specific gates
Multi-task with gates + uncertainty weighting (full v2)
```

### 16.4. Ablation theo cross-level interaction

```text
Full model
- no cross-level interaction (chỉ dùng individual views và contextual views độc lập)
```

Mục tiêu: kiểm tra cross-level interaction có thực sự đóng góp vào performance không.

### 16.5. Ablation theo site encoding

```text
Full site: observed features + site random intercept (in-distribution)
Site features only (out-of-distribution / leave-intersection-out)
No site view
```

Mục tiêu: tách biệt tác động của observed infrastructure features và site-specific memorization.

---

## 17. Data Leakage và Methodological Cautions

### 17.1. Không dùng target làm input

Nếu các behaviors được quan sát đồng thời, không dùng behavior A làm input để dự báo behavior B. Điều này tạo data leakage.

**Setup khuyến nghị: Setup A — Context-only features làm input**

```text
Input:  Rider role, Rider traits, Road context, Environment, Site
Output: Risky behaviors
```

### 17.2. Overfitting

Dataset khoảng vài nghìn observations, không nên dùng model quá lớn.

```text
Small encoders (d ≤ 32)
Dropout (p = 0.2–0.3)
L2 regularization
Early stopping
Cross-validation
Leave-intersection-out validation
```

### 17.3. Causal caution

Mô hình dự báo không chứng minh quan hệ nhân quả.

```text
✓ Dùng: associated with, predictive contribution, model-based importance
✗ Tránh: causes, leads to, proves that
```

### 17.4. Negative transfer là kết quả nghiên cứu, không phải thất bại

Nếu MTL Transfer Ratio < 1 cho một task → báo cáo và giải thích. Kết quả này gợi ý rằng hành vi đó có cơ chế rủi ro khác biệt, không chia sẻ representation với các behaviors khác — là insight có giá trị.

---

## 18. Why This Model Is Optimized for This Dataset

| Yêu cầu | M2G-Net v2 |
|---|---|
| Không quá đơn giản | Có gated fusion, cross-level interaction, MTL |
| Không quá phức tạp | Không dùng GCN/RetNet/Transformer lớn; encoder nhỏ |
| Đúng bản chất tabular | Small encoders, categorical embeddings, regularized fusion |
| Site generalization | Site context dùng observed features + regularized intercept |
| Interpretable | Gate weights + Integrated Gradients + ablation |
| Defensible MTL claim | Single-task baselines + MTL Transfer Ratio |
| Theory-grounded | Mỗi view neo vào framework lý thuyết cụ thể |

---

## 19. Expected Contributions

### 19.1. Methodological contribution

Đề xuất một extension từ decision tree sang theory-guided multi-view multi-task gated fusion có phân tầng, cross-level interaction và uncertainty-weighted loss.

### 19.2. Predictive contribution

Kiểm tra liệu multi-view fusion có cải thiện dự báo risky behaviors so với decision tree, logistic regression, random forest, XGBoost/LightGBM, early-fusion MLP, late fusion và single-task baselines.

### 19.3. Interpretability contribution

Cung cấp phân tích:

```text
View-level contribution qua gate weights (per-sample)
View-level attribution qua Integrated Gradients (population-level)
Cross-level interaction: individual × contextual
Task-specific importance
Ablation-based evidence
Learned uncertainty σ_k per task
```

### 19.4. Policy contribution

```text
Nhóm rider nào có rủi ro gì?
Trong bối cảnh nào rủi ro tăng?
Individual và contextual factors tương tác như thế nào?
View nào nên được ưu tiên trong intervention cho từng behavior?
```

---

## 20. Slide Flow để trình bày với thầy

### Slide 1: Title

**M2G-Net v2: A Theory-Guided Multi-View Fusion Extension for Motorcycle Risky Behavior Prediction**

Thông điệp: Em muốn đề xuất một extension có kiểm soát từ paper gốc, không phải thay thế paper gốc.

---

### Slide 2: Paper gốc đã làm gì?

```text
Dataset: observed motorcycle riders at intersections
Groups: food delivery, ride-hailing, private riders
Outcomes: red-light running, no turn signal, helmet behavior, mobile phone use
Method: decision tree analysis
```

Thông điệp: Decision tree rất mạnh vì tạo rule dễ hiểu.

---

### Slide 3: Decision tree hoạt động như thế nào?

Minh họa:

```text
Rider type → Time slot → Traffic / Weather / Riding opposite way → Risk category
```

Thông điệp: Decision tree học rule tuần tự theo từng nhánh.

---

### Slide 4: Khoảng trống nhỏ

```text
Decision tree chọn split cục bộ ở từng node.
Các nhánh có thể không đồng bộ.
Khó so sánh hệ thống mức đóng góp của từng nhóm thông tin.
Không nắm bắt được individual × contextual interaction một cách hệ thống.
```

Thông điệp: Đây không phải lỗi của decision tree, mà là cơ hội để thử multi-view modeling có phân tầng.

---

### Slide 5: Ý tưởng chính

```text
Chia dataset thành 2 tầng view lý thuyết:
Individual level: Rider role, Rider traits
Contextual level: Road/intersection context, Environmental context, Site context

Học cross-level interaction: individual × contextual
Fusion bằng regularized task-specific gated fusion + residual early fusion
Loss: uncertainty-weighted multi-task
```

Thông điệp: Không gộp tất cả biến ngay từ đầu; học từng view riêng, học interaction, rồi fusion.

---

### Slide 6: Architecture

```text
Individual → Encoders → h_ind
Contextual → Encoders → h_ctx
Cross-Level Interaction → h_inter
→ Regularized Task-Specific Gated Fusion + Residual
→ Uncertainty-Weighted Multi-Task Outputs
```

Thông điệp: Mỗi task có thể học trọng số view khác nhau; individual × contextual interaction được học có cấu trúc.

---

### Slide 7: Outputs

```text
1. Probability of each risky behavior
2. Overall risk profile
3. Gate weight contribution (per-sample)
4. Integrated Gradients attribution (population-level)
5. Learned task uncertainty σ_k
```

Thông điệp: Model không chỉ dự báo mà còn hỗ trợ diễn giải ở nhiều tầng.

---

### Slide 8: Baselines

```text
Decision Tree
Logistic Regression
Random Forest
XGBoost / LightGBM
Early-Fusion MLP
Late Fusion
Single-Task MLP (4 models riêng biệt — để detect negative transfer)
M2G-Net v2
```

Thông điệp: Em sẽ không claim model mới tốt hơn nếu chưa so sánh công bằng, kể cả với single-task baselines.

---

### Slide 9: Evaluation

```text
ROC-AUC, PR-AUC, F1-score, Balanced accuracy
MTL Transfer Ratio (phát hiện negative transfer)
Leave-intersection-out validation (generalization sang site mới)
```

Thông điệp: Leave-intersection-out kiểm tra khả năng generalize; MTL Transfer Ratio kiểm tra liệu multi-task có giúp hay gây hại.

---

### Slide 10: Ablation

```text
Full model
- no rider role view
- no road context view
- no environment view
- no site features
- no cross-level interaction
- no gating (early fusion only)
- no multi-task (single-task)
- no uncertainty weighting
```

Thông điệp: Ablation cho biết component nào thật sự quan trọng.

---

### Slide 11: Expected value

```text
Nếu model tốt hơn:
→ multi-view fusion có phân tầng và cross-level interaction có ích.

Nếu model không tốt hơn về accuracy:
→ gate weights và IG attribution vẫn cung cấp view-level insights mà decision tree không cho.
→ Kết quả về negative transfer và σ_k vẫn có giá trị nghiên cứu.
```

Thông điệp: Dù kết quả thế nào, nghiên cứu vẫn có giá trị.

---

### Slide 12: Request

```text
Xin access dataset đã ẩn danh.
Cam kết bắt đầu bằng replication baseline.
Không chia sẻ dữ liệu ra ngoài.
Sản phẩm: notebook + report + baseline comparison + ablation analysis + view attribution report.
```

Thông điệp: Rủi ro thấp, giá trị nghiên cứu rõ ràng.

---

## 21. Pitch ngắn để nói với thầy

> Thưa thầy, em rất quan tâm đến bài *Not the same* vì dataset có cấu trúc rất phù hợp để thử một hướng mở rộng về predictive modeling. Paper gốc dùng decision tree để phân tích vai trò của rider type và road context đối với risky behaviors. Em muốn giữ decision tree làm baseline chính, sau đó thử một mô hình theory-guided multi-view multi-task gated fusion.
>
> Ý tưởng cốt lõi là tổ chức dataset thành hai tầng có ý nghĩa lý thuyết: individual level gồm rider role và rider traits, contextual level gồm road/intersection context, environmental context và site context. Mỗi view có encoder riêng, và model học cross-level interaction giữa individual và contextual representations trước khi dùng task-specific gated fusion để dự báo nhiều hành vi rủi ro cùng lúc với uncertainty-weighted loss.
>
> Mục tiêu của em không phải chứng minh model phức tạp hơn thì chắc chắn tốt hơn. Em muốn kiểm tra một cách nghiêm túc liệu multi-view representation learning có cải thiện prediction và interpretation so với decision tree, logistic regression, random forest/XGBoost, early-fusion MLP và single-task baselines hay không. Em cũng sẽ làm ablation study và dùng Integrated Gradients để phân tích view contribution ở population level, không chỉ dựa vào gate weights.
>
> Nếu thầy cho phép, em chỉ cần phiên bản dataset đã ẩn danh. Em sẽ bắt đầu bằng replication/baseline trước, sau đó mới thử fusion model. Sản phẩm cuối sẽ là code notebook, bảng so sánh baseline, ablation study, view attribution analysis và short technical report.

---

## 22. One-Minute Version

> Em muốn mở rộng bài *Not the same* bằng cách giữ decision tree làm baseline, rồi thử một theory-guided multi-view multi-task gated fusion model có phân tầng individual–contextual. Thay vì chỉ encode từng view riêng, em tổ chức rider role và rider traits thành individual level, còn road context, environmental context và site context thành contextual level, sau đó học cross-level interaction giữa hai tầng. Model dùng regularized task-specific gated fusion kết hợp residual early fusion để dự báo nhiều risky behaviors cùng lúc với uncertainty-weighted loss. Em sẽ so sánh với decision tree, logistic regression, random forest/XGBoost, early-fusion MLP và single-task baselines để detect negative transfer; đồng thời làm ablation study và dùng Integrated Gradients để phân tích view contribution ở population level. Mục tiêu không phải thay thế paper gốc, mà là kiểm tra xem cấu trúc multi-view phân tầng có tận dụng được thông tin tiềm ẩn trong dataset quan sát tốt hơn không.

---

## 23. 30-Second Version

> Em muốn thử một extension từ decision tree sang theory-guided multi-view fusion có phân tầng. Rider role và rider traits là individual level; road context, environment và intersection là contextual level. Model học cross-level interaction giữa hai tầng, dùng gate riêng cho từng hành vi rủi ro, và dùng uncertainty-weighted loss để cân bằng gradient giữa các tasks. Em sẽ giữ decision tree làm baseline, thêm single-task baselines để detect negative transfer, và dùng Integrated Gradients để phân tích view attribution rõ hơn gate weights.

---

## 24. Câu chốt mạnh nhất

> **The value of this project is not that the proposed model is more complex than decision trees, but that it can systematically test whether theoretically distinct sources of risk — at individual and contextual levels, and in their interaction — contribute independently or interactively to unsafe riding behavior across multiple outcomes simultaneously.**

Bản tiếng Việt:

> **Giá trị của dự án này không nằm ở việc mô hình mới phức tạp hơn decision tree, mà nằm ở việc nó kiểm tra một cách hệ thống liệu các nguồn rủi ro khác nhau về mặt lý thuyết — ở tầng individual, tầng contextual và trong tương tác giữa hai tầng — đóng góp độc lập hay tương tác với nhau trong việc hình thành nhiều hành vi lái xe không an toàn cùng lúc.**

---

## 25. Final Recommended Claim

Claim nên dùng:

> **This project does not aim to replace the original decision-tree analysis. Instead, it aims to test whether a theory-guided multi-view multi-task gated fusion model — with hierarchical individual–contextual encoding and cross-level interaction — can provide complementary predictive and interpretive insights into motorcycle risky riding behaviors.**

Tránh claim quá mạnh như:

```text
✗ This model will outperform all baselines.
✗ This model proves causal effects.
✗ This model replaces decision tree.
✗ This is a completely new architecture.
```

Claim an toàn và thuyết phục hơn:

```text
✓ This model tests a controlled methodological extension with hierarchical structure.
✓ This model preserves interpretability through view-level gates, Integrated Gradients and ablation.
✓ This model may reveal whether individual and contextual sources of risk interact differently across behaviors.
✓ This model provides uncertainty estimates for each predicted behavior as additional research output.
```

---

## 26. Checklist trước khi xin dataset

```text
[ ] Hiểu rõ decision tree baseline trong paper gốc
[ ] Có framework diagram của M2G-Net v2 (phân tầng individual–contextual)
[ ] Có danh sách input views và tầng phân loại của từng view
[ ] Có danh sách target behaviors
[ ] Có lý thuyết neo cho từng view (bảng theory mapping)
[ ] Có baseline comparison plan, bao gồm single-task baselines
[ ] Có ablation plan, bao gồm ablation theo cross-level interaction
[ ] Có validation plan, đặc biệt leave-intersection-out
[ ] Hiểu site encoding strategy: khi nào dùng random intercept, khi nào zero-out
[ ] Có kế hoạch Integrated Gradients để phân tích view attribution
[ ] Có cam kết không chia sẻ dữ liệu
[ ] Có output cụ thể: notebook + report + baseline comparison + ablation analysis + view attribution report
```

---

## 27. Kết luận

**M2G-Net v2** là phiên bản cải thiện cân bằng được bốn yêu cầu:

1. **Có novelty thực chất:** không chỉ thêm layer, mà thêm cấu trúc lý thuyết — phân tầng individual–contextual, cross-level interaction, uncertainty-weighted loss và attribution qua Integrated Gradients.

2. **Không quá đà:** không dùng mô hình spatio-temporal/graph/RetNet khi dataset không có time-series hoặc graph rõ ràng; site context dùng observed features thay vì pure learned embedding để tránh overfit.

3. **Defensible:** giữ decision tree làm baseline, thêm single-task baselines để detect negative transfer, báo cáo MTL Transfer Ratio, và không che giấu kết quả âm tính.

4. **Theory-grounded:** mỗi view được neo vào framework lý thuyết cụ thể, biến view decomposition thành claim kiểm chứng được, không chỉ là engineering choice.

Cách định vị tốt nhất:

> Đây không phải một mô hình nhằm thay thế paper gốc. Đây là một extension có kiểm soát để kiểm tra xem cấu trúc multi-view phân tầng tiềm ẩn trong dataset quan sát — và tương tác giữa individual và contextual factors — có thể được khai thác tốt hơn cho prediction và interpretation hay không.
# Archived Draft Notice

This file is an older methodology draft and may contain stale notation,
unmasked-loss formulas, or stronger claims than the current evidence supports.
Use `docs/MATHEMATICAL_FORMULAS.md`, `docs/ARCHITECTURE_VISUALIZATION.md`,
`README.md`, `config.py`, and the `src/` code as the current source of truth.
