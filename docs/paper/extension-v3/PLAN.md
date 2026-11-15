# M2G-NET v3 — kế hoạch khai báo trước (viết trước khi sửa code)

Ngày lập: 2026-11-15. Dữ liệu synthetic. Không có bất kỳ kết luận nào về dữ liệu thực.

## 0. Giả định rút ra từ việc đọc mã (bước 1)

1. Hàm sinh dữ liệu (`scripts/generate_paper_simulation.py`) là logistic tuyến tính trên
   contrast ±1 của 8 biến phân loại + offset site, cộng 3 tích đôi (interaction/partial).
   Hàm thật có độ phức tạp thấp; 2100 hàng train, ~300 hàng validation, 600 hàng test.
2. Trần oracle ≈ 0.65/0.76/0.77 → mọi mô hình đang cách trần ≈ 0.02 AUC. Khoảng cách này
   chủ yếu là phương sai ước lượng (ít dữ liệu, early stopping trên AUC validation 300 hàng,
   một lần khởi tạo), không phải thiếu năng lực biểu diễn (các ablation gate/interaction
   nằm trong nhiễu).
3. Brier của m2g_net kém hơn early_fusion_mlp/catboost ≈ 0.012–0.014. Nguyên nhân rõ nhất:
   `USE_FOCAL_LOSS=True` (α=0.25, γ=2). Focal loss không phải proper scoring rule, làm xác
   suất đầu ra co về giữa; scorer dùng `raw_probability` (chưa calibrate) nên Brier chịu
   thiệt trực tiếp. Hai MLP baseline dùng BCE thuần.
4. `scripts/evaluate/paper_pipeline.py` chọn kiến trúc theo tên trong `NETWORKS`; v3 phải
   được đăng ký ở đây và được chọn bằng validation, không dùng test split.
5. `scripts/run_extension.py` gọi `verify_sources(plan)` với hash của code đóng băng v2.
   Sửa `src/`/`config.py` sẽ làm hash đổi → runner mới cho 1711–1730 phải kiểm hash theo
   `FREEZE.json` (v3), và phải chứng minh đối thủ không đổi hành vi (xem 3.4).
6. Checkpoint `model.pt` hiện lưu một state_dict; ensemble cần lưu danh sách thành viên.
   Scorer chỉ cần `predictions.csv`, nên định dạng mới không ảnh hưởng chấm điểm.

## 1. Ba hướng nâng cấp đã cân nhắc (bước 2)

A. **Proper scoring loss**: v3 tắt focal (dùng BCE masked + uncertainty weighting như cũ).
   Hợp lý vì nhắm thẳng vào điểm yếu Brier; không đổi kiến trúc, không đổi công thức mô hình.
   Rủi ro: AUC gần như không đổi (focal ít đổi thứ hạng) → có thể không đạt +0.005 AUC.

B. **Ensemble theo seed khởi tạo** (M mô hình cùng kiến trúc, cùng split, seed con khác
   nhau, trung bình xác suất). Hợp lý vì phần lớn khoảng cách tới oracle là phương sai;
   trung bình M fit giảm phương sai dự đoán, cải thiện đồng thời AUC và Brier, và tính
   ổn định theo tập (18/20) chính là thứ ensemble mua được. Rủi ro: chi phí ×M; nếu lệch
   (bias) chiếm ưu thế thì lợi ích < 0.005; catboost vốn là ensemble nên khoảng cách với
   catboost có thể hẹp hơn.

C. **Giảm dung lượng / tăng regularization** (VIEW_DIM nhỏ hơn, weight decay lớn hơn,
   early stopping trên NLL). Rủi ro: cần sweep, mà validation 300 hàng quá nhiễu để chọn;
   dễ tự lừa bằng tuning.

**Chọn: A + B gộp thành một biến thể duy nhất `m2g_net_v3` = kiến trúc TGMVMTGFNetV2
"full" giữ nguyên, loss BCE (không focal), ensemble M=5 seed khởi tạo.** Chỉ một biến thể
được đóng băng; không chọn giữa nhiều biến thể bằng test. Nếu khảo sát phát triển cho
thấy A hoặc B đơn lẻ tốt hơn trên validation, ghi lại nhưng vẫn chỉ đóng băng một cấu hình.

Không đổi công thức mô hình (mục 1–6 của `docs/MATHEMATICAL_FORMULAS.md`). Chỉ cần cập
nhật mục 7 (câu "Focal loss is enabled") và thêm công thức ensemble
\(\hat y_k = \frac1M\sum_m \hat y_k^{(m)}\) cho v3.

## 2. Siêu tham số mới (tất cả trong `config.py`)

```
V3_USE_FOCAL_LOSS = False
V3_ENSEMBLE_SIZE  = 5
V3_MEMBER_SEED_OFFSET = 1000   # seed thành viên m = split_seed + m*V3_MEMBER_SEED_OFFSET
```
Mọi hằng số khác dùng chung với m2g_net (LR, WD, batch, epochs, patience, gate prior 0.1).

## 3. Kế hoạch bước -> kiểm tra

3.1 `src/loss.py`: không đổi API; `UncertaintyWeightedLoss(use_focal=...)` đã có tham số.
    `scripts/train.py::fit_model_for_split` thêm tham số `use_focal=cfg.USE_FOCAL_LOSS`
    (mặc định giữ hành vi cũ). -> kiểm tra: gọi với mặc định cho kết quả bitwise bằng cũ
    trên 1 dev set (3.4).
3.2 `scripts/evaluate/paper_pipeline.py`: thêm `"m2g_net_v3"` vào `NETWORKS`; trong
    `select_network`, nếu tên là v3: với prior 0.1 (cố định, không sweep), fit
    `V3_ENSEMBLE_SIZE` thành viên với `split_seed = seed + m*offset`, `use_focal=False`;
    dự đoán = trung bình xác suất các thành viên trên validation và test; `score` báo cáo
    = AUC validation của ensemble; `model.pt` lưu `{"ensemble_members": [payload...]}`.
    -> kiểm tra: `--smoke` chạy xong, `predictions.csv` có đủ 4 task × 600 hàng, xác suất
    trong [0,1].
3.3 Khảo sát phát triển (chỉ seed 1601, 1701–1710; chỉ validation split, chấm bằng oracle
    trên hàng validation eligible để giảm nhiễu nhãn thiếu): so v3 với m2g_net cũ và
    early_fusion_mlp theo macro AUC và Brier trên validation. Mục đích: xác nhận chiều
    hướng (v3 ≥ cũ), KHÔNG dùng để chọn giữa nhiều biến thể. -> kiểm tra: bảng
    `docs/paper/extension-v3/dev-validation.csv`; không mở `predictions` test.
3.4 Kiểm tra tương đương đối thủ: chạy lại `m2g_net` và `early_fusion_mlp` bằng code mới
    trên `outputs/paper-extension/data/interaction-1704`, seed 42; so `predictions.csv`
    với unit đóng băng trong `outputs/paper-extension/units/`. -> kiểm tra: `cmp` bằng
    nhau (hoặc sai số ≤1e-12 nếu khác định dạng in). Nếu không bằng → dừng, báo.
3.5 Tests: `tests/test_gate_prior_smoothing.py` (bắt buộc nếu `src/fusion.py` bị đụng —
    kế hoạch này không đụng), toàn bộ `tests/`. -> kiểm tra: báo số pass/fail thật.
3.6 FREEZE: ghi `docs/paper/extension-v3/FREEZE.json` = cấu hình v3 + sha256 của
    `src/*.py`, `config.py`, `scripts/train.py`, `scripts/evaluate/paper_pipeline.py`,
    `baselines/run_all_baselines.py`, `scripts/run_extension_v3.py`. DỪNG, xin xác nhận.
3.7 Sau xác nhận: `scripts/run_extension_v3.py` (clone từ `run_extension.py`): seeds
    1711–1730, models = 8 đối thủ + `m2g_net_v3`, training seeds 42 & 101, kiểm hash theo
    FREEZE.json trước và sau; sinh dữ liệu + check_data; fit; chấm bằng
    `score_predictions`; xuất `results-locked.csv`, `paired-deltas-locked.csv`,
    `criteria.csv`, `CONCLUSION.md`. Chạy đúng một lần.

## 4. Tiêu chí thành công (đã khai báo, không đổi sau khi thấy kết quả)

Trên 20 tập LOCKED TEST, mỗi tập lấy trung bình 2 training seed:
1. interaction và partial, với TỪNG đối thủ trong {m2g_net, shared_gate, no_interaction,
   early_fusion, single_task_mlp, early_fusion_mlp, logistic_regression, catboost}:
   hiệu macro AUC (v3 − đối thủ) > 0 ở ≥ 18/20 tập VÀ trung bình > +0.005.
2. additive: chỉ báo cáo; v3 không thua catboost và m2g_net cũ về trung bình.
3. Brier: v3 không kém early_fusion_mlp và catboost về trung bình ở cả 3 cơ chế.
4. Báo mean ± MCSE (SD/√20), min–max, số tập dương. Không p-value.
Bất kỳ điều kiện nào hỏng → kết luận "chưa vượt trội", kèm bảng đầy đủ.

## 5. Ước lượng chi phí LOCKED TEST

Mỗi tập: 8 đối thủ ≈ 77 s + v3 ≈ 5×14 ≈ 72 s, ×2 training seed ≈ 300 s. 60 tập (20 seed × 3
cơ chế) ≈ 5 giờ CPU đơn luồng; 4 worker song song ≈ 1.3 giờ.

## 6. Phân công gợi ý (agent rẻ hơn làm được gì)

Giao được (cơ học, có spec rõ, có kiểm tra tự động): 3.1, 3.2, 3.4, 3.5, 3.7 (chạy runner,
không diễn giải). Cần người/agent mạnh hơn: thiết kế đã xong ở đây; review diff 3.2 (sai
seed thành viên, rò test vào selection, lưu checkpoint sai); đọc bảng 3.3 và quyết định
freeze; viết `CONCLUSION.md` theo đúng tiêu chí mục 4.
