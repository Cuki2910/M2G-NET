# m2g_net_v3 — LOCKED TEST conclusion

**Kết luận: CHƯA VƯỢT TRỘI theo tiêu chí đã khai báo trước (Tiêu chí 1 trượt 7/16 điều kiện).**

Phạm vi: dữ liệu mô phỏng (synthetic), 20 bộ sinh seed 1711–1730 × 3 cơ chế × 2 training seed (42, 101; lấy trung bình mỗi bộ). Chạy MỘT lần sau FREEZE (`FREEZE.json`, `refrozen.at` mới nhất) và sau khi người dùng xác nhận. 1080/1080 đơn vị chạy xong, 0 lỗi. Không có p-value; báo mean, SD, MCSE = SD/√n, min–max, số bộ dương. Không chạy lại với biến thể nào.
Đây không phải kiểm chứng trên dữ liệu thực.

Nguồn số liệu: `criteria.csv`, `per-set-locked.csv`, `results-locked.csv`, `summary-locked.csv`, `locked-manifest.json` (cùng thư mục).

## Tiêu chí 1 — macro AUC, v3 − đối thủ (cần ≥18/20 bộ dương VÀ mean > +0.005)

| Đối thủ | interaction: mean ± MCSE (min–max), dương | Đạt | partial: mean ± MCSE (min–max), dương | Đạt |
|---|---|---|---|---|
| m2g_net | +0.0046 ± 0.0007 (−0.0007…+0.0103), 19/20 | **Không** (mean < 0.005) | +0.0052 ± 0.0009 (−0.0022…+0.0118), 18/20 | Đạt |
| shared_gate | +0.0029 ± 0.0008 (−0.0023…+0.0091), 16/20 | **Không** | +0.0018 ± 0.0011 (−0.0096…+0.0103), 15/20 | **Không** |
| no_interaction | +0.0040 ± 0.0009 (−0.0041…+0.0118), 17/20 | **Không** | +0.0049 ± 0.0007 (−0.0002…+0.0105), 18/20 | **Không** (mean < 0.005) |
| early_fusion | +0.0041 ± 0.0007 (−0.0020…+0.0111), 19/20 | **Không** (mean < 0.005) | +0.0037 ± 0.0009 (−0.0033…+0.0104), 16/20 | **Không** |
| single_task_mlp | +0.0186 ± 0.0014, 20/20 | Đạt | +0.0247 ± 0.0015, 20/20 | Đạt |
| early_fusion_mlp | +0.0115 ± 0.0013, 20/20 | Đạt | +0.0137 ± 0.0014, 20/20 | Đạt |
| logistic_regression | +0.0885 ± 0.0021, 20/20 | Đạt | +0.0885 ± 0.0021, 20/20 | Đạt |
| catboost | +0.0103 ± 0.0012, 20/20 | Đạt | +0.0139 ± 0.0015, 20/20 | Đạt |

9/16 điều kiện đạt, 7/16 trượt. Tất cả điều kiện trượt là so với các biến thể cùng họ kiến trúc (m2g_net, shared_gate, no_interaction, early_fusion). v3 hơn cả bốn đối thủ ngoài họ (single_task_mlp, early_fusion_mlp, logistic_regression, catboost) ở cả hai cơ chế.

Hai ca sát ngưỡng, ghi rõ để không đọc sai: interaction vs m2g_net đạt 19/20 bộ dương nhưng mean +0.0046 < +0.005; partial vs no_interaction đạt 18/20 nhưng mean +0.0049 < +0.005. Cả hai vẫn tính là trượt, ngưỡng không được đổi.

## Tiêu chí 2 — additive (chỉ báo cáo; v3 không được thua catboost và m2g_net về trung bình)

Đạt: vs catboost +0.0062 ± 0.0014 (16/20), vs m2g_net +0.0039 ± 0.0008 (18/20).
Báo cáo thêm (không phải tiêu chí): ở additive v3 thua single_task_mlp (−0.0023, 8/20 dương), early_fusion_mlp (−0.0023, 6/20) và logistic_regression (−0.0059, 3/20).

## Tiêu chí 3 — Brier (mean v3 − đối thủ ≤ 0, trung bình trên 3 cơ chế)

Đạt: vs early_fusion_mlp −0.0030 ± 0.0003 (0/20 bộ có v3 tệ hơn), vs catboost −0.0042 ± 0.0003.
Theo từng cơ chế: interaction và partial đều tốt hơn cả hai đối thủ. Additive: tốt hơn catboost (−0.0010) nhưng kém early_fusion_mlp một chút (+0.0002 ± 0.0003, 13/20 bộ kém hơn). Tiêu chí khai báo là trung bình trên 3 cơ chế nên vẫn đạt; hàng additive được nêu để minh bạch.

## Tiêu chí 4

Không dùng p-value, không dùng chữ "có ý nghĩa thống kê". Mọi con số ở trên là mean ± MCSE, min–max, số bộ dương.

## Điều cần nói rõ

- **Vì Tiêu chí 1 trượt nên kết luận là "chưa vượt trội".** LOCKED TEST không chạy lại.
- Một điều chỉnh trước khi chạy, làm trước khi có dữ liệu locked: `run_extension_v3.py` được thêm dòng Brier `aggregate` (trung bình 3 cơ chế) cho đúng lời văn giao thức, giữ nguyên các dòng theo cơ chế. Ghi trong `FREEZE.json` (`refrozen.reason`).
- Các vòng dev (gate, skip, weight decay, ensemble) không được áp dụng; cấu hình đóng băng là M5. Kết quả dev dự báo trước đó (khoảng +0.004 đến +0.006 so với các biến thể cùng họ) khớp với kết quả locked ở quy mô độ lớn.
- Không rút ra kết luận về hiệu quả trên dữ liệu thực; dữ liệu là mô phỏng.
