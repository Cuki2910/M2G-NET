# Kế hoạch Cải thiện Mô hình M2G-Net v2

Tài liệu này tổng hợp các điểm đánh giá phản biện chuyên sâu (dựa trên tiêu chuẩn của tạp chí *Transportation Research* và *Accident Analysis & Prevention*), phân loại theo mức độ hoàn thành để chuẩn bị cho việc xuất bản nghiên cứu.

## ✅ Đã xử lý (Completed)

1. **Định vị & Ý nghĩa Thực tiễn (Positioning & Practical Implications)**
   - **Vấn đề:** Tên gọi "Theory-Guided" chung chung và thiếu mô tả ứng dụng thực tế.
   - **Giải pháp đã làm:** Cập nhật `README.md`, liên kết rõ ràng việc chia "View" với các lý thuyết hành vi giao thông (VD: *Theory of Planned Behavior*). Bổ sung đoạn nhấn mạnh tính thực tiễn: "hỗ trợ các nhà hoạch định thiết kế nút giao an toàn và tối ưu hóa hệ thống giám sát".

2. **Xử lý Mất cân bằng Lớp (Class Imbalance - Tier 4)**
   - **Vấn đề:** Dữ liệu hành vi vi phạm thường mất cân bằng, dễ làm mô hình sụp đổ nếu dùng Cross Entropy thông thường.
   - **Giải pháp đã làm:** Kích hoạt `USE_FOCAL_LOSS = True` trong `config.py` và cập nhật tài liệu `ARCHITECTURE_VISUALIZATION.md`.

---

## 🚧 Cần xử lý tiếp theo (Pending Improvements)

### 1. Dữ liệu Thực tế (Ưu tiên Tuyệt đối - Tier 1)
- **Vấn đề:** Đánh giá hiện tại phụ thuộc hoàn toàn vào dữ liệu tổng hợp (synthetic data), sẽ bị desk-reject ở các tạp chí top đầu.
- **Hành động:** 
  - Thu thập/sử dụng tập dữ liệu quan sát hành vi giao thông thực tế (từ camera giao thông hoặc dữ liệu mở).
  - Chạy lại toàn bộ pipeline huấn luyện và đánh giá trên dữ liệu này.

### 2. Phân tích Nhân quả & Trọng số Cổng (Causal vs. Gate Weights - Tier 3)
- **Vấn đề:** Trọng số cổng (Gate weights / $\alpha_k$) chỉ thể hiện "sự chú ý" (attention) của mô hình, không phải quan hệ nhân quả.
- **Hành động:** 
  - Báo cáo rõ ràng trong bài báo rằng Gate Weights không thay thế phân tích nhân quả.
  - Sử dụng và báo cáo kết quả từ các công cụ giải thích Post-hoc đã có trong pipeline (như *Integrated Gradients*) để phân tích tầm quan trọng của đặc trưng (Feature Importance).

### 3. Biện luận sức mạnh Đa nhiệm (Multi-task Synergy - Tier 5)
- **Vấn đề:** Cần lập luận tại sao lại nhóm 4 hành vi này (vượt đèn đỏ, không xi nhan, không đội mũ, dùng điện thoại) vào cùng một mô hình.
- **Hành động:** 
  - Bổ sung vào bài viết khoa học (manuscript) lập luận rằng các hành vi vi phạm này có sự tương quan nội tại (VD: người có thiên hướng chấp nhận rủi ro thường vi phạm nhiều lỗi cùng lúc). Việc học đa nhiệm giúp mô hình học được đặc trưng ẩn (latent traits) chung tốt hơn là huấn luyện 4 mô hình độc lập.

### 4. Báo cáo Bằng chứng Khách quan (Evidence Needed)
- **Ablation Study Định lượng:** Phải chứng minh được mô-đun *Cross-Level Interaction* và *Task-Specific Gated Fusion* mang lại khác biệt lớn và có ý nghĩa thống kê so với một mạng MLP cơ bản (Base MLP).
- **Phân tích ổn định (Gate Stability):** Cần chạy thử nhiều random seed (ví dụ: 10 lần) để báo cáo mức độ biến động (variance) của các Gate weights, đảm bảo mô hình không hội tụ vào các cấu hình chú ý ngẫu nhiên.
- **Đánh giá Độ tin cậy (Calibration/ECE):** Báo cáo Expected Calibration Error để chứng minh xác suất (probability) do mô hình đưa ra có thể tin cậy cho việc ra quyết định.

### 5. Nâng cấp xử lý Yếu tố Địa điểm (Site-Aware Modeling - Tier 2) (Tùy chọn nâng cao)
- **Vấn đề:** Việc dùng weight-decay embedding cho `h_site_id` chỉ tạo ra các hệ số chặn (intercept) tĩnh.
- **Hành động:** Nếu bài toán yêu cầu tổng quát hóa (generalize) sang các nút giao mới chưa từng gặp trong tập train, nên xem xét áp dụng Bayesian Hierarchical Modeling cho lớp layer cuối cùng để ước lượng chuẩn xác các biến động ngẫu nhiên theo không gian (spatial random effects).
