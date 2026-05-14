# Methods, Terms, Symbols, and Measurement Glossary

Tài liệu này định nghĩa các phương pháp, thuật ngữ, ký hiệu và chỉ số đo lường
được dùng trong M2G-Net v2. Có thể dùng như phụ lục thuật ngữ cho bài viết,
luận văn, hoặc báo cáo kỹ thuật. Dữ liệu hiện tại trong repository là dữ liệu
tổng hợp kiểm chứng phương pháp, trừ khi được thay bằng dữ liệu quan sát thực.

## 1. Ký hiệu chung

| Ký hiệu | Ý nghĩa |
|---|---|
| \(N\) | Số mẫu quan sát trong tập dữ liệu hoặc mini-batch, tùy ngữ cảnh. |
| \(K\) | Số tác vụ dự đoán nhị phân. Trong cấu hình hiện tại, \(K=4\). |
| \(V\) | Số view đầu vào. Trong M2G-Net, \(V=5\): rider role, rider traits, road context, environment, site-observed. |
| \(x^{(n)}\) | Đầu vào của mẫu thứ \(n\). |
| \(x_i\) | View đầu vào thứ \(i\). |
| \(y_k^{(n)}\) | Nhãn nhị phân của mẫu \(n\) cho tác vụ \(k\), nhận giá trị 0 hoặc 1. |
| \(\hat{y}_k^{(n)}\) | Xác suất mô hình dự đoán cho lớp dương của tác vụ \(k\). |
| \(m_k^{(n)}\) | Mask nhãn: bằng 1 nếu nhãn tác vụ \(k\) của mẫu \(n\) được quan sát, bằng 0 nếu thiếu hoặc không áp dụng. |
| \(s\) | Mã định danh site/intersection sau khi encode. |
| \(S_{\mathrm{train}}\) | Số site xuất hiện trong tập train. |
| \(s_{\mathrm{unk}}\) | Mã site dành cho site chưa từng thấy trong train. |
| \(d\) | Số chiều biểu diễn ẩn của mỗi view. |
| \(h_i\) | Biểu diễn ẩn của view thứ \(i\). |
| \(h_{\mathrm{ind}}\) | Biểu diễn cấp cá nhân, ghép từ rider role và rider traits. |
| \(h_{\mathrm{ctx}}\) | Biểu diễn cấp bối cảnh, ghép từ road context, environment, và site-aware context. |
| \(h_{\mathrm{site}}\) | Biểu diễn site-aware, gồm đặc trưng site quan sát được và site intercept học được. |
| \(h_{\mathrm{int}}\) | Biểu diễn tương tác cross-level giữa cá nhân và bối cảnh. |
| \(\tilde{h}_j\) | Đầu vào thứ \(j\) của gate, gồm 5 view và 1 interaction đã chiếu về cùng chiều. |
| \(\alpha_{k,j}\) | Trọng số gate của tác vụ \(k\) dành cho đầu vào gate \(j\). |
| \(T\) | Gate temperature, điều khiển độ sắc của phân phối gate trước khi smoothing. |
| \(\lambda\) | Gate prior weight, trọng số trộn với prior đều trong công thức sparsemax uniform mixing. |
| \(p_0\) | Prior đều trên các đầu vào gate, bằng \(\frac{1}{V+1}\mathbf{1}_{V+1}\). |
| \(z_{\mathrm{gated}}^{(k)}\) | Biểu diễn sau gated fusion cho tác vụ \(k\). |
| \(z_{\mathrm{early}}\) | Biểu diễn early-fusion baseline dùng để trộn residual. |
| \(z_k\) | Biểu diễn cuối cùng đưa vào head dự đoán của tác vụ \(k\). |
| \(s_k=\log\tau_k^2\) | Tham số log-variance học được trong uncertainty-weighted loss. |
| \(\tau_k\) | Độ bất định học được của tác vụ \(k\). |
| \(\theta\) | Tập tham số học được của mô hình. |
| \(\tau\) hoặc threshold | Ngưỡng chuyển xác suất thành nhãn dự đoán 0/1. Không nên nhầm với \(\tau_k\) của uncertainty loss. |

## 2. Thuật ngữ mô hình và phương pháp

| Thuật ngữ | Định nghĩa và cách hiểu trong bài |
|---|---|
| M2G-Net | Multi-View Multi-Task Gated Fusion Network, kiến trúc học sâu dùng nhiều view đầu vào, nhiều tác vụ dự đoán và cơ chế gated fusion theo tác vụ. |
| Multi-view learning | Học từ nhiều nhóm đặc trưng có ý nghĩa khác nhau, ví dụ đặc trưng người lái, đường, môi trường và site. Mục tiêu là giữ cấu trúc ngữ nghĩa thay vì gộp tất cả cột ngay từ đầu. |
| Multi-task learning | Huấn luyện một mô hình chung cho nhiều tác vụ liên quan. Các tác vụ chia sẻ encoder nhưng có head dự đoán riêng. |
| View-specific encoder | Bộ mã hóa riêng cho từng view. Mỗi encoder biến đặc trưng categorical hoặc numerical của view đó thành biểu diễn ẩn \(h_i\). |
| Site-aware representation | Biểu diễn site/intersection kết hợp đặc trưng site quan sát được với embedding site học được. Thành phần này giống site-specific intercept, nhưng không phải một mixed-effects model đầy đủ. |
| Unknown-site id | Mã dự phòng cho site không xuất hiện trong train. Khi đánh giá out-of-site nghiêm ngặt, site intercept có thể bị tắt để tránh dùng thông tin site không hợp lệ. |
| Cross-level interaction | Thành phần học tương tác giữa biểu diễn cá nhân và biểu diễn bối cảnh. Trong M2G-Net, nó dùng tổng các projection và tích từng phần tử để mô hình hóa cá nhân trong bối cảnh cụ thể. |
| Gated fusion | Cơ chế trộn các view và interaction bằng trọng số gate theo từng tác vụ. Gate cho biết mô hình đang dựa nhiều hơn vào nguồn biểu diễn nào cho từng tác vụ. |
| Task-specific gate | Gate riêng cho mỗi tác vụ. Điều này cho phép tác vụ A ưu tiên một view khác tác vụ B. |
| Sparsemax | Hàm biến logits thành phân phối xác suất có tổng bằng 1 và có thể tạo trọng số đúng bằng 0. Khác softmax, sparsemax có thể chọn một tập nhỏ đầu vào gate. |
| Sparsemax uniform prior mixing | Công thức gate hiện tại: \(\alpha_k=(1-\lambda)\operatorname{sparsemax}(\mathrm{logits}/T)+\lambda p_0\). Khi \(\lambda>0\), mọi đầu vào gate có lower bound \(\lambda/(V+1)\), nên gate không còn sparse tuyệt đối. |
| Gate temperature | Tham số \(T\) chia logits trước sparsemax. \(T\) thấp làm gate sắc hơn, \(T\) cao làm gate đều hơn. |
| Temperature annealing | Lịch thay đổi temperature trong quá trình huấn luyện để điều chỉnh độ sắc của gate theo epoch. |
| Residual early-fusion blending | Cơ chế trộn biểu diễn gated fusion với biểu diễn early fusion bằng hệ số sigmoid học được. Nó giúp mô hình vẫn có đường biểu diễn gộp cơ bản nếu gate chưa ổn định. |
| Prediction head | Lớp dự đoán riêng cho từng tác vụ, biến \(z_k\) thành xác suất \(\hat{y}_k\). |
| Masked loss | Hàm mất mát chỉ tính trên các nhãn có mask bằng 1, phù hợp với dữ liệu nhiều tác vụ có nhãn thiếu. |
| Uncertainty-weighted loss | Cách cân bằng các tác vụ bằng tham số bất định học được. Tác vụ có loss nhiễu hơn có thể được giảm trọng số tương đối. |
| Binary cross-entropy, BCE | Loss phổ biến cho phân loại nhị phân, đo sai khác giữa nhãn 0/1 và xác suất dự đoán. |
| Focal loss | Biến thể của BCE giảm ảnh hưởng của mẫu dễ và tập trung hơn vào mẫu khó hoặc lớp hiếm. Trong cấu hình hiện tại, focal loss là tùy chọn và mặc định tắt. |
| Positive-class weighting | Tăng trọng số cho lớp dương khi lớp dương hiếm. Dùng để giảm thiên lệch về lớp âm trong dữ liệu mất cân bằng. |
| Calibration | Mức độ xác suất dự đoán khớp với tần suất thực tế. Ví dụ, trong nhóm mẫu được dự đoán khoảng 0.70, nếu khoảng 70% thật sự là lớp dương thì mô hình được calibrated tốt. |
| Temperature scaling | Hiệu chỉnh hậu nghiệm bằng cách chia logits cho một temperature học trên validation set. Mục tiêu là cải thiện calibration mà thường không thay đổi thứ hạng dự đoán. |
| Threshold tuning | Chọn ngưỡng xác suất trên validation set để tối ưu F1 hoặc balanced accuracy. Chỉ ảnh hưởng metric phụ thuộc threshold, không ảnh hưởng ROC-AUC hoặc PR-AUC. |
| Ablation study | Thí nghiệm bỏ từng thành phần, view hoặc cơ chế để đo đóng góp của thành phần đó. AUC drop càng lớn thì thành phần bị bỏ càng quan trọng trong bối cảnh thí nghiệm. |
| Leave-site-out validation | Đánh giá tổng quát hóa theo site bằng cách giữ một hoặc nhiều site làm test và train trên các site còn lại. Nghiêm ngặt hơn random split nếu mục tiêu là out-of-site generalization. |
| Independent test set | Tập test độc lập với train/validation. Trong repository hiện tại, tập này vẫn là synthetic out-of-site test nếu chưa thay bằng dữ liệu thực. |
| Repeated runs | Huấn luyện nhiều lần với các random seed khác nhau để ước lượng độ ổn định của metric. |
| Paired t-test | Kiểm định thống kê so sánh hai mô hình trên cùng các seed hoặc cùng fold. Dùng để hỏi chênh lệch trung bình có đáng kể thống kê hay không. |
| Confidence interval, CI | Khoảng tin cậy của một ước lượng, ví dụ mean ROC-AUC. CI hẹp hơn thường cho thấy ước lượng ổn định hơn. |
| Integrated Gradients | Phương pháp giải thích mô hình bằng cách tích lũy gradient từ baseline đến input thật. Dùng để ước lượng đóng góp của feature vào dự đoán. |
| Gate weight interpretation | Trọng số gate là tín hiệu mô hình đang sử dụng view nào tương đối nhiều hơn. Không nên diễn giải trực tiếp là quan hệ nhân quả. |

## 3. Ma trận nhầm lẫn và ký hiệu phân loại

| Ký hiệu | Định nghĩa |
|---|---|
| TP, true positive | Số mẫu lớp dương được dự đoán đúng là dương. |
| TN, true negative | Số mẫu lớp âm được dự đoán đúng là âm. |
| FP, false positive | Số mẫu lớp âm bị dự đoán nhầm là dương. |
| FN, false negative | Số mẫu lớp dương bị dự đoán nhầm là âm. |
| Positive class | Lớp sự kiện cần phát hiện, ví dụ một hành vi rủi ro xuất hiện. |
| Negative class | Lớp không có sự kiện cần phát hiện. |
| Decision threshold | Ngưỡng xác suất để đổi \(\hat{y}\) thành nhãn. Ví dụ threshold 0.5 nghĩa là \(\hat{y}\ge0.5\) được dự đoán là dương. |

## 4. Chỉ số đo lường

| Metric | Công thức hoặc ý nghĩa | Diễn giải |
|---|---|---|
| Accuracy | \((TP+TN)/(TP+TN+FP+FN)\) | Tỷ lệ dự đoán đúng. Có thể gây hiểu nhầm khi dữ liệu mất cân bằng. |
| Precision | \(TP/(TP+FP)\) | Trong các mẫu mô hình dự đoán dương, bao nhiêu mẫu thật sự dương. Precision cao nghĩa là ít false positive. |
| Recall, Sensitivity, TPR | \(TP/(TP+FN)\) | Trong các mẫu thật sự dương, mô hình bắt được bao nhiêu. Recall cao nghĩa là ít false negative. |
| Specificity, TNR | \(TN/(TN+FP)\) | Trong các mẫu thật sự âm, mô hình loại đúng bao nhiêu. Specificity cao nghĩa là ít false positive. |
| FPR | \(FP/(FP+TN)=1-\mathrm{specificity}\) | Tỷ lệ lớp âm bị báo động nhầm là dương. |
| FNR | \(FN/(FN+TP)=1-\mathrm{recall}\) | Tỷ lệ lớp dương bị bỏ sót. |
| F1-score | \(2TP/(2TP+FP+FN)\), tương đương \(2\cdot\frac{precision\cdot recall}{precision+recall}\) | Trung bình điều hòa của precision và recall. Hữu ích khi lớp dương hiếm và cần cân bằng FP/FN. Phụ thuộc threshold. |
| Balanced Accuracy | \((sensitivity+specificity)/2\) | Trung bình giữa khả năng bắt lớp dương và lớp âm. Hữu ích hơn accuracy khi dữ liệu mất cân bằng. Phụ thuộc threshold. |
| ROC-AUC, AUC | Diện tích dưới đường ROC, trong đó ROC vẽ TPR theo FPR khi quét mọi threshold. | Metric không phụ thuộc threshold, đo khả năng xếp hạng mẫu dương cao hơn mẫu âm. 0.5 gần ngẫu nhiên, 1.0 là hoàn hảo. Trong bài, AUC thường nghĩa là ROC-AUC nếu không ghi rõ. |
| PR-AUC, Average Precision | Diện tích/tóm tắt dưới đường Precision-Recall khi quét threshold. | Nhạy với mất cân bằng lớp. Khi lớp dương hiếm, PR-AUC thường phản ánh chất lượng phát hiện lớp dương rõ hơn ROC-AUC. |
| Brier score | \(\frac{1}{N}\sum_n(\hat{y}^{(n)}-y^{(n)})^2\) | Sai số bình phương trung bình của xác suất dự đoán. Thấp hơn là tốt hơn. Vừa phản ánh discrimination vừa phản ánh calibration. |
| ECE, Expected Calibration Error | \(\sum_b \frac{|B_b|}{N}\left|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\right|\), với \(B_b\) là bin xác suất. | Đo độ lệch giữa confidence dự đoán và tần suất đúng trong từng bin. Thấp hơn là tốt hơn. |
| Macro average | Trung bình các metric trên các tác vụ, bỏ qua giá trị không xác định nếu có. | Dùng để tóm tắt hiệu năng multi-task mà không để một tác vụ duy nhất chi phối toàn bộ báo cáo. |
| AUC drop | AUC full model trừ AUC khi bỏ một thành phần. | Dùng trong ablation. Drop lớn hơn gợi ý thành phần đó đóng góp nhiều hơn trong setting thí nghiệm. |
| MTL transfer ratio | \(AUC_{\mathrm{multi-task}}/(AUC_{\mathrm{single-task}}+\epsilon)\) | Tỷ lệ so sánh multi-task với single-task. Lớn hơn 1 gợi ý multi-task có lợi cho tác vụ đó. |
| Mean | Trung bình số học của metric qua seed, fold hoặc task. | Tóm tắt hiệu năng trung tâm. Cần đọc kèm độ lệch chuẩn hoặc CI. |
| Standard deviation, SD | Độ lệch chuẩn của metric qua seed hoặc fold. | SD cao nghĩa là kết quả kém ổn định hơn. |
| 95% CI | Khoảng tin cậy 95% quanh mean. | Cho biết độ bất định của ước lượng mean. Không phải khoảng chứa 95% cá thể trong dữ liệu. |
| p-value | Xác suất quan sát được chênh lệch ít nhất như hiện tại nếu giả thuyết không khác biệt là đúng. | p nhỏ, ví dụ < 0.05, thường được xem là bằng chứng thống kê cho khác biệt. Không đo kích thước hiệu ứng. |
| t-statistic | Thống kê kiểm định trong t-test. | Dấu và độ lớn phản ánh hướng và cường độ tương đối của chênh lệch so với sai số chuẩn. |

## 5. Cách đọc metric trong bài

- ROC-AUC và PR-AUC là metric không phụ thuộc threshold. Chúng phù hợp để so
  sánh khả năng xếp hạng xác suất giữa các mô hình.
- F1, balanced accuracy, sensitivity, specificity, precision, accuracy phụ
  thuộc threshold. Nếu dùng threshold được tune trên validation set, phải ghi
  rõ là validation-tuned threshold.
- ECE và Brier score đánh giá chất lượng xác suất, không chỉ đúng sai sau khi
  đặt threshold. Mô hình có AUC cao vẫn có thể calibration kém.
- Với dữ liệu mất cân bằng, không nên chỉ báo cáo accuracy. Nên đọc cùng
  ROC-AUC, PR-AUC, F1, balanced accuracy và calibration.
- Macro metric trong M2G-Net là trung bình qua các tác vụ. Khi báo cáo kết quả
  quan trọng, nên kèm cả metric từng tác vụ để tránh che khuất tác vụ yếu.
- Gate weights và Integrated Gradients là công cụ diễn giải mô hình, không phải
  bằng chứng nhân quả. Chúng cho biết mô hình sử dụng tín hiệu thế nào trong dữ
  liệu và cấu hình hiện tại.

## 6. Quy ước báo cáo khuyến nghị

Khi viết kết quả cho bài, nên dùng quy ước sau:

| Nội dung | Quy ước |
|---|---|
| Metric chính | Macro ROC-AUC cho discrimination, macro ECE/Brier cho calibration. |
| Metric phụ | PR-AUC, F1, balanced accuracy, sensitivity, specificity theo từng tác vụ. |
| Threshold | Báo cáo rõ threshold mặc định 0.5 hay validation-tuned threshold. |
| Calibration | Nếu dùng temperature scaling, báo cáo ECE trước và sau scaling. |
| Baseline comparison | Báo cáo mean, 95% CI và paired t-test khi có repeated runs. |
| Ablation | Báo cáo full model AUC, ablated AUC và AUC drop. |
| Out-of-site evaluation | Ghi rõ site intercept bật hay tắt, và site trong test có xuất hiện ở train không. |
| Interpretability | Tránh diễn giải gate/IG là nguyên nhân trực tiếp. Dùng cụm "model reliance" hoặc "relative contribution". |
| Data caveat | Với dữ liệu synthetic, giới hạn kết luận ở proof-of-concept và feasibility. |
