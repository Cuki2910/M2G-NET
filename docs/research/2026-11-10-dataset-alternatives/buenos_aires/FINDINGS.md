# Buenos Aires — kiểm tra file thật (2026-11-10)

Chưa train, chưa sửa src/config/manuscript. Chỉ tải và profile.

## Truy cập
- Catalog page: 502. API `package_show`: `web_fetch` bị cắt; `curl` mặc định bị WAF "Request Rejected"; browser 503 (lỗi server chập chờn).
- Một lần `curl` với User-Agent trình duyệt thông thường (không proxy, không loop) lấy được JSON metadata và 4 file CSV xe máy 2022-2025 cùng `diccionario_motos.xlsx` từ `cdn.buenosaires.gob.ar`. HTTP 200. Hành vi WAF chỉ chặn UA mặc định của curl. Metadata: license `cc-by`, `isopen: true` (package_show.json).
- Checksum: `raw/SHA256SUMS`.

## Cấu trúc
- CSV delimiter `;`, UTF-8 BOM, CRLF. Một dòng = một occupant (`id_ubicacion` 1 = conductor, 2 = pasajero, 3/4 = pasajero extra). Khóa vehicle = (year, `id_moto`); `id_moto` restart mỗi năm.
- 2023 có 100 dòng trống hoàn toàn (loại bỏ).
- Cột: dia_habil, turno, punto_observacion (intersection), zona (1 Centro, 2 Norte, 3 Oeste, 4 Sur), uso_vehiculo, tiene_patente, antiguedad, cantidad_ocupantes, proteccion_total, tipo_distraccion, sexo/edad_ocupante, proteccion_ocupante.
- Mã (dictionary): `proteccion_ocupante` 1 = Sí, 2 = No, 9 = không quan sát. `tipo_distraccion` 1 = Celular, 2 = Fuma, 3 = Come, 4 = Ninguno, 9 = không quan sát; chỉ quan sát cho conductor.

## Class support (chỉ conductor, n = 10346)
| Năm | n | Helmet = No | Helmet unobs (9) | Celular | Distr unobs (9) |
|---|---|---|---|---|---|
| 2022 | 2477 | 289 | 23 | 404 | 0 |
| 2023 | 2225 | 178 | 5 | 369 | 0 |
| 2024 | 2281 | 244 | 42 | 495 | 37 |
| 2025 | 3363 | 377 | 167 | 798 | 0 |
Tổng: helmet No 1088 / Yes 9021 / unobs 237; Celular 2066 / Ninguno 8035 / khác (fuma, come, unobs) 245.

## Group / split
- 78 intersection khác nhau; chỉ 21 xuất hiện cả 4 năm. 4 zona. Có `punto_observacion` để group-aware split (GroupKFold / leave-intersection-out) hoặc split theo năm.
- Trong dữ liệu thực tế `id_ubicacion` chỉ có 1 (conductor) và 2 (pasajero); mã 3/4 của dictionary không xuất hiện (`cantidad_ocupantes` tối đa 2).

## Điểm yếu còn lại
- Chỉ 2 endpoints (helmet, phone), không có red-light/turn-signal. Thu hẹp scope cần quyết định của tác giả.
- Helmet "unobserved" 2025 cao (167, ~5%); `proteccion_total` không khớp `proteccion_ocupante` ở 133 dòng conductor (96 dòng total=2 nhưng driver helmet=1, suy luận: pasajero không đội; chưa kiểm). Không dùng `proteccion_total` làm nhãn conductor.
- Quan sát tại đèn đỏ (đang chờ), tập 3 observation-day khác nhau; chưa rõ năm nào cùng ngày/operativo. Không có rider ID, không có ảnh.
- Mô tả dictionary thiếu nhãn rõ cho `antiguedad`, `tiene_patente` mã 9 (đoán "No observado"; chưa xác minh).
- Phương pháp lấy mẫu 2022-2025 chưa kiểm (xem mục dưới; chỉ đọc được thiết kế 2016). Chưa kiểm nhãn tại chính nguồn (không có inter-rater).

## Phương pháp lấy mẫu (tìm thêm, 2026-11-10)
- Trang báo cáo chính thức `buenosaires.gob.ar/gcaba_historico/investigaciones-de-seguridad-vial-0`: "Acceso denegado" (fetch). Không bypass thêm. Báo cáo 2022-2025: chưa đọc được.
- Tài liệu duy nhất đọc được: Vergara Parra (2016), thiết kế mẫu của đợt **2016**, không phải 2022-2025. https://www.aacademica.org/albano.blas.vergara/47.pdf (CC BY-NC-ND; fetch bị cắt trước mục "Coeficientes de ponderación").
- Nội dung đọc được (2016): quan sát không tham gia tại giao lộ có đèn tín hiệu; mẫu theo giai đoạn, phân tầng theo zona, 45 điểm quan sát chọn ngẫu nhiên; mục tiêu moto 1.925 ca (±3 điểm %, 95%), DEFF 1,2.
- Còn thiếu: áp dụng cho 2022-2025 chưa kiểm chứng. Các file CSV không có cột trọng số, nên prevalence thô không đại diện cho dân số CABA. Mục đích dự đoán/phân loại có thể ổn, nhưng không báo cáo như ước lượng prevalence.
