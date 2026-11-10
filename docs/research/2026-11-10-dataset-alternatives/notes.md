# coNotes — Dataset alternatives, discussion only

Ngày hội thoại: 2026-11-10. Người dùng: không bắt buộc MOTOR, cần dataset phù hợp; browse để bàn luận, chưa bắt đầu empirical. Không tải data/annotation/codebook files, không model/test, không sửa manuscript, không email.

## Bốn câu hỏi

1. Có nguồn giữ nguyên các endpoints và categorical context không?
2. Có public tabular roadside data tránh phải xây video pipeline không?
3. Những nguồn nào đổi estimand sang self-report hoặc chỉ cung cấp perception labels?
4. Access, rights, grouping và full-page reading đã xác nhận đến đâu?

## Search và nguồn

Queries lượt đầu: `motorcycle rider safety behaviour observational dataset helmet phone turn signal Hanoi data Mendeley`; `site:data.mendeley.com motorcycle risky riding behavior dataset observational`; `motorcyclist multiple risky behaviours dataset red light helmet mobile phone observational open data repository`; `Risky riding behaviours among motorcyclists in Malaysia data availability`; `site:data.buenosaires.gob.ar Factores de Riesgo Vial Motovehículos`.

Queries tiếp: `motorcycle helmet phone dataset observational data repository`; `motorcycle roadside observation helmet phone dataset open data csv Buenos Aires Ghana`; `Buenos Aires motovehículos diccionario distraccion`; `Mahasarakham University Social Network Survey data Mendeley`; `Motorcycle taxi programme increases safe riding data`; `motorcycle dataset multi-label tabular safety`.

### Buenos Aires — Relevamientos Anuales de Factores de Riesgo Vial

Official catalog: https://data.buenosaires.gob.ar/dataset/estudio-observacional-factores-riesgo-vial . Web_fetch 394cadb7-1 trả description: observational annual studies, motorcycle helmet và driver distraction, có resource links. Response bị cắt; chưa đọc đủ catalog. Official resource search preview có các trường `id_moto`, `id_ubicacion`, `punto_observacion`, `dia_habil`, `turno`, `uso_vehiculo`, `sexo_ocupante`, `edad_ocupante`, `proteccion_ocupante`, `tipo_distraccion`, missing description. Các semantics/code values/group meanings vẫn chưa kiểm chứng. Không gọi `id_ubicacion` là site ID: trong related auto dictionary nó là occupant position; motorcycle dictionary phải đọc riêng.

Resource discovered: https://data.buenosaires.gob.ar/lv/dataset/estudio-observacional-factores-riesgo-vial/resource/834215e5-1703-4955-880f-066740d90334 . Fetch 503 website error. Không lấy mirror locale như independent evidence. Raw CSV URL xuất hiện trong search results, không fetch/download. Không tính counts/year harmonization, license hay confirm phone endpoint. Có field distraction KHÔNG tự chứng minh phone use. Không có bằng chứng đủ red-light/turn-signal endpoints từ lượt này. Catalog là lead ưu tiên, chưa dataset được chọn.

### Malaysia — exact-topic studies, raw access unknown

https://eprints.qut.edu.au/205312/ và https://doi.org/10.1016/j.trf.2020.08.031 — search abstract mô tả helmet/red-light/phone/turn-signal cùng demographic/contextual factors. QUT fetch bị host blocked; publisher gặp CAPTCHA, không bypass. Chưa full-source read, không nâng counts/prevalence/license thành verified. Raw download/access/permission chưa chứng minh; chưa xác nhận on-request. Đây là study lead, chưa public dataset.

https://journalofroadsafety.org/article/142387-rush-hours-the-risky-behaviours-of-food-delivery-and-non-food-delivery-motorcycle-riders — web_fetch 636026b2-2 trả key findings liên quan helmets, red lights, phone stopped/moving, turn signals; full response bị cắt. Không full methods/data-availability read, không claim raw records public. Không ghép các study/site thành cùng một dataset.

### Hanoi — closest contract, known prior on-request evidence

https://doi.org/10.1016/j.aap.2024.107762 ; https://research.tudelft.nl/en/publications/not-the-same-how-delivery-ride-hailing-and-private-riders-roles-i/ . Lượt này institutional fetch bị cắt; dùng prior audit có chỉ vị trí data-availability và mapping tại `docs/OBSERVATIONAL_DATA_ACCESS.md`, đã Read. Theo hồ sơ cũ: data on request, access chưa có, draft email chưa gửi. Không xác nhận khả năng/tốc độ cấp quyền hiện tại; không đoán CSV synthetic là dữ liệu thật.

### Thailand — empirical self-report, not roadside events

https://data.mendeley.com/datasets/4zzd5wz45z/2 — official description fetch 39373b07-4 nêu social-media questionnaire về demographics/unsafe riding. Page ghi đang xem previous version. Full page bị cắt; không pin latest/license hoặc báo counts verified. Không xem self-reported frequencies như observed binary event labels.

https://escholarship.org/uc/item/22z5c3tx — fetch 636026b2-1 đọc hết landing content trả về: university student social-media survey, multiple risky behaviors và friendship nominations, raw survey/anonymized IDs được mô tả. Chỉ đọc landing abstract, không toàn thesis. Không dùng làm roadside data, không khẳng định sample adequacy/deep model phù hợp. Dataset DOI tìm thấy: https://doi.org/10.7910/DVN/K3BS8L ; Harvard landing 403, license/access chưa kiểm chứng. Search dataset abstract và thesis abstract có population counts khác nhau, chưa phân giải; không trích con số như một sample size thống nhất.

### Ghana — avoid Figshare equals dataset mistake

https://doi.org/10.6084/m9.figshare.33010838 ; https://api.figshare.com/v2/articles/33010838 . API fetch 636026b2-3 cho thấy file list có JPEG và figshare_url thuộc `/articles/figure/`; response bị cắt. Không full API read; đủ để không dùng DOI Figshare làm bằng chứng raw tabular dataset. Không download JPEG. Publisher-linked Figshare page 403. Raw data availability chưa kiểm chứng.

### Other leads not promoted

HELMET OSF https://osf.io/4pwj8/overview — fetch bị cắt; detection/tracking/helmet lead không chứng minh multi-outcome tabular data. Mendeley accident-severity và Bogotá crash datasets: search only, không mở rộng thành candidate tốt cho risky behavior vì thay câu hỏi và crash-conditioned population. Không khẳng định các nguồn không có giá trị cho nghiên cứu khác.

## Giới hạn đọc bắt buộc

Chưa hoàn tất full-page reading cho các candidate chính do tool truncation/access failures. Do đó đây là discovery/shortlist, KHÔNG completed source audit. Không đưa số mẫu, license hoặc release date vào kết luận verified. Một landing abstract được đọc đủ không thay full study/codebook. Dừng access-blocked routes; không đổi config, login, headers/proxy hay cài browser CLI.

## Nhận định phương pháp — đề xuất, không fact dataset

- Ưu tiên authorized tabular observations có nhiều outcomes trên cùng observation unit, contextual predictors, codebook và grouping.
- Nếu chỉ có helmet/distraction thì thu hẹp endpoints là redesign cần chốt; không chế thêm targets từ composite/redundant label.
- Không dùng `proteccion_total`/helmet-derived aggregates làm input dự đoán helmet; không dùng distraction-derived fields làm input cùng target. Codebook phải chứng minh vai trò trước.
- Không random-split occupant rows cùng motorcycle hoặc video frames; grouping tùy actual ID semantics. Không hứa có site-out/year-out nếu chưa kiểm data coverage/schema.
- Nếu cho phép đổi hẳn domain, cần tìm riêng benchmark multi-task tabular có outcomes thực; không ngầm hiểu câu 'không nhất thiết MOTOR' là đồng ý bỏ motorcycle safety.

## So sánh candidate khác — cập nhật trong session 2026-11-10

Phạm vi: (1) self-report nhiều hành vi; (2) helmet perception; (3) MOTOR và các nguồn chưa chứng minh raw access. Không tải dataset mới, không train.

- Thailand Mendeley v2: https://data.mendeley.com/datasets/4zzd5wz45z/2 . Fetch thấy mô tả social-media questionnaire và danh sách CSV, questionnaire PDF, description PDF. Đây là previous version; latest/license chưa kiểm chứng. Description PDF https://data.mendeley.com/public-files/datasets/4zzd5wz45z/files/293689f6-ace6-47f0-ad97-d3a3af68acc0/file_downloaded đọc được nhóm covariates demographics/exposure và lịch sử violations/accidents/injury, nhưng bị cắt khi tới MRBQ items. Chưa chứng minh đủ bốn targets hiện tại. Dự đoán self-report khác dự đoán hành vi quan sát; không lấy crash/injury history làm input mặc định khi chưa khóa thời gian và mục tiêu.
- MSU: https://escholarship.org/uc/item/22z5c3tx . Đọc hết landing abstract: self-report helmet, phone, multiple occupants, speeding, cùng demographics, decision preferences và friendship IDs. Phù hợp câu hỏi social-network/risk behavior, không phải road-site context. Raw access/license vẫn chưa xác minh; counts không dùng do bất nhất đã ghi trước. Friendship linkage cần cân nhắc leakage/dependence khi thiết kế evaluation.
- HELMET: https://osf.io/4pwj8/overview . Fetch chỉ trả navigation shell, không đủ nội dung dataset. Search và https://github.com/LinHanhe/Helmet_use_detection mô tả hướng detection/tracking/helmet; fetch GitHub cũng bị cắt trước README. Chưa xác nhận đầy đủ schema/license. Không gọi đây là nguồn đã đủ bốn hành vi hoặc ready-to-use tabular.
- MOTOR: https://huggingface.co/datasets/varunpaturkar/MOTOR . Fetch lần này bị cắt trước nội dung schema; không nâng claims mới từ snippet. Giữ kết luận prior audit: behavior/legality chưa tương đương bốn outcomes, rights/release/group mapping chưa đóng.
- Ghana Figshare: giữ trạng thái prior audit, chưa chứng minh raw tabular records; DOI figure không đủ chứng nhận data release.

Đính chính nhận định trước: hai tasks vẫn là multi-task; giảm từ bốn xuống hai không tự chứng minh giả thuyết gating yếu hay model không hữu ích. Support có positive examples không đủ kết luận train/evaluation khả thi. Thiếu weights không tự chứng minh mẫu không đại diện; chỉ chưa đủ cơ sở suy rộng prevalence. Chưa có benchmark chứng minh Buenos Aires chạy kịp deadline hoặc phù hợp kiến trúc hiện tại.

Khuyến nghị thiết kế (không phải kết quả thực nghiệm): nếu cần public tabular mà giữ nhiều behavior targets, Thailand Mendeley đáng audit tiếp trước, nhưng phải chấp nhận estimand self-report. Nếu giữ đúng observational task hiện tại, các candidate này chưa thay thế Hanoi/Malaysia; Buenos Aires vẫn là empirical candidate thu hẹp, không phải xác nhận model fit.
