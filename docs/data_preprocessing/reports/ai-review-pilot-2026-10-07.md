# Pilot OpenAI review nhãn — 07/10/2026

Đã tích hợp vào [notebook 03](../../../notebooks/data_preprocessing/03_annotation_review.ipynb), phần **6–9**. Key xác thực được, model `gpt-6-luna` có trong danh sách model của tài khoản. Các cell API chạy độc lập với phần CUDA/PyTorch của notebook và dùng artifacts visual đã nghiệm thu.

**Kết luận: API/schema hoạt động; chất lượng nhận xét chưa đạt để chạy tiếp 212 case.** Không sửa annotation nguồn hoặc CSV quyết định của người review. Pilot dùng **10 ảnh duy nhất**, thử ba cấu hình trên cùng các ảnh đó; không coi JSON hợp lệ hoặc confidence tự báo là nhãn đã đúng.

**Kiểm thử nội dung bổ sung:** [bảng đối chiếu 10 case](content-test-2026-10-07.md) trong phần 10 notebook phân biệt mô tả có cơ sở với đề xuất đủ căn cứ. Có 7 đề xuất phù hợp cho sàng lọc và 3 case cần xem lại, bổ sung `(2,7318)` vì vùng cạnh trái bị lon che. Đây không phải accuracy nhãn; gate vẫn FAIL, không gọi thêm API.

## Các lần thử và chi phí

| Prompt / reasoning | Thời điểm API thực chạy, UTC+7 | Request | Response completed | Chi phí ước tính USD |
|---|---|---:|---:|---:|
| v1 / none | 20:30:22–20:30:56 | 10 | 10 | 0.003670125 |
| v2 / none | 20:37:23–20:37:59 | 10 | 9 | 0.004724250 |
| v3 / low | 20:42:39–20:43:40 | 10 | 10 | 0.006059750 |
| Tổng, trên 10 ảnh | | **30** | **29** | **0.014454125** |

Chi phí tính từ usage từng receipt, gồm input, cached input, cache writes và output (kể cả response incomplete), theo giá Standard short-context của [OpenAI Docs](https://developers.openai.com/api/docs/models/gpt-6-luna). Đây là ước tính, không phải hóa đơn. Reparse/chạy lại cùng request fingerprint dùng cache và không gọi API lại.

v1 quá dễ giữ mọi bbox vì chứa được vật thể. Một response có bản nháp và message cuối; parser đã được sửa bằng regression test để đọc message cuối, reparse receipt cũ không gửi lại ảnh. v2 thêm geometry metadata và quy ước bbox visible nhưng một response bị incomplete ở giới hạn output; không cắt JSON ở đầu để giả là response hoàn tất. v3 dùng reasoning low, phân biệt khoảng trống góc của bbox xoay với mở rộng cực trị; cả 10 response cuối hợp lệ.

## Nhận xét cuối của model và audit nội dung

| Đề xuất của model | Số case |
|---|---:|
| regenerate | 6 |
| keep_with_justification | 3 |
| pending | 1 |

Audit độc lập bằng 10 PNG, cutout gốc và crop native: **FAIL** ở ít nhất hai điểm cần tính thận trọng:

- `(4,15271)`: model kết luận chắc chắn bbox thừa phía trên. Cutout có tay cầm mảnh ở góc trên trái, còn ảnh native có vệt tối mảnh sát vật che phía trước. Chưa đủ căn cứ loại khả năng đây là fragment hợp lệ mà polygon bỏ sót; không chấp nhận kết luận chắc chắn chỉ từ phần thân lớn.
- `(4,9534)`: model đề xuất tái sinh như lỗi rõ ràng, dù bbox/polygon chỉ lệch **8 px** ở cạnh dưới, bằng khoảng **2,03%** chiều cao bbox, trong cảnh có che khuất. Với contour giản lược và owner-mask gốc thiếu, bằng chứng chưa đủ để kết luận chắc chắn.

Ảnh QA/crop native được giữ ở `data/processed/synthetic_annotation_review_v1/ai/gpt-6-luna_v1/qa/`, có đường dẫn trong audit JSON.

Đây là audit mức phù hợp của nhận xét với bằng chứng hiện có, không phải chứng nhận GT pixel cho 10 case hay phép đo accuracy trên toàn dataset. Tất cả đề xuất vẫn tách khỏi nhãn đã nghiệm thu. **212 ảnh còn lại chưa gửi.**

## Cách dùng phần API trong notebook

1. Giữ key trong `.env` ở root hoặc biến môi trường `OPENAI_API_KEY`. `.env` đã được Git ignore; [.env.example](../../../.env.example) chỉ chứa tên biến trống.
2. Chạy cell cấu hình phần 6 với `AI_RUN_PILOT=True`; chạy cell phần 7. Cùng input/prompt/config dùng receipt cache. Sửa prompt/reasoning/ảnh làm đổi fingerprint và tạo request mới có phí.
3. Xem 10 kết quả cùng ảnh. Phần chọn case hiện đề xuất AI và quyết định manual riêng; output đã lưu có ví dụ.
4. `pilot_quality_audit.json` phải chứa audit nội dung cho đúng fingerprints, không chỉ PASS schema. Hiện file này ghi `passed=false`; continuation bị chặn. Sau khi có đánh giá mới đạt, mới đặt `AI_RUN_REMAINING=True` và chạy phần 8.
5. Cell cuối ghi trạng thái `PILOT_QUALITY_FAILED_NO_CONTINUATION`, usage, chi phí và thời gian. Cache/result/audit ở `data/processed/synthetic_annotation_review_v1/ai/gpt-6-luna_v1/`; snapshots theo run ID ở `runs/data_preprocessing/ai_review/`.

Snapshot cấu hình hiện tại: [annotation_review_ai_v1.json](../../../configs/data/annotation_review_ai_v1.json) (workflow v1, prompt v3). HTTP client dùng thư viện chuẩn Python, đọc/thu cutout dùng Pillow; không cần OpenAI SDK/dotenv. Các cell CUDA người dùng thêm được giữ nguyên; chúng chưa được nghiệm thu trong phiên API này.

## Kiểm chứng và bước tiếp theo

**48 CPU tests đạt**, gồm 11 ca API review: chọn pilot có task/severity coverage; hai image input và schema không cho tuyên bố mask đã xác minh; reject sai IDs/incomplete response; cache không rebill và đổi ảnh invalidates; dotenv ưu tiên environment; lỗi API không echo key; continuation cần audit đúng pilot; chi phí cached input; final-message parsing; geometry metadata; reasoning low.

Đã thực gọi API trên 10 ảnh; kiểm chứng receipt/usage, reparse cache và output notebook. Key không nằm trong notebook/config/log; source và CSV manual giữ checksum. Không chạy model PDP/GPU hoặc full suite PyTorch trong bước này.

Tiếp theo cần đánh giá các case fragment mảnh/near-threshold bằng reviewer hoặc model mạnh hơn, rồi chạy lại audit pilot trước khi mở rộng. Mask/bbox repair và nghiệm thu nguồn synth là bước riêng; thay threshold/thu bbox theo polygon để làm hết cảnh báo chưa được thực hiện.
