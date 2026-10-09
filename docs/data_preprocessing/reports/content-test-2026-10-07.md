# Kiểm thử nội dung AI review — 07/10/2026

**Kết luận: chưa đạt để chạy tiếp 212 ảnh.** Đã đối chiếu 10 nhận xét với crop native, cutout, bbox/polygon và edge zoom. **7 đề xuất phù hợp để sàng lọc; 3 kết luận mạnh hơn bằng chứng.** Đây là mức tương thích với rubric đối chiếu, không phải accuracy nhãn/bbox/mask.

Notebook: [03_annotation_review.ipynb](../../../notebooks/data_preprocessing/03_annotation_review.ipynb), **phần 10**. Thời gian cell nghiệm thu **2026-10-07T22:15:30+07:00 → 2026-10-07T22:15:31+07:00**, run `3e68a2cc-2e2e-4ba8-adcd-4189c5d739e4`. Chỉ đọc phản hồi cached: **0 API calls mới**, không thêm chi phí; không sửa source hoặc CSV manual.

## Đối chiếu từng case

| Task nguồn / ann | Chênh cạnh bbox–polygon | Đề xuất model | Kết quả nội dung |
|---|---|---|---|
| 1 / 30575 | trái 42 px | regenerate | Phù hợp để sàng lọc |
| 1 / 42502 | trên 7 px | keep_with_justification | Phù hợp để sàng lọc |
| 2 / 7318 | trái 136 px | regenerate | Cần review thêm; kết luận quá chắc |
| 2 / 41775 | trên 12 px | keep_with_justification | Phù hợp để sàng lọc |
| 3 / 38483 | phải 109 px | regenerate | Phù hợp để sàng lọc |
| 3 / 58188 | dưới 11 px | keep_with_justification | Phù hợp để sàng lọc |
| 4 / 15271 | trên 123 px | regenerate | Cần review thêm; kết luận quá chắc |
| 4 / 9534 | dưới 8 px | regenerate | Cần review thêm; kết luận quá chắc |
| 2 / 34269 | trái 45 px | pending | Phù hợp để sàng lọc |
| 3 / 49657 | trên 20 px | regenerate | Phù hợp để sàng lọc |

Ba case cần xem lại:

- **2 / 7318:** cạnh trái lệch 136 px nhưng vùng đó bị lon che và có mảnh đỏ cạnh lon. Mô tả hướng lệch đúng; kết luận không có pixel mục tiêu chưa đủ căn cứ. Model vẫn đề xuất regenerate, confidence 0,94, needs_human=false.
- **4 / 15271:** phần thân đen được mô tả đúng; model chưa xử lý đủ khả năng tay cầm mảnh từ cutout nằm dọc vật che. Confidence 0,94 chưa thay thế bằng chứng về ownership của pixel.
- **4 / 9534:** chênh đáy 8 px, khoảng 2,03% chiều cao bbox. Với che khuất và contour giản lược, kết luận lỗi rõ ràng cần tái sinh là mạnh hơn bằng chứng.

Bảy case còn lại có nhận xét phù hợp để chọn bước kiểm tra tiếp. Ba đề xuất regenerate trong nhóm này cũng chưa chứng nhận bbox sai theo owner-mask hoặc mask sửa đã đúng; không tự dùng các điểm này làm nhãn đã nghiệm thu.

## Cách kiểm thử và giới hạn

- Bộ đối chiếu [annotation_review_content_reference_v1.json](../../../configs/data/annotation_review_content_reference_v1.json) do Codex lập sau khi xem 10 crop native, cutout và zoom cạnh; không phải mask GT do người gán nhãn xác minh.
- Khóa case/source fingerprint và SHA-256 nội dung nhận xét để tránh dùng lại điểm cho output khác. Thay model/prompt/output phải đối chiếu mới.
- Chấm riêng mô tả, đề xuất sàng lọc và cách xử lý bất định. Không lấy số token/schema/confidence tự báo làm bằng chứng semantic correctness.
- Mô tả: 7 SUPPORTED / 3 PARTIAL. Đề xuất tương thích rubric: 7/10; ba case cần xem lại đều chưa bật needs_human. Owner-mask gốc thiếu, nên ground_truth_accuracy để null và label_repairs_accepted=0.
- Từ v1 sang v3 có 7/10 quyết định thay đổi. Prompt, rubric và reasoning cũng thay đổi; đây là so sánh lịch sử cấu hình, không phải đo repeatability cùng prompt hay accuracy toàn dataset.

## Artifact và cách xem

- Report HTML tại `runs/data_preprocessing/content_review/3e68a2cc-2e2e-4ba8-adcd-4189c5d739e4/content_report.html`: mở từng case để xem ảnh native/edge zoom, nhận xét gốc và ghi chú đối chiếu.
- Cùng thư mục có summary.json, case_content_checks.json, geometry.json, prompt_decision_comparison.json và source_asset_sha256.json.
- Chạy riêng hai cell phần 10 của notebook; không cần key, OpenAI SDK hay PyTorch/GPU. Cell cuối ghi thời điểm/status và kiểm tra input/source/CSV không đổi.
- 11 tests cơ học API/cache/parser/gate đã chạy lại và đạt. Những tests này không chứng minh nội dung model đúng; nội dung được đối chiếu trực quan như bảng trên. AI DevKit lint offline vẫn thiếu package trong npm cache.

Gate pilot được giữ FAIL và bổ sung case 2/7318 vào nhóm cần xem lại; chưa gửi 212 ảnh còn lại. Bước tiếp theo là xác minh các fragment/ownership bằng mask hoặc reviewer, hoặc đánh giá model mạnh hơn với bộ đối chiếu này trước khi mở rộng.
