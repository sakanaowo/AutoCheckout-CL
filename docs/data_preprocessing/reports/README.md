# Báo cáo tiền xử lý theo lần chạy

Các báo cáo ghi trạng thái **tại thời điểm thực nghiệm**. Trạng thái hiện hành xem [chỉ mục tiền xử lý](../README.md); cách chạy lại xem [OPERATIONS](../OPERATIONS.md).

| Ngày | Báo cáo | Bằng chứng chính |
|---|---|---|
| 07/10 | [Audit đầu vào](data-audit-2026-10-07.md) | Inventory/COCO và protocol nguồn |
| 07/10 | [Nghiệm thu tài nguyên](preprocessing-acceptance-2026-10-07.md) | Decode/hash, nền/cutout, 222 case nghi vấn |
| 07/10 | [Split 5 task](split-100-4x25-2026-10-07.md) | Release native, checksum và holdout lock |
| 07/10 | [Visual review](annotation-review-2026-10-07.md) | 222 card/gallery và CSV ban đầu |
| 07/10 | [Pilot Luna](ai-review-pilot-2026-10-07.md), [audit nội dung](content-test-2026-10-07.md) | API/schema đạt, nội dung chưa đạt |
| 07/10 | [Pilot Sol](sol-pilot-2026-10-07.md) | 10 ảnh khác, gate nội dung FAIL |
| 08/10 | [Batch Sol 200](sol-batch-2026-10-08.md) | 200 responses; đề xuất chưa áp dụng tại thời điểm chạy |
| 08/10 | [Resolution 202](annotation-resolution-202-2026-10-08.md) | API reviewer cuối, 9 giữ/193 tái ghép, geometry/task/CSV PASS |

Báo cáo mới dùng tên `chu-de-YYYY-MM-DD.md`, ghi run ID, thời gian thực chạy có UTC offset, config/code revision, input/output/checksums, nghiệm thu và giới hạn. Artifacts lớn nằm trong `runs/`; báo cáo trỏ đến chúng thay vì sao chép gallery vào docs.
