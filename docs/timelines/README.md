# Nhật ký thực nghiệm

Các báo cáo lưu mục đích, thời điểm chạy, cấu hình, artifacts và gate nghiệm thu. Trạng thái remote của Batch xem trong submission.json hoặc chạy cell theo dõi notebook.

| Ngày | Công việc | Báo cáo |
|---|---|---|
| 07/10/2026 | Audit/split 5 task và review nhãn synth | [Tiền xử lý](../data_preprocessing/README.md) |
| 07/10/2026 | Pilot GPT-6.1 Sol trên 10 ảnh mới | [Pilot Sol](../data_preprocessing/reports/sol-pilot-2026-10-07.md) |
| 08/10/2026 | Batch GPT-6.1 Sol, đúng 200 ảnh, 2 deferred | [Batch và nghiệm thu notebook](../data_preprocessing/reports/sol-batch-2026-10-08.md) |
| 08/10/2026, 13:16–13:34 UTC+7 | API chốt và pipeline xử lý thật 202 ảnh | [Resolution 202 / notebook 04](../data_preprocessing/reports/annotation-resolution-202-2026-10-08.md) |
| 08/10/2026 | Tổ chức docs, chỉ mục code/config/notebook và runbook; commit theo chức năng | [Chỉ mục triển khai](../project/IMPLEMENTATION_INDEX.md), [vận hành](../data_preprocessing/OPERATIONS.md), [kiểm chứng](../project/PROGRESS.md) |
| 09/10/2026 | Sửa dependency pins theo log Conda; tạo handoff, model notebook chưa chạy | [Handoff agent tiếp theo](../project/HANDOFF_2026-10-09.md), [môi trường](../environment/README.md) |
| 09/10/2026, 15:12:32–15:13:43 UTC+7 | Notebook 01 limited PDP foundation PASS trên CPU/máy 3060; tạo notebook 02 full CUDA, chưa chạy | [Bằng chứng và phạm vi model](model-foundation-2026-10-09.md) |
| 09/10/2026, 16:12:09–16:12:57 UTC+7 | Full pretrained PDP CUDA 640/800 smoke PASS trên RTX 3060, PyTorch fallback; sửa device sau Lightning teardown | [Báo cáo full baseline](model-foundation-2026-10-09.md#full-baseline-cuda-đạt-trên-rtx-3060) |
| 09/10/2026 | Khép phạm vi notebook processor/loader/PDP nền; cập nhật docs hiện hành và thứ tự bước tiếp theo | [Kết luận nghiệm thu](../project/PDP_FOUNDATION_ACCEPTANCE.md) |

[Nhật ký chi tiết 06–08/10](history-2026-10-06-08.md) · [Tiến độ dự án](../project/PROGRESS.md) · [Notebook review](../../notebooks/data_preprocessing/03_annotation_review.ipynb).
