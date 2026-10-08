# Tiến độ AutoCheckout-CL

Cập nhật 08/10/2026. [Bàn giao](AGENT_HANDOFF.md) · [Kế hoạch](IMPLEMENTATION_PLAN.md) · [Chỉ mục triển khai](IMPLEMENTATION_INDEX.md) · [Nhật ký chi tiết](../timelines/history-2026-10-06-08.md).

| Phần | Trạng thái / bằng chứng | Việc tiếp theo |
|---|---|---|
| Tổ chức repo | Tài liệu vận hành, báo cáo có ngày, nguồn tham khảo và lịch sử tách riêng; notebook theo giai đoạn | Cập nhật chỉ mục cùng mỗi thay đổi |
| Khôi phục mã nền | 189 file khôi phục từ HEAD sau reset; code/runner/tests và kết quả cũ được giữ | Kiểm chứng model/Vast.ai trước sử dụng |
| Audit tài nguyên | 57.710 file decode/hash; 208 nền, 7.502 cutout/200 SKU | Tái sử dụng catalog đã kiểm tra |
| Real-only split | 22.494 train/1.503 val/6.003 test; native geometry; 9 gate đạt | Giữ test lock; nghiệm thu processor/loader |
| Visual và AI review | 222 card/gallery, Luna 10 + Sol 10 + Batch 200; gate API đạt, gate nội dung lịch sử FAIL | Đọc reports như lịch sử đề xuất |
| Resolution 202 | 202 ảnh/2.846 object/200 SKU; 9 giữ/193 tái ghép; owner-mask/RLE cho 2.721 object, task JSON và CSV acceptance PASS | Bản augmentation riêng; chưa tích hợp training |
| Scope còn lại | 20 pilot còn pending; toàn bộ 20.000 synth chưa phát hành lại | Xác định scope xử lý bổ sung riêng |
| PDP nền/processor | [Notebook 01](../../notebooks/modeling/01_processor_loader_pdp_acceptance.ipynb) đã tạo, chưa chạy; có gates processor/loader 640–800 và PDP smoke/resume | S0–S1: environment, loader/masking, bbox round-trip, loss/optimizer/resume |
| ConvNeXt-V2-Base | Chưa có adapter kiểm chứng | S2 sau PDP nền; feature maps/weights/forward/backward |
| Vast.ai training/eval | Chưa train/đánh giá trên protocol mới | Pilot EXP-B1/B2 rồi 5 task, calibration chỉ trên val |
| LoRA/K=3/FSA/freeze/overlap | Chưa triển khai các ablation mới | Sau baseline, mỗi thay đổi một run |

Báo cáo hiện hành cho scope đã xử lý: [Resolution 202](../data_preprocessing/reports/annotation-resolution-202-2026-10-08.md). CSV có **193 regenerate đã áp dụng, 9 giữ, 20 pending**; không dùng hàng đợi JSON ban đầu hoặc output cũ để suy ra pending hiện tại. Bản ảnh thật và holdout không đổi.

Kiểm chứng 08/10 cho lần tổ chức/commit: **78 CPU tests đạt** trên 11 suites native/resize/audit/group/split/task-config/taskcfg/visual/API/Batch/resolution. Command dùng `pytest.main` và multiprocessing `fork` trong kernel Python 3.14 local (4 deprecation warnings); xem các suites trong [runbook](../data_preprocessing/OPERATIONS.md). `tests/test_make_task_json.py` không collect được do thiếu `pycocotools`, tách khỏi nhóm regression đã đạt; cần chạy trong environment model đã chuẩn bị. Model/GPU/full suite chưa nghiệm thu; tests API/schema không phải accuracy GT.

Kiểm tra tổ chức tài liệu: **284 liên kết nội bộ hợp lệ**, **49 cell code tiền xử lý compile**, **8 config JSON hợp lệ**, `git diff --check` đạt. Kiểm tra read-only dataset thực tế đạt 202 ảnh/2.846 object/2.721 regenerated mask objects và CSV 193/9/20; không gọi API mới hoặc chạy training. Preview đầy đủ giữ local ở `runs/notebook_versions/`; bản notebook trong git giữ output nghiệm thu chữ/bảng.
