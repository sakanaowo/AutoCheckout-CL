# Tiến độ AutoCheckout-CL

**Cập nhật 20/10/2026: baseline EXP-B1/EXP-B2 hoàn tất 5 task ngày 15/10, 19:28:27 UTC+7.** [Báo cáo kết quả](FINAL_BACKBONE_COMPARISON_2026-10-20.md) thay trạng thái dừng/chuyển máy ngày 13/10. Pilot/transition, foundation/ConvNeXt adapter, CLI/runner và S6 đã có bằng chứng riêng.

Final test B1/B2: AP50 **62,33%/84,40%**, AP[.50:.95] **51,55%/71,37%**, cAcc **2,58%/21,74%**; forgetting **6,63/8,67 điểm**. Đã tính lại counting 10 stage, calibration validation Task 5 và AP cuối; hashes/checkpoints khớp. 63 tests evaluation PASS. Hoàn tất budget 16 epochs/task chưa chứng minh hội tụ; LoRA/K=3/FSA/freeze bổ sung vẫn là ablation chưa chạy.

[Bàn giao](AGENT_HANDOFF.md) · [Kế hoạch](IMPLEMENTATION_PLAN.md) · [Chỉ mục triển khai](IMPLEMENTATION_INDEX.md) · [Snapshot training đến 13/10](../timelines/training-handoff-through-2026-10-13.md).

| Phần | Trạng thái / bằng chứng | Việc tiếp theo |
|---|---|---|
| Tổ chức repo | Tài liệu vận hành, báo cáo có ngày, nguồn tham khảo và lịch sử tách riêng; notebook theo giai đoạn | Cập nhật chỉ mục cùng mỗi thay đổi |
| Khôi phục mã nền | 189 file khôi phục từ HEAD sau reset; code/runner/tests và kết quả cũ được giữ | Tái sử dụng code đã kiểm chứng; bổ sung tests theo thay đổi mới |
| Audit tài nguyên | 57.710 file decode/hash; 208 nền, 7.502 cutout/200 SKU | Tái sử dụng catalog đã kiểm tra |
| Real-only split | 22.494 train/1.503 val/6.003 test; native geometry; 9 gate đạt | Giữ test lock; processor/loader đã đạt trên phạm vi notebook |
| Visual và AI review | 222 card/gallery, Luna 10 + Sol 10 + Batch 200; gate API đạt, gate nội dung lịch sử FAIL | Đọc reports như lịch sử đề xuất |
| Resolution 202 | 202 ảnh/2.846 object/200 SKU; 9 giữ/193 tái ghép; owner-mask/RLE cho 2.721 object, task JSON và CSV acceptance PASS | Bản augmentation riêng; chưa tích hợp training |
| Scope còn lại | 20 pilot còn pending; toàn bộ 20.000 synth chưa phát hành lại | Xác định scope xử lý bổ sung riêng |
| Dependency/Conda | [Kiểm tra 20/10](FINAL_BACKBONE_COMPARISON_2026-10-20.md): Conda pdp Python 3.10.22, Torch cu121, 64 pins, pip check/imports và 63 tests evaluation PASS | Driver local chưa hoạt động; metrics dùng CPU, GPU acceptance cũ không thay kiểm tra máy hiện tại |
| Handoff 09/10 | [Snapshot trước kiểm chứng model](HANDOFF_2026-10-09.md) giữ làm lịch sử; [kết luận hiện hành](PDP_FOUNDATION_ACCEPTANCE.md) | Dùng kết luận mới khi tiếp quản trạng thái model |
| PDP nền/processor | [Notebook 01](../../notebooks/modeling/01_processor_loader_pdp_acceptance.ipynb) đạt PASS_LIMITED_PDP_FOUNDATION, 11 gate, CPU tiny smoke; regression 27/27, resume 1/1; [bằng chứng](../timelines/model-foundation-2026-10-09.md) | Giữ evidence foundation; native kernel máy training đã có nghiệm thu riêng |
| Full PDP baseline CUDA | [Notebook 02](../../notebooks/modeling/02_full_pdp_baseline_acceptance.ipynb), run d967dc7b: PASS_FULL_PDP_CUDA_SMOKE trên RTX 3060, 640/800 batch 1 FP32; resume/state/predictions/frozen hashes đạt; peak allocated 3,49/4,22 GiB | Native kernel chưa đạt (PyTorch fallback); ConvNeXt và CLI/runner đã có nghiệm thu riêng; không chứng nhận convergence |
| Pretrained public | Script tải + notebook tự nhận cache path đã có; 3 files SenseTime detector tải thật, ~161 MB, pinned revision/SHA256, offline reuse đạt | Tái sử dụng cache/revision đã khóa; không cần tải lại để bắt đầu adapter |
| ConvNeXt-V2-Base | [Notebook 03](../../notebooks/modeling/03_convnext_v2_base_acceptance.ipynb) PASS 8/8 gates, 29 tests; run aeeb770b trên 3060, 640/800 peak 3,683/4,416 GiB | Adapter đã được dùng trong full baseline; xem báo cáo cuối |
| CLI/runner native | [Notebook 04](../../notebooks/modeling/04_training_runtime_acceptance.ipynb) PASS 10/10 gates, 78 tests; ResNet/ConvNeXt 640/800, step 1→4, checkpoint metadata/resume/predict-only/bbox native | Runtime đã dùng cho pilot/transition/full baseline; giữ nghiệm thu trước thay đổi mới |
| S6 calibration/evaluation | Policy val-only/test-only reload đã dùng cho full baseline; final thresholds B1/B2 0,42/0,35, NMS 0,45; tính lại khớp | Giữ oracle tắt; thay đổi policy phải chọn trên validation trong run riêng |
| Training notebooks 01–03 | Pilot/transition và baseline **cả hai backbone đủ 5 task**; campaign COMPLETED 15/10 | Giữ artifacts và outputs lịch sử; notebook báo cáo evaluation 02 tổng hợp kết quả |
| GPU training/eval | B2 4090→L4→4080 SUPER; B1 4080 SUPER. [Kết quả cuối](FINAL_BACKBONE_COMPARISON_2026-10-20.md) đã đối chiếu checkpoints/predictions/policies | Phân tích validation, hội tụ, ablation và thêm seeds; không còn task baseline chờ train |
| LoRA/K=3/FSA/freeze/overlap | Chưa triển khai các ablation mới | Sau baseline, mỗi thay đổi một run |

Báo cáo hiện hành cho scope đã xử lý: [Resolution 202](../data_preprocessing/reports/annotation-resolution-202-2026-10-08.md). CSV có **193 regenerate đã áp dụng, 9 giữ, 20 pending**; không dùng hàng đợi JSON ban đầu hoặc output cũ để suy ra pending hiện tại. Bản ảnh thật và holdout không đổi.

Kiểm chứng 08/10 cho lần tổ chức/commit: **78 CPU tests đạt** trên 11 suites native/resize/audit/group/split/task-config/taskcfg/visual/API/Batch/resolution. Command dùng `pytest.main` và multiprocessing `fork` trong kernel Python 3.14 local (4 deprecation warnings); xem các suites trong [runbook](../data_preprocessing/OPERATIONS.md). `tests/test_make_task_json.py` không collect được do thiếu `pycocotools`, tách khỏi nhóm regression đã đạt; cần chạy trong environment model đã chuẩn bị. Đó là snapshot 08/10; ngày 09/10 đã đạt limited foundation và full CUDA Task 1 smoke, full suite vẫn chưa xác nhận. Tests API/schema không phải accuracy GT.

Snapshot kiểm tra tổ chức tài liệu 08–09/10: **284 liên kết nội bộ hợp lệ**, **49 cell code tiền xử lý compile**, **8 config JSON hợp lệ**, `git diff --check` đạt. Kiểm tra read-only dataset thực tế đạt 202 ảnh/2.846 object/2.721 regenerated mask objects và CSV 193/9/20; không gọi API mới hoặc chạy training. Preview đầy đủ giữ local ở `runs/notebook_versions/`; bản notebook trong git giữ output nghiệm thu chữ/bảng.


## Bước tiếp theo

1. Dùng báo cáo cuối và notebook evaluation 02 làm bảng kết quả baseline; giữ snapshot/tham số/hash để tái lập.
2. Phân tích trên validation: thiếu đếm lớp mới, lỗi ảnh hard, confidence theo nhóm task và tác động top-1/NMS. Không chọn threshold bằng test.
3. Đối chiếu learning curves để đánh giá hội tụ; 16 epochs/task hiện là budget đã chạy, không phải kết luận tối ưu.
4. Thêm seeds và ablation độc lập nếu tiếp tục nghiên cứu; giữ baseline gốc. LoRA/K=3/FSA/freeze/synthetic là các hướng riêng, chưa được triển khai trong baseline này.
5. Khôi phục driver/kiểm chứng CUDA local trước công việc cần GPU. Evaluation từ predictions có thể chạy CPU trong Conda pdp.
