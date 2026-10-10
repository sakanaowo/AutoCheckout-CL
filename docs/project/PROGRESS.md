# Tiến độ AutoCheckout-CL

Cập nhật 10/10/2026. **Hoàn tất local foundation, adapter ConvNeXt và CLI/runner smoke ở notebooks 01–04**; [runtime acceptance](TRAINING_RUNTIME_ACCEPTANCE.md); [kết luận nghiệm thu](PDP_FOUNDATION_ACCEPTANCE.md) ghi phạm vi foundation trước các nghiệm thu adapter/runtime. [Bàn giao](AGENT_HANDOFF.md) · [Kế hoạch](IMPLEMENTATION_PLAN.md) · [Chỉ mục triển khai](IMPLEMENTATION_INDEX.md) · [Nhật ký chi tiết](../timelines/history-2026-10-06-08.md).

| Phần | Trạng thái / bằng chứng | Việc tiếp theo |
|---|---|---|
| Tổ chức repo | Tài liệu vận hành, báo cáo có ngày, nguồn tham khảo và lịch sử tách riêng; notebook theo giai đoạn | Cập nhật chỉ mục cùng mỗi thay đổi |
| Khôi phục mã nền | 189 file khôi phục từ HEAD sau reset; code/runner/tests và kết quả cũ được giữ | Kiểm chứng model/Vast.ai trước sử dụng |
| Audit tài nguyên | 57.710 file decode/hash; 208 nền, 7.502 cutout/200 SKU | Tái sử dụng catalog đã kiểm tra |
| Real-only split | 22.494 train/1.503 val/6.003 test; native geometry; 9 gate đạt | Giữ test lock; processor/loader đã đạt trên phạm vi notebook |
| Visual và AI review | 222 card/gallery, Luna 10 + Sol 10 + Batch 200; gate API đạt, gate nội dung lịch sử FAIL | Đọc reports như lịch sử đề xuất |
| Resolution 202 | 202 ảnh/2.846 object/200 SKU; 9 giữ/193 tái ghép; owner-mask/RLE cho 2.721 object, task JSON và CSV acceptance PASS | Bản augmentation riêng; chưa tích hợp training |
| Scope còn lại | 20 pilot còn pending; toàn bộ 20.000 synth chưa phát hành lại | Xác định scope xử lý bổ sung riêng |
| Dependency/Conda | [Conda sạch 10/10](CONDA_CLEAN_INSTALL_ACCEPTANCE.md) Python 3.10.22, 64 pins, pip check/imports, 105 tests, full ResNet/ConvNeXt CUDA 640 và notebook dry run PASS; bổ sung setuptools 81 để sửa pkg_resources | Cài cùng requirements trong env Python 3.10 riêng trên 4090; metadata trong real pilot; native kernel còn chờ |
| Handoff 09/10 | [Snapshot trước kiểm chứng model](HANDOFF_2026-10-09.md) giữ làm lịch sử; [kết luận hiện hành](PDP_FOUNDATION_ACCEPTANCE.md) | Dùng kết luận mới khi tiếp quản trạng thái model |
| PDP nền/processor | [Notebook 01](../../notebooks/modeling/01_processor_loader_pdp_acceptance.ipynb) đạt PASS_LIMITED_PDP_FOUNDATION, 11 gate, CPU tiny smoke; regression 27/27, resume 1/1; [bằng chứng](../timelines/model-foundation-2026-10-09.md) | Phần notebook foundation đã hoàn tất; full CUDA smoke đạt ở hàng dưới; runtime đã có notebook 04; native kernel còn chờ |
| Full PDP baseline CUDA | [Notebook 02](../../notebooks/modeling/02_full_pdp_baseline_acceptance.ipynb), run d967dc7b: PASS_FULL_PDP_CUDA_SMOKE trên RTX 3060, 640/800 batch 1 FP32; resume/state/predictions/frozen hashes đạt; peak allocated 3,49/4,22 GiB | Native kernel chưa đạt (PyTorch fallback); ConvNeXt và CLI/runner đã có nghiệm thu riêng; không chứng nhận convergence |
| Pretrained public | Script tải + notebook tự nhận cache path đã có; 3 files SenseTime detector tải thật, ~161 MB, pinned revision/SHA256, offline reuse đạt | Tái sử dụng cache/revision đã khóa; không cần tải lại để bắt đầu adapter |
| ConvNeXt-V2-Base | [Notebook 03](../../notebooks/modeling/03_convnext_v2_base_acceptance.ipynb) PASS 8/8 gates, 29 tests; run aeeb770b trên 3060, 640/800 peak 3,683/4,416 GiB | [Readiness](CONVNEXT_READINESS.md); native kernel, evaluator và pilot còn chờ |
| CLI/runner native | [Notebook 04](../../notebooks/modeling/04_training_runtime_acceptance.ipynb) PASS 10/10 gates, 78 tests; ResNet/ConvNeXt 640/800, step 1→4, checkpoint metadata/resume/predict-only/bbox native | [Báo cáo](TRAINING_RUNTIME_ACCEPTANCE.md); S6 đạt kỹ thuật; môi trường training rồi pilot |
| S6 calibration/evaluation | [Evaluation 01](../../notebooks/evaluation/01_count_calibration_acceptance.ipynb) PASS 7/7 gates; grid/NMS, val-only policy, test-only reload, oracle tắt; [báo cáo](COUNT_CALIBRATION_ACCEPTANCE.md) | Đã nghiệm thu CPU; sinh policy từ val checkpoint pilot thật |
| Training notebooks 01–03 | Đã tạo pilot/transition/5-task notebooks, headless executor, tmux, logging và resume; artifacts dưới data; [dry run](TRAINING_NOTEBOOK_DRY_RUN.md), [runbook](TRAINING_NOTEBOOK_RUNBOOK.md) | Cấu hình settings/data/weights trên 4090, chạy real pilot rồi chọn baseline_epochs từ val |
| Vast.ai training/eval | Chưa train/đánh giá trên protocol mới | Pilot EXP-B1/B2 rồi 5 task, calibration chỉ trên val |
| LoRA/K=3/FSA/freeze/overlap | Chưa triển khai các ablation mới | Sau baseline, mỗi thay đổi một run |

Báo cáo hiện hành cho scope đã xử lý: [Resolution 202](../data_preprocessing/reports/annotation-resolution-202-2026-10-08.md). CSV có **193 regenerate đã áp dụng, 9 giữ, 20 pending**; không dùng hàng đợi JSON ban đầu hoặc output cũ để suy ra pending hiện tại. Bản ảnh thật và holdout không đổi.

Kiểm chứng 08/10 cho lần tổ chức/commit: **78 CPU tests đạt** trên 11 suites native/resize/audit/group/split/task-config/taskcfg/visual/API/Batch/resolution. Command dùng `pytest.main` và multiprocessing `fork` trong kernel Python 3.14 local (4 deprecation warnings); xem các suites trong [runbook](../data_preprocessing/OPERATIONS.md). `tests/test_make_task_json.py` không collect được do thiếu `pycocotools`, tách khỏi nhóm regression đã đạt; cần chạy trong environment model đã chuẩn bị. Đó là snapshot 08/10; ngày 09/10 đã đạt limited foundation và full CUDA Task 1 smoke, full suite vẫn chưa xác nhận. Tests API/schema không phải accuracy GT.

Kiểm tra tổ chức tài liệu: **284 liên kết nội bộ hợp lệ**, **49 cell code tiền xử lý compile**, **8 config JSON hợp lệ**, `git diff --check` đạt. Kiểm tra read-only dataset thực tế đạt 202 ảnh/2.846 object/2.721 regenerated mask objects và CSV 193/9/20; không gọi API mới hoặc chạy training. Preview đầy đủ giữ local ở `runs/notebook_versions/`; bản notebook trong git giữ output nghiệm thu chữ/bảng.


## Bước tiếp theo

Theo [runtime acceptance](TRAINING_RUNTIME_ACCEPTANCE.md) và [S6 acceptance](COUNT_CALIBRATION_ACCEPTANCE.md):
**Chuẩn bị pilot local → chạy pilot EXP-B1/B2 trên RTX 4090 cùng budget và policy val → full teacher Task 1→2 smoke trên cùng máy → đủ 5 task**.
Người dùng quyết định ngày 10/10 bỏ notebook nghiệm thu môi trường Vast.ai riêng; metadata và số đo runtime ghi trong các run pilot/training.
Local notebooks 01–04 đã có bằng chứng; full training/E1/E2 chưa hoàn tất. Synthetic pending/LoRA/K=3/FSA/freeze bổ sung là track sau baseline.
