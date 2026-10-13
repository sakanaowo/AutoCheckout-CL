# Tiến độ AutoCheckout-CL

**13/10/2026 09:56 UTC+7: baseline STOPPED theo người dùng để chuyển sang 4080 SUPER.**
ConvNeXt Task 1 COMPLETED (test M1 AP 83,70%, cAcc 73,16% cho 100 SKU).
Task 2 resume từ checkpoint **6.414**, epoch index 4; observed stop 6.420.
Checkpoint/backup/Task 1 parent đã audit, giữ đầy đủ artifacts; ResNet chưa bắt đầu.
[Bàn giao chuyển GPU](../environment/TRANSFER_TO_4080_2026-10-13.md).
Snapshot RUNNING L4 bên dưới là lịch sử trước khi dừng.

**Cập nhật 12/10/2026: baseline RUNNING trên NVIDIA L4 GCP từ 23:46:00 UTC+7.**
ConvNeXt Task 1 đã resume từ step 31.550 và có updates mới, losses hữu hạn.
Tmux `ac-baseline-l4` độc lập SSH; notebook 03/console/runner/task logs đang cập nhật.
[Setup, kiểm chứng resume và cách theo dõi](../environment/GCP_L4_SETUP_2026-10-12.md).
Đoạn dừng RTX 4090 bên dưới là snapshot lịch sử, đã được yêu cầu tiếp tục trên L4.

**DỪNG THEO YÊU CẦU NGƯỜI DÙNG — 11/10/2026 15:08:40 UTC+7.**
Baseline ConvNeXt dừng ở Task 1, completed 11 epochs; checkpoint step 31.550
có thể resume trên GPU tương thích. **Giữ artifact**, không tự khởi chạy lại.
[Báo cáo dừng và chuyển GPU](BASELINE_4090_STOP_REVIEW_2026-10-11.md) ghi validation,
audit, manifest và điều kiện tiếp tục. ResNet chưa bắt đầu, full baseline chưa hoàn tất.


Cập nhật **13/10/2026**. **Hoàn tất local foundation, adapter ConvNeXt và CLI/runner smoke ở notebooks modeling 01–04**; [runtime acceptance](TRAINING_RUNTIME_ACCEPTANCE.md); [kết luận nghiệm thu](PDP_FOUNDATION_ACCEPTANCE.md) ghi phạm vi foundation trước các nghiệm thu adapter/runtime. [Bàn giao](AGENT_HANDOFF.md) · [Kế hoạch](IMPLEMENTATION_PLAN.md) · [Chỉ mục triển khai](IMPLEMENTATION_INDEX.md) · [Nhật ký chi tiết](../timelines/history-2026-10-06-08.md).

**Real pilot Task 1 trên RTX 4090 đã COMPLETED lúc 18:14:57 UTC+7 ngày 10/10**,
campaign `vast4090-seed0-640-v1`: [review và hướng notebook 02](TASK1_PILOT_4090_REVIEW.md).
Hai models đủ 21.752 steps, checkpoint/predictions/policy val và 100 prototypes; B1/B2 test AP
26,88%/73,54%, cAcc 3,50%/30,57%.
**Notebook 02 transition smoke đã COMPLETED lúc 09:13:40 ngày 11/10**, 2 phút 43 giây;
[review](TASK_TRANSITION_4090_REVIEW.md) ghi teacher/PPG và old memory preservation, cùng giới hạn
0/25 prototypes mới và test subset không có GT lớp mới.
[Benchmark baseline GPU](BASELINE_4090_CONFIGURATION.md) đã PASS 20 cases, parent checkpoints giữ nguyên:
batch/effective 8/8, workers 8, FP32 + TF32. Người dùng ưu tiên hội tụ theo validation.
**Notebook 03 khởi chạy 11/10 lúc 10:54:46 UTC+7**, tmux `ac-baseline`,
ConvNeXt Task 1–5 trước rồi ResNet Task 1–5. 16 epochs/task là budget khởi đầu,
chưa chứng minh hội tụ; notebook/console/task logs được lưu trong lúc training.
Kiểm chứng thêm hai process cùng train Task 2, batch 8: 4 phép đo 60 giây PASS,
VRAM GPU-wide 19,30 GiB, không OOM; tốc độ từng model giảm mạnh. Giữ baseline chạy tuần tự.

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
| Training notebooks 01–03 | Pilot/transition COMPLETED; notebook 03 ConvNeXt Task 1 COMPLETED, Task 2 dừng trên L4, checkpoint 6.414; bản đầy đủ giữ trong data, Git giữ tóm tắt và traceback | Chuyển campaign hiện tại, kiểm chứng 4080, resume cùng budget khi được yêu cầu |
| GPU training/eval | [L4 setup](../environment/GCP_L4_SETUP_2026-10-12.md): 58 regression + 4 kernel tests, full-state resume đạt; Task 1 test M1 AP 83,70%, cAcc 73,16%; [transfer audit](../environment/TRANSFER_TO_4080_2026-10-13.md) đạt | Task 2–5 ConvNeXt và toàn bộ ResNet còn lại; giữ holdout/policy val và protocol |
| LoRA/K=3/FSA/freeze/overlap | Chưa triển khai các ablation mới | Sau baseline, mỗi thay đổi một run |

Báo cáo hiện hành cho scope đã xử lý: [Resolution 202](../data_preprocessing/reports/annotation-resolution-202-2026-10-08.md). CSV có **193 regenerate đã áp dụng, 9 giữ, 20 pending**; không dùng hàng đợi JSON ban đầu hoặc output cũ để suy ra pending hiện tại. Bản ảnh thật và holdout không đổi.

Kiểm chứng 08/10 cho lần tổ chức/commit: **78 CPU tests đạt** trên 11 suites native/resize/audit/group/split/task-config/taskcfg/visual/API/Batch/resolution. Command dùng `pytest.main` và multiprocessing `fork` trong kernel Python 3.14 local (4 deprecation warnings); xem các suites trong [runbook](../data_preprocessing/OPERATIONS.md). `tests/test_make_task_json.py` không collect được do thiếu `pycocotools`, tách khỏi nhóm regression đã đạt; cần chạy trong environment model đã chuẩn bị. Đó là snapshot 08/10; ngày 09/10 đã đạt limited foundation và full CUDA Task 1 smoke, full suite vẫn chưa xác nhận. Tests API/schema không phải accuracy GT.

Kiểm tra tổ chức tài liệu: **284 liên kết nội bộ hợp lệ**, **49 cell code tiền xử lý compile**, **8 config JSON hợp lệ**, `git diff --check` đạt. Kiểm tra read-only dataset thực tế đạt 202 ảnh/2.846 object/2.721 regenerated mask objects và CSV 193/9/20; không gọi API mới hoặc chạy training. Preview đầy đủ giữ local ở `runs/notebook_versions/`; bản notebook trong git giữ output nghiệm thu chữ/bảng.


## Bước tiếp theo

Theo [runtime acceptance](TRAINING_RUNTIME_ACCEPTANCE.md) và [S6 acceptance](COUNT_CALIBRATION_ACCEPTANCE.md):
**Pilot/transition/benchmark và ConvNeXt baseline Task 1 hoàn tất → L4 dừng ở Task 2 → chuyển artifacts hiện tại/kiểm chứng 4080 → resume từ 6.414 cùng budget khi được yêu cầu**.
Người dùng quyết định ngày 10/10 bỏ notebook nghiệm thu môi trường Vast.ai riêng; metadata và số đo runtime ghi trong các run pilot/training.
Local notebooks 01–04 và real Task 1 pilot đã có bằng chứng; E2/full training chưa hoàn tất. Synthetic pending/LoRA/K=3/FSA/freeze bổ sung là track sau baseline.
