# Snapshot bàn giao training đến 13/10/2026

Lưu nội dung phần đầu handoff (điều chỉnh đường dẫn liên kết) trước cập nhật 20/10. Các trạng thái RUNNING/STOPPED bên dưới là lịch sử; baseline hai backbone đã hoàn tất 15/10. Xem [báo cáo cuối](../project/FINAL_BACKBONE_COMPARISON_2026-10-20.md).

# Bàn giao AutoCheckout-CL

**DỪNG THEO NGƯỜI DÙNG — 13/10/2026 09:56 UTC+7, chuẩn bị chuyển L4 → 4080 SUPER.**
ConvNeXt Task 1 đã COMPLETED; Task 2 dừng ở observed step 6.420, checkpoint resume
**6.414**, backup 6.365. ResNet chưa bắt đầu. Đã đọc/audit checkpoints, tensors hữu hạn,
đủ states; GPU trống, không còn training. Giữ toàn bộ artifacts, không tự launch lại.
[Điểm dừng, manifest và paths cần chuyển](../environment/TRANSFER_TO_4080_2026-10-13.md).
Settings đích `data/training/settings_baseline_4080.json`; máy 4080 chưa kiểm chứng.
Các mô tả RUNNING bên dưới là snapshot trước điểm dừng này.

**Cập nhật máy đích 12/10/2026: NVIDIA L4 trên GCP**, thay cho phương án 4080 SUPER.
[Setup L4](../environment/GCP_L4_SETUP_2026-10-12.md) đã đạt Python/CUDA/native kernel,
58 regression + 4 kernel tests; data đã tải/giải nén bằng rclone. Checkpoint step
31.550 và backup khớp hashes nguồn, ConvNeXt strict restore/640px forward trên L4 đạt.
Smoke resume trên bản checkpoint riêng đã đạt 2 updates đến 31.552: Lightning
restore đủ model/optimizer/scheduler/PDP memory, batch 8, finite losses, peak allocated
6,847 GiB; checkpoint baseline gốc giữ nguyên ở 31.550.
Settings: `data/training/settings_baseline_l4.json`; plan chỉ đổi roots, campaign/budget
giữ nguyên. **Baseline RUNNING trên L4 từ 12/10/2026 23:46:00 UTC+7**,
tmux `ac-baseline-l4`, ConvNeXt Task 1 resume từ 31.550 đã có updates mới.
Notebook 03 và bản execution autosave 15 giây, console tổng `tmux_baseline_l4.log`
trong campaign; `baseline_launch_l4.json` giữ receipt/PID attempt paths.
Session detached vẫn train khi SSH ngắt. Snapshot Vast.ai bên dưới là lịch sử.

**DỪNG THEO YÊU CẦU NGƯỜI DÙNG — 11/10/2026 15:08:40 UTC+7.**
Baseline ConvNeXt dừng ở Task 1, completed 11 epochs; checkpoint step 31.550
có thể resume trên GPU tương thích. **Giữ artifact**, không tự khởi chạy lại.
[Báo cáo dừng và chuyển GPU](../project/BASELINE_4090_STOP_REVIEW_2026-10-11.md) ghi validation,
audit, manifest và điều kiện tiếp tục. ResNet chưa bắt đầu, full baseline chưa hoàn tất.


Snapshot nền **11/10/2026**; trạng thái baseline/GPU mới nhất ở đầu tài liệu. Notebook modeling 01/02 đã đạt processor/loader và full pretrained ResNet CUDA smoke.
[ConvNeXt notebook 03](../project/CONVNEXT_READINESS.md) đã đạt 8/8 gates, 29 tests trên RTX 3060 (run aeeb770b).
CLI/runner và notebook 04 có triển khai/kiểm chứng riêng tại [báo cáo runtime](../project/TRAINING_RUNTIME_ACCEPTANCE.md).
Các run model local dùng PyTorch fallback. [S6 calibration/evaluation](../project/COUNT_CALIBRATION_ACCEPTANCE.md) đã nghiệm thu kỹ thuật bằng CPU, policy val và test-only reload.
[Pilot Task 1 RTX 4090](../project/TASK1_PILOT_4090_REVIEW.md) đã COMPLETED ngày 10/10 lúc 18:14:57 UTC+7: đủ hai models, 21.752 steps/model, 100/100 prototypes, policy val và test metrics. B1/B2 test M1 AP 26,88%/73,54%, cAcc 3,50%/30,57%.
[Notebook 02 transition smoke](../project/TASK_TRANSITION_4090_REVIEW.md) đã COMPLETED ngày 11/10 lúc 09:13:40 UTC+7: 8 steps/model, teacher frozen/16 calls, PPG 62/46, old prototypes/private prompt rows giữ nguyên. Chưa có prototype lớp mới; scope 16 ảnh/7 lớp mới chỉ đạt kỹ thuật. [Benchmark GPU baseline](../project/BASELINE_4090_CONFIGURATION.md) đã PASS 20 cases: batch/effective 8/8, workers 8, FP32 + TF32. Notebook 03 khởi chạy lúc 10:54:46 ngày 11/10 UTC+7, tmux `ac-baseline`, ConvNeXt Task 1–5 trước rồi ResNet Task 1–5. Budget khởi đầu 16 epochs/task; ưu tiên validation, chưa nghiệm thu hội tụ hoặc full baseline.

[Snapshot đầu ngày 09/10](../project/HANDOFF_2026-10-09.md) giữ bằng chứng dependency/branch tại lúc viết, trước các run model đã đạt;
đọc cùng kết luận hiện hành, không dùng các đoạn “chưa chạy” trong snapshot làm trạng thái model mới nhất.
Đo hai model cùng train Task 2 ở batch 8 cũng PASS: GPU memory used 19,30 GiB,
không OOM, nhưng B1/B2 chỉ 6,77/5,44 ảnh/s so với 16,80/10,94 khi chạy riêng.
Giữ notebook 03 chạy tuần tự; xem [benchmark đồng thời](../project/BASELINE_4090_CONFIGURATION.md).

[Khảo sát resource/cost 11/10](../project/GPU_COST_REVIEW_2026-10-11.md): batch 8 peak GPU-wide
9,59/9,72 GiB B1/B2; current price 0,4153 USD/h, DLPerf 49,57. Shortlist host 4090
khác giá 0,422–0,459 (DLPerf 97–123), hoặc hai GPU riêng 2×3090 giá 0,323 tổng /
2×4070 Ti SUPER giá 0,425 tổng để nghiên cứu parallel. Chưa thuê/chuyển instance,
chưa benchmark PDP trên các offers mới; không coi DLPerf là tốc độ PDP đã kiểm chứng.
[Nhật ký model](../timelines/model-foundation-2026-10-09.md) giữ chi tiết runs PASS/FAIL và các sửa lỗi.
Các làm rõ của người dùng có ưu tiên hơn handoff gốc.
Quyết định 10/10/2026: bỏ notebook nghiệm thu môi trường Vast.ai riêng; bước tiếp là chuẩn bị pilot local và thực chạy pilot/chuyển Task 1→2/baseline 5 task trên RTX 4090. Ghi metadata/VRAM/thời gian trong các run đó; fixture/debug nhỏ tiếp tục dùng CPU/3060 local.
Yêu cầu tiếp nối: mọi artifacts của training notebooks ghi dưới `data/training/` để upload; logging từng step/epoch/task và resume phải có evidence dry run local. [Notebooks 01–03](../../notebooks/training/README.md) dùng [settings](../../configs/training/notebooks.json), headless executor/tmux theo [runbook](../project/TRAINING_NOTEBOOK_RUNBOOK.md). Không tạo lại notebook environment riêng; real pilot 4090 đã hoàn tất trong campaign `vast4090-seed0-640-v1`, full training còn chờ. Dùng `data/training/settings_4090.json` cho notebook 02; notebook 03 mặc định đọc `data/training/settings_baseline_4090.json`, budget 16 đã khóa; checkpoint đang giữ để chuyển GPU.

[Conda cài sạch 10/10](../project/CONDA_CLEAN_INSTALL_ACCEPTANCE.md) đã đạt trên local RTX 3060: Python 3.10.22,
105 tests, ResNet/ConvNeXt 640 step 1→4 và notebook pilot CPU dry run. Requirements bổ sung **setuptools==81.0.0**:
lần cài mới kéo setuptools 84 khiến Lightning thiếu pkg_resources dù pip check đạt.
Artifacts ở `data/training/validation/conda_py310_clean_install_20261010/`; env thử còn trong /tmp.
Không chuyển project sang Python 3.9/3.12; trên máy thuê dùng env Conda Python 3.10 riêng.

