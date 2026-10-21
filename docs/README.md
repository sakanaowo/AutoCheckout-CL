# Tài liệu AutoCheckout-CL

Cập nhật **20/10/2026**. Protocol **5 task 100+4×25, seed 0**. Baseline hai backbone đã hoàn tất ngày 15/10 trên campaign `vast4090-seed0-640-v1`; metrics cuối được tính lại từ predictions ngày 20/10. [Báo cáo kết quả](project/FINAL_BACKBONE_COMPARISON_2026-10-20.md) là nguồn trạng thái mới nhất; các báo cáo dừng/chuyển GPU 11–13/10 là lịch sử.

| Điểm vào | Mục đích |
|---|---|
| [Synthetic v1 lưu trên Drive, xóa local](data_preprocessing/reports/synth-v1-retirement-2026-10-21.md) | Quy ước v1/v2, inventory/checksums, phụ thuộc lịch sử và hướng dẫn restore; v2 chưa sinh |
| [Quyết định ảnh ghép v2](project/synthetic-data-v2-decisions-2026-10-21.md), [kế hoạch](project/synthetic-data-v2-plan-2026-10-20.md), [thiết kế](project/synthetic-data-v2-design-2026-10-20.md) | Nhánh `feat/synthetic-data-v2`: D1–D21 (val/test RPC gốc, 71.642 ảnh, camera theo hình dạng, cổng cAcc-M2); kiểm kê → cắt bổ sung camera 0 → pilot → full release → loader; chưa triển khai |
| [Báo cáo cuối hai backbone](project/FINAL_BACKBONE_COMPARISON_2026-10-20.md) | Protocol, full metrics, forgetting/counting, biểu đồ, đối chiếu PDP và evidence kiểm tra lại |
| [Bàn giao](project/AGENT_HANDOFF.md), [snapshot đầu ngày 09/10](project/HANDOFF_2026-10-09.md) | Bàn giao hiện hành và snapshot lịch sử trước kiểm chứng model |
| [Kết luận processor/loader/PDP foundation](project/PDP_FOUNDATION_ACCEPTANCE.md) | Phạm vi notebook đã hoàn tất, evidence CPU/CUDA, giới hạn và 5 bước tiếp theo |
| [Chỉ mục triển khai](project/IMPLEMENTATION_INDEX.md) | Code mới, config, notebook, tests và đầu ra của từng phần |
| [Kế hoạch](project/IMPLEMENTATION_PLAN.md), [tiến độ](project/PROGRESS.md) | Phần đã đạt và thứ tự công việc tiếp theo |
| [ConvNeXt và train readiness](project/CONVNEXT_READINESS.md) | Notebook 03 đã đạt CUDA; các gate training còn thiếu |
| [CLI/runner acceptance](project/TRAINING_RUNTIME_ACCEPTANCE.md) | Notebook 04: ResNet/ConvNeXt, resolution, checkpoint, resume và bbox native |
| [S6 calibration/evaluation acceptance](project/COUNT_CALIBRATION_ACCEPTANCE.md) | Evaluation 01: policy val lưu riêng, test-only reload, oracle tắt, không cần GPU |
| [Training notebook runbook](project/TRAINING_NOTEBOOK_RUNBOOK.md), [dry run](project/TRAINING_NOTEBOOK_DRY_RUN.md) | Pilot/transition/baseline 5 task; artifacts data, tmux/headless, logs và resume |
| [Pilot Task 1](project/TASK1_PILOT_4090_REVIEW.md), [transition Task 1→2](project/TASK_TRANSITION_4090_REVIEW.md), [cấu hình baseline 4090](project/BASELINE_4090_CONFIGURATION.md) | Kết quả real run, giới hạn smoke và 20 benchmark cases để chọn batch/TF32; epochs chờ validation |
| [Dependencies/môi trường](environment/README.md), [Conda cài sạch](project/CONDA_CLEAN_INSTALL_ACCEPTANCE.md) | Python 3.10, setuptools/pkg_resources, 105 tests và GPU/notebook checks trong env mới |
| [Setup L4 GCP](environment/GCP_L4_SETUP_2026-10-12.md), [chuyển sang 4080 SUPER](environment/TRANSFER_TO_4080_2026-10-13.md) | 62 tests, kernel/resume thật, Task 1 metrics, checkpoint Task 2 và paths/hashes cần chuyển |
| [Báo cáo dừng baseline 4090 ngày 11/10](project/BASELINE_4090_STOP_REVIEW_2026-10-11.md) | Snapshot lịch sử trước L4: 11 epochs ConvNeXt, val AP 82,83%, checkpoint step 31.550 |
| [Tài nguyên và chi phí GPU Vast.ai](project/GPU_COST_REVIEW_2026-10-11.md) | Snapshot ngày 11/10: VRAM/CPU/RAM, giá/offers và ngưỡng hiệu quả theo throughput |
| [Tiền xử lý](data_preprocessing/README.md), [hướng dẫn vận hành](data_preprocessing/OPERATIONS.md) | Cách đọc kết quả, chạy lại và tiếp tục pipeline |
| [Schema](data_preprocessing/formats.md) | Tọa độ, COCO, task mapping và artifacts |
| [Báo cáo thực nghiệm](data_preprocessing/reports/README.md) | Kết quả theo ngày/run; giữ nguyên kết luận tại thời điểm chạy |
| [Nhật ký](timelines/README.md) | Mốc thực nghiệm và lịch sử triển khai |
| [Tổng quan kiến thức](project/overview.md) | Giải thích bài toán/PDP; trạng thái thực hiện xem bàn giao |
| [Tài liệu nguồn](references/README.md) | Handoff gốc, phân tích paper, PDF và notebook tham khảo |
| [Lịch sử tháng 09](archive/README.md) | Kế hoạch/tiến độ/GCP cũ |
| [Templates AI DevKit](ai/) | Templates kế thừa cho các phase phát triển |

Notebook có [chỉ mục riêng](../notebooks/README.md). Quy tắc thêm mới: quyết định/kế hoạch vào `project/`; hướng dẫn vào thư mục nghiệp vụ; báo cáo có ngày vào `data_preprocessing/reports/`; mốc thời gian vào `timelines/`; tư liệu nguồn vào `references/`; tài liệu đã thay thế vào `archive/`. Cập nhật chỉ mục và tiến độ trong cùng thay đổi triển khai.

Code dùng lại đặt trong `tools/`, `autocheckout/`, `pdp/`; notebook điều phối và ghi bằng chứng. Config version ở `configs/data/`; dữ liệu ảnh/mask ở `data/processed/`, run receipts/QA/snapshot ở `runs/`, hai thư mục này không vào git. Giữ notebook với mục tiêu và output nghiệm thu; ảnh gallery đầy đủ ở artifacts local.
