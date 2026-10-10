# Training trên Vast.ai

GPU mục tiêu RTX 4090. Người dùng quyết định ngày 10/10/2026 bỏ notebook nghiệm thu môi trường Vast.ai riêng. Chuẩn bị code/config/notebook local; thực chạy pilot EXP-B1/B2, full-model Task 1→2 smoke và baseline 5 task trên GPU training đã chọn. Notebook train ghi GPU/versions/kernel mode, config/split lock, timestamp, steps/loss/VRAM/thời gian và checkpoint artifacts. Chưa có pilot hoặc full training mới.

Chỉ bắt đầu sau khi các notebook trong [modeling](../modeling/README.md) đạt nghiệm thu.

## Notebooks và vận hành

| Notebook | Công việc |
|---|---|
| [01_task1_pilot.ipynb](01_task1_pilot.ipynb) | Pilot EXP-B1/B2 cùng budget; dry run có interruption/resume |
| [02_task_transition.ipynb](02_task_transition.ipynb) | Task 1→2, teacher/PPG/prototype/current-task subset; parent từ pilot |
| [03_five_task_baseline.ipynb](03_five_task_baseline.ipynb) | Baseline mới đủ 5 task; snapshot metrics/policy ngay sau mỗi task |

[Runbook chi tiết](../../docs/project/TRAINING_NOTEBOOK_RUNBOOK.md) hướng dẫn settings, copy/upload data/weights, tmux, đọc logs, resume và download artifacts. [Báo cáo dry run](../../docs/project/TRAINING_NOTEBOOK_DRY_RUN.md) giữ evidence local.

Mọi artifacts mới ghi dưới `data/training/`; template settings ở [notebooks.json](../../configs/training/notebooks.json), mặc định CPU dry run. Đặt `dry_run=false` trên 4090 để dùng pretrained/model/data thật. Chọn `baseline_epochs` từ pilot val trước notebook 03. Executor tạo bản notebook có outputs dưới data, dùng đúng interpreter và không giới hạn thời gian cell train.
