# Training và resume trên GPU

Ngày **13/10/2026**, baseline đã dừng trên L4 GCP theo người dùng để chuyển sang
**4080 SUPER**. Pilot EXP-B1/B2 và transition smoke đã hoàn tất trên 4090;
ConvNeXt baseline Task 1 hoàn tất trên L4, Task 2 resume từ checkpoint **6.414**.
ResNet baseline chưa bắt đầu. [Bàn giao chuyển máy](../../docs/environment/TRANSFER_TO_4080_2026-10-13.md)
ghi artifacts và điều kiện tiếp tục; chưa khởi chạy trên GPU đích.

Người dùng đã bỏ notebook nghiệm thu môi trường riêng. Notebook training giữ
GPU/versions/kernel mode, config/split lock, timestamps, steps/loss/VRAM và artifacts.

Chỉ bắt đầu sau khi các notebook trong [modeling](../modeling/README.md) đạt nghiệm thu.

## Notebooks và vận hành

| Notebook | Công việc |
|---|---|
| [01_task1_pilot.ipynb](01_task1_pilot.ipynb) | Pilot EXP-B1/B2 cùng budget; dry run có interruption/resume |
| [02_task_transition.ipynb](02_task_transition.ipynb) | Task 1→2, teacher/PPG/prototype/current-task subset; parent từ pilot |
| [03_five_task_baseline.ipynb](03_five_task_baseline.ipynb) | Baseline mới đủ 5 task; snapshot metrics/policy ngay sau mỗi task |

[Runbook chi tiết](../../docs/project/TRAINING_NOTEBOOK_RUNBOOK.md) hướng dẫn settings, copy/upload data/weights, tmux, đọc logs, resume và download artifacts. [Báo cáo dry run](../../docs/project/TRAINING_NOTEBOOK_DRY_RUN.md) giữ evidence local.

Mọi artifacts mới ghi dưới `data/training/`; template settings ở [notebooks.json](../../configs/training/notebooks.json), mặc định CPU dry run. Real baseline dùng `dry_run=false`, budget đã khóa 16 epochs/task. Khi chuyển máy, truyền `--settings data/training/settings_baseline_4080.json`; notebook 03 vẫn giữ default legacy 4090 nếu không override. Executor dùng đúng interpreter, lưu live outputs mỗi 15 giây và không giới hạn thời gian cell train.

Notebook 03 trong Git giữ tóm tắt thực nghiệm và traceback dừng; bản log đầy đủ của
lượt L4 nằm ở `data/training/vast4090-seed0-640-v1/baseline_notebook_full_l4_at_commit_20261013.ipynb`
và các execution folders. Copy campaign hiện tại; gói Drive ngày 11/10 chưa có tiến độ mới.
