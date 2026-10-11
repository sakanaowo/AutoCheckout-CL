# Báo cáo dừng baseline RTX 4090 — 11/10/2026

Người dùng yêu cầu dừng để tối ưu chi phí/hiệu năng và chỉ xóa artifact nếu không
thể tiếp tục trên GPU khác. **Đã dừng training; giữ artifact vì checkpoint hợp lệ
và workflow hỗ trợ resume trên GPU tương thích. Máy đích chưa được kiểm chứng.**

## Run và điểm dừng

- Campaign `vast4090-seed0-640-v1`, phase baseline, thứ tự EXP-B2 → EXP-B1.
- ConvNeXt-V2-Base Task 1, 21.752 ảnh; 640px, batch/effective 8/8, workers 8,
  FP32 + TF32, seed 0, budget khởi đầu 16 epochs/task.
- Bắt đầu 10:54:46, dừng hoàn toàn 15:08:40 UTC+7, wall time **4 giờ 13 phút 54 giây**.
- Hoàn tất **11 epochs**, đang epoch 12/16; observed step **31.579**.
- Checkpoint dùng để tiếp tục: **step 31.550, epoch index 11**, khoảng
  72.52% budget optimizer updates Task 1. Resume chạy lại
  **29 updates** sau checkpoint. Không tuyên bố shuffle/minibatches giống hệt chạy liên tục.
- ResNet chưa bắt đầu; Task 2–5 chưa chạy; chưa có task_final hoặc predictions
  val/test của baseline. Không có metrics counting/test/forgetting baseline cuối.
- Chi phí ước tính riêng khoảng thời gian notebook trên instance: **1.76 USD**
  tại 0,4153 USD/h, không phải invoice. Instance còn running nên billing tiếp tục
  sau khi training dừng. Không shutdown/destroy instance hoặc xóa dữ liệu.

## Validation đã thu được

| Epoch hoàn tất | Steps | AP | AP50 | AP75 |
|---|---:|---:|---:|---:|
| 1 | 2719 | 21.45% | 27.90% | 25.60% |
| 2 | 5438 | 51.97% | 63.65% | 60.60% |
| 3 | 8157 | 66.95% | 80.86% | 77.46% |
| 4 | 10876 | 73.48% | 88.15% | 84.54% |
| 5 | 13595 | 76.80% | 91.71% | 88.08% |
| 6 | 16314 | 79.14% | 94.01% | 90.64% |
| 7 | 19033 | 80.40% | 95.46% | 92.07% |
| 8 | 21752 | 81.06% | 96.13% | 92.77% |
| 9 | 24471 | 81.92% | 96.68% | 93.62% |
| 10 | 27190 | 82.23% | 97.01% | 93.63% |
| 11 | 29909 | 82.83% | 97.57% | 94.28% |

Val AP tăng 21,45→82,83%; ba delta gần nhất khoảng +0,85 / +0,31 / +0,60 điểm
phần trăm. Chưa đạt tiêu chí đề xuất tăng <0,5 điểm trong ba epochs liên tiếp;
không xác nhận hội tụ từ run bị dừng này. Pilot và baseline đổi effective batch,
TF32 và số lượt ảnh/updates; không quy toàn bộ cải thiện cho riêng một thay đổi.
Prototype chỉ tích lũy ở epoch cuối, nên chưa dùng checkpoint giữa task làm teacher
Task 2; phải hoàn tất Task 1 trước. Test chưa được dùng để chọn budget.

## Kiểm tra checkpoint và quyết định artifact

- `last.ckpt`: 780.234.350 bytes, step 31.550. Backup `.prev`: cùng số bytes,
  step 31.274. Cả hai torch.load(map_location='cpu') thành công.
- 1.156 tensors mỗi checkpoint, tensors numeric hữu hạn; đầy đủ state_dict,
  optimizer_states (1), lr_schedulers (1), PDP caches/prototypes/counters và loops.
- SHA256 của last: `e5b4c411a512208d8533ff9da5049d1fe4705b7ff45439c9d4985960491b7295`.
- Resume trước đó từ step 547 đã kiểm chứng hash model, optimizer, scheduler và
  PDP state đều bằng checkpoint. Đây là bằng chứng resume cùng máy, không phải
  benchmark/restore thực tế trên GPU đích khác.
- Training contract khóa batch/effective, n_gpus=1, seed, shuffle, freeze, LR,
  scheduler/policy và checksum task_config/train. **Không khóa tên GPU**.
- Plan relocation check đạt khi đổi roots, giữ content hashes; có thể chuyển
  NVIDIA GPU tương thích môi trường CUDA/PyTorch, giữ một GPU cho mỗi experiment.
- **Giữ toàn bộ baseline (~1,5 GiB), notebook executions (~50 MiB), logs và
  checkpoint backup. Không xóa artifact nào.** Điều kiện xóa của người dùng chưa xảy ra.
- Pilot/transition, pretrained và release dữ liệu giữ nguyên.

[Audit checkpoint](../../data/training/vast4090-seed0-640-v1/baseline_checkpoint_portability_audit.json),
[stop receipt](../../data/training/vast4090-seed0-640-v1/baseline_stop_report.json),
[val CSV](../../data/training/vast4090-seed0-640-v1/baseline_validation_curve.csv).

## Tiếp tục trên máy khác

[Manifest chuyển máy](../../data/training/vast4090-seed0-640-v1/baseline_resume_manifest.json)
ghi paths cần giữ và HEAD/code patch. Copy code đúng revision và các sửa TF32/logging,
release + raw images, pretrained cache gồm blobs/snapshots, settings và toàn bộ
campaign có pilot/transition COMPLETED. Giữ symlinks hoặc copy đầy đủ content.

Cài env pdp Python 3.10 với dependencies pinned; kiểm chứng GPU/native kernel trên
máy đích. RTX 30/40 là ứng viên cùng stack; RTX 50 cần env mới và regression riêng.
Không copy binary torch_extensions giữa kiến trúc GPU khác; dùng cache mới và
biên dịch kernel trên máy đích. Chỉ sửa roots nếu đổi đường dẫn, giữ campaign,
experiments, 16 epochs, batch/effective 8/8, workers 8, TF32, seed và data hashes.

```bash
python -m tools.execute_training_notebook \
  --notebook notebooks/training/03_five_task_baseline.ipynb \
  --settings data/training/settings_baseline_4090.json \
  --live-notebook notebooks/training/03_five_task_baseline.ipynb --save-interval 15
```

Executor sẽ resume ConvNeXt Task 1 từ last.ckpt, verify model/optimizer/scheduler/PDP
state, rồi tiếp tục protocol. Hai GPU chỉ dùng được hiệu quả khi có runner gán mỗi
experiment một GPU riêng; không đổi n_gpus=2 trong task đang resume. Không tự launch
lại từ handoff; người dùng đã yêu cầu dừng. GPU/offer đích chưa được thuê.

## Trạng thái vận hành

Runner nhận SIGINT, exit **75**, process receipt INTERRUPTED; progress INTERRUPTED.
Notebook executor ghi FAIL/CellExecutionError vì wrapper chuyển exit 75 thành lỗi
cell; đây là dừng theo yêu cầu, không phải NaN/OOM. Output và error traceback được
lưu đầy đủ trong execution folder và
[`baseline_notebook_full_at_commit.ipynb`](../../data/training/vast4090-seed0-640-v1/baseline_notebook_full_at_commit.ipynb).
Notebook trong Git giữ tóm tắt điểm dừng và traceback gốc; log đầy đủ giữ ngoài Git. tmux ac-baseline có
pane dead, không còn process training; không khởi chạy lại. `baseline_launch.json`
đã chuyển STOPPED_BY_USER. Không tạo summary COMPLETED giả cho baseline.
