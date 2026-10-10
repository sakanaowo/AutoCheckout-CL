# Dry run local: notebooks pilot, transition và baseline

Ngày 10/10/2026, UTC+7. Bộ [training notebooks 01–03](../../notebooks/training/README.md) đã được chạy bằng headless executor trong **tmux detached**, trên CPU. Các lệnh launch tmux trả về ngay và notebooks vẫn hoàn tất trong background; không có SSH client gắn vào các sessions test này. [Runbook](TRAINING_NOTEBOOK_RUNBOOK.md) ghi lệnh dùng trên 4090.

## Bằng chứng notebook

Campaign kiểm tra cuối: `data/training/dry_run/notebooks-final-20261010/`. Settings receipt: `data/training/validation/settings.json`. Tất cả timestamps sinh khi thực chạy.

| Notebook | Execution ID | Bắt đầu → kết thúc UTC+7 | Kết quả |
|---|---|---|---|
| Pilot Task 1 | `48fd65ce-4f86-46a3-97e8-dd0e11021d34` | 10:00:52 → 10:01:09 | COMPLETED, 3 code cells; 2 experiments |
| Task 1→2 | `c7a3c931-a0c2-4b56-83db-f8f603151143` | 10:02:40 → 10:02:51 | COMPLETED, 2 code cells; 2 experiments |
| Baseline 5 task | `e40091ca-d1fc-48e3-883f-55d13e08a2e5` | 10:02:51 → 10:03:47 | COMPLETED, 2 code cells; 5 tasks/experiment |

Mỗi execution folder giữ source/executed notebook, settings, console log, per-cell timestamps và execution.json. Các runner attempts giữ từng command/env/git diff, stdout/stderr và `.process.json` với PID/host/start/exit code. Live task logging gồm step_metrics.jsonl, progress.json, checkpoint_status.json, run_info.json, epoch validation JSON và Lightning CSV.

- Pilot cả hai cases cố ý dừng ở optimizer step 1, exit **75**, giữ last.ckpt và không có final checkpoint/predictions. Tiến trình mới resume tới step **2**; hash model/optimizer/scheduler/PDP memory sau restore đều khớp. Không đổi physical batch 1/effective batch 2.
- Task 2 lấy checkpoint parent từ pilot, ghi parent MD5, teacher present/frozen, teacher forwards >0. Prototype cache được mang qua và telemetry ghi đúng pseudo labels (0 trong fixture này); không coi 0 là đã kiểm chứng chất lượng PPG.
- Baseline train mới Task 1 rồi Task 2–5, **2 optimizer steps/task**; đủ final checkpoints, val/test predictions, raw mAP/forgetting/counting và policy val cho từng stage.
- Metrics/policy được snapshot sau từng task vào `stage_metrics/task_<t>/`; không phải đợi cả 5 task mới có kết quả. Oracle mặc định tắt. Input annotation/task config hashes giữ nguyên.
- Rerun baseline cùng settings/campaign lúc **10:08:50–10:08:52**, execution `5f0bde5f-c525-48a7-b28c-2beae86d4da9`, COMPLETED; cả 10 task của hai cases được skip, không tạo invocation training mới. Verification receipt ở `data/training/validation/verification.json` đối chiếu outputs, epoch logs, teacher counters và policy stages của các snapshots.

CPU dry profile: random ResNet18, encoder 1/decoder 2, 96px, 6 labels chia 2+1+1+1+1; B1/B2 dùng cùng profile nhỏ. `proto_correct_only=0` chỉ trong dry run để thực thi cache. Real run vẫn dùng ResNet50/ConvNeXt public pretrained và correct-only prototype policy. Không dùng các metrics/checkpoints fixture để báo nghiên cứu hoặc khởi tạo real baseline.

## Tests và lỗi đã sửa

Regression **88 tests PASS**, 0 failures/errors/skips, receipts ở `data/training/validation/training_notebooks_regression.xml` và `regression.log`. Coverage gồm notebook config/roots, uploaded campaign đổi đường dẫn, batch conflict, prerequisites, subprocess PID và anti-overlap, per-task snapshots, CLI import/runtime/resume, teacher/prototypes, evaluator/grid/NMS và run provenance. Ruff và shell/Python/notebook syntax đạt; 252 internal links của các docs thay đổi được kiểm tra.

Các tests mới đã có tín hiệu FAIL trước fix:

1. Resume không dùng `.prev` khi last.ckpt bị thiếu trong rotation; đã thêm fallback khi file chính không tồn tại.
2. Không có logging step/teacher/errors bền vững; đã thêm TrainingProgress và checkpoint receipt.
3. Lightning có thể xử lý Ctrl+C nội bộ rồi fit() trả về; CLI cũ lưu task_final cho task chưa hoàn tất. Nay kiểm tra `trainer.interrupted`, trả exit 75 và giữ task resumable.
4. Validation AP chỉ in console, không có artifact epoch; nay lưu AP/AP50/AP75 và log CSV để chọn pilot budget bằng val.
5. Upload campaign sang root khác bị coi là đổi cấu hình; nay so cấu hình/content hashes, cho phép đổi đường dẫn.
6. Baseline chỉ evaluate cuối protocol; nay chạy/evaluate/snapshot từng task, policy riêng cho từng prefix.

Executor ban đầu dùng synchronous KernelManager nên local startup bị treo trước cell đầu; lượt đó giữ trạng thái INTERRUPTED. Đã chuyển sang AsyncKernelManager, cleanup kernel khi kết thúc và giữ connection files dưới data. Một lượt startup khác FAIL vì sandbox chặn local sockets; các lượt tmux cuối đã thực chạy ngoài giới hạn socket đó và COMPLETED. Không sửa các receipt lỗi thành PASS.

## Giới hạn và bàn giao

Dry run kiểm chứng notebook orchestration, persistence/logging, resume, teacher và evaluator trên fixture nhỏ. Chưa train pilot/backbone đầy đủ trên 4090, chưa xác nhận convergence hoặc chất lượng đủ 200 SKU. Baseline_epochs của real settings vẫn cần chọn từ pilot val curves; không có budget nghiên cứu đã nghiệm thu chỉ từ dry run.

Artifacts mới đều dưới data để upload, không vào Git. Code/config/notebook templates ở repo; các run modeling/evaluation lịch sử giữ nguồn evidence riêng. Quyết định bỏ notebook môi trường Vast.ai riêng vẫn có hiệu lực.
