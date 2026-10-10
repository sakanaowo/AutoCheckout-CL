# Vận hành notebooks training: data artifacts, tmux và resume

Ngày 10/10/2026. Chuẩn bị code/notebooks tại local, chạy thực nghiệm chính trên RTX 4090 đã chọn. Theo quyết định người dùng, bỏ notebook nghiệm thu môi trường Vast.ai riêng. Ba notebook gọi runner/PDP/evaluator hiện có; executor chỉ giúp notebook chạy ngầm và lưu outputs.

## 1. Thứ tự và cấu hình

1. [01_task1_pilot.ipynb](../../notebooks/training/01_task1_pilot.ipynb): pilot Task 1 của EXP-B1/EXP-B2, cùng data/batch/resolution/epochs.
2. [02_task_transition.ipynb](../../notebooks/training/02_task_transition.ipynb): copy riêng checkpoint Task 1, smoke Task 2 trên derived subset, kiểm tra teacher/PPG/cache/current-task loader.
3. [03_five_task_baseline.ipynb](../../notebooks/training/03_five_task_baseline.ipynb): Task 1 mới từ pretrained theo budget đã chọn rồi Task 2–5, evaluator và snapshot metrics ngay sau mỗi task.

Dùng Python 3.10 của Conda `pdp` đã chuẩn bị. Cần `nbclient`, `nbformat`, `ipykernel`, dependencies PDP và `tmux`. Executor tự tạo kernelspec dưới data với chính `sys.executable`; không cần đăng ký kernel tên pdp. Không cài package hoặc tải weights trong notebook. Gói/environment setup hiện có ở [README môi trường](../environment/README.md).

Copy template thành cấu hình vận hành nằm trong data:

```bash
mkdir -p data/training
cp configs/training/notebooks.json data/training/settings_4090.json
```

Sửa `data/training/settings_4090.json`:

| Trường | Cách đặt |
|---|---|
| `campaign_id` | Tên riêng cho thí nghiệm, ví dụ `native-seed0-640-v1`; giữ nguyên để resume |
| `artifact_root` | `data/training`, hoặc đường tuyệt đối tới folder data trên persistent disk |
| `dry_run` | `false` cho real run; `true` cho CPU fixture, output tự tách vào `dry_run/` |
| `experiments` | `["EXP-B1", "EXP-B2"]`; chạy lần lượt trên một GPU |
| `raw_root`, `release_root` | `data/archive`, `data/processed/rpc_100-4x25_seed0_native_v1` hoặc roots đã upload |
| `pretrained_root` | `data/pretrained/hf-cache`, gồm cả snapshots và blobs của hai weights đã khóa |
| `resolution` | 640 cho pilot chính; đổi sang 800 cần campaign mới |
| `batch_size`, `effective_batch` | Mặc định 1/2. Effective phải chia hết physical; accumulation = effective/physical, một GPU |
| `pilot_epochs` | Mặc định 2 là budget khởi đầu; hai models dùng cùng số epochs và optimizer steps |
| `transition_epochs`, `transition_samples` | Mặc định 1 epoch, tối đa 16 ảnh train Task 2; diagnostics riêng |
| `baseline_epochs` | Mặc định `null`; sau pilot chọn số nguyên dương từ loss/val curves trước notebook 03 |
| `checkpoint_every_minutes` | Mặc định 2, và checkpoint cuối mỗi epoch |
| `num_workers` | Mặc định 4 cho real run; dry run dùng 0 |
| `count_nms_grid` | Mặc định `"0.45"`; nếu tune NMS, các candidates chỉ được chọn trên val |
| `test_interruption` | Chỉ áp dụng dry run: cố ý dừng step 1 rồi resume; real run không tự dừng |

Giữ physical/effective batch, epochs, resolution và dữ liệu của phase đang resume. Plan và input hashes được khóa; thay cấu hình thí nghiệm cần campaign mới. `baseline_epochs` có thể chọn sau pilot vì các phase có plan riêng. Epoch budget giữ cơ chế cập nhật prototype ở epoch cuối của PDP hiện tại; không cắt task giữa epoch bằng một max-step tùy ý.

## 2. Upload data và pretrained

Trên local weights hiện nằm dưới runs, có thể copy cache đã tải vào data trước khi upload:

```bash
mkdir -p data/pretrained
cp -a runs/pretrained/hf-cache data/pretrained/
```

Giữ cả thư mục cache, gồm `blobs/` và `snapshots/`, vì snapshots có thể là symlink. Revision detector `83ecd26945199939cb82806f988debdb71e6f43e`; ConvNeXt `9e250ed8f88b436472d2b24af26f82a8aa8c719d`. Khi thiếu detector, downloader có sẵn nhận `--cache-dir data/pretrained/hf-cache --manifest data/pretrained/deformable_detr.json`. Không trộn các pretrained khác vào cùng campaign.

Repo/code phải có cùng source trên máy 4090. Upload folder data với cấu trúc giữ nguyên, ví dụ thay USER/HOST/PORT bằng thông tin SSH thực tế:

```bash
rsync -a --partial --info=progress2 -e 'ssh -p PORT' data/ USER@HOST:/workspace/AutoCheckout-CL/data/
```

Paths tương đối trong settings được resolve từ repo root. Khi đưa campaign đã chạy sang một máy/path khác, giữ dữ liệu và weights cùng content hashes; sửa roots trong settings cho đúng vị trí mới. Plan so sánh cấu hình và content hashes, cho phép đổi đường dẫn khi resume; model/processor dựng lại từ checkpoint metadata. Upload toàn bộ campaign, gồm các parent checkpoints cho Task 2+.

## 3. Chạy notebook ngầm và reconnect

Trên máy 4090, từ repo root, tạo tmux session. Gõ lệnh Python **bên trong session**:

```bash
cd /workspace/AutoCheckout-CL
tmux new-session -s ac-pilot
```

```bash
conda activate pdp
python -m tools.execute_training_notebook \
  --notebook notebooks/training/01_task1_pilot.ipynb \
  --settings data/training/settings_4090.json
```

Nhấn **Ctrl+B**, thả phím rồi nhấn **D** để detach. Notebook tiếp tục khi SSH disconnect; reconnect bằng:

```bash
tmux ls
tmux attach -t ac-pilot
```

Sau notebook 01, chạy notebook 02 bằng cùng lệnh, đổi `--notebook` sang `02_task_transition.ipynb`. Sau review pilot/transition và chọn `baseline_epochs`, chạy `03_five_task_baseline.ipynb`. Có thể dùng cùng tmux session hoặc các session `ac-transition`, `ac-baseline`; chỉ chạy một workflow của cùng campaign tại một thời điểm. File lock và PID receipts chặn job trùng đang sống.

tmux giữ session qua SSH timeout/detach theo [manual chính thức](https://man.openbsd.org/tmux). Instance reboot/termination hoặc disk mất không được tmux bảo vệ; checkpoint/artifact cần nằm trên disk bền vững và được tải về trước khi kết thúc thuê máy.

Executor dùng `--cell-timeout -1` mặc định để cell train dài không bị timeout; tự lưu `notebook_executed.ipynb` sau mỗi code cell và khi lỗi. [nbclient](https://nbclient.readthedocs.io/en/latest/client.html) hỗ trợ execution hooks và tắt timeout. Trong lúc cell train chạy, đọc runner/task logs dưới data để theo dõi trực tiếp; console notebook hiển thị cell start/finish và stream outputs sau cell.

Nếu chạy trực tiếp từ IDE, chọn kernel Conda pdp và đặt trước cell setup:

```python
import os
os.environ["AUTOCHECKOUT_TRAINING_SETTINGS"] = "/workspace/AutoCheckout-CL/data/training/settings_4090.json"
```

Notebook đọc cùng settings/runner. Dùng tmux/headless executor khi cần độc lập với IDE hoặc phiên SSH; bản notebook source trong git giữ hướng dẫn, các bản có output tự lưu dưới data.

## 4. Artifacts và logging

```text
data/training/<campaign_id>/
  pilot|transition|baseline/
    plan.json                     # cấu hình resolved, input/parent/weights hashes, budget
    summary.json                  # chỉ COMPLETED khi đủ artifacts và kiểm tra teacher/data đạt
    attempts/<attempt_id>/
      execution.json              # git commit/diff, lệnh/env, timestamps, exit codes, errors
      EXP-Bx_taskN_*_runner.log    # stdout + stderr đầy đủ của từng tiến trình
      *.process.json              # PID/host/process start, RUNNING/COMPLETED/INTERRUPTED/FAIL
    EXP-B1|EXP-B2/
      resolved_config.sh, resolved_cli.json
      task_<t>/
        train.log                 # log engine/PDP, line buffered
        step_metrics.jsonl        # một dòng/optimizer step: loss/LR/epoch/batch/time/VRAM/teacher/PPG
        progress.json             # tiến độ mới nhất, step/status/error
        training_start.json       # restored step, teacher hiện diện/frozen, old/seen classes
        val_epoch_0000.json        # AP/AP50/AP75 trên val_task sau mỗi epoch
        checkpoint_status.json    # checkpoint timestamp, optimizer step, bytes, backup existence
        last.ckpt, last.ckpt.prev  # trong task đang dở; task hoàn tất thì dọn checkpoint resume
        task_final.pth            # final model/config/processor/prototype memory
        runtime.json, loading_report.json, run_info.json
        pred_val.npz, pred_test.npz
        lightning_logs/version_*/metrics.csv
      calibration_count*.json     # policy chọn trên val, oracle tắt
      metrics_cl_val.json, metrics_cl_test.json, metrics_count_test.json
      stage_metrics/task_<t>/     # baseline: snapshot metrics/policy ngay sau từng task
  transition_inputs/tasks/       # derived subset riêng, giữ release gốc
  notebook_executions/<notebook>/<execution_id>/
    notebook_source.ipynb, notebook_executed.ipynb
    settings.json, execution.json, console.log, error.log (nếu lỗi)
    kernels/, kernel_runtime/, ipython/
```

Dry run có thêm `data/training/dry_run/<campaign_id>/fixture/`, dùng ảnh/model nhỏ riêng. Hash annotations và task config đối chiếu trước/sau. Real weights có MD5 và pinned revisions; git diff/source của từng session lưu trong execution/run_info.

Để xem trực tiếp khi reconnect, thay campaign/experiment/task đúng run đang hoạt động:

```bash
watch -n 10 cat data/training/native-seed0-640-v1/pilot/EXP-B1/task_1/progress.json
tail -F data/training/native-seed0-640-v1/pilot/EXP-B1/task_1/step_metrics.jsonl
tail -F data/training/native-seed0-640-v1/pilot/EXP-B1/task_1/train.log
```

Loss fields gồm `tr`, `ce`, `bbox`, `giou`, `car`, `QL`, `DDL` và metrics validation khi có. Loss JSONL là scalar từ callback của optimizer step vừa xong; `seconds_per_optimizer_step` là trung bình từ đầu session hiện tại, không gồm bootstrap trước fit. VRAM là peak allocated/reserved trong tiến trình, không phải toàn instance. `run_info.json` giữ từng session/optimizer steps/resume verification/seconds; xem total_seconds khi cộng các lượt interruption. COCO AP=-1 có nghĩa không có GT hợp lệ cho phạm vi đó.

## 5. Resume sau interruption

- **Chỉ mất SSH:** reconnect vào tmux đang sống; không mở thêm run mới. PID/lock giúp tránh hai tiến trình ghi cùng campaign.
- **Notebook/tiến trình bị ngắt:** chạy lại cùng notebook/settings/campaign. Runner skip task có final + hai predictions; thiếu predictions thì predict-only; task chưa final tiếp tục bằng `last.ckpt`, dựng lại model/processor và restore optimizer/scheduler/PDP memory.
- **Mất điện/instance reboot:** khởi động lại environment đã chuẩn bị, mount/upload lại toàn bộ data rồi chạy cùng lệnh. Resume quay về checkpoint đã lưu gần nhất; công việc sau checkpoint có thể phải chạy lại. Nếu chưa có checkpoint đầu tiên, task bắt đầu lại.
- **`last.ckpt` thiếu:** dùng `last.ckpt.prev` tự động nếu tồn tại. Nếu `last.ckpt` tồn tại nhưng hỏng, giữ bản hỏng để kiểm tra rồi copy `.prev` thành `last.ckpt`; code không tự bỏ qua lỗi load của một file có mặt.
- **PID receipt vẫn RUNNING sau hard kill parent:** kiểm tra PID trên đúng host và reconnect vào runner nếu còn sống. Nếu cần dừng chủ động, gửi SIGINT cho process group của PID runner (`kill -INT -- -PID`, thay PID bằng số trong receipt), đợi thoát rồi resume. Không chạy chồng job khi runner cũ còn sống.

Checkpoint ghi file tạm, rotate `.prev`, rồi rename atomic; lưu ở ranh giới optimizer step và cuối epoch. Ctrl+C/Lightning interruption trả exit 75, không ghi task_final cho task chưa hoàn tất. `--verify_resume 1` đối chiếu hash model/optimizer/scheduler/PDP memory sau restore. Shuffle trong real run vẫn bật; không tuyên bố thứ tự minibatch sau resume giống hệt run liên tục.

Budget/config đổi sau khi task đã final cần campaign mới. Full baseline có folder riêng, train lại Task 1 đủ budget; transition checkpoint pilot không được dùng thay Task 1 baseline. Policy đã chọn trên val được giữ khi rerun; baseline dùng policy/snapshot riêng sau từng task để không trộn stage sets.

## 6. Dry run local và bàn giao

Giữ `dry_run=true` hoặc thêm `--dry-run` trong lệnh executor. Chạy ba notebook theo thứ tự. Dry run dùng CPU random ResNet18 nhỏ/6 labels và `proto_correct_only=0` để thực thi cache/teacher; B1/B2 lúc này là hai case workflow cùng profile nhỏ. Không dùng metrics hoặc checkpoint này cho baseline thật. Real run giữ default `proto_correct_only=1` và pretrained/backbone đúng config.

Dry run kiểm tra: chuẩn bị inputs dưới data, hai cases pilot, dừng/resume step 1, teacher frozen và teacher forwards ở Task 2+, đủ 5 task, logging/CSV/epoch val metrics, calibration val-only/raw mAP/counting/forgetting và rerun skip task hoàn tất. Regression receipts nằm ở `data/training/validation/`. [Báo cáo dry run](TRAINING_NOTEBOOK_DRY_RUN.md) ghi timestamps và bằng chứng cuối cùng.

Khi real run kết thúc, sync campaign về máy local hoặc storage bền vững:

```bash
rsync -a --partial --info=progress2 -e 'ssh -p PORT' \
  USER@HOST:/workspace/AutoCheckout-CL/data/training/native-seed0-640-v1/ \
  data/training/native-seed0-640-v1/
```

Data/artifacts không vào Git. Giữ code/config/notebook template trong Git, và toàn bộ campaign dưới data để có thể tiếp tục hoặc đối chiếu kết quả.
