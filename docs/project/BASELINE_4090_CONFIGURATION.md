# Cấu hình notebook 03 sau pilot và transition

**DỪNG THEO YÊU CẦU NGƯỜI DÙNG — 11/10/2026 15:08:40 UTC+7.**
Baseline ConvNeXt dừng ở Task 1, completed 11 epochs; checkpoint step 31.550
có thể resume trên GPU tương thích. **Giữ artifact**, không tự khởi chạy lại.
[Báo cáo dừng và chuyển GPU](BASELINE_4090_STOP_REVIEW_2026-10-11.md) ghi validation,
audit, manifest và điều kiện tiếp tục. ResNet chưa bắt đầu, full baseline chưa hoàn tất.

Các mô tả run đang hoạt động bên dưới là snapshot trước khi dừng.

Ngày **11/10/2026, UTC+7**. Ưu tiên của người dùng: **hội tụ trên validation**,
không đặt giới hạn tổng thời gian baseline. Pilot và transition smoke đã hoàn tất.
Notebook 03 khởi chạy **11/10 lúc 10:54:46 UTC+7**, tmux `ac-baseline`:
**ConvNeXt Task 1–5 trước, rồi ResNet Task 1–5**. Dùng 16 epochs/task làm budget
khởi đầu; chưa nghiệm thu hội tụ hoặc kết quả baseline cuối.

## Phần nền giữ nguyên

- EXP-B2 PDP ConvNeXt-V2-Base trước, rồi EXP-B1 PDP ResNet50; tuần tự trên một RTX 4090.
- Native real-only release, seed 0, protocol **100+4×25**, holdout/checksums đã khóa.
- 640px, pretrained revisions hiện hành; baseline train lại Task 1 từ pretrained,
  không dùng Task 1 pilot/transition thay Task 1 baseline.
- Freeze backbone/encoder/decoder theo native config; không bật LoRA/K=3/FSA/synthetic
  hoặc freeze bổ sung. Giữ LR 1e-4 / LR_old 1e-5, effective batch giống nhau giữa B1/B2;
  chưa có bằng chứng để tự scale LR theo batch.
- Prototype mean, correct-only; checkpoint mỗi 2 phút và cuối epoch, resume audit bật.
- Val sau mỗi epoch; sau mỗi task có predictions/mAP/counting/forgetting/policy snapshot.
  Counting NMS 0,45, score grid 0,10–0,90; chọn trên val, test dùng policy khóa, oracle tắt.

## Đo GPU thực tế

Artifacts ở `data/training/validation/baseline_gpu_benchmark_20261011/`:
[summary](../../data/training/validation/baseline_gpu_benchmark_20261011/summary.json),
[biểu đồ GPU](../../data/training/validation/baseline_gpu_benchmark_20261011/gpu_profiles.png).
**20 cases PASS**, không OOM; parent pilot hashes không đổi. Đo bằng PDP model/loader/CLI
thật trên derived subsets riêng, Task 2 có teacher và PPG. Thêm timing callback sau
warmup, đo ít nhất 16 optimizer steps và 8 giây; dừng chủ động trước final/evaluation,
không tạo research checkpoints hoặc dùng metrics benchmark để kết luận chất lượng.
Prototype path của epoch cuối hoạt động trong các phép đo.

| Task 2 profile | ResNet ảnh/giây | ConvNeXt ảnh/giây | GPU utilization B1/B2 | Peak allocated B1/B2 |
|---|---:|---:|---:|---:|
| Batch 1, workers 4, FP32 | 4,31 | 4,06 | 51% / 63% | 3,32 / 3,75 GiB |
| Batch 4, workers 4, FP32 | 11,94 | 8,22 | 75% / 80% | 5,06 / 5,45 GiB |
| Batch 8, workers 4, FP32 | 15,19 | 8,87 | 86% / 92% | 7,42 / 7,75 GiB |
| Batch 16, workers 4, FP32 | 16,44 | 9,72 | 91% / 94% | 12,30 / 12,55 GiB |
| Batch 8, workers 8, FP32 + TF32 | **16,29** | **10,56** | **73% / 88%** | **7,42 / 7,75 GiB** |
| Batch 16, workers 8, FP32 + TF32 | 19,10 | 11,90 | 92% / 97% | 12,30 / 12,55 GiB |

Tăng workers từ 4 lên 8/16 ở batch 16 FP32 không tạo cải thiện lớn: ResNet
16,44→16,28→16,11 ảnh/s, ConvNeXt 9,72→9,81→9,96 ảnh/s.
Task 1 batch 16 FP32 cũng PASS: B1/B2 27,21/17,64 ảnh/s, peak allocated khoảng 11,5 GiB.

Các phép đo ngắn là bằng chứng throughput/finite loss/VRAM, chưa là validation về
accuracy của batch/TF32 mới hoặc memory ở Task 3–5. GPU utilization là số đo driver
trên cửa sổ ngắn, không phải phép đo toàn bộ Tensor Core throughput. TF32 có thể làm
thời gian GPU mỗi batch ngắn hơn nên utilization giảm trong khi ảnh/giây tăng.

## Kiểm chứng hai model cùng chạy trên một GPU

Sau câu hỏi của người dùng về VRAM, đã đo lại batch 8/effective 8, workers 8 mỗi
process, FP32 + TF32 ở Task 2. Derived train subset có 2.048 ảnh, cùng seed và input
cho các lần đo. Mỗi model chạy riêng 60 giây sau warmup, rồi hai process được đồng
bộ sau warmup để cùng train hơn 60 giây. Không đổi cấu hình MPS hoặc workflow.
[Summary đo đồng thời](../../data/training/validation/parallel_gpu_benchmark_20261011/summary.json)
và raw logs/telemetry nằm trong cùng folder.

| Model | Chạy riêng | Khi hai model cùng train |
|---|---:|---:|
| EXP-B1 ResNet50 | 16,80 ảnh/s | 6,77 ảnh/s |
| EXP-B2 ConvNeXt-V2-Base | 10,94 ảnh/s | 5,44 ảnh/s |

**Cả 4 phép đo PASS**, loss hữu hạn, teacher/PPG hoạt động, parent checkpoints và
input hashes giữ nguyên. Hai process có cửa sổ train chồng nhau 60,22 giây.
Khi cùng chạy: tổng peak allocated 15,17 GiB, tổng peak reserved 18,32 GiB;
GPU-wide memory used đo bằng nvidia-smi đạt **19,30 GiB**, utilization trung bình
khoảng **98,5–98,8%**. Không có task_final hoặc predictions nghiên cứu mới.

Vì vậy **batch 8 có thể chạy đồng thời về bộ nhớ**. Batch 16 của hai model có tổng
allocated khoảng 24,85 GiB / reserved 26,51 GiB, vượt GPU có 23,99 GiB usable theo
nvidia-smi; không dùng hai batch 16 cùng lúc. Allocated chỉ tính tensors; reserved
bao gồm vùng caching allocator quản lý, không cộng allocated thêm lần nữa vào
reserved. Xem [PyTorch memory management](https://docs.pytorch.org/docs/2.2/notes/cuda.html#memory-management).

GPU utilization cao khi cùng chạy đi kèm tốc độ từng model giảm mạnh. Phép đo này
chưa cho thấy lợi ích để chuyển baseline sang đồng thời, nên giữ **tuần tự** trên
một RTX 4090. Không suy ra thời gian full baseline từ việc hai process đủ VRAM:
chưa đo validation/prediction xen kẽ, Task 3–5 hoặc giai đoạn một model đã xong.
Notebook hiện có vòng lặp experiment tuần tự và khóa cả campaign; mở hai notebook
cùng settings sẽ bị chặn. Muốn chạy đồng thời cần thiết kế runner và receipts/resume
cho hai jobs; không bỏ khóa campaign để thử trên artifacts nghiên cứu.

## Hardware profile đề xuất

[settings_baseline_4090.json](../../data/training/settings_baseline_4090.json):

```json
{
  "resolution": 640,
  "batch_size": 8,
  "effective_batch": 8,
  "num_workers": 8,
  "tf32": true,
  "baseline_epochs": 16
}
```

Không accumulation. Vẫn dùng FP32 tensors/Lightning `32-true`; TF32 cho phép backend
matmul/convolution dùng Tensor Cores, không phải FP16/BF16 AMP. AMP chưa được đo/nghiệm thu.
Chọn batch 8 vì batch 16 chỉ nhanh hơn 13–17% với TF32 nhưng dùng thêm khoảng 5 GiB
allocated, tăng effective batch và giảm steps/epoch. Batch 16 là lựa chọn throughput
cao hơn đã đo, không phải điều kiện để model học tốt hơn hoặc bắt buộc lấp đầy VRAM.

Settings gốc của pilot/transition giữ nguyên. Hardware settings mới chỉ dùng cho phase
baseline trong cùng campaign; không dùng nó để resume các phase cũ.
TF32 hiện được truyền từ settings→plan→environment→native CLI, mặc định false/0;
plan cũ không có trường này vẫn tương thích khi TF32 tắt. Đổi TF32 trong phase có plan
bị từ chối giống đổi batch. **20 regression tests config/workflow/runner PASS**.

## Chưa chốt số epochs hội tụ

Pilot có val AP tăng mạnh từ epoch 1 sang 2, nên không có plateau để suy ra epochs tối ưu.
Với effective batch 8, Task 1 có **2.719 steps/epoch**:

- 2 epochs: 5.438 steps, chỉ bằng một phần tư pilot cũ; không giữ mặc định này khi đổi batch.
- 8 epochs: 21.752 steps, bằng mốc pilot cũ.
- **16 epochs: 43.504 steps**, budget ứng viên để tăng gấp đôi update budget pilot.
  Đây là budget khởi đầu dùng khi người dùng yêu cầu chạy notebook 03 ngày 11/10,
  chưa phải số epochs đã được validation xác nhận hội tụ.

Để chọn budget cần thêm learning curves trên val với hardware profile mới, giữ cùng
budget giữa hai models và kiểm tra từng task. Tiêu chí plateau có thể chốt trước run,
ví dụ val AP tăng dưới 0,5 điểm phần trăm trong 3 epochs liên tiếp, đồng thời xem lớp
yếu/prototype coverage và counting val. Đây là tiêu chí đề xuất, không phải gate nghiên
cứu đã có trong CLI. Không dùng test để chọn budget.

Workflow hiện tại dùng một `baseline_epochs` chung cho tất cả task; chưa có budget tự
điều chỉnh theo validation từng task. Không thể kết luận cả 5 task hội tụ chỉ từ đường
cong Task 1 hoặc từ transition smoke.

**Không thêm EarlyStopping trực tiếp vào pipeline hiện tại:** prototype chỉ tích lũy
ở epoch cuối `args.epochs-1`; dừng sớm trước đó có thể không tạo memory cho task sau.
Cần chọn fixed budget từ val hoặc thiết kế/kiểm chứng lifecycle prototype trước khi
chuyển sang adaptive stopping. Notebook 03 vẫn giữ guard baseline_epochs dương;
run hiện hành đặt 16 và khóa plan. Không sửa epochs của phase đang chạy khi resume.

## Log notebook và chạy ngầm

Launcher `data/training/vast4090-seed0-640-v1/run_baseline_tmux.sh` chạy trong tmux
`ac-baseline`, nhận `--live-notebook notebooks/training/03_five_task_baseline.ipynb
--save-interval 15`. Notebook source và bản notebook_executed.ipynb trong execution
folder được lưu atomic mỗi 15 giây, cả khi cell train đang chạy hoặc tạm không có
output. Runner stdout/stderr được forward vào notebook đồng thời giữ log file riêng;
console tổng ở `data/training/vast4090-seed0-640-v1/tmux_baseline.log`.

Nếu notebook source bị sửa bên ngoài, mirror dừng để giữ sửa tay; bản trong data
tiếp tục lưu. **13 executor/workflow tests PASS**, gồm output của tiến trình con khi
cell còn chạy, lỗi và bảo vệ sửa tay. Snapshot notebook ban đầu giữ riêng trong data.
Reconnect bằng `tmux attach -t ac-baseline`; khi đã ở tmux khác dùng
`tmux switch-client -t ac-baseline`. Đóng SSH/detach để job tiếp tục.

Đã ngắt runner có kiểm soát (exit 75) và resume từ checkpoint step 547 để bật
forwarding log training vào notebook. Execution hiện hành bắt đầu 11:01:35 UTC+7,
đã qua step 648, loss hữu hạn và mirror ACTIVE. Metadata các đường log/execution
ở `data/training/vast4090-seed0-640-v1/baseline_launch.json`; bản lỗi notebook của
lượt ngắt chủ động giữ trong history, không phải lỗi run đang hoạt động.

## Cách chạy khi đã chọn budget

Đặt `baseline_epochs` đã chọn trong hardware settings, rồi dùng
[notebook 03](../../notebooks/training/03_five_task_baseline.ipynb). Notebook mặc định
đọc đúng hardware file; executor trong tmux nhận `--settings data/training/settings_baseline_4090.json`.
Sau khi phase đã bắt đầu, không đổi batch/TF32/epochs/data khi resume.

Trong training, GPU có thể lên gần 100% với batch 16; profile batch 8 sử dụng ít
bộ nhớ hơn và có throughput tốt. Các đoạn checkpoint, matching CPU, COCO evaluation
và calibration có thể làm GPU nghỉ. Ưu tiên ảnh/giây và validation thay vì ép mọi
thời điểm hoặc mọi tài nguyên phải đạt 100%.

## Ước lượng thời gian notebook 03

Số ảnh train theo Task 1–5: 21.752 / 12.534 / 12.129 / 11.850 / 13.377;
tổng 71.642 lượt ảnh cho một epoch ở mỗi task, mỗi model. Dùng throughput Task 2
batch 8 + TF32 đã đo (B1 16,29, B2 10,56 ảnh/s) cho cả 5 task:

| Epochs mỗi task, mỗi model | Training thuần B1 | Training thuần B2 | Tổng thời gian dự kiến, hai models tuần tự |
|---|---:|---:|---:|
| 8 | 9,8 giờ | 15,1 giờ | khoảng 28–35 giờ |
| 16 | 19,5 giờ | 30,2 giờ | khoảng 55–70 giờ |
| 24 | 29,3 giờ | 45,2 giờ | khoảng 80–105 giờ |

Training thuần = epochs × 71.642 / throughput; chưa có validation, checkpoint,
khởi tạo, predictions hoặc metrics/calibration. Khoảng tổng là ước lượng vận hành
có dự phòng cho những phần đó và sai số benchmark, chưa phải wall time đã đo.
Task 1 không có teacher nên có thể nhanh hơn; Task 3–5 chưa benchmark và PPG có
thể thay đổi khi tăng số lớp hoặc khi model học tốt hơn. Không bao gồm SSH/instance
bị ngắt hay các run khảo sát bổ sung để chọn budget. Nếu ưu tiên hội tụ theo val và
cần nhiều epochs hơn, tổng thời gian tăng tương ứng; 16 epochs chưa bảo đảm hội tụ.
