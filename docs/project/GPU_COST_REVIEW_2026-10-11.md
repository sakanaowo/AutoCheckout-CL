# Tài nguyên PDP và lựa chọn GPU Vast.ai

**DỪNG THEO YÊU CẦU NGƯỜI DÙNG — 11/10/2026 15:08:40 UTC+7.**
Baseline ConvNeXt dừng ở Task 1, completed 11 epochs; checkpoint step 31.550
có thể resume trên GPU tương thích. **Giữ artifact**, không tự khởi chạy lại.
[Báo cáo dừng và chuyển GPU](BASELINE_4090_STOP_REVIEW_2026-10-11.md) ghi validation,
audit, manifest và điều kiện tiếp tục. ResNet chưa bắt đầu, full baseline chưa hoàn tất.

Các mô tả run đang hoạt động bên dưới là snapshot trước khi dừng.

Khảo sát **11/10/2026, UTC+7**, khi baseline ConvNeXt đang chạy Task 1. Mục tiêu:
chọn phần cứng theo bộ nhớ đủ dùng và **chi phí hoàn thành cùng budget**, thay vì
lấp đầy VRAM. Chỉ đọc telemetry/API; baseline hiện tại tiếp tục chạy.

## Ưu tiên người dùng và cập nhật lúc 15:01 UTC+7

Người dùng yêu cầu **tiết kiệm chi phí nhưng sử dụng tối đa hiệu năng**. Tiêu chí:
chi phí hoàn thành cùng budget thấp nhất, cân nhắc wall time; throughput/giá trên
PDP là quyết định cuối. Không dùng VRAM utilization hay TFLOPS quảng cáo để chốt.

Offers thay đổi trong lúc khảo sát. Snapshot mới ở
`data/training/validation/gpu_cost_review_20261011/latest_marketplace_offers.json`
và `latest_dual_gpu_offers.json` cho shortlist hiện hành:

| Ứng viên | USD/giờ | DLPerf offer | ID |
|---|---:|---:|---:|
| 1× RTX 4080 SUPER | **0,2422** | 71,82 | 46468361 |
| 2× RTX 4080 SUPER | **0,4678 tổng** | 143,63 tổng | 46468358 |
| 2× RTX 3090 | **0,3228 tổng** | 83,92 tổng | 48372895 |
| 1× RTX 4090 giá gần hiện tại | 0,4089 | 98,57 | 53860967 |

**Ưu tiên benchmark 4080 SUPER** ở snapshot mới: DLPerf/USD tham khảo cao hơn
shortlist một GPU trước đó. Máy này chỉ cần đạt 58,3% throughput hiện tại để hòa
vốn compute: ResNet 9,80 / ConvNeXt 6,38 ảnh/s. Nếu đạt 80% tốc độ hiện tại,
chi phí training tương ứng giảm khoảng 27%; đây là phép tính điều kiện, chưa phải
kết quả cross-GPU. Offer API báo VRAM khác cấu hình retail cho một số bundles;
trước thuê phải xác minh bộ nhớ mỗi GPU và nhận dạng thiết bị, không suy ra từ
tên GPU hoặc trường tổng VRAM của bundle.

Để giảm wall time hai models, khảo sát 2×4080 SUPER, mỗi model một GPU riêng.
Để hạ mức tiền mỗi giờ, 2×3090 là ứng viên. Hai phương án chưa được benchmark PDP
và cần runner có scheduling mỗi experiment một GPU; current notebook vẫn tuần tự.
Các bảng bên dưới giữ snapshot trước đó và các điều kiện hòa vốn tại thời điểm viết.

## Tài nguyên đã đo

640px, FP32 + TF32, teacher/PPG thật ở Task 2. Peak là của các phép đo đã chạy,
không phải upper bound đã chứng minh cho mọi task, độ phân giải hoặc batch.

| Model / physical = effective batch | Peak allocated | Peak reserved | Peak GPU memory used, nvidia-smi |
|---|---:|---:|---:|
| ResNet50 / 8 | 7,42 GiB | 9,09 GiB | **9,59 GiB** |
| ConvNeXt-V2-Base / 8 | 7,75 GiB | 9,22 GiB | **9,72 GiB** |
| ResNet50 / 16 | 12,30 GiB | 13,11 GiB | **13,61 GiB** |
| ConvNeXt-V2-Base / 16 | 12,55 GiB | 13,40 GiB | **13,90 GiB** |

[Benchmark gốc](../../data/training/validation/baseline_gpu_benchmark_20261011/summary.json)
và GPU CSV trong cùng folder cung cấp allocated/reserved và GPU-wide memory.
Không cộng allocated thêm vào reserved: reserved đã bao gồm allocated.
Task 3–5 và GPU khác chưa đo peak; cuDNN/allocator có thể chọn workspace khác.

Run ConvNeXt Task 1 hiện hành: allocated 6,85 / reserved 7,65 GiB, nvidia-smi
memory used khoảng 8,15 GiB. Hai lần lấy mẫu GPU utilization là 81% và 89%, power
khoảng 270–271 W trên limit 450 W. Clock quan sát 2.550 MHz, memory 10.251 MHz,
temperature 42°C. Các snapshot này không phải thống kê peak/mean toàn run.

RAM CPU cộng theo PSS của runner + training + workers khoảng **6,24–6,48 GiB**.
RSS cộng khoảng 40 GiB đếm trùng nhiều trang shared giữa các worker; không dùng
con số đó làm RAM thực cần thuê. Mẫu CPU 2 giây sử dụng khoảng **2,83 logical cores**.
Đây là mẫu Task 1, không phải peak CPU/RAM của Task 2 có teacher hoặc các lần load/save.
Chưa có CPU/PSS peak của ResNet trong full baseline hiện hành.

Sizing đề xuất để benchmark máy mới:

- Một model batch 8: **GPU 16 GB, 8–16 effective vCPU, RAM 32 GB, NVMe 80–100 GB**.
  16 GB là mục tiêu đủ dư địa từ peak đã đo, cần preflight Task 1/2 trên GPU mới.
- Hai model trên hai GPU riêng: mỗi GPU ít nhất 16 GB, khoảng 24–32 effective vCPU,
  RAM 64 GB, NVMe ít nhất 100 GB; mỗi model giữ batch/effective 8/8.
- Batch 16: đã đo dưới 14 GiB trên 4090, nhưng tăng effective batch sẽ đổi số
  optimizer updates; không sửa batch/epochs trong phase baseline đang khóa.

## Vì sao VRAM không cho biết GPU đang dùng bao nhiêu TFLOPS

82,97 TFLOPS là thông số đỉnh được Vast báo cho GPU hiện tại. Chưa có đo FLOP/s
thực hoặc SM/Tensor Core occupancy của PDP. Không suy ra GPU chỉ dùng 1/3 compute
từ 8/24 GiB VRAM; cũng không nhân 82,97 với nvidia-smi utilization để ra FLOP/s.
Utilization của nvidia-smi là thời gian có kernel đang chạy, theo
[NVIDIA](https://docs.nvidia.com/deploy/nvidia-smi/index.html), không phải phần trăm
hiệu suất arithmetic đỉnh. Matching, launch/synchronization, bandwidth và input
pipeline đều có thể giới hạn throughput.

Đã thử hai model trên một 4090: 19,30 GiB memory, GPU gần 99%, nhưng ResNet
16,80→6,77 và ConvNeXt 10,94→5,44 ảnh/s. Vì vậy một GPU đủ VRAM cho hai process
không đồng nghĩa lợi về thời gian. Xem [benchmark đồng thời](BASELINE_4090_CONFIGURATION.md).

## Giá thuê và chất lượng host

Nguồn [Vast REST API](https://vast.ai/developers/api), public offers và read-only
current instance. [Snapshot tính toán](../../data/training/validation/gpu_cost_review_20261011/cost_comparison.json)
giữ giá, query, offer IDs và điều kiện hòa vốn; raw responses lưu trong cùng folder.
Một GPU: verified, on-demand, reliability ≥0,99, RAM ≥32 GB, effective vCPU ≥8,
disk ≥80 GiB và bandwidth ≥300 MB/s, driver hỗ trợ CUDA ≥12,1. Hai GPU dùng RAM
≥64 GB, effective vCPU ≥24, storage pricing 100 GiB. Giá gồm allocated storage
theo query; current instance dùng giá allocation thực. Không bao gồm phí data transfer.

| Offer | GPU memory | USD/giờ | DLPerf của offer | ID |
|---|---:|---:|---:|---:|
| Instance hiện tại, RTX 4090 | 24 GB | **0,4153** | **49,57** | 55154372 |
| RTX 4090, host khác | 24 GB | 0,4222 | 97,33 | 25753562 |
| RTX 4090, cao nhất DLPerf/USD trong shortlist một GPU | 24 GB | 0,4593 | 123,28 | 47635768 |
| RTX 4070 Ti SUPER | 16 GB | 0,2190 | 50,85 | 55340748 |
| RTX 4080 SUPER | 16 GB | 0,2489 | 55,65 | 51999238 |
| RTX 3090 Ti | 24 GB | 0,2222 | 46,19 | 35612014 |
| RTX A4000 | 16 GB | 0,1055 | 21,17 | 43678946 |
| 2× RTX 3090 | 24 GB mỗi GPU | **0,3228 tổng** | 83,92 tổng offer | 48372895 |
| 2× RTX 4070 Ti SUPER | 16 GB mỗi GPU | **0,4250 tổng** | 104,59 tổng offer | 55340755 |

Current instance có GPU compute 0,4000 USD/h + storage 0,0153 USD/h, 32 effective
vCPU và khoảng 504 GiB RAM được báo bởi API. CPU/RAM lớn không nhất thiết là phần
được tính tiền riêng; giá thuê của cả bundle mới quyết định chi phí.
DLPerf hiện tại 49,57 và GPU memory bandwidth metadata 386,7 GB/s thấp hơn các
offer 4090 khác (97–123, khoảng 875–879 GB/s). Đây là tín hiệu chọn host để thử,
không chứng minh GPU hỏng, không xác định nguyên nhân, không bảo đảm PDP nhanh gấp đôi.

Vast định nghĩa [DLPerf](https://console.vast.ai/faq/) là dự đoán gần đúng cho
workloads DL phổ biến, có thể kém chính xác với task khác. PDP có attention kernel,
teacher/PPG và frozen components, nên chỉ dùng điểm đó để shortlist.

## Tính ngưỡng tiết kiệm bằng tốc độ PDP

Với cùng số ảnh/budget: chi phí training tỷ lệ **USD/giờ ÷ ảnh/giây**.
GPU mới tiết kiệm nếu tốc độ tương đối cao hơn tỷ lệ giá mới/giá hiện tại.
Benchmark Task 2 chạy riêng 60 giây: ResNet **16,80**, ConvNeXt **10,94 ảnh/s**.

| GPU mới | Phải đạt ít nhất tốc độ hiện tại | ResNet tối thiểu | ConvNeXt tối thiểu |
|---|---:|---:|---:|
| 4090 host 0,4222/h | 101,7% | 17,08 ảnh/s | 11,13 ảnh/s |
| 4090 host 0,4593/h | 110,6% | 18,58 ảnh/s | 12,10 ảnh/s |
| 4070 Ti SUPER 0,2190/h | 52,7% | 8,86 ảnh/s | 5,77 ảnh/s |
| 4080 SUPER 0,2489/h | 59,9% | 10,07 ảnh/s | 6,56 ảnh/s |
| 3090 Ti 0,2222/h | 53,5% | 8,99 ảnh/s | 5,86 ảnh/s |
| A4000 0,1055/h | 25,4% | 4,27 ảnh/s | 2,78 ảnh/s |

Các ngưỡng này chỉ cho compute/throughput, chưa cộng chuyển dữ liệu, setup,
validation, calibration và chi phí giữ GPU idle. Đánh giá cuối bằng thời gian và
tiền của cùng công việc, không chỉ training FPS. Budget 50–65 giờ ở giá hiện tại
tương ứng **20,76–26,99 USD** cho toàn run, không phải tiền còn lại kể từ lúc khảo sát.

## Khuyến nghị

1. Nếu giữ một GPU, **ưu tiên benchmark host 4090 tốt hơn**, trước khi hạ GPU:
   giá gần hiện tại, DLPerf tham khảo cao hơn rõ. Offer 25753562 cần nhanh hơn
   khoảng 1,7% để tiết kiệm; 47635768 cần nhanh hơn 10,6%.
2. Nếu muốn hai model cùng chạy và giảm wall time, **shortlist 2×3090 0,323 USD/h**
   hoặc **2×4070 Ti SUPER 0,425 USD/h**. Mỗi model dùng GPU riêng. Cần benchmark
   đồng thời trên hai GPU thật và đổi runner: notebook hiện tại tuần tự, không tự
   phân phối hai experiments khi máy có thêm GPU. DLPerf của bundle là tổng offer,
   không dùng nguyên số đó làm tốc độ mỗi GPU.
3. Nếu chọn một GPU 16 GB, **4070 Ti SUPER/4080 SUPER** là ứng viên phù hợp bộ nhớ;
   chỉ chốt hiệu quả sau khi đạt ngưỡng FPS hòa vốn ở cùng batch 8 và cùng inputs.
   A4000 có giá giờ thấp nhất nhưng điểm throughput proxy thấp, có thể kéo dài run.

Chưa thể tuyên bố một GPU tốt nhất cho PDP chỉ từ specs hoặc DLPerf. Chốt bằng
benchmark isolated Task 1/2 trên máy ứng viên, kiểm tra native kernel, loss hữu
hạn, VRAM và tốc độ, giữ nguyên effective batch/epochs/precision/data. Tiếp tục
kiểm tra peak khi tới Task 3–5. Không đổi cấu hình hoặc thuê/chuyển instance trong
lượt khảo sát này; current baseline vẫn chạy.

A100/H100 trong query mở rộng có giá từ khoảng 0,719/2,070 USD/h; memory 40–80 GB
chưa có nhu cầu ở batch 8. RTX 50-series không là drop-in cho env PyTorch 2.2.2+cu121:
[PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) bổ sung Blackwell với CUDA 12.8;
cần env riêng và regression custom kernel trước khi cân nhắc.
