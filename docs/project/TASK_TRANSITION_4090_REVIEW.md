# Notebook 02: chuyển Task 1→2 trên RTX 4090

Kiểm tra ngày **11/10/2026, UTC+7**. Campaign `vast4090-seed0-640-v1`.
Attempt `2955c937-6201-4268-a8ee-52249f1d9dcd` chạy **09:10:57 → 09:13:40**,
tổng **2 phút 43 giây**. Summary **COMPLETED**, `dry_run=false`,
`research_metrics=false`; cả hai runner exit 0, notebook không còn error output.

Kết luận: **đạt smoke chuyển task trong scope nhỏ**, chưa nghiệm thu học Task 2,
prototype cho lớp mới, chất lượng pseudo-label hoặc khả năng chống quên trên full val.

## Kết quả kỹ thuật

| Bằng chứng | EXP-B1 ResNet50 | EXP-B2 ConvNeXt |
|---|---:|---:|
| Optimizer steps | 8 | 8 |
| Teacher present / frozen | true / true | true / true |
| Teacher forward calls trong telemetry | 16 | 16 |
| PPG pseudo-labels được thêm | 62 | 46 |
| Peak allocated VRAM | 3,46 GiB | 3,87 GiB |
| Thời gian task process, gồm setup/predict | 60,5 giây | 78,1 giây |
| Prototype cũ hợp lệ | 100/100 | 100/100 |
| Prototype mới hợp lệ | 0/25 | 0/25 |

Đã đọc checkpoint và đối chiếu trực tiếp:

- Parent pilot và bản copy trong transition khớp hashes đã khóa trong plan.
- Cả 100 prototype cũ và feature caches cũ giữ nguyên giữa parent và Task 2 final.
- Backbone/encoder/decoder tensors giữ nguyên; private prompt rows của 100 lớp cũ
  giữ nguyên. 106 tensors khác thay đổi theo native config, gồm shared prompts;
  `freeze_shared_after_task1=0` của baseline vẫn có hiệu lực.
- PREV=100, CUR=25, seen=125; model/processor dựng đúng từ checkpoint.
- Telemetry đủ steps 1–8, training losses hữu hạn; có final checkpoint và val/test predictions.
- **5/5 input hashes khớp**; train GT chỉ có current-task labels 100–124.
- Không thực hiện interruption/resume Task 2 trong lượt này. `verified=false` là chưa
  kiểm tra resume, không phải một phép resume thất bại.

## Giới hạn chất lượng và metrics

Train subset chỉ **16 ảnh / 45 current-task objects / 7 trong 25 lớp mới**,
physical/effective batch 1/2, 1 epoch, correct-only prototype policy bật.
18 lớp mới không có GT trong train subset. Các lớp mới còn lại chưa tạo được prototype
hợp lệ trong 8 steps tại các batches lấy mẫu; final vẫn thiếu labels 100–124.

Val_task có 2 ảnh / 2 objects mới: AP/AP50/AP75 của lớp mới đều **0** ở cả hai models.
Đây là kết quả học lớp mới chưa đạt trong smoke, không phải bằng chứng convergence.

Test subset có **2 ảnh / 8 objects cũ / 0 objects Task 2**. Vì vậy test mAP_C và metrics
counting của nhóm mới là **NaN do không có GT**, không phải training loss NaN hoặc crash.
AP50 old=100% trên hai ảnh không chứng minh không quên trên full holdout.
Test cAcc=0% của cả hai cases cũng chỉ là diagnostic của hai ảnh.

62/46 pseudo-labels chứng minh PPG được thực thi. Chưa có phép đối chiếu từng pseudo-label
với old GT để đánh giá precision/recall; không dùng số lượng này để xếp hạng chất lượng
teacher hoặc kết luận B1 tốt hơn B2.

## Bước tiếp theo

Notebook 02 đã kiểm chứng đường teacher/PPG/current-task loader và checkpoint memory.
Không dùng checkpoint transition hoặc policy từ 2 ảnh làm baseline nghiên cứu.

Trước notebook 03 vẫn cần đo batch/throughput cho cả models (Task 2 có teacher), kiểm tra
validation learning curves để chọn budget. `baseline_epochs` tiếp tục `null`; pilot 2
epochs vẫn tăng AP nên chưa có plateau. Full baseline sẽ train lại Task 1 từ pretrained
rồi lần lượt Task 2–5 với cùng protocol/budget giữa B1/B2.

Nếu cần đánh giá chất lượng chuyển task trước full baseline, dùng một subset có coverage
đại diện cho 25 SKU mới và full-val old/new metrics trong experiment riêng, không đổi
plan hoặc subset của smoke đã hoàn tất. LoRA/K=3/FSA/synthetic tiếp tục là ablation sau baseline.

Bằng chứng:
[summary](../../data/training/vast4090-seed0-640-v1/transition/summary.json),
[plan](../../data/training/vast4090-seed0-640-v1/transition/plan.json),
[notebook 02](../../notebooks/training/02_task_transition.ipynb),
[pilot review](TASK1_PILOT_4090_REVIEW.md).
