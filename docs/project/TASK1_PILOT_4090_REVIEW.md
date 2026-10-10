# Pilot Task 1 trên RTX 4090 — kết quả và hướng tiếp theo

Ngày kiểm tra: **10/10/2026, UTC+7**. Campaign `vast4090-seed0-640-v1`, source commit
`7edc557`. Pilot **COMPLETED** lúc **18:14:57**, executor exit **0**; hai runner hoàn tất
với exit 0. Bắt đầu lượt đầu lúc 13:46:28, tổng thời gian thực tế **4 giờ 28 phút 29 giây**,
gồm interruption/resume, train, validation, predictions và evaluation.

Đã hoàn tất pilot kỹ thuật và có kết quả Task 1. **Chưa nghiệm thu convergence,
Task 1→2 hoặc baseline 5 task.** `baseline_epochs` tiếp tục để `null`.

## 1. Phạm vi và bằng chứng

- Hai cases: PDP ResNet50 / ConvNeXt-V2-Base; pretrained detector/backbone theo revisions
  khóa trong workflow, không phải hai models có cùng nguồn pretraining.
- Train Task 1: **21.752 ảnh, 100 SKU**; 640px, FP32, TF32 tắt, physical batch 1,
  effective batch 2, 2 epochs, **21.752 optimizer steps/model**.
- Freeze backbone/encoder/decoder, tối ưu theo native PDP config. LoRA, prototype K=3,
  synthetic và freeze bổ sung chưa bật.
- Release real-only, protocol 100+4×25; giữ split nghiên cứu đã khóa.
  Đây không phải official RPC test2019 benchmark.
- **24/24 input hashes khớp** plan; checkpoint/val annotation/val prediction hashes
  khớp calibration policy của cả hai cases. Policy chọn trên **val**, oracle tắt.
- Các loss trong telemetry hữu hạn; đủ final checkpoint, val/test predictions và metrics.
- B1 bị ngắt ở step 5.815, resume từ step 5.568; model/optimizer/scheduler/PDP state
  khớp sau restore. Có **247 telemetry rows chạy lại**; đồ thị loại trùng bằng cách giữ
  lần cuối của mỗi optimizer step. Shuffle bật nên không khẳng định thứ tự minibatch
  sau resume giống một lượt chạy liên tục.
- B2 không bị ngắt; `resume_verification.verified=false` có nghĩa chưa thực hiện resume,
  không phải một phép kiểm tra resume bị thất bại.

Bằng chứng:
[summary](../../data/training/vast4090-seed0-640-v1/pilot/summary.json),
[phân tích JSON](../../data/training/vast4090-seed0-640-v1/pilot/analysis/pilot_review.json),
[notebook có outputs](../../data/training/vast4090-seed0-640-v1/notebook_executions/01_task1_pilot/4a115078-667f-4397-bc63-4d92af149df3/notebook_executed.ipynb).

## 2. Kết quả Task 1

AP trong bảng là **M1 mAP@A** trên 100 lớp đã thấy: 1.464 ảnh val / 5.766 ảnh test
có GT của nhóm lớp này. Counting dùng toàn bộ 1.503 val / 6.003 test và chỉ đếm 100
SKU đã thấy. AP50 và cAcc là tỷ lệ; ACD là tổng sai lệch count theo SKU trung bình/ảnh.

| Chỉ số | EXP-B1 ResNet50 | EXP-B2 ConvNeXt |
|---|---:|---:|
| Val AP, epoch 1 → 2 | 16,64% → 28,17% | 52,86% → 73,62% |
| Val AP50, cuối pilot | 33,84% | 87,94% |
| Test AP | 26,88% | 73,54% |
| Test AP50 | 32,30% | 87,84% |
| Val cAcc | 3,99% | 30,07% |
| Test cAcc | 3,50% | 30,57% |
| Test ACD, thấp hơn tốt hơn | 5,63 | 1,73 |
| Test mCCD, thấp hơn tốt hơn | 0,9390 | 0,2951 |
| Test mCIoU, cao hơn tốt hơn | 0,1510 | 0,7302 |
| Test cAcc easy / medium / hard | 8,67% / 1,63% / 0,20% | 46,56% / 29,71% / 15,55% |
| Score threshold chọn trên val | 0,29 | 0,35 |
| Counting NMS IoU | 0,45 | 0,45 |
| Peak allocated VRAM | 2,82 GiB | 3,00 GiB |
| Prototype hợp lệ cho Task 1 | 100/100 | 100/100 |

M2 có protocol xử lý predictions đè lên SKU chưa học khác M1. Test M2 AP/AP50:
B1 **34,65%/41,69%**, B2 **75,12%/89,76%**; không trộn chúng với M1 trong so sánh.

![Validation, classification loss và counting](../../data/training/vast4090-seed0-640-v1/pilot/analysis/pilot_comparison.png)

## 3. Diễn giải

**B2 là cấu hình tốt hơn rõ rệt trong pilot hiện tại.** Test AP cao hơn B1 khoảng
46,66 điểm phần trăm và test cAcc cao hơn 27,07 điểm. Đây là kết quả của hai cấu hình
pretrained/backbone cụ thể, một seed và budget 2 epochs; chưa chứng minh chất lượng
chống quên hoặc kết luận chung về mọi cấu hình ResNet/ConvNeXt.

**Cả hai vẫn đang học.** Val AP tăng 11,53 điểm ở B1 và 20,76 điểm ở B2 từ epoch 1
sang epoch 2. Hai điểm validation chưa cho thấy plateau để chọn một số epochs cuối
cùng. Median classification loss của 500 steps cuối giảm từ 0,440 xuống 0,363 ở B1
và 0,258 xuống 0,131 ở B2. Tổng loss bị query loss chi phối: median QL khoảng 75,
nhân lambda 0,1 đóng góp khoảng 7,5; không đọc tổng loss đơn độc làm thước đo detector.

**Counting còn là hạn chế.** cAcc yêu cầu count vector của toàn bộ SKU đã thấy trong
một ảnh khớp GT; AP cao không đảm bảo checkout đúng toàn bộ. B2 vẫn sai count vector
ở khoảng 69,4% ảnh test; cAcc hard chỉ 15,55%.

Tái kiểm tra trên val với đúng policy đã khóa cho thấy:

| Diagnostic val | B1 | B2 |
|---|---:|---:|
| GT objects thuộc 100 lớp đã thấy | 9.161 | 9.161 |
| Predicted objects sau NMS/threshold | 2.974 | 7.844 |
| Tổng count thiếu theo cặp ảnh/SKU | 7.357 | 2.006 |
| Tổng count thừa theo cặp ảnh/SKU | 1.170 | 689 |
| Số SKU không có prediction qua threshold | 12 | 1 |

Count thiếu/thừa có thể do bỏ sót, sai SKU hoặc hậu xử lý; các số này không tự phân
rã thành lỗi định vị/classification. RPC category **108** (model label **54**) có 72
object val nhưng 0 predictions qua threshold ở cả hai models. Cần kiểm tra scores,
nhầm lớp và ảnh của SKU này trên val trước khi kết luận nguyên nhân.

**Đủ prototype chưa bảo đảm prototype tốt.** Tất cả 100 vectors có 256 dimensions,
hữu hạn và norm khác 0. B1 có cache từ **3–100 mẫu/lớp**, label 54 chỉ 3 mẫu; B2 có
100 mẫu/lớp. Việc label 54 của B2 vẫn không vượt threshold trên val cho thấy coverage
và chất lượng prediction là hai bằng chứng khác nhau. Correct-only cache policy đã bật.

CSV diagnostic:
[B1 theo SKU](../../data/training/vast4090-seed0-640-v1/pilot/analysis/EXP-B1_val_count_by_sku.csv),
[B2 theo SKU](../../data/training/vast4090-seed0-640-v1/pilot/analysis/EXP-B2_val_count_by_sku.csv).

## 4. Notebook tiếp theo: Task 1→2

Có đủ điều kiện đầu vào để chạy
[02_task_transition.ipynb](../../notebooks/training/02_task_transition.ipynb).
Giữ settings/campaign hiện hành, `dry_run=false`, batch 1/effective 2, resolution 640;
chạy cả B1 và B2 từ checkpoint pilot tương ứng. Notebook copy parent vào transition
folder và không sửa checkpoint pilot/release gốc.

Đặt `AUTOCHECKOUT_TRAINING_SETTINGS` tới `data/training/settings_4090.json` nếu chạy IDE.
Nếu chạy executor, dùng `--settings data/training/settings_4090.json` như runbook;
không dùng mặc định dry-run template.

Scope mặc định **16 ảnh train Task 2, 1 epoch, 8 optimizer steps/model**, 2 ảnh cho
mỗi val/prediction subset. First-16 train subset hiện có 45 object của **7/25 lớp mới**,
10/16 ảnh có 26 object thuộc 6 lớp cũ theo full GT dùng riêng để audit. Loader train
chỉ đọc current-task GT. Scope này kiểm tra chuyển task, không đo convergence Task 2,
không nghiệm thu đủ prototype cho toàn bộ 25 lớp mới và không đánh giá forgetting
trên full holdout.

Bằng chứng cần đọc sau notebook 02:

1. Parent checkpoint/hash đúng; model/processor dựng đúng; PREV=100 và seen=125.
2. Teacher present, frozen và được gọi; training losses hữu hạn khi đường teacher/PPG hoạt động.
3. Prototype cũ được mang sang; ghi missing prototypes của lớp mới và PPG pseudo-label
   counters. Nếu counter bằng 0, báo đúng và kiểm tra teacher scores/threshold trên
   phạm vi audit, không hạ ngưỡng chỉ để ép counter dương.
4. Loader chỉ có labels 100–124, release/holdout và parent hashes không đổi.
5. Có final checkpoint, predictions, telemetry và COMPLETED summary. Metrics của
   2 ảnh chỉ là diagnostics, không thay cho đánh giá old/new trên full val.

## 5. Trước notebook baseline 5 task

- Giữ B1 làm baseline đối chiếu; ưu tiên phân tích B2 nhưng chưa bỏ B1 chỉ từ pilot.
- Hoàn tất transition smoke trước; không dùng pilot Task 1 thay một baseline đã hội tụ.
- Quét batch/throughput riêng cho cả hai models, gồm 2/2 để đo bỏ accumulation và các
  batch 4/8/16 phù hợp VRAM. Task 2 có teacher và prompt memory nên cần đo lại VRAM;
  không suy thẳng khả năng chứa batch từ Task 1.
- Giữ cùng data/resolution/effective batch và budget giữa B1/B2; tăng effective batch
  làm giảm số optimizer steps/epoch. Batch/precision thay đổi cần experiment/campaign
  riêng và ghi provenance, không sửa plan đã hoàn tất.
- TF32/AMP chỉ thử sau finite-loss/kernel checks theo kế hoạch. Tối ưu cũng cần xem
  loader và các đồng bộ scalar/CPU trong matching/logging.
- Chạy thêm validation epochs theo các budget đã định trước để tìm plateau; không
  chốt `baseline_epochs=2`, cũng không chọn budget bằng test. Nếu so nhiều budget,
  dùng campaign riêng cho từng config và tính tới prototype cập nhật ở epoch cuối.
- Đánh giá sâu các lớp yếu và counting trên val; mọi calibration mới ghi policy mới,
  không thay artifact 0,29/0,35 hiện tại để khớp test.

Kết luận vận hành: **notebook 02 sẵn sàng chạy smoke**, còn **notebook 03 chưa có budget
convergence được chốt**. LoRA/K=3/FSA/synthetic tiếp tục để sau baseline.
