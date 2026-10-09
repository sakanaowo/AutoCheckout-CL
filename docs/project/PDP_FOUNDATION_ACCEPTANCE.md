# Kết luận nghiệm thu processor/loader và PDP nền

Cập nhật **09/10/2026**, timezone **Asia/Bangkok**. **Hoàn tất phạm vi tạo notebook, kiểm chứng processor/loader/PDP nền và full pretrained CUDA smoke.** Báo cáo này ghi phạm vi notebook 01–02. Phần tiếp nối đã có [nghiệm thu ConvNeXt](CONVNEXT_READINESS.md) và [CLI/runner notebook 04](TRAINING_RUNTIME_ACCEPTANCE.md); dùng hai báo cáo đó cho trạng thái mới nhất.

## Bằng chứng đã đạt

| Phần | Kết quả được chứng minh | Run / thời gian thực |
|---|---|---|
| [Notebook 01](../../notebooks/modeling/01_processor_loader_pdp_acceptance.ipynb) | `PASS_LIMITED_PDP_FOUNDATION`, **11/11 gates**; release/checksums/mapping, current-task GT separation trên 5 task JSON; processor và loader 640/800, CPU tiny PDP losses/gradient/optimizer/freeze và resume; sources/holdout giữ nguyên | `65687942-4332-40db-a82e-76ae0429c100`; 09/10, **15:12:32–15:13:43 UTC+7** |
| [Notebook 02](../../notebooks/modeling/02_full_pdp_baseline_acceptance.ipynb) | `PASS_FULL_PDP_CUDA_SMOKE`, **7/7 gates**; full pretrained ResNet50/PDP trên CUDA 640/800, finite losses/gradient, classifier cập nhật, frozen hashes giữ nguyên, prediction reload tương đương; Lightning restore model/optimizer/scheduler/PDP state và global step | `d967dc7b-d665-4a8b-9ca0-0e27895e4bdc`; 09/10, **16:12:09–16:12:57 UTC+7** |
| Regression notebook 01 | **27/27 baseline tests + 1/1 Lightning resume test**, 0 failures/errors/skips; tests teacher/PPG/prototypes/task transition dùng fixture nhỏ | JUnit trong run 65687942 |
| Công cụ acceptance/downloader | **18/18 tests CPU+GPU**, 0 failures/errors/skips, gồm regression tái hiện và sửa lỗi CPU/CUDA sau Lightning teardown | `acceptance_regression.xml` trong run d967dc7b |
| [Downloader public](../../tools/download_pdp_pretrained.py) | Tải đúng 3 files detector, tự nhận cache path, pinned revision, SHA256 và offline reuse; không cần tải dataset COCO | `SenseTime/deformable-detr`, revision `83ecd26945199939cb82806f988debdb71e6f43e` |

Notebook 01 dùng random ResNet18 nhỏ ở 96 px trên CPU. Processor 640/800 kiểm tra **2 ảnh thật mỗi task**, normalization, aspect ratio, bbox round-trip/postprocess, padding/mask, collation và shuffle tái lập. Train chỉ chứa current-task labels; full GT chỉ dùng audit.

Notebook 02 dùng full ResNet50 pretrained detector COCO, encoder/decoder **6/6**, **300 queries**, **d_model 256**, **4 feature levels**, prompt **100×10**, DDL 0.15 và query loss 0.1. Giữ freeze backbone/encoder/decoder theo baseline; LoRA/K=3/FSA/freeze shared bổ sung tắt. Batch là **2 ảnh thật train Task 1**, physical batch **1**, FP32, TF32 tắt; mỗi resolution có **5 optimizer steps** gồm manual step và Lightning global step 1→4.

| RTX 3060 12 GB, batch 1 FP32 | Peak allocated VRAM | Kết quả |
|---|---:|---|
| 640 | **3,49 GiB** | PASS |
| 800 | **4,22 GiB** | PASS |

RTX 3060 này đủ cho smoke trên; số đo chưa xác nhận batch lớn, teacher full model ở Task 2+, ConvNeXt hoặc full training đủ VRAM. **Native CUDA kernel chưa đạt; run dùng PyTorch CUDA fallback.**

## Artifacts và giới hạn kết luận

Artifacts local, không đưa vào git:

- `runs/modeling/processor_loader_pdp/65687942-4332-40db-a82e-76ae0429c100/`: config/run/gates/summary, baseline/resume logs và JUnit.
- `runs/modeling/full_pdp_baseline/d967dc7b-d665-4a8b-9ca0-0e27895e4bdc/`: summary/gates, 640/800 results/load reports, frozen hashes, configs/checkpoints và worker logs; **notebook_executed.ipynb** có outputs PASS, **acceptance_regression.xml** có 18 tests.
- `runs/pretrained/`: detector cache và receipt/checksums. Worker model dùng local files sau bước tải public.

Các run FAIL trước được giữ nguyên. Outputs lỗi trong file notebook 02 gốc thuộc run cũ; bằng chứng PASS nằm trong bản executed của run d967dc7b. [Nhật ký](../timelines/model-foundation-2026-10-09.md) giải thích sửa Ninja metadata, validator FrozenBatchNorm counters và device sau interruption.

Kết luận này chứng nhận foundation và full pretrained **Task 1 smoke**. Chưa chứng nhận convergence, mAP/cAcc, tái lập bảng benchmark của paper, full teacher/PPG trên 5 task, native kernel, ConvNeXt, calibration/counting hoặc môi trường Vast.ai. Prototype state restore không chứng nhận đã học đủ prototype cho 100 SKU. CLI/runner defaults chưa được notebook có processor explicit này nghiệm thu. Full test suite vẫn chưa được xác nhận.

Protocol giữ **100+4×25, seed 0, 224 slots/225 outputs**, real-only release và holdout đã khóa. Đây là split nghiên cứu có cross-task image reuse với current-task GT; không đổi protocol/split để chạy tiếp.

## Công việc tiếp nối được xác định khi kết thúc notebook 02

| Thứ tự | Công việc | Điều kiện nghiệm thu |
|---|---|---|
| **1 — Khép phần runtime của S1** | Nối resolution explicit vào `pdp/main.py`/`engine.py` cho train/val/predict, checkpoint lưu processor config; cấu hình runner trỏ release hiện hành, seed/protocol đúng; kiểm tra effective batch và đếm optimizer steps thực tế | Notebook kiểm chứng đường CLI/runner thật: bbox về native coordinates, cùng processor policy 640/800, loss/optimizer/resume và run artifacts. Không dùng legacy paths/GCP defaults |
| **2 — S2, adapter ConvNeXt-V2-Base** | Factory lấy channels/strides/stage indices từ timm, nối feature masks và input projections; nạp backbone pretrained riêng, chỉ chuyển tensor detector tương thích và ghi load report | Notebook adapter: feature shapes/masks ở 640/800, finite loss/gradient, checkpoint/config reload tương đương. Giữ ResNet baseline để đối sánh; không dùng nguyên ResNet out_indices cho ConvNeXt |
| **3 — Chuẩn bị training và evaluation** | Kiểm chứng GPU/driver/toolkit/compiler/native kernel trên RTX 4090 Vast.ai do người dùng chuẩn bị; hoàn thiện S6 lưu/nạp calibration policy, chọn threshold trên val rồi khóa cho test, tắt oracle trong pipeline chính | Notebook môi trường/kernel có forward/backward equivalence; evaluator fixtures đúng và calibration provenance. Smoke fallback trên 3060 chưa nghiệm thu toolchain Vast.ai |
| **4 — E1, pilot Task 1** | EXP-B1 PDP ResNet và EXP-B2 PDP ConvNeXt trên cùng real-only split/resolution/optimizer-step budget/evaluator; đo batch/VRAM/latency và loss/val curves | Có checkpoints/resume, số steps thực, metrics val, load reports, timestamps và notebook outputs. Chọn budget dựa trên pilot; không đổi đồng thời backbone và ablation |
| **5 — E2, đủ 5 task** | Kiểm chứng full-model Task 1→2 teacher/PPG/prototype state và current-task-only loader trước khi chạy đủ protocol | Metrics sau mỗi task, forgetting/counting, policy val được áp cố định lên test và artifacts tái lập; chỉ từ đây mới có kết quả nghiên cứu đầy đủ |

Synthetic 202/20 pilot pending xử lý ở track riêng, không chặn baseline real-only. LoRA/K=3/FSA/freeze shared bổ sung và generator overlap là ablation sau baseline, mỗi thay đổi có run riêng. [Kế hoạch](IMPLEMENTATION_PLAN.md) giữ tiêu chí từng milestone; [PROGRESS](PROGRESS.md) ghi trạng thái hiện hành.


Notebook S2 [03 ConvNeXt-V2-Base](../../notebooks/modeling/03_convnext_v2_base_acceptance.ipynb) đã đạt CUDA 640/800, 8/8 gates và 29 tests trong run aeeb770b. CLI/runner tiếp nối được ghi trong [runtime acceptance](TRAINING_RUNTIME_ACCEPTANCE.md). [Đối chiếu điều kiện/train readiness](CONVNEXT_READINESS.md) ghi phạm vi và các gate còn lại; trạng thái này không thay đổi kết quả baseline đã đạt ở trên.
