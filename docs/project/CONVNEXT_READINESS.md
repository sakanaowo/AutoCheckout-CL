# ConvNeXt-V2-Base và điều kiện training

Cập nhật 10/10/2026. **Adapter đã nghiệm thu CUDA trên RTX 3060.** Notebook 03 đạt `PASS_CONVNEXT_ADAPTER_CUDA_SMOKE`; phần CLI/runner tiếp nối được theo dõi trong [nghiệm thu runtime](TRAINING_RUNTIME_ACCEPTANCE.md). Full training, calibration và môi trường Vast.ai có tiêu chí riêng.

## Bằng chứng adapter

Run `aeeb770b-7e25-4ad2-92bd-053c1bacf6e6`, **09/10/2026 23:16:09–23:17:32 UTC+7**, ghi đủ **8/8 gates PASS** và **29 regression tests**, không failures/errors/skips. Summary, JUnit, checkpoints và hashes weights đã được đọc/đối chiếu trên máy chạy; artifacts nằm tại `runs/modeling/convnext_adapter/<run_id>/`.

| RTX 3060, batch 1 FP32 | Peak allocated | Kết quả |
|---|---:|---|
| 640 | 3,683 GiB | PASS |
| 800 | 4,416 GiB | PASS |

Đã kiểm chứng metadata features channels 256/512/1024, strides 8/16/32, projections về 256 dimensions và level thứ tư stride 64; masks/normalization; finite losses/gradients, optimizer update, frozen hashes; Lightning state restore và cold model/config reload. Mỗi resolution có 5 optimizer steps trong notebook 03. Đây là Task 1 smoke trên 2 ảnh thật, chưa phải convergence hoặc học đủ prototypes 100 SKU.

Backbone `timm/convnextv2_base.fcmae_ft_in22k_in1k`, revision `9e250ed8f88b436472d2b24af26f82a8aa8c719d`. Detector `SenseTime/deformable-detr`, revision `83ecd26945199939cb82806f988debdb71e6f43e`. Backbone pretrained được giữ nguyên qua detector construction/transfer; input projections/classifier/prompts khởi tạo mới, chỉ chuyển detector core tương thích. Các cơ chế này nằm ở [backbones.py](../../pdp/models/backbones.py) và được CLI dùng qua [runtime.py](../../pdp/runtime.py).

Run dùng **PyTorch CUDA fallback**; native kernel chưa đạt, log environment ghi thiếu `CUDA_HOME`. Đây không phải lỗi OOM hoặc lỗi adapter. Kết quả 3060 không chứng nhận toolkit/compiler trên máy Vast.ai.

## Cách chạy lại

1. Mở [notebook 03](../../notebooks/modeling/03_convnext_v2_base_acceptance.ipynb), chọn Conda `pdp` Python 3.10 và restart kernel.
2. Dùng release native đã khóa và summary PASS của notebook 02. Có thể đặt `AUTOCHECKOUT_BASELINE_SUMMARY`, `AUTOCHECKOUT_DATA_ROOT`, `AUTOCHECKOUT_RELEASE_ROOT` nếu chuyển máy.
3. Giữ batch 1, FP32, 640/800. Cache weights đã có trên máy hiện tại; `offline=True` chỉ dùng cache, `False` cho phép tải public theo revision khóa. Notebook không cài package.
4. Run All, lưu notebook có outputs và giữ run artifacts. Không đổi run FAIL cũ thành PASS.

## Việc tiếp theo

- [Notebook 04](../../notebooks/modeling/04_training_runtime_acceptance.ipynb) kiểm chứng cấu hình này qua CLI/runner thật, interruption/resume, predict-only và bbox native. Đọc [báo cáo runtime](TRAINING_RUNTIME_ACCEPTANCE.md) để lấy kết quả mới nhất.
- [S6 kỹ thuật đã đạt](COUNT_CALIBRATION_ACCEPTANCE.md): threshold/NMS chọn trên val, lưu/nạp policy cho test, oracle mặc định tắt. Sinh policy mới từ val của checkpoint pilot thật.
- Người dùng đã bỏ notebook nghiệm thu GPU/toolchain Vast.ai riêng ngày 10/10/2026. Ghi GPU/versions/kernel mode và số đo runtime trong pilot; native kernel chưa có evidence nghiệm thu mới.
- Pilot EXP-B1 ResNet/EXP-B2 ConvNeXt cùng split/resolution/optimizer-step budget/evaluator, rồi kiểm chứng full teacher/PPG Task 1→2 trước 5 task.

Không cần chờ synthetic pending để chạy baseline real-only. LoRA/K=3/FSA/freeze bổ sung và generator overlap là ablation sau baseline. Tests 23 contract ở commit `41827ed` là bằng chứng trước runtime; run 29 tests và CUDA ở trên là bằng chứng mới hơn.
