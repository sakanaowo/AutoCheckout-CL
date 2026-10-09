# Điều kiện ConvNeXt-V2-Base và training Vast.ai — 09/10/2026

**Có thể bắt đầu nghiệm thu adapter ở notebook 03. Chưa đủ điều kiện xác nhận train dài ConvNeXt trên Vast.ai.** Notebook đã được tạo, chưa chạy; sửa giao diện backbone và policy warm-start có tests nhẹ, nhưng runtime CUDA/weights của ConvNeXt vẫn pending.

## Đối chiếu phần 1 và 2

[Kết luận foundation](PDP_FOUNDATION_ACCEPTANCE.md) và [báo cáo thực chạy](../timelines/model-foundation-2026-10-09.md) ghi:

- Environment Conda Python 3.10.22/pins đã PASS trên máy RTX 3060.
- Notebook 01: `PASS_LIMITED_PDP_FOUNDATION`, processor/loader 640/800, GT separation 5 task, tiny CPU smoke/optimizer/resume; baseline regression 27/27 và resume 1/1.
- Notebook 02: full pretrained ResNet50/PDP CUDA 640/800, batch 1 FP32, losses/gradient/optimizer/frozen hashes/Lightning restore PASS. Native kernel chưa đạt; dùng PyTorch CUDA fallback.
- File notebook 02 tracked hiện có output của run **ed32adc8-9d66-403f-9016-919f3c8f9363**, 09/10 **16:17:51–16:19:09 UTC+7**, `PASS_FULL_PDP_CUDA_SMOKE`, peak allocated 3,487/4,224 GiB. Báo cáo trước đó ghi run d967dc7b cũng PASS; giữ cả hai làm lịch sử.

Lượt rà soát này đọc docs và output notebook trong repo; **run artifacts/pretrained cache của các lượt CUDA không có trong workspace hiện tại**, nên chưa tái kiểm tra hashes/JUnit/checkpoint của máy đã chạy và không chạy lại model. `.venv` ở workspace này thiếu torch/timm/numpy/pytest; không cài packages theo yêu cầu người dùng. Khi chạy notebook 03 ở máy khác, cần summary/gates thật của baseline và release/weights tương ứng, hoặc chạy lại notebook 02.

Phạm vi 1–2 đã đủ để tiến hành adapter riêng; **S1 tổng thể vẫn còn đường CLI/runner**. Smoke không chứng nhận convergence, full-model teacher Task 2+, native kernel hoặc môi trường Vast.ai.

## Phần 3 đã chuẩn bị

| Thành phần | Triển khai / kiểm chứng hiện tại |
|---|---|
| [Notebook 03](../../notebooks/modeling/03_convnext_v2_base_acceptance.ipynb) | Environment/baseline/release/weights → CUDA 640/800 → source preservation; run ID/timestamps/gates/logs; chưa thực chạy |
| [backbones.py](../../pdp/models/backbones.py) | Chọn stages/channels/strides từ `feature_info`; giữ 3 stage cuối cho Deformable DETR, hỗ trợ checkpoint metadata và local pretrained file |
| [ConvEncoder](../../pdp/models/modeling_deformable_detr.py) | Gọi factory metadata, lấy đúng feature outputs và downsample masks; bỏ assumptions out_indices của ResNet cho ConvNeXt |
| `plan_detector_transfer` | Chuyển tensor detector core tương thích; reset input projections/classifier/prompts, giữ ConvNeXt backbone riêng; từ chối core thiếu/sai shape/unexpected |
| [tests](../../tests/test_backbone_adapter.py) | Tái hiện lỗi index 4 ở ConvNeXt trước sửa; kiểm tra contract 640/800, giữ ResNet/single-level, metadata reload, warm-start/core rejection và local-file provider |
| [RunEvidence](../../autocheckout/model_acceptance.py) | Pass status có scope riêng cho ConvNeXt, giữ default notebook 02; [test](../../tests/test_model_acceptance.py) |

Nguồn backbone: [model card timm](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k) và [timm v0.9.12](https://github.com/huggingface/pytorch-image-models/blob/v0.9.12/timm/models/convnext.py). Tag model đã có trong phiên bản đang pin. Notebook khóa revision **9e250ed8f88b436472d2b24af26f82a8aa8c719d**; detector COCO giữ revision **83ecd26945199939cb82806f988debdb71e6f43e**. Khi người dùng chạy, notebook tải/reuse config+safetensors public, ghi SHA256 và dùng local file cho constructor; không cài package. Có offline mode.

Theo metadata/model card, ConvNeXt-V2-Base có 4 native stages channels 128/256/512/1024, strides 4/8/16/32. Adapter chọn channels 256/512/1024, strides 8/16/32; projections cho detector có d_model 256 và thêm level stride 64. Notebook sẽ kiểm tra cả 4 projection outputs, mask geometry/binary, normalization, finite losses/gradients/optimizer/freeze, Lightning resume và **cold model/config reload**.

Test chạy local chỉ dùng feature-provider doubles và tensor-shape policy; không giả định có torch/timm, không chứng nhận pretrained features thật hoặc CUDA. Regression downloader hiện cần huggingface_hub và regression model cần ML stack, nên không chạy được trong workspace này; không tính chúng vào tests đã đạt mới.

## Cách chạy notebook 03

1. Dùng môi trường Conda Python 3.10 đã cài dependencies theo [README môi trường](../environment/README.md). Restart kernel để import code mới.
2. Copy/check release và baseline artifacts ở đúng máy. Mặc định notebook tìm `runs/modeling/full_pdp_baseline/latest.json`; có thể đặt `AUTOCHECKOUT_BASELINE_SUMMARY` trỏ summary PASS. Chỉ marker/docs không đủ để qua gate baseline evidence.
3. Đặt `AUTOCHECKOUT_DATA_ROOT`/`AUTOCHECKOUT_RELEASE_ROOT` nếu paths khác. Chọn `offline=True` nếu weights của hai snapshot đã cache; default khi thực chạy có thể tải weights public, không gọi OpenAI.
4. Run cells theo thứ tự, batch 1 FP32 ở 640/800. Nếu OOM/FAIL, giữ run và logs; chưa có PASS tổng thể. Save notebook có outputs vào snapshot theo run ID.
5. Chỉ khi đủ gates mới ghi `PASS_CONVNEXT_ADAPTER_CUDA_SMOKE`. `full_training_ready=False` vẫn giữ để tách các điều kiện training chưa hoàn tất.

## Các điều kiện còn thiếu để train trên Vast.ai

| Gate | Vì sao cần | Bằng chứng phải có |
|---|---|---|
| S2 runtime ConvNeXt | Code/contract tests chưa thay thế model thật | Notebook 03 PASS ở 640/800, weights/load reports, cold reload và Lightning resume |
| S1 CLI/runner | `pdp/main.py`/`engine.py` vẫn khởi tạo processor theo pretrained/default; `configs/exp/common.sh` vẫn dùng `checkout_800`, SenseTime ResNet và require_kernel=1 | Cờ/config explicit backbone/pretrained/size/root, train/val/predict thống nhất; chạy CLI/runner smoke và resume đúng release native |
| Instance/toolchain | Run 3060 chỉ chứng minh môi trường đã chạy; native kernel trước đó lỗi compiler | RTX 4090 Vast.ai driver/PyTorch/CUDA/nvcc/compiler đúng, kernel forward/backward equivalence; hoặc chọn fallback có đo throughput/VRAM và ghi policy rõ |
| Budget/VRAM | Task 1 batch 1 chưa đo ConvNeXt/teacher full ở task mới | Batch/effective batch, optimizer-step count, VRAM/latency; chuyển full Task 1→2 với teacher/PPG/prototype state |
| Evaluation S6 | Calibration/counting pipeline chính còn thiếu policy artifact và tắt oracle | Fixtures đúng, threshold/NMS chọn trên val, policy/provenance khóa rồi áp cho test |
| Pilot E1 | Smoke không chứng minh học hội tụ | PDP ResNet vs ConvNeXt cùng split/resolution/steps/evaluator; loss/val curves, checkpoints và resume, rồi quyết định budget đủ 5 task |

**Hiện đủ chuẩn bị/chạy notebook nghiệm thu S2 trong environment phù hợp; chưa đủ nghiệm thu full training.** Không cần chờ 20 pilot synth để bắt đầu real-only baseline. LoRA/K=3/FSA/freeze bổ sung và generator overlap là track sau baseline.

## Kiểm chứng của lượt chuẩn bị này

`python3 -m pytest tests/test_backbone_adapter.py tests/test_model_acceptance.py -q -o addopts='' --tb=short`: **23 tests đạt**. Đã có tín hiệu FAIL trước sửa cho ConvNeXt index ngoài phạm vi, policy warm-start/status và bảo vệ pretrained backbone khỏi detector initialization. Tests chỉ dùng feature-provider doubles/shape policy; ML/CUDA gates nằm trong notebook 03 và chưa thực chạy.

Detector `post_init` có thể khởi tạo lại Conv/Linear của backbone nếu chưa đánh dấu loaded. Factory bảo vệ các module đã được timm nạp theo [Transformers 4.37.2 initialization contract](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/modeling_utils.py). Notebook ghi hash weights ngay khi timm trả model và đối chiếu sau detector construction, rồi sau warm-start để chặn trường hợp chỉ cấu hình pretrained=True nhưng tensor thực tế đã bị reset.
