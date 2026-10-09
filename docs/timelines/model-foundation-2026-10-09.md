# Model foundation và full baseline — 09/10/2026

## Bằng chứng notebook 01 do người dùng thực chạy

[Notebook 01](../../notebooks/modeling/01_processor_loader_pdp_acceptance.ipynb) có output và artifacts
`runs/modeling/processor_loader_pdp/65687942-4332-40db-a82e-76ae0429c100/`.
Đã đọc lại summary/gates/JUnit trong phiên tiếp quản; không chạy lại run này.

- Thời gian thực: **09/10/2026 15:12:32–15:13:43 Asia/Bangkok**; git `9d689a4`.
- Conda `pdp`, Python 3.10.22; environment ghi torch 2.2.2/torchvision 0.17.2, CUDA runtime 12.1,
  GPU NVIDIA GeForce RTX 3060. **Config device là CPU**, không phải CUDA full-model run.
- **`PASS_LIMITED_PDP_FOUNDATION`**, đủ 11 gates: environment, release lock, current-task/full-GT separation,
  processor 640/800, loader 640/800, PDP smoke, baseline regression, Lightning resume, source preservation.
- Baseline JUnit: **27 tests, 0 failures/errors/skips**; resume JUnit: **1 test, 0 failures/errors/skips**.
- Model smoke: tiny random ResNet18, encoder 1/decoder 2, 96 px. Không nghiệm thu full pretrained PDP,
  GPU/kernel, ConvNeXt hoặc convergence. Cross-task image reuse là quy ước của split đã khóa.
- Hai run trước giữ nguyên lịch sử RUNNING/INTERRUPTED; không sửa thành PASS.

## Deliverable tiếp theo, chưa thực chạy full CUDA

[Notebook 02](../../notebooks/modeling/02_full_pdp_baseline_acceptance.ipynb) kiểm chứng full pretrained ResNet50/PDP
trên ảnh train Task 1, CUDA FP32 batch 1 ở 640 và 800. Cần local detector snapshot có config/preprocessor/weights.
Notebook chỉ hướng dẫn người dùng tải riêng, không cài dependencies/tải model khi execution.

Mỗi resolution có worker process, loading report, losses/gradient, optimizer/frozen hash, two-pass predictions,
controlled Lightning interruption/resume, peak VRAM và artifacts/checkpoint/config/checksums.
Nếu native kernel khả dụng, gọi test CUDA forward/backward equivalence có sẵn; ghi fallback riêng.
RTX 3060 có thể dùng nếu đủ VRAM đo được; OOM giữ gate đã đạt và tổng hợp PARTIAL.
Không bắt buộc 4090 cho smoke. Pilot/training chính vẫn theo kế hoạch RTX 4090 Vast.ai.

Công cụ dùng lại:

- [model_acceptance.py](../../autocheckout/model_acceptance.py): persist gates/status và chặn PASS cho missing/OOM/interruption;
  validator reject core pretrained thiếu/sai, chỉ cho classifier thay số lớp và PDP parameters mới.
- [full_baseline_acceptance.py](../../pdp/full_baseline_acceptance.py): full worker, dùng engine/loader/inference/checkpoint hiện có.
  Scoped factory adapters cung cấp checkpoint config/loading report; không thu nhỏ architecture.
- [tests evidence](../../tests/test_model_acceptance.py) và [tests Lightning](../../tests/test_full_baseline_acceptance.py):
  kiểm tra status/loading report và restore thật trên tiny CPU fixture với prototype cache không rỗng;
  cố ý làm sai trainable weights/live memory trước restore, so sánh model/optimizer/scheduler/PDP state và predictions.

Notebook 02 lúc tạo có execution_count=None, outputs rỗng. Trong lúc hoàn thiện, người dùng đã bắt đầu run e1f12f37: FAIL ở gate environment do pip check/Ninja, chưa chạy full model CUDA. Outputs lỗi được giữ nguyên.
Công cụ có kiểm chứng CPU fixture, không suy ra full pretrained CUDA đã chạy.
Defaults processor trong CLI/runner vẫn cần nối resolution explicit và nghiệm thu trước training.
Sau full baseline smoke: adapter ConvNeXt-V2-Base → pilot cùng split/budget → đủ 5 task → ablation riêng.

## Kiểm chứng công cụ trong phiên tạo notebook

Chạy bằng Python Conda `pdp` hiện có, `CUDA_VISIBLE_DEVICES=''`, không cài packages:

```bash
python -m pytest tests/test_model_acceptance.py tests/test_full_baseline_acceptance.py -q
```

Kết quả **11 tests đạt, 0 failures/errors/skips**, gồm 9 tests status/loading report và 2 tests worker/Lightning.
Đây là tiny CPU fixture và gate logic; notebook 02/full pretrained/CUDA chưa thực thi.
AI DevKit lint/memory offline không chạy được vì npm cache thiếu ai-devkit; không coi đây là lint/memory PASS.

## Lỗi environment khi người dùng bắt đầu notebook 02

Run `e1f12f37-cb95-495a-9970-ddb9f6b7aeac` bắt đầu 15:36:53, FAIL tại environment lúc 15:37:04
ngày 09/10/2026 Asia/Bangkok. Pip check báo Ninja 1.11.1.1 unsupported platform.
Đã tái hiện exit 1, xác định WHEEL có dòng trống trước Tags làm parser đọc 0 Tag; binary Ninja vẫn chạy.
Wheel Ninja 1.13.0 từ PyPI đã kiểm tra checksum và metadata hợp lệ/khớp supported tags trong cùng Conda,
không cài package. Requirements đổi riêng Ninja 1.13.0, giữ model stack và packaging 24.2.
[README môi trường](../environment/README.md#cấu-hình-project-và-lỗi-ninja-ở-notebook-02) có lệnh sửa Conda hiện có;
[Conda config](../../configs/environment/pdp.yml) dành cho tạo mới. Chưa có pip check PASS sau người dùng cài sửa.

## Tự tải checkpoint public theo yêu cầu người dùng

Thay bước chuẩn bị/copy đường dẫn bằng [script downloader](../../tools/download_pdp_pretrained.py).
Notebook 02 gọi script khi pretrained_dir rỗng, tự nhận snapshot_dir vào CONFIG; vẫn dùng local-only trong worker model.
Lấy weights detector COCO, không tải dataset COCO và không cài dependencies.

Tải thật 3 files từ SenseTime/deformable-detr ngày 09/10/2026 **15:52:28–15:52:33 Asia/Bangkok**:
commit `83ecd26945199939cb82806f988debdb71e6f43e`, model.safetensors 160.769.532 bytes,
SHA256 `caf1e3e61283c6ce35cd2d9adaa7033cf40997d4dfe434003bcdb9085cc8cf9b`.
Receipt/checksums ở `runs/pretrained/deformable_detr.json`; cache `runs/pretrained/hf-cache/`, được ignore.
Script mặc định khóa commit này; có --revision và --offline.
Chạy lại offline đạt ngày 09/10, 15:54:32 với cùng hashes; không gọi mạng/tải lại payload.
4 tests downloader đạt với mock ở biên HF; full baseline CUDA chưa được suy ra từ việc tải weights.

## Kiểm tra kết quả run 7169ed25 và sửa validator

Run `7169ed25-a405-4eb8-aebc-08cefc041668`: **15:58:25–15:59:27 ngày 09/10/2026 Asia/Bangkok**, status FAIL.
Environment, foundation, release lock, pretrained files và sources preserved PASS.
Cả cuda_640/cuda_800 FAIL tại audited pretrained load, trước model forward/backward;
peak allocated/reserved đều 0, chưa có bằng chứng OOM hoặc 3060 đủ/thiếu VRAM.

Loading report: 38 params prompts/query_tf mới, 2 classifier shapes 91→225 (đúng chủ đích),
4 unexpected `num_batches_tracked` ở BatchNorm downsample của ResNet.
`DeformableDetrFrozenBatchNorm2d._load_from_state_dict` chủ động bỏ counter này;
checkpoint public chứa counters, không phải thiếu pretrained weights của backbone/encoder/decoder.
Validator trước đó từ chối mọi unexpected keys, gây false rejection.

Đã sửa: chỉ cho phép chính xác counters suy ra từ live FrozenBatchNorm modules; core tensors/counters
ngoài modules này vẫn bị reject. Giữ loading report raw, thêm ignored_frozen_batch_norm_counters.json.
Không sửa outputs/run FAIL cũ thành PASS. Tái hiện policy cũ từ chối cùng report; full checkpoint ResNet50/PDP
nạp lại trên CPU, validation mới PASS với đúng 4 counters, không missing backbone/encoder/decoder.
13 tests acceptance/Lightning và 4 tests downloader đạt (17 tổng), không failure/error/skip.
Full CUDA forward/backward/resume chưa chạy lại sau sửa; cần run ID mới.

Native custom kernel cũng chưa build được: log lỗi compiler tại torch ATen/core/List_inl.h
("need typename" và conversion), nên dùng PyTorch fallback với require_custom_kernel=False.
Sửa validator không chứng nhận native kernel; toolchain cần nghiệm thu riêng trước native-kernel training.

## Full baseline CUDA đạt trên RTX 3060

Run người dùng `21d281b1-c6ca-4d32-985e-e7145788e219` ngày 09/10, 16:07:00–16:08:03 Asia/Bangkok
đã qua pretrained load và GPU optimizer step, nhưng FAIL khi prediction sau interruption.
Lightning teardown chuyển model về CPU; helper vẫn đưa input lên CUDA. Đã tái hiện chính lỗi này
trong test CUDA tiny fixture trên RTX 3060 (FAIL trước sửa), thêm module.to(device) sau controlled interruption,
rồi test CPU+GPU resume đạt. Tổng 18 tests acceptance/downloader đạt, 0 failure/error/skip.

Đã thực chạy lại toàn bộ notebook bằng nbclient, kernel tạm dùng Conda pdp hiện có, không cài dependencies,
không ghi đè outputs file notebook của người dùng. Run mới **d967dc7b-d665-4a8b-9ca0-0e27895e4bdc**:

- Thời gian thật: **2026-10-09T16:12:09+07:00 → 2026-10-09T16:12:57+07:00** (Asia/Bangkok).
- **PASS_FULL_PDP_CUDA_SMOKE**, đủ 7 gates PASS; full pretrained ResNet50 encoder/decoder 6/6, prompt 100×10,
  protocol 100+4×25, 225 outputs, train Task 1 images [1,2], batch vật lý 1, FP32, TF32 tắt.
- Cả 640 và 800: losses/gradient finite, classifier cập nhật, frozen params bất biến, predictions reload tương đương,
  checkpoint khôi phục optimizer/scheduler/PDP state, global step 1→4; tổng 5 optimizer steps gồm manual step.
- GPU RTX 3060; peak allocated **3,4866 GiB (640)** / **4,2243 GiB (800)**. Đủ cho smoke cấu hình này;
  không suy ra batch lớn/ConvNeXt/full five-task teacher/training đủ VRAM.
- **PyTorch CUDA fallback**; custom_kernel_accepted=False, native kernel/toolchain chưa nghiệm thu.
- Source/weights/release/holdout preservation PASS. Không chứng nhận convergence, benchmark, ConvNeXt,
  calibration/counting, synthetic augmentation, Vast.ai hoặc đường CLI/runner mặc định.

Artifacts ở `runs/modeling/full_pdp_baseline/d967dc7b-d665-4a8b-9ca0-0e27895e4bdc/`:
summary/gates, 640/800 result/load report/config/last.ckpt/task_final.pth và worker logs,
`notebook_executed.ipynb` có outputs, `acceptance_regression.xml` 18 tests CPU+GPU.
Bản notebook thứ hai trong `runs/notebook_versions/02_full_pdp_baseline_acceptance_e1b18b10-dc34-4b87-8521-7a33b601a20b.ipynb`.
Các run FAIL trước giữ nguyên. Bước tiếp: đường train CLI/runner explicit resolution, adapter ConvNeXt,
pilot cùng split/budget trên RTX 4090 Vast.ai; native kernel là gate riêng.
