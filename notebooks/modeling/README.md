# Kiểm chứng model

**Hoàn tất phạm vi notebook processor/loader/PDP nền và full CUDA smoke.** [Kết luận nghiệm thu](../../docs/project/PDP_FOUNDATION_ACCEPTANCE.md) ghi bằng chứng, phần S1 runtime còn lại và thứ tự công việc tiếp theo.

[01_processor_loader_pdp_acceptance.ipynb](01_processor_loader_pdp_acceptance.ipynb) đã đạt **`PASS_LIMITED_PDP_FOUNDATION`** trong run `65687942-4332-40db-a82e-76ae0429c100`, ngày 09/10/2026, 15:12:32–15:13:43 Asia/Bangkok. Máy có RTX 3060; model smoke thực chạy CPU. Artifacts ghi đủ 11 gate PASS, regression 27/27 và resume 1/1; xem [báo cáo](../../docs/timelines/model-foundation-2026-10-09.md). Chọn kernel Python 3.10 sau khi tự cài [dependencies](../../docs/environment/README.md), sửa `AUTOCHECKOUT_DATA_ROOT`/`AUTOCHECKOUT_RELEASE_ROOT` nếu cần, rồi chạy từ setup đến cell kết thúc.

Notebook có các gate:

1. Environment, release/checksums và protocol 100+4×25, 224 slots/225 outputs.
2. Nhãn current task/full GT tách riêng trên toàn bộ 5 task JSON.
3. Processor 640/800 trên 2 ảnh thật mỗi task: normalization, bbox round-trip/postprocess, padding/mask và source preservation.
4. Loader/collation/shuffle dùng code `pdp/main.py`, seed tái lập, workers=0.
5. PDP nhỏ trên batch ảnh thật Task 1 ở 96 px: forward/backward, detection/query/DDL losses, gradient, optimizer groups/step và frozen params.
6. Tests PDP nền/teacher/PPG/prototypes/task transition và test Lightning interruption/resume có sẵn; lưu logs/JUnit theo run ID.

Smoke dùng random ResNet18, encoder 1/decoder 2, prompt 4×2, 300 queries/256 dimensions; mapping thật của release. Mọi gate đạt chỉ cho `PASS_LIMITED_PDP_FOUNDATION`. Tắt một phần cho PARTIAL; lỗi cho FAIL. Không dùng kết quả này để nghiệm thu full pretrained ResNet50/PDP, full model 640/800, GPU/kernel, ConvNeXt hay convergence.

Output ở `runs/modeling/processor_loader_pdp/<run_id>/`: config, run/summary/gates JSON, logs và JUnit XML. Run timestamps được sinh khi thực chạy. Lưu notebook có outputs sau khi chạy, có thể lưu thêm bản theo run ID vào `runs/notebook_versions/`. Notebook không cài dependencies, không tải pretrained/API, không tạo lại split hay dùng full GT/test holdout để train.

Input: release real-only đã nghiệm thu trong [tiền xử lý](../data_preprocessing/README.md). Synthetic 202 chưa tích hợp trong notebook này. Full pretrained CUDA smoke đã đạt ở notebook 02. Bước sau: nối/kiểm chứng processor trong đường CLI/runner training, rồi adapter ConvNeXt-V2-Base, native kernel/evaluator và pilot Vast.ai; xem [kế hoạch](../../docs/project/IMPLEMENTATION_PLAN.md).


## Full baseline CUDA — notebook 02

[02_full_pdp_baseline_acceptance.ipynb](02_full_pdp_baseline_acceptance.ipynb) đã đạt **PASS_FULL_PDP_CUDA_SMOKE** trong run `d967dc7b-d665-4a8b-9ca0-0e27895e4bdc` trên RTX 3060: 640/800, batch 1 FP32, peak allocated 3,49/4,22 GiB. Dùng PyTorch fallback, native kernel chưa đạt. Bản notebook đã chạy nằm ở `runs/modeling/full_pdp_baseline/d967dc7b-d665-4a8b-9ca0-0e27895e4bdc/notebook_executed.ipynb`; outputs lỗi cũ trong file gốc được giữ. [Báo cáo](../../docs/timelines/model-foundation-2026-10-09.md#full-baseline-cuda-đạt-trên-rtx-3060).
Dùng full pretrained ResNet50, encoder/decoder 6/6, prompt 100×10, 300 queries và 225 outputs.
Chọn Conda `pdp` Python 3.10; để `CONFIG['pretrained_dir']` rỗng, gate pretrained tự gọi
[download_pdp_pretrained.py](../../tools/download_pdp_pretrained.py), tải detector checkpoint public
SenseTime/deformable-detr và tự nhận đường dẫn cache. Không tải dataset COCO, không cài dependencies.
Script lấy đúng config.json/preprocessor_config.json/model.safetensors (~161 MB), mặc định khóa revision
`83ecd26945199939cb82806f988debdb71e6f43e`, ghi commit/hash vào run artifacts và tái sử dụng cache
`runs/pretrained/hf-cache/`. `pretrained_offline=True` dùng cache không gọi mạng; local path vẫn là override tùy chọn.

Có thể tải trước từ repo root, không cần copy đường dẫn:

```bash
python -m tools.download_pdp_pretrained
```

Đã kiểm chứng tải thật 3 files và dùng cache offline; [4 tests downloader](../../tests/test_download_pdp_pretrained.py)
kiểm tra khóa revision/files/hash, offline cache và reject snapshot thiếu/sai revision.

RTX 3060 12 GB trong run d967dc7b đã đủ cho smoke batch 1 FP32 ở cả 640/800; peak allocated 3,49/4,22 GiB. Không bắt buộc 4090 để lặp lại smoke cấu hình này.
Không đảm bảo đủ VRAM chỉ từ tên GPU. OOM ở resolution nào thì ghi PARTIAL, giữ gate đã đạt;
chuyển gate còn thiếu sang GPU đủ bộ nhớ. RTX 4090 Vast.ai vẫn là môi trường pilot/training theo kế hoạch.

Worker [full_baseline_acceptance.py](../../pdp/full_baseline_acceptance.py) dùng losses/optimizer/loader/inference/resume
hiện có, chạy một subprocess mỗi resolution. Có scoped adapters để cung cấp full checkpoint config và loading report
cho engine, cùng processor explicit; không thay đổi defaults của CLI/runner. Gates gồm pretrained core loading,
CUDA forward/backward, gradient, optimizer/frozen hashes, predictions và controlled interruption/restore qua Lightning.
Nếu có native kernel, chạy thêm test forward/backward equivalence hiện có; fallback có trường riêng, không báo native PASS.
Output: `runs/modeling/full_pdp_baseline/<run_id>/`, có logs, peak VRAM, configs/checksums, last.ckpt và task_final.pth.

Cả 640/800 và gates provenance/source đạt → `PASS_FULL_PDP_CUDA_SMOKE`; không chứng nhận convergence,
CL teacher/PPG đủ 5 task, ConvNeXt hay Vast.ai. Notebook source trong run là snapshot trước thực thi;
lưu notebook có outputs sau thực chạy. Hai checkpoint format được giữ, không coi task_final.pth là resume đầy đủ.

Kiểm chứng công cụ bằng fixture CPU (không phải full model CUDA):

```bash
python -m pytest tests/test_model_acceptance.py tests/test_full_baseline_acceptance.py -q
```

Các tests kiểm tra không gán PASS cho OOM/missing/interruption, reject pretrained core thiếu/sai,
và khôi phục thật model/optimizer/scheduler/nonempty prototype state qua Lightning trên tiny fixture.


## ConvNeXt-V2-Base — notebook 03

[03_convnext_v2_base_acceptance.ipynb](03_convnext_v2_base_acceptance.ipynb) đã tạo, **chưa chạy**. Sau foundation/full ResNet smoke, notebook kiểm tra adapter metadata, pretrained ConvNeXt riêng, detector core warm-start có audit, feature maps/masks/projections 640/800, losses/gradient/optimizer, Lightning resume và cold model/config reload. Dùng snapshot revision đã khóa; khi thực chạy có thể tải/reuse weights public, không cài dependencies.

Restart Conda kernel để nạp code mới. Cần baseline summary/gates thật từ máy đã chạy; dùng `AUTOCHECKOUT_BASELINE_SUMMARY` nếu không có marker latest ở máy hiện tại. Run ID/output nằm trong `runs/modeling/convnext_adapter/`; save notebook với outputs sau execution. PASS chỉ cho `PASS_CONVNEXT_ADAPTER_CUDA_SMOKE`, không chứng nhận train dài hay Vast.ai. [Điều kiện và training readiness](../../docs/project/CONVNEXT_READINESS.md).
