# Kiểm chứng model

[01_processor_loader_pdp_acceptance.ipynb](01_processor_loader_pdp_acceptance.ipynb) đã tạo, **chưa chạy**. Chọn kernel Python 3.10 sau khi tự cài [dependencies](../../docs/environment/README.md), sửa `AUTOCHECKOUT_DATA_ROOT`/`AUTOCHECKOUT_RELEASE_ROOT` nếu cần, rồi chạy từ setup đến cell kết thúc.

Notebook có các gate:

1. Environment, release/checksums và protocol 100+4×25, 224 slots/225 outputs.
2. Nhãn current task/full GT tách riêng trên toàn bộ 5 task JSON.
3. Processor 640/800 trên 2 ảnh thật mỗi task: normalization, bbox round-trip/postprocess, padding/mask và source preservation.
4. Loader/collation/shuffle dùng code `pdp/main.py`, seed tái lập, workers=0.
5. PDP nhỏ trên batch ảnh thật Task 1 ở 96 px: forward/backward, detection/query/DDL losses, gradient, optimizer groups/step và frozen params.
6. Tests PDP nền/teacher/PPG/prototypes/task transition và test Lightning interruption/resume có sẵn; lưu logs/JUnit theo run ID.

Smoke dùng random ResNet18, encoder 1/decoder 2, prompt 4×2, 300 queries/256 dimensions; mapping thật của release. Mọi gate đạt chỉ cho `PASS_LIMITED_PDP_FOUNDATION`. Tắt một phần cho PARTIAL; lỗi cho FAIL. Không dùng kết quả này để nghiệm thu full pretrained ResNet50/PDP, full model 640/800, GPU/kernel, ConvNeXt hay convergence.

Output ở `runs/modeling/processor_loader_pdp/<run_id>/`: config, run/summary/gates JSON, logs và JUnit XML. Run timestamps được sinh khi thực chạy. Lưu notebook có outputs sau khi chạy, có thể lưu thêm bản theo run ID vào `runs/notebook_versions/`. Notebook không cài dependencies, không tải pretrained/API, không tạo lại split hay dùng full GT/test holdout để train.

Input: release real-only đã nghiệm thu trong [tiền xử lý](../data_preprocessing/README.md). Synthetic 202 chưa tích hợp trong notebook này. Bước sau: kiểm chứng full baseline/pretrained/processor trong đường training, rồi adapter ConvNeXt-V2-Base và pilot Vast.ai; xem [kế hoạch](../../docs/project/IMPLEMENTATION_PLAN.md).
