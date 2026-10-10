# Kiểm chứng cài Conda sạch — Python 3.10

Ngày 10/10/2026, UTC+7. **Đạt `PASS_CLEAN_CONDA_VALIDATION`** trên máy local RTX 3060.
Tạo Conda environment mới từ [pdp.yml](../../configs/environment/pdp.yml), cài Torch cu121,
[requirements.txt](../../requirements.txt) và package editable; không dùng packages từ Conda pdp cũ.
Bắt đầu cài **12:44:58**, hoàn tất kiểm chứng **12:55:38**.

Environment còn ở `/tmp/autocheckout-conda-py310-20261010`; đặt trong /tmp vì ổ chứa repo chỉ còn khoảng 2,6 GB
trước khi chạy. Đây là vị trí tạm, không bảo đảm tồn tại qua reboot. Logs, checkpoints và notebook outputs
giữ dưới `data/training/validation/conda_py310_clean_install_20261010/` (~909 MB).

## Dependency cần sửa sau lần cài đầu

Lần đầu cài requirements và `pip check` exit 0 nhưng import Lightning thất bại:
`lightning-utilities==0.10.1` cần `pkg_resources`, còn setuptools được cài tự động là **84.0.0**.
[Setuptools 82.0.0](https://setuptools.pypa.io/en/latest/history.html#v82-0-0) đã loại bỏ module này.
Môi trường pdp cũ có setuptools 81.0.0 nên không phát hiện lỗi qua kiểm tra packages đã cài sẵn.

Requirements bổ sung duy nhất **setuptools==81.0.0**. Cài lại requirements trong environment mới,
imports và `pip check` đều exit 0. Python/model pins còn lại giữ nguyên.
Lỗi này khác lỗi NumPy/source build trên Python 3.12: stack hiện hành vẫn dùng Python 3.10.

## Kết quả kiểm chứng

| Kiểm tra | Bằng chứng |
|---|---|
| Interpreter độc lập | Python **3.10.22**, sys.prefix thuộc env mới; sys.path không mượn site-packages pdp cũ |
| Requirements | **64 pins khớp**, không mismatch; pip check exit 0 |
| Imports | Torch/torchvision, NumPy/SciPy, Transformers/timm, Lightning/torchmetrics, COCO, OpenCV và notebook stack đạt |
| CUDA | Torch **2.2.2+cu121**, torchvision **0.17.2+cu121**, runtime **12.1**, RTX 3060; CUDA tensor forward/backward đạt |
| Regression | **105 tests PASS**, 0 failures/errors/skips; gồm CPU/CUDA optimizer/resume, teacher/PPG/prototypes, CLI/runner, calibration và logging |
| Full-model EXP-B1 | ResNet50 pretrained, 640 px, physical/effective batch 1/2, step **1 → 4**, peak training allocated **3.61 GiB**, PASS |
| Full-model EXP-B2 | ConvNeXt-V2-Base pretrained, cùng cấu hình/budget, step **1 → 4**, peak **3.79 GiB**, PASS |
| Notebook headless | Notebook training 01, CPU fixture EXP-B1, **3 code cells hoàn tất**; interruption step 1 và resume step 2 đạt |

Hai GPU cases gọi [acceptance worker](../../tools/pdp_runtime_acceptance.py) và runner thật.
Dùng lại subset đã kiểm chứng ở runtime notebook 04: 4 ảnh train Task 1, 2 ảnh prediction mỗi split,
2 epochs, FP32, TF32 tắt, shuffle=0. Detector/backbone lấy từ cache local đã khóa revision,
không tải weights mới. Worker xác nhận hash model/optimizer/scheduler/PDP memory sau restore,
frozen parameters giữ nguyên, predict-only cho cùng predictions, và runner skip task hoàn tất.

Notebook dry run dùng tiny random ResNet18/6 labels riêng. Kết quả GPU subset và CPU fixture kiểm chứng
đường chạy/dependencies; không dùng làm accuracy/convergence hay pilot đủ budget.
Chưa thực chạy trên RTX 4090/Vast.ai và chưa nghiệm thu native deformable-attention kernel.

## Receipts

Trong artifact folder:

- `installation.json`: timestamps, commands, exit codes, nguồn/requirements trước sửa và các bước cài ban đầu.
- `source/`, `requirements-after.txt`: input snapshot và requirements sau pin setuptools.
- `imports-before.log`: tín hiệu FAIL với setuptools 84; `imports_after.log`, `environment.json`: PASS với setuptools 81.
- `validation.json`: kết quả cuối, 105 tests, GPU cases và notebook execution ID.
- `regression.xml`, `regression.log`: kết quả regression cuối.
- `gpu/EXP-B1_640/`, `gpu/EXP-B2_640/`: result.json, runner logs, checkpoint state hashes, task_final.pth, val/test predictions.
- `notebooks/dry_run/clean-conda-pilot/`: fixture, checkpoint/logs và notebook có outputs;
  execution ID `add0fd8c-4c95-48b9-94a3-e452cb7b639c`.
- `pip_freeze_after.log`, `conda_export_after.log`: phiên bản packages và environment sau kiểm chứng.

Giữ lịch sử lỗi: lượt regression đầu gặp quota /tmp khi tích lũy checkpoints test
(`quota-fail-regression.log/xml`, 49 failures, 4 errors, 52 passed). Chỉ dọn thư mục fixtures tạm do lượt này tạo,
rồi chạy lại với `tmp_path_retention_policy=failed`; lượt cuối 105 PASS.
Script điều phối tạm có một lỗi shadow tên biến sau regression, giữ receipt
`validation-orchestration-fail.json`; đã sửa script tạm và tiếp tục các bước còn lại.
Các lỗi lịch sử không được đổi thành PASS.

## Cài cùng cấu hình trên máy thuê

Từ repo đã có requirements cập nhật, dùng environment riêng Python 3.10.
Chỉ chạy lệnh create khi environment `pdp310` chưa tồn tại:

```bash
cd /workspace/AutoCheckout-CL
conda env create -n pdp310 -f configs/environment/pdp.yml
conda activate pdp310
python -c "import sys; print(sys.executable, sys.version); assert sys.version_info[:2] == (3, 10)"
python -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python -c "import lightning, torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Mở tmux session rồi activate `pdp310` bên trong trước khi chạy headless notebook.
IDE chọn kernel/interpreter của env này. Tiếp tục theo [runbook training](TRAINING_NOTEBOOK_RUNBOOK.md):
real pilot EXP-B1/B2 trên 4090, chuyển Task 1→2 và baseline 5 task.
Theo quyết định người dùng, không bổ sung notebook nghiệm thu môi trường Vast.ai riêng.
