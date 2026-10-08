# Dependencies và chuẩn bị môi trường

Tài liệu này liệt kê các gói cần cài cho AutoCheckout-CL. Người dùng tự cài trên máy/instance muốn sử dụng và tự cấu hình Vast.ai. Việc bổ sung tài liệu chưa nghiệm thu môi trường model hoặc GPU.

## Python và bộ phiên bản

Dùng **Python 3.10**, theo `requires-python` trong [pyproject.toml](../../pyproject.toml). Dùng virtual environment riêng cho repo. Bộ phiên bản bên dưới phục vụ code PDP hiện có; kiểm chứng tương thích trước khi nâng PyTorch/Transformers/timm hoặc tích hợp backbone mới.

| Nhóm | Dependency / phiên bản | Khai báo |
|---|---|---|
| Tensor và vision | `torch==2.2.2`, `torchvision==0.17.2` | Cài riêng theo CPU/CUDA |
| Tính toán | `numpy==1.24.4`, `scipy==1.10.1`, `scikit-learn==1.3.2` | [requirements.txt](../../requirements.txt) |
| Ảnh, biểu đồ, tiến trình | `pillow==10.2.0`, `matplotlib==3.7.4`, `tqdm==4.66.1` | requirements.txt |
| COCO annotations/evaluation | `pycocotools==2.0.7` | requirements.txt |
| Hugging Face | `transformers==4.37.2`, `tokenizers==0.15.1`, `huggingface-hub==0.20.3`, `safetensors==0.4.2` | requirements.txt |
| Backbone | `timm==0.9.12` | requirements.txt |
| Training/metrics | `lightning==2.1.3`, `pytorch-lightning==2.1.3`, `torchmetrics==1.3.0.post0` | requirements.txt |
| Build CUDA extension | `ninja==1.11.1.1` | requirements.txt; toolchain hệ thống bên dưới |
| Tests/lint | `pytest==8.3.3`, `ruff==0.6.9` | [requirements-dev.txt](../../requirements-dev.txt) |
| Notebook | `ipykernel==7.3.0`, `nbclient==0.10.0`, `nbformat==5.10.4` | [requirements-notebooks.txt](../../requirements-notebooks.txt) |

`ipykernel` dùng để chọn kernel trong IDE; `nbclient`/`nbformat` dùng khi thực thi và lưu notebook bằng code. JupyterLab/server là tùy chọn nếu làm việc qua VS Code. Các helper OpenAI hiện dùng standard library, không bắt buộc cài OpenAI SDK. API key ở environment hoặc `.env` local; xem [OPERATIONS](../data_preprocessing/OPERATIONS.md).

## Lệnh cài để dùng sau

Chạy từ repo root trên Linux. Tạo `.venv` nếu chưa có; với môi trường đã có, kiểm tra interpreter là Python 3.10 trước khi cài.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Chọn **một** bản PyTorch phù hợp môi trường. Bộ `torch 2.2.2` / `torchvision 0.17.2` và indexes dưới đây theo [hướng dẫn PyTorch chính thức](https://pytorch.org/get-started/previous-versions/#v222).

CPU cho kiểm thử/tiền xử lý local:

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1 cho instance GPU tương thích do bạn tự chuẩn bị:

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

Sau đó cài dependencies chung, công cụ phát triển/notebook và package repo:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt \
  -r requirements-notebooks.txt
python -m pip install -e .
```

Nếu `.venv` được tạo bởi `uv` và chưa có pip, có thể thay `python -m pip install` bằng `uv pip install --python .venv/bin/python`; dùng cùng tên gói, requirement files và PyTorch index. Không dùng Python hệ thống 3.14 để cài bộ dependencies model này.

Trong VS Code, chọn interpreter/kernel **`.venv/bin/python`**. Có thể đăng ký kernel ở prefix của môi trường nếu cần:

```bash
python -m ipykernel install --sys-prefix --name autocheckout \
  --display-name "AutoCheckout (Python 3.10)"
```

## Dependencies hệ thống cho Vast.ai

Bạn tự chọn image, cấu hình driver và đường dẫn trên instance. Khi chạy GPU cần có:

- NVIDIA driver tương thích bản PyTorch CUDA đã chọn.
- CUDA toolkit có `nvcc` để build custom deformable-attention extension; wheel PyTorch không thay thế compiler này. Với stack `cu121` ở trên, chuẩn bị toolkit CUDA 12.1 tương ứng.
- C/C++ compiler, `make`/build tools và Python 3.10 headers nếu build extension từ source.
- RTX 4090 dùng compute capability **8.9** khi đặt `TORCH_CUDA_ARCH_LIST`; xác minh GPU thực tế trước khi build.

Xem [PyTorch CUDA extension](https://docs.pytorch.org/docs/2.2/cpp_extension.html) và [bảng GPU NVIDIA](https://developer.nvidia.com/cuda-gpus). `scripts/setup_vm.sh` là script GCP kế thừa, có sudo/path assumptions cũ; dùng tài liệu này để chuẩn bị stack của instance thay vì chạy script đó nguyên trạng.

## Kiểm tra sau khi bạn cài

Kiểm tra xung đột dependencies và các imports chính:

```bash
python -m pip check
python -c "import torch, torchvision, transformers, timm, lightning, pycocotools; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Nếu dùng uv thay pip, chạy `uv pip check --python .venv/bin/python`. Trên local CPU, CUDA không khả dụng là kết quả dự kiến. Trên instance GPU, cần xác nhận `torch.cuda.is_available()` và tên GPU, `nvcc`, custom kernel, forward/backward trong notebook nghiệm thu riêng.

Regression dữ liệu gồm suite từng thiếu `pycocotools`:

```bash
python -m pytest tests/test_native_checkout.py tests/test_resize.py \
  tests/test_audit_rpc.py tests/test_groups.py tests/test_make_split.py \
  tests/test_make_task_config.py tests/test_make_task_json.py tests/test_taskcfg.py \
  tests/test_annotation_review.py tests/test_ai_annotation_review.py \
  tests/test_ai_annotation_batch.py tests/test_annotation_resolution.py -q
```

Sau đó chạy các test PDP/metrics/runner phù hợp với môi trường đã chuẩn bị; ghi kết quả thực tế, không tính các test CUDA bị skip là đã nghiệm thu GPU. Cài xong dependency chưa chứng nhận processor 640/800, loader, ConvNeXt adapter hoặc convergence.

Bước triển khai tiếp theo: notebook kiểm chứng môi trường/processor/loader/PDP nền theo [kế hoạch](../project/IMPLEMENTATION_PLAN.md). Mỗi lần thực chạy ghi run ID, thời điểm bắt đầu/kết thúc, phiên bản thư viện và host/GPU; tài liệu này chỉ là hướng dẫn cài.
