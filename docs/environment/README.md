# Dependencies và chuẩn bị môi trường

Tài liệu này liệt kê các gói cần cài cho AutoCheckout-CL. [Cài Conda sạch ngày 10/10](../project/CONDA_CLEAN_INSTALL_ACCEPTANCE.md) đã đạt trên RTX 3060 local với Python 3.10; instance Vast.ai/4090 được chuẩn bị và ghi metadata khi thực chạy pilot.

## Python và bộ phiên bản

Dùng **Python 3.10**, theo `requires-python` trong [pyproject.toml](../../pyproject.toml). Dùng virtual environment riêng cho repo. Bộ phiên bản bên dưới phục vụ code PDP hiện có; kiểm chứng tương thích trước khi nâng PyTorch/Transformers/timm hoặc tích hợp backbone mới.

| Nhóm | Dependency / phiên bản | Khai báo |
|---|---|---|
| Tensor và vision | `torch==2.2.2`, `torchvision==0.17.2` | requirements.txt khóa phiên bản; cài trước theo CPU/CUDA index |
| Tính toán | `numpy==1.24.4`, `scipy==1.10.1`, `scikit-learn==1.3.2` | [requirements.txt](../../requirements.txt) |
| Ảnh, biểu đồ, tiến trình | `pillow==10.2.0`, `matplotlib==3.7.4`, `tqdm==4.66.1` | requirements.txt |
| COCO annotations/evaluation | `pycocotools==2.0.7` | requirements.txt |
| Hugging Face | `transformers==4.37.2`, `tokenizers==0.15.1`, `huggingface-hub==0.20.3`, `safetensors==0.4.2` | requirements.txt |
| Backbone | `timm==0.9.12` | requirements.txt |
| Training/metrics | `lightning==2.1.3`, `pytorch-lightning==2.1.3`, `torchmetrics==1.3.0.post0` | requirements.txt |
| Build CUDA extension | `ninja==1.13.0` | requirements.txt; wheel manylinux2014 có Tag headers hợp lệ; toolchain hệ thống bên dưới |
| Tests/lint | `pytest==8.3.3`, `ruff==0.6.9` | [requirements.txt](../../requirements.txt) |
| Packaging/build compatibility | `packaging==24.2` | requirements.txt; đáp ứng wheel ≥24.0 và Lightning ≥20.0,<25.0 |
| Setuptools runtime compatibility | `setuptools==81.0.0` | requirements.txt; cung cấp pkg_resources cho lightning-utilities 0.10.1 |
| Typing compatibility | `typing-extensions==4.13.2` | requirements.txt; đáp ứng Jupyter Client ≥4.13.0 và Lightning <6.0 |
| Notebook | `ipykernel==7.3.0`, `nbclient==0.10.0`, `nbformat==5.10.4` | [requirements.txt](../../requirements.txt) |

`ipykernel` dùng để chọn kernel trong IDE; `nbclient`/`nbformat` dùng khi thực thi và lưu notebook bằng code. JupyterLab/server là tùy chọn nếu làm việc qua VS Code. Các helper OpenAI hiện dùng standard library, không bắt buộc cài OpenAI SDK. API key ở environment hoặc `.env` local; xem [OPERATIONS](../data_preprocessing/OPERATIONS.md).

## Lệnh cài để dùng sau

Chạy từ repo root trên Linux. Có thể dùng **Conda hoặc venv**; không bắt buộc uv. Chọn một environment Python 3.10 rồi dùng `python -m pip` của chính environment đó.

Nếu dùng Conda, environment `pdp` đã tạo thì chỉ cần activate:

```bash
# Chỉ chạy lệnh create nếu environment pdp chưa tồn tại.
conda create --name pdp python=3.10
conda activate pdp
python -c "import sys; print(sys.executable, sys.version)"
python -m pip --version
```

`sys.executable` và đường dẫn pip phải cùng thuộc environment `pdp`. Không cần tạo thêm `.venv` bên trong Conda.

Để tạo môi trường mới từ cấu hình của project, chạy từ repo root:

```bash
conda env create -f configs/environment/pdp.yml
conda activate pdp
```

[pdp.yml](../../configs/environment/pdp.yml) chỉ khai báo Python 3.10/pip và biến môi trường;
runtime vẫn dùng một nguồn [requirements.txt](../../requirements.txt). Sau activation, cài Torch từ đúng
CPU/cu121 index bên dưới rồi requirements. Môi trường `pdp` đã có thì dùng activation và phần sửa Ninja,
không cần tạo lại hoặc chạy solver cập nhật toàn bộ Conda.

Nếu dùng venv, tạo `.venv` nếu chưa có; với môi trường đã có, kiểm tra interpreter là Python 3.10 trước khi cài:

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
python -m pip install -r requirements.txt
python -m pip install -e .
```

Nếu `.venv` được tạo bởi `uv` và chưa có pip, có thể thay `python -m pip install` bằng `uv pip install --python .venv/bin/python`; dùng cùng tên gói, requirement files và PyTorch index. Không dùng Python hệ thống 3.14 để cài bộ dependencies model này.

Trong VS Code, chọn interpreter/kernel của environment đã cài: `.../miniconda3/envs/pdp/bin/python` khi dùng Conda, hoặc **`.venv/bin/python`** khi dùng venv. Có thể đăng ký kernel ở prefix của môi trường nếu cần:

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

## Lỗi ResolutionImpossible với typing-extensions

Log ngày 09/10/2026 xác nhận `typing-extensions==4.9.0` xung đột với `jupyter-client>=8.9.0` do `ipykernel==7.3.0` yêu cầu: các bản Jupyter Client đó cần `typing-extensions>=4.13.0`. Requirements hiện dùng **4.13.2**. Metadata của [Jupyter Client 8.10.0](https://pypi.org/pypi/jupyter-client/8.10.0/json) và [typing-extensions 4.13.2](https://pypi.org/pypi/typing-extensions/4.13.2/json) hỗ trợ Python 3.10.

Torch/torchvision hiện được pin **2.2.2/0.17.2** trong requirements để các gói như coco-eval/timm không kéo Torch mới vào. Vẫn cài Torch trước từ đúng CPU/CUDA index rồi mới cài `-r requirements.txt`. Đây là vấn đề constraints, không phải lỗi tạo Conda environment. Nâng riêng typing-extensions trong environment vẫn không giải quyết được nếu requirements tiếp tục yêu cầu 4.9.0.

Sau khi cập nhật requirements trong checkout đang sử dụng:

```bash
conda activate pdp
# CPU; dùng index cu121 ở phần trên nếu đây là instance CUDA tương thích.
python -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
```

Các lệnh trên để người dùng chạy trên environment của mình; chưa xác nhận cài/import/model thành công chỉ từ sửa requirements. Xem [pip dependency resolution](https://pip.pypa.io/en/stable/topics/dependency-resolution/#dealing-with-dependency-conflicts).

## Xung đột wheel/packaging sau khi cài thành công

Log có `Successfully installed` nhưng còn báo `wheel 0.47.0 requires packaging>=24.0` vì requirements cũ đã hạ packaging xuống 23.2. [Wheel 0.47.0](https://pypi.org/pypi/wheel/0.47.0/json) cần packaging ≥24.0; [Lightning 2.1.3](https://pypi.org/pypi/lightning/2.1.3/json) yêu cầu ≥20.0,<25.0. Pin hiện hành **packaging==24.2** nằm trong cả hai giới hạn và hỗ trợ Python 3.10.

Trong Conda environment đã cài các gói, chỉ cần cập nhật gói này rồi kiểm tra:

```bash
conda activate pdp
python -m pip install packaging==24.2
python -m pip check
```

Đồng thời dùng requirements đã cập nhật để lần cài sau không hạ packaging lại về 23.2. Không dùng upgrade packaging không giới hạn vì phiên bản ≥25.0 không đáp ứng constraint của Lightning 2.1.3. Chỉ khi `pip check` không còn conflict mới tiếp tục import và nghiệm thu notebook; log cài thành công chưa chứng nhận model/GPU.

## Import Lightning lỗi: No module named pkg_resources

Lần cài Conda sạch ngày 10/10/2026 kéo `setuptools==84.0.0`. Cài requirements và `pip check` đều exit 0,
nhưng import Lightning thất bại vì `lightning-utilities==0.10.1` gọi `import pkg_resources`.
Module này đã bị loại bỏ từ [Setuptools 82.0.0](https://setuptools.pypa.io/en/latest/history.html#v82-0-0).
Conda pdp cũ có setuptools 81.0.0 nên không bộc lộ lỗi khi kiểm tra môi trường đã cài sẵn.
Sau khi khóa setuptools 81, env mới đạt 105 tests, hai full-model CUDA cases và notebook pilot dry run;
xem [báo cáo cài sạch](../project/CONDA_CLEAN_INSTALL_ACCEPTANCE.md).

Requirements hiện khóa **setuptools==81.0.0** để giữ API mà stack Lightning đang dùng.
Trong đúng environment Python 3.10:

```bash
python -m pip install -r requirements.txt
python -m pip check
USE_TF=0 python -c "import lightning, pytorch_lightning, torchmetrics; print('imports PASS')"
```

`pip check` kiểm tra metadata dependency; cần kiểm tra imports và đường chạy model để phát hiện API đã bị gỡ.
Không nâng setuptools lên ≥82 khi vẫn dùng lightning-utilities 0.10.1.

## Cấu hình project và lỗi Ninja ở notebook 02

Giữ stack model phù hợp mã PDP vendored: Python 3.10, torch/torchvision 2.2.2/0.17.2,
Transformers 4.37.2, timm 0.9.12, Lightning 2.1.3 và numpy 1.24.4. Đây là cấu hình project
đã có bằng chứng limited foundation; không tuyên bố toàn bộ pins là môi trường benchmark paper nguyên bản.
Không nâng cả stack chỉ để xử lý công cụ build. Packaging giữ **24.2** để đáp ứng constraint Lightning <25.

Ngày 09/10, đã tái hiện `pip check` exit 1 trong Conda `pdp` (Linux x86_64, Python 3.10.22,
pip 26.2.1): `ninja 1.11.1.1 is not supported on this platform`.
File WHEEL của bản đã cài có dòng trống trước các Tag headers; parser của pip đọc được **0 Tag**.
Binary `ninja --version` vẫn chạy; chuỗi tags nhiều linux của wheel thực tế có giao với supported tags.
Lỗi quan sát được là metadata wheel, không phải bằng chứng RTX 3060/CUDA không tương thích.

Project đổi riêng **ninja==1.13.0**. [PyPI Ninja 1.13.0](https://pypi.org/project/ninja/1.13.0/)
có Python ≥3.8 và Linux x86_64 manylinux2014/glibc ≥2.17. Đã tải wheel để kiểm tra metadata/checksum
trong `/tmp`, không cài: WHEEL có Tag headers hợp lệ và khớp platform Conda hiện tại.
Việc này chưa chứng nhận kernel CUDA build/forward/backward thành công.

Trong environment hiện có, người dùng chạy:

```bash
conda activate pdp
python -m pip install --no-cache-dir --force-reinstall --only-binary=:all: ninja==1.13.0
python -m pip check
ninja --version
```

Không sửa tay WHEEL trong site-packages, không bỏ assertion `pip check`, không nâng packaging ≥25.
Sau khi pip check exit 0, restart kernel Conda `pdp` và chạy notebook 02 từ setup với run ID mới;
giữ run cũ FAIL làm lịch sử. Notebook tự tải checkpoint public khi `CONFIG['pretrained_dir']` rỗng; đường dẫn local là override tùy chọn.
Có thể chuẩn bị trước bằng `python -m tools.download_pdp_pretrained` từ repo root, không cần chép đường dẫn.
Nếu thiếu toolkit/compiler, mặc định `require_custom_kernel=False` cho phép CUDA PyTorch fallback;
native-kernel acceptance vẫn False. Với True, cần toolchain CUDA tương thích và gate kernel đạt.

Full baseline có thể kiểm tra trên RTX 3060 batch 1/FP32 nếu đủ VRAM; không bắt buộc 4090 cho smoke.
OOM ở 800 giữ kết quả 640 và tổng hợp PARTIAL. GPU RTX 4090 Vast.ai vẫn là môi trường pilot/training chính.
Xem [notebook 02](../../notebooks/modeling/02_full_pdp_baseline_acceptance.ipynb) và
[hướng dẫn modeling](../../notebooks/modeling/README.md).

## Kiểm tra sau khi bạn cài

Kiểm tra xung đột dependencies và các imports chính:

```bash
python -m pip check
python -c "import torch, torchvision, transformers, timm, lightning, pycocotools; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Nếu dùng uv thay pip, chạy `uv pip check --python .venv/bin/python`. Trên local CPU, CUDA không khả dụng là kết quả dự kiến. Trên instance GPU, xác nhận CUDA/tên GPU và ghi metadata runtime trong pilot/training; người dùng đã bỏ notebook nghiệm thu môi trường riêng. Native-kernel acceptance vẫn cần bằng chứng kernel thực chạy.

Regression dữ liệu gồm suite từng thiếu `pycocotools`:

```bash
python -m pytest tests/test_native_checkout.py tests/test_resize.py \
  tests/test_audit_rpc.py tests/test_groups.py tests/test_make_split.py \
  tests/test_make_task_config.py tests/test_make_task_json.py tests/test_taskcfg.py \
  tests/test_annotation_review.py tests/test_ai_annotation_review.py \
  tests/test_ai_annotation_batch.py tests/test_annotation_resolution.py -q
```

Sau đó chạy các test PDP/metrics/runner phù hợp với môi trường đã chuẩn bị; ghi kết quả thực tế, không tính các test CUDA bị skip là đã nghiệm thu GPU. Cài xong dependency chưa chứng nhận processor 640/800, loader, ConvNeXt adapter hoặc convergence.

Local Conda pdp/processor/loader và full pretrained CUDA smoke đã đạt trên RTX 3060 bằng PyTorch fallback; [kết luận nghiệm thu](../project/PDP_FOUNDATION_ACCEPTANCE.md). Adapter ConvNeXt đã có nghiệm thu notebook 03; [runtime notebook 04](../project/TRAINING_RUNTIME_ACCEPTANCE.md) ghi kiểm chứng CLI/runner. S6 calibration đã nghiệm thu kỹ thuật; bước tiếp là pilot/chuyển task/baseline trên 4090 theo [runbook training](../project/TRAINING_NOTEBOOK_RUNBOOK.md). Mỗi lần thực chạy ghi run ID, thời điểm bắt đầu/kết thúc, phiên bản thư viện và host/GPU; xem [cài Conda sạch](../project/CONDA_CLEAN_INSTALL_ACCEPTANCE.md) để tái dựng môi trường đã kiểm chứng local.
