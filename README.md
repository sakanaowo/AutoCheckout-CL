# AutoCheckout-CL

Đồ án: hệ thống thanh toán tự động bằng camera. Hệ thống nhận diện từng sản phẩm trên quầy, không cần mã vạch, rồi đếm số lượng và tính tiền. Khi cửa hàng nhập thêm sản phẩm mới, mô hình học thêm các SKU đó mà không train lại từ đầu, không lưu ảnh của các đợt trước, và vẫn nhận ra các SKU cũ (class-incremental object detection).

- **Dữ liệu:** RPC (Retail Product Checkout): 200 SKU, 30.000 ảnh quầy có bbox.
- **Phương pháp:** PDP (*Beyond Prompt Degradation: Prototype-guided Dual-pool Prompting for Incremental Object Detection*, CVPR 2026), đã sửa lỗi và cải tiến cho RPC.

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/overview.md](docs/overview.md) | Tổng quan dự án, mục tiêu bài toán và giải thích paper PDP cho người có nền tảng DL |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Kế hoạch: quyết định, dữ liệu, các bản sửa F1–F12, thí nghiệm, mốc kiểm soát |
| [PROGRESS.md](PROGRESS.md) | Tiến độ từng task, nhật ký, handoff |
| [GCP_TRAINING_GUIDE.md](GCP_TRAINING_GUIDE.md) | Vận hành máy ảo GCP: cài đặt, dữ liệu, chạy thí nghiệm, lấy kết quả |
| [docs/formats.md](docs/formats.md) | Định dạng file và thư mục dùng chung giữa các phần code |

## Cấu trúc

```
pdp/           code PDP_IOD (bản gốc ở commit đầu, mỗi bản sửa là một commit riêng; nguồn: pdp/UPSTREAM.md)
autocheckout/  thư viện dùng chung: cấu hình task, file dự đoán, chỉ số mAP và đếm, lưu vết lần chạy
tools/         công cụ dữ liệu (DL1–DL6) và đánh giá (V2, V3, V6)
baselines/     baseline truy xuất DINOv2 (E5)
configs/       cấu hình task, chia tập, và từng thí nghiệm (configs/exp/*.sh)
scripts/       cài VM, tải dữ liệu, chuẩn bị dữ liệu, chạy thí nghiệm
tests/         unit test và smoke test (CPU)
```

## Chạy test (CPU)

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2
pip install -r requirements.txt -r requirements-dev.txt && pip install -e .
pytest            # khoảng 2 phút; 3 test kernel CUDA chỉ chạy trên GPU
```

Train và đánh giá trên GPU: xem [GCP_TRAINING_GUIDE.md](GCP_TRAINING_GUIDE.md).

## Nguồn và giấy phép

- Thư mục `pdp/` lấy từ [zyt95579/PDP_IOD](https://github.com/zyt95579/PDP_IOD) (commit `7702d91`). Code này dựa trên [MD-DETR](https://github.com/GauravBh1010tt/MD-DETR) và Deformable DETR của Hugging Face Transformers (Apache 2.0). Repo PDP_IOD gốc **không có file LICENSE**; mọi quyền với code gốc thuộc về tác giả của nó. Repo này chỉ dùng cho mục đích học tập, phi thương mại.
- Dataset RPC có giấy phép CC BY-NC-SA 4.0 và **không** được đưa lên repo.
