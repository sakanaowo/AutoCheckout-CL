# AutoCheckout-CL

Nghiên cứu nhận diện và đếm sản phẩm tại quầy thanh toán bằng học liên tục theo lớp. Đợt đầu kiểm chứng PDP theo paper, sau đó thay backbone bằng **ConvNeXt-V2-Base** trên cùng dữ liệu để đo tác động.

- **Protocol chính:** 5 task **100+4×25 SKU**, seed 0; nguồn nhập 4 task chỉ dùng tham khảo.
- **Dữ liệu:** release ảnh thật đã nghiệm thu: **22.494 train / 1.503 val / 6.003 test**. Split nghiên cứu độc lập; một phần test2019 nguồn được dùng train, nên không gọi là official RPC test2019 benchmark.
- **Synthetic:** bản riêng 202 ảnh đã nghiệm thu: 9 giữ / 193 tái ghép thật với owner masks, provenance và mapping 5 task. CSV còn 20 pilot pending; chưa thêm vào release ảnh thật hoặc phát hành lại toàn bộ nguồn synth.
- **GPU train:** RTX 4090 24 GB trên **Vast.ai**. Processor 640/800 và model/GPU chưa được nghiệm thu; LoRA, K=3, FSA và freeze bổ sung là ablation sau baseline.
- **Quy trình:** mỗi bước dữ liệu/model/train/eval có notebook ghi mục tiêu, config, output và thời điểm chạy thật. Kết quả tháng 09 giữ làm lịch sử.

## Điểm vào

| Tài liệu | Nội dung |
|---|---|
| [docs/README.md](docs/README.md) | Mục lục tài liệu hiện hành, tham khảo và lịch sử |
| [Bàn giao](docs/project/AGENT_HANDOFF.md) | Quyết định, trạng thái code và cách tiếp quản |
| [Kế hoạch](docs/project/IMPLEMENTATION_PLAN.md), [tiến độ](docs/project/PROGRESS.md) | Thứ tự triển khai và bằng chứng kiểm chứng |
| [notebooks/README.md](notebooks/README.md) | Quy trình notebook và thứ tự thực hiện |
| [Split 5 task](notebooks/data_preprocessing/02_split_and_acceptance.ipynb), [báo cáo](docs/data_preprocessing/reports/split-100-4x25-2026-10-07.md) | Đã chạy; 9 kiểm tra nghiệm thu đạt, config và test lock cố định |
| [Review](notebooks/data_preprocessing/03_annotation_review.ipynb), [resolution 202](notebooks/data_preprocessing/04_annotation_resolution.ipynb) | Visual/AI/Batch → áp dụng và nghiệm thu ảnh thay thế |
| [Chỉ mục triển khai](docs/project/IMPLEMENTATION_INDEX.md), [vận hành](docs/data_preprocessing/OPERATIONS.md) | Code/config/tests mới, cách chạy lại và tiếp tục |
| [Schema dữ liệu](docs/data_preprocessing/formats.md) | Native geometry mới và quy ước legacy để đối chiếu |

## Cấu trúc

```text
docs/
  project/             bàn giao, kế hoạch và tiến độ hiện hành
  data_preprocessing/  schema, hướng dẫn; reports/ lưu báo cáo có ngày
  references/          handoff nguồn, giải thích paper và PDF
  archive/             kế hoạch, tiến độ và hướng GCP tháng 09
  timelines/           nhật ký theo thời điểm thực nghiệm
notebooks/
  data_preprocessing/  audit → nghiệm thu → split → review → resolution
  modeling/            kiểm chứng PDP nền và backbone (chưa chạy)
  training/            pilot/full protocol trên Vast.ai (chưa chạy)
  evaluation/          đánh giá detector và đếm (chưa chạy)
  references/          notebook thành viên nhóm
configs/               task mapping, data release lock và cấu hình thí nghiệm
pdp/                   lõi PDP/Deformable DETR, teacher và PPG
autocheckout/          mapping, metrics và provenance
tools/                 CLI dữ liệu, split, task JSON và evaluation
scripts/               runner và công cụ vận hành kế thừa
tests/                 fixtures và kiểm thử
baselines/             baseline truy xuất
results/               báo cáo/kết quả lịch sử
data/archive/          đầu vào giữ nguyên (không đưa vào git)
data/processed/        release và queue sinh từ notebook (không đưa vào git)
runs/                  log, QA và bản notebook theo run ID (không đưa vào git)
```

Các file kế hoạch/tiến độ/hướng dẫn GCP ở root là đường dẫn vào tài liệu đã tổ chức lại. Tái sử dụng tools/tests/configs đã khôi phục; kiểm chứng defaults cũ trước khi train trên Vast.ai.

## Kiểm chứng và bước tiếp theo

CPU tests dữ liệu/cấu hình/visual/API/resolution có kiểm chứng riêng; xem [tiến độ](docs/project/PROGRESS.md) và [lệnh kiểm tra](docs/data_preprocessing/OPERATIONS.md). Chưa chạy toàn bộ suite/model/GPU. Release giữ nguyên pixel gốc; bước tiếp theo là notebook kiểm chứng processor, loader, label masking và PDP nền, đồng thời quyết định scope 20 pilot còn lại và tích hợp bản synth đã nghiệm thu vào config augmentation riêng.

`pyproject.toml` hiện yêu cầu Python 3.10; dependencies model trong repo là cấu hình kế thừa. Chuẩn bị môi trường riêng trên Vast.ai và kiểm tra PyTorch/CUDA/deformable-attention kernel trước pilot. Không bật `--shutdown` của runner như cơ chế kết thúc thuê instance.

## Nguồn và giấy phép

- Thư mục `pdp/` lấy từ [zyt95579/PDP_IOD](https://github.com/zyt95579/PDP_IOD) (commit `7702d91`). Code này dựa trên [MD-DETR](https://github.com/GauravBh1010tt/MD-DETR) và Deformable DETR của Hugging Face Transformers (Apache 2.0). Repo PDP_IOD gốc **không có file LICENSE**; mọi quyền với code gốc thuộc về tác giả của nó. Repo này chỉ dùng cho mục đích học tập, phi thương mại.
- Dataset RPC có giấy phép CC BY-NC-SA 4.0 và **không** được đưa lên repo.
