# AutoCheckout-CL

Nghiên cứu nhận diện và đếm sản phẩm tại quầy thanh toán bằng học liên tục theo lớp. Đợt đầu kiểm chứng PDP theo paper, sau đó thay backbone bằng **ConvNeXt-V2-Base** trên cùng dữ liệu để đo tác động.

- **Protocol chính:** 5 task **100+4×25 SKU**, seed 0; nguồn nhập 4 task chỉ dùng tham khảo.
- **Dữ liệu:** release ảnh thật đã nghiệm thu: **22.494 train / 1.503 val / 6.003 test**. Split nghiên cứu độc lập; một phần test2019 nguồn được dùng train, nên không gọi là official RPC test2019 benchmark.
- **Synthetic:** bản riêng 202 ảnh đã nghiệm thu: 9 giữ / 193 tái ghép thật với owner masks, provenance và mapping 5 task. CSV còn 20 pilot pending; chưa thêm vào release ảnh thật hoặc phát hành lại toàn bộ nguồn synth.
- **Nghiệm thu model:** foundation, ConvNeXt adapter và [CLI/runner](docs/project/TRAINING_RUNTIME_ACCEPTANCE.md) đã đạt. Pilot và transition smoke đã hoàn tất trên 4090; [môi trường L4 GCP](docs/environment/GCP_L4_SETUP_2026-10-12.md) đạt native kernel và resume thật. LoRA/K=3/FSA/freeze bổ sung là ablation sau baseline.
- **Baseline hoàn tất:** cả ResNet-50 và ConvNeXt-V2-Base đủ 5 task, campaign kết thúc 15/10; kiểm tra lại artifacts ngày 20/10. Final test AP50 **62,33% / 84,40%**, cAcc **2,58% / 21,74%** (ResNet / ConvNeXt). [Báo cáo chi tiết](docs/project/FINAL_BACKBONE_COMPARISON_2026-10-20.md) có metrics từng task, forgetting, counting và đối chiếu paper.
- **Quy trình:** mỗi bước dữ liệu/model/train/eval có notebook ghi mục tiêu, config, output và thời điểm chạy thật. Kết quả tháng 09 giữ làm lịch sử.

## Điểm vào

| Tài liệu | Nội dung |
|---|---|
| [docs/README.md](docs/README.md) | Mục lục tài liệu hiện hành, tham khảo và lịch sử |
| [Kết quả baseline hai backbone](docs/project/FINAL_BACKBONE_COMPARISON_2026-10-20.md) | Báo cáo cuối, bảng/biểu đồ, kiểm chứng từ predictions và giới hạn so sánh paper |
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
  modeling/            foundation + ConvNeXt CUDA; notebook 04 kiểm chứng CLI/runner
  training/            pilot/transition và baseline hai backbone đủ 5 task
  evaluation/          S6 và notebook báo cáo baseline cuối đã có bằng chứng
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

Training notebooks mới ghi logs/checkpoints/predictions/policies và bản notebook đã execute dưới `data/training/`; [runbook](docs/project/TRAINING_NOTEBOOK_RUNBOOK.md) hướng dẫn upload folder data, chạy ngầm với tmux và resume. [Dry-run report](docs/project/TRAINING_NOTEBOOK_DRY_RUN.md) là evidence kiểm chứng local.

Các file kế hoạch/tiến độ/hướng dẫn GCP ở root là đường dẫn vào tài liệu đã tổ chức lại. Tái sử dụng tools/tests/configs đã khôi phục; kiểm chứng môi trường và kernel trên GPU đích trước khi resume.

## Kiểm chứng và bước tiếp theo

Phạm vi notebook processor/loader/PDP nền đã hoàn tất: notebook 01 đạt 11/11 gates; notebook 02 đạt 7/7 gates full CUDA smoke trên RTX 3060, batch 1 FP32, peak allocated 3,49/4,22 GiB ở 640/800. [Kết luận nghiệm thu](docs/project/PDP_FOUNDATION_ACCEPTANCE.md) giữ phạm vi local; [setup L4](docs/environment/GCP_L4_SETUP_2026-10-12.md) bổ sung 58 regression + 4 kernel tests và smoke resume thật. Full baseline đã hoàn tất; 16 epochs/task chưa tự chứng minh hội tụ.

Ngày 20/10 đã tính lại counting 10 stage, calibration validation Task 5 và final detection AP của cả hai backbone: khớp artifacts; 63 tests evaluation đạt. Conda `pdp` Python 3.10.22 cài đủ 64 pins, pip check/imports đạt; driver NVIDIA local chưa hoạt động nên kiểm tra này dùng CPU. Bước tiếp: phân tích lỗi trên validation, kiểm tra hội tụ và lập ablation/multiple-seed riêng. Giữ release/holdout và baseline gốc; không chọn policy mới dựa trên test. ConvNeXt có AP/counting tốt hơn nhưng forgetting trung bình cao hơn (8,67 so với 6,63 điểm AP50).

`pyproject.toml` yêu cầu Python 3.10; dependencies đã pin trong requirements. Chuẩn bị environment riêng và build CUDA kernel trên máy đích. Không bật `--shutdown` của runner như cơ chế kết thúc thuê instance.

## Nguồn và giấy phép

- Thư mục `pdp/` lấy từ [zyt95579/PDP_IOD](https://github.com/zyt95579/PDP_IOD) (commit `7702d91`). Code này dựa trên [MD-DETR](https://github.com/GauravBh1010tt/MD-DETR) và Deformable DETR của Hugging Face Transformers (Apache 2.0). Repo PDP_IOD gốc **không có file LICENSE**; mọi quyền với code gốc thuộc về tác giả của nó. Repo này chỉ dùng cho mục đích học tập, phi thương mại.
- Dataset RPC có giấy phép CC BY-NC-SA 4.0 và **không** được đưa lên repo.
