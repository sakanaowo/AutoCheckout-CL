# Tài liệu AutoCheckout-CL

Cập nhật 09/10/2026. Protocol **5 task 100+4×25, seed 0**; training trên **RTX 4090 Vast.ai**. Điểm tiếp quản là bàn giao và chỉ mục triển khai bên dưới.

| Điểm vào | Mục đích |
|---|---|
| [Bàn giao](project/AGENT_HANDOFF.md), [snapshot đầu ngày 09/10](project/HANDOFF_2026-10-09.md) | Bàn giao hiện hành và snapshot lịch sử trước kiểm chứng model |
| [Kết luận processor/loader/PDP foundation](project/PDP_FOUNDATION_ACCEPTANCE.md) | Phạm vi notebook đã hoàn tất, evidence CPU/CUDA, giới hạn và 5 bước tiếp theo |
| [Chỉ mục triển khai](project/IMPLEMENTATION_INDEX.md) | Code mới, config, notebook, tests và đầu ra của từng phần |
| [Kế hoạch](project/IMPLEMENTATION_PLAN.md), [tiến độ](project/PROGRESS.md) | Phần đã đạt và thứ tự công việc tiếp theo |
| [ConvNeXt và train readiness](project/CONVNEXT_READINESS.md) | Điều kiện notebook 03 và các gate còn thiếu trên Vast.ai |
| [Dependencies/môi trường](environment/README.md) | Python 3.10, gói cần cài, lệnh CPU/CUDA và kiểm tra sau cài |
| [Tiền xử lý](data_preprocessing/README.md), [hướng dẫn vận hành](data_preprocessing/OPERATIONS.md) | Cách đọc kết quả, chạy lại và tiếp tục pipeline |
| [Schema](data_preprocessing/formats.md) | Tọa độ, COCO, task mapping và artifacts |
| [Báo cáo thực nghiệm](data_preprocessing/reports/README.md) | Kết quả theo ngày/run; giữ nguyên kết luận tại thời điểm chạy |
| [Nhật ký](timelines/README.md) | Mốc thực nghiệm và lịch sử triển khai |
| [Tổng quan kiến thức](project/overview.md) | Giải thích bài toán/PDP; trạng thái thực hiện xem bàn giao |
| [Tài liệu nguồn](references/README.md) | Handoff gốc, phân tích paper, PDF và notebook tham khảo |
| [Lịch sử tháng 09](archive/README.md) | Kế hoạch/tiến độ/GCP cũ |
| [Templates AI DevKit](ai/) | Templates kế thừa cho các phase phát triển |

Notebook có [chỉ mục riêng](../notebooks/README.md). Quy tắc thêm mới: quyết định/kế hoạch vào `project/`; hướng dẫn vào thư mục nghiệp vụ; báo cáo có ngày vào `data_preprocessing/reports/`; mốc thời gian vào `timelines/`; tư liệu nguồn vào `references/`; tài liệu đã thay thế vào `archive/`. Cập nhật chỉ mục và tiến độ trong cùng thay đổi triển khai.

Code dùng lại đặt trong `tools/`, `autocheckout/`, `pdp/`; notebook điều phối và ghi bằng chứng. Config version ở `configs/data/`; dữ liệu ảnh/mask ở `data/processed/`, run receipts/QA/snapshot ở `runs/`, hai thư mục này không vào git. Giữ notebook với mục tiêu và output nghiệm thu; ảnh gallery đầy đủ ở artifacts local.
