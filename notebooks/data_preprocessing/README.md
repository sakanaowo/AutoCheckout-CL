# Notebook tiền xử lý

Chạy theo thứ tự **01 → 01b → 01c → 02** cho tài nguyên/split. **03 → 04** là nhánh review và xử lý synth độc lập với real-only release.

| Notebook | Mục đích / trạng thái |
|---|---|
| [01_data_audit](01_data_audit.ipynb) | Inventory/COCO; đã chạy |
| [01b_preprocessing_acceptance](01b_preprocessing_acceptance.ipynb) | Decode/hash và nghiệm thu background/cutout; đã chạy |
| [01c_protocol_comparison](01c_protocol_comparison.ipynb) | Phân tích trước quyết định 5 task; giữ output lịch sử |
| [02_split_and_acceptance](02_split_and_acceptance.ipynb) | Native split và task JSON; 9 gate đạt, test lock cố định |
| [03_annotation_review](03_annotation_review.ipynb) | Visual 222 case; Luna/Sol pilots, content audit và Batch 200; gate nội dung lịch sử FAIL |
| [04_annotation_resolution](04_annotation_resolution.ipynb) | API chốt 202 case, tái ghép thật, owner mask/RLE/task/CSV acceptance PASS |

CSV hiện hành có **193 regenerate đã áp dụng / 9 giữ / 20 pending ngoài scope**. Notebook 03 lưu output trước resolution; đọc notebook 04 và manifest/CSV hiện tại để biết trạng thái mới.

[Hướng dẫn chạy lại và các cờ API/apply](../../docs/data_preprocessing/OPERATIONS.md) · [Code/config/tests](../../docs/project/IMPLEMENTATION_INDEX.md) · [Báo cáo](../../docs/data_preprocessing/reports/README.md).

API cells chạy độc lập với cells torch/CUDA trong notebook 03. Submit/refresh/apply mặc định tắt, key ở environment hoặc `.env`, không lưu trong notebook. Mỗi lần chạy lưu run ID, mục đích, timestamps thực, config, output và nghiệm thu; dữ liệu nguồn ở `data/archive` giữ nguyên.
