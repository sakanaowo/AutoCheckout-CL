# Tiền xử lý dữ liệu

Trạng thái 08/10/2026: **real-only release đã khóa**, **202 case synth đã xử lý**, **20 case pilot còn pending**. Chưa thêm synth vào release ảnh thật; chưa phát hành lại toàn bộ 20.000 ảnh synth.

| Bước | Notebook | Kết quả hiện tại |
|---|---|---|
| Inventory/COCO | [01_data_audit](../../notebooks/data_preprocessing/01_data_audit.ipynb) | Kiểm kê dữ liệu nhập, ảnh thiếu và protocol nguồn |
| Tài nguyên tiền xử lý | [01b_preprocessing_acceptance](../../notebooks/data_preprocessing/01b_preprocessing_acceptance.ipynb) | Full decode/hash 57.710 file; 208 nền, 7.502 cutout/200 SKU; phát hiện 222 case cần xem |
| Chọn protocol | [01c_protocol_comparison](../../notebooks/data_preprocessing/01c_protocol_comparison.ipynb) | Phân tích nguồn 4 task; quyết định sau đó là 5 task 100+4×25 |
| Split/nhãn task | [02_split_and_acceptance](../../notebooks/data_preprocessing/02_split_and_acceptance.ipynb) | Native real-only: 22.494 train, 1.503 val, 6.003 test; 9 gate đạt |
| Visual/AI/Batch review | [03_annotation_review](../../notebooks/data_preprocessing/03_annotation_review.ipynb) | 222 card/gallery; 2 pilot ×10 và Batch 200 có đề xuất, các gate nội dung lịch sử FAIL |
| Áp dụng và nghiệm thu | [04_annotation_resolution](../../notebooks/data_preprocessing/04_annotation_resolution.ipynb) | 202 ảnh/2.846 object: 9 giữ, 193 tái ghép; owner-mask/RLE xác minh 2.721 object; CSV còn 20 pending ngoài scope |

Đọc [hướng dẫn vận hành](OPERATIONS.md) trước khi chạy lại, [chỉ mục code/config/tests](../project/IMPLEMENTATION_INDEX.md) khi sửa triển khai, [báo cáo theo run](reports/README.md) khi truy nguồn. [Schema](formats.md) phân biệt native release với legacy 800.

Scope 202 nằm ở `data/processed/rpc_synth_resolved_202_v1`, có mapping 5 task nhưng là bản augmentation riêng. Gate này không chứng nhận mask gốc của 9 ảnh giữ, accuracy GT của API, toàn bộ synth hay mục tiêu overlap 20–45%. Processor/loader/PDP nền là bước tiếp theo; không chia lại holdout đã khóa.
