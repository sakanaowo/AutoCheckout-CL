# Review trực quan nhãn synthetic — 07/10/2026

[Notebook 03](../../../notebooks/data_preprocessing/03_annotation_review.ipynb) đã chạy **16:39:22–16:40:11 UTC+7**, 49 giây, run ID `520cad15-9b84-452f-809c-b3ee23c6c02a`. Trạng thái `VISUAL_REVIEW_READY_REPAIR_PENDING`.

## Kết quả đã kiểm chứng

| Hạng mục | Kết quả |
|---|---|
| Queue bbox/polygon lệch | 222 annotation / 222 ảnh, 52 case IoU dưới 0,9 |
| Ảnh đối chiếu | Đủ 222 PNG, giải mã được, đúng ID và geometry nguồn |
| Hiển thị notebook | 3 case lệch nhất; cell chọn case hiện ảnh đối chiếu và cutout nguồn |
| Nghiệm thu visual | PASS; crop giữ cả bbox và mọi mảnh polygon, có giới hạn canvas |
| Quyết định có evidence/reviewer | 0; CSV vẫn 222 dòng pending và byte-identical trước/sau run |
| Sửa nhãn | Chưa áp sửa; gate repair pending |
| Dữ liệu nguồn | Checksum annotation, 222 ảnh và cutout liên quan giữ nguyên |

Mỗi PNG gồm toàn cảnh khay, crop nguyên bản và crop có bbox đỏ/polygon cyan. Hai crop dùng cùng vùng và tỉ lệ; không sửa pixels/labels nguồn. IoU ở đây so sánh bbox lưu trong annotation với khung bao polygon, không phải chất lượng dự đoán detector.

## Cách review một case

1. Mở notebook, xem phần **2. Xuất ảnh đối chiếu**. Output đã lưu chứa ba case lệch nhất; gallery có đủ 222 case tại `runs/data_preprocessing/annotation_review/520cad15-9b84-452f-809c-b3ee23c6c02a/visuals/index.html`.
2. Trong phần **3. Chọn một case**, đặt `CASE_KEY = (2, 7318)` hoặc cặp task/annotation cần xem, rồi chạy cell đó. Khi mở kernel mới, chạy các cell trước để khởi tạo dữ liệu. Task ID là task nguồn 4-task để truy vết; mapping training chính vẫn là 5 task.
3. Đối chiếu pixels nguyên bản, bbox, polygon và cutout. Cutout chưa mang biến đổi/che khuất của ảnh ghép; không dùng nó như owner-mask đã xác minh.
4. Mở `data/processed/synthetic_annotation_review_v1/review_decisions.csv`, tìm đúng cặp `source_task,annotation_id`; điền `decision,evidence,reviewer`. Các lựa chọn và evidence cần có được giải thích trong phần **4. Ghi quyết định** của notebook. Notebook không tự đặt quyết định hoặc ghi đè CSV đã tồn tại.
5. Chạy lại cell chọn case để xem quyết định mới; chạy cell cuối để cập nhật số đã review. Quyết định cần tái sinh/thay mask chưa chứng minh repair đã được thực hiện.

Để chọn case khác trong cùng kernel, chỉ chạy lại cell chọn case. Gallery/PNG là snapshot theo run ID; khi chạy lại toàn bộ notebook, giữ notebook version và artifacts của run trước theo quy trình chung.

## Artifact và việc còn lại

- `runs/data_preprocessing/annotation_review/<run_id>/visuals/`: 222 ảnh đối chiếu và gallery HTML.
- `visual_manifest.json`: ID, path, IoU, crop box và checksum từng PNG.
- `source_asset_sha256.json`: checksum ảnh/cutout đầu vào cho lần chạy.
- `summary.json`: thời điểm, môi trường, nghiệm thu visual và trạng thái review/repair.
- `data/processed/synthetic_annotation_review_v1/`: queue, CSV quyết định và summary lần chạy mới nhất.

Còn review từng case có bằng chứng, rồi giữ nhãn với giải trình hoặc sửa/tái sinh có mask và provenance. Sau khi áp sửa, cập nhật JSON phụ thuộc và nghiệm thu geometry, mapping 5 task và leakage trước khi thêm synth vào training. Split ảnh thật đã khóa không bị thay đổi trong công việc này.

Kiểm chứng CPU: `tests/test_annotation_review.py` có 4 ca đạt (crop chứa cả hai geometry ở giữa/sát mép/fractional coordinates; ảnh đối chiếu có hai màu và không làm đổi nguồn). Regression dữ liệu/cấu hình cùng các test này đạt **37 tests**. CLI AI DevKit lint không chạy được offline vì package chưa có trong npm cache; kết quả trên dựa vào pytest và thực thi notebook, không phải lint AI DevKit.
