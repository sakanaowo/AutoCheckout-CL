# Xử lý tự động 202 ảnh — 08/10/2026

**Đã xử lý 202/202 case: 9 giữ có giải trình, 193 ảnh tái ghép thật.** CSV review đã chốt 202 dòng với reviewer OpenAI và evidence/output; không còn pending trong phạm vi này. 20 dòng pilot ngoài phạm vi 202 giữ nguyên, còn pending.

Notebook: [04_annotation_resolution.ipynb](../../../notebooks/data_preprocessing/04_annotation_resolution.ipynb). [Cấu hình](../../../configs/data/annotation_resolution_202_v1.json). [HTML trước/sau](../../../runs/data_preprocessing/annotation_resolution/f3292ec6-ce43-47e5-a07f-9198d60b1e43/resolution_report.html).

## API và thời điểm thực chạy

Theo yêu cầu mới “xử lý phần data 202 ảnh”, API được dùng làm reviewer cuối. Model **gpt-6.1-sol**, reasoning **medium**, prompt `rpc-annotation-final-resolution-v1`, nhận card + crop native + zoom cạnh + cutout. Action chỉ keep hoặc regenerate; geometry còn bất định dẫn đến tái ghép, không phải chờ người duyệt hoặc đoán tọa độ trên JPEG cũ.

API run `74db1f34-e4ec-4c56-bc5a-6e78fb5cc2f4`: **2026-10-08T13:16:30+07:00 → 2026-10-08T13:24:42+07:00**. Hai shard 201+1 request, đều completed, đủ 202 output hợp lệ, không lỗi. Batch IDs: `batch_6ac73593616081908803e2a5d5a78a5d`, `batch_6ac73595a1688190bfbe104128e94dd3`. Chi phí lượt chốt **ước tính $1,00714975**, từ usage và [giá Batch](https://developers.openai.com/api/docs/models/gpt-6.1-sol); không phải hóa đơn. Đây là lượt bổ sung so với batch 200 ảnh trước.

Apply/acceptance run `f3292ec6-ce43-47e5-a07f-9198d60b1e43`: **2026-10-08T13:31:19+07:00 → 2026-10-08T13:34:20+07:00**, CPU, không dùng GPU/cloud train. Chạy từ receipts cached, không gọi lại API.

## Việc đã áp dụng

- **9 giữ:** dùng JPEG nguồn và geometry nguồn nguyên bản; ghi giải trình AI. Original owner-map không có, nên không chứng nhận mask GT của các ảnh này.
- **193 tái ghép:** dùng đúng mọi cutout/instance của từng ảnh nguồn và nền trong catalog 208 đã kiểm tra. Giữ multiset SKU/instance, tạo ảnh thay thế, lưu seed/rotation/resize/placement/background SHA/owner map. Không flip sản phẩm. Nếu placement theo template không giữ đủ instance/visibility, dùng grid fallback được ghi trong recipe.
- Owner pixels dùng alpha ≥128, visibility mỗi instance ≥0,70. Bbox/area lấy từ owner map mới; COCO segmentation RLE bảo toàn fragment nhỏ, không lấy polygon giản lược làm bbox.
- Tái ghép là cách xử lý thận trọng khi không đủ căn cứ giữ; **193 không có nghĩa là 193 bbox nguồn đã được chứng minh sai**. Ảnh thay thế có nội dung/pose mới, không phải sửa tọa độ của cùng JPEG cũ.

## Dataset và nghiệm thu

Bản riêng: `data/processed/rpc_synth_resolved_202_v1/`, image root là repository root. Có annotations.json, resolution_manifest.json, images/ (193 JPEG mới), owner_masks/ (193 NPZ), recipes/ và 10 task JSON train/GT-full theo mapping 5 task 100+4×25 seed0 đã khóa.

- **202 ảnh / 2.846 object / 200 SKU**, giữ đúng multiset instance/SKU từng ảnh và toàn bộ scope.
- **2.721 object trên ảnh tái ghép** đã kiểm tra RLE → owner map → bbox/area; native bounds/dimensions/visibility/path/IDs đạt.
- Task labels nằm đúng range của từng task, RPC mapping dùng TaskConfig đã khóa; ảnh nhiều SKU có thể xuất hiện ở nhiều task như release real-only hiện tại.
- CSV `review_decisions.csv`: **193 regenerate (đã áp dụng) / 9 keep / 20 pending ngoài phạm vi**. Evidence ghi status `regenerated` hoặc `kept_by_ai`, ảnh output, target annotation ID và mask/decision fingerprint.
- Raw annotations/photos/cutouts/backgrounds và real-only release/test lock giữ checksum. Không thêm dữ liệu này vào release ảnh thật hoặc thay holdout.
- **40 tests đạt**: batch/parser/cache, action bất định → regeneration, COCO RLE theo Fortran order, owner/bbox/area, pixels thực sự chứa instance, giữ SKU khi boxes trùng, cache dataset, bảo toàn quyết định manual ngoài scope. Command: `python3 -m pytest tests/test_annotation_resolution.py tests/test_ai_annotation_batch.py tests/test_ai_annotation_review.py tests/test_annotation_review.py -q -o addopts='' --tb=short`.

Acceptance trong notebook đã chạy và PASS cho phạm vi 202. Điều này kết hợp review AI cho keep và geometry từ mask mới cho regenerate; không diễn giải thành GT accuracy của API. Không chứng nhận mục tiêu overlap 20–45% hoặc GRN/model: các phần đó thuộc thực nghiệm sau.

## Việc còn lại

20 case pilot không thuộc yêu cầu 202 vẫn pending; toàn bộ nguồn synth 20.000 ảnh chưa được phát hành lại. Bản 202 đã xử lý có thể được dùng làm augmentation có version sau khi loader/processor được kiểm chứng. Bước mô hình tiếp theo vẫn là notebook PDP nền → ConvNeXt-V2-Base → pilot trên RTX 4090 Vast.ai.
