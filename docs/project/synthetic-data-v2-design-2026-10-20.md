# Thiết kế đề xuất: dữ liệu ảnh ghép v2

Ngày lập: 20/10/2026. Nhánh: `feat/synthetic-data-v2`, base đọc code `9aa1b6e`. **Trạng thái: đề xuất để nhóm duyệt; chưa triển khai generator, chưa sinh dữ liệu, chưa train.**

## 1. Mục tiêu và phạm vi

Cá cần kiểm tra nguyên liệu Hiển đã tách nền, tạo bộ ghép tự động và bàn giao dữ liệu có thể train, thay vì chỉnh tay hàng chục nghìn ảnh. Đầu ra nghiên cứu mong muốn là model học từ ảnh ghép nhưng nhận diện/đếm được trên ảnh quầy thật; chất lượng ảnh ghép chỉ được nghiệm thu sơ bộ bằng kiểm tra dữ liệu, khả năng chuyển sang ảnh thật phải đo bằng model sau đó.

[Ba tài liệu họp nguyên văn](../references/meeting-2026-10-20/README.md) là đầu vào. Đoạn chat mới nhất xác nhận An và Cá còn cần chốt cách xử lý. Giữ hai mức:

- Hướng đã rõ: train bằng ảnh ghép, val/test ảnh thật giữ nguyên; kiểm tra cutout sẵn có, ghép cảnh gần thực tế.
- Đề xuất từ handoff: sinh v2 thay bộ 20.000 ảnh cũ; không có SKU tương lai; chỉ nhãn hiện tại cho train; các tham số và FULL-SYN cần review trước thực hiện.

Lần thay đổi hiện tại chỉ đọc code/data metadata và lập kế hoạch. Phạm vi triển khai đầu tiên đề xuất là **inventory → gallery nguyên liệu → pilot 200 ảnh/task → nghiệm thu → full release → smoke loader**. FSA, sửa audit teacher, calibration theo nhóm và full training là các gói tiếp theo; không ghép việc xây generator với thay model thành một nghiệm thu duy nhất.

## 2. Quy tắc repo áp dụng

- [CLAUDE.md](../../CLAUDE.md): công việc nhỏ, có căn cứ, thay đổi đúng phạm vi, kiểm chứng hành vi.
- [Chỉ mục docs](../README.md): kế hoạch/quyết định ở `docs/project/`; nguồn ở `docs/references/`; báo cáo dữ liệu có ngày ở `docs/data_preprocessing/reports/`; tiến độ và chỉ mục cập nhật cùng thay đổi.
- [Hợp đồng dữ liệu](../data_preprocessing/formats.md) phải cập nhật **trước code** nếu bổ sung schema.
- [Quy tắc notebook](../../notebooks/README.md): thuật toán ở module/CLI; notebook điều phối và ghi run ID, thời gian thực với UTC offset, revision, config, seed, môi trường, checksum, kết quả/giới hạn. Giữ quy ước `Asia/Bangkok` của repo (UTC+7); không ghi timestamp chạy giả.
- Dataset mới ở `data/processed/`, receipts/gallery ở `runs/`, không đưa vào git. Không sửa `data/archive`, release real v1, dữ liệu resolved 202 hay holdout đã khóa.
- Không gọi reviewer API, tải model hoặc dùng GPU trong gói inventory/generator mặc định.

## 3. Những gì đã đọc trực tiếp ở nhánh này

| Nguồn | Kết quả kiểm tra và ý nghĩa |
|---|---|
| `data/archive/cutouts/sam/index.json` | 7.502 record; đủ 200 SKU; 10–48/SKU; toàn bộ đường dẫn cutout tồn tại. Mỗi record có `source`, `camera`, `ann_id`, `bbox`, `size`, `file`. |
| Camera metadata | ID 0: 876; 1: 2.417; 2: 3.143; 3: 1.066. Không cần suy camera ID bằng hình dạng mask. Nếu chỉ giữ camera 2, có **2 SKU còn dưới 5 cutout**, ngay trước khi review mask. |
| Ảnh nguồn | Toàn bộ `source` nối được với `instances_train2019.json`, nhưng thư mục ảnh `train2019/` không có local. Có thể kiểm kê/ghép từ PNG hiện có; tái chạy SAM cần thêm ảnh nguồn. |
| `data/archive/scripts/02_extract_objects.py` | Có resize từng cutout theo `max_side=1000`; 893 record có cạnh lớn nhất 1000. Không mặc định pixel PNG của mọi cutout còn cùng tỷ lệ vật lý. Comment camera của script là manh mối, chưa phải xác minh ánh xạ góc chụp. |
| `tools/annotation_resolution.py` | Có compositor, owner mask, bbox/RLE, visibility và recipe; phục vụ thay thế 193 ảnh, còn phụ thuộc bố cục/box ảnh cũ và grid fallback. Không phải generator mới theo task/phân bố train. |
| Fingerprint resolution | Hash bao gồm toàn file `annotation_resolution.py`. Tránh sửa/refactor file này chỉ để dùng chung helper, vì có thể phá khả năng reuse artifacts đã nghiệm thu. |
| `tools/make_task_json.py` | `build_task_coco` ánh xạ RPC ID → label; `task_files` chọn ảnh theo lớp có trong ảnh, chưa có `generation_task`. Gộp các scene task 2 chứa lớp task 1 rồi export chung sẽ kéo ảnh task 2 về task 1. |
| `autocheckout/taskcfg.py` | Có mapping chuẩn, 200 SKU trong 5 data task, 24 slot dự phòng. Không tạo mapping mới và không ánh xạ hai lần. |
| `pdp/datasets/coco_hug.py`, `pdp/main.py` | Loader đọc `img_folder/file_name`; train/test image roots tách được. Processor resize khi load. Đầu vào mới phải giữ đúng root và native bbox. |
| `configs/exp/native/common.sh` | Baseline dùng cùng RAW_ROOT cho train/test trong script, F14 tắt. FULL-SYN cần wrapper riêng ở gói train; không đổi defaults B1/B2. |
| Tests hiện có | Có `test_annotation_resolution.py`, `test_make_task_json.py`, `test_native_checkout.py`, `test_taskcfg.py`; có thể dùng làm regression. Lần lập kế hoạch chưa chạy lại các suite này. |

Đây là kiểm tra metadata/code có mục đích, **không phải** full decode/hash lại 7.502 ảnh hay nghiệm thu chất lượng bằng mắt. Số 208 nền và lịch sử resolution lấy từ báo cáo trước; gói inventory phải đối chiếu lại catalog nền.

## 4. Lựa chọn thiết kế

| Hướng | Lợi ích | Giới hạn / quyết định |
|---|---|---|
| Chia lại nhãn 20.000 ảnh cũ | Nhanh, dùng ảnh đã có | Không xóa SKU tương lai trong pixel; không khôi phục seed/mask chủ. Không chọn cho mục tiêu handoff mới. |
| Mở rộng compositor resolution ngay trong file cũ | Có sẵn hình học và kiểm thử | Làm đổi fingerprint lịch sử; logic template/grid chưa phù hợp bố cục v2. Không chọn cách sửa trực tiếp. |
| **Generator v2 riêng, tái dùng mapping/IO và helper hình học ổn định** | Tách version, có task ownership, provenance, bảo toàn baseline | Phải bổ sung sampling/placement/QA. Đây là hướng đề xuất. |

Không cần viết lại loader hay framework train để sinh ảnh. Không dùng image generation AI cho việc ghép: ảnh sản phẩm/SKU phải được bảo toàn và hình học phải xác định được từ mask.

## 5. Luồng dữ liệu và hợp đồng đề xuất

```mermaid
flowchart LR
    A[Cutout index và PNG] --> B[Inventory và review nguyên liệu]
    C[Nền khay và nguồn train] --> B
    D[TaskConfig và thống kê train thật] --> E[Lập recipe theo task]
    B --> E
    E --> F[Render RGB và owner mask]
    F --> G[Bbox và RLE từ visible mask]
    G --> H[Train JSON: chỉ lớp hiện tại]
    G --> I[GT full: chỉ audit]
    H --> J[Nghiệm thu và smoke loader]
    I --> J
```

### Layout đề xuất

```text
configs/data/synth_v2.json                       cấu hình versioned, phù hợp convention JSON hiện tại
autocheckout/synthetic.py                       catalog, thống kê, sampling, export
autocheckout/compositing.py                     render một scene và geometry
tools/synth_v2.py                               CLI inventory/pilot/generate/verify
notebooks/data_preprocessing/05_synthetic_v2.ipynb
data/processed/rpc_synth_v2_<version>/
  images/task_<t>/                              JPEG, relative paths từ dataset root
  owner_masks/task_<t>/                         PNG uint16; 0 là nền
  recipes/task_<t>/                             recipe deterministic
  annotations/task_<t>_rpc.json                 full GT canonical, category_id RPC 1..200
  tasks/synth_v2/                               5 cặp train/gt_full, model labels 0..199
  task_config.json                             copy byte-identical mapping real v1
  manifest.json                                identity, hashes, config, phiên bản
runs/data_preprocessing/synthetic_v2/<run_id>/   thời gian, QA, gallery, review và receipts
```

- Mỗi ảnh có `generation_task` bất biến. Export task t chỉ lấy ảnh `generation_task == t`, không tìm ngược mọi ảnh chứa SKU t trong union.
- Task 1: toàn bộ vật thuộc lớp 0..99 và được gắn nhãn. Task t≥2: tối thiểu 1 vật mới, có vật cũ theo tỷ lệ đã duyệt; không có label ≥ `seen_classes(t)`.
- `train` chỉ nhãn lớp task t; `gt_full` giữ mọi vật với `labeled` để audit. Hai file có cùng image IDs/file_names. Loader train không đọc `gt_full`.
- Canonical JSON dùng RPC IDs; chỉ `build_task_coco` chuyển sang model labels khi export từng task. Không truyền task JSON đã remap vào `--train-source` vốn chờ RPC IDs.
- Dataset image root là thư mục release synth mới; real evaluation root vẫn `data/archive`. Khi chuẩn bị training bundle, copy byte-identical `val_task_<t>.json`, `val_full.json`, `test_full.json` từ release thật; không tái split/remap.
- `owner == instance_id` là visible mask cuối. Bbox phải bằng `mask_bbox(mask)`, `area == mask.sum()`, RLE round-trip đúng. **Không** yêu cầu IoU hình chữ nhật với silhouette ≥0.98.
- Rendering alpha và annotation mask tách rõ: owner lấy support `alpha >=128` sau transform, shadow không được tính là vật. Feather ở viền không được làm mất vật nhỏ; kiểm tra ảnh RGB song song với geometry.
- Mỗi vật lưu cutout/hash, SKU, camera/source, scale, rotation, vị trí, layer, màu/bóng, full-mask area và visible fraction; mỗi ảnh có background/hash, seed, config identity, JPEG parameters.
- Seed theo `(dataset_seed, task_id, image_index, retry_index)` bằng hàm hash ổn định, không dùng Python `hash()`. Workers không làm đổi kế hoạch sampling hay thứ tự JSON.
- Byte-identical áp dụng payload ảnh/mask/annotation/recipe trong cùng code/config/assets và phiên bản thư viện. Run timestamps/host/duration ở receipt riêng, không nhét vào payload rồi yêu cầu checksum giống giữa hai lần chạy.
- Fingerprint khác thì output version mới; không âm thầm ghi đè dữ liệu đã duyệt. Không publish một release thiếu ảnh do placement thất bại.

### Thống kê train thật

Chỉ dùng `splits/train.json` và ảnh thuộc split train để đo số vật, level, bbox area/aspect, overlap bbox, màu/sáng. Bbox overlap là proxy, **không phải** phần trăm diện tích mask bị che; nếu thiếu mask thật thì ghi rõ giới hạn.

Không suy kích thước vật lý từ pixel PNG đã resize; dùng bbox/metadata nguồn để kiểm tra scale, rồi đối chiếu thống kê theo SKU trên train thật. Không khẳng định có thể khôi phục chính xác tỷ lệ gốc nếu crop/resize metadata không đủ.

Số ảnh full đề xuất bằng log B2: `[21752, 12534, 12129, 11850, 13377]`, tổng **71.642 ảnh**. Đây là tổng lượt ảnh theo task, không phải 22.494 ảnh train thật duy nhất. Bằng số ảnh/epochs không đồng nghĩa bằng số object supervision; phải báo cáo cả vật mới/cũ và ngân sách FSA bổ sung.

## 6. Những điều phải chốt bằng evidence và review

| Điểm | Đề xuất ban đầu | Cách khép |
|---|---|---|
| Camera → góc và tư thế | Metadata camera có sẵn; chưa dùng heuristic để thay thông tin này | Gallery theo camera/SKU, xác minh nguồn và review. Không gọi độ vuông vức là góc chụp đã xác định. |
| ≥5 cutout/SKU | Ngưỡng nghiệm thu nguyên liệu | Nếu thiếu sau lọc, báo SKU cụ thể; bổ sung tư thế được duyệt hoặc lấy lại ảnh nguồn. Không tự lấy mask bị loại. |
| Vật mới 40–60%, visible ≥70% | Tham số pilot, chưa là tối ưu | Review gallery và phân bố train. Quy tắc làm tròn số vật phải rõ ở số lượng nhỏ. |
| Không flip, rotate toàn vòng | Theo handoff | Unit test không phản chiếu chữ; visual review hình dáng và nhãn. |
| Cân bằng SKU ±20% | Theo handoff | Tính quota trước render, kiểm tra feasibility và counts sau render. Không bù bằng việc lặp cùng ảnh. |
| Phân bố scale/overlap | Fit train-only, không tự đoán | Báo cáo phân bố riêng từng SKU/camera; chốt config sau pilot. |
| Giống ảnh thật | Gallery tối thiểu 100 ảnh, phủ 5 task/3 level, xem thêm nhóm lỗi | Nhóm ghi reviewer, quyết định, lý do và config hash. Metrics hình ảnh không thay visual QA. |
| Dùng real statistics | Được handoff cho phép | Ghi “train bằng ảnh ghép, dùng thống kê train thật để thiết kế”, không quảng bá hoàn toàn không dùng dữ liệu thật. |
| DINO/CLIP, CycleGAN, trộn real | Chưa làm ở v2 đầu tiên | Gói/thí nghiệm riêng nếu nhóm quyết định; không tự tải model hay trộn dữ liệu. |

## 7. Điều kiện chuyển sang train

Generator không giải quyết tự động teacher tự tin nhưng sai, prototype drift hoặc thiếu recall nhãn giả. I3/F14 là biện pháp dự kiến, cần audit trên đúng preprocessing.

Gói train sau data cần: export/load FSA ConvNeXt round-trip giữ backbone/config; audit teacher dùng processor checkpoint và full GT để phân loại nhầm mới→cũ; lưu policy threshold theo nhóm chọn trên val; log theo handoff; thử task 1 trước task 2–5.

Điểm dừng phải so **val với val** cùng định nghĩa metric, không so val FULL-SYN với test B2. Handoff vừa cấm dùng val để chỉnh generator vừa đề xuất cải thiện ảnh sau khi val kém: nhóm phải chốt đây là go/no-go một lần hay vòng model selection trên val. Nếu dùng val lặp, ghi rõ điều đó và không gọi val là holdout chưa dùng. Test luôn chỉ báo cáo.

FULL-SYN thay nhiều yếu tố và có FSA thêm compute; dùng để đo cả cấu hình tổng hợp, không quy hiệu quả riêng cho ảnh ghép. FULL-REAL giúp so dữ liệu nhưng còn khác mức nhãn thiếu/SKU tương lai, cần nêu giới hạn.

## 8. Bàn giao

Mốc đầu tiên để Cá/An/Hiển xem: inventory đủ/thiếu, gallery camera/mask, thống kê train và đề xuất config. Mốc thứ hai: 1.000 ảnh pilot, train/gt_full, gallery và nghiệm thu. Chỉ sau review pilot mới sinh full.

Phân công đề xuất: Cá điều phối pipeline và evidence; Hiển hỗ trợ nguồn cutout/giải thích extraction; An cùng Cá chốt protocol/chất lượng. Đây không phải xác nhận phân công của nhóm.

[Kế hoạch từng bước](synthetic-data-v2-plan-2026-10-20.md).
