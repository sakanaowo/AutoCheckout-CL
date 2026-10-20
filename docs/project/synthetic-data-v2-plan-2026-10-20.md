# Synthetic Data v2 Implementation Plan

> **For agentic workers:** Khi người dùng yêu cầu triển khai, dùng `superpowers:executing-plans` để thực hiện từng task và đánh dấu checkbox. Kế hoạch này chưa phải lệnh bắt đầu triển khai hoặc chạy full dataset/train.

**Goal:** Bàn giao bộ train ảnh ghép 5 task có nhãn, provenance và bằng chứng chất lượng, bắt đầu bằng kiểm kê cutout và pilot để nhóm duyệt.

**Architecture:** Generator độc lập với baseline/resolution cũ. Catalog và statistics train-only tạo recipe theo task; renderer tạo RGB/owner mask; exporter remap đúng một lần và verifier chặn lỗi trước release.

**Tech Stack:** Conda `pdp`, Python 3.10, NumPy, Pillow, pycocotools, pytest; dependency đã có. Config JSON theo convention repo; chưa thêm model/API.

**Spec:** [Thiết kế đề xuất và bằng chứng đọc code](synthetic-data-v2-design-2026-10-20.md). Nguồn: [handoff cuộc họp](../references/meeting-2026-10-20/README.md).

**Status:** DRAFT FOR TEAM REVIEW — lập trên `feat/synthetic-data-v2`, base `9aa1b6e`; chưa có code v2, chưa có dataset v2, chưa có training run mới.

## Global Constraints

- Protocol `100+4×25`, seed mapping `0`; dùng nguyên `task_config.json` real v1. 24 slot dự phòng không sinh dữ liệu.
- Không sửa `data/archive`, real release/holdout, resolved 202, hoặc defaults baseline B1/B2.
- Task 1 đủ nhãn; task t≥2 chỉ lớp hiện tại có nhãn train; full GT chỉ audit. Không có SKU tương lai trong pixel của scene.
- Không flip. Không dùng ảnh/statistics val/test để fit generator.
- Pilot `200 ảnh/task`; full counts đề xuất `[21752,12534,12129,11850,13377]` chỉ sau nhóm duyệt pilot.
- `new_ratio=[0.4,0.6]`, `min_visible=0.7`, số vật mỗi SKU mới ±20%, ≥5 cutout/SKU là tham số đề xuất chờ nghiệm thu feasibility, không tự âm thầm nới.
- Bbox/area/RLE từ owner mask cuối; native pixels, processor resize sau.
- Notebook/run artifacts ghi thời gian thực, checksum và giới hạn; không lưu gallery/data vào git, không giả trạng thái PASS.
- Chỉ plan/docs ở lượt hiện tại; mọi task dưới đây là công việc **chưa thực hiện**.

## Review Focus

1. Scene task 2 có SKU task 1 không được lọt vào file train task 1 — kiểm thử Task 4.
2. Cutout sau lọc thiếu 5/SKU hoặc mất source/camera không được thay thế im lặng — Task 1.
3. Feather/overlap có nhãn đúng về toán nhưng RGB không còn vật dễ thấy — Task 3 + visual gate Task 5.
4. Worker count/resume/timestamps làm đổi payload hoặc mất ảnh khi placement fail — Task 4/5.
5. Root ảnh khác giữa synth và real, category remap hai lần, full GT đi vào train — Task 4/6.

## Mốc và phạm vi

| Mốc | Bàn giao | Phụ thuộc | Điều kiện chuyển tiếp |
|---|---|---|---|
| M0 | Inventory, gallery cutout, camera/scale questions | Dữ liệu hiện có | Nhóm chốt danh sách nguyên liệu/quy tắc |
| M1 | Sampler + compositor + exporter có tests | M0 | Geometry/task/reproducibility đạt trên fixtures |
| M2 | 200 ảnh/task, QA/gallery và review record | M1 | Nhóm duyệt hash config/dataset pilot |
| M3 | Full release, manifest và báo cáo | M2 | Full mechanical QA đạt, số lượng đủ |
| M4 | Smoke loader 640, bundle val/test giữ nguyên | M3 | Có thể chuyển cho gói train riêng |

M0 làm CPU, không phụ thuộc GPU driver. Chưa ước lượng giờ sinh full/dung lượng trước khi benchmark pilot. Ngân sách GPU/FSA không nằm trong nghiệm thu generator.

## Task 1: Inventory nguyên liệu, train statistics và hợp đồng

**Files:** tạo `autocheckout/synthetic.py`, `tools/synth_v2.py`, `configs/data/synth_v2.json`, `tests/test_synthetic_catalog.py`; tạo notebook `notebooks/data_preprocessing/05_synthetic_v2.ipynb`; sửa `docs/data_preprocessing/formats.md` trước code schema.

**Interfaces:**
- `inventory_assets(raw_root: Path, task_config: TaskConfig) -> dict`: đọc index, PNG, nền và nguồn; xuất catalog dùng relative paths/hash, camera ID gốc, counts, issues, không suy góc camera.
- `measure_train_statistics(train_coco: dict, image_root: Path) -> dict`: số vật/level, bbox area/aspect/overlap, sáng/màu và source IDs; không nhận val/test để fit.
- `load_reviewed_catalog(catalog: dict, decisions: list[dict]) -> dict`: giữ quyết định `keep/reject/pending`, reviewer/reason, không dùng `pending` cho full generation.
- CLI đề xuất `python -m tools.synth_v2 inventory --config configs/data/synth_v2.json --run-dir <run_dir>`.

- [ ] Viết contract ở `formats.md` theo spec: camera/source, `generation_task`, canonical RPC ID, model label, ảnh root, owner/RLE và scope deterministic payload.
- [ ] Viết fixture nhỏ, tests `test_inventory_joins_source_camera`, `test_missing_cutout_is_reported`, `test_rejected_sku_shortage_blocks_generation`, `test_train_statistics_only_use_listed_images`; assert count/source/camera đúng, file hỏng có lỗi riêng, `<5` không được coi đạt.
- [ ] Chạy `conda run -n pdp python -m pytest tests/test_synthetic_catalog.py -q`; xác nhận test mới fail do chưa có implementation.
- [ ] Implement interfaces và gallery catalog theo SKU/camera; checksum file, alpha/bounds, nền train-only. Giữ index raw bất biến; capture raw/release lock hashes trước/sau.
- [ ] Chạy lại test trên cho PASS; notebook inventory thật ghi số kiểm tra, ảnh lỗi/thiếu, camera coverage, source join, disk estimate nguyên liệu. Không gọi đây là semantic acceptance.
- [ ] Xuất CSV review và gallery ≥5 cutout/SKU; nhóm xem tối thiểu 1.000 cutout, xem riêng trường hợp camera/scale bất thường. Ghi quyết định trước khi dùng catalog full.
- [ ] Cập nhật báo cáo có ngày + chỉ mục notebook/implementation/progress khi có run thật; checkpoint thay đổi theo chức năng sau kiểm chứng.

**Gate M0:** có giải thích camera IDs, chính sách fallback cho 2 SKU thiếu nếu chỉ dùng camera 2; 893 cutout cạnh 1000 được xem xét scale. Không tái chạy SAM khi chưa có ảnh nguồn.

## Task 2: Lập kế hoạch scene theo task và quota

**Files:** mở rộng `autocheckout/synthetic.py`, config; tạo `tests/test_synthetic_sampling.py`.

**Interfaces:**
- `plan_scenes(catalog: dict, statistics: dict, task_config: TaskConfig, config: dict) -> list[dict]`.
- Mỗi recipe gồm `schema_version`, `generation_task`, `image_index`, `image_id`, `seed`, `level`, `background_id`, `objects`; object có `instance_id`, `rpc_category_id`, `cutout_id`, `labeled`, camera và scale cơ sở. Renderer bổ sung transform thực tế.
- Seed/IDs từ task + index cố định; quota lập toàn dataset trước khi chia workers. Kế hoạch scene được lưu trước render.

- [ ] Viết tests `test_task1_has_only_current_skus`, `test_incremental_scene_has_old_and_new_without_future`, `test_new_sku_quota_within_twenty_percent`, `test_cutout_use_balanced_within_sku`, `test_infeasible_ratio_fails_explicitly`, `test_seed_does_not_depend_on_worker_order`.
- [ ] Chạy `conda run -n pdp python -m pytest tests/test_synthetic_sampling.py -q`, xác nhận fail rồi implement sampler.
- [ ] Lấy `(level, object_count)` theo phân bố train. Với t≥2 chọn số vật mới trong `[ceil(0.4*n), floor(0.6*n)]`, thêm điều kiện ít nhất 1 mới và 1 cũ; nếu tập rỗng, báo config infeasible, không tự đổi tỷ lệ.
- [ ] Phân bổ SKU mới bằng quota cân bằng trong mỗi task; chọn cutout luân phiên có shuffle seed ổn định. SKU cũ lấy trong union task trước, ghi policy sampling rõ trong config; không dùng source future.
- [ ] Xác minh PASS tests và thống kê recipe pilot trước render. Nếu không thể đồng thời đạt quota/level/count, trả báo cáo xung đột cho nhóm, không drop scene/SKU.

## Task 3: Renderer scene và nhãn visible-mask

**Files:** tạo `autocheckout/compositing.py`, `tests/test_synthetic_compositing.py`. Tái dùng `mask_bbox`, `encode_rle`, `decode_rle` từ `tools/annotation_resolution.py` qua import; không sửa file cũ chỉ để chuyển helper.

**Interface:** `render_scene(recipe: dict, catalog: dict, config: dict, raw_root: Path) -> dict` trả `image` (PIL RGB), `owner` (NumPy uint16), `annotations_rpc` (list), `recipe` đã resolve. Không ghi file trong hàm này.

- [ ] Viết tests trên mask bất đối xứng/fragment: `test_no_horizontal_reflection`, `test_final_owner_matches_rle_bbox_area`, `test_occlusion_checks_all_previous_instances`, `test_small_mask_survives_edge_processing`, `test_rgb_contains_owned_object`, `test_impossible_layout_returns_failure`.
- [ ] Chạy `conda run -n pdp python -m pytest tests/test_synthetic_compositing.py -q` để thấy fail.
- [ ] Implement scale có căn cứ nguồn/train stats, rotation 0–360°, alpha support `>=128`, erosion/feather cấu hình được; không áp một scale pixel PNG chung khi resize nguồn khác nhau.
- [ ] Implement placement gần biên vật đã đặt; kiểm tra cả vật mới và mọi vật trước còn ≥0.7 visible, vật nằm trong vùng khay cho phép. Bounded retries; không fallback grid âm thầm hoặc bỏ vật để báo thành công.
- [ ] Thêm shadow chung hướng sáng, màu/sáng/mờ/nhiễu nhẹ; giữ mask trước photometric effects. Xuất geometry từ owner cuối, shadow không chiếm owner.
- [ ] Chạy tests cho PASS và regression `conda run -n pdp python -m pytest tests/test_annotation_resolution.py -q`. Kiểm tra trực quan fixture và một ít scene thật trước pilot.

## Task 4: Writer, task export và manifest

**Files:** mở rộng `autocheckout/synthetic.py`, `tools/synth_v2.py`; tạo `tests/test_synthetic_export.py`, `tests/test_synthetic_reproducibility.py`.

**Interfaces:**
- `export_task_annotations(coco_rpc: dict, task_id: int, task_config: TaskConfig) -> dict[str, dict]`: bắt buộc mọi image thuộc `generation_task==task_id`, tạo đúng cặp train/gt_full qua `build_task_coco`.
- `write_scene(rendered: dict, output_root: Path) -> dict`: ảnh JPEG, owner PNG16, recipe và file hashes; không ghi absolute paths vào payload.
- `generate_dataset(recipes: list[dict], catalog: dict, config: dict, raw_root: Path, output_root: Path, workers: int) -> dict`: dựng identity/manifest, resume theo recipe/config/asset hashes.
- CLI `pilot` ép 200/task và output riêng; `generate` dùng full config chỉ sau review record trùng pilot config; `verify` đọc release, không render lại.

- [ ] Viết test cụ thể: scene task 2 gồm RPC của task 1 và task 2; export chỉ task 2, train chỉ label mới, full giữ cả hai; **không đưa ảnh vào task 1**.
- [ ] Thêm tests remap đúng một lần, annotation IDs duy nhất, train/full cùng ảnh, reserved labels không xuất hiện, paths portable dưới image root, shadow/recipe không lọt vào COCO target.
- [ ] Thêm tests `test_serial_parallel_payload_bytes_equal`, `test_resume_keeps_payload_hashes`, `test_changed_config_refuses_existing_output`, `test_failed_scene_prevents_release_complete`, `test_run_timestamp_outside_payload`.
- [ ] Chạy hai test files cho FAIL; implement writes tạm→rename, canonical ordering, SHA-256 và reuse helper `autocheckout.io.save_json`. MD5 thêm cho tương thích runner khi cần.
- [ ] Không dùng `make_task_json.main` trên union scene của mọi task. Reuse hàm ánh xạ trên từng canonical task JSON riêng; không sửa hành vi legacy nếu không cần.
- [ ] Chạy tests mới và `conda run -n pdp python -m pytest tests/test_make_task_json.py tests/test_taskcfg.py -q`; PASS không thay thế kiểm tra pilot dataset.

## Task 5: QA, pilot 200/task và review nhóm

**Files:** tạo `tools/synth_v2_qa.py`, `tests/test_synthetic_qa.py`; hoàn thiện notebook 05; báo cáo thực chạy `docs/data_preprocessing/reports/synth-v2-YYYY-MM-DD.md`.

**Interface:** `verify_dataset(output_root: Path, task_config: TaskConfig, locked_release: Path) -> dict` trả `gates`, `counts`, `failures`, `hashes`, `limits`. Không tự sửa dữ liệu bị lỗi.

- [ ] Tests cố ý cài lỗi: future SKU, old label train, bbox lệch mask, RLE hỏng, ảnh thiếu/trùng hash, background holdout, annotation ngoài ảnh, quota lệch, config/recipe mismatch; tất cả phải FAIL đúng gate.
- [ ] Run `conda run -n pdp python -m pytest tests/test_synthetic_qa.py -q` trước/sau implementation.
- [ ] Chạy CLI pilot qua notebook: `conda run -n pdp python -m tools.synth_v2 pilot --config configs/data/synth_v2.json --out-dir data/processed/rpc_synth_v2_pilot_v1 --workers 1 --run-dir <run_dir>`.
- [ ] Chạy `verify`; cùng pilot render lại ở output tạm với worker count khác, đối chiếu payload hashes; không nhận timestamps manifest làm lỗi giả.
- [ ] Xuất gallery 100 ảnh phủ 5 task × 3 level, mỗi stratum tối thiểu 1 nếu có; đánh dấu mới/cũ bằng màu khác nhau, thêm ảnh có nguy cơ mask/occlusion. Báo cáo counts, distributions, failures, retry rates, seconds/image và bytes/image.
- [ ] Đo scale/aspect/overlap, màu/sáng so với train thật; ghi mức lệch thay vì tự tạo ngưỡng PASS thống kê chưa được duyệt. FID/KID không bắt buộc giai đoạn này.
- [ ] An/Cá và người hỗ trợ cutout review; lưu `pilot_review.json` có reviewer/date, config hash, manifest hash, keep/change và lý do.

**Gate M2:** geometry/protocol/provenance đạt **và** có review pilot đồng ý. “Tests pass” không thay được duyệt độ chân thực. Nếu chỉnh tham số thì output/pilot version mới, giữ lịch sử.

## Task 6: Full release và bàn giao loader

**Files:** bổ sung `tests/test_synthetic_loader.py`; hoàn thiện CLI/notebook 05, `docs/data_preprocessing/OPERATIONS.md`, formats, các chỉ mục và báo cáo. Không sửa engine hay baseline.

- [ ] Dùng benchmark pilot tính disk/CPU time và headroom cho JPEG/PNG/RLE/recipe; kiểm tra dung lượng trống trước full. Không hứa thời gian/chi phí khi chưa đo.
- [ ] Sinh 71.642 ảnh theo counts đã duyệt vào version mới; ghi fail/resume nếu ngắt; không tuyên bố release hoàn tất với scene thiếu.
- [ ] Full decode/hash/geometry/protocol checks, counts theo task/SKU/level, duplicate scan; so lại raw/real holdout hashes.
- [ ] Copy byte-identical task config và real val/test JSON vào bundle khi cần; train root = synth dataset, test root = raw archive. Lưu cả nguồn và checksum.
- [ ] Test loader trên fixture và sample thật 5 task: processor 640, native→normalized→native boxes, label ranges, ảnh root, no flip, train không có GT cũ; assert val/test bytes không đổi.
- [ ] Chạy `conda run -n pdp python -m pytest tests/test_synthetic_loader.py tests/test_native_checkout.py -q`; smoke CPU batch load không được gọi là nghiệm thu model/GPU.
- [ ] Notebook kết thúc với manifest, gate summary, giới hạn semantic QA, đường dẫn tải/chuyển máy; cập nhật report/index/PROGRESS/timeline với thời gian chạy thật.

**Gate M4:** dữ liệu sẵn sàng cho smoke train, không phải chứng nhận cAcc/AP hay production.

## Gói kế tiếp: FULL-SYN / FULL-REAL (lập kế hoạch riêng sau data)

| Việc | Code phải đọc/kiểm chứng | Điều kiện trước campaign |
|---|---|---|
| FSA ConvNeXt → export → PDP | `pdp/runtime.py`, `pdp/main.py`, `pdp/models/backbones.py`, `configs/exp/common.sh` | Reload đúng backbone/config/weights, không silently reinitialize; thống nhất epochs FSA và PDP task 1, ghi compute bổ sung |
| Audit teacher | `pdp/ppg_audit.py`, `pdp/ppg.py` | Processor đúng runtime; tách nhầm SKU mới/cũ/tương lai/background; full GT chỉ dùng chấm |
| I3/F14 + counters | `pdp/engine.py`, native config mới | 0.5/0.5 theo handoff; đếm trước/sau filter, regression labels/teacher, giữ baseline |
| Ngưỡng theo nhóm | `autocheckout/counting.py`, `tools/eval_count.py`, policy schema | Val-only search, lưu vector thresholds và NMS/tie-break, test chỉ reload |
| Log/task summary và runner | `autocheckout/training_workflow.py`, native scripts | Checkpoint/data hashes, module train/freeze, losses/grad, time/cost, per-SKU AP/confusion |
| Task 1 go/no-go | Predictions/policy val B2 và FULL-SYN | Val–val cùng metric; nhóm chốt −15 điểm hay ngưỡng khác, quy tắc dùng val để quyết định rerun |
| FULL-REAL đối chứng | Cùng method với FULL-SYN, root real riêng | Budget riêng; mô tả khác biệt old/future unlabeled objects và số supervision |

Không chạy train, cài driver, thuê GPU, CycleGAN, DINO/CLIP, LoRA, K=3 hoặc demo Task 6 trong gói generator này.

## Kiểm tra kế hoạch và bàn giao

- [x] Đọc code/rules/docs và metadata nguyên liệu trên nhánh hiện tại.
- [x] Phân biệt quyết định nhóm với tham số đề xuất; ghi các mâu thuẫn của handoff.
- [x] Chỉ định module, giao diện, tests, gates và đầu ra từng mốc.
- [x] Tách deterministic payload khỏi run receipts và bảo toàn lịch sử resolution.
- [ ] Nhóm duyệt thiết kế/plan và các câu hỏi M0.
- [ ] Người dùng yêu cầu bắt đầu triển khai; mốc đầu tiên là Task 1, không tự nhảy đến full generation.

Lệnh và file v2 trong task ở trên là **đề xuất**, chưa tồn tại tại thời điểm lập kế hoạch. Người nhận đọc cả spec và plan; giữ các câu hỏi còn mở trong review record, không diễn giải checkbox chưa chọn thành việc đã hoàn thành.
