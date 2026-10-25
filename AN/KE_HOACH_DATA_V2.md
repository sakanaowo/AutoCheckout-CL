# Kế hoạch làm dữ liệu ảnh ghép v2

File theo dõi tiến độ phần dữ liệu. Cập nhật mỗi khi xong một task hoặc có quyết định mới.

- Quyết định gốc: [D1–D21](../docs/project/synthetic-data-v2-decisions-2026-10-21.md)
- Plan kỹ thuật chi tiết (interface, tests): [plan v2](../docs/project/synthetic-data-v2-plan-2026-10-20.md)
- Thiết kế: [design v2](../docs/project/synthetic-data-v2-design-2026-10-20.md)

**Quy ước trạng thái:** `[ ]` chưa làm · `[~]` đang làm · `[x]` xong (có bằng chứng) · `[!]` bị chặn, chờ quyết định.

## 1. Mục tiêu

Bộ train **71.642 ảnh ghép** cho 5 task (100+4×25 SKU, seed 0), cộng bộ đánh giá lấy từ RPC gốc. Yêu cầu:
- nhãn đúng theo từng task;
- có nguồn gốc (hash, seed, config) cho từng ảnh và từng vật;
- đã qua kiểm tra tự động và duyệt bằng mắt;
- loader đọc được, sẵn sàng cho FULL-SYN.

Không thuộc phạm vi file này: train, FSA, cổng dừng sau task 1 (thuộc gói train).

## 2. Các cổng duyệt

| Cổng | Khi nào | Người duyệt quyết | Điều kiện qua |
|---|---|---|---|
| **M0** | Xong phần nguyên liệu (Giai đoạn 1) | Nhóm hình dạng từng SKU, cutout giữ/loại, tỉ lệ chai/lon đứng–nằm | Mọi SKU có ≥ 5 cutout ở góc được dùng, hoặc có quyết định riêng cho SKU thiếu |
| **M1** | Xong bộ sinh (Giai đoạn 2) | — (tự động) | Toàn bộ test PASS; chạy lại cho ra đúng từng byte |
| **M2** | Xong pilot 200 ảnh/task (Giai đoạn 3) | Ảnh ghép đủ giống thật chưa, có cần chỉnh tham số không | Kiểm tra tự động đạt **và** có biên bản duyệt |
| **M4** | Xong bản đầy đủ + bộ đánh giá + loader (Giai đoạn 4) | — | Đủ số ảnh, kiểm tra đạt, loader đọc đúng |

## 3. Danh sách task

### Giai đoạn 0 — Chuẩn bị

| ID | Task | Đầu ra | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| T0.1 | Commit các doc quyết định 21/10 lên `feat/synthetic-data-v2` | Commit `6d19507` | `git log` | [x] |
| T0.2 | Cập nhật hợp đồng dữ liệu `formats.md` trước khi viết code: catalog (`cutout_set`, `shape_group`, SHA-256), `generation_task`, recipe, bộ đánh giá | `formats.md` mục 7 (7.1 nguyên liệu, 7.2 release, 7.3 bộ đánh giá) | Review diff | [x] |
| T0.3 | Tạo conda env riêng `sam` (ultralytics + Torch CUDA) để cắt cutout; không đụng env `pdp` | Env `sam`: ultralytics 8.4.183, torch 2.14.1+cu130, CUDA OK | Chạy SAM trên 1 ảnh: làm cùng T1.4 | [x] |

### Giai đoạn 1 — Nguyên liệu (cổng M0)

| ID | Task | Đầu ra | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| T1.1 | Tải **toàn bộ ảnh camera 0** của train2019 (13.434 ảnh, khoảng 2,2 GB) qua Kaggle API, tải từng file; tiếp tục được nếu bị ngắt | `data/archive/train2019/` (chỉ camera 0), danh sách + SHA-256 | 13.434/13.434, 0 thiếu/thừa, tất cả mở được (2592×1944), 2,48 GB; `runs/data_preprocessing/train2019_extract/manifest_cam0.json` | [x] |
| T1.2 | Kiểm kê catalog cutout: hash, camera, nguồn, kích thước, `cutout_set` | `data/processed/rpc_synth_v2_v1/catalog/cutouts_sam_v1.json` | 7.502/7.502 dùng được, 0 lỗi; 7 test PASS. Chặn SKU < 5 làm ở T1.8 khi có nhóm hình dạng | [x] |
| T1.3 | Lọc val2019: bỏ 208 ảnh làm nền → 5.792 ảnh; đo thống kê thiết kế (số vật/level, kích thước theo SKU, chồng lấn, màu/sáng) | `data/processed/rpc_synth_v2_v1/design/` (val_excluded_backgrounds, design_val, design_statistics, receipt) | 5.792 ảnh (1.933/1.921/1.938), 71.108 vật; từ chối test2019 (test) | [x] |
| T1.4 | Kiểm tra tương đương SAM: cắt lại 20–30 ảnh nguồn **đã có cutout**, so mask mới với mask cũ | `runs/data_preprocessing/sam_equivalence/comparison.json` | **Đạt:** 42/42 cặp, IoU mask thấp nhất 0,993, trung vị 0,998 (7 cặp trùng tuyệt đối). sam2.1_b.pt SHA-256 `f1a9cf2d…5e2e47` | [x] |
| T1.5 | Cắt bổ sung camera 0 cho nhóm dạng trụ (drink, canned_food, alcohol, milk; khoảng 51 SKU), mục tiêu 15–20 cutout/SKU, cùng script/tham số/trọng số | `data/processed/cutouts_cam0_v1/` | 4.047 cutout mới (loại 2.353, chủ yếu vì IoU); gộp với bộ cũ: 11.075 cutout, 474 trùng nguồn giữ bản cũ; camera 0 trung vị 23/SKU | [x] |
| T1.6 | Duyệt cutout bằng mắt, loại thủ công lỗi (dính mâm xoay, mask vỡ, hộp thấy hai mặt) | `catalog/reviews.json`, mỗi cutout có lý do; trang tổng hợp loại theo lý do `runs/data_preprocessing/m0_review/rejected_<lý do>_*.jpg` | Duyệt 2 vòng + sửa bằng SAM-L; **chờ người dùng xác nhận trang tổng hợp** | [x] |
| T1.7 | Gán **nhóm hình dạng** cho 200 SKU (dẹt/trụ/hộp cứng) và đo tỉ lệ chai/lon đứng–nằm trên mẫu val2019 | `catalog/skus.json`: 123 flat (camera 2) / 64 upright (camera 0 + 1% camera 2) / 13 mixed (50/50) | Mình xem cả 20 trang tờ duyệt; người dùng chấp nhận 21/10; không SKU nào thiếu cutout | [x] |
| T1.8 | Gallery theo SKU/camera/nhóm và **họp duyệt M0** | Gallery + `m0_review.json` | Có người duyệt, ngày, quyết định | [x] |

### Giai đoạn 2 — Bộ sinh ảnh (cổng M1)

| ID | Task | Đầu ra | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| T2.1 | Lập kế hoạch cảnh: số vật theo level, tỉ lệ vật mới 40–60% (kẹp `[1, n−1]`), SKU cũ chọn đều, quota cân bằng, seed ổn định | `plan_scenes` + recipe | Test: task 1 chỉ SKU task 1; không SKU tương lai; quota ±20%; seed không phụ thuộc số worker | [x] |
| T2.2 | Renderer: xoay/scale theo val, chồng lấn, mỗi vật còn thấy ≥ 70%, bóng, màu/sáng; bbox/RLE lấy từ mask cuối | `render_scene` | Test: không lật ảnh, bbox khớp mask, vật nhỏ không mất | [x] |
| T2.3 | Ghi file, xuất nhãn theo `generation_task`, manifest, chạy tiếp được nếu bị ngắt | `generate_dataset`, CLI | Test: ảnh task 2 không lọt vào task 1; ID lớp chỉ ánh xạ một lần; chạy song song ra đúng từng byte như chạy tuần tự | [x] |

### Giai đoạn 3 — Pilot (cổng M2)

| ID | Task | Đầu ra | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| T3.1 | Công cụ kiểm tra dữ liệu (`verify_dataset`) | Báo cáo kiểm tra tự động | Test cố ý cài lỗi: phải FAIL đúng chỗ | [x] |
| T3.2 | Sinh pilot 200 ảnh/task, đo s/ảnh, MB/ảnh, tỉ lệ đặt vật thất bại | `rpc_synth_v2_pilot_v1` | `verify` đạt | [x] |
| T3.3 | Gallery 100 ảnh (5 task × 3 level) + so phân bố với val2019, **họp duyệt M2** | `pilot_review.json` | Có người duyệt | [x] M2 duyệt 21/10 |

### Giai đoạn 4 — Bản đầy đủ (cổng M4)

| ID | Task | Đầu ra | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| T4.1 | Sinh đủ 71.642 ảnh `[21752, 12534, 12129, 11850, 13377]` | Release v2 + manifest | Đủ số ảnh, kiểm tra tự động đạt, không trùng | [x] o15, verify 14/14 (22/10) |
| T4.2 | Bộ đánh giá từ RPC gốc: val2019 đã lọc, test2019 đầy đủ, theo task, giữ `level`; thêm tập 6.003 ảnh cũ để so B1/B2 | `eval/` + checksum | Test: 5.792 / 24.000 ảnh, nhãn trong phạm vi các lớp đã học | [x] |
| T4.3 | Smoke test loader 640: tọa độ box, nhãn, thư mục ảnh, không nhãn cũ trong train | Test loader | PASS | [x] |
| T4.4 | Báo cáo `synth-v2-<ngày>.md`, cập nhật PROGRESS/chỉ mục | Báo cáo | Link hợp lệ | [x] |

### Song song (phục vụ cổng dừng của gói train, chạy CPU)

| ID | Task | Trạng thái |
|---|---|---|
| P1 | Code cAcc-M2 trong `counting.py`/`eval_count.py` + test | [x] |
| P2 | Tính lại cAcc-M2 task 1 của B2 từ `pred_val.npz` → mốc B | [x] |

## 4. Nhật ký quyết định trong lúc làm

Câu hỏi phát sinh sẽ được đưa kèm phương án đề xuất; người dùng chọn. Ghi lại đây sau khi chốt.

| Ngày | Câu hỏi | Phương án chọn | Lý do |
|---|---|---|---|
| 21/10 | Cắt lại toàn bộ cutout hay giữ và bổ sung? | Giữ 7.502 cutout + cắt bổ sung camera 0 + duyệt bằng mắt + kiểm tra tương đương | Cùng phương pháp thì cắt lại không tốt hơn; chỗ thiếu thật là góc ngang của chai/lon |
| 21/10 | Phạm vi tải train2019 | Toàn bộ camera 0 (13.434 ảnh) | Danh sách SKU dạng trụ chốt ở M0; tải một lần, chi phí thêm nhỏ |
| 21/10 | Môi trường chạy SAM | Env `sam` riêng | Giữ env `pdp` (64 gói đã ghim phiên bản) không đổi |
| 21/10 | Chọn camera cho từng SKU thế nào? | Tờ duyệt từng SKU: mình điền đề xuất, nhóm xác nhận | Chia theo nhóm hàng sai ở hộp thịt chữ nhật, hộp giấy |
| 21/10 | Tư thế đứng của chai/lon | Giữ khoảng 1% (người dùng chọn; đề xuất là 0%) | Val có dưới 0,5%; giữ một ít để phòng quầy thật có chai đứng |
| 21/10 | Đưa `AN/` lên git? | Commit lên nhánh | Cả nhóm xem được, có lịch sử |
| 21/10 | Notebook cho M0 | Một notebook `05_synthetic_v2` khi chốt M0 | Đúng quy tắc repo, không lặp việc |
| 21/10 | Chốt camera theo SKU | Chấp nhận đề xuất: 123 flat / 64 upright / 13 mixed | Sai sót sẽ lộ ra ở gallery pilot M2, sửa được từng SKU trong `skus.json` |
| 21/10 | Tỉ lệ tư thế cho 13 SKU mixed | 50/50 cố định | Chỉ 6,5% catalog; đơn giản |
| 21/10 | Cách làm T1.6 | Mình duyệt toàn bộ 4.958 cutout được dùng; nhóm xem trang tổng hợp các cutout bị loại | Lọc tự động không bắt được lỗi dính mâm xoay |
| 21/10 | Mâm xoay dính vào mask: khoảng 15–20% cutout camera 2, gần như toàn bộ ở SKU trong suốt/trắng | Loại, rồi SKU nào còn < 8 cutout thì cắt thêm ảnh camera 2 từ zip bằng đúng quy trình cũ và duyệt phần mới | Dùng quy trình đã kiểm chứng; viết bước tự xóa mâm thì rủi ro ăn mất bao bì trắng/trong |
| 21/10 | Mảnh mâm nhỏ ở mép | Loại luôn: thấy mâm là loại | Nhãn bbox/mask phải khớp sản phẩm |
| 21/10 | 105 hộp chụp ngang thấy hai mặt (mask sạch) | Giữ loại | Mỗi SKU đã đủ cutout thẳng mặt; tư thế hai mặt không có trên khay |
| 21/10 | Bố cục bản đầy đủ (M2) | A: giữ cách rải pilot v1; code xếp gọn để sẵn, mặc định tắt | Khoảng cách giữa các vật và độ chồng bbox đã khớp val; xếp gọn chỉ thu hẹp khoảng trống trong cụm nhưng làm vật bị che thật nhiều hơn (40–47% so với 28–39%) và độ phủ bbox vượt val; sai số tổng 0,099 so với 0,123 |
| 21/10 | Ngưỡng dừng sau task 1 khi B2 M2 = 85,79% (không phải ~73% như lúc chốt D17) | Giữ Δ: 🟢 ≥ 72,8%, 🟡 57,8–72,8%, 🔴 < 57,8%; FSA kiểm tra sớm < 52,8% | Hai bên đo cùng cách; giữ 60% nghĩa là chấp nhận kém B2 tới 26 điểm |
| 21/10 | Bố cục bản đầy đủ, vòng 2 (người dùng thấy ảnh ghép xếp chồng nhiều) | **o15**: mỗi vật còn thấy ≥ 90%, xếp gọn; 15% cảnh (và giỏ quá đầy) cho phép còn thấy ≥ 75%. Thay quyết định A | Đo che khuất bằng SAM trên 450 ảnh val (đã kiểm định trên ảnh ghép): vật có vết khuyết rõ 2,8/4,5/5,2%; bản A 15/17/19%; c90 0,4/1,2/1,5%; o15 2,7/5,5/4,4%. Bản A đã sinh được giữ ở thư mục superseded |
| 22/10 | 3 cảnh hard (1003694, 1005824, 5008992) không xếp được sau 16 lượt | **Chạy lại 16 lượt ở mức 75% khi cả 16 lượt đều hỏng** (người dùng chọn, thay vì bỏ 3 cảnh). 71.639 ảnh đã sinh không đổi, giữ đúng số ảnh đã chốt | Thử riêng: ở mức 75% cả 3 cảnh xếp được sau 0–5 lượt; cấu hình cũ chỉ dành 4 lượt cuối cho mức 75% |
| 22/10 | Xoá bản A cũ `rpc_synth_v2_layoutA_superseded` (23 GB) | Xoá (người dùng đồng ý) | o15 đã thay; manifest/plan receipt/verify report giữ ở `runs/data_preprocessing/synthetic_v2/layoutA_record/` |
| 22/10 | Gói train FULL-SYN: trần epoch FSA / dự đoán test ở vòng cổng / máy train | **8 epoch chọn theo AP50 val; bỏ test ở vòng cổng; 4090 vast.ai như B2** (người dùng chọn theo đề xuất) | `docs/project/full-syn-train-plan-2026-10-22.md` |
| 25/10 | Hình thức chạy gói train FULL-SYN | **Code F0.1–F0.8 + notebook training mới + cập nhật tài liệu + gói dữ liệu đẩy lên Drive trên nhánh này; người dùng tự ssh vào vast để setup dữ liệu, code, môi trường rồi chạy.** Notebook là log chạy, đi song song với log của code dưới `data/` (gitignore) | Ý của người dùng; plan mục 4.1 |
| 25/10 | Số notebook, ngưỡng dừng nhãn giả, gói Drive | **2 notebook: smoke `04` + `05_full_syn_r<N>` (mỗi vòng một file mới để đối chứng); dừng khi precision nhãn giả task 2 < 0,5; một tar đẩy lên `drive:AutoCheckout/2026-10-25` (stream thẳng chỉ khoảng 2 MB/s nên dừng; đóng gói ở local, người dùng tự upload); người dùng tự push** | Cổng là ngưỡng số nên chạy liền một mạch, không để GPU thuê ngồi chờ; plan mục 4.1 |

## 5. Câu hỏi đang chờ

- (Cho gói train) `pdp/main.py` đọc `train_task_<t>.json` và `val_task_<t>.json` từ cùng `--task_ann_dir`, nhưng train ảnh ghép nằm ở `rpc_synth_v2_v1/tasks/synth_v2/` còn val ở `rpc_eval_official_v1/`. Đề xuất: runner tạo một thư mục task gồm symlink tới hai nguồn (như test T4.3 đang làm), `--train_img_dir` = release root, `--test_img_dir` = `data/archive`, `--pred_ann_dir` = `rpc_eval_official_v1`. Chốt khi lên plan gói train

## 6. Nhật ký thực hiện

| Ngày | Việc đã làm | Bằng chứng |
|---|---|---|
| 21/10 | Token Kaggle hoạt động; tải từng file qua API (dấu `/` ghi thành `%2F`); ảnh test2019 tải về có MD5 trùng bản local | — |
| 21/10 | T0.1: commit doc quyết định | `6d19507` |
| 21/10 | T1.1: tải từng file bị Kaggle chặn (HTTP 429) sau khoảng 50 file; chuyển sang tải zip đầy đủ 25,3 GB vào `data/cache/kaggle/`, sẽ chỉ giải nén train2019 camera 0. Đã xóa công cụ tải từng file | — |
| 21/10 | T0.3: tạo env `sam` lần đầu lỗi vì kênh `defaults` chưa chấp nhận điều khoản; chạy lại bằng conda-forge | `runs/data_preprocessing/sam_env/create.log` |
| 21/10 | T0.2: `formats.md` mục 7. Bộ đánh giá dùng tên file như release cũ (`val_task_<t>`, `val_full`, `test_full`) để runner dùng được | — |
| 21/10 | T1.1: zip 27,2 GB tải xong, không lỗi; giải nén `retail_product_checkout/train2019/*_camera0-*`. Zip có hai bản train2019 (gốc và trong `retail_product_checkout/`), dùng bản sau | `train2019_extract/manifest_cam0.json` |
| 21/10 | T1.7 (đo trước): chai/lon đứng < 0,5% trong val. Trong 17.719 vật nhóm trụ, bbox nhỏ nhất chỉ bằng 0,48 lần p90 của SKU; 40 bbox nhỏ nhất có khoảng 5 vật đứng. Hộp thịt chữ nhật (`canned_food`) nằm úp mặt, nên cần chọn camera theo SKU | — |
| 21/10 | T1.4 đạt: SAM chạy 0,19 s/ảnh trên 3060 | `sam_equivalence/` |
| 21/10 | T1.5 mở rộng từ khoảng 51 lên **cả 200 SKU** (32 ảnh camera 0/SKU, dự kiến còn khoảng 20 sau khi lọc). Lý do: chỉ khoảng 20 phút GPU; tờ duyệt có thể chỉ ra hộp cứng cũng cần góc ngang; cutout không dùng thì bỏ qua | `runs/data_preprocessing/cutouts_cam0_v1/extract.log` |
| 21/10 | T1.8: lệnh `review-sheets` (mỗi SKU một hàng: crop val, camera 2, camera 0) và hàm `merge_catalogs` gộp catalog bỏ trùng nguồn. Trang chạy thử: chai trong val nằm ngang, khớp camera 0, không khớp camera 2 (nắp). SKU 80 có 3 cutout camera 2 thủng mất phần nắp, lọc tự động không bắt được, cần loại ở T1.6 | — |
| 21/10 | T1.6 xong phần mình duyệt: bộ cũ + camera 0 loại 404/4.958 (8,1%: 248 camera 2, 156 camera 0). Thêm quy tắc: hộp đứng chụp camera 0 mà thấy rõ mặt thứ hai (≥ khoảng 30% bề rộng) thì loại | `rejects_manual.json` |
| 21/10 | Cắt bổ sung camera 2 cho 12 SKU thiếu (640 ảnh nguồn từ zip → 364 đạt, 243 không trùng nguồn) và camera 0 cho SKU 121 (2/40 đạt). Duyệt 243 cutout mới: loại 146, chủ yếu dính mâm ở gói trong suốt/trắng | `rejects_supp.json`, `cutouts_cam2_supp_v1/` |
| 21/10 | Sau duyệt: mọi SKU có ≥ 8 cutout sạch ở góc chính. Tự quyết hai điểm nhỏ: SKU 82/160/166/168 không còn cutout nắp sạch → bỏ phần 1% tư thế đứng (chỉ nằm ngang); SKU 121 (lon nước cốt dừa) chỉ 2 cutout camera 0 → đổi tỉ lệ 70% nắp / 30% thân | `catalog/skus.json` |
| 21/10 | Bù cutout (người dùng chọn): 28 SKU dưới 15 cutout sạch → cắt hết ảnh còn lại trong zip (1.920 ảnh camera 2 + toàn bộ camera 0 của SKU 82/198) → 1.047 cutout mới không trùng, giữ 913. Sau bù: ít nhất 11, 10% thấp nhất 17, trung vị 20. Còn 3 SKU dưới 15 vì đã dùng hết nguồn: 121 (11, SAM hay hỏng), 68 (14, dính mâm), 198 (14) | `rejects_supp2.json` |
| 21/10 | Người dùng thấy có cutout bị loại trông bình thường → duyệt lại cả 684 cutout bị loại ở cỡ ảnh lớn (250px), gán lý do từng cái: dính mâm 215 / mảnh mâm nhỏ 185 / mask vỡ 35 / hộp hai mặt 105 / **loại nhầm 144** (khôi phục) | `m0_review/rereview/labels.json` |
| 21/10 | Sửa 435 cutout lỗi mask bằng cách cắt lại đúng ảnh gốc với **SAM 2.1 large** (cùng crop/hậu xử lý/ngưỡng QC; trọng số SHA-256 `ab7e1ac9…`): 377 qua QC, mình duyệt cạnh nhau cũ/mới → nhận 195. SAM-L sửa rất tốt nắp chai/vành lon bị thủng; với gói trong suốt/trắng chỉ sửa được khoảng một nửa | `cutouts_repair_v1/`, `catalog/cutouts_repl_v1.json` |
| 21/10 | Kết quả: lấy lại 339/684 (144 khôi phục + 195 sửa). Còn loại 345: dính mâm 125, mảnh mâm 94, mask vỡ 21, hộp hai mặt 105 (loại vì tư thế, mask vẫn sạch). Mỗi SKU vẫn ≥ 11 cutout sạch, trung vị 20. Lưu ý cho bộ sinh: chọn cutout theo `reviews.json` (keep), vì cùng ảnh gốc có thể có cả bản cũ bị loại và bản SAM-L được giữ | — |
| 21/10 | **M0 chốt** (người dùng đồng ý): `catalog/cutouts.json` + `m0_review.json`; notebook `05_synthetic_v2` chạy lại các bước CPU (run `57ab8898`, PASS_M0) | `runs/data_preprocessing/synthetic_v2/57ab8898/summary.json` |
| 21/10 | Phát hiện: 48% số vật trong ảnh val là bản lặp của một SKU đã có trong ảnh (khách mua nhiều đơn vị). Bộ sinh lấy **kiểu giỏ hàng** (số đơn vị mỗi SKU) từ ảnh val cùng level, thay vì chọn từng vật độc lập. Đo trên val: tâm vật trung bình (0,505; 0,552), độ lệch chuẩn 0,157/0,187/0,207 theo level; độ sáng vật so với cutout: camera 2 khoảng 0,85, camera 0 khoảng 0,95; JPEG val khoảng q75 | `configs/data/synth_v2.json` |
| 21/10 | T2.1 xong: `plan_scenes` (kiểu giỏ, 40–60% vật mới kẹp [1, n−1], SKU cân bằng, SKU cũ chia đều, cutout không lặp trong ảnh, seed ổn định), 11 test. T2.2 đang làm: `autocheckout/compositing.py` (xoay không lật, đặt vật giữ ≥ 70% phần thấy, bóng, màu/sáng, nhãn từ owner mask, RLE nén) + `fit_scales`, 7 test. **Chưa làm:** chạy `fit_scales` trên dữ liệu thật, writer/export/CLI (T2.3), pilot | — |
| 21/10 | T2.2–T2.3 xong: `fit_scales` (kích thước theo SKU–camera khớp val), renderer, writer tiếp tục được khi bị ngắt, `generate`/`export`/`verify`/`gallery`. Sửa trong lúc chạy: SKU 121 bị tính nhầm tư thế đứng; cảnh chật thì nới độ rải, rồi đặt đều toàn khay, cuối cùng đặt vật to trước; giới hạn tổng diện tích giỏ (p99 của val) khi lập kế hoạch | `configs/data/synth_v2.json` |
| 21/10 | T3.1–T3.2: pilot `rpc_synth_v2_pilot_v1` 1.000/1.000 ảnh, 83 s với 16 worker (bản đầy đủ ước khoảng 1,7 giờ); **14/14 kiểm tra tự động đạt** (bắt đúng 5 loại lỗi cố ý cài trong test); tái lập 0/180 file khác giữa 1 và 16 worker. Hiệu chỉnh 1 vòng theo val: kích thước ×1,048, độ rải ×0,9. So với val: số vật/ảnh 7,14/12,31/17,29 vs 7,15/12,38/17,29; độ phủ bbox p50 0,29/0,41/0,47 vs 0,32/0,40/0,46; diện tích bbox theo SKU synth/val trung vị 0,98. Ảnh 1840px (cỡ nền), không phải 1850 như D10 | `rpc_synth_v2_pilot_v1/verify_report.json` |
| 21/10 | T3.3: gallery `runs/data_preprocessing/synthetic_v2/pilot_v1_gallery/` (5 trang có khung xanh = mới có nhãn, đỏ = cũ không nhãn; 3 trang đặt ảnh ghép trên, ảnh thật dưới). Nhận xét của mình: ảnh thật xếp vật sát nhau thành cụm/hàng, ảnh ghép rải hơn; vật ghép phẳng, không có độ dày | — |
| 21/10 | M2: thêm chế độ xếp gọn (`pack_mode=compact`, `pack_prob`, `aligned_prob`) và chỉ số `basket_density`; thử 12 cấu hình trên 400 ảnh. Không cấu hình nào khớp cả mật độ cụm lẫn độ phủ bbox. Đo thêm: vật có hàng xóm cách < 3% cạnh ảnh 94/97/98% (val 98/99/100%), IoU bbox lớn nhất p90 0,25/0,26/0,26 (val 0,23/0,25/0,26). **Người dùng chọn A**, M2 duyệt | `rpc_synth_v2_pilot_v1/pilot_review.json` |
| 21/10 | T4.1 bắt đầu: kế hoạch 71.642 cảnh (vật mới ở task 2–5 chiếm 50,0–50,1%), sinh với 16 worker khoảng 11 ảnh/s (ước 1 giờ 50 phút) | `rpc_synth_v2_v1/plans/` |
| 21/10 | T4.2 xong: lệnh `eval-set` → `rpc_eval_official_v1`: val_full 5.792 ảnh / 71.108 vật, val_task_1..5 = 5.562/3.291/3.100/3.053/3.412 ảnh, test_full 24.000 ảnh / 294.333 vật, nhãn 0..199. Tập so B1/B2 (6.003 ảnh) trùng id, file, box với test mới (chỉ khác trường `group` của split cũ) nên lọc dự đoán theo id là so được | `rpc_eval_official_v1/manifest.json` |
| 21/10 | T4.3 xong: `tests/test_synthetic_loader.py` (4 test PASS trên pilot + bộ đánh giá): ảnh 640, box khớp bbox gốc < 1px, chỉ nhãn task hiện tại | — |
| 21/10 | P1: `drop_unlearned` trong `counting.py` + `eval_count --unlearned-iou 0.5` (policy ghi tham số, file kết quả hậu tố `_m2`), 3 test. P2: tính lại B2 task 1 trên val cũ (cùng NMS 0,45, lưới 0,10–0,90): M1 = 73,19% (khớp số cũ), **M2 = 85,79%** trên 1.464 ảnh, ngưỡng 0,27 | `runs/data_preprocessing/synthetic_v2/b2_task1_cacc_m2.json` |
| 21/10 | **T4.1 xong:** 71.642/71.642 ảnh, 0 lỗi, 5.817 s (12,3 ảnh/s); export + verify (RLE mẫu 2.000) **14/14 đạt**, 23 GB. So val: số vật/ảnh 7,11/12,35/17,25 vs 7,15/12,38/17,29; phủ bbox 0,32/0,41/0,48 vs 0,32/0,40/0,46; diện tích SKU trung vị 0,993. T4.4: báo cáo `docs/data_preprocessing/reports/synth-v2-2026-10-21.md`, cập nhật PROGRESS/chỉ mục, formats.md 7.2 theo cấu trúc thật (1840px, `plans/`, `labels/`) | `rpc_synth_v2_v1/verify_report.json` |
| 21/10 | Kiểm tra lại chất lượng (người dùng thấy nhiều vật xếp chồng): bản A che gấp ~4 lần ảnh thật vì hiệu chỉnh theo độ phủ bbox (ảnh thật phủ bbox do xếp khít, không do đè). Val và test giống nhau trên mọi chỉ số bố cục. Thêm `occluded_scene_prob`/`occluded_min_visible` + nới khi giỏ quá đầy; sửa `generate` báo lỗi khi worker bị kill (trước đó treo khi hết RAM). Pilot v2 (o15) 1.000/1.000, verify 14/14 | `rpc_synth_v2_pilot_v2/pilot_review.json` |
| 22/10 | Sinh o15 bản đầy đủ (12 worker, 9,1 ảnh/s): 71.639/71.642, 3 cảnh hard lỗi. Job thoát mã 1 vì một tiến trình root `auto_poweroff` (tạo 01:16, tắt máy 01:36) đổi chủ `runs/.../full_v2/` thành root, nên không ghi được biên nhận; ảnh không ảnh hưởng. Thêm lượt chạy lại ở mức 75% + 1 test; sinh nốt 3 cảnh → 71.642/71.642, 0 lỗi | `runs/data_preprocessing/synthetic_v2/full_v2/auto_poweroff/generate.log`, `full_v2_finish/` |
| 22/10 | **T4.1 xong (o15):** export + verify (RLE mẫu 2.000) **14/14 đạt**, 23 GB, manifest `ad6d899e…`. So val: số vật/ảnh 7,11/12,35/17,25 vs 7,15/12,38/17,29; phủ bbox 0,27/0,35/0,41 vs 0,32/0,40/0,46; mật độ cụm 0,77/0,79/0,87 vs 0,82/0,90/0,98; diện tích SKU trung vị 1,019. Vật bị che > 10% (nhãn thật) 5,8/6,8/9,8%. Báo cáo cập nhật theo o15 | `rpc_synth_v2_v1/verify_report.json` |
| 22/10 | Rà lại dữ liệu sau khi xoá bản A (script độc lập): hash mọi file nhãn + 2.000 ảnh mẫu khớp manifest; id không trùng; nhãn train task t đúng và đủ lớp của task t; bbox trong ảnh; 208 nền không có trong val/test; task_config hai gói giống nhau; 34 test đạt; xem 9 ảnh ngẫu nhiên có box. Viết plan gói train FULL-SYN | `docs/project/full-syn-train-plan-2026-10-22.md` |
| 21/10 | Thử lọc tự động: độ đặc của mask bắt mask thủng/vỡ (SKU 80/163 chỉ 0,50–0,62; có 270 cutout < 0,85); chỉ số pixel trắng sát mép không bắt được dính mâm xoay (42491/42499 ra 0) | — |
| 21/10 | T1.2 + T1.3: `autocheckout/synthetic.py`, `tools/synth_v2.py` (`inventory`, `stats`), 7 test PASS. Bbox trong ảnh thật chồng nhiều: phần bbox bị bbox khác phủ, trung vị 0,32 (easy) / 0,40 (medium) / 0,46 (hard). Đây là đo trên bbox, không phải che khuất thật; dùng để so pilot với val theo cùng cách đo. Màu khay ngoài vật: RGB trung vị khoảng 143 | `design/design_statistics.json` |
