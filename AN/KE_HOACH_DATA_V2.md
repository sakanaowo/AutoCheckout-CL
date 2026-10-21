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
| T1.6 | Duyệt cutout bằng mắt, loại thủ công lỗi (dính mâm xoay, mask vỡ, hộp thấy hai mặt) | `catalog/reviews.json`, mỗi cutout có lý do; trang tổng hợp loại theo lý do `runs/data_preprocessing/m0_review/rejected_<lý do>_*.jpg` | Duyệt 2 vòng + sửa bằng SAM-L; **chờ người dùng xác nhận trang tổng hợp** | [~] |
| T1.7 | Gán **nhóm hình dạng** cho 200 SKU (dẹt/trụ/hộp cứng) và đo tỉ lệ chai/lon đứng–nằm trên mẫu val2019 | `catalog/skus.json`: 123 flat (camera 2) / 64 upright (camera 0 + 1% camera 2) / 13 mixed (50/50) | Mình xem cả 20 trang tờ duyệt; người dùng chấp nhận 21/10; không SKU nào thiếu cutout | [x] |
| T1.8 | Gallery theo SKU/camera/nhóm và **họp duyệt M0** | Gallery + `m0_review.json` | Có người duyệt, ngày, quyết định | [!] chờ xác nhận |

### Giai đoạn 2 — Bộ sinh ảnh (cổng M1)

| ID | Task | Đầu ra | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| T2.1 | Lập kế hoạch cảnh: số vật theo level, tỉ lệ vật mới 40–60% (kẹp `[1, n−1]`), SKU cũ chọn đều, quota cân bằng, seed ổn định | `plan_scenes` + recipe | Test: task 1 chỉ SKU task 1; không SKU tương lai; quota ±20%; seed không phụ thuộc số worker | [ ] |
| T2.2 | Renderer: xoay/scale theo val, chồng lấn, mỗi vật còn thấy ≥ 70%, bóng, màu/sáng; bbox/RLE lấy từ mask cuối | `render_scene` | Test: không lật ảnh, bbox khớp mask, vật nhỏ không mất | [ ] |
| T2.3 | Ghi file, xuất nhãn theo `generation_task`, manifest, chạy tiếp được nếu bị ngắt | `generate_dataset`, CLI | Test: ảnh task 2 không lọt vào task 1; ID lớp chỉ ánh xạ một lần; chạy song song ra đúng từng byte như chạy tuần tự | [ ] |

### Giai đoạn 3 — Pilot (cổng M2)

| ID | Task | Đầu ra | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| T3.1 | Công cụ kiểm tra dữ liệu (`verify_dataset`) | Báo cáo kiểm tra tự động | Test cố ý cài lỗi: phải FAIL đúng chỗ | [ ] |
| T3.2 | Sinh pilot 200 ảnh/task, đo s/ảnh, MB/ảnh, tỉ lệ đặt vật thất bại | `rpc_synth_v2_pilot_v1` | `verify` đạt | [ ] |
| T3.3 | Gallery 100 ảnh (5 task × 3 level) + so phân bố với val2019, **họp duyệt M2** | `pilot_review.json` | Có người duyệt | [ ] |

### Giai đoạn 4 — Bản đầy đủ (cổng M4)

| ID | Task | Đầu ra | Kiểm tra | Trạng thái |
|---|---|---|---|---|
| T4.1 | Sinh đủ 71.642 ảnh `[21752, 12534, 12129, 11850, 13377]` | Release v2 + manifest | Đủ số ảnh, kiểm tra tự động đạt, không trùng | [ ] |
| T4.2 | Bộ đánh giá từ RPC gốc: val2019 đã lọc, test2019 đầy đủ, theo task, giữ `level`; thêm tập 6.003 ảnh cũ để so B1/B2 | `eval/` + checksum | Test: 5.792 / 24.000 ảnh, nhãn trong phạm vi các lớp đã học | [ ] |
| T4.3 | Smoke test loader 640: tọa độ box, nhãn, thư mục ảnh, không nhãn cũ trong train | Test loader | PASS | [ ] |
| T4.4 | Báo cáo `synth-v2-<ngày>.md`, cập nhật PROGRESS/chỉ mục | Báo cáo | Link hợp lệ | [ ] |

### Song song (phục vụ cổng dừng của gói train, chạy CPU)

| ID | Task | Trạng thái |
|---|---|---|
| P1 | Code cAcc-M2 trong `counting.py`/`eval_count.py` + test | [ ] |
| P2 | Tính lại cAcc-M2 task 1 của B2 từ `pred_val.npz` → mốc B | [ ] |

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

## 5. Câu hỏi đang chờ

(Trống)

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
| 21/10 | Thử lọc tự động: độ đặc của mask bắt mask thủng/vỡ (SKU 80/163 chỉ 0,50–0,62; có 270 cutout < 0,85); chỉ số pixel trắng sát mép không bắt được dính mâm xoay (42491/42499 ra 0) | — |
| 21/10 | T1.2 + T1.3: `autocheckout/synthetic.py`, `tools/synth_v2.py` (`inventory`, `stats`), 7 test PASS. Bbox trong ảnh thật chồng nhiều: phần bbox bị bbox khác phủ, trung vị 0,32 (easy) / 0,40 (medium) / 0,46 (hard). Đây là đo trên bbox, không phải che khuất thật; dùng để so pilot với val theo cùng cách đo. Màu khay ngoài vật: RGB trung vị khoảng 143 | `design/design_statistics.json` |
