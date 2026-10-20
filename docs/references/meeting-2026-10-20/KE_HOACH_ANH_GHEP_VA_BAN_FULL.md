# Kế hoạch: bản chạy full option và bộ ảnh ghép để train

Ngày 20/10/2026. Đối tượng đọc: cả nhóm.

## 0. Các quyết định đã chốt

| # | Quyết định |
|---|---|
| 1 | **Tập train dùng toàn bộ ảnh ghép.** Val và test giữ nguyên ảnh thật, không đổi. |
| 2 | **Task 1 có nhãn đầy đủ:** ảnh ghép task 1 chỉ chứa 100 SKU của task 1, nên mọi vật trong ảnh đều có nhãn. |
| 3 | **Task 2–5:** ảnh có SKU mới (gắn nhãn) xen lẫn SKU cũ (**không gắn nhãn**). SKU cũ được model thầy gán nhãn giả, giống tình huống ở quầy thật. Ảnh **không bao giờ** chứa SKU của task sau. |
| 4 | Chạy luôn **một bản full option**, gồm mọi sửa lỗi và cải tiến đã có bằng chứng. |
| 5 | Sau mỗi task phải ghi log chi tiết (mục 2). |

Lý do của quyết định 2: ảnh thật của task 1 hiện chỉ có nhãn cho khoảng **50% số vật** (138.280/276.091). Nửa còn lại là SKU của task sau nhưng bị coi là nền. Model học rằng "vật này là nền", đến task sau lại phải học ngược lại.

## 1. Bản chạy full option: FULL-SYN

### 1.1 Những gì được bật

| Thành phần | Giá trị | Lý do |
|---|---|---|
| Backbone | ConvNeXt-V2-Base (IN-22k) | Hướng nhóm đã chọn |
| **FSA**: task 1 fine-tune toàn bộ model trên 100 SKU | `FINETUNE_ARGS` + `--save_hf 1`, rồi chạy PDP với `--repo_name <FSA>/task_1/hf_model` | Sửa việc encoder/decoder được train cùng ResNet nhưng ghép với ConvNeXt (B2). Tháng 9, PDP đạt val mAP50 0,83 khi bắt đầu từ FSA, so với 0,11 khi bắt đầu từ COCO bị khóa. |
| F1–F13 | Bật như B1 và B2 | Sửa lỗi code gốc, đã kiểm tra |
| **F14**: mỗi vật chỉ một nhãn giả | `--pseudo_dedup_iou 0.5` | Khi tắt, cAcc tụt về 0,10 sau task 5 (E4_noF14) |
| **I3**: bỏ nhãn giả "cũ" đè lên vật mới đã có nhãn | `--pseudo_gt_iou 0.5` | Cột "Mới": 0,683 khi tắt, 0,925 khi bật |
| **NMS không phân biệt lớp** khi đếm | IoU 0,45 (đã có trong `counting.dedup_detections`) | Một vật không bị đếm 2 lần, ví dụ vừa là SKU cũ vừa là SKU mới |
| **Ngưỡng đếm riêng cho từng nhóm task**, chọn trên **val** | **Cần viết thêm code**: hiện `select_threshold` chỉ chọn 1 ngưỡng chung | SKU mới thường có điểm tin cậy thấp hơn. Ngưỡng do val quyết định, không cố định "mới thấp, cũ cao". |
| Ngưỡng nhãn giả của model thầy | τ_h 0,5 / τ_l 0,2 (mặc định của paper) | Giữ nguyên, không tinh chỉnh |
| Huấn luyện | 640px, batch 8, 16 epoch/task, seed 0, TF32 | Giống B2 để so sánh được |
| Dữ liệu | Ảnh ghép v2 (mục 3–5); số ảnh mỗi task **bằng số ảnh thật B2 đã dùng** (xem trong log B2) | So sánh công bằng với B2 |

### 1.2 Những gì không bật (chưa có bằng chứng)
I2 `--augment` (biến đổi màu đã làm trong bộ sinh ảnh, bật thêm sẽ thành biến đổi hai lần), I4 `--prototype_nearest`, I5 `--freeze_shared_after_task1`, LoRA, K=3 prototype. Mỗi thứ này chỉ thêm vào sau, **từng thứ một**.

### 1.3 Điểm dừng ngay trong lần chạy (không tốn thêm GPU)
Sau task 1 (FSA + PDP), đánh giá trên **val thật** rồi so với task 1 của B2:
- Nếu kém nhiều (ví dụ cAcc thấp hơn trên 15 điểm), **dừng**. Cải thiện ảnh ghép rồi chạy lại task 1, không chạy tiếp task 2–5.
- Lý do: FSA trên ảnh ghép làm backbone quen với ảnh ghép. Từ task 2 trở đi backbone bị khóa, nên chênh lệch giữa ảnh ghép và ảnh thật sẽ đi theo đến cuối.

### 1.4 Bản đối chứng (nên có)
**FULL-REAL** dùng cấu hình giống hệt FULL-SYN nhưng train trên ảnh thật. Nếu không có bản này, không tách được phần cải thiện do phương pháp (so với B2) và phần mất mát do ảnh ghép (FULL-SYN so với FULL-REAL).

### 1.5 Việc code cần làm trước khi chạy
1. Bộ sinh ảnh ghép v2 (mục 5).
2. Chọn ngưỡng đếm theo từng nhóm task trên val, rồi áp dụng các ngưỡng đó cho test.
3. Script cấu hình `configs/exp/native/EXP-FULL-SYN.sh`: FSA cho ConvNeXt, rồi PDP với các cờ ở mục 1.1.
4. Ghi log theo mục 2.
5. Thống kê nhãn giả so với nhãn thật bị ẩn: làm được vì ảnh ghép biết hết vật cũ nằm ở đâu.

## 2. Log bắt buộc sau mỗi task

Mỗi task ghi `task_<t>/task_log.json` (máy đọc) và `task_<t>/SUMMARY.md` (người đọc). Một phần đã có sẵn: `run_info.json`, `train.log` (ghi args), CSVLogger.

| Nhóm | Nội dung |
|---|---|
| **Thông tin chạy** | run id; git commit và dirty diff; câu lệnh đầy đủ; **toàn bộ args**; GPU, driver, phiên bản torch/transformers; giờ bắt đầu và kết thúc; số giờ GPU; chi phí ước tính |
| **Dữ liệu** | task id; danh sách SKU mới và SKU đã học; đường dẫn và md5 file annotation; số ảnh, số vật có nhãn, số vật cũ không nhãn; **phiên bản, config hash và seed của bộ sinh ảnh** |
| **Model** | backbone; checkpoint khởi tạo (đường dẫn, md5); số tham số được train và bị khóa **theo từng module**; learning rate theo từng nhóm; danh sách cờ F/I đang bật |
| **Quá trình train (theo epoch)** | từng thành phần loss (cls, bbox, giou, L_Q, L_DDL); lr; độ lớn gradient; thời gian mỗi epoch; số ảnh/giây; bộ nhớ GPU tối đa |
| **Nhãn giả (task ≥ 2)** | số nhãn giả trên mỗi ảnh; số bị I3 loại; số bị F14 loại; phân bố điểm tin cậy; **precision/recall so với vật cũ thật** (từ file GT đầy đủ của ảnh ghép) |
| **Đánh giá (val và test)** | AP50 cho lớp Cũ/Mới/Tất cả và theo từng nhóm task; cAcc, ACD, mCCD, mCIoU, mCCS **tổng và theo easy/medium/hard**; **ngưỡng đã chọn cho từng nhóm** và NMS; một dòng của ma trận độ chính xác; forgetting; AP từng SKU (CSV); **top-20 cặp nhầm lẫn** (SKU mới bị dự đoán thành SKU cũ nào) |
| **Sản phẩm đầu ra** | đường dẫn và md5 của checkpoint, hf_model, file dự đoán val/test |

## 3. Ảnh ghép: hiện trạng tài nguyên

| Tài nguyên | Hiện trạng | Vấn đề |
|---|---|---|
| **7.502 cutout** (SAM, RGBA) | Đủ 200 SKU, **10–48 cutout/SKU**, alpha nhị phân, không rỗng | Cutout lấy từ ảnh đơn chụp bằng **4 camera: từ trên, 45°, 30° và ngang**. Ảnh quầy chụp từ trên xuống. Góc ngang dùng được nếu tương ứng một tư thế nằm ổn định trên khay; góc **30° và 45°** cho thấy 2–3 mặt và bị méo phối cảnh, không bao giờ có trên quầy. Chưa có ai xem bằng mắt toàn bộ; đã tự động loại 2.098 cutout. Mép do SAM tạo có thể dính khay hoặc cắt mất bao bì. |
| **208 nền** | Khay 1840×1840 làm từ ảnh val2019 (thuộc tập train), đã kiểm tra không trùng val/test (`background_sources_train_only: PASS`) | Một số nền còn mảng màu hoặc đốm mờ sau inpainting, cần kiểm tra bằng mắt |
| 20.000 ảnh ghép cũ | Chia 4 task | **Không dùng lại.** Sai số task, chỉ che khuất tối đa 15%, **lật ngang 50% làm chữ và logo bị ngược**, 222 bbox lệch với mask (lệch tối đa 137px), không lưu nền, seed hay phép biến đổi |
| Ảnh thật để tham chiếu | Train trung bình khoảng 12,3 vật/ảnh, có trường `level` (easy/medium/hard) | Chỉ được lấy **thống kê từ tập train thật**. **Không** dùng val/test để tinh chỉnh bộ sinh ảnh. |

Tham khảo IncreACO (phần PEA):
- **Chọn tư thế:** bỏ những tư thế không thể xuất hiện trên quầy.
- **Đặt vật:** đặt sát cạnh vật trước, ít che nhau, phóng to/thu nhỏ 0,5–0,7.
- **Kiểm tra che khuất theo dạng vật:** vật dẹt dễ bị che, chai lọ khó bị che.
- **Đưa vào ảnh của task mới:** sản phẩm cũ với α = 0,05, mỗi ảnh có ít nhất một sản phẩm mới.
- **Làm ảnh giống thật hơn bằng CycleGAN.** Đây là bước làm cAcc tăng từ khoảng 26% lên khoảng 75%.

## 4. Yêu cầu bộ ảnh ghép v2

### A. Chọn cutout
1. **Xác định góc chụp từng cutout.** Ưu tiên lấy từ metadata camera của RPC train2019 (nối từ ảnh gốc của cutout). Nếu không có, tự phân loại bằng độ vuông vức của mask (diện tích mask/hình chữ nhật bao nhỏ nhất) và "độ ổn định" của IncreACO, rồi xem bằng mắt.
   - **Từ trên:** giữ, ưu tiên cao.
   - **Ngang:** giữ nếu là một tư thế nằm ổn định (mặt lớn, gần chữ nhật). Bỏ các mặt quá hẹp như đáy chai hoặc cạnh hộp.
   - **30° và 45°:** bỏ, hoặc chỉ giữ dưới 10%.
   - **Kiểm tra theo từng SKU:** so phân bố tỉ lệ cạnh và diện tích bbox của ảnh ghép với ảnh train thật. SKU nào lệch nhiều thì xem lại cutout.
   - (Tùy nhóm quyết) chọn cutout theo độ giống (DINOv2/CLIP) với khoảng 20 vật cắt từ ảnh train thật mỗi SKU. Nếu làm, phải ghi rõ là có dùng một ít ảnh thật.
   - Có thể thêm biến đổi phối cảnh nhẹ (±5°) để mô phỏng vật ở mép ảnh.
2. Bỏ cutout có mask lỗi (dính khay, mất góc, lẫn hai vật). Xem bằng mắt **ít nhất 5 cutout/SKU** và ghi danh sách loại bỏ.
3. Sau khi lọc, mỗi SKU còn **≥ 5 cutout**. Nếu ít hơn thì báo cáo, không âm thầm bỏ qua.
4. Làm mịn mép: thu mask vào 1–2px rồi làm mờ mép (feather) để không có viền sáng.

### B. Bố cục
5. Số vật mỗi ảnh lấy **theo phân bố easy/medium/hard của ảnh train thật** (đo từ annotation train). Tỉ lệ ba mức giống tập train.
6. **Kích thước:** một hệ số tỉ lệ chung, chỉnh sao cho phân bố diện tích bbox của ảnh ghép khớp với ảnh train thật, cộng dao động khoảng ±10%. Cutout cùng camera đã giữ đúng tỉ lệ kích thước giữa các SKU.
7. **Xoay** tự do 0–360°. **Không lật** vì chữ và logo sẽ bị ngược.
8. **Đặt vật:** sát cạnh vật đã có (như IncreACO), không chồng lên mép khay. Mức che khuất **đo từ ảnh train thật** (độ chồng bbox), không đoán. Mỗi vật phải còn **nhìn thấy ≥ 70%** (có thể chỉnh); nếu không đạt thì đặt lại.

### C. Độ chân thực
9. Thêm bóng đổ mềm, cùng một hướng sáng trong mỗi ảnh.
10. Mỗi vật: dao động nhẹ độ sáng, tương phản và nhiệt màu, hoặc khớp màu theo nền. Mỗi ảnh: dao động sáng, mờ nhẹ, nhiễu, nén JPEG chất lượng khoảng 85–95.
11. Kích thước ảnh ra giống nền (1840) hoặc ảnh thật (1830). Loader sẽ tự đổi về 640.
12. (Giai đoạn 2, nếu cần) CycleGAN chuyển ảnh ghép sang phong cách ảnh thật, train trên **ảnh train thật** và ảnh ghép.

### D. Nhãn
13. **Tính bbox từ mask phần nhìn thấy cuối cùng**, sau khi mọi vật đã đặt xong. Đây là lỗi của bộ cũ.
14. Lưu **mask (RLE) của phần nhìn thấy** và một ảnh "mask chủ" cho biết mỗi pixel thuộc vật nào.
15. Không có bbox rỗng hoặc tràn ra ngoài ảnh. Diện tích bbox và mask nhất quán (IoU ≥ 0,98).

### E. Chia task (quan trọng nhất)
16. **Task 1:** chỉ dùng 100 SKU của task 1, gắn nhãn tất cả.
17. **Task t ≥ 2:** mỗi ảnh có **≥ 1 SKU mới**. Vật mới chiếm khoảng 40–60% (có thể chỉnh); số còn lại là SKU **cũ** từ task 1 đến t−1. **Chỉ gắn nhãn SKU mới.** Tuyệt đối **không có SKU của task sau t**.
18. Mỗi task xuất **2 file**:
    - `train` (chỉ nhãn task hiện tại) cho loader;
    - `gt_full` (đủ nhãn mọi vật) chỉ dùng để kiểm tra và đo chất lượng nhãn giả, **không đưa vào train**.
19. Mã nhãn 0..199 theo `task_config.json` của bản phát hành `rpc_100-4x25_seed0_native_v1`.

### F. Số lượng và cân bằng
20. Số ảnh mỗi task bằng số ảnh thật B2 đã dùng.
21. Mỗi SKU **mới** của mỗi task có số vật gần bằng nhau, chênh lệch tối đa ±20%. Phân bố cutout trong SKU cũng đều nhau.

### G. Nghiệm thu (phải đạt trước khi train)
22. Kiểm tra tự động:
    - không có SKU tương lai;
    - không có nhãn SKU cũ trong file `train`;
    - bbox khớp mask;
    - số vật mỗi SKU đạt ngưỡng;
    - không trùng ảnh;
    - nền không thuộc val/test.
23. So thống kê ảnh ghép với ảnh train thật (bảng và biểu đồ):
    - số vật/ảnh;
    - diện tích và tỉ lệ cạnh của bbox;
    - độ chồng bbox;
    - phân bố độ sáng và màu.
    - Có thể thêm FID/KID.
24. Xem bằng mắt **100 ảnh ngẫu nhiên** đã vẽ bbox, gồm task 1 và task 3. Ghi lỗi gặp phải.
25. Chạy lại cùng seed và config phải ra **file giống hệt từng byte**.

### H. Truy vết
26. Mỗi ảnh lưu:
    - seed và nền dùng;
    - với mỗi vật: cutout id, SKU, góc xoay, tỉ lệ, vị trí, thứ tự lớp chồng, tỉ lệ còn nhìn thấy, tham số màu và bóng;
    - vật đó có nhãn hay không.
27. Lưu `manifest.json`: phiên bản code, config, checksum mọi file, thời gian chạy, kết quả nghiệm thu.

## 5. Prompt giao cho AI hoặc người làm phần ảnh ghép

```text
Bối cảnh
--------
Dự án AutoCheckout-CL (repo hiện tại, nhánh develop): phát hiện và đếm sản phẩm theo học tăng dần
(class-incremental, không replay) trên RPC. Giao thức 100+4×25 SKU, seed 0. Bản phát hành ảnh thật:
data/processed/rpc_100-4x25_seed0_native_v1/ (task_config.json, splits/, tasks/real/).
Val và test là ảnh thật, giữ nguyên, KHÔNG được dùng để sinh hoặc tinh chỉnh ảnh ghép.
Tài nguyên có sẵn: 7.502 cutout RGBA tạo bằng SAM (10–48 cutout/SKU, chụp bằng 4 camera: trên/45°/30°/ngang)
và 208 nền khay 1840×1840 (nguồn val2019, đã kiểm tra không trùng val/test). Đọc
docs/data_preprocessing/reports/preprocessing-acceptance-2026-10-07.md và split-100-4x25-2026-10-07.md
để biết vị trí file, index và các lỗi của bộ ảnh ghép cũ (lật 50%, che tối đa 15%, bbox lệch mask,
không lưu provenance). KHÔNG dùng lại 20.000 ảnh ghép cũ.

Nhiệm vụ
--------
Viết bộ sinh ảnh ghép v2 (tools/synth_v2.py + configs/data/synth_v2.yaml + tests) để tạo tập TRAIN
cho 5 task, cùng notebook hoặc báo cáo nghiệm thu. Mục tiêu: ảnh giống ảnh quầy thật nhất có thể.
Model sẽ train HOÀN TOÀN bằng ảnh ghép và đánh giá trên ảnh thật.

Yêu cầu bắt buộc
----------------
1. Chọn cutout: xác định góc chụp từng cutout, ưu tiên từ metadata camera RPC train2019 (nối với
   ảnh gốc). Nếu thiếu, phân loại bằng độ vuông vức của mask và độ ổn định của IncreACO =
   (diện tích mask / diện tích mask lớn nhất của SKU) × (độ đặc so với hình chữ nhật bao nhỏ nhất,
   chuẩn hóa theo SKU). Giữ góc từ trên. Giữ góc ngang nếu là tư thế nằm ổn định. Bỏ 30°/45°
   (hoặc giữ dưới 10%, tham số trong config). Kiểm tra theo từng SKU: so tỉ lệ cạnh và diện tích
   bbox với ảnh train thật, rồi báo cáo SKU lệch nhiều.
   Loại mask lỗi. Xuất danh sách loại bỏ có lý do, và một gallery ≥ 5 cutout/SKU để người xem.
   Nếu SKU nào còn < 5 cutout thì báo cáo. Thu mask vào 1–2px và làm mờ mép.
2. Bố cục: số vật/ảnh và tỉ lệ easy/medium/hard lấy từ annotation của TẬP TRAIN THẬT.
   Kích thước dùng một hệ số chung, chỉnh để phân bố diện tích bbox khớp ảnh train thật, cộng
   dao động ±10%. Xoay 0–360°. KHÔNG LẬT. Đặt vật sát cạnh vật trước như IncreACO (chọn điểm
   trên mép vật cũ, lấy mẫu Gauss quanh điểm đó). Mức che khuất khớp ảnh train thật. Mỗi vật còn
   nhìn thấy ≥ min_visible (mặc định 0,7); nếu không đạt thì đặt lại, tối đa N lần, rồi ghi log.
3. Độ chân thực: bóng đổ mềm cùng hướng sáng trong một ảnh. Dao động màu nhẹ cho từng vật.
   Dao động sáng, mờ, nhiễu và nén JPEG (85–95) cho cả ảnh. Kích thước ảnh ra giống nền.
   Mọi tham số nằm trong config.
4. Nhãn: bbox, diện tích và mask RLE tính từ mask PHẦN NHÌN THẤY cuối cùng. Lưu ảnh mask chủ
   (PNG, mỗi pixel ghi id vật). IoU giữa bbox và mask ≥ 0,98. Không có bbox rỗng hoặc tràn ảnh.
5. Chia task:
   - Task 1 chỉ dùng 100 SKU của task 1, gắn nhãn TẤT CẢ.
   - Task t≥2: mỗi ảnh có ≥ 1 SKU mới. Tỉ lệ vật mới nằm trong khoảng new_ratio (mặc định
     0,4–0,6); phần còn lại là SKU CŨ (task < t), KHÔNG gắn nhãn trong file train. TUYỆT ĐỐI
     không có SKU của task > t.
   - Mã nhãn 0..199 theo task_config.json.
   - Xuất COCO JSON cho mỗi task:
     tasks/synth_v2/train_task_<t>.json (chỉ nhãn task t, định dạng giống tasks/real để
     tools/make_task_json.py --train-source đọc được) và train_task_<t>_gt_full.json (đủ mọi
     vật, có cờ labeled true/false), chỉ dùng để kiểm tra.
6. Số lượng: số ảnh mỗi task là tham số (mặc định bằng số ảnh thật B2 đã dùng). Số vật của mỗi SKU
   mới chênh lệch tối đa ±20%. Cutout trong mỗi SKU được dùng đều nhau.
7. Truy vết và tái lập: mỗi ảnh lưu seed, nền, và với mỗi vật: cutout_id, sku, label_id, góc,
   tỉ lệ, x/y, thứ tự lớp, tỉ lệ nhìn thấy, tham số màu/bóng, labeled. manifest.json gồm git commit,
   hash của config, checksum mọi file, thời gian, kết quả nghiệm thu. Cùng seed và config phải ra
   file giống hệt từng byte. Chạy song song nhiều process vẫn phải tất định (seed riêng theo
   từng ảnh).

Nghiệm thu (báo cáo docs/data_preprocessing/reports/synth-v2-<ngày>.md + notebook)
--------------------------------------------------------------------------------
- Test tự động (pytest):
  - không có SKU tương lai;
  - file train không có nhãn SKU cũ;
  - bbox khớp mask;
  - không có bbox lỗi;
  - không lật;
  - tái lập từng byte;
  - nền không thuộc val/test;
  - mỗi SKU đạt số vật tối thiểu.
- So sánh thống kê ảnh ghép với ẢNH TRAIN THẬT (bảng và biểu đồ):
  - số vật/ảnh theo mức;
  - diện tích và tỉ lệ cạnh bbox;
  - độ chồng bbox;
  - histogram độ sáng và màu;
  - KID/FID trên 1.000 ảnh mỗi bên (nếu làm được).
- Gallery 100 ảnh ngẫu nhiên đã vẽ bbox (có ảnh của task 1 và task ≥ 2; vật không nhãn vẽ màu khác).
- Bảng tổng hợp mỗi task: số ảnh, số vật có nhãn, số vật không nhãn, số SKU, min/max số vật mỗi SKU.

Không được làm
--------------
- Không dùng val/test (ảnh hoặc thống kê) để sinh hay tinh chỉnh ảnh.
- Không gắn nhãn SKU cũ trong file train (sẽ thành replay).
- Không đưa SKU tương lai vào ảnh.
- Không sửa bản phát hành ảnh thật hoặc file trong tasks/real.
- Không train model trong nhiệm vụ này.

Bàn giao
--------
- Code, config và tests đạt.
- Lệnh chạy một dòng.
- Dữ liệu 5 task.
- Manifest và báo cáo nghiệm thu.
- Danh sách cutout bị loại.
- Các vấn đề còn mở.
Trước khi sinh toàn bộ, chạy thử 200 ảnh/task và gửi gallery để nhóm duyệt.
```

## 6. Thứ tự làm

1. Viết bộ sinh ảnh v2, chạy thử 200 ảnh/task, nhóm duyệt gallery.
2. Sinh toàn bộ và nghiệm thu (mục 4G).
3. Viết code chọn ngưỡng theo nhóm, ghi log theo mục 2, script FULL-SYN.
4. Chạy FULL-SYN task 1 và **kiểm tra điểm dừng** (mục 1.3).
5. Nếu đạt, chạy tiếp task 2–5. Chạy song song FULL-REAL nếu đủ ngân sách.
6. Sau đó mới đến: so sánh với IncreACO (183+17), thêm seed, demo task 6.

## Thuật ngữ nhanh

- **Cutout:** ảnh một sản phẩm đã tách khỏi nền, kèm kênh trong suốt (RGBA).
- **Mask:** ảnh đen trắng đánh dấu pixel nào thuộc vật.
- **Mask chủ:** mỗi pixel ghi vật nào đang nằm trên cùng.
- **Nhãn giả:** model của task trước (model thầy) tự gán nhãn cho SKU cũ trong ảnh mới.
- **Replay:** dùng lại dữ liệu có nhãn của lớp cũ. Nhóm không dùng.
- **NMS không phân biệt lớp:** nếu hai khung chồng nhau nhiều thì giữ khung có điểm cao nhất, bất kể là SKU nào.
- **FID/KID:** chỉ số đo hai tập ảnh "giống nhau" đến đâu. Càng thấp càng giống.
- **Tái lập từng byte:** chạy lại ra file giống hệt, để kết quả kiểm tra được.
