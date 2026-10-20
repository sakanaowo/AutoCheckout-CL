# Handoff: xử lý ảnh ghép và train model — 20/10/2026

Người nhận: người phụ trách **sinh ảnh ghép** và **train/đánh giá model**.
Tài liệu này tự đủ để bắt đầu làm. Chi tiết nằm trong các file ở mục 9.

---

## 1. Dự án trong 5 dòng

- **AutoCheckout-CL:** quầy tính tiền tự động. Camera chụp khay từ trên xuống, model phát hiện và **đếm** từng SKU.
- **Học tăng dần** (class-incremental object detection), **không replay**: thêm SKU mới mà không train lại từ đầu và không dùng lại dữ liệu có nhãn của SKU cũ.
- **Phương pháp:** PDP (CVPR 2026), dựa trên Deformable DETR, dùng prompt pool chung và riêng, cùng nhãn giả cho lớp cũ do model thầy tạo.
- **Dữ liệu:** RPC (200 SKU), giao thức **100+4×25**: task 1 có 100 SKU, task 2–5 mỗi task thêm 25 SKU. Seed 0.
- **Code:** repo `AutoCheckout-CL`, nhánh `develop`. Code gốc của tác giả nằm ở `../PDP_IOD`.

## 2. Hiện trạng

### 2.1 Đã có
- Bản phát hành ảnh thật `data/processed/rpc_100-4x25_seed0_native_v1/`: train 22.494 / val 1.503 / test 6.003 ảnh. Chia theo nhóm ảnh, test đã khóa, nền không lấy từ val/test.
  - Thư mục `data/` **chỉ có trên máy train**, không có trên máy Mac.
- Hai bản baseline đã chạy đủ 5 task (640px, batch 8, 16 epoch/task, TF32, 4090/4080S):

| Sau task 5 (test) | B1: ResNet-50 | B2: ConvNeXt-V2-Base |
|---|---:|---:|
| AP50 | 62,33% | **84,40%** |
| cAcc (đếm đúng cả giỏ) | 2,58% | **21,74%** |
| Forgetting | **6,63** | 8,67 |

- AP50 của B2 cho SKU **Cũ / Mới** sau từng task: task 2: 98,2 / 95,2; task 3: 96,7 / 85,5; task 4: 92,2 / 76,4; task 5: 85,9 / 74,7.
  - **Cột Mới giảm qua từng task**, và SKU mới bị đếm thiếu (sau task 5 chỉ đếm được khoảng 43%).
- Tài nguyên ảnh ghép:
  - **7.502 cutout** SAM (RGBA, 10–48 cutout/SKU, đủ 200 SKU);
  - **208 nền khay** 1840×1840.
  - Bộ 20.000 ảnh ghép cũ **không dùng**: sai số task (4 task), lật ngang 50%, bbox lệch mask, không lưu provenance.

### 2.2 Chẩn đoán vì sao kết quả thấp
1. **Baseline đang TẮT các cải tiến đã có trong code.**
   - `configs/exp/native/common.sh` đặt `--pseudo_dedup_iou 0` (F14 tắt), không có `--pseudo_gt_iou` (I3 tắt), và không có FSA: backbone, encoder và decoder bị khóa ngay từ task 1.
2. **B2 bị lệch:** encoder/decoder được train trên COCO cùng ResNet-50 nhưng nhận đặc trưng ConvNeXt. Phần nối (`input_proj`) được khởi tạo mới và chỉ học với lr 1e-5.
   - **Không có** encoder/decoder nào được train trên ImageNet; ImageNet chỉ pretrain backbone. Cách sửa là **FSA**: task 1 fine-tune toàn bộ model.
3. **SKU mới bị gán nhầm thành SKU cũ.** Model thầy gán nhãn giả "SKU cũ" lên vật thực ra là SKU mới, khi I3 tắt.
   - Thí nghiệm tháng 9 (cấu hình cũ): cột Mới đạt 0,683 khi tắt I3, 0,925 khi bật.
4. **Task 1 chỉ có nhãn cho khoảng 50% số vật** (138.280/276.091). Nửa còn lại là SKU của task sau nhưng bị coi là nền, nên model phải "học ngược" ở task sau.

## 3. Quyết định của nhóm

| # | Quyết định |
|---|---|
| 1 | **Train hoàn toàn bằng ảnh ghép.** Val và test là ảnh thật, giữ nguyên. |
| 2 | **Task 1:** ảnh ghép chỉ chứa 100 SKU của task 1, **gắn nhãn tất cả**. |
| 3 | **Task 2–5:** SKU mới (gắn nhãn) + SKU cũ (**không gắn nhãn**, model thầy tạo nhãn giả). **Không bao giờ** đưa SKU của task sau vào ảnh. Gắn nhãn SKU cũ sẽ thành replay: **cấm**. |
| 4 | Chạy luôn **một bản full option: FULL-SYN** (mục 5). Nên chạy kèm **FULL-REAL** để đối chứng. |
| 5 | Sau mỗi task **ghi log chi tiết** (mục 6). |
| 6 | **Không lật ảnh** ở bất kỳ bước nào, vì chữ và logo sẽ bị ngược. |
| 7 | Chọn ngưỡng đếm trên **val**. Test chỉ dùng để báo cáo. |

## 4. Gói việc 1: sinh ảnh ghép v2 (làm trước)

**Prompt đầy đủ và 27 yêu cầu:** [KE_HOACH_ANH_GHEP_VA_BAN_FULL.md](KE_HOACH_ANH_GHEP_VA_BAN_FULL.md), mục 4 và 5. Có thể giao nguyên prompt đó cho AI coding agent hoặc làm theo. Các điểm quan trọng nhất:

### 4.1 Rủi ro số 1: góc chụp của cutout
- Cutout lấy từ ảnh đơn của RPC train2019, chụp bằng **4 camera: từ trên, 45°, 30° và ngang**. Camera quầy chụp **từ trên xuống**.
- Cách xử lý:
  - **Từ trên:** giữ, ưu tiên cao.
  - **Ngang:** giữ **nếu** là tư thế nằm ổn định trên khay (mặt lớn, gần chữ nhật). Bỏ mặt hẹp như đáy chai hoặc cạnh hộp.
  - **30° và 45°:** bỏ, hoặc giữ dưới 10%. Ảnh ở góc này thấy 2–3 mặt và bị méo phối cảnh.
- Cách xác định góc:
  - **A:** lấy từ metadata camera của train2019, nối từ ảnh gốc của cutout. **Việc đầu tiên: kiểm tra index cutout có lưu ảnh gốc hoặc id camera không.**
  - **B:** nếu không có, phân loại bằng độ vuông vức của mask và "độ ổn định" của IncreACO, rồi xem bằng mắt.
- Kiểm tra lại theo từng SKU: so phân bố tỉ lệ cạnh và diện tích bbox của ảnh ghép với **ảnh train thật**.
- (Tùy nhóm quyết) chọn cutout theo độ giống DINOv2/CLIP với khoảng 20 vật cắt từ ảnh train thật mỗi SKU. Nếu làm thì phải ghi rõ là có dùng một ít ảnh thật.

### 4.2 Các yêu cầu khác
- **Bố cục:**
  - số vật/ảnh và tỉ lệ easy/medium/hard lấy từ annotation **train thật** (trung bình khoảng 12,3 vật/ảnh);
  - một hệ số tỉ lệ chung, chỉnh cho diện tích bbox khớp ảnh thật, cộng dao động ±10%;
  - xoay 0–360°;
  - đặt sát cạnh vật trước như IncreACO;
  - mỗi vật còn nhìn thấy ≥ 70%.
- **Độ chân thực:**
  - thu mask vào 1–2px rồi làm mờ mép;
  - bóng đổ mềm;
  - dao động màu, sáng, mờ, nhiễu;
  - nén JPEG chất lượng 85–95.
- **Nhãn:**
  - bbox tính từ **mask phần nhìn thấy cuối cùng**;
  - lưu RLE và ảnh mask chủ.
- **Chia task:**
  - task t ≥ 2: mỗi ảnh có ≥ 1 SKU mới, vật mới chiếm 40–60%;
  - mỗi task xuất `train_task_<t>.json` (chỉ nhãn task t) và `train_task_<t>_gt_full.json` (đủ nhãn, **chỉ để kiểm tra**);
  - mã nhãn 0..199 theo `task_config.json`.
- **Số lượng:** số ảnh mỗi task bằng số ảnh thật B2 đã dùng (xem trong log B2). Số vật của mỗi SKU mới chênh tối đa ±20%.
- **Nghiệm thu trước khi train:**
  - pytest kiểm tra: không có SKU tương lai, không có nhãn SKU cũ trong file train, bbox khớp mask, không lật, tái lập từng byte;
  - so thống kê với ảnh train thật;
  - xem 100 ảnh đã vẽ bbox.
- **Chạy thử 200 ảnh/task** để nhóm duyệt rồi mới sinh toàn bộ.
- **Không dùng val/test** (ảnh hay thống kê) để sinh hoặc tinh chỉnh ảnh.

## 5. Gói việc 2: train FULL-SYN

### 5.1 Cấu hình

| Thành phần | Cờ / cách làm |
|---|---|
| Backbone | ConvNeXt-V2-Base `convnextv2_base.fcmae_ft_in22k_in1k` (timm) |
| **FSA** (task 1 fine-tune toàn bộ) | Như tháng 9: `FINETUNE_ARGS` (`configs/exp/common.sh`) + `--save_hf 1` → `task_1/hf_model`. Sau đó PDP chạy với `--repo_name <FSA>/task_1/hf_model`. **Cần kiểm tra việc export hf_model chạy được với backbone ConvNeXt** (tháng 9 mới làm với ResNet). |
| F1–F13 | Bật như B1/B2 |
| **F14** | `--pseudo_dedup_iou 0.5` |
| **I3** | `--pseudo_gt_iou 0.5` |
| Nhãn giả | τ_h 0,5 / τ_l 0,2 (mặc định) |
| Đếm | NMS không phân biệt lớp, IoU 0,45 (`autocheckout/counting.py: dedup_detections`, đã có) |
| **Ngưỡng đếm theo nhóm task** | **Cần viết code**: hiện `select_threshold` chỉ chọn một ngưỡng. Chọn mỗi nhóm task một ngưỡng trên val, rồi áp cho test. |
| Train | 640px, batch 8, 16 epoch/task, seed 0, TF32. **Giữ giống B2.** |
| **Không bật** | `--augment`, `--augment_flip`, `--prototype_nearest`, `--freeze_shared_after_task1`, LoRA. Chưa có bằng chứng; mỗi thứ thêm sau, từng cái một. |

- Script mới: `configs/exp/native/EXP-FULL-SYN.sh`, dựa trên `EXP-B2.sh` + `native/common.sh`, thay đường dẫn task annotation sang ảnh ghép.
  - `tools/make_task_json.py` có `--train-source NAME=PATH`.
  - Val/test vẫn trỏ về bản phát hành ảnh thật.
- **FULL-REAL** (đối chứng): giống hệt FULL-SYN nhưng train trên `tasks/real`.

### 5.2 Điểm dừng sau task 1
- Đánh giá trên **val thật** rồi so với task 1 của B2 (test cAcc 73,16%, AP50 98,35%).
- Nếu cAcc kém hơn khoảng 15 điểm, **dừng**: cải thiện ảnh ghép rồi chạy lại task 1.
  - Lý do: FSA trên ảnh ghép làm backbone quen ảnh ghép, và backbone bị khóa từ task 2.

### 5.3 Chi phí tham khảo
- 4090 khoảng 0,4–0,46 USD/giờ. ConvNeXt khoảng 10,9 ảnh/giây; B2 chạy hết khoảng 50–65 giờ cho 5 task.
- FULL-SYN và FULL-REAL mỗi bản khoảng 2,5–3 ngày GPU, khoảng 25–35 USD.
- **Thời gian FSA chưa đo thực tế** vì fine-tune toàn bộ chậm hơn.

## 6. Log bắt buộc sau mỗi task

Mỗi task ghi `task_<t>/task_log.json` và `task_<t>/SUMMARY.md`. Đã có sẵn `run_info.json`, `train.log` và CSVLogger; phần còn thiếu cần bổ sung.

- **Thông tin chạy:**
  - run id;
  - git commit và diff;
  - câu lệnh và **toàn bộ args**;
  - GPU và phiên bản thư viện;
  - thời gian, số giờ GPU, chi phí.
- **Dữ liệu:**
  - SKU mới và SKU đã học;
  - file annotation và md5;
  - số ảnh, số vật có nhãn, số vật không nhãn;
  - phiên bản, config hash và seed của bộ sinh ảnh.
- **Model:**
  - checkpoint khởi tạo và md5;
  - tham số train/khóa theo từng module;
  - lr theo từng nhóm;
  - các cờ F/I đang bật.
- **Train (theo epoch):**
  - cls, bbox, giou, L_Q, L_DDL;
  - lr, độ lớn gradient;
  - số ảnh/giây, bộ nhớ GPU tối đa.
- **Nhãn giả (task ≥ 2):**
  - số nhãn giả/ảnh;
  - số bị I3 loại, số bị F14 loại;
  - phân bố điểm;
  - **precision/recall so với `gt_full`**.
- **Đánh giá trên val và test:**
  - AP50 Cũ/Mới/Tất cả và theo từng nhóm;
  - cAcc, ACD, mCCD, mCIoU, mCCS, **tổng và theo easy/medium/hard**;
  - ngưỡng đã chọn cho từng nhóm;
  - một dòng của ma trận độ chính xác, forgetting;
  - AP từng SKU;
  - **top-20 cặp nhầm SKU mới thành SKU cũ**.
- **Đầu ra:** checkpoint, hf_model, file dự đoán, kèm md5.

## 7. Bẫy cần tránh

| Bẫy | Hậu quả |
|---|---|
| Gắn nhãn SKU cũ trong ảnh task mới | Thành replay; mất luận điểm "không replay" |
| Đưa SKU tương lai vào ảnh | Lộ lớp tương lai, sai giao thức |
| Dùng val/test để chỉnh bộ sinh ảnh hoặc chọn ngưỡng trên test | Kết quả bị thổi phồng, không công bằng |
| Lật ảnh | Chữ và logo ngược; nhiều SKU RPC chỉ khác nhau ở chữ |
| Đổi nhiều thứ cùng lúc ngoài FULL-SYN | Không biết thứ gì có tác dụng |
| Bỏ qua điểm dừng sau task 1 | Tốn 2 ngày GPU cho một bản đã hỏng từ đầu |
| Tính bbox trước khi đặt xong mọi vật | Lỗi bbox lệch mask giống bộ cũ |
| Chỉ báo cáo ảnh dễ | Không so được với RPC/IncreACO (họ báo cáo đủ 3 mức) |

## 8. Câu hỏi còn mở (nhóm chốt dần)

1. Index cutout có id camera hoặc ảnh gốc không? Nếu không có thì dùng cách B ở mục 4.1.
2. Có dùng một ít ảnh train thật để chọn cutout (DINOv2/CLIP) không? Việc này ảnh hưởng câu chuyện "chỉ cần ảnh đơn".
3. Các tham số tạm đặt: vật mới chiếm 40–60%, còn nhìn thấy ≥ 70%, ngưỡng dừng −15 điểm cAcc.
4. Nếu ảnh ghép không đạt điểm dừng: trộn khoảng 10% ảnh thật, hay dùng CycleGAN như IncreACO (ảnh ghép thô khoảng 26% cAcc, sau CycleGAN khoảng 75%)?
5. Việc sau FULL-SYN: so với IncreACO theo giao thức 183+17 (IncreACO đạt cAcc 74,3%), thêm seed, demo task 6 với SKU Việt Nam.

## 9. Thông tin đã xác minh khác (đọc để có đủ context)

### 9.1 Code của nhóm khác code gốc PDP thế nào
Kiểm tra trực tiếp `../PDP_IOD` (commit `7702d91`) với `pdp/` và đối chiếu với PDF paper.

- **Code gốc chạy trên RPC sẽ ra sai phương pháp.** Nhóm đã sửa F1–F13, các bản sửa **đang bật** ở B1, B2 và FULL-SYN:

| Mã | Lỗi trong code gốc → cách sửa |
|---|---|
| F1 | Viết cứng 80 lớp COCO (`total_classes = 80`) → đọc cấu hình task của RPC |
| F2 | Prompt riêng từ task 2 trở đi bằng 0 và không bao giờ học → `init_task_prompts()` |
| F3 | L_DDL bị tắt (`use_ddl_loss = False`, loss bị bỏ đi) → tính và cộng vào loss, λ 0,15 |
| F4 | L_Q tính trong `no_grad` → cho gradient chạy qua |
| F5/F8 | Teacher chạy không có prompt; file `checkpoint07.pth` viết cứng; thiếu file thì PPG tắt mà không báo lỗi → teacher là bản sao đóng băng của model cuối task trước |
| F6 | PPG chỉ xét top-5 cặp mỗi ảnh, dùng `<=` làm lọt lớp mới, lấy sai đặc trưng → viết lại `pdp/ppg.py` |
| F7 | Prototype lấy cả dự đoán sai → chỉ lấy dự đoán đúng (theo paper) |
| F11 | Kernel CUDA không được nạp, chỉ chậm hơn → nạp từ `transformers`. Con số "chậm 20 lần" do nhóm đo, chưa đo lại |
| F12 | Không xáo dữ liệu khi chạy 1 GPU → `shuffle` |
| F13 | `post_init()` xóa prior bias của classifier → khôi phục |

- **Cấu hình khác code gốc.** Paper **không ghi** batch, epoch, kích thước ảnh. Giá trị gốc lấy từ `run.sh`/`main.py`.

| | Code gốc | Nhóm |
|---|---|---|
| Ảnh | 800/1333 | 640 |
| Batch hiệu dụng | 32 (viết cứng) | 8 |
| Epoch | 8 (viết cứng) | 16 |
| GPU | 2 | 1, có TF32 |

- Lý do dùng batch 8: thí nghiệm tháng 9, batch 32 cho val mAP50 **0,085**, batch 4 cho **0,675** (cùng dữ liệu và lượng tính toán). Mỗi task RPC có ít ảnh hơn COCO khoảng 10 lần.
- **16 epoch chưa được kiểm tra hội tụ.** Mới chạy 1 seed.
- **Giống paper:**
  - lr 1e-4 (phần mới) / 1e-5 (phần cũ);
  - 100 prompt chung, prompt dài 10;
  - λ_Q 0,1;
  - ngưỡng PPG 0,5 / 0,2 / 0,5;
  - mỗi lớp 1 prototype, bộ nhớ 100 vector/lớp.
- Chi tiết và dòng code làm bằng chứng: [REPORT_20-10.md](REPORT_20-10.md) mục 4.

### 9.2 B2 lệch encoder/decoder: chi tiết
- `plan_detector_transfer` (`pdp/models/backbones.py`) khởi tạo mới backbone, `input_proj`, `class_embed` và prompt khi đổi backbone. Encoder/decoder giữ nguyên từ checkpoint COCO+ResNet (`SenseTime/deformable-detr`).
- Ở B2, encoder/decoder bị khóa; `input_proj` (phần nối) chỉ học với lr_old 1e-5. Nghĩa là encoder/decoder "nhìn" đặc trưng ConvNeXt qua một lớp chuyển đổi gần như chưa học.
- Không cần train lại từ đầu. FSA ở task 1 (mở khóa toàn bộ) là đủ, chi phí khoảng một task fine-tune.

### 9.3 Cách tính các chỉ số (để đọc log cho đúng)
- **AP50 Cũ/Mới/Tất cả** (mAP@P/C/A): chỉ tính trên các SKU đã học, và **chỉ trên các ảnh test có chứa SKU của nhóm đang đo**.
- **cAcc, ACD, mCCD, mCIoU, mCCS:** tính trên **toàn bộ 6.003 ảnh test**, chỉ đếm các SKU đã học; SKU chưa học trong ảnh bị bỏ qua. cAcc chỉ tính đúng khi đếm đúng **mọi** SKU đã học trong ảnh, nên tụt mạnh hơn AP50.
- Ngưỡng đếm được chọn trên val (lưới 0,05–0,95, chọn ngưỡng nhỏ nhất đạt cAcc cao nhất); NMS 0,45.
- Forgetting tính từ ma trận độ chính xác.
- Code: `autocheckout/cl_metrics.py`, `autocheckout/counting.py`, `tools/eval_count.py`.
- Các dòng ở những task khác nhau **không cùng thước đo** vì số SKU tăng. Muốn xem riêng phần quên thì nhìn cột "Cũ".

### 9.4 B1/B2 kết luận được đến đâu
- **Kết luận được:** trong cấu hình khóa backbone, ConvNeXt cho kết quả cuối tốt hơn rõ (AP50 +22, cAcc +19), nhất quán ở mọi task và mọi mức khó.
- **Chưa kết luận được:**
  - bên nào quên ít hơn (6,63 so với 8,67, mới 1 seed);
  - kiến trúc ConvNeXt có tốt hơn nói chung không, vì đổi cùng lúc pretrain (COCO so với IN-22k), kích thước (25M so với 88M tham số) và độ khớp encoder.
- Câu viết an toàn trong báo cáo: xem REPORT mục 2.

### 9.5 Kết quả tháng 9 (nhánh `main`, pipeline khác, chỉ để tham khảo)
- Bật FSA + F14: AP50 khoảng 94,7%, cAcc khoảng 36–43%, gần như không quên.
- PDP bắt đầu từ FSA: val mAP50 **0,83**; bắt đầu từ COCO bị khóa: **0,11**.
- Tắt F14 (E4_noF14): test cAcc **0,10** sau task 5. Nhãn giả trùng tăng từ 3,7% lên 31% qua các task.
- I3: cột Mới **0,683 → 0,925**.
- Các số này **không so trực tiếp** được với B1/B2 (khác pipeline và dữ liệu) nhưng cho thấy hướng đúng.

### 9.6 Vì sao cột "Mới" giảm qua từng task
1. Model thầy gán nhãn giả SKU cũ lên vật thực ra là SKU mới (I3 tắt). Đây là nguyên nhân chính.
2. Backbone, encoder và decoder bị khóa với đặc trưng chưa thấy SKU mới. FSA chỉ giúp task 1; từ task 2 vẫn khóa.
3. Classifier thiên về lớp cũ, và ngưỡng chung cho mọi lớp làm SKU mới (điểm thấp hơn) bị loại.
4. SKU mới đã bị coi là nền ở các task trước (vấn đề "nửa có nhãn").
- Cách sửa: I3, F14, FSA, ngưỡng theo nhóm, và ảnh ghép có nhãn đầy đủ.

### 9.7 Train chỉ bằng ảnh ghép: hợp lệ nhưng rủi ro
- Hợp lệ: đây chính là giao thức chính thức của RPC/IncreACO (train trên ảnh đơn hoặc ảnh ghép, test trên ảnh quầy).
- Rủi ro chênh lệch ảnh ghép/ảnh thật. Theo bảng 1 của IncreACO, cAcc trung bình (không học tăng dần):

| Dữ liệu train | cAcc |
|---|---:|
| Ảnh ghép kiểu RPC gốc | 9,27% |
| Ảnh ghép IncreACO (thô) | 26,52% |
| Ảnh ghép đã CycleGAN | 75,30% |
| Ghép + CycleGAN | 77,15% |

- Bài học: **chất lượng ảnh ghép quyết định kết quả**, và CycleGAN là bước tạo khác biệt lớn nhất.

### 9.8 So sánh với IncreACO (WACV 2021): điều kiện
- IncreACO học tăng dần **183 + 17**: cAcc **74,3%**, ACD 0,44, mCCD 0,04, mCIoU 96,51%, mCCD cũ/mới 0,03/0,09.
- Khác biệt khi so:
  - họ dùng Mask R-CNN, ảnh 800×800;
  - **có replay** (SKU cũ có nhãn với α = 0,05);
  - ảnh ghép + CycleGAN;
  - test trên toàn bộ test2019. Nhóm dùng tập test 6.003 ảnh tự chia.
- Muốn so thì phải chạy **giao thức 183+17** (mỗi nhóm hàng lấy 1 SKU làm SKU mới). Khi viết, ghi rõ "so tham khảo, khác test split và có/không replay".
- Lợi thế để nêu: nhóm **không replay**.

### 9.9 Số liệu dữ liệu và tài nguyên
- Ảnh thật theo task (ảnh train / vật có nhãn task đó):

| Task | Ảnh train | Vật có nhãn |
|---|---:|---:|
| 1 | 21.752 | 138.280 |
| 2 | 12.534 | 34.197 |
| 3 | 12.129 | 32.320 |
| 4 | 11.850 | 32.859 |
| 5 | 13.377 | 38.435 |

  Bản capped giới hạn 6.000 ảnh/task. Một ảnh có thể thuộc nhiều task.
- Classifier có **224 ô / 225 đầu ra**: 200 SKU cộng 24 ô dự phòng, chưa dùng. Task 6 (demo) có thể dùng các ô dự phòng này (cần mở rộng `task_config`; chưa thử).
- Nền 208 ảnh làm từ val2019; split giữ mọi nhóm ảnh có val2019 ở train, val/test chỉ lấy từ test2019 (đã kiểm tra không lộ đề). Một số nền còn vết inpainting.
- Cutout: 2.098 đã bị lọc tự động (2.043 do IoU, 55 do fill), chưa có review bằng mắt.
- Ảnh ghép cũ: 20.000 ảnh, 238.013 vật; 202 ảnh lỗi đã xử lý (193 ghép lại, 9 giữ), còn 20 case chưa xử lý. **Không dùng cho FULL-SYN.**

### 9.10 Chi phí và tốc độ (nhóm đo)
- 4090 khoảng 0,4–0,46 USD/giờ.
- Tốc độ train: ConvNeXt khoảng 10,9 ảnh/giây, ResNet khoảng 16,8 ảnh/giây.
- B1 hoặc B2 chạy đủ 5 task mất khoảng 50–65 giờ.
- B2 đã chạy qua nhiều máy (4090 → L4 → 4080S) bằng cơ chế chạy tiếp từ checkpoint; kết quả không bị ảnh hưởng.

### 9.11 Demo
- Task 6: 5–10 SKU Việt Nam học từ ảnh ghép (cutout tự chụp), tính tổng tiền theo `catalog.csv`.
- Hướng dẫn chụp: [HUONG_DAN_CHUP_DATA_DEMO.md](HUONG_DAN_CHUP_DATA_DEMO.md). Code app/demo **chưa có**.

### 9.12 Những gì CHƯA xác minh
- File tham số thực tế của B1/B2 (`resolved_cli.json`) và các file dự đoán: nằm trong `data/` trên máy train, chưa đọc.
- Export FSA với ConvNeXt; thời gian chạy FSA.
- 16 epoch đã hội tụ chưa; độ dao động giữa các seed.
- Index cutout có id camera không.
- Cờ `--freeze_shared_after_task1`: tài liệu cũ ghi "chưa khóa đúng các prompt shared". Không dùng trong FULL-SYN.

## 10. Tài liệu liên quan

| File | Nội dung |
|---|---|
| [KE_HOACH_ANH_GHEP_VA_BAN_FULL.md](KE_HOACH_ANH_GHEP_VA_BAN_FULL.md) | **Yêu cầu đầy đủ và prompt cho bộ sinh ảnh**, cấu hình FULL-SYN, danh sách log |
| [REPORT_20-10.md](REPORT_20-10.md) | Tổng quan dự án, kết quả B1/B2, khác biệt với code gốc (F1–F14), thuật ngữ |
| `AutoCheckout-CL/docs/project/FINAL_BACKBONE_COMPARISON_2026-10-20.md` | Số liệu chi tiết B1/B2 |
| `AutoCheckout-CL/docs/data_preprocessing/reports/preprocessing-acceptance-2026-10-07.md` | Kiểm tra cutout, nền và lỗi của bộ ảnh ghép cũ |
| `AutoCheckout-CL/docs/data_preprocessing/reports/split-100-4x25-2026-10-07.md` | Cách chia split và task, số liệu theo task |
| `AutoCheckout-CL/configs/exp/native/` | Script B1/B2 và `common.sh` |
| `AutoCheckout-CL/configs/exp/common.sh` | FSA/E4 tháng 9 (`FINETUNE_ARGS`, `FSA_ARGS`) |
| `AutoCheckout-CL/pdp/main.py` | Các cờ `--pseudo_*`, `--save_hf`, `--augment*` |
| `AutoCheckout-CL/autocheckout/counting.py` | NMS khi đếm, chọn ngưỡng |
| Paper PDP và IncreACO | Trong thư mục `AN/` (phần sinh ảnh: IncreACO mục 3.1) |
