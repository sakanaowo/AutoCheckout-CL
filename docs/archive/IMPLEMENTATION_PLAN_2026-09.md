# Kế hoạch lịch sử AutoCheckout-CL — tháng 09/2026

Phần bên dưới được giữ nguyên làm tư liệu của hướng GCP trước đây. Các quyết định về VM, nguồn dữ liệu, độ phân giải cố định, đường dẫn, thí nghiệm và kết quả không phải cấu hình hiện hành; khi khác nhau, ưu tiên kế hoạch investigation ở trên.

### Kế hoạch cũ: giai đoạn 1 — ảnh quầy thật RPC

- Phiên bản 1.3, ngày 28/09/2026. Trạng thái: **đang implement**; tiến độ từng task ghi ở [PROGRESS.md](PROGRESS_2026-09.md). Máy ảo: `auto-cl` (L4), xem mục 3.1. Các thay đổi so với v1.2 (sau lượt rà soát trước khi code) liệt kê ở Phụ lục B.
- Tài liệu liên quan: paper PDP (file PDF trong repo), [phân tích trước đó](<../references/Phân tích PDP cho đồ án thanh toán tự động (RPC).md>), [hướng dẫn máy ảo](GCP_TRAINING_GUIDE.md).
- Code gốc tham chiếu: [PDP_IOD](https://github.com/zyt95579/PDP_IOD), commit `7702d91`. Mọi số dòng trong tài liệu này tính theo commit đó; link trỏ cố định vào commit.
- Các điểm ghi "đã kiểm chứng" là đã chạy code hoặc đọc nguồn gốc. Các con số ghi "ước tính" phải thay bằng số đo ở pilot.

---

## 1. Mục tiêu và phạm vi

### 1.1 Bài toán

Camera chụp các sản phẩm khách đặt trên quầy, hệ thống nhận diện từng sản phẩm (SKU), đếm số lượng và tính tổng tiền. Cửa hàng nhập thêm sản phẩm theo từng đợt, nên mô hình phải học thêm SKU mới mà **không train lại từ đầu và không lưu ảnh của các đợt trước**, đồng thời vẫn nhận ra SKU cũ.

### 1.2 Phạm vi giai đoạn 1

- **Trong phạm vi:**
  - Dùng ảnh quầy thật của RPC (`val2019` và `test2019`, tổng 30.000 ảnh có bbox) cho cả train, val và test.
  - Sửa code PDP cho đúng paper và chạy được trên RPC.
  - Xây các baseline, đánh giá mAP theo từng task và đánh giá đếm/tính tiền (cAcc).
- **Ngoài phạm vi (để sau):**
  - Ảnh sản phẩm đơn (`train2019`) và ảnh ghép của nhóm (mục 11).
  - Demo webcam với sản phẩm Việt Nam (mục 10), tùy QĐ-5.

### 1.3 Kịch bản mô phỏng

- Cửa hàng mở với 100 SKU (task 1), sau đó có 4 đợt nhập hàng, mỗi đợt 25 SKU (task 2–5).
- Mỗi đợt có ảnh quầy mới. **Nhân viên chỉ gán bbox cho SKU mới.** SKU cũ vẫn xuất hiện trong ảnh nhưng không có nhãn; PPG phải tự gán nhãn giả cho chúng.
- Không giữ ảnh của các đợt trước. Chỉ được giữ trọng số, prompt và prototype (các vector embedding, không phải ảnh).
- Kịch bản này trùng với giao thức IOD của paper: ảnh của task t có thể chứa vật của lớp cũ và lớp tương lai, nhưng chỉ lớp của task t có nhãn.

### 1.4 Khi nào giai đoạn 1 được coi là xong

1. Code PDP đã sửa; mỗi sửa đổi có test tự động; smoke test chạy hết trên cả CPU và GPU.
2. Tập test được chọn một lần và **khóa** (md5 lưu trong repo); không bao giờ dùng test để chọn tham số.
3. Có bảng kết quả E0–E5 (mục 7) trên cùng tập test:
   - mAP@C, mAP@P, mAP@A sau từng task;
   - ma trận độ chính xác theo task;
   - cAcc, ACD, mCCD, mCIoU sau task cuối.
4. Có ablation tối thiểu A1–A4.
5. Mọi con số tái tạo được từ file dự đoán và cấu hình đã lưu.

### 1.5 Khác biệt so với tài liệu phân tích trước

- Test chính trên ảnh quầy thật, không dùng ảnh ghép.
- Code gốc cần 12 nhóm sửa (F1–F12, mục 6.3), không chỉ lỗi kích thước pool 80 lớp. Chỉ sửa kích thước pool là **chưa đủ**: prompt riêng của task ≥ 2 vẫn không học.
- Nhóm đã có máy ảo GPU chạy được (mục 3.1), không bị chặn GPU như lo ngại trước đây.

---

## 2. Quyết định

### 2.1 Đã chốt hoặc đề xuất mặc định

| Nội dung | Giá trị | Trạng thái | Lý do |
|---|---|---|---|
| Dữ liệu train giai đoạn 1 | Chỉ ảnh quầy thật | Nhóm đã chốt | Ảnh ghép đang làm |
| Tập test | Ảnh quầy thật, khóa cố định | Nhóm đã chốt | Ảnh ghép không đo được khoảng cách giữa ảnh ghép và ảnh thật |
| Đơn vị khi chia tập | **Nhóm ảnh**, không chia theo từng ảnh (mục 4.2) | Đề xuất | Có các nhóm ảnh chụp liên tiếp cùng một bộ hàng |
| Chia lớp | 100 + 4×25; phân tầng theo 17 nhóm hàng; seed 0 | Đề xuất | Mô phỏng catalog ban đầu rồi nhập hàng theo đợt; mỗi đợt có đủ loại hàng |
| Độ phân giải | 800×800 (640 là phương án dự phòng) | Đề xuất | Giống IncreACO; chốt lại sau khi đo tốc độ ở pilot |
| Ảnh dùng cho nhiều task | Như paper: một ảnh có thể xuất hiện ở nhiều task, mỗi task chỉ giữ nhãn lớp của task đó | Đề xuất | So sánh được với paper |
| Chọn tham số, ngưỡng, cấu hình | Chỉ dựa trên val | Đề xuất | Test chỉ dùng để báo cáo |

### 2.2 Cần nhóm chốt

| Mã | Câu hỏi | Đề xuất | Phải chốt trước |
|---|---|---|---|
| QĐ-1 | Số slot lớp của mô hình | **Đã chốt 28/09: 224** (200 SKU RPC + 24 dự phòng cho demo); classifier 225 đầu ra | — |
| QĐ-2 | Kịch bản "nhân viên chỉ gán nhãn SKU mới trên ảnh quầy" có phù hợp cách trình bày đề tài không | **Đã chốt 28/09: có** | — |
| QĐ-3 | Dọn dữ liệu project cũ | **Đã xong 28/09**: ổ 200 GB đã được format; sau đó VM cũ `anmetarayban` và cả 2 ổ của nó đã bị xóa (phải tắt deletion protection trước). Còn 14 snapshot của ổ boot cũ (khoảng 30 GB, khoảng 40 nghìn VND/tháng) chờ nhóm quyết định | — |
| QĐ-7 | Code PDP_IOD gốc không có LICENSE, có được đưa lên repo GitHub public không | **Chốt lại 28/09: đẩy lên repo public, ghi rõ nguồn** (`pdp/UPSTREAM.md`). Nhóm đã được nhắc là code gốc không có giấy phép. VM lấy code bằng `git clone`/`git fetch` | — |
| QĐ-4 | Ngân sách | Còn khoảng 4 triệu VND (≈ 153 USD theo tỷ giá trong bảng giá GCP), **hết hạn 24/10/2026** (mục 3.6) | — |
| QĐ-6 | Cấu hình VM (mục 3.5) | **Đã chọn phương án B (28/09)**: VM `auto-cl`. Hiện đang chạy on-demand; nên chuyển sang Spot khi R1/R2 xong | — |
| QĐ-5 | Demo webcam với sản phẩm Việt Nam (task 6) có nằm trong phạm vi đồ án không | Có, nếu còn thời gian | Giai đoạn 8 |

---

## 3. Hạ tầng

### 3.1 Máy ảo đang dùng: `auto-cl` (tạo và kiểm tra ngày 28/09/2026)

| Thuộc tính | Giá trị |
|---|---|
| Tên / zone / project | `auto-cl` / `us-central1-c` / `project-95a0d104-9d0f-4aa1-ba0` |
| Loại máy | `g2-standard-4` (4 vCPU, 16 GB RAM) |
| GPU | 1 × NVIDIA **L4**, 23 GB dùng được, compute capability 8.9 |
| Kiểu cấp phát | **STANDARD (on-demand)**, khoảng 18.400 VND/giờ. Nên chuyển sang **Spot** (khoảng 11.100 VND/giờ) sau khi R1/R2 chạy được; cách chuyển ở mục 2.5 của hướng dẫn VM |
| Ổ | 1 ổ boot 100 GB `pd-balanced`, còn trống khoảng 80 GB, ghi khoảng 178 MB/s; snapshot hằng ngày, giữ 14 ngày |
| Phần mềm | Image `common-cu129-ubuntu-2204-nvidia-580`: driver 580.178.04, CUDA 12.9, Python 3.10.12; không có sẵn PyTorch; thiếu `python3.10-venv` và `unzip` |
| VM cũ `anmetarayban` (V100) | **Đã xóa cùng 2 ổ ngày 28/09** (QĐ-3) |

### 3.2 Hệ quả cho việc implement

- **Spot (khi chuyển sang):** code bắt buộc phải resume được giữa chừng (R1), và script chạy phải tự bỏ qua phần đã xong (R2). Hai task này cũng hữu ích ngay cả khi chạy on-demand.
- **Ổ và RAM:** ổ `pd-balanced` nhanh (khoảng 178 MB/s), nhưng RAM chỉ 16 GB. Vẫn nên thu nhỏ ảnh trước (DL2) để dữ liệu train (khoảng 2 GB) nằm gọn trong page cache, và để dung lượng ổ 100 GB đủ dùng.
- **Độ chính xác số:** kernel deformable attention chỉ chạy float32/float64 (đã kiểm chứng trong mã nguồn kernel). Mặc định **train FP32**. L4 có hỗ trợ bf16/TF32, nhưng chỉ bật AMP khi đã ép kernel chạy FP32 và đo được là nhanh hơn; việc này không bắt buộc.
- **Môi trường Python:** image không có sẵn PyTorch. Dùng venv riêng với `torch 2.2.2 + cu121` (chạy được trên L4), vì code gắn với `transformers 4.37.2` và `lightning 2.1.3`. Tổ hợp torch 2.2.2 + transformers 4.37.2 đã chạy được model PDP trong bước phân tích. Lệnh cài đặt ở [GCP_TRAINING_GUIDE.md](GCP_TRAINING_GUIDE.md).
- **Kernel CUDA:** code gốc không bao giờ dùng được kernel (F11), nên hiện tại luôn chạy bản PyTorch chậm hơn.

### 3.3 Dung lượng lưu trữ (ước tính)

| Hạng mục | Dung lượng |
|---|---|
| File zip RPC tải từ Kaggle | 25,3 GB theo lần tải thật ngày 28/09 (lúc lập plan, API báo 15,9 GB). Đã xóa sau khi giải nén; khi cần ảnh sản phẩm đơn thì tải lại (khoảng 4 phút trên VM) |
| Ảnh quầy gốc, vuông, cạnh khoảng 1750–1890 px (`val2019` + `test2019`) | Khoảng 6 GB (trung bình khoảng 200 KB/ảnh, đo trên 14.194 tên file) |
| Ảnh quầy đã thu nhỏ về 800×800 | Khoảng 2 GB (ước tính) |
| Mỗi lần chạy 5 task | Khoảng 2 GB: trọng số cuối mỗi task khoảng 0,28 GB × 5, cộng file dự đoán, cộng một checkpoint resume tạm 0,52 GB |
| Khoảng 20 lần chạy | Khoảng 40 GB |

Tổng cộng khoảng 65 GB nếu giữ lại file zip, hoặc khoảng 50 GB nếu xóa zip sau khi giải nén. Ổ boot hiện còn trống khoảng 70 GB; ổ 200 GB đã trống hoàn toàn.

### 3.4 Ngân sách GPU (ước tính)

**Mốc để ước tính.** Theo paper Deformable DETR (Table 1), train 50 epoch trên COCO (khoảng 118 nghìn ảnh) mất 325 giờ GPU V100, tức khoảng 0,2 giây/ảnh; suy luận đạt 19 ảnh/giây. Ảnh 800×800 có khoảng 0,75 lần số pixel so với ảnh COCO, do đó:

| Loại bước | Ước tính (giây/ảnh) | Giải thích |
|---|---|---|
| Fine-tune toàn bộ mô hình (B1) | 0,15 | 1 lượt forward + backward |
| PDP task 1 | 0,2 | Thêm 1 lượt forward không tính gradient để lấy query |
| PDP task ≥ 2 | 0,3 | Thêm 2 lượt forward của teacher |
| Suy luận (2 lượt) | 0,1 | |

Tất cả giả định kernel chạy được. Nếu rơi về bản PyTorch thì chậm hơn đáng kể (chưa đo).

**Số ảnh mỗi task (ước tính theo xác suất, số thật in ra ở DL5):** task 1 khoảng 21,8 nghìn ảnh; mỗi task 25 SKU khoảng 12,6 nghìn ảnh.

| Cấu hình | Định nghĩa | Một lần chạy 5 task |
|---|---|---|
| **Chuẩn** | Tối đa 6.000 ảnh/task (lấy mẫu cố định, phân tầng theo mức), 6 epoch | Khoảng 15 giờ (task 1: 2 giờ; task 2–5: 12 giờ; đánh giá: 1 giờ) |
| **Đầy đủ** | Toàn bộ ảnh, 8 epoch (như paper) | Khoảng 45 giờ |

| Nhóm thí nghiệm (mục 7) | Giờ GPU (ước tính) |
|---|---|
| Pilot P1–P3 | Khoảng 5 |
| E0 (12 epoch) + FSA (cấu hình chuẩn) | Khoảng 11 + 3 |
| E1–E4 (cấu hình chuẩn) | Khoảng 55 |
| E5 | Khoảng 4 |
| Ablation A1–A9 (3 task, dùng lại task 1 khi có thể) | Khoảng 55 |
| **Tổng tối thiểu** | **Khoảng 135** |
| Tùy chọn: E4 cấu hình đầy đủ; thêm 2 seed cho E3/E4 | Khoảng +55; khoảng +60 |

**Số đo thật trên L4 (28/09/2026, `pdp/benchmark.py`, ảnh 800×800, có kernel CUDA):**

| Đo | Kết quả |
|---|---|
| Bước train PDP của task ≥ 2 (teacher 2 lượt + PPG + student 2 lượt + backward), batch 4 | **0,336 giây/ảnh**, bộ nhớ GPU cao nhất 6,2 GB (batch 2: 0,374 giây/ảnh, 4,5 GB) |
| Suy luận 2 lượt, batch 1 | 110 ms/ảnh ở 800 px; 87 ms/ảnh ở 640 px |
| Kernel deformable attention so với bản PyTorch | 0,07–0,10 ms so với 1,7–1,9 ms mỗi lượt forward |

Tính lại theo số đo: task 1 chưa đo, tạm ước khoảng 0,22 giây/ảnh vì không có teacher. Một lần chạy 5 task ở cấu hình chuẩn mất khoảng **17 giờ L4**: task 1 khoảng 2,2 giờ, task 2–5 khoảng 13,4 giờ, dự đoán khoảng 1,2 giờ. Cả plan tối thiểu (pilot, E0, FSA, DET, E1–E5, A1–A9) khoảng **165 giờ L4, tức khoảng 7 ngày chạy liên tục**; nếu chạy Spot thì tốn khoảng 1,9 triệu VND.

Một GPU chạy liên tục 24/7 được tối đa 168 giờ/tuần. Tính cả thời gian bị Spot thu hồi và thời gian sửa lỗi, thực tế nên dự trù 2–3 tuần chạy GPU. Chi phí quy ra tiền cho từng cấu hình: xem mục 3.5.

### 3.5 Chi phí theo cấu hình và lần chạy thử đầu tiên

**Giá Spot tại us-central1** (lấy từ Cloud Billing Catalog API ngày 28/09/2026, đơn vị VND):

| Hạng mục | Giá |
|---|---|
| GPU V100 / L4 / T4 | 34.942 / 8.765 / 5.465 mỗi giờ |
| CPU và RAM dòng N1 | 494,6 mỗi vCPU-giờ; 66,3 mỗi GiB-giờ |
| CPU và RAM dòng G2 | 391,0 mỗi vCPU-giờ; 45,8 mỗi GiB-giờ |
| Ổ `pd-standard` / `pd-balanced` / snapshot | 1.043 / 2.609 / 1.304 mỗi GiB-tháng |

**Các phương án cấu hình:**

| Phương án | Cấu hình | VND/giờ | Tốc độ so với V100 (ước tính) | Chi phí cho plan tối thiểu (khoảng 135 giờ V100) | Ổ đĩa mỗi tháng |
|---|---|---|---|---|---|
| A (VM cũ) | `n1-standard-8` + V100, giữ 2 ổ `pd-standard` | 40.888 | 1 | Khoảng 5,5 triệu (**vượt credit**) | 313 nghìn |
| A' | `n1-standard-4` + V100 | 37.915 | 1 | Khoảng 5,1 triệu (vượt credit) | 313 nghìn |
| **B (đề xuất)** | `g2-standard-4` (4 vCPU, 16 GB) + **L4 24 GB**, 1 ổ boot 150 GB `pd-balanced` | 11.062 | Khoảng 0,75 | Khoảng 2,0 triệu | 391 nghìn |
| C | `n1-standard-4` + T4 16 GB, giữ ổ cũ | 8.438 | Khoảng 0,28 | Khoảng 4,1 triệu, chạy lâu gấp khoảng 3,6 lần | 313 nghìn |

Ghi chú về các con số:

- Tỉ lệ tốc độ lấy từ benchmark ResNet công khai, không phải từ model của nhóm: V100 nhanh khoảng 3,6 lần T4; L4 nhanh khoảng 2,7 lần T4. Phải đo lại bằng model thật (xem cuối mục).
- Máy G2 (L4) **không dùng được ổ `pd-standard`**. Vì vậy phương án B cần tạo VM và ổ đĩa mới; sau khi chuyển xong thì xóa VM và 2 ổ cũ để không tốn 313 nghìn VND/tháng tiền ổ.
- Ổ `pd-balanced` 150 GB đọc được khoảng 180 MiB/s, nhanh gấp khoảng 7 lần ổ `pd-standard` 200 GB hiện tại. Image cần tối thiểu 100 GB và tự nó dùng 28 GB.
- Quota của project chỉ cho **1 GPU chạy cùng lúc** trên toàn tài khoản (V100, L4, T4 hoặc P100). Zone `us-central1-b` có đủ cả 3 loại GPU.
- 8 vCPU là thừa: ảnh đã thu nhỏ thì 4 vCPU đủ nạp dữ liệu cho 1 GPU.

**Lần chạy thử đầu tiên (để đánh giá tính khả thi).**

- Điều kiện trước: đã làm xong T0, DL1–DL6, F1–F12, R1/R2/R4, V1–V2. Các việc này chủ yếu viết code trên Mac, không tốn GPU.
- Trên VM cần:
  - cài môi trường và build kernel: khoảng 1 giờ;
  - tải và xử lý dữ liệu: khoảng 1 giờ;
  - smoke test và đo tốc độ: khoảng 0,5 giờ;
  - pilot: xem bảng dưới.
- Số giờ VM trong bảng đã nhân 1,5 để dự phòng sửa lỗi.

| Phạm vi | Giờ GPU tính toán (quy về V100) | A: giờ VM / chi phí | B: giờ VM / chi phí | C: giờ VM / chi phí |
|---|---|---|---|---|
| Tối thiểu: smoke test + P2 | Khoảng 1,3 | Khoảng 6 giờ / 230 nghìn | Khoảng 6,5 giờ / 70 nghìn | Khoảng 11 giờ / 90 nghìn |
| Đầy đủ pilot: P1 + P2 + P3 (gồm FSA) | Khoảng 5 | Khoảng 11 giờ / 460 nghìn | Khoảng 14 giờ / 150 nghìn | Khoảng 31 giờ / 260 nghìn |

Việc đầu tiên sau khi cài môi trường xong: chạy một benchmark tổng hợp khoảng 15 phút (ảnh ngẫu nhiên 800×800, chạy đủ 4 lượt forward như PDP ở task ≥ 2) để đo số giây/ảnh thật. Cách đọc kết quả:

- Về chi phí, phương án B còn rẻ hơn A' chừng nào L4 chưa chậm hơn V100 quá khoảng 3,4 lần (37.915 / 11.062).
- Nếu L4 chậm hơn từ 2 lần trở lên thì thời gian chạy cũng dài gấp đôi trở lên. Khi đó phải so với hạn dùng của credit (QĐ-4) để chọn giữa B và A'.

### 3.6 Hạn credit 24/10/2026 và chạy Spot

**Thời gian còn lại:** 26 ngày (tính từ 28/09). Khi credit hết hạn hoặc tiêu hết: nếu tài khoản billing đã nâng cấp lên trả phí (dùng được GPU thì thường là đã nâng cấp, nhưng nên kiểm tra trong Billing), phí sẽ tính vào phương thức thanh toán. Vì vậy phải tải kết quả về và **xóa VM và ổ trước 24/10**, hoặc chủ động chấp nhận trả tiền. Nên đặt budget alert trong Billing.

**Spot khác on-demand ở những điểm sau** (theo [tài liệu GCP về Spot](https://docs.cloud.google.com/compute/docs/instances/spot)):

| Điểm | Spot | Ảnh hưởng tới project |
|---|---|---|
| Tốc độ | Cùng loại máy, cùng GPU; tài liệu không nói Spot chạy chậm hơn | Không đổi |
| Bị thu hồi | Google có thể tắt VM bất cứ lúc nào, chỉ báo trước tối đa 30 giây; xác suất "thường thấp" nhưng thay đổi theo ngày và zone | Mất phần việc kể từ checkpoint gần nhất (≤ 30 phút với R1), cộng thời gian chờ bật lại |
| Tự bật lại | Không | Phải bật tay, hoặc dùng R5 |
| Thời gian chạy tối đa | Không giới hạn (khác với loại preemptible cũ bị giới hạn 24 giờ) | Chạy dài được |
| Còn máy hay không | Có lúc hết GPU Spot, không bật được | Có thể tạm chuyển về STANDARD (cùng lệnh `set-scheduling`) để không trễ hạn |
| Giá | Có thể đổi mỗi ngày; ngày 28/09 là khoảng 11.100 VND/giờ so với 18.400 VND/giờ của on-demand | Theo dõi Billing |

Dữ liệu thực tế của project: VM Spot cũ (V100, zone `us-central1-b`) **không bị thu hồi lần nào** trong nhật ký hoạt động từ 14/09 đến 28/09, có lần chạy liên tục khoảng 17 giờ. Mẫu này nhỏ và không phải L4, nên chỉ để tham khảo.

**Chi phí tới 24/10** (plan tối thiểu khoảng 180 giờ L4, cộng khoảng 25 giờ VM bật để cài đặt và sửa lỗi, cộng ổ `auto-cl` khoảng 226 nghìn):

| Cách chạy | Tổng chi phí | Còn dư so với 4 triệu |
|---|---|---|
| Toàn bộ on-demand | Khoảng 4,0 triệu | Gần như không còn (2 ổ của VM cũ đã xóa ngày 28/09 nên không còn khoản 270 nghìn tiền ổ) |
| On-demand khi cài đặt và pilot, **Spot cho các lần chạy chính** (đề xuất) | Khoảng 2,7 triệu | Khoảng 1,3 triệu (đủ cho **một** trong hai: E4 cấu hình đầy đủ, khoảng 0,8 triệu; hoặc thêm seed, khoảng 0,9 triệu) |

**Mốc gợi ý để kịp hạn** (thời gian GPU cho E0–E5 khoảng 4 ngày chạy liên tục trên L4; A1–A4 khoảng 1,5 ngày):

| Mốc | Việc | Chạy trên |
|---|---|---|
| Đến khoảng 06/10 | Code T0–R4 và V1–V2 (chủ yếu viết trên Mac, không tốn GPU); cài VM và chuẩn bị dữ liệu DL1–DL6 | VM on-demand, chỉ bật khi cần |
| Khoảng 07–08/10 | Smoke test, pilot, mốc G1; **test resume** (tự tắt VM giữa chừng) | On-demand |
| Khoảng 09/10 | Chuyển `auto-cl` sang Spot | — |
| Khoảng 09–19/10 | E0, FSA, E1–E5, mốc G2; sau đó A1–A4 (A5–A9 nếu còn thời gian) | Spot |
| Khoảng 20–22/10 | V5, V6; tải kết quả về; mốc G3 | Spot hoặc tắt VM |
| Trước 24/10 | Xóa VM và ổ, hoặc quyết định trả tiền để chạy tiếp | — |

Nếu bị trễ, cắt theo thứ tự: A5–A9, rồi E2, rồi dùng ablation 3 task. Giữ lại E0, E3, E4, E5 và A1–A4.

---

## 4. Dữ liệu

### 4.1 Nguồn

- Kaggle `diyer22/retail-product-checkout-dataset`, phiên bản 5, file zip 25,3 GB (tải thật 28/09), giấy phép **CC BY-NC-SA 4.0** (chỉ dùng phi thương mại, không đưa dữ liệu lên repo).
- Dữ liệu gồm 200 SKU thuộc 17 nhóm hàng.
- Ảnh quầy vuông, cạnh không cố định: lấy mẫu 1.300 ảnh (qua bản mirror trên HuggingFace, 28/09) thấy cạnh từ khoảng 1750 đến 1890 px, không phải đúng 1800. DL2 co từng ảnh theo cạnh riêng của nó. Có 3 mức độ đông:

  | Mức | Số SKU/ảnh | Số vật/ảnh |
  |---|---|---|
  | easy | 3–5 | 3–10 |
  | medium | 5–8 | 10–15 |
  | hard | 8–10 | 15–20 |

- `val2019` có 6.000 ảnh (2.000 mỗi mức); `test2019` có 24.000 ảnh (8.000 mỗi mức). Trung bình 12,26 vật/ảnh.
- Annotation theo định dạng COCO, `category_id` từ 1 đến 200. `rpctool` đọc trường `level` của ảnh, nên có thể trường này tồn tại; DL1 sẽ kiểm tra.
- Cần tài khoản Kaggle (API token) để tải. **Không commit file `kaggle.json`.**
- Đã kiểm tra các nguồn khác (28/09):
  - Bản mirror trên HuggingFace (`benjamintli/retail-product-checkout`, giống hệt `SAxSHADOW/retail-product-checkout`) tải được không cần đăng nhập, nhưng **chỉ có ảnh, bbox và nhãn**: không có tên file gốc, không có `level`, không có id ảnh.
  - Thiếu tên file thì không chia tập theo nhóm ảnh chụp liên tiếp được (mục 4.2); thiếu `level` thì không phân tầng theo mức độ đông được. Vì vậy **dùng bản gốc trên Kaggle**.
  - Baidu Drive (link trên trang RPC) khó dùng từ ngoài Trung Quốc.

### 4.2 Phát hiện: các nhóm ảnh chụp liên tiếp (cần DL1 xác nhận)

Tôi phân tích 14.194/24.000 tên file của `test2019`, dạng `YYYYMMDD-HH-MM-SS-<hậu tố>.jpg`:

- Ảnh đi theo cụm liên tiếp có cùng hậu tố, chủ yếu cụm 3 ảnh (3.377 cụm).
- Trong một cụm, các ảnh cách nhau trung vị khoảng **11 giây**; giữa hai cụm khác nhau là khoảng **33 giây**.
- Có 1.705/2.614 hậu tố xuất hiện lại ở thời điểm hoặc ngày khác.

Nhiều khả năng đây là cùng một bộ hàng (hoặc cùng một danh sách mua) được chụp nhiều lần. Nếu chia ngẫu nhiên từng ảnh, các ảnh gần giống nhau sẽ rơi vào cả train lẫn test, và kết quả bị thổi phồng. Vì vậy:

- **Chia tập theo nhóm.** Tạm định nghĩa một nhóm là các ảnh cùng hậu tố, gộp chung cho cả `val2019` và `test2019`.
- DL1 xác nhận bằng annotation: các ảnh cùng hậu tố có cùng tập (SKU, số lượng) không? Hậu tố có trùng giữa hai tập `val2019` và `test2019` không?
- Nếu có ảnh khác hậu tố nhưng trùng danh sách hàng thì gộp chúng vào cùng nhóm.

### 4.3 Chia tập

Sau khi gộp `val2019` và `test2019` thành một kho chung (đánh lại `image_id` nếu trùng), chia theo nhóm, phân tầng theo mức độ đông, seed 0:

| Tập | Nguồn | Khoảng số ảnh | Dùng để |
|---|---|---|---|
| **test** | Chỉ các nhóm lấy từ `test2019` | ≈ 6.000 (≈ 2.000 mỗi mức) | Đánh giá cuối mỗi task. **Khóa suốt đồ án** |
| **val** | Chỉ các nhóm lấy từ `test2019` | ≈ 1.500 (≈ 500 mỗi mức) | Theo dõi khi train, chọn ngưỡng đếm, ra quyết định ở các mốc G |
| **train** | Phần còn lại (`val2019` + phần còn lại của `test2019`) | ≈ 22.500 | Chia vào các task |
| train_pilot | Tập con của train, lấy theo nhóm | 3.000 | Pilot |

Lý do chọn như vậy:

- Test chỉ lấy từ `test2019` để sau này kết quả "chỉ train bằng ảnh ghép" vẫn gần với giao thức gốc của RPC.
- 6.000 ảnh test chứa khoảng 73 nghìn vật, trung bình khoảng 370 vật mỗi SKU; sai số chuẩn của cAcc khoảng ±0,65 điểm.
- Nếu một nhóm có ảnh ở cả `val2019` và `test2019` thì cả nhóm đi vào train.

Điều kiện chấp nhận:

- Không nhóm nào nằm ở 2 tập.
- Trong test, mỗi SKU có ít nhất 50 vật; trong val, mỗi SKU có ít nhất 15 vật (nếu không đạt thì đổi seed và ghi lại seed đã dùng).

### 4.4 Chia lớp và cách đánh số nhãn

- File `configs/tasks_100-4x25_seed0.json` gồm:
  - thứ tự 200 SKU;
  - task của từng SKU;
  - bảng đổi `category_id` RPC (1..200) sang nhãn nội bộ (0..199, theo thứ tự task);
  - 24 slot dự phòng (200..223) cho task 6.
- Cách chia: trong mỗi nhóm hàng, trộn SKU (seed 0), rồi rải xen kẽ các nhóm hàng. Kết quả là task 1 chiếm khoảng một nửa mỗi nhóm hàng, phần còn lại chia đều cho task 2–5.
- Classifier có 225 đầu ra. Slot 224 (`n_classes - 1` trong code gốc) **không bao giờ là nhãn đích**: loss focal dạng sigmoid biểu diễn "không có vật" bằng vector toàn 0, và code gốc che slot này khi train, bỏ nó khi đánh giá (đã kiểm tra trong `DeformableDetrLoss.loss_labels`).

### 4.5 File JSON cho từng task (định dạng code PDP đọc được)

| File | Nội dung |
|---|---|
| `train_task_t.json` | Ảnh train có ít nhất 1 SKU của task t; chỉ giữ nhãn của task t; có trường `area` (bộ tiền xử lý của code **bắt buộc** phải có `area`) |
| `train_task_t_capped.json` | Tối đa 6.000 ảnh (cấu hình chuẩn) |
| `val_task_t.json` | Tương tự trên val (dùng theo dõi mAP@C khi train) |
| `val_full.json`, `test_full.json` | Giữ **toàn bộ** nhãn 200 SKU; bộ đánh giá tự lọc theo task |
| `*_gt_full.json` của train | Nhãn đầy đủ của ảnh train, **chỉ dùng cho V4** (đo chất lượng nhãn giả), không bao giờ đưa vào train |

Hàm sinh JSON nhận **nhiều nguồn ảnh**, mỗi nguồn có nhãn đánh dấu (ảnh thật; sau này thêm ảnh ghép), để giai đoạn 2 không phải viết lại.

---

## 5. Cấu trúc repo và luồng chạy

```
AutoCheckout-CL/
├── pdp/                 # code PDP_IOD (commit đầu = bản gốc 7702d91, các commit sau = từng bản sửa; nguồn ghi ở pdp/UPSTREAM.md)
├── autocheckout/        # thư viện dùng chung: TaskConfig, định dạng file dự đoán, ghi file an toàn, chỉ số
├── tools/               # audit_rpc, resize, make_split, make_task_config, make_task_json, predict, eval_cl, eval_count, ppg_audit, summarize
├── baselines/           # adapt.py (B1), retrieval (B3)
├── configs/             # split, task, file tham số của từng thí nghiệm
├── scripts/             # setup_vm.sh, download_rpc.sh, prepare_data.sh, run_exp.sh
├── tests/               # unit test (CPU) + smoke test
├── docs/formats.md      # định dạng file và thư mục dùng chung giữa các phần code
├── results/             # kết quả tải về từ VM (chỉ số, log; không có checkpoint)
├── IMPLEMENTATION_PLAN.md, GCP_TRAINING_GUIDE.md, PROGRESS.md
└── (không commit) data/, runs/, *.pth, *.ckpt, *.npz, kaggle.json, .venv/
```

Lưu ý kỹ thuật:

- Code PDP dùng import kiểu top-level (`import utils`, `from datasets...`, `from models...`), nên phải chạy với thư mục `pdp/` là thư mục làm việc (hoặc đứng đầu `sys.path`). Vì thư mục `pdp/datasets` trùng tên gói `datasets` của HuggingFace, **không cài gói `datasets`** vào venv.
- `autocheckout` được cài ở chế độ editable (`pip install -e .`) để cả `pdp/` lẫn `tools/` import được.
- Định dạng dữ liệu, file dự đoán và thư mục run được mô tả ở [docs/formats.md](../data_preprocessing/formats.md).

Luồng xử lý **một task t** (sau khi sửa):

1. Tạo model; `set_task_id(t-1)`; nạp `task_final.pth` của task t-1 (với t = 1: nạp checkpoint COCO hoặc checkpoint FSA).
2. Khởi tạo phần prompt riêng của task t (F2).
3. Nạp teacher từ `task_final.pth` của task t-1 (F5, chỉ khi t ≥ 2).
4. Train với resume (R1). Mỗi batch:
   - teacher 2 lượt, rồi PPG sinh nhãn giả;
   - student lượt 1 (không tính gradient) để lấy query, lượt 2 tính loss.
5. Epoch cuối: cập nhật prototype cho lớp mới (F7).
6. Lưu `task_final.pth`.
7. V1: dự đoán trên val và test. V2/V3: tính chỉ số.

---

## 6. Danh sách công việc

Mỗi việc gồm: việc cần làm, đầu ra, và **tiêu chí nghiệm thu** (ưu tiên test tự động).

### 6.1 Giai đoạn 0: nền tảng

| ID | Việc | Nghiệm thu |
|---|---|---|
| T0.1 | Khởi tạo git cho thư mục này; đẩy lên `github.com/AnNguyen05092004/AutoCheckout-CL` (public, QĐ-7). Commit đầu: tài liệu. Commit thứ hai: code PDP_IOD gốc, không sửa gì (ghi rõ nguồn và commit `7702d91`). Tạo `.gitignore` cho `data/ runs/ *.pth *.ckpt kaggle.json .venv/` | Các file trong `pdp/` giống hệt blob của upstream; `git log` tách được "code gốc" và "bản sửa" |
| T0.2 | File phiên bản thư viện (`requirements.txt` dùng chung, `requirements-dev.txt` cho test; PyTorch cài riêng theo máy); môi trường CPU `.venv` trên Mac để chạy unit test | `pytest` chạy được trên Mac |
| T0.3 | Trên VM: tạo venv, chạy `nvidia-smi`, kiểm tra `torch.cuda` thấy L4. **Thử build kernel** bằng một script nhỏ gọi `torch.utils.cpp_extension.load` trên thư mục `transformers/kernels/deformable_detr`, độc lập với code PDP; việc nối kernel vào code PDP làm ở F11 | Import được module kernel; thời gian build được ghi lại |
| T0.4 | Tạo `/data/rpc` và `/data/runs` trên ổ boot (lệnh ở mục 4 của hướng dẫn VM). Nếu thiếu chỗ thì tăng ổ lên 150 GB | `df -h` cho thấy đủ chỗ theo mục 3.3 |
| T0.5 | Tải RPC bằng Kaggle API; chỉ giải nén `val2019/`, `test2019/` và 2 file JSON tương ứng | Số file ảnh là 6.000 và 24.000; JSON đọc được |

### 6.2 Giai đoạn 1: dữ liệu

| ID | Việc | Nghiệm thu |
|---|---|---|
| DL1 | `tools/audit_rpc.py`: số ảnh và số vật theo mức; số vật theo SKU; kích thước ảnh gốc (mong đợi ảnh vuông, cạnh khoảng 1750–1890 px); có đủ trường `area`, `iscrowd`, `level`, `supercategory` không; `image_id` có trùng giữa val và test không; **xác nhận nhóm ảnh** (mục 4.2); md5 trùng | Báo cáo audit lưu trong repo; chốt khóa nhóm |
| DL2 | `tools/resize.py`: thu nhỏ về 800×800 vào một thư mục duy nhất `checkout_800/` (co bbox và `area` theo; ghi lại kích thước gốc) | Vẽ bbox ngẫu nhiên trên 20 ảnh để kiểm tra; tổng số vật không đổi |
| DL3 | `tools/make_split.py`: chia theo nhóm (mục 4.3); ghi seed, danh sách ảnh val/test và md5 các file split vào `configs/splits/` trong repo (md5 của `test_full.json` nằm trong `manifest.json` của DL5, bản sao cũng được commit) | Các điều kiện ở mục 4.3 đều đạt; chạy lại ra đúng cùng md5 |
| DL4 | `tools/make_task_config.py`: chia lớp (mục 4.4) | Mỗi task có số SKU đúng; phân bố nhóm hàng được in ra |
| DL5 | `tools/make_task_json.py`: sinh các file ở mục 4.5 (nhận nhiều nguồn ảnh) | In số ảnh và số vật mỗi task; JSON đọc được bằng `CocoDetection` của code; unit test trên JSON giả nhỏ |
| DL6 | Tập pilot: `train_pilot` 3.000 ảnh (lấy theo nhóm), cho 2 task 100+25 | Như DL5 |

### 6.3 Giai đoạn 2: sửa code PDP cho đúng paper (F1–F12)

Nguyên tắc:

- Mỗi bản sửa là **một commit riêng, kèm một test**.
- Không sửa lan sang phần không liên quan.
- Mọi thay đổi hành vi đều có cờ bật/tắt khi cần so sánh.

Các file được dẫn chiếu (trong thư mục gốc hoặc `models/` của PDP_IOD): [prompt.py](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/models/prompt.py), [engine.py](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/engine.py), [main.py](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/main.py), [modeling_deformable_detr.py](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/models/modeling_deformable_detr.py), [image_processing_deformable_detr.py](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/models/image_processing_deformable_detr.py).

**F1: Cấu hình task cho RPC** (bắt buộc để chạy được)
- *Hiện trạng:* `task_info_coco` viết cứng 80 lớp COCO ([create_coco_instance.py#L8-L55](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/datasets/create_coco_instance.py#L8-L55)); `main.py#L224-L227` gọi hàm này; `task_num_classes` chỉ tính cho `n_tasks` task đang chạy.
- *Sửa:*
  - `task_info_rpc(file cấu hình DL4)` trả về `task_map` và `label2name`, gồm cả task 6 (24 slot);
  - `--n_classes 225`;
  - `task_num_classes` lấy từ **toàn bộ** file cấu hình để pool riêng có đủ 224 slot, trong khi `--n_tasks` chỉ là số task thực chạy.
- *Nghiệm thu:* `task_map` khớp file cấu hình; tổng số slot là 224; offset liên tục.

**F2: Pool riêng: kích thước và khởi tạo prompt của task mới** (lỗi nghiêm trọng, đã kiểm chứng, có từ MD-DETR)
- *Hiện trạng:*
  - `total_classes = 80` ([prompt.py#L18](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/models/prompt.py#L18)), `pool_sizes` tính theo tỉ lệ với 80 (prompt.py#L22-L24), `private_size = 80` (prompt.py#L60).
  - `gram_schmidt` chỉ chạy lúc tạo model với `task_count = 0` (prompt.py#L44-L55), nên phần prompt của task ≥ 2 **toàn số 0 và có gradient bằng 0**: không bao giờ học.
  - Với 200 lớp: task đầu 100 lớp thì crash; chia 4×50 thì task 3–4 không có prompt nào.
- *Sửa:*
  - `private_size` = tổng số slot (224); `pool_sizes` = số lớp từng task.
  - Thêm hàm khởi tạo phần prompt của task hiện tại (gọi `gram_schmidt` cho p, k, a của cả 6 layer).
  - Gọi hàm này trong `main.py` **sau** `set_task_id` (main.py#L280-L283), **sau** khi nạp checkpoint task trước (main.py#L285-L305), và **trước** khi tạo optimizer.
  - Khi resume giữa task (R1), trọng số đã train được nạp đè lên sau bước này, nên không bị mất.
- *Nghiệm thu (CPU):*
  1. Phần prompt của task mới khác 0 và có gradient.
  2. Phần prompt của task cũ không đổi sau 1 bước optimizer.
  3. |cos| giữa key mới và key cũ nhỏ hơn 1e-2.
  4. Cấu hình 100+4×25 với 224 slot không bị lỗi chỉ số.
  - Bước phân tích đã có bản chạy thử của test này và đạt cả 4 điểm với 200 slot; cần chạy lại với 224 slot.

**F3: L_DDL** (đã kiểm chứng: hiện không chạy)
- *Hiện trạng:*
  - `use_ddl_loss = False` (prompt.py#L69); loss chỉ được tính trong `if train and self.ortho_mu > 0` (prompt.py#L253), mà `ortho_mu = 0`.
  - Decoder bỏ luôn loss mà module prompt trả về: `p_list, _, output = ...` ([modeling_deformable_detr.py#L1393](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/models/modeling_deformable_detr.py#L1393)).
- *Sửa:*
  - L_DDL (công thức 9–10 của paper) chỉ phụ thuộc tham số prompt, không phụ thuộc ảnh đầu vào. Vì vậy tính **một lần mỗi bước train** ngay từ tham số của module prompt: với mỗi layer, lấy góc giữa từng prompt chung và từng prompt riêng đang dùng (các task trước bị detach như code gốc), rồi lấy trung bình 6 layer. Cách này cho cùng giá trị với việc tính trong decoder, nhưng không phải sửa đường trả về của decoder và không bị tính thừa ở lượt forward không gradient hay ở teacher. Paper không nói cách gộp các layer; ghi chú rõ trong code.
  - θ = 90°, λ = 0,15 (tham số `--ddl_lambda`), không phụ thuộc `ortho_mu`.
  - Trainer cộng λ·L_DDL vào loss tổng và ghi vào `loss_dict['loss_ddl']`; thêm khóa này vào `short_map` (engine.py#L432), nếu không phần log sẽ báo `KeyError`.
- *Nghiệm thu:* λ = 0 và λ > 0 cho loss khác nhau; pool chung nhận gradient từ L_DDL; giá trị khớp cách tính thủ công trên tensor nhỏ.

**F4: L_Q có gradient** (đã kiểm chứng, có từ MD-DETR)
- *Hiện trạng:* `query_tf` và cross-entropy nằm trong `torch.no_grad()` (engine.py#L314-L332), nên L_Q không tạo gradient.
- *Sửa:* vẫn chạy lượt forward thứ nhất trong `no_grad`, nhưng tính `query_tf(query)` và cross-entropy **bên ngoài** `no_grad`.
- *Nghiệm thu:* `query_loss.requires_grad` là True; `query_tf` nhận gradient khác 0 từ riêng L_Q.

**F5: Teacher** (sai khác với paper)
- *Hiện trạng:*
  - Teacher được nạp ở bước train đầu tiên, với đường dẫn và tên file viết cứng `checkpoint07.pth` (engine.py#L267-L299).
  - Thiếu file thì chỉ in cảnh báo rồi âm thầm tắt cả chưng cất lẫn PPG.
  - Teacher chạy 1 lượt, **không truyền query**, nên decoder không có prompt (engine.py#L304). Điều này khác với cách mô hình đã được train.
- *Sửa:*
  - Thêm tham số `--prev_ckpt`; nạp teacher một lần lúc bắt đầu train; báo lỗi nếu không có file.
  - Đặt `task_count` của teacher = t-2, để teacher chỉ dùng prompt của các task cũ.
  - Teacher suy luận 2 lượt giống Evaluator; đặc trưng cho PPG lấy từ lượt thứ 2.
- *Nghiệm thu:* teacher và student cùng trọng số, cùng `task_count` thì cho cùng đầu ra; thiếu file thì báo lỗi.

**F6: Sinh nhãn giả (PPG)** (lỗi đã kiểm chứng)
- *Hiện trạng:*
  - `post_process` chọn top-k theo cặp (query, lớp) với k = `bg_thres_topk` = 5 ([image_processing_deformable_detr.py#L1341-L1402](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/models/image_processing_deformable_detr.py#L1341-L1402), [run.sh#L28](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/run.sh#L28)), sau đó mới lọc lớp cũ. Ảnh RPC có trung bình 12 vật, nên 5 ứng viên là quá ít.
  - `labels <= PREV` (engine.py#L210) nhận nhầm cả lớp đầu tiên của task hiện tại.
  - Đặc trưng của ứng viên được lấy theo **vị trí trong danh sách top-k**, không theo chỉ số query (engine.py#L219-L227).
- *Sửa:*
  - Che logit mọi lớp ≥ PREV trước khi chọn ứng viên.
  - Mỗi query chỉ giữ 1 nhãn (lớp có score cao nhất).
  - Lấy top-k **query** (k = 50, là tham số) và trả về chỉ số query.
  - Đổi `<=` thành `<`.
  - Lấy đặc trưng bằng chỉ số query thật.
  - Giữ nguyên ngưỡng theo paper: τh = 0,5, τl = 0,2, θs = 0,5. Giữ hộp ở dạng cxcywh chuẩn hóa như code gốc.
- *Nghiệm thu:* với đầu ra giả lập biết trước, đặc trưng lấy ra đúng hàng query; không có nhãn giả nào ≥ PREV; một query không sinh 2 nhãn.

**F7: Prototype**
- *Hiện trạng:* code lấy mọi query được ghép Hungarian với GT của lớp mới, kể cả query phân loại sai (engine.py#L342-L376). Paper nói chỉ lấy vật "correctly classified".
- *Sửa:* chỉ đưa query vào bộ nhớ khi lớp có score cao nhất trùng lớp GT. Giữ nguyên: chỉ cập nhật ở epoch cuối, mỗi lớp tối đa 100 vector, prototype là trung bình.
- *Nghiệm thu:* query phân loại sai không vào bộ nhớ; hết task, mọi lớp mới đều có prototype (nếu thiếu thì in ra danh sách lớp thiếu).

**F8: Tham số hóa epoch, đường dẫn, checkpoint**
- *Hiện trạng:*
  - Số epoch bị ép cứng bằng 8 (main.py#L244-L247).
  - Đường dẫn và tên checkpoint viết cứng (engine.py#L271-L278, `run.sh`).
  - Checkpoint đầy đủ 0,52 GiB được lưu mỗi epoch (engine.py#L456-L480, engine.py#L521-L540).
  - Callback Lightning (main.py#L235) ghi mọi task vào chung một thư mục.
- *Sửa:*
  - Số epoch và đường dẫn thành tham số.
  - Cuối task lưu `task_final.pth` gồm trọng số, prototype và bộ nhớ prototype, không có optimizer (khoảng 0,28 GiB). File này dùng làm teacher và để khởi tạo task sau.
  - Checkpoint dùng để resume do R1 quản lý.
- *Nghiệm thu:* đổi số epoch vẫn chạy được và PPG vẫn bật (teacher nạp được).

**F9: Validate và đánh giá**
- *Hiện trạng:*
  - Validation trong lúc train đi qua `common_step`, nên chạy cả teacher (engine.py#L495-L519).
  - Sau mỗi task, code chạy 3 lần suy luận riêng cho C, P, A (main.py#L315-L369).
- *Sửa:*
  - Validation trong lúc train dùng `eval_mode` (không có teacher) trên val của task hiện tại.
  - Cuối task gọi V1: suy luận **một lần** trên val và test, lưu dự đoán. Các chỉ số được tính sau bằng V2/V3.
- *Nghiệm thu:* trên cùng tập ảnh con như code gốc, V2 cho mAP khớp với cách tính 3 lần của code gốc (lệch ≤ 0,1 điểm).

**F10: Báo cáo tham số được train**
- *Hiện trạng:*
  - Code đóng băng tham số theo tên (engine.py#L566-L577).
  - Optimizer có 2 nhóm (engine.py#L587-L624): `class_embed` và `prompts` dùng lr 1e-4. Nhóm lr 1e-5 gồm `input_proj`, `query_position_embeddings`, `reference_points`, `level_embed`, `bbox_embed`. Các tham số này dùng chung cho mọi task, nên cũng có thể gây quên.
- *Sửa:* ghi vào log mỗi task bảng tổng hợp số tham số được train theo nhóm. Không đổi hành vi.
- *Nghiệm thu:* cấu hình mặc định (225 lớp, 224 slot) cho 69,12M tham số tổng và **35,00M** tham số được train (đã tính ở bước lập plan).
- *Ghi chú (khác Deformable DETR gốc, giữ nguyên để giống code PDP):* model được tạo từ `DeformableDetrConfig()` mặc định của HuggingFace, nên **không có loss phụ ở các layer decoder** (`auxiliary_loss=False`), chi phí ghép cặp của lớp là 1 và trọng số `loss_ce` là 1; các tham số `--set_cost_*`, `--*_loss_coef` trong `main.py` không được dùng. B1 (E0, FSA) dùng cùng cấu hình này để các thí nghiệm so sánh được với nhau.

**F11: Kernel CUDA** (đã kiểm chứng: code gốc không bao giờ dùng kernel)
- *Hiện trạng:*
  - [load_custom.py#L23](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/models/load_custom.py#L23) tìm kernel ở `<thư mục cha của repo>/kernels/deformable_detr`, là đường dẫn không tồn tại.
  - Lỗi nạp kernel chỉ ghi warning (modeling_deformable_detr.py#L56-L65), và mỗi lần gọi đều rơi về bản PyTorch nhờ `try/except` (modeling_deformable_detr.py#L695-L711).
  - Kernel chỉ hỗ trợ float32/float64 (`AT_DISPATCH_FLOATING_TYPES`), nên khi chạy FP16 cũng rơi về bản PyTorch.
- *Sửa:*
  - Trỏ tới thư mục `transformers/kernels/deformable_detr` của gói đã cài (có sẵn trong bản 4.37.2). Khi build đặt `TORCH_CUDA_ARCH_LIST=8.9` (L4).
  - Ghi log rõ đang dùng kernel hay bản PyTorch.
  - Thêm tham số `--require_kernel` để báo lỗi thay vì âm thầm chuyển sang bản PyTorch.
- *Nghiệm thu (trên VM):* log báo dùng kernel; đầu ra của kernel và bản PyTorch trên cùng input lệch ≤ 1e-4; đo tốc độ của cả hai.

**F12: Xáo trộn dữ liệu train** (đã kiểm chứng)
- *Hiện trạng:* DataLoader train không có `shuffle` ([main.py#L267-L268](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/main.py#L267-L268)).
  - Tác giả chạy 2 GPU nên Lightning tự thêm `DistributedSampler` có xáo trộn.
  - Với 1 GPU, Lightning 2.1.3 dùng `SequentialSampler`: **mọi epoch có cùng một thứ tự**. Đã chạy thử để xác nhận.
- *Sửa:* thêm `shuffle=True` cho DataLoader train (thứ tự vẫn tái lập được nhờ `seed_everything`).
- *Nghiệm thu:* hai epoch liên tiếp có thứ tự khác nhau; cùng seed thì chạy lại ra cùng thứ tự.

**F13: Khởi tạo prior cho đầu phân loại** (phát hiện khi chạy pilot trên L4, 28/09)
- *Hiện trạng:*
  - Code đặt bias của classifier theo prior của focal loss (p = 0,01) trong `DeformableDetrForObjectDetection.__init__`. Nhưng hàm khởi tạo trọng số của HuggingFace chạy **sau đó** và đặt mọi bias `Linear` về 0: `post_init()` khi tạo model mới, và bước khởi tạo lại lớp lệch kích thước khi nạp checkpoint COCO 91 lớp vào model 225 lớp.
  - Hậu quả: mọi lớp bắt đầu ở p = 0,5 trên mọi query. `loss_ce` khởi đầu khoảng 730 (bình thường vài đơn vị); model coi gần như cả 300 query là có vật.
  - Với AdamW, bias chỉ dịch khoảng lr = 1e-4 mỗi bước, nên trong khoảng 1.100 bước của cấu hình chuẩn nó không thể tự về gần −4,6.
- *Sửa:* đặt lại bias = −log(0,99/0,01) khi đầu phân loại không được nạp nguyên từ checkpoint (model mới, hoặc số lớp khác checkpoint). Cờ `--prior_init_classifier 0` giữ hành vi gốc (P1).
- *Nghiệm thu:* model mới và checkpoint khác số lớp thì bias = prior; checkpoint cùng số lớp (FSA) giữ nguyên bias đã train (`tests/test_pdp_f13_prior_init.py`).

### 6.4 Giai đoạn 3: hạ tầng chạy

| ID | Việc | Nghiệm thu |
|---|---|---|
| R1 | **Resume khi Spot bị thu hồi.** Dùng `ModelCheckpoint` riêng cho từng task (`save_last=True`, lưu mỗi N bước ≈ 30 phút và cuối epoch). **Ghi checkpoint an toàn:** ghi ra file tạm rồi đổi tên, giữ lại bản trước, vì Spot có thể tắt máy đúng lúc đang ghi (đã kiểm tra: hàm `_atomic_save` của Lightning 2.1.3 ghi thẳng vào file đích nên **không** an toàn; cần một `CheckpointIO` riêng); gọi `fit(ckpt_path=last)` nếu đã có checkpoint. Dùng `on_save_checkpoint`/`on_load_checkpoint` để lưu thêm bộ nhớ prototype và `batch_counter`. Trả scheduler từ `configure_optimizers`, thay cho việc gọi tay ở engine.py#L457. Teacher không cần nằm trong checkpoint vì nạp lại được từ `task_final.pth` của task trước | Dừng tiến trình giữa epoch 2 rồi chạy lại: tiếp tục từ checkpoint gần nhất; epoch, lr và prototype đều đúng. Resume giữa epoch không tái lập được chính xác thứ tự dữ liệu; ghi chú điều này |
| R2 | `scripts/run_exp.sh configs/exp/<tên>.sh [--shutdown]`, chạy tuần tự các task (file cấu hình là bash, đặt `EXP`, `N_TASKS`, `ARGS`, tùy chọn `START_TASK`, `REUSE_TASK1`, `SKIP_EVAL`). Task đã có `task_final.pth` và dự đoán thì bỏ qua; có `last.ckpt` thì resume. Sau mỗi task chạy V1–V3. Khi kết thúc (**kể cả khi lỗi**) và có `--shutdown` thì tắt VM | Chạy lại lệnh sau khi đã xong thì không train lại; xóa dự đoán của task 3 thì chỉ chạy lại dự đoán task 3 |
| R3 | Lưu vết mỗi lần chạy: `args.json`, git hash + diff, phiên bản thư viện, md5 của `test_full.json`, log, thời gian mỗi task, bộ nhớ GPU cao nhất. Thư mục `runs/<thí nghiệm>/task_<t>/` | Từ thư mục run biết chính xác code, cấu hình và dữ liệu đã dùng |
| R5 | *(Tùy chọn, nên có khi chạy Spot)* Tự bật lại khi bị thu hồi: một script trên Mac, cứ khoảng 10 phút kiểm tra nhật ký `compute.instances.preempted`; nếu VM vừa bị thu hồi thì `start` lại. Một dịch vụ khởi động trên VM tự chạy lại R2 nếu còn thí nghiệm dở. Không bật lại khi VM tắt do `--shutdown` | Giả lập bằng cách tự tắt VM giữa chừng: job tự chạy tiếp mà không cần thao tác tay |
| R4 | Smoke test. **CPU (Mac):** 2 task × 1 epoch × 20 ảnh, trọng số ngẫu nhiên. **GPU (VM):** 200 ảnh thật, có kernel; đo giây/ảnh và bộ nhớ | Chạy hết: lưu/nạp checkpoint, có prototype, dự đoán, chỉ số |

### 6.5 Giai đoạn 4: đánh giá

| ID | Việc | Nghiệm thu |
|---|---|---|
| V1 | Lõi trong `pdp/inference.py`, dùng chung cho validation, dự đoán cuối task và `main.py --predict_only 1` (thay cho `tools/predict.py`): suy luận 2 lượt; che các lớp chưa học và slot 224; giữ **top-100 cặp (query, lớp) mỗi ảnh như hậu xử lý gốc**, kèm chỉ số query; lưu `pred_<split>.npz` theo [docs/formats.md](../data_preprocessing/formats.md) (nhãn model; đổi sang `category_id` RPC bằng `TaskConfig`). mAP (V2) dùng toàn bộ các cặp để so được với paper; đếm (V3) và demo lấy **lớp cao nhất của mỗi query** (`top1_per_query`) | Số dòng/ảnh ≤ 100; không có nhãn ≥ số lớp đã học; mAP tính từ file khớp với cách tính của code gốc (F9) |
| V2 | `tools/eval_cl.py`. **M1 (giao thức paper):** mAP@C/P/A, mỗi chỉ số tính trên tập ảnh con chứa lớp tương ứng. **M2 (thực tế cửa hàng):** dùng toàn bộ test, bỏ các dự đoán chồng (IoU ≥ 0,5) lên SKU chưa học, vì ở thời điểm task t cửa hàng chưa bán các SKU đó. AP50 là chỉ số chính, kèm AP50:95; tách theo easy/medium/hard; ma trận (sau task t × nhóm lớp g); độ quên | M1 khớp code gốc (F9); unit test trên dữ liệu giả có đáp án |
| V3 | `tools/eval_count.py`: **chọn ngưỡng trên val** (tối đa cAcc) rồi áp dụng cho test. Tính cAcc, ACD, mCCD, mCIoU (tự cài theo công thức của `rpctool`, đối chiếu với `rpctool` trên 1 file). Số của `rpctool` (ngưỡng dò ngay trên tập đang đánh giá, tức "nhìn đáp án") chỉ ghi kèm để so với leaderboard. Với các task giữa chừng: cAcc chỉ tính trên SKU đã học | Kết quả tự cài và `rpctool` trùng nhau khi dùng cùng ngưỡng |
| V4 | `tools/ppg_audit.py`: chạy teacher + PPG trên 1.000 ảnh train của task t có nhãn đầy đủ. Đo precision/recall của nhãn giả theo từng nhánh (tin cậy cao / qua prototype) và theo nhóm hàng; ma trận nhầm giữa các SKU cùng nhóm | Có bảng cho pilot và cho E4 |
| V5 | Chỉ số hệ thống: thời gian train mỗi task; dung lượng tăng thêm mỗi task (prompt 72 KiB/lớp, bộ nhớ prototype 100 KiB/lớp); độ trễ suy luận trên GPU và CPU ở 640 và 800 | Có bảng |
| V6 | Tổng hợp: bảng `.md`/`.csv`, biểu đồ mAP theo task và độ quên | Sinh lại được bằng một lệnh |

### 6.6 Giai đoạn 5: cải tiến cho RPC (mỗi mục có cờ bật/tắt để ablation)

| ID | Cải tiến | Mặc định trong E4 | Vì sao / rủi ro |
|---|---|---|---|
| I1 | **Thích nghi ở task 1 (FSA):** B1 fine-tune toàn bộ mô hình trên dữ liệu task 1, lưu dạng HF (`save_pretrained` kèm processor), rồi PDP task 1 dùng `--repo_name` trỏ tới đó và đóng băng như cũ | Bật | Đặc trưng COCO đóng băng khó phân biệt các SKU gần giống nhau. Rủi ro: SKU tương lai có trong ảnh task 1 bị học thành nền. P3 sẽ kiểm tra |
| I2 | Augmentation (chỉ khi train): xoay bội số 90°, đổi màu nhẹ, đa tỉ lệ (`shortest_edge` ngẫu nhiên 640–800). **Mặc định không lật ảnh**: lật một chiều tạo ra bao bì có chữ và logo bị ngược gương, không bao giờ gặp trên quầy thật, trong khi RPC có nhiều SKU chỉ khác nhau ở chữ; lật cả hai chiều thì trùng với xoay 180°. Lật để thành cờ riêng nếu muốn thử. Chèn trước bộ tiền xử lý ở [coco_hug.py#L43-L54](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/datasets/coco_hug.py#L43-L54) | Bật | Code gốc hoàn toàn không có augmentation. Nghiệm thu: unit test tọa độ box sau biến đổi, và vẽ ra để kiểm tra |
| I3 | Bỏ nhãn giả chồng (IoU ≥ 0,5) lên GT của task hiện tại | Bật | Tránh cùng một vật có 2 nhãn khác nhau |
| I4 | Nhận ứng viên qua prototype chỉ khi prototype gần nhất đúng là lớp teacher dự đoán | Tắt (A7) | Chống nhầm giữa các SKU cùng hãng khác vị |
| I5 | Đóng băng các tham số dùng chung sau task 1 (`input_proj`, `query_tf`, `query_position_embeddings`, `reference_points`, `level_embed`, `bbox_embed`) | Tắt (A8) | Giảm quên, nhưng có thể giảm khả năng học lớp mới |

### 6.7 Giai đoạn 6: baseline

| ID | Việc | Nghiệm thu |
|---|---|---|
| B1 | Không viết `baselines/adapt.py` riêng, mà chạy `main.py` với các cờ `--use_prompts 0 --optim_groups detr --freeze ''`, cộng thêm `--joint 1` (E0), `--save_hf 1` (FSA), cấu hình 1 lớp và `--pred_ann_dir` (detector cho E5). Mục đích là fine-tune Deformable DETR toàn bộ, không dùng prompt. Chia learning rate theo nhánh có sẵn trong code gốc ([engine.py#L601-L617](https://github.com/zyt95579/PDP_IOD/blob/7702d91d595e5ceed5df333d50c68444d7075ef9/engine.py#L601-L617)): lr 1e-4, backbone 1e-5, `sampling_offsets`/`reference_points` × 0,1. Có 3 chế độ: (a) 200 lớp → **E0**; (b) task 1 → **FSA** (I1); (c) 1 lớp "sản phẩm" → detector cho E5 | Chạy được cả 3 chế độ; E0 có đầu ra đánh giá được bằng V1–V3 |
| B2 | Cờ bật/tắt thành phần: `--use_shared`, `--use_private`, `--pseudo {none,threshold,ppg}`, `--ddl_lambda`, `--topk`, các cờ I3–I5, cờ augmentation. Thêm các cờ trả về **hành vi gốc** của từng bản sửa cho P1 (khởi tạo prompt task mới F2, gradient L_Q F4, teacher không prompt F5, cách chọn ứng viên cũ F6, prototype không lọc F7, không xáo trộn F12) | Mỗi cờ có test nhỏ; E1, E2, P1 và các ablation chỉ khác nhau ở cờ |
| B3 | **E5 (truy xuất):** detector B1c + DINOv2 (ViT-S/14 hoặc B/14) trích đặc trưng từ vùng cắt. **Detector B1c (nhóm chốt 28/09, phương án b):** train class-agnostic trên ảnh train của task 1 với **mọi box** trong ảnh, không kèm tên SKU. Tức là giả định cửa hàng đánh box "sản phẩm" (không cần biết SKU) ngay từ đầu. Đây là lợi thế của E5 so với PDP (PDP chỉ có box của SKU task 1) và phải ghi rõ khi báo cáo. Mỗi SKU lưu tối đa 100 embedding (cùng ngân sách với bộ nhớ prototype của PDP); gán nhãn theo prototype hoặc kNN gần nhất; ngưỡng "không chắc" chọn trên val. Task mới chỉ cần thêm embedding, không train lại | Đánh giá được bằng V1–V3 như các phương pháp khác |

---

## 7. Thí nghiệm

**Pilot** (2 task 100+25, `train_pilot`, 4 epoch, đánh giá trên toàn bộ val):

| Mã | Cấu hình | Để trả lời câu hỏi |
|---|---|---|
| P1 | Code gốc, chỉ sửa những gì cần để chạy được: F1, kích thước pool (phần đầu của F2), F8. Thiếu F8 thì teacher tìm file `checkpoint07.pth` không có (pilot chỉ chạy 4 epoch) và PPG bị tắt mà không báo lỗi | Các lỗi còn lại ảnh hưởng tới kết quả bao nhiêu |
| P2 | F1–F12 | Các bản sửa có làm tăng mAP@P không |
| P3 | P2 + FSA | FSA giúp hay hại mAP@C của task 2 |

**Thí nghiệm chính** (5 task, cấu hình chuẩn trừ khi ghi khác; mọi số liệu đều trên cùng tập test đã khóa):

| Mã | Cấu hình | Vai trò |
|---|---|---|
| E0 | B1a: fine-tune toàn bộ trên cả 200 lớp một lần (12 epoch) | Cận trên |
| E1 | Train tuần tự: 1 pool chung, không pool riêng, không nhãn giả; nền FSA | Cận dưới (mức quên khi không chống) |
| E2 | Tương đương MD-DETR: chỉ pool riêng, nhãn giả theo ngưỡng cố định 0,65, top-5 (như `run.sh` của MD-DETR); nền FSA | Baseline chính của paper |
| E3 | PDP đã sửa trên nền FSA (I1), không bật I2–I5, không augmentation | PDP |
| E3_coco | PDP đã sửa, đúng như paper: nền Deformable DETR COCO đóng băng, không bật I1–I5 | Tái hiện paper; cho thấy vì sao cần FSA |
| **E4** | E3 + I2 + I3 (tức PDP + I1 + I2 + I3) | **Phương pháp chính** |

Từ 29/09, E1–E4 cùng dùng nền FSA (nhóm chốt sau pilot, phụ lục C): trên nền COCO đóng băng, mọi phương pháp chỉ đạt khoảng 0,1 mAP50 ở pilot, nên so sánh giữa chúng không có ý nghĩa.
| E5 | B3, truy xuất bằng DINOv2 | Phương án thay thế / dự phòng |

**Ablation trên E4** (3 task 100+25+25 nếu thiếu GPU; các ablation chỉ ảnh hưởng task ≥ 2 thì dùng lại task 1 của E4):

| Mã | Thay đổi | Bắt buộc |
|---|---|---|
| A1 | PPG → ngưỡng cố định (chỉ τh, như Table 5 hàng 1 của paper) | Có |
| A2 | Bỏ pool chung | Có |
| A3 | Bỏ pool riêng | Có |
| A4 | top-k 5 so với 50 | Có |
| A5 | Bỏ FSA | Nên có |
| A6 | Bỏ augmentation | Nên có |
| A7 | Bật I4 | Nên có |
| A8 | Bật I5 | Nên có |
| A9 | Tắt L_DDL | Nên có |

Tùy chọn nếu còn ngân sách:
- Chạy E4 ở cấu hình đầy đủ.
- Chạy thêm 2 seed cho E3 và E4 để có độ lệch chuẩn. Nên làm nếu chênh lệch giữa các phương pháp nhỏ hơn 2 điểm.

---

## 8. Mốc kiểm soát

Mọi quyết định tại các mốc dựa trên **val**. Các ngưỡng "rõ rệt" dưới đây là đề xuất, nhóm có thể điều chỉnh.

| Mốc | Điều kiện để đi tiếp | Nếu không đạt |
|---|---|---|
| **G0** (trước pilot) | Toàn bộ test F/R/V xanh; smoke test CPU và GPU chạy hết; kernel chạy được; tập test đã khóa | Không chạy pilot |
| **G1** (sau pilot) | (a) P2 tốt hơn P1 ở mAP@P task 2. (b) Đã đo giây/ảnh; chọn cấu hình chuẩn hay đầy đủ theo QĐ-4. (c) So P3 với P2 | (a) không đạt: rà lại F2, F5, F6 trước khi đi tiếp. (c) FSA làm giảm mAP@C task 2 ≥ 3 điểm: bỏ FSA hoặc giảm số epoch FSA |
| **G2** (sau E0, E3, E4, E5) | So mAP@A và cAcc | E4 kém E5 ≥ 3 điểm mAP@A hoặc ≥ 5 điểm cAcc: chuyển phương pháp chính sang E5, giữ PDP làm đối chứng. E0 thấp: vấn đề nằm ở detector/dữ liệu chứ không ở continual learning, nên ưu tiên chỉnh độ phân giải, augmentation, số epoch trước khi làm ablation |
| **G3** (trước báo cáo) | Mọi bảng sinh lại được từ file dự đoán đã lưu; tập test chỉ dùng để báo cáo | Sửa trước khi viết báo cáo |

---

## 9. Thứ tự thực hiện

```
T0.1 → T0.2 → [F2, F4, F6, F12 + test trên CPU] ─┐
T0.3 → T0.4 → T0.5 → DL1 → DL2 → DL3 → DL4 → DL5/DL6 ─┤
[F1, F3, F5, F7–F11] → R1–R4 → V1–V3 ────────────────┴→ G0
G0 → P1, P2, P3 (+ V4) → G1
G1 → B1 (E0, FSA) → E1, E2, E3, E4 ; B3 → E5 → G2
G2 → A1–A9 → V5, V6 → (demo) → báo cáo → G3
```

- **Đường găng:** môi trường VM và kernel → dữ liệu (DL1–DL5) → sửa code và hạ tầng chạy → pilot → E3/E4.
- **Làm song song được trên Mac không cần GPU:** các bản sửa F và test CPU; công cụ dữ liệu (thử trên JSON giả); V2 và V3 (thử trên dữ liệu giả).

---

## 10. Demo và tính tiền (nên có, tùy QĐ-5)

1. **Bảng giá:** file CSV giá cho 200 SKU của RPC (giá giả định, vì RPC không có giá) và giá thật cho sản phẩm demo.
2. **Pipeline:** ảnh → dự đoán (V1 + ngưỡng đã chọn trên val) → số lượng từng SKU → hóa đơn và tổng tiền.
3. **Giao diện đơn giản:** chụp từ webcam hoặc tải ảnh lên, hiển thị bbox và hóa đơn.
4. **Task 6 (24 slot):**
   - Chụp sản phẩm Việt Nam trên nền trắng, camera nhìn từ trên xuống, và gán bbox chỉ cho sản phẩm mới (dùng CVAT hoặc Label Studio).
   - Train thêm task 6 bằng chính pipeline CL; đây chính là minh họa "nhập hàng mới" cho đề tài.
   - Lưu ý: camera và ánh sáng khác RPC nên độ chính xác sẽ thấp hơn. Sản phẩm RPC (hàng Trung Quốc) không có sẵn để demo.

## 11. Khi có ảnh ghép (giai đoạn 2)

- Ảnh ghép chỉ được thêm vào train, như một nguồn mới trong DL5. **Tập test giữ nguyên.**
- Chỉ cắt sản phẩm từ `train2019` (ảnh sản phẩm đơn).
- Thí nghiệm so sánh:
  - (a) chỉ ảnh thật (tức E4);
  - (b) ảnh thật + ảnh ghép của SKU mới;
  - (c) chỉ ảnh ghép cho SKU mới, tức kịch bản không cần chụp ảnh quầy khi nhập hàng.
- Nếu làm thêm tập test phụ bằng ảnh ghép: chia ảnh sản phẩm đơn theo góc chụp và báo cáo mức chênh so với ảnh thật.

## 12. Rủi ro

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Đặc trưng đóng băng không đủ để nhận SKU mới | mAP@C task 2–5 thấp hơn nhiều so với E0 | Bật FSA; thử mở băng một phần decoder (ngược với I5); hoặc chuyển sang E5 |
| FSA dạy mô hình coi SKU tương lai là nền | P3 có mAP@C task 2 thấp hơn P2 | Mốc G1 |
| Nhãn giả nhầm giữa các SKU gần giống nhau | V4 cho precision thấp ở nhánh prototype | Bật I4, tăng θs |
| Rò rỉ dữ liệu giữa các tập | Test cao bất thường so với val | Kiểm tra nhóm ở DL1, chia theo nhóm ở DL3 |
| Spot bị thu hồi liên tục | Tiến độ chậm | R1/R2; dùng cấu hình chuẩn |
| Lệch phiên bản thư viện | Lỗi khi import hoặc khi nạp checkpoint | Cố định phiên bản; lưu phiên bản vào mỗi run (R3) |
| Hết ngân sách GPU | Theo dõi chi phí trong Billing; đặt budget alert | Phương án B; cấu hình chuẩn; ablation 3 task |
| Hết dung lượng đĩa | `df -h` | Chỉ giữ `task_final.pth` và dự đoán; xóa `last.ckpt` khi task xong |

---

## Phụ lục A. Số liệu đã kiểm chứng và nguồn

| Số liệu | Nguồn / cách kiểm |
|---|---|
| Cấu hình VM `auto-cl` và VM cũ; quota GPU; loại GPU có ở zone; phần mềm trên `auto-cl` (SSH) | `gcloud compute instances describe`, `regions describe`, `accelerator-types list`, ngày 28/09/2026 |
| Giá Spot/ổ đĩa (VND, USD) | Cloud Billing Catalog API, Compute Engine, us-central1, ngày 28/09/2026 |
| G2 không hỗ trợ `pd-standard` | [Tài liệu GCP về các loại ổ đĩa](https://docs.cloud.google.com/compute/docs/disks) |
| Tỉ lệ tốc độ V100/L4/T4 (ResNet, chỉ để ước tính) | [Dell MLPerf trên T4](https://www.dell.com/support/kbdoc/en-us/000132094/deep-learning-performance-on-t4-gpus-with-mlperf-benchmarks), [DEEP-GAP](https://arxiv.org/pdf/2604.14552) |
| pd-standard 0,12 MiB/s mỗi GiB | [Tài liệu GCP về hiệu năng ổ đĩa](https://docs.cloud.google.com/compute/docs/disks/performance) |
| 325 giờ V100 cho 50 epoch COCO; 19 FPS | [Deformable DETR, Table 1](https://arxiv.org/abs/2010.04159) |
| Kaggle RPC: 15,9 GB, v5, CC BY-NC-SA 4.0; ảnh quầy trung bình khoảng 200 KB; quy luật tên file | Kaggle API (thông tin dataset và danh sách file) |
| Thống kê RPC (số ảnh, số vật, mức độ đông) | [Paper RPC](https://arxiv.org/abs/1901.07249) |
| 69,12M / 35,00M tham số; checkpoint 0,52 / 0,26 GiB | Tạo model với 225 lớp và 224 slot trên CPU |
| Lỗi F2, F3, F4, F6, F11, F12 | Chạy code (CPU, trọng số ngẫu nhiên) và đọc mã nguồn kernel |
| Checkpoint `SenseTime/deformable-detr` (300 query, không two-stage, không box refine) | HF Hub config |

## Phụ lục B. Thay đổi so với v1.2 (rà soát trước khi code, 28/09/2026)

| Mục | Thay đổi | Lý do |
|---|---|---|
| 2.2 | Chốt QĐ-1 (224 slot), QĐ-2 (có); thêm QĐ-7 (repo chỉ ở local); QĐ-3 cập nhật việc xóa VM cũ | Nhóm trả lời ngày 28/09 |
| 4.4 | Viết lại ý nghĩa slot 224 | Đọc `DeformableDetrLoss.loss_labels`: "không có vật" là vector đích toàn 0, slot 224 không bao giờ là đích |
| 5 | Thêm `autocheckout/`, `docs/formats.md`, `PROGRESS.md`; ghi chú cách chạy `pdp/` và xung đột tên `datasets` | Để các phần code viết song song khớp định dạng |
| F3 | Tính L_DDL một lần mỗi bước từ tham số prompt, không đi qua decoder | Cùng giá trị, ít sửa code hơn, không tính thừa |
| F10 | Ghi chú model dùng cấu hình HF mặc định (không loss phụ, cost lớp 1) | Khác Deformable DETR gốc; giữ nguyên để giống PDP |
| R1 | Xác nhận `_atomic_save` của Lightning 2.1.3 không an toàn khi bị tắt máy giữa lúc ghi | Đọc mã nguồn Lightning |
| V1 | Giữ top-100 cặp (query, lớp) kèm chỉ số query; đếm lấy lớp cao nhất mỗi query | v1.2 vừa đòi "mỗi query 1 nhãn" vừa đòi mAP khớp code gốc (F9); hai yêu cầu này mâu thuẫn |
| I2 | Mặc định không lật ảnh | Lật tạo bao bì chữ ngược gương, không có trên quầy thật |
| B2 | Thêm cờ trả về hành vi gốc cho P1 | P1 cần chạy lại được hành vi gốc của từng lỗi |

## Phụ lục C. Thay đổi và phát hiện trong lúc implement (28–28/09/2026)

| Mục | Thay đổi / phát hiện | Ghi chú |
|---|---|---|
| F2 | Gram-Schmidt gốc không trực giao được khi prompt cũ đã train (cos tới 0,17) → khởi tạo prompt task mới bằng phép chiếu lên phần bù trực giao (QR) | Test phát hiện |
| F9 / V1 | Một đường suy luận duy nhất (`pdp/inference.py`); CLI là `main.py --predict_only 1` | |
| V2 | COCOeval chạy 1 lần mỗi task rồi gom tập con (chính xác tuyệt đối, test đối chiếu từng số); chỉ vùng diện tích "all". Nhanh khoảng 5 lần | Không dùng được `accumulate(p)` của pycocotools |
| R1 | `_atomic_save` của Lightning không an toàn; dùng callback riêng ghi file tạm rồi đổi tên. Ghi ở đầu batch kế tiếp để resume không lặp lại batch | Test: tổng số bước không đổi sau resume |
| B1 | Tên backbone `backbone.0` của nhánh lr gốc không khớp tên tham số HuggingFace → cấu hình B1 dùng `--lr_backbone_names backbone`. Stem và `layer1` của backbone luôn đóng băng (hành vi chuẩn của Deformable DETR) | |
| B1c / E5 | Chọn phương án b: detector class-agnostic train trên mọi box của ảnh task 1 (28/09) | Lợi thế của E5, ghi rõ khi báo cáo |
| Dữ liệu | Ảnh quầy RPC vuông nhưng cạnh 1750–1890 px (không cố định 1800); mirror HuggingFace thiếu tên file và `level` → bắt buộc dùng Kaggle | Kiểm tra 28/09 |
| QĐ-7 | Đổi thành đẩy public, ghi nguồn; VM lấy code bằng `git` | 28/09 |
| Test CPU | Ảnh test 96 px (ở 64 px, batch 1 ảnh làm GroupNorm backward trên CPU lỗi) | Không ảnh hưởng ảnh 800 px |
| F13 | Bias của classifier luôn bị HuggingFace đặt về 0 (p = 0,5), xóa mất prior của focal loss → thêm F13 | Phát hiện qua `loss_ce` khoảng 730 ở phút đầu của pilot; đã dừng pilot, sửa, chạy lại |
| Cấu hình | `configs/exp/*.sh` cho mọi thí nghiệm của mục 7 (P1–P3, FSA_pilot, E0, FSA, DET, E1–E4, A1–A9) | Test parse mọi file |
| Batch hiệu dụng | 32 (code gốc, gộp gradient) → **4** cho mọi thí nghiệm sau pilot. Cùng lượng tính toán nhưng gấp 8 lần số bước tối ưu. Pilot ở 32 (364 bước ở task 1): P2 đạt val mAP50 0,042, FSA_pilot đạt 0,085. Mô hình định vị tốt (AP50 không phân biệt lớp 0,73–0,93) nhưng chỉ 14–20% đúng SKU. FSA_pilot_eb4 (cùng dữ liệu, batch 4) đạt 0,675, 75% đúng SKU | Phát hiện qua pilot 28/09. Một task RPC có ít ảnh hơn khoảng 10 lần so với một task COCO của paper. Cấu hình pilot giữ batch 32 để tái lập. P2_eb4/P3_eb4 kiểm tra lại cho PDP |
| Nền của E1–E4 | Chẩn đoán sau pilot: PDP trên nền COCO đóng băng đạt 0,110 mAP50 task 1 (P2_eb4); trên nền FSA đạt 0,829, lớp cũ/mới sau task 2 là 0,827/0,669, không lớp nào thiếu prototype (P3_eb4). Nhóm chốt (29/09): E1, E2, E3 chạy trên nền FSA như E4; thêm E3_coco (đúng như paper) để tái hiện. Bỏ P1 | Khoảng +17 giờ GPU cho E3_coco |
| F14 | Nhánh prototype của PPG chấp nhận query phụ trên vật đã có nhãn (đặc trưng khớp prototype), không có bước loại trùng. Nhãn giả trùng tích lũy qua các đợt (E4: 3,7% ở task 2 → 31% ở task 5), mô hình đếm thừa (cAcc test đợt 5 là 0,10). Thêm `--pseudo_dedup_iou` (NMS không phân biệt lớp giữa các nhãn giả): audit task 5 cho nhãn trùng từ 2.505 còn 21, precision từ 0,60 lên 0,89 | Phát hiện 30/09 khi phân tích E4. Nhóm chốt cùng ngày: bật (`--pseudo_dedup_iou 0.5`) cho mọi run; E4 chạy lại, bản cũ đổi tên thành E4_noF14 để đối chứng |
