# Tổng quan tình trạng AutoCheckout-CL — 20/10/2026

Báo cáo tóm tắt sau khi đọc tài liệu và code nhánh `develop`. Mục cuối có giải thích thuật ngữ.

## 1. Tóm tắt một câu

Phần **nghiên cứu cơ bản đã xong** (chạy PDP với 2 backbone đủ 5 task trên ảnh thật), nhưng **phần ứng dụng chưa bắt đầu**: chưa có app tính tiền, chưa có demo, chưa dùng ảnh ghép (cắt nền, ghép lại) vào train.

## 2. Đã làm được

| Hạng mục | Nội dung |
|---|---|
| Dữ liệu ảnh thật | 22.494 ảnh train / 1.503 val / 6.003 test, chia theo nhóm ảnh để tránh lộ đề; tập test đã khóa |
| Cách chia đợt | 5 task: 100 SKU ban đầu, sau đó 4 đợt mỗi đợt thêm 25 SKU (gọi tắt **100+4×25**) |
| Code | PDP lấy từ repo gốc của tác giả, bổ sung lưu/chạy tiếp, chạy 640px, thêm ConvNeXt-V2 |
| Đánh giá | Đo cả độ chính xác phát hiện (AP) và độ chính xác đếm (cAcc...). Ngưỡng chọn trên val, test chỉ dùng để báo cáo |
| Train | Hai backbone chạy đủ 5 task, xong ngày 15/10; số liệu kiểm tra lại ngày 20/10 khớp |

### Kết quả cuối (test, đủ 200 SKU)

| Chỉ số | ResNet-50 | ConvNeXt-V2-Base |
|---|---:|---:|
| AP50 (phát hiện đúng) ↑ | 62,33% | **84,40%** |
| cAcc (đếm đúng cả giỏ) ↑ | 2,58% | **21,74%** |
| Forgetting (mức quên) ↓ | **6,63** | 8,67 |
| cAcc trên ảnh khó | 0% | 8,59% |

Nhận xét:
- ConvNeXt tốt hơn rõ về phát hiện và đếm, nhưng **quên nhiều hơn**.
- cAcc 21,74% vẫn **quá thấp** cho thanh toán thật (khoảng 4/5 giỏ bị tính sai).
- **SKU mới bị đếm thiếu nặng**: ở đợt cuối, model chỉ đếm được khoảng 43% số SKU mới thực có (ResNet chỉ khoảng 9%).

Báo cáo chi tiết: `docs/project/FINAL_BACKBONE_COMPARISON_2026-10-20.md`.

### Kết quả qua từng task (test, AP50 / cAcc)

| Sau task | Số SKU đã học | ResNet-50 | ConvNeXt-V2-Base |
|---|---:|---:|---:|
| 1 | 100 | 83,16% / 17,52% | 98,35% / 73,16% |
| 2 | 125 | 79,14% / 7,46% | 97,54% / 53,32% |
| 3 | 150 | 73,94% / 4,60% | 94,66% / 29,97% |
| 4 | 175 | 67,99% / 3,18% | 89,72% / 22,29% |
| 5 | 200 | 62,33% / 2,58% | 84,40% / 21,74% |

Ngay từ task 1, trước khi học liên tục, hai model đã chênh khoảng 15 điểm AP50.

### B1 và B2 đã đủ để kết luận backbone nào tốt hơn chưa?

**Hai bản giống và khác nhau ở đâu:**

| | B1: ResNet-50 | B2: ConvNeXt-V2-Base | Giống hay khác |
|---|---|---|---|
| Dữ liệu, split, seed | RPC 100+4×25, seed 0 | Như B1 | Giống |
| Ảnh, batch, epoch, số bước train | 640px, batch 8, 16 epoch/task | Như B1 | Giống |
| Phương pháp PDP và phần bị khóa | Khóa backbone/encoder/decoder | Như B1 | Giống |
| Dữ liệu pretrain của backbone | COCO (đi kèm detector) | ImageNet-22k → 1k | **Khác** |
| Kích thước backbone | Khoảng 25 triệu tham số | Khoảng 88 triệu tham số | **Khác** (gấp 3,5 lần) |
| Khớp với encoder/decoder | Khớp, vì được train cùng nhau | Không khớp; phần nối được khởi tạo lại | **Khác** |
| Máy train | 4080 SUPER | 4090 → L4 → 4080 SUPER, có chạy tiếp | Khác, nhưng không ảnh hưởng kết quả |

**Kết luận được gì, chưa kết luận được gì:**

| Câu hỏi | Kết luận | Vì sao |
|---|---|---|
| Trong cấu hình đã chạy, bên nào cho kết quả cuối tốt hơn? | **ConvNeXt, kết luận được** | Chênh rất lớn (AP50 +22, cAcc +19 điểm), trong khi đổi seed thường chỉ dao động 1–2 điểm; nhất quán trên val/test, mọi task, mọi nhóm lớp, mọi mức độ khó |
| Bên nào chống quên tốt hơn? | **Chưa kết luận được** | Chỉ chênh 2 điểm (6,63 so với 8,67), mới 1 seed; hai model quên từ hai mức đỉnh khác nhau (98% so với 83%) |
| Kiến trúc ConvNeXt có tốt hơn ResNet nói chung? | **Chưa kết luận được** | Khác cùng lúc 3 thứ (pretrain, kích thước, độ khớp; xem bảng trên), không tách được yếu tố nào tạo ra khác biệt |
| Nếu không khóa backbone thì sao? | **Chưa biết** | Phần lớn khoảng cách đến từ đặc trưng có sẵn của backbone (task 1 đã chênh 15 điểm). Nếu bật FSA (fine-tune toàn bộ, xem mục 3), ResNet có thể bắt kịp một phần |
| Bên nào hợp hơn để triển khai tại quầy? | **Chưa biết** | Chưa đo tốc độ suy luận và VRAM |

**Cần chạy thêm:**

| Muốn trả lời | Thí nghiệm |
|---|---|
| Kết quả có ổn định không | Thêm 2 seed cho mỗi backbone; hoặc ít nhất tính bootstrap trên test (không cần train lại) |
| Kiến trúc hay pretrain/kích thước tạo khác biệt | So với ConvNeXt-V2-Tiny (cỡ ResNet-50) hoặc ResNet-50 pretrain ImageNet-22k |
| Khi không khóa backbone | Chạy cả hai với FSA và F14 |
| Chọn để triển khai | Đo thời gian xử lý mỗi ảnh và VRAM |

**Cách viết an toàn trong báo cáo đồ án:** "Trong cấu hình PDP khóa backbone trên RPC 100+4×25 (1 seed), thay ResNet-50 (pretrain COCO) bằng ConvNeXt-V2-Base (pretrain ImageNet-22k) tăng AP50 cuối 22 điểm và cAcc 19 điểm. Mức quên không giảm. Chưa tách riêng được tác động của kiến trúc, kích thước model và dữ liệu pretrain."

## 3. Vấn đề quan trọng nhất: baseline mới đang tắt các bản sửa của tháng 9

Tháng 9 (nhánh `main`), nhóm phát hiện PDP nguyên bản không chạy tốt trên RPC và đã sửa hai chỗ:
- **FSA**: fine-tune toàn bộ detector trên 100 SKU đầu trước khi học tăng dần. Lý do: model học từ COCO không phân biệt được các SKU na ná nhau (cùng hãng, khác vị).
- **F14**: loại các nhãn giả bị trùng. PPG của paper sinh nhãn giả trùng, tăng từ 3,7% lên 31% qua các đợt.

Với hai bản sửa này, kết quả tháng 9 đạt khoảng **94,7% AP50, cAcc 36–43%, gần như không quên**.

Baseline mới (tháng 10) **cố ý chạy đúng cấu hình paper**, nên đã **tắt cả FSA và F14** (khóa backbone/encoder/decoder, `pseudo_dedup_iou 0`). Hai đợt dùng pipeline khác nhau nên không so trực tiếp được, nhưng đây gần như chắc chắn là lý do lớn khiến cAcc hiện tại thấp.

**Đề xuất ưu tiên số 1:** chạy thêm một campaign ConvNeXt có bật lại FSA và F14.

## 4. Hai bản đã chạy có giống code gốc của paper không?

**Không giống hoàn toàn.** Cả hai bản chạy **phương pháp PDP theo paper**, nhưng code đã sửa nhiều so với repo gốc của tác giả (`zyt95579/PDP_IOD`, commit `7702d91`). Mọi thay đổi nằm trong các commit riêng; xem bằng `git log -- pdp/`. Các khác biệt chia làm 4 lớp.

### 4.1. Sửa lỗi code để khớp với paper (đang BẬT ở cả 2 bản)

Repo gốc có nhiều chỗ **không làm đúng như paper mô tả**, nên chạy nguyên code gốc cũng không ra phương pháp của paper.

Bảng dưới đã được **kiểm tra trực tiếp trên code** (repo gốc ở `../PDP_IOD`, so với `pdp/` của nhóm) và đối chiếu với PDF paper (mục 4.2, 4.3, 5.1). Cột "Bằng chứng trong code gốc" chỉ ra đúng dòng gây lỗi.

| Mã | Lỗi trong code gốc | Bằng chứng trong code gốc | Đã sửa | Giải thích dễ hiểu |
|---|---|---|---|---|
| F1 | Chia task và kích thước pool viết cứng cho COCO | `prompt.py`: `total_classes = 80`, `private_size = int(80)`; `main.py` lấy task từ `task_info_coco()` (danh sách tên lớp COCO) | Đọc file cấu hình task của RPC; pool = tổng số lớp | Code gốc chỉ chạy được với COCO. Với RPC (task 1 có 100 lớp), pool riêng chỉ có 80 ô, nên các task sau bị cắt mất hoặc lỗi chỉ số |
| F2 | Prompt riêng của task ≥ 2 **bằng 0 và không bao giờ học** | `gram_schmidt()` chỉ điền giá trị cho các ô của task đầu (`task_count = 0` lúc khởi tạo), các ô còn lại để 0; không có chỗ nào khởi tạo lại khi sang task mới | Thêm `init_task_prompts()`: khởi tạo ô của task mới (ngẫu nhiên, trực giao với ô cũ) khi bắt đầu mỗi task | Mỗi đợt hàng mới cần "bộ nhớ riêng" (private prompt). Ở code gốc, bộ nhớ này bằng 0, và vector 0 không nhận được gradient nên mãi mãi bằng 0. Từ đợt 2 trở đi coi như không có bộ nhớ riêng |
| F3 | Loss L_DDL **viết đúng công thức nhưng không bao giờ chạy** | `use_ddl_loss = False`; thêm nữa, nhánh này chỉ chạy khi `ortho_mu > 0`, mà model tạo prompt với `ortho_mu = 0`; và giá trị loss trả về bị decoder bỏ đi (`p_list, _, output = ...`) | Tính L_DDL mỗi bước train và cộng vào loss (λ = 0,15) | L_DDL giúp prompt chung và prompt riêng "không giẫm chân nhau". Code gốc có viết hàm nhưng tắt hẳn, nên thành phần này của paper không có tác dụng |
| F4 | Loss L_Q được tính nhưng không có gradient | `engine.py`: `query_loss` được tính bên trong `with torch.no_grad():` | Cho gradient chạy qua `query_tf` | Có tính điểm phạt nhưng không truyền ngược về model, nên phần đó không bao giờ học. Giống chấm bài mà không trả bài cho học sinh |
| F5, F8 | Teacher chạy **không có prompt**; tên file checkpoint viết cứng; thiếu file thì PPG tắt mà chỉ in cảnh báo | Teacher gọi model không truyền `query`, mà decoder chỉ dùng prompt khi `query is not None`; file `checkpoint07.pth` viết cứng (`main.py` cũng ép `epochs = 8`, bỏ qua tham số dòng lệnh); thiếu file thì `print("Warning...")` rồi `old_model = None` | Teacher là bản sao đóng băng của model cuối đợt trước, chạy có prompt như lúc nó được train | Teacher là "model cũ", dùng để nhắc model mới về SKU cũ. Code gốc chạy model cũ thiếu prompt, tức là không giống model cũ thật. Nếu đổi số epoch thì không tìm thấy file, và PPG tự tắt, chỉ có một dòng cảnh báo trong log |
| F6 | PPG chọn sai ứng viên | `post_process(..., bg_thres_topk=5)` chỉ giữ **5** cặp (vật, lớp) điểm cao nhất mỗi ảnh, xét cả lớp mới; điều kiện lớp cũ là `labels <= PREV` (nhãn đánh số từ 0, nên lọt lớp mới đầu tiên); đặc trưng lấy theo vị trí trong danh sách 5 cặp thay vì đúng chỉ số query | Viết lại PPG (`pdp/ppg.py`): xếp hạng từng query trên lớp cũ, tối đa 50 query, mỗi query một nhãn, dùng `< PREV`, lấy đúng đặc trưng | PPG gán nhãn giả cho SKU cũ trong ảnh mới. Ảnh RPC có khoảng 12 món nhưng code gốc chỉ xét tối đa 5. Nó còn so sai đặc trưng với prototype, nên nhánh "kiểm tra bằng prototype" gần như so ngẫu nhiên |
| F7 | Prototype lấy cả dự đoán sai | Code gốc đưa mọi query được ghép với vật thật vào bộ nhớ, không xét model đoán đúng hay sai. Paper ghi rõ chỉ lấy "correctly classified instances" | Chỉ lấy query mà lớp điểm cao nhất trùng lớp thật | Prototype là "hình mẫu" của SKU. Lấy cả dự đoán sai thì hình mẫu bị lẫn |
| F11 | CUDA kernel không bao giờ được nạp | `load_custom.py` tìm thư mục `kernels/deformable_detr` nằm ngoài repo (không có), lỗi thì chỉ ghi cảnh báo và dùng bản PyTorch | Nạp kernel từ thư viện `transformers`; có cờ bắt buộc phải có kernel | Chỉ ảnh hưởng tốc độ: theo đo của nhóm, chậm khoảng 20 lần; kết quả không đổi |
| F12 | Không xáo trộn dữ liệu khi chạy **1 GPU** | `DataLoader` không có `shuffle`. Chạy 2 GPU như tác giả thì Lightning tự xáo trộn; chạy 1 GPU như nhóm thì không | Bật `shuffle` rõ ràng | Với cấu hình gốc 2 GPU thì không lỗi. Nhóm chạy 1 GPU nên phải sửa, nếu không model học ảnh theo đúng một thứ tự mỗi epoch |
| F13 | Giá trị khởi tạo của classifier bị xóa | `modeling_deformable_detr.py`: đặt bias theo prior 0,01, nhưng ngay sau đó `post_init()` chạy `_init_weights` và đưa bias của mọi lớp `Linear` về 0 | Khôi phục prior sau khi khởi tạo | Classifier cần bắt đầu ở trạng thái "hầu như không có vật". Bias bằng 0 thì lúc đầu model đoán mọi lớp đều 50%, loss khởi đầu rất lớn và học chậm |

Đã kiểm tra thêm: các bản sửa trên **thực sự được dùng** khi train. `main.py` gọi `set_teacher()` và `init_task_prompts()`; `engine.py` cộng L_DDL vào loss, dùng `ppg.py`, và cho L_Q có gradient. Các cờ liên quan để mặc định là bật (`ddl_lambda 0.15`, `query_loss_grad 1`, `ppg_legacy 0`, `prior_init_classifier 1`, `proto_correct_only 1`, `shuffle 1`).

Quy mô: `engine.py` lệch khoảng 340 dòng so với gốc, `main.py` khoảng 410 dòng, `prompt.py` khoảng 80 dòng. Có thêm file mới: `ppg.py`, `checkpointing.py`, `runtime.py`, `inference.py`…

**Sửa so với bản trước của báo cáo này:**
- F3 không phải "sai công thức". Công thức trong code khớp phương trình (10) của paper, chỉ là bị tắt.
- F5 không phải "nạp sai thời điểm". Lỗi thật là teacher chạy thiếu prompt, cộng với tên file viết cứng.
- F12 chỉ là lỗi khi chạy 1 GPU.
- Con số "chậm 20 lần" ở F11 lấy từ đo đạc của nhóm, báo cáo này chưa tự đo lại.

### 4.2. Cấu hình chạy khác code gốc (đổi cho hợp RPC và phần cứng)

Lưu ý quan trọng: **paper không ghi batch size, số epoch, kích thước ảnh hay số GPU.** Các giá trị ở cột "Code gốc" lấy từ `run.sh` và `main.py` của tác giả, không phải từ paper.

| | Code gốc (`run.sh`, `main.py`) | Bản đã chạy | Giải thích dễ hiểu |
|---|---|---|---|
| Dữ liệu | COCO, 80 lớp; `run.sh` chạy 3 task (40+20+20). Bảng chính của paper dùng 4 task | RPC, 200 SKU, 5 task | Bài toán khác hẳn: COCO là người, xe, chó…; RPC là các gói hàng rất giống nhau. Vì vậy **không so trực tiếp được số của nhóm với bảng trong paper** |
| Kích thước ảnh | Mặc định của processor: cạnh ngắn 800, cạnh dài tối đa 1333 | 640×640 (ảnh RPC vuông) | Ảnh nhỏ hơn thì train nhanh hơn và tốn ít bộ nhớ GPU hơn, nhưng có thể mất chi tiết của vật nhỏ |
| Batch hiệu dụng | 32, viết cứng: `accumulate_grad_batches = 32/(n_gpus × batch_size)` | 8 | Batch là số ảnh model xem trước mỗi lần tự sửa. Mỗi task RPC ít ảnh hơn COCO khoảng 10 lần; batch 32 thì cả task chỉ được vài trăm lần sửa, không kịp học. Batch 8 cho số lần sửa gấp 4 |
| Epoch mỗi task | 8, viết cứng trong vòng lặp của `main.py` (bỏ qua `--epochs`) | 16 | Số lượt học hết dữ liệu. Chọn 16 làm ngân sách cố định; chưa chứng minh là đủ (chưa kiểm tra hội tụ) |
| GPU | 2 GPU, mỗi GPU batch 2 | 1 GPU, batch 8, bật TF32 | TF32 là chế độ tính nhanh hơn trên GPU đời mới, sai số rất nhỏ. Dùng 1 GPU để vừa ngân sách thuê máy |

**Giống code gốc và paper:** khóa backbone/encoder/decoder, chỉ train classifier và prompt; learning rate 1e-4 (phần mới) / 1e-5 (phần cũ); 100 prompt chung; số prompt riêng bằng tổng số lớp (paper mục 5.1); prompt dài 10 (theo `run.sh`, paper không ghi); λ_DDL = 0,15 và λ_Q = 0,1; ngưỡng PPG 0,5 / 0,2 / 0,5 (paper mục 5.1); mỗi lớp một prototype là vector trung bình, chỉ cập nhật ở epoch cuối, bộ nhớ 100 vector mỗi lớp (paper mục 4.3 và code gốc).

Giới hạn kiểm tra: thư mục kết quả train (`data/`) không có trên máy này, nên báo cáo chưa đọc được file tham số thực tế của từng lần chạy (`resolved_cli.json`). Cấu hình trên lấy từ `configs/exp/native/common.sh`, giá trị mặc định trong `main.py`, và tài liệu cấu hình baseline của nhóm.

### 4.3. Đổi backbone (chỉ ở B2)

B2 thay ResNet-50 bằng **ConvNeXt-V2-Base**; paper không làm việc này. Encoder/decoder vẫn lấy từ checkpoint Deformable DETR vốn được train với ResNet. Phần nối với backbone mới (`input_proj`), classifier và prompt được khởi tạo lại từ đầu.

### 4.4. Cải tiến riêng của nhóm (code có sẵn nhưng đang TẮT)

Baseline cố ý tắt những cải tiến này để đo đúng phương pháp paper:
- **FSA**: fine-tune toàn bộ detector ở task 1.
- **F14**: loại nhãn giả bị trùng.
- **I2–I5**: augmentation khi train; bỏ nhãn giả chồng lên box của task hiện tại; kiểm tra prototype gần nhất; freeze shared.

### Kết luận
- **B1 (ResNet-50)**: phương pháp paper, code đã sửa lỗi cho khớp paper, cấu hình chỉnh cho RPC. Đây là bản tái dựng PDP đúng nghĩa nhất, nhưng **không phải chạy nguyên code gốc**.
- **B2 (ConvNeXt)**: như B1, thêm việc đổi backbone.
- Chạy "đúng 100% code gốc" thì không có ý nghĩa: code không chạy được trên RPC (F1) hoặc chạy ra sai phương pháp (F2–F8).
- Khi viết báo cáo đồ án, cần ghi rõ các khác biệt ở mục 4.2 (nhất là batch 8 thay vì 32, ảnh 640 thay vì 800).

## 5. Phần ảnh ghép (synthetic)

### Hiện có gì
- **Nguyên liệu (dùng được):** 7.502 ảnh sản phẩm đã tách nền (cutout) cho 200 SKU, và 208 ảnh khay trống.
- **Bộ 20.000 ảnh ghép cũ:** được ghép từ nguyên liệu trên, nhưng **chưa dùng để train**.
- **Bản 202 ảnh đã sửa:** 202 ảnh có nhãn lỗi được xử lý (193 ảnh ghép lại, 9 ảnh giữ nguyên). Bản này đứng riêng, chưa đưa vào train. Còn 20 case chưa xử lý.

### Vấn đề của bộ 20.000 ảnh cũ

1. **Sai cách chia đợt.** Bộ này được sinh khi nhóm còn chia 4 đợt (48, 68, 47, 37 SKU). Cách chia mới 100+4×25 xếp SKU vào đợt khác, nên phải viết lại file nhãn theo đợt mới.
   - Đây **không phải vẽ lại box**. Chỉ cần chạy script lọc và đổi ID lớp; repo đã có sẵn `tools/make_task_json.py` hỗ trợ trộn nhiều nguồn ảnh.
   - Hệ quả nhỏ: một ảnh có thể chứa SKU của nhiều đợt mới. Khi train một đợt, các SKU không thuộc đợt đó không có nhãn và bị coi là nền. Ảnh thật cũng đang chịu quy ước này.
2. **Không lưu mask gốc** (xem thuật ngữ). Trong 238.013 vật, 222 vật (0,09%) có bbox lệch với đường viền; số này đã xử lý. Phần còn lại khớp nhau và **đủ tin cậy để train**, chỉ là không chứng minh được tuyệt đối.
3. **Không lưu cách ghép** (nền nào, seed, xoay/phóng to bao nhiêu), nên không tái tạo được ảnh.
4. **Gần như không có che khuất:** mỗi vật bị che tối đa khoảng 15%. Ảnh thật khó có nhiều vật chồng nhau, và đó chính là chỗ model đang yếu.

### Hai cách để dùng ảnh ghép

| | A. Dùng lại 20.000 ảnh cũ | B. Sinh bộ ảnh ghép mới |
|---|---|---|
| Làm gì | Viết lại file nhãn theo 5 đợt, thay 202 ảnh lỗi bằng bản đã sửa | Dùng cutout và nền có sẵn để ghép ảnh mới, theo đúng 5 đợt |
| Công sức | Khoảng nửa ngày đến 1 ngày | Khoảng 2–3 ngày viết code; sinh ảnh trên CPU vài giờ |
| Ưu điểm | Nhanh, rẻ | Điều chỉnh được mức che khuất 20–45%; lưu đủ mask/seed; mỗi ảnh chỉ chứa SKU của đợt đó |
| Nhược điểm | Không có che khuất, nên ít giúp ảnh khó | Phải viết thêm code đặt vật và kiểm soát chồng lấn |

Với cả hai cách, **phần tốn nhất là train lại** (nhiều ngày GPU cho 5 đợt), không phải xử lý dữ liệu. Cách B cũng là công cụ cần cho demo Task 6 (sản phẩm Việt Nam).

Khi trộn ảnh ghép với ảnh thật **không bị lộ đề**: nền được tạo từ ảnh val2019 (thuộc tập train), và test chỉ lấy từ test2019. Loader cần chỉnh nhẹ, vì hiện chỉ đọc một thư mục ảnh cho mỗi đợt.

## 6. Dữ liệu thật hiện tại có đủ không?

- **Đủ để nghiên cứu trên RPC.** Mỗi đợt có khoảng 12.000–22.000 ảnh. Đợt 1 đạt AP50 98%, val và test sát nhau. Kết quả tụt là do học liên tục (quên, đếm thiếu SKU mới) và cấu hình khóa model, không phải do thiếu ảnh.
- **Không đủ cho câu chuyện của đồ án.** Hiện mỗi SKU mới được học từ hàng nghìn **ảnh quầy thật đã vẽ box**. Ngoài đời, khi nhập hàng mới, cửa hàng chỉ có vài chục ảnh chụp từng sản phẩm riêng lẻ. Ảnh ghép chính là cách biến ảnh đơn lẻ thành ảnh quầy để học, và **bắt buộc cho demo Task 6**.

## 7. Việc chưa làm

1. App tính tiền và demo: chưa có code. Mới có hướng dẫn chụp dữ liệu demo (`AN/HUONG_DAN_CHUP_DATA_DEMO.md`): 5–10 SKU Việt Nam kèm `catalog.csv` có giá.
2. Ảnh ghép chưa đưa vào train (mục 5).
3. Các thử nghiệm mở rộng chưa chạy: FSA, F14, LoRA, prototype K=3, freeze shared. Cờ `--freeze_shared_after_task1` hiện chưa khóa đúng các prompt shared.
4. Mới chạy 1 seed và chưa kiểm tra hội tụ (16 epoch mỗi đợt có thể chưa đủ).
5. Driver GPU RTX 3060 local chưa chạy được. Dữ liệu và checkpoint không nằm trên máy Mac.

## 8. Đề xuất thứ tự làm

1. Chạy campaign **ConvNeXt + FSA + F14**: khả năng cao nhất để tăng cAcc.
2. Phân tích lỗi trên val: vì sao SKU mới bị đếm thiếu (ngưỡng điểm, NMS hay classifier).
3. Song song: chụp dữ liệu Task 6, viết pipeline demo đơn giản (ảnh → đếm → tra giá trong `catalog.csv` → tổng tiền).
4. Ảnh ghép: làm cách A nếu cần kết quả nhanh; làm cách B cho demo và để cải thiện ảnh khó.
5. Sau cùng mới tới LoRA, K=3 và chạy thêm seed.

## 9. Giải thích thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **SKU** | Một mã sản phẩm cụ thể, ví dụ "Coca lon 330ml". RPC có 200 SKU |
| **Continual / Incremental Learning (học liên tục)** | Model học thêm lớp mới theo từng đợt mà không train lại từ đầu và không giữ lại ảnh cũ |
| **Task / đợt** | Một lần nhập hàng mới. Mỗi task, model chỉ được thấy nhãn của các SKU mới |
| **Protocol** | Quy định chia SKU vào các task, ví dụ 100+4×25 |
| **Replay-free** | Không lưu lại ảnh cũ để ôn tập; chỉ giữ trọng số model |
| **Forgetting (độ quên)** | Mức tụt độ chính xác của SKU cũ sau khi học SKU mới. Càng thấp càng tốt |
| **Backbone** | Phần mạng trích đặc trưng ảnh. Ở đây so sánh ResNet-50 (cũ) với ConvNeXt-V2-Base (mới hơn, mạnh hơn) |
| **Detector / Deformable DETR** | Model phát hiện vật: tìm vị trí (box) và đoán lớp. PDP xây trên Deformable DETR |
| **PDP** | Phương pháp trong bài báo nhóm dùng: thêm "prompt" (vector học được) chia hai nhóm, *shared* dùng chung các task và *private* riêng từng task, để học task mới mà ít quên task cũ |
| **Prototype** | Vector trung bình đại diện cho mỗi SKU, dùng để nhận ra SKU cũ |
| **PPG / nhãn giả** | Khi train task mới, ảnh có SKU cũ nhưng không có nhãn. Model cũ tự đoán và gán "nhãn giả" cho chúng để model mới không quên |
| **FSA** | Fine-tune toàn bộ detector trên task đầu trước khi bắt đầu học liên tục (bản sửa của nhóm, tháng 9) |
| **F14** | Bản sửa loại nhãn giả bị trùng trên cùng một vật (của nhóm, tháng 9) |
| **Freeze (đóng băng)** | Khóa một phần model không cho cập nhật khi train |
| **Fine-tune** | Train tiếp một model đã học sẵn trên dữ liệu mới |
| **Pretrained** | Model đã được người khác train sẵn trên bộ dữ liệu lớn (COCO, ImageNet) |
| **LoRA, K=3** | Hai kỹ thuật mở rộng dự kiến thử sau: LoRA thêm ít tham số nhỏ để tinh chỉnh; K=3 cho mỗi SKU 3 prototype thay vì 1 |
| **Bbox** | Hình chữ nhật bao quanh vật, là nhãn mà detector học |
| **Mask / polygon** | Mask: bản đồ đánh dấu từng pixel thuộc vật nào. Polygon: đường viền rút gọn từ mask |
| **Mask gốc (owner mask)** | Mask lúc ghép ảnh, ghi pixel nào thuộc vật nào sau khi các vật che nhau. Có nó thì chứng minh được bbox đúng và tái tạo được ảnh |
| **Cutout** | Ảnh một sản phẩm đã xóa nền, dùng để ghép |
| **Synthetic / ảnh ghép** | Ảnh quầy tạo bằng cách dán cutout lên ảnh khay trống |
| **Che khuất (occlusion)** | Vật này che một phần vật kia. Ảnh quầy thật có nhiều, ảnh ghép cũ gần như không có |
| **SAM** | Model tách vật khỏi nền, nhóm đã dùng để tạo cutout |
| **Train / val / test** | Train: ảnh để học. Val: ảnh để chọn tham số (như ngưỡng). Test: ảnh chỉ dùng để chấm điểm cuối, không được dùng để chỉnh model |
| **Leakage (lộ đề)** | Ảnh giống nhau lọt vào cả train và test, làm điểm cao giả |
| **Epoch** | Một lượt model học hết toàn bộ ảnh train |
| **Seed** | Số khởi tạo ngẫu nhiên. Chạy nhiều seed để chắc kết quả không do may mắn |
| **Hội tụ** | Model đã học đến mức không cải thiện thêm |
| **AP50** | Độ chính xác phát hiện, tính một box là đúng khi trùng ít nhất 50% với box thật. Càng cao càng tốt |
| **cAcc** | Tỉ lệ ảnh mà model đếm đúng **toàn bộ** số lượng từng SKU, tức là tính tiền đúng cả giỏ. Chỉ số quan trọng nhất cho thanh toán |
| **ACD** | Trung bình mỗi ảnh đếm sai bao nhiêu món |
| **Ngưỡng (threshold)** | Mức điểm tự tin tối thiểu để tính một box là có thật |
| **NMS** | Bước gộp các box trùng nhau trên cùng một vật, để không đếm một vật hai lần |
| **Campaign** | Một lần chạy train đủ 5 task với một cấu hình cố định |
| **Ablation** | Thí nghiệm bật/tắt từng thành phần để đo tác dụng riêng của nó |
| **Loss** | "Điểm phạt" đo model đoán sai bao nhiêu. Model học bằng cách giảm loss |
| **Gradient** | Hướng cần chỉnh trọng số để giảm loss. Phần nào không nhận gradient thì không học |
| **Teacher** | Bản sao đóng băng của model sau đợt trước, dùng để nhắc model mới về SKU cũ |
| **Classifier** | Phần cuối của model, quyết định box đó là SKU nào |
| **Encoder / decoder** | Hai khối giữa của Deformable DETR: encoder xử lý đặc trưng ảnh, decoder dùng các "query" để tìm từng vật |
| **Query** | Một "ô tìm kiếm" của decoder; mỗi query cố tìm một vật trong ảnh |
| **Checkpoint** | File lưu trọng số model tại một thời điểm, để chạy tiếp hoặc dùng lại |
| **Augmentation** | Biến đổi ảnh khi train (xoay, đổi màu, phóng to) để model học được nhiều trường hợp hơn |
| **CUDA kernel** | Đoạn code chạy trực tiếp trên GPU NVIDIA cho nhanh; chỉ ảnh hưởng tốc độ |
| **Batch** | Số ảnh model xem trước mỗi lần cập nhật trọng số |
