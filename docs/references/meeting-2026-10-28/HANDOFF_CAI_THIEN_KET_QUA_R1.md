# Handoff: hướng cải thiện sau FULL-SYN vòng 1

Ngày: 28/10/2026. Người viết: An.
Tài liệu này trả lời các câu hỏi mở của nhóm sau vòng 1 và đề xuất **việc cần làm tiếp, xếp theo thứ tự ưu tiên**.
Mọi số liệu lấy từ notebook `notebooks/training/05_full_syn_r1.ipynb`, plan `docs/project/full-syn-train-plan-2026-10-22.md`, báo cáo ảnh ghép `docs/data_preprocessing/reports/synth-v2-2026-10-21.md` và code `pdp/`. Chỗ nào là **giả thuyết** thì ghi rõ.

---

## 1. Tóm tắt trong 30 giây

- Vòng 1 **qua mọi cổng**. Sau task 5, test 6.003 ảnh đạt **cAcc 43,15%** (B2: 21,74%), AP50 94,9%, gần như **không quên** (−0,5 điểm). Mới chạy **1 seed**.
- **Phần học tăng dần không phải nút thắt.** Nhãn giả đúng ≥ 98,5%, AP50 không giảm qua các task.
- **Nút thắt là nhận sai từng vật trên ảnh thật.** Model nhận sai khoảng **6–7% số vật**; mỗi ảnh có khoảng 12 vật nên chỉ hơn 40% số ảnh đếm đúng hoàn toàn. Nguyên nhân chính là ảnh ghép chưa đủ giống ảnh thật, cộng với các SKU trông giống nhau.
- ⇒ Ưu tiên **sửa dữ liệu, cấu hình train và cách đếm**. **Chưa cần đổi phương pháp** (prompt, PDP).
- Có **3 điểm cấu hình rẻ chưa tận dụng**: learning rate không bao giờ giảm, augment đang tắt, ảnh chỉ 640px.

### Bảng kết quả vòng 1

| Sau task | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| val cAcc-M2 | 79,4% | 73,6% | 65,0% | 56,3% | **43,7%** |
| Ngưỡng tốt nhất của nhóm SKU task 1 | 0,83 | 0,68 | 0,53 | 0,37 | **0,10** |

- Test sau task 5, chia theo độ khó: easy 62,4%, medium 40,5%, hard 26,7%.
- Mốc tham khảo: **IncreACO** (học tăng dần, 183 + 17 SKU, có giữ ảnh cũ) đạt **74,3%**. **E0** (ảnh thật, học một lần đủ 200 SKU) đạt 83,6%.

---

## 2. Trả lời các câu hỏi của nhóm

### 2.1 Data: có cần làm ảnh ghép thật hơn nữa không?

**Có. Đây là hướng có lợi lớn nhất**, nhưng nên làm "thật hơn" theo thứ tự rẻ → đắt.

**Căn cứ**

| Căn cứ | Ý nghĩa |
|---|---|
| Tỉ lệ nhận sai mỗi vật khoảng 6,4–7,4%, gần như nhau ở easy, medium và hard. E0 chỉ sai 1,5%, IncreACO 2,4% | Lỗi nằm ở **nhận diện từng vật**, không phải do ảnh đông |
| AP50 val của FSA gần như phẳng từ epoch 3 (0,920 → 0,924) | Train thêm trên ảnh ghép không giúp gì nữa: model đã học hết những gì ảnh ghép dạy được |
| Chỉ có **5.877 cutout**, ít nhất 11 mỗi SKU. 72.000 ảnh ghép chỉ là sắp xếp lại các cutout này | Thiếu **đa dạng** (góc chụp, ánh sáng, bản thân vật), không phải thiếu số ảnh |

**Việc nên làm**

1. **S1. Cắt thêm cutout** từ train2019, mục tiêu 80–150 mỗi SKU. Ưu tiên vật trụ nằm ngang (camera 0). Đây là thay đổi **dự kiến có tác dụng lớn nhất**.
2. **S2. Mở rộng biến đổi màu và ánh sáng.** Hiện `global_gain` chỉ 0,96–1,04 và `object_color` 0,03 (`configs/data/synth_v2.json`). Một cách khác là chuyển phần màu và ánh sáng (tần số thấp) từ 208 ảnh nền sang ảnh ghép. Cách này giữ nguyên chữ in trên bao bì.
3. **S3. Bóng đổ mềm** dưới vật. Hiện vật ghép phẳng, không có bóng.
4. **S4. Vật lạ không nhãn** (ví dụ hộp bút, như trong ảnh thật). Cần nhóm **quyết định lại D15**, vì D15 hiện đang cấm.
5. **CycleGAN: để sau cùng.** Tốn khoảng 5–8 ngày và 45–70 giờ GPU. Cần ảnh thật không nhãn (phải tách từ val2019) và phải quyết định lại D20. Rủi ro: làm méo chữ, đổi màu bao bì, làm tăng nhầm SKU. Chỉ làm khi S1–S3 không đủ.

**Về số lượng 72.000 ảnh:** không gây hại nhưng nhiều khả năng **thừa**. Thêm ảnh không thêm đa dạng, nên nên dùng tiền cho cutout mới thay vì thêm ảnh. Xem thêm R4 ở mục 2.4.

### 2.2 Data: ảnh ghép có nên chỉnh lại độ chồng lên nhau không?

**Không nên tăng độ chồng. Nên xếp vật khít hơn.**

Hai khái niệm này khác nhau:
- **Độ chồng (che khuất):** vật này đè lên vật kia.
- **Độ khít (mật độ cụm):** các vật nằm sát nhau nhưng không đè.

| Chỉ số (easy / medium / hard) | Ảnh ghép | Ảnh thật (val) | Kết luận |
|---|---|---|---|
| Vật bị che rõ (đo bằng mask SAM) | 2,7 / 5,5 / 4,4% | 2,8 / 4,5 / 5,2% | **Đã khớp**, không cần chỉnh |
| Mật độ cụm (vật xếp khít) | 0,77 / 0,79 / 0,87 | 0,82 / 0,90 / 0,98 | **Ảnh ghép thưa hơn**, cần chỉnh |

Bản ảnh ghép đầu tiên từng cho vật đè nhau nhiều (15–19% vật bị che), sai so với ảnh thật nên đã được sửa thành bố cục hiện tại (o15).

**Việc nên làm**
- **S5.** Giảm `pack_gap` và tăng `compact_prob` để giỏ khít hơn, mục tiêu mật độ cụm ≈ ảnh thật.
- **D3** (làm trước S5, không tốn GPU): kiểm tra NMS 0,45 có đang xoá nhầm vật thật nằm sát nhau ở ảnh hard không, bằng cách quét IoU 0,3–0,7.

### 2.3 Model: prompt chung có nên đóng băng sau task 1 không?

**Nên thử như một thí nghiệm, nhưng không phải việc ưu tiên số 1.**

- **Paper PDP gốc không đóng băng.** Paper viết shared pool được *"progressively refined with data from new tasks"*, tức được tinh chỉnh tiếp ở mọi task. Code gốc cũng chỉ khoá prompt riêng (private) của task cũ. Vòng 1 làm giống paper (`--freeze_shared_after_task1 0`).
- **Code đã có sẵn cờ** `--freeze_shared_after_task1 1` (cấu hình `configs/exp/A8.sh`, có test `tests/test_pdp_i5_freeze_shared.py`).
- **Lý do nên thử:** ngưỡng tốt nhất tụt dần (0,83 → 0,10) dù AP50 không giảm. Nghĩa là model vẫn **xếp hạng đúng** nhưng **điểm tin cậy giảm đều**. *Giả thuyết:* các phần dùng chung (prompt chung, ở lr 1e-4; input_proj, bbox_embed, … ở lr 1e-5) trôi dần qua các task, trong khi hàng `class_embed` của SKU cũ bị khoá (`pdp/engine.py:532`). Đóng băng prompt chung là cách rẻ nhất để kiểm tra giả thuyết này.
- **Chi phí:** dùng lại checkpoint task 1 của vòng 1, chỉ chạy lại task 2–5 (khoảng 22 giờ, khoảng 9 USD).
- **Nên làm D2 trước** (miễn phí): vẽ phân bố điểm của box đúng và box sai qua 5 stage để xác nhận điểm có thực sự trôi không.

Các hướng đổi phương pháp khác, **chưa ưu tiên**:
- **P2IOD** (CVPR'26, thay prompt pool bằng MLP và gộp model): paper chỉ báo AP50 và không so với PDP, trong khi nút thắt hiện tại không nằm ở prompt.
- **Mở khoá một phần decoder** ở task ≥ 2 với lr nhỏ: SKU mới học kém hơn SKU cũ (ở task 4, AP50 0,906 so với 0,957).

### 2.4 Model: xem lại ngưỡng theo nhóm, learning rate và các tham số

| ID | Vấn đề | Đề xuất | Chi phí |
|---|---|---|---|
| **R1** | **Learning rate không bao giờ giảm.** `--lr_drop` mặc định là 40 epoch (`pdp/main.py:89`), nhưng FSA chỉ chạy 8 epoch và PDP 16 epoch mỗi task. B2 cũng bị như vậy | Giảm lr ×0,1 ở khoảng 75% số epoch (FSA: epoch 3 trên 4; PDP: epoch 12 trên 16). Deformable DETR thường tăng AP rõ sau lần giảm lr, và điểm tin cậy cũng sắc hơn | Chỉ đổi cờ |
| **R2** | **Augment đang tắt** (`--augment 0`), nên mỗi ảnh được xem y hệt nhau 16 lần | Bật `--augment 1` (xoay 90°, chỉnh màu nhẹ, nhiều cỡ ảnh). Code có sẵn ở `pdp/augment.py` | Chỉ đổi cờ |
| **R3** | Ảnh 640px làm chữ in bị mờ, trong khi nhiều SKU chỉ khác nhau ở chữ | Thử **800px**. **Không lên 1800px**: quá chậm, tốn bộ nhớ, lệch với ảnh pretrain | Chậm hơn khoảng 1,5 lần |
| **R4** | Train thừa epoch: AP50 của FSA phẳng từ epoch 3 | FSA chạy 3–4 epoch thay vì 8. Với PDP, xem đường val theo epoch trong log vòng 1 rồi quyết định | Tiết kiệm 30–50% thời gian GPU |
| **T1** | **Ngưỡng theo nhóm chỉ thêm khoảng +0,5 điểm** (43,15 → 43,68%). Ngưỡng ở task 5 chạm đáy lưới 0,10 | Hạ đáy lưới xuống 0,01. Chọn ngưỡng trên nửa val, chấm trên nửa còn lại để biết số val lạc quan đến đâu. Không kỳ vọng tăng nhiều | Chỉ CPU |
| **T2** | NMS 0,45 có thể xoá nhầm vật thật nằm sát nhau | Quét NMS IoU 0,3–0,7 (D3) | Chỉ CPU |
| **T3** | Chưa dùng TTA | Dự đoán thêm 3 bản xoay 90° rồi gộp kết quả; thường thêm 0,5–2 điểm | Chỉ thời gian dự đoán |

### 2.5 Chạy: chỉ 5 task có đủ để đánh giá continual learning không?

**Đủ cho kết quả chính. Nên có thêm một chuỗi dài hơn làm thí nghiệm phụ.**

- **5 task là chuẩn trong các paper cùng lĩnh vực:**
  - PDP và P2IOD dùng COCO 40 + 10×4 (5 task).
  - VOC dùng tối đa 4 task.
  - IncreACO chỉ có 2 giai đoạn (183 + 17).

  Giao thức 100 + 4×25 của nhóm **không ngắn hơn** các paper này.
- **Nhưng chính kết quả cho thấy cần kiểm tra chuỗi dài hơn.** cAcc giảm đều mỗi task (79 → 44%) và ngưỡng đã chạm đáy 0,10 ở task 5. Một chuỗi dài hơn sẽ cho biết model **có tiếp tục đứng vững không**.
- **Đề xuất:** sau vòng 2, chạy thêm **100 + 10×10** (11 task) làm thí nghiệm phụ. Task 1 giống hệt hiện tại nhờ chia task theo seed 0, nên **dùng lại được FSA và task 1**. Chỉ cần sinh lại ảnh ghép cho task 2–11 (khoảng 2 giờ CPU) và train 10 task nhỏ (*ước tính* 25–35 giờ GPU).
- **Lưu ý cho task 6 (ảnh nhóm tự chụp):** model có 224 ô lớp (200 SKU RPC + 24 ô dự phòng), nên **thêm tối đa 24 SKU mới** mà không phải sửa kiến trúc.

### 2.6 Các hướng cải thiện khác (chưa nằm trong các câu hỏi trên)

Đã kiểm trong code để loại những thứ đã làm rồi:
- NMS khi đếm đã gộp không phân biệt lớp (`autocheckout/counting.py`).
- Ảnh ghép đã có nén JPEG, nhiễu và làm mờ (`configs/data/synth_v2.json`).

| ID | Việc | Vì sao có thể giúp | Chi phí | Ưu tiên |
|---|---|---|---|---|
| **C1** | **Cân lại điểm giữa lớp cũ và lớp mới sau train.** Có hai cách: Weight Aligning (chuẩn hoá độ lớn trọng số `class_embed` giữa các nhóm task), hoặc cộng một hệ số bù cho mỗi nhóm, chọn trên val | Ngưỡng theo nhóm chỉ quyết định "giữ hay bỏ box". Khi SKU cũ và SKU mới **tranh nhau cùng một vật**, ngưỡng không đổi được lớp thắng, còn cân lại điểm thì đổi được. Nhắm thẳng vào hiện tượng điểm tụt qua các task | Chỉ CPU | **Cao**, sau D1 và D2 |
| **C2** | **Bộ phân loại lại vùng cắt**: cắt từng vật ở độ phân giải gốc, đưa qua một model phân loại nhỏ để chọn lại SKU, chỉ dùng cho các cặp hay nhầm | Giải quyết chữ mờ ở 640px mà không phải tăng cả ảnh lên 1800px | Train một model nhỏ | Cao, nếu D1 cho thấy nhầm do chữ |
| **S6** | **Ảnh ghép tập trung vào cặp hay nhầm**: đặt các SKU giống nhau vào cùng ảnh và cho xuất hiện nhiều hơn | Model phải học cách phân biệt đúng những cặp nó hay sai | Sinh lại ảnh | Cao, sau D1 |
| **S7** | **Tăng tần suất SKU yếu**: SKU có AP thấp hoặc ít cutout thì xuất hiện nhiều hơn | Hiện SKU xuất hiện theo phân bố giỏ hàng, chưa tính SKU nào học kém | Sinh lại ảnh | Trung bình |
| **S8** | **Rà cutout bằng mắt** | Chưa ai xem hết cutout. Mép do SAM tạo có thể dính khay hoặc cắt mất bao bì | Vài giờ công người | Trung bình, rẻ |
| **S9** | **Kiểm tra lỗi ở mép vật ghép**: model có thể nhận vật nhờ viền cắt, thứ ảnh thật không có | Nếu D1 cho thấy nhiều lỗi sót vật: thử làm mép mềm hơn, ăn mòn mép ngẫu nhiên, hoặc hoà trộn Poisson | Sinh lại ảnh | Thấp, chỉ khi D1 gợi ý |
| **R5** | **Lấy trung bình trọng số các epoch cuối** (EMA / checkpoint averaging) | Thường làm điểm ổn định và sắc hơn | Ít | Trung bình |
| **M5** | **Cho backbone học thêm ở task sau**: mở tầng cuối của backbone với lr rất nhỏ, có teacher giữ kiến thức cũ | FSA chỉ fine-tune backbone trên SKU task 1, nên backbone chưa thấy SKU task 2–5. Có thể là lý do SKU mới học kém (AP50 0,906 so với 0,957) | Train lại | Trung bình, có rủi ro quên |
| **P1** | **Tự gán nhãn trên ảnh thật không nhãn** (ví dụ chia đôi val2019: một nửa để thích nghi, một nửa để chọn model) | Cách mạnh nhất để thu hẹp khoảng cách ảnh ghép – ảnh thật | Train lại | **Đổi giao thức**: cả nhóm phải quyết |
| **E1** | **Khoảng tin cậy cho kết quả test** (bootstrap) | Luận văn cần báo kèm ±. Hiện chỉ có ước lượng ±0,65 điểm | Chỉ CPU | Làm khi viết báo cáo |

**D1 (phân rã lỗi) quyết định nên làm hướng nào:**

| Kết quả D1 cho thấy | Nên làm |
|---|---|
| Lỗi chủ yếu là **nhầm giữa SKU cũ và SKU mới** | C1 (rẻ nhất, chỉ CPU) |
| Lỗi chủ yếu là **nhầm giữa các SKU giống nhau trong cùng nhóm** | S6, C2, R3 (800px) |
| Lỗi chủ yếu là **bỏ sót vật** | D3 (quét NMS), S5 (giỏ khít hơn), S9 |
| Lỗi chủ yếu là **đếm thừa** | D3, T1 (ngưỡng) |

---

## 3. Phát hiện thêm cần nhóm biết

1. **Bảng test stage 1–4 đang dùng M1, nên FULL-SYN bị phạt** (stage 1: 28,65% so với B2 73,16%).
   - Ảnh ghép không chứa SKU tương lai, nên model không học cách bỏ qua vật lạ.
   - B2 được lợi vì ảnh thật có sẵn SKU tương lai làm nền.
   - Theo D16, M2 mới là chỉ số đúng. **Cần tính M2 trên test cho stage 1–4** (D4, chỉ CPU).
2. **Chưa chứng minh được ảnh ghép tốt hơn ảnh thật.**
   - E4 tháng 9 (ResNet-50, ảnh thật, FSA, augment, I3, lọc trùng sau dự đoán) đạt **42,4%** trên cùng 6.003 ảnh test, gần bằng FULL-SYN.
   - Nhiều khả năng phần hơn B2 đến chủ yếu từ **FSA, I3 và lọc trùng (F14)**, không phải từ ảnh ghép.
   - ⇒ Cần thí nghiệm **FULL-REAL** (mục 5).
3. **Ảnh ghép task ≥ 2 dùng lại cutout của SKU cũ.**
   - Thực chất đây là một dạng giữ mẫu cũ (exemplar augmentation, giống IncreACO), và có thể là lý do chính khiến model gần như không quên.
   - Luận văn **phải ghi rõ** điểm này, không nên gọi là hoàn toàn "replay-free".
4. **Mới có 1 seed.**
   - Seed train đang cố định `--seed 0` trong `configs/exp/native/common.sh:23`.
   - Cần đưa thành tham số và chạy thêm seed 1, 2 cho cấu hình cuối.

---

## 4. Danh sách việc theo thứ tự ưu tiên

Cột **Ai** và **Trạng thái** để nhóm điền.

| # | ID | Việc | Cần | Chi phí | Ai | Trạng thái |
|---|---|---|---|---|---|---|
| 0 | – | **Kéo `data/training/full-syn-r1/` (7,6 GB) về và backup.** Mọi việc phân tích đều cần thư mục này | – | – | | |
| 1 | D1 | **Phân rã lỗi task 5 trên val:** sót vật, đếm thừa, nhầm lớp (cùng hay khác nhóm hàng). Tách theo độ khó, nhóm hàng, vật nằm/đứng, số cutout của SKU. Liệt kê các cặp SKU hay nhầm | CPU | vài giờ | | |
| 2 | D2 | Phân bố điểm của box đúng và box sai qua 5 stage | CPU | vài giờ | | |
| 3 | D3, D4, T1, T3, **C1** | Quét NMS; M2 trên test stage 1–4; kiểm ngưỡng trên nửa val; TTA; cân lại điểm lớp cũ/mới | CPU / dự đoán | 1–2 ngày | | |
| 4 | **S8** | Rà cutout bằng mắt, loại cutout lỗi | Người | vài giờ | | |
| 5 | R1–R5 | Thử từng thay đổi bằng **FSA 3–4 epoch** (khoảng 2–2,5 giờ, khoảng 1 USD mỗi lần), so bằng val cAcc-M2 | GPU | khoảng 5 USD | | |
| 6 | S1–S9 | Sinh lại ảnh ghép, **tối đa 2 thay đổi mỗi lần**, chọn theo kết quả D1, đo bằng FSA ngắn | CPU + GPU | vài ngày | | |
| 7 | C2 | Bộ phân loại lại vùng cắt (nếu D1 cho thấy nhầm do chữ) | GPU | 1–2 ngày | | |
| 8 | r2 | Gộp các thay đổi thắng, chạy đủ 5 task, **chấm test một lần** | GPU | khoảng 35 giờ, khoảng 15 USD (ít hơn nếu có R4) | | |
| 9 | I5, M5 | Đóng băng prompt chung; mở tầng cuối backbone ở task sau. Chạy lại task 2–5 | GPU | khoảng 22 giờ mỗi thí nghiệm | | |
| 10 | – | FULL-REAL, seed 1–2, chuỗi 100 + 10×10, E1 | GPU / CPU | xem mục 5 | | |
| – | P1 | Tự gán nhãn trên ảnh thật: **chờ nhóm quyết định đổi giao thức** | – | – | | |

---

## 5. Thí nghiệm cần có để luận văn đứng vững

| Thí nghiệm | Trả lời câu hỏi |
|---|---|
| **FULL-REAL**: B2 + FSA + I3 + F14 trên ảnh thật | Ảnh ghép có thật sự hơn ảnh thật không? |
| FULL-SYN học một lần 200 SKU | Mất bao nhiêu điểm do ảnh ghép, bao nhiêu do học tăng dần (so với E0 83,6%) |
| FULL-SYN tắt I3 / F14 | Mỗi bản sửa đóng góp bao nhiêu |
| 3 seed cho cấu hình cuối | Kết quả có ổn định không, sai số bao nhiêu |
| 100 + 10×10 | Model có đứng vững khi số task tăng không |

---

## 6. Quy tắc cần giữ

- **Test chỉ chấm một lần mỗi vòng.** Mọi lựa chọn (ngưỡng, NMS, TTA, số epoch, thay đổi dữ liệu) phải chọn **trên val**. Không chấm lại test của vòng 1 với cách đếm mới rồi chọn bản tốt nhất.
- **Mỗi lần chỉ đổi ít thứ** (tối đa 2 thay đổi), để biết thay đổi nào có tác dụng.
- Báo cáo **M2 làm chỉ số chính**, kèm M1 để minh bạch.
- Ghi rõ trong luận văn: chỉ 1 seed (cho đến khi có thêm); và ảnh ghép có dùng lại cutout SKU cũ.

## 7. Những điều còn là giả thuyết

| Giả thuyết | Kiểm bằng |
|---|---|
| Điểm tin cậy tụt do các phần dùng chung trôi | D2, I5 |
| Lỗi chính là nhầm giữa các SKU giống nhau, đặc biệt SKU khác nhau ở chữ in | D1 |
| Thêm cutout có tác dụng lớn nhất | S1 với FSA ngắn |
| FULL-REAL ≥ FULL-SYN | FULL-REAL |
| Giảm lr và bật augment sẽ tăng điểm | R1, R2 |
| SKU cũ và SKU mới tranh nhau cùng một vật, nên cân lại điểm sẽ giúp | D1, C1 |
| Backbone chỉ thích nghi với SKU task 1, nên SKU mới học kém hơn | M5 |
| Model nhận vật nhờ viền cắt của ảnh ghép | D1, S9 |
