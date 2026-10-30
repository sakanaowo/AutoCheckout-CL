# Phân tích kết quả FULL-SYN vòng 1 — 29/10/2026

Người làm: さかな (với Claude).

Tài liệu này kiểm lại các giả thuyết trong [handoff họp 28/10 của An](../references/meeting-2026-10-28/HANDOFF_CAI_THIEN_KET_QUA_R1.md) bằng chính kết quả của vòng 1 ([plan mục 11](full-syn-train-plan-2026-10-22.md#11-kết-quả-vòng-1-2710)). Mọi phép kiểm chạy trên máy local (CPU và RTX 3060), không train lại, không chấm lại test để chọn cấu hình.

**Hướng của nhóm:** giữ phương pháp của paper (PDP), nhưng đưa kết quả lên mức dùng được cho production. Vì vậy tài liệu không xét hướng làm ảnh ghép "thật hơn" (bóng đổ, màu sắc, CycleGAN).

## Đọc nhanh

1. **Cách chấm điểm đúng.** Số đo trên val gần như trùng số đo trên test, nên chọn cấu hình dựa trên val là tin được.
2. **Chỉnh cách đếm không giúp thêm.** Đổi ngưỡng hay đổi cách lọc box trùng chỉ thay đổi kết quả khoảng ±0,3 điểm.
3. **Lỗi lớn nhất là nhận nhầm sản phẩm** (đúng vị trí nhưng sai SKU). Nếu hết lỗi này, tỉ lệ đếm đúng cả giỏ lên từ 43% đến khoảng 67%.
4. **Nhầm xảy ra giữa các SKU anh em** (cùng dòng sản phẩm, khác vị), và chỉ xuất hiện khi SKU anh em được học ở đợt sau. Đặc trưng mà model dùng để phân biệt không tách được các cặp này: khi nhầm, SKU đúng thường xếp thứ hai, sát ngay sau SKU sai.
5. **Không sửa được sau khi train.** Đã thử 6 cách: chỉnh ngưỡng, chỉnh bước lọc box trùng, cân điểm theo nhóm task, trả phần dùng chung về task 1, xếp lại bằng prototype của PDP, và dự đoán ở ảnh to hơn. Không cách nào tăng cAcc. Muốn tăng phải train lại.
6. **Các giả thuyết khác của handoff không đứng vững:** SKU cũ và mới "tranh nhau", điểm tin cậy tụt làm model sai, và E4 đã đạt 42,4% trên cùng tập test.

Bản trước của tài liệu này cho rằng nguyên nhân chính là "bộ ảnh mẫu chỉ có một mặt sản phẩm". Kiểm thêm thì điều này **sai**; xem mục 3.3.

**Việc nên làm tiếp:** phải train lại. Các hướng và cách chọn ở mục 7. Đã chọn chạy thử hướng A (M5) trên riêng task 2 trước; code, cách chấm và mức lệch so với paper ở mục 8. **Kết quả 30/10: `stop`** (cAcc-M2 +1,21 điểm, ngưỡng là +3; không quên thêm SKU task 1), nên chuyển sang hướng B (mục 8.6).

## Một số khái niệm dùng trong tài liệu

| Từ | Nghĩa |
|---|---|
| Task / stage | Model học 5 đợt: 100 SKU, rồi 4 đợt mỗi đợt thêm 25 SKU. "Stage 3" là model sau khi học xong đợt 3 |
| cAcc | Tỉ lệ ảnh mà model đếm đúng hết mọi món (như hoá đơn đúng từng dòng). Chỉ số chính |
| M1, M2 | Hai cách chấm cAcc khi model chưa học hết 200 SKU (giải thích ở mục 1) |
| Ngưỡng | Mỗi box có một điểm tin cậy từ 0 đến 1; chỉ box có điểm trên ngưỡng mới được đếm |
| NMS | Bước bỏ các box trùng nhau trên cùng một vật, để một vật không bị đếm hai lần |
| Nhầm / sót / thừa | Nhầm: khoanh đúng vật nhưng sai SKU. Sót: không đếm vật đó. Thừa: đếm một vật không có |

## 1. Cách chấm có đúng không (câu hỏi của Minh)

**Câu hỏi:** sau mỗi task, val có chấm luôn cả những SKU model chưa học không?

**Trả lời:** có hai cách chấm, và nhóm dùng cách đúng để quyết định.

- **M1** chỉ đếm SKU đã học, nhưng vẫn phạt khi model khoanh nhầm một sản phẩm chưa học thành SKU đã học. Ảnh ghép không bao giờ có sản phẩm lạ, nên model chưa học cách bỏ qua chúng, và M1 rất thấp ở task 1–4. Minh thấy đúng ở chỗ này.
- **M2** bỏ qua mọi box nằm trên sản phẩm chưa học. Ngoài đời, sản phẩm cửa hàng chưa bán thì không có trên quầy, nên M2 mới là chỉ số phản ánh thực tế. **Các trạm kiểm tra của vòng 1 đều dùng M2.**
- Ở task 5 model đã học đủ 200 SKU, nên M1 và M2 bằng nhau.

Về dữ liệu train (ý của An): ảnh ghép của task t có khoảng một nửa là SKU mới (có nhãn) và một nửa là SKU cũ (không nhãn; model của task trước tự gán nhãn). Ví dụ task 3 có 148.229 vật, trong đó 74.062 vật có nhãn.

**Kiểm thêm:** chấm M2 trên test (dùng ngưỡng đã chọn trên val, không chỉnh gì thêm):

| Sau task | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| M2 trên test | 79,4% | 73,5% | 64,8% | 55,9% | 44,0% |
| M2 trên val | 79,4% | 73,6% | 65,0% | 56,3% | 43,7% |
| M1 trên test (số trong plan) | 28,0% | 24,2% | 22,1% | 25,7% | 44,0% |

Val và test lệch nhau dưới 0,5 điểm.

## 2. Chỉnh cách đếm có giúp không

**Cách kiểm:** chia val làm hai nửa. Chọn ngưỡng trên nửa thứ nhất, chấm trên nửa thứ hai, lặp lại 5 lần với 5 cách chia khác nhau. Cách này cho biết một thay đổi có giúp thật hay chỉ khớp với dữ liệu dùng để chọn.

| Cách đếm (sau task 5) | cAcc trên nửa giữ lại |
|---|---|
| Một ngưỡng chung (như B2) | 42,7% |
| Mỗi nhóm SKU một ngưỡng (vòng 1 dùng cách này) | 43,0% |
| Mỗi nhóm một ngưỡng, cho phép ngưỡng thấp tới 0,01 | 43,0% |
| Không lọc box trùng | 36,3% |
| Lọc box trùng, nhiều mức khác nhau | 38,4–42,9%; mức đang dùng là tốt nhất |

Mỗi con số dao động khoảng ±0,3 điểm giữa các lần chia. **Kết luận:** các cách đếm cho kết quả gần như nhau, không còn gì để lấy thêm. Số trên val chỉ lạc quan khoảng 1 điểm.

## 3. Model sai ở đâu

### 3.1 Loại lỗi

Sau task 5, val có 71.108 vật trên 5.792 ảnh, trung bình 12 vật mỗi ảnh.

- 92,8% số vật được đếm đúng.
- 4,8% bị **nhầm SKU**: 2,1% nhầm sang SKU cùng nhóm hàng (sữa thành sữa khác), 2,7% sang nhóm hàng khác.
- 2,4% bị **sót**.
- Ngoài ra có thêm số box **thừa** bằng 1,7% số vật.

Lỗi gần như đều ở mọi độ khó của ảnh (easy 6,8%, medium 6,8%, hard 7,8%). Lỗi tập trung vào một số SKU: 20 SKU tệ nhất chiếm 39% tổng số lỗi.

Nếu xoá được hết từng loại lỗi thì cAcc lên được bao nhiêu:

| Nếu không còn | cAcc sau task 5 |
|---|---|
| (hiện tại) | 43% |
| sót | 51% |
| thừa | 48% |
| sót và thừa | 60% |
| **nhầm SKU** | **67%** |

**Kết luận:** nhầm SKU là chỗ cần sửa nhất.

### 3.2 Có phải SKU cũ và SKU mới "tranh nhau" không

Handoff đoán rằng khi học SKU mới, model kéo vật cũ sang lớp mới (đề xuất C1: cân lại điểm giữa lớp cũ và lớp mới).

**Kiểm:** nếu đúng thì phần lớn lỗi phải là "vật cũ bị gọi tên SKU mới".
- Thực tế hai chiều cân bằng: vật cũ gọi thành SKU mới 1.412 lần, vật mới gọi thành SKU cũ 1.522 lần.
- Tỉ lệ nhầm sang SKU của đợt khác chỉ hơi cao hơn mức nhầm ngẫu nhiên (90% so với 84%).

Các cặp hay nhầm nhất là **cùng dòng sản phẩm, khác vị**: 98_milk với 97_milk, 74_drink với 73_drink, 36_instant_drink với 34_instant_drink.

**Kết luận:** giả thuyết "tranh nhau" gần như sai. Model nhầm vì sản phẩm trông giống nhau.

### 3.3 Có phải do bộ ảnh mẫu chỉ có một mặt sản phẩm không

**Giả thuyết ban đầu.** Ảnh ghép được cắt từ ảnh mẫu của RPC (mỗi ảnh một sản phẩm trên bàn xoay). Với 79 trong 201 SKU, RPC không chụp mặt sau. Xem [ảnh đối chiếu](../../runs/analysis_r1/confusions.png): ảnh mẫu của 98_milk và 97_milk chỉ có mặt sau, còn trên quầy thật hộp hay ngửa mặt trước. Sau task 5, nhóm SKU một mặt nhầm 7,4%, nhóm hai mặt chỉ 3,2%.

**Kiểm đối chứng.** Nếu nguyên nhân là thiếu ảnh mặt còn lại, thì:
- ngay sau khi học xong (task 1), nhóm một mặt đã phải nhầm nhiều hơn;
- B2, train trên ảnh quầy thật nên thấy đủ mọi mặt, thì không được có khoảng chênh này.

| | Nhầm SKU: có hai mặt | Nhầm SKU: một mặt |
|---|---|---|
| FULL-SYN, ngay sau task 1 (SKU task 1) | 0,8% | 0,8% |
| B2, ngay sau task 1 (SKU task 1) | 0,9% | 0,4% |
| FULL-SYN sau task 5 | 3,3% | 7,1% |
| B2 sau task 5 | 12,2% | 16,7% |

(Đo trên 6.003 ảnh test cũ để so được với B2, ngưỡng chọn trên val của từng model.)

**Kết luận:** giả thuyết sai.
- Ngay sau khi học, nhóm một mặt nhầm không nhiều hơn.
- Khoảng chênh chỉ xuất hiện về sau, và B2 (đã thấy đủ mọi mặt) cũng có khoảng chênh cỡ đó (khoảng 4 điểm).

Như vậy nhóm SKU một mặt vốn có nhiều "anh em" giống mình hơn. Lỗi xảy ra khi các SKU anh em đó được học ở đợt sau, không phải vì thiếu ảnh. Hệ quả: cắt thêm cutout hay tải thêm camera không phải là hướng sửa chính.

### 3.4 Điểm tin cậy tụt dần có làm model sai không

Handoff dẫn dãy ngưỡng 0,83 → 0,10 qua các task làm bằng chứng điểm tin cậy tụt mạnh, và đề xuất khoá các phần dùng chung của model (I5).

**Kiểm 1, đọc lại số.** Dãy 0,83 → 0,10 là ngưỡng chọn theo M1. Ở các task đầu, ngưỡng M1 bị đẩy cao để không đếm nhầm sản phẩm chưa học. Ngưỡng theo M2 đã thấp (0,10–0,32) ngay từ task 1. Điểm tin cậy có tụt thật, nhưng vừa phải: điểm thường gặp của một box đúng thuộc SKU task 1 giảm từ 0,95 xuống 0,82 sau 4 task.

**Kiểm 2, thử trực tiếp.** Lấy model sau task 5, trả các phần dùng chung về đúng giá trị sau task 1, rồi chạy lại trên 1.500 ảnh val bằng RTX 3060:

| Biến thể | cAcc | Điểm thường gặp của box đúng |
|---|---|---|
| Model sau task 5 (gốc) | 43,1% | 0,76–0,82 |
| Trả bộ chọn prompt (`query_tf`) về task 1 | 43,1% | không đổi |
| Trả thêm prompt dùng chung về task 1 | 43,1% | 0,94–0,95 |

cAcc trong bảng là cAcc sau khi chọn lại ngưỡng cho từng biến thể.

**Kết luận:**
- Prompt dùng chung quyết định thang điểm: trả về task 1 thì điểm cao lại như lúc đầu.
- Nhưng số lỗi không giảm: SKU task 4 bớt nhầm, SKU task 5 lại nhầm nhiều hơn. Điểm tụt chỉ là thang điểm thay đổi, và việc chọn ngưỡng đã bù được.
- Phép thử này chưa phải là train lại với các phần dùng chung bị khoá, nên chưa loại hẳn I5, nhưng kỳ vọng thấp.

### 3.5 Vì sao cAcc giảm từ 79% xuống 44% qua các task

Có hai lý do cộng lại:
- **Mỗi ảnh có nhiều vật cần đếm đúng hơn.** Sau task 1, mỗi ảnh có khoảng 6 vật thuộc SKU đã học; sau task 5 là 12 vật. Muốn ảnh đúng thì mọi vật đều phải đúng, nên càng nhiều vật càng khó. Lý do này chiếm khoảng 1/3 mức giảm.
- **Nhận từng vật kém dần.** Tỉ lệ lỗi mỗi vật tăng từ 4,5% lên 9,0%. Lý do này chiếm khoảng 2/3 còn lại.

**Mốc cho production:** với 12 vật mỗi ảnh, muốn đếm đúng 80% số giỏ thì mỗi vật chỉ được sai khoảng 1,8%. Hiện tại là 9%, tức phải giảm lỗi khoảng 5 lần.

### 3.6 Vì sao model nhầm các SKU anh em

**Kiểm 1, model có "thấy" SKU đúng không.** Với mỗi box bị gán sai SKU sau task 5, xem SKU đúng đứng thứ mấy trong các lựa chọn của model cho box đó:

| SKU đúng đứng | Tỉ lệ box nhầm |
|---|---|
| Thứ hai | 59% |
| Thứ ba trở xuống | 33% |
| Không có trong các lựa chọn | 8% |

Khi SKU đúng đứng thứ hai, điểm của nó thường bằng khoảng 72% điểm của SKU bị chọn. Model gần đúng, nhưng không tách được hai SKU.

**Kiểm 2, đặc trưng có tách được không.** PDP lưu sẵn một "prototype" cho mỗi SKU: đặc trưng trung bình của SKU đó. Prototype của các cặp hay nhầm gần như trùng nhau. Độ giống (cosine) của 98/97_milk là 0,94, của 74/73_drink là 0,88, trong khi hai SKU bất kỳ thường chỉ khoảng 0,55. Nói cách khác, ở tầng đặc trưng, model "nhìn" hai SKU này gần như là một.

**Thử sửa sau khi train (đều không được):**

| Cách thử | cAcc trên nửa val giữ lại |
|---|---|
| Cân điểm theo nhóm task (C1) | giảm 0,3–0,9 điểm |
| Xếp lại hai SKU đứng đầu theo độ giống với prototype | giảm 0,5–0,8 điểm hoặc giữ nguyên |
| Dự đoán ở ảnh 800px thay vì 640px (model train ở 640px) | giảm từ 42% xuống 22% |

Ảnh 800px làm model kém đi rõ rệt vì model không quen cỡ ảnh này. Dù vậy, số lần nhầm 98_milk thành 97_milk giảm từ 86 xuống 67 trên 142 hộp. Đây là dấu hiệu yếu rằng ảnh to hơn giúp phân biệt chữ in, nếu model được train ở cỡ đó.

**Kết luận:** lỗi nằm ở đặc trưng của model. Trong PDP, backbone, encoder và decoder bị khoá sau FSA, mà FSA chỉ được train trên 100 SKU của task 1. Các SKU anh em học ở đợt sau vì vậy phải phân biệt bằng đặc trưng chưa từng được chỉnh cho chúng. Đây vẫn là giả thuyết; muốn kiểm phải train lại (mục 7).

### 3.7 Model có tự biết giỏ nào mình không chắc không

Ngoài đời, có thể để model tự làm những giỏ nó chắc, còn giỏ khó thì chuyển cho nhân viên. Kiểm sau task 5: xếp giỏ theo mức chắc chắn (box có điểm sát ngưỡng, hoặc hai SKU có điểm gần nhau trên cùng một box), rồi xem các giỏ chắc nhất đúng bao nhiêu.

| Model tự làm | Chuyển nhân viên | Số giỏ model làm đúng |
|---|---|---|
| 100% số giỏ | 0% | 44% |
| 50% giỏ chắc nhất | 50% | 65% |
| 20% giỏ chắc nhất | 80% | 78% |
| 10% giỏ chắc nhất | 90% | 78% |

**Kết luận:** model sai mà vẫn tự tin. Vì vậy hiện chưa dùng được cách chuyển giỏ khó cho nhân viên.

## 4. Cấu hình train

- **Learning rate không giảm lần nào** trong cả vòng 1, ở cả FSA lẫn PDP. Code gốc của paper cũng vậy. Deformable DETR thường giảm learning rate 10 lần ở gần cuối, và việc này hay giúp tăng điểm (R1).
- **PDP task 1 chạy 16 epoch là thừa:** AP50 trên val gần như không đổi từ epoch đầu (0,928 → 0,934). Task 3 và 5 dừng tăng ở khoảng epoch 10–12. Bớt epoch tiết kiệm được giờ GPU (R4).

## 5. Một chỗ sai trong handoff

Handoff viết E4 (tháng 9, ảnh thật) đạt 42,4% "trên cùng 6.003 ảnh test". Thực ra E4 được chấm trên tập test của cách chia dữ liệu tháng 9, là một file khác. Vì vậy chỉ nói được E4 cùng tầm với FULL-SYN, không so trực tiếp được.

## 6. Xếp lại các đề xuất của handoff

| Đề xuất | Sau khi kiểm | Ưu tiên |
|---|---|---|
| FULL-REAL: cùng phương pháp, train trên ảnh quầy thật | Lý do "thiếu mặt sản phẩm" đã sai (mục 3.3). Vẫn cần để trả lời câu hỏi ảnh ghép hay ảnh thật tốt hơn | Trung bình |
| R1 giảm learning rate, R4 bớt epoch | Đã xác nhận learning rate không giảm, task 1 thừa epoch. Chỉ cần đổi cờ, ghép được vào FULL-REAL | **Cao** |
| R3 ảnh 800px | Dự đoán ở 800px với model 640px giảm nhầm 98/97_milk (mục 3.6). Cần train ở 800px mới biết | **Cao** |
| R2 bật augment | Chưa kiểm được ở local | Trung bình |
| I5 khoá phần dùng chung | Điểm tụt không làm model sai thêm (mục 3.4) | Thấp |
| C1 cân điểm, xếp lại bằng prototype | Đã thử, không giúp (mục 3.6) | Bỏ |
| S1 cắt thêm ảnh mẫu | Lỗi không do thiếu mặt sản phẩm (mục 3.3) | Thấp |
| M5 cho backbone/encoder học thêm ở task sau | Đặc trưng không tách được SKU anh em học ở đợt sau (mục 3.6) | **Cao** |
| T1, D3 chỉnh ngưỡng, lọc box trùng | Không giúp (mục 2) | Bỏ |
| Làm ảnh ghép thật hơn (S2, S3, CycleGAN) | Theo hướng của nhóm, và không sửa được lỗi chính | Bỏ |

## 7. Các hướng train lại

Cả ba hướng đều giữ phương pháp của paper (PDP, học tăng dần bằng nhãn giả), và đều nên ghép thêm hai chỉnh sửa rẻ: giảm learning rate ở cuối mỗi task (R1) và bớt epoch thừa (R4).

| Hướng | Làm gì | Nhắm vào | Chi phí GPU (4090) | Rủi ro |
|---|---|---|---|---|
| **A. Đặc trưng học tiếp (M5)** | Ở task 2–5, mở khoá tầng cuối backbone và encoder với learning rate nhỏ; model thầy giữ kiến thức cũ | Đặc trưng chưa từng được chỉnh cho SKU học sau (mục 3.6) | Dùng lại FSA và task 1; chạy lại task 2–5, khoảng 22 giờ ở tốc độ vòng 1 (cộng phần chậm hơn, mục 8.4) | Lệch paper (paper khoá các phần này); có thể quên nhiều hơn |
| **B. Ảnh 800px (R3)** | Train lại cả vòng ở 800px | Chữ in trên bao bì bị mờ ở 640px | Cả vòng, khoảng 50 giờ (chậm hơn khoảng 1,5 lần) | Không đổi phương pháp; tốn nhất |
| **C. FULL-REAL** | Cùng cấu hình vòng 1, train trên ảnh quầy thật (split B2) | Trả lời câu hỏi ảnh ghép hay ảnh thật tốt hơn | Khoảng 30–40 giờ | Không trực tiếp sửa lỗi SKU anh em |

Đề xuất: làm **A** trước. Rẻ nhất, nhắm đúng nguyên nhân tìm được, và nếu A không giúp thì giả thuyết "đặc trưng bị khoá" sai, lúc đó chuyển sang B.

## 8. Chạy thử M5 trên task 2

Ngày 29/10 nhóm chọn thử hướng A trước, nhưng chỉ chạy **task 2** để biết giả thuyết "đặc trưng bị khoá" đúng hay sai trước khi tốn cả vòng.

### 8.1 Lệch so với paper bao nhiêu

Đếm trên checkpoint của vòng 1. Model có 130,5M tham số.

| Phần | Tham số | Paper (như vòng 1) | M5 |
|---|---|---|---|
| Backbone, 3 tầng đầu | 59,7M | Khoá | Khoá |
| Backbone, tầng cuối (`stages_3`) | 27,5M | Khoá | Train, `lr_old` = 1e-5 |
| Encoder | 4,5M | Khoá | Train, `lr_old` = 1e-5 |
| Decoder | 6,1M | Khoá | Khoá |
| Prompt (mỗi task một bộ riêng) | 29,0M | Train | Train |
| Phần dùng chung khác (`input_proj`, `bbox_embed`…) | 3,1M | Train, `lr_old` | Train, `lr_old` |

- Phần dùng chung bị sửa khi học task mới tăng từ **2,4% lên khoảng 27%** model; phần bị khoá giảm từ 75% xuống khoảng 51%. Prompt không tính vào đây, vì mỗi task có bộ riêng nên không ghi đè lên task cũ.
- M5 thêm một cơ chế mà paper không có: loss giữ output của encoder gần với model thầy (model sau task trước) trên cùng ảnh (`--feature_kd`). Model thầy của paper chỉ dùng để tạo nhãn giả.
- Ở task 2, prototype của SKU task 1 vẫn khớp với model thầy, vì model thầy chính là model task 1. Từ task 3, model thầy đã có đặc trưng bị sửa, nên prototype của SKU task 1 phải được làm lại. Phần này **chưa làm**; code chặn chạy M5 quá task 2.

### 8.2 Production có chấp nhận được không

Production không chấm theo mức giống paper. M5 giữ nguyên kiến trúc, số tham số và tốc độ suy luận. Cái giá là mỗi lần thêm SKU phải phát hành lại cả model (khoảng 520 MB) thay vì chỉ prompt và head, và train chậm hơn (mục 8.4). Rủi ro thật là **quên SKU cũ**, nên cách chấm dưới đây có điều kiện riêng cho nó. Trong báo cáo, vòng 1 giữ làm bản tái hiện đúng paper, M5 là biến thể "PDP + fine-tune một phần".

### 8.3 Cách chấm lần chạy thử

So trên val với task 2 của vòng 1 (cAcc-M2 73,64% trên 5.702 ảnh, sai số chuẩn khoảng 0,6 điểm; AP50 của SKU task 1 là 94,96%). Quy tắc nằm trong `full_syn.pilot_verdict`:

| Kết quả | Quyết định |
|---|---|
| cAcc-M2 tăng ≥ 3 điểm (khoảng 5 lần sai số chuẩn) **và** AP50 SKU task 1 giảm ≤ 1 điểm | `go`: làm lại prototype của SKU task 1, rồi chạy task 3–5 |
| cAcc-M2 tăng < 3 điểm | `stop`: giả thuyết sai, chuyển sang hướng B (800px, giữ khoá như paper) |
| cAcc-M2 tăng nhưng AP50 SKU task 1 giảm > 1 điểm | `retry`: thử lại một lần với `feature_kd` lớn hơn |

Chỉ đổi một biến so với vòng 1: learning rate, số epoch, batch, ảnh 640px giữ nguyên. R1 (giảm learning rate) và R4 (bớt epoch) để sau, không gộp vào lần thử này.

### 8.4 Đo trên máy local

Chạy task 2 trên RTX 3060, batch 2 (batch hiệu dụng 8 như vòng 1), 30 bước tối ưu, từ task 1 của vòng 1. Attention dùng bản PyTorch thay cho kernel biên dịch, nên số tuyệt đối chậm hơn trên 4090; chỉ tỉ lệ M5/gốc là đáng tin.

| | Gốc (`EXP-FULL-SYN`) | M5 | M5 so với gốc |
|---|---|---|---|
| Tham số được train | 32,18M | 64,18M | +32,0M (đúng tầng cuối backbone + encoder) |
| Giây mỗi bước tối ưu (8 ảnh) | 4,90 | 5,12 | +4,5% |
| VRAM đỉnh, batch 2 | 6,10 GiB | 7,05 GiB | +0,95 GiB |

Suy ra cho 4090: task 2 của vòng 1 mất 5,36 giờ, nên task 2 với M5 khoảng **5,6 giờ** (cả 4 task khoảng 22 giờ, như mục 7). VRAM ở batch 8: vòng 1 dùng 7,5 GiB; M5 thêm khoảng 4 × 0,95 GiB, tức khoảng 11–12 GiB, vẫn dưới 24 GiB. Số này là ước lượng; notebook ghi VRAM thật vào `step_metrics.jsonl`.

- Loss giữ đặc trưng (`FKD`) ở mức khoảng 0,008 ngay từ bước đầu: đó là nhiễu do dropout của encoder khi train, không phải do đặc trưng đã trôi. Với trọng số 1, loss này nhỏ so với loss phát hiện, nên giữ đặc trưng còn lỏng. Trọng số 1 là chọn tạm, chưa có căn cứ; nếu quên nhiều thì nhánh `retry` tăng nó lên.
- Ở batch 8, M5 hết bộ nhớ trên 3060 (12 GB). Vòng 1 trên 4090 dùng 7,5 GiB ở batch 8 cho task 2.

### 8.5 Đã làm

- `pdp/main.py`, `pdp/engine.py`: cờ `--tune_after_task1` (mở khoá theo tên, từ task 2, ở `lr_old`) và `--feature_kd` (MSE giữa output encoder của model và của model thầy). Test: `tests/test_pdp_m5_tune_features.py`.
- `configs/exp/native/EXP-FULL-SYN-M5.sh`: `EXP-FULL-SYN` cộng `--tune_after_task1 stages_3,encoder --feature_kd 1`.
- `autocheckout/full_syn.py`: setting `pdp_experiment` và `seed_campaign`; `stage_seed` dùng lại FSA và task 1 của vòng 1 (link, kiểm md5); `stage_tasks(plan, last=2)`; `pilot_verdict`. Test trong `tests/test_full_syn.py`; dry run notebook 06 chạy hết.
- Notebook [06_full_syn_m5_pilot](../../notebooks/training/06_full_syn_m5_pilot.ipynb), settings mẫu `configs/training/full_syn_m5_pilot.json`, cách chạy ở [runbook mục 7.1](TRAINING_NOTEBOOK_RUNBOOK.md#71-chạy-thử-m5-trên-task-2-29102026).
- Gói seed `data/packages/full_syn_r1_seed_20261029.tar` (1,06 GB, 62 file, md5 `2ae021116a7c0124018051766ba23ca4`).

### 8.6 Kết quả (30/10): `stop`

Chạy trên RTX 4090 (vast.ai), đúng cấu hình ở 8.3, campaign `full-syn-m5-pilot`. Notebook [06](../../notebooks/training/06_full_syn_m5_pilot.ipynb) in:

> STOP - cAcc-M2 0.7364 -> 0.7485 (+0.0121), AP50 task-1 classes 0.9496 -> 0.9473 (-0.0023): no gain, the frozen-feature hypothesis fails; next is 800 px (direction B)

| Trên val, sau task 2 | Vòng 1 | M5 | Chênh |
|---|---|---|---|
| cAcc-M2 (5.702 ảnh) | 73,64% | **74,85%** | **+1,21** (ngưỡng `go`: +3) |
| cAcc-M1, ngưỡng theo nhóm | 24,43% | 26,93% | +2,50 |
| cAcc-M1 theo độ khó: dễ / vừa / khó | 42,01 / 21,34 / 9,96% | 46,40 / 22,90 / 11,51% | +4,39 / +1,56 / +1,55 |
| AP50 SKU task 1 (`mAP_P`) | 94,96% | 94,73% | −0,23 |
| AP50 SKU task 2 (`mAP_C`) | 94,81% | 95,78% | +0,97 |
| AP50 tất cả (`mAP_A`) | 94,44% | 94,50% | +0,06 |
| Quên SKU task 1 (`forgetting`) | −1,58 | −1,35 | |
| Nhãn giả: precision / recall | 99,72 / 99,61% | 99,72 / 99,61% | như nhau (cùng model thầy) |

Các cặp nhầm nhiều nhất (nhãn số, như bảng của vòng 1):

| Đúng → đoán | M5 | Vòng 1 |
|---|---|---|
| 41 → 59 | 19 | 32 |
| 84 → 85 | 10 | 9 |
| 101 → 11 | 5 | 4 |
| 98 → 111 | 5 | 2 |
| 116 → 66 | 3 | 3 |
| 73 → 72 | 3 | 5 |
| 117 → 66, 69 → 70, 31 → 87 | 2 mỗi cặp | ngoài top 20 |
| 64 → 63 | 2 | 2 |

Đọc kết quả:

- **Giả thuyết "đặc trưng bị khoá" không giải thích được phần lớn lỗi.** Mở khoá 32M tham số (tầng cuối backbone và encoder) chỉ thêm 1,21 điểm cAcc-M2, khoảng 2 lần sai số chuẩn của val, xa ngưỡng 3 điểm. Chỉ có một lần chạy mỗi bên, nên chưa biết dao động giữa các lần train là bao nhiêu.
- **Không quên thêm:** AP50 SKU task 1 chỉ giảm 0,23 điểm, `forgetting` còn nhỏ hơn vòng 1. Loss `FKD` nằm trong khoảng 0,0083–0,0091 suốt 16 epoch, không tăng dần. Vậy phần lệch paper ở 8.1 không gây hại ở task 2, nhưng cũng không đem lại đủ.
- **Cặp nặng nhất giảm, các cặp khác gần như không đổi:** 41 → 59 từ 32 xuống 19, các cặp còn lại đứng yên hoặc tăng nhẹ. Đặc trưng học tiếp tách được một phần cặp tệ nhất, không sửa được lỗi SKU anh em nói chung.
- Theo quy tắc 8.3, chuyển sang **hướng B** (800px, giữ khoá như paper). Phần làm lại prototype cho M5 từ task 3 (8.1) không cần làm nữa.

Chi phí đo thật trên 4090:

| | Ước ở 8.4 | Đo |
|---|---|---|
| Tham số được train | 64,18M | 64,18M |
| VRAM đỉnh, batch 8 (`peak_allocated_gib`) | 11–12 GiB | 10,4 GiB (reserved 12,7 GiB) |
| Giây mỗi bước tối ưu (trung vị) | | 0,765 |
| Giờ train task 2 (`train_hours`) | 5,6 | 6,27 (vòng 1: 5,36) |

Mỗi epoch có khoảng 2 phút 45 giây chấm val (`eval_epochs=1`), cộng lại khoảng 42 phút cho 16 epoch. Vòng 1 chạy trên một máy 4090 khác, nên chưa tách được bao nhiêu trong 0,9 giờ chênh là do M5, bao nhiêu là do máy.

Kết quả backup ở `drive:AutoCheckout/2026-10-30/full_syn_m5_pilot_20261030.zip` (479 MiB, 96 mục, md5 `96c914b1b583ab5ea406be148c3e3699`): cả campaign trừ `cache/`, gồm `task_2/task_final.pth`, `step_metrics.jsonl`, `train.log`, `stages/`, notebook đầy đủ và console log trong `notebook_executions/`, cùng file settings. Giải nén ở gốc repo. 14 symlink (sang `full-syn-r1` và `data/processed`) mang đường dẫn tuyệt đối của vast; nối lại như [handoff mục 5](HANDOFF_M5_PILOT_2026-10-30.md#5-khi-chạy-xong).

## Phụ lục: chi tiết kỹ thuật

- Dữ liệu: campaign `data/training/full-syn-r1/` (giải nén từ `training.zip`). Script ở `runs/analysis_r1/` (không vào git): `d1_errors.py`, `d1b_cross_task.py`, `d1c_per_object.py`, `d1d_ceilings.py`, `d1e_one_side.py` (mục 3), `d2_scores.py` (3.4), `t1_split_half.py` (2), `hybrid_predict.sh` và `hybrid_eval.py` (3.4), `sheet.py` (ảnh đối chiếu), `coverage.py` (3.7), `h1_b2_control.py` (3.3), `h2_rank.py`, `h3_group_bias.py`, `h4_features.py` + `h4_rerank.py`, `res_predict.sh` + `h5_res_eval.py` (3.6). Dự đoán ở 960px hết bộ nhớ trên 3060 nên không có kết quả.
- Một vật tính là "khoanh đúng" khi box trùng với vật ít nhất 50% (IoU ≥ 0,5). Ngưỡng và cách lọc trùng lấy từ policy vòng 1, trừ mục 2 và mục 3.4 có chọn lại.
- Mục 3.4: chạy lại model gốc trên 3060 cho đúng dự đoán của 4090. Mức thay đổi trọng số từ task 1 đến task 5: bộ chọn prompt (`query_tf`) 130%, prompt dùng chung 33–160%, `input_proj` 14%, `bbox_embed` 5%; backbone, encoder, decoder và phần phân loại của SKU cũ không đổi. `query_tf` thay đổi nhiều vì nó khớp cờ `--new_params prompts` nên được train ở learning rate cao ở mọi task. Cờ `--freeze_shared_after_task1` có khoá `query_tf` nhưng không khoá prompt dùng chung.
- Mục 3.3: "có ảnh hai mặt" nghĩa là SKU có bộ ảnh `-back` trong `instances_train2019.json`. Train2019 có 53.739 ảnh, mỗi ảnh một vật.
- Mục 5: md5 file test của E4 là `8a508281…` (`/data/rpc/tasks/100-4x25_seed0`); tập 6.003 ảnh là `e9b5da5e…`.
