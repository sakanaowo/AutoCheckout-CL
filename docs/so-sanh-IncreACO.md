# So sánh với IncreACO (WACV 2021)

*Lập ngày 06/10/2026. Cập nhật mỗi khi có run mới.*

**Bài so sánh:** Yang và cộng sự, *IncreACO: Incrementally Learned Automatic Check-out with Photorealistic Exemplar Augmentation*, WACV 2021. Bài này cũng làm đếm sản phẩm học tăng dần trên RPC, nên là đối thủ trực tiếp.

**Mục tiêu của nhóm:** phương pháp của nhóm ít nhất phải tốt hơn IncreACO. Tài liệu này ghi lại:

- ngưỡng cần vượt;
- những gì đã so được và những gì chưa so công bằng được;
- khoảng cách hiện tại;
- các hướng cải thiện.

Số liệu của nhóm nằm trong `results/experiments/<run>/metrics_count_test*.json` (tập test khóa, 6.003 ảnh).

## 0. Giải thích các chỉ số

**Ví dụ dùng xuyên suốt mục này.** Một giỏ hàng thật có 3 Coca, 2 Pepsi và 1 gói snack (6 món). Mô hình đếm ra 3 Coca, 1 Pepsi và 2 snack: nó nhận nhầm một lon Pepsi thành snack.

### Chỉ số đếm (theo bộ công cụ chính thức của RPC)

Ký hiệu ↑ nghĩa là càng cao càng tốt, ↓ là càng thấp càng tốt.

| Chỉ số | Ý nghĩa | Trong ví dụ |
|---|---|---|
| **cAcc** ↑ (checkout accuracy) | Tỉ lệ ảnh mà **mọi** SKU đều được đếm đúng, tức hóa đơn đúng hoàn toàn. Đây là chỉ số quan trọng nhất, vì chỉ cần sai 1 món là khách bị tính sai tiền | Ảnh này sai, tính là 0 |
| **ACD** ↓ (average counting distance) | Trung bình mỗi ảnh lệch tổng cộng bao nhiêu món, cộng độ lệch của mọi SKU | \|3−3\| + \|1−2\| + \|2−1\| = **2** |
| **mCCD** ↓ (mean category counting distance) | Với mỗi SKU: tổng độ lệch chia cho tổng số thật, cộng dồn trên mọi ảnh. Sau đó lấy trung bình trên các SKU. Có thể hiểu là "đếm sai bao nhiêu phần trăm" của mỗi SKU | Coca 0/3, Pepsi 1/2, snack 1/1 → trung bình **0,5** |
| **mCIoU** ↑ (mean category IoU) | Với mỗi SKU: tổng min(dự đoán, thật) chia cho tổng max(dự đoán, thật). Sau đó lấy trung bình trên các SKU. Bằng 1 khi đếm khớp hoàn toàn | Coca 3/3, Pepsi 1/2, snack 1/2 → **0,67** |
| **mCCS** → 1 (mean category confidence score, IncreACO đề xuất) | Với mỗi SKU: tổng số dự đoán chia cho tổng số thật. Sau đó lấy trung bình trên các SKU. Bằng 1 là đếm vừa đủ, nhỏ hơn 1 là đếm thiếu, lớn hơn 1 là đếm thừa | Coca 3/3, Pepsi 1/2, snack 2/1 → **1,17** |

**Lưu ý về mCCS.** Thừa và thiếu có thể bù trừ nhau. Trong ví dụ, tổng số món dự đoán vẫn là 6, đúng bằng thực tế, dù hóa đơn sai. Vì vậy mCCS cho biết mô hình **có lệch về một phía không** (hay đếm thiếu hay đếm thừa), còn mCCD cho biết **sai nhiều hay ít**. Luôn phải đọc hai chỉ số này cùng nhau.

### Chỉ số nhận diện

- **mAP50** ↑: đo chất lượng khoanh hộp và gọi đúng tên từng sản phẩm.
  - Một hộp dự đoán được tính là đúng khi đúng SKU và chồng lên hộp thật ít nhất 50% (IoU ≥ 0,5).
  - Với mỗi SKU, AP tổng hợp hai yếu tố: tìm được bao nhiêu vật thật và có bao nhiêu báo nhầm, xét trên mọi mức điểm tin cậy. mAP là trung bình AP của các SKU.
- **mmAP** (hay AP50:95) ↑: giống mAP50 nhưng khắt khe hơn. Lấy trung bình trên các mức chồng lấn từ 50% đến 95%, nên hộp phải khớp rất sát mới được điểm cao.
- **mAP cao chưa chắc cAcc cao.** Mỗi ảnh có khoảng 12 món. Nếu mỗi món đúng với xác suất 98% thì cả ảnh đúng khoảng 0,98¹² ≈ 78%. Nếu chỉ đúng 93% thì cả ảnh đúng khoảng 42%.

### Các khái niệm khác trong tài liệu

- **SKU cũ / SKU mới.** Tại đợt *t*, "mới" là các SKU vừa học trong đợt *t*, "cũ" là mọi SKU học ở các đợt trước. Ký hiệu `_o` (old) và `_n` (new), ví dụ mCCD_o, mCCS_n.
- **Nhóm 1 … nhóm 5.** Nhóm 1 là 100 SKU gốc; nhóm 2–5 là 25 SKU thêm vào ở mỗi đợt 2–5.
- **Ngưỡng đếm.** Mô hình gán cho mỗi vật một điểm tin cậy từ 0 đến 1, và chỉ vật có điểm ≥ ngưỡng mới được đếm.
  - Ngưỡng cao thì sót vật thật; ngưỡng thấp thì đếm cả vật báo nhầm.
  - Ngưỡng được chọn trên tập val sao cho cAcc cao nhất, rồi giữ nguyên khi đo trên test.
- **NMS** (lọc trùng). Khi hai hộp chồng nhau từ 50% trở lên, chỉ giữ hộp có điểm cao hơn, để một món không bị đếm hai lần.
  - Mask R-CNN của IncreACO luôn có bước này.
  - DETR của nhóm vốn không có, nên nhóm báo cáo cả hai bản: có và không có NMS.
- **Độ quên.** Sau khi học thêm các đợt mới, AP của SKU cũ tụt đi bao nhiêu so với lúc vừa học xong chúng.
- **Học một lần (oracle, cận trên)** là train cả 200 SKU cùng lúc, có đủ nhãn. **Học tăng dần** là học từng đợt, mỗi đợt chỉ có nhãn của SKU mới.

## 1. Ngưỡng cần vượt

Lấy từ Bảng 1 và Bảng 3 của IncreACO, đã chuyển sang dạng 0–1:

| Chỉ số | Học một lần 200 SKU | Học tăng dần 183 + 17 |
|---|---|---|
| cAcc ↑ | 0,7715 | **0,743** |
| ACD ↓ | 0,41 | **0,44** |
| mCCD ↓ | 0,03 | **0,04** |
| mCIoU ↑ | 0,9672 | **0,9651** |
| mCCD SKU cũ ↓ / SKU mới ↓ | 0,03 / 0,04 | **0,03 / 0,09** |
| mCCS SKU cũ / SKU mới (→ 1) | 0,993 / 0,990 | **0,992 / 1,004** |
| mAP50 / mmAP (AP50:95) | 0,9837 / 0,8182 | không báo cáo |

Theo mức độ đông (học một lần, cAcc): easy 0,8806; medium 0,7731; hard 0,6614.

## 2. IncreACO làm gì

1. **PEA (ảnh ghép giống thật).** Họ cắt sản phẩm từ 53.739 ảnh sản phẩm đơn (`train2019`) rồi dán thành ảnh quầy, theo 3 tiêu chí:
   - bỏ các tư thế không thể nằm trên quầy (hệ số ổn định ≥ 0,6);
   - đặt sản phẩm sát nhau;
   - giới hạn độ che khuất theo hình dạng từng loại sản phẩm.

   Sau đó họ dùng CycleGAN để ảnh trông như ảnh chụp thật. Họ không dùng ảnh quầy thật nào để train.
2. **Học tăng dần.** Mô hình là Mask R-CNN (bỏ nhánh mask).
   - Chưng cất kiểu LwF: mô hình cũ làm "thầy" cho phần phân loại và phần hồi quy hộp của lớp cũ.
   - Với lớp mới: học trên nhãn thật của toàn bộ C_o + C_n lớp.
   - Dữ liệu đợt mới được sinh từ ảnh ghép, có trộn sản phẩm cũ theo hệ số hồi tưởng α. Họ chọn α = 0,05, nhưng mục 4.1.2 lại ghi 0,5.
3. **Kịch bản.** Một bước duy nhất: 183 SKU, rồi thêm 17 SKU (mỗi nhóm hàng lấy 1 SKU). Đợt mới dùng 13.075 ảnh ghép, 18k vòng lặp.

## 3. Hai thiết lập khác nhau ở đâu

| | IncreACO | Nhóm |
|---|---|---|
| Ảnh train | Chỉ ảnh ghép từ ảnh sản phẩm đơn | Ảnh quầy thật (`val2019` + phần còn lại của `test2019`), 22.494 ảnh |
| Ảnh test | Toàn bộ 24.000 ảnh `test2019` | 6.003 ảnh thuộc `test2019`, chia theo nhóm ảnh, khóa md5 |
| Số bước học tăng dần | 1 bước (+17 SKU, 8,5% tổng số) | 4 bước × 25 SKU (50% tổng số) |
| Dữ liệu SKU cũ khi học đợt mới | **Có nhãn**: ảnh ghép chứa SKU cũ | **Không có nhãn**: SKU cũ trong ảnh phải đoán nhãn giả (PPG) |
| Mô hình | Mask R-CNN, có NMS | Deformable DETR + PDP (prompt) |
| Số lần chạy | 1 | 1 (seed 0) |

**Hệ quả khi so sánh:**

- **Học một lần (E0 so với Oracle của họ):** nhóm có lợi thế dữ liệu, vì train bằng ảnh thật. Thắng ở đây **không** chứng minh phương pháp tốt hơn.
- **Học tăng dần (E4 so với 183 + 17):** nhóm chịu thiệt.
  - Phải qua 4 bước, với lượng SKU mới gấp 6 lần.
  - Không được xem lại SKU cũ có nhãn.
  - Muốn kết luận "tốt hơn" một cách công bằng thì cần thêm một run theo đúng giao thức 183 + 17 (mục 6, hướng C).
- **Tập test:** cả hai đều lấy từ `test2019` với tỉ lệ easy/medium/hard như nhau, nhưng của nhóm chỉ là một phần. Sai số chuẩn của cAcc là ±0,65 điểm.

## 4. Kết quả so sánh

Mọi số của nhóm dưới đây đều có NMS khi đếm (IoU 0,5), vì Mask R-CNN của IncreACO vốn có NMS. Bản không NMS nằm ở mục 5.

### 4.1 Học một lần 200 SKU: E0 so với IncreACO Oracle (Syn + Render)

| Chỉ số | IncreACO | E0 | Chênh lệch |
|---|---|---|---|
| cAcc easy | 0,8806 | **0,902** | +2,1 điểm |
| cAcc medium | 0,7731 | **0,857** | +8,4 |
| cAcc hard | 0,6614 | **0,751** | +9,0 |
| **cAcc trung bình** | 0,7715 | **0,836** | **+6,5** |
| ACD | 0,41 | **0,27** | |
| mCCD | 0,03 | **0,022** | |
| mCIoU | 0,9672 | **0,978** | |
| mAP50 | 0,9837 | **0,992** | |
| mmAP | 0,8182 | **0,855** | |

E0 hơn ở mọi chỉ số, rõ nhất ở ảnh đông (hard +9 điểm). Lý do chính là dữ liệu: ảnh quầy thật so với ảnh ghép. Nếu không có NMS, E0 chỉ đạt cAcc 0,680, thấp hơn IncreACO, vì DETR báo trùng.

### 4.2 Học tăng dần: E4 so với IncreACO 183 + 17

| Chỉ số | IncreACO (sau 1 bước) | E4 sau đợt 2 (1 bước, 100 + 25) | **E4 sau đợt 5 (200 SKU)** |
|---|---|---|---|
| **cAcc** | **0,743** | 0,484 | 0,424 |
| ACD | **0,44** | 0,96 | 1,63 |
| mCCD | **0,04** | 0,129 | 0,134 |
| mCIoU | **0,965** | 0,895 | 0,880 |
| mCCD SKU cũ | **0,03** | 0,059 | 0,121 |
| mCCD SKU mới | **0,09** | 0,409 | 0,226 |
| mCCS SKU cũ | 0,992 | 1,007 | 0,988 |
| mCCS SKU mới | 1,004 | 1,095 | 1,041 |
| Tỉ lệ so với mức học một lần (cAcc) | **96%** | – | 51% |

Với E4, "SKU mới" là 25 SKU của đợt vừa học, "SKU cũ" là mọi SKU học trước đó.

**Kết luận:** ở phần học tăng dần, nhóm **chưa** tốt hơn IncreACO. Ngay cả khi so một bước với một bước (đợt 2), cAcc vẫn kém 26 điểm và sai số đếm SKU mới gấp 4,5 lần.

### 4.3 Khoảng cách tới mục tiêu (E4 đợt 5 so với IncreACO tăng dần)

| Chỉ số | Mục tiêu | E4 hiện tại | Cần cải thiện |
|---|---|---|---|
| cAcc | ≥ 0,743 | 0,424 | +32 điểm |
| ACD | ≤ 0,44 | 1,63 | giảm khoảng 3,7 lần |
| mCCD SKU cũ | ≤ 0,03 | 0,121 | giảm khoảng 4 lần |
| mCCD SKU mới | ≤ 0,09 | 0,226 | giảm khoảng 2,5 lần |
| mCIoU | ≥ 0,965 | 0,880 | +8,5 điểm |

**Những điểm nhóm đã ngang hoặc hơn:**

- **Không quên.** Nhóm SKU đầu (100 SKU) giữ AP50 0,982 sau 4 bước; mCCD của nhóm này chỉ tăng từ 0,047 lên 0,065.
- **Không đếm lệch về một phía:** mCCS xấp xỉ 1, như IncreACO.
- **Học một lần:** hơn IncreACO.

## 5. Phát hiện mới từ mCCS / mCCD (06/10)

Cách đọc mCCS và mCCD xem ở mục 0. Tóm tắt: mCCS gần 1 mà mCCD cao nghĩa là có ảnh thừa, có ảnh thiếu, nhưng tổng số thì vừa.

Bảng dưới là mCCD/mCCS theo nhóm SKU của từng đợt, đo sau đợt 5 (có NMS):

| Run | Nhóm 1 (100 SKU gốc) | Nhóm 2 | Nhóm 3 | Nhóm 4 | Nhóm 5 | cAcc |
|---|---|---|---|---|---|---|
| E0 (cận trên) | 0,022 / 1,00 | 0,027 / 1,01 | 0,022 / 1,00 | 0,020 / 1,00 | 0,023 / 1,00 | 0,836 |
| **E4** | 0,065 / 1,02 | 0,226 / 0,96 | 0,169 / 0,95 | 0,190 / 0,93 | 0,226 / 1,04 | 0,424 |
| E4_noF14 | 0,073 / 1,05 | 0,262 / 1,03 | 0,184 / 0,95 | 0,196 / 0,89 | 0,250 / 0,93 | 0,396 |
| E3 (không có I2, I3) | 0,086 / 1,05 | 1,109 / **2,01** | 0,500 / 0,87 | 0,518 / 0,50 | 0,589 / 0,48 | 0,200 |
| E2 (kiểu MD-DETR) | 0,729 / **0,28** | 0,825 / 0,21 | 0,567 / 0,61 | 0,572 / 1,26 | 3,582 / **4,56** | 0,009 |
| E1 (train tuần tự) | 0,859 / **0,14** | 0,795 / 0,43 | 0,767 / 0,53 | 0,764 / 0,81 | 4,337 / **5,32** | 0,003 |
| E5 (truy xuất DINOv2) | 0,794 / 0,31 | 0,717 / 0,39 | 0,696 / 0,38 | 0,786 / 0,27 | 0,762 / 0,36 | 0,002 |

1. **Chẩn đoán của E4 được làm rõ.** Trước đây nhóm cho rằng mô hình "kém tự tin với SKU mới" nên đếm thiếu. Thực tế mCCS của SKU mới xấp xỉ 1, tức **tổng số không thiếu**.
   - Lỗi là **đếm nhiễu theo từng ảnh**: mCCD của các nhóm 2–5 là 0,17–0,23, gấp 8–10 lần E0. Nhóm 1 chỉ 0,065.
   - Cách giải thích hợp lý nhất: điểm số của SKU mới thấp, nên ngưỡng đếm bị kéo xuống (0,26, so với 0,35 ở đợt 1). Khi đó một số vật thật vẫn bị bỏ sót, đồng thời nhận thêm một số phát hiện sai hoặc nhầm SKU. Hai loại lỗi bù trừ nhau trong tổng số.
   - Đây vẫn là **giả thuyết**, cần phân rã lỗi để kiểm chứng (mục 6, hướng B).
2. **Khoảng cách với IncreACO nằm ở các SKU học tăng dần, không nằm ở 100 SKU gốc.** Nhóm 1 có mCCD 0,065, gần mức 0,03–0,05 của IncreACO và FSA. Các nhóm 2–5 cao hơn 2–2,5 lần mức SKU mới của IncreACO (0,09).
3. **E3 cho bằng chứng định lượng về vai trò của I3.** Nhóm 2 bị đếm gấp đôi (mCCS 2,01), còn nhóm 4–5 chỉ đếm được một nửa (0,50 / 0,48). Nghĩa là SKU mới bị gán nhầm thành SKU của nhóm 2, đúng như phân tích nhãn giả trước đó.
4. **E1 và E2 quên rồi bị SKU mới "nuốt".** SKU cũ chỉ còn đếm được 14–28%. Nhóm vừa học bị đếm gấp 4–5 lần, vì vật cũ bị nhận thành SKU mới.
5. **E4_noF14 (khi không có NMS):** nhóm 1 có mCCS 1,34, tức đếm thừa 34% do nhãn giả trùng. NMS đưa về 1,05. Điều này khớp với phát hiện F14.

## 6. Hướng cải thiện (xếp theo chi phí tăng dần)

| Mã | Việc | Chi phí | Kỳ vọng / cách đo |
|---|---|---|---|
| A | **Ngưỡng đếm theo nhóm task**, chọn trên val. Hiện cả 200 SKU dùng chung một ngưỡng, trong khi điểm số của SKU gốc và SKU mới khác nhau | Không train lại; vài phút trên VM. Cần thêm code trong `autocheckout/counting.py` | Giảm mCCD của nhóm 2–5 và tăng cAcc. Nếu không đổi thì giả thuyết ở mục 5.1 sai |
| B | **Phân rã lỗi đếm** của E4: bỏ sót, nhầm SKU (đúng vị trí, sai lớp) hay phát hiện thừa; nhầm sang SKU nào, có cùng nhóm hàng không | Không train lại; dùng file `pred_test.npz` có sẵn | Biết cần sửa phân loại hay sửa định vị |
| C | **Run theo đúng giao thức IncreACO**: 183 SKU, rồi thêm 17 SKU (1 SKU mỗi nhóm hàng), 1 bước | 1 lần FSA cho 183 SKU + 1 đợt PDP. Cần ước tính giờ GPU trước khi chạy | Hàng so sánh công bằng duy nhất với 74,3%. Cần tạo cấu hình task mới |
| D | **Tăng khả năng học SKU mới**: mở băng một phần mô hình nền FSA, hoặc train đợt mới lâu hơn hay với LR riêng | Mỗi thử nghiệm khoảng 1 run E4 | IncreACO cho thấy chỉ train lớp cuối thì SKU mới học rất kém (Hình 8), giống tình trạng nền đóng băng của nhóm |
| E | **Ảnh ghép từ `train2019`** (giai đoạn 2 của kế hoạch): tạo thêm ảnh có SKU mới; nếu nhóm chấp nhận dùng ảnh catalog thì trộn thêm một ít SKU cũ (α nhỏ) | Lớn: pipeline cắt, dán và đổi phong cách ảnh | IncreACO cho thấy α tác động lớn đến cân bằng cũ/mới. Lưu ý: chỉ dùng ảnh ghép thô thì cAcc chỉ 26,5%, cần dán lên nền quầy thật hoặc dùng CycleGAN. Có dùng ảnh SKU cũ thì phải báo cáo như một thiết lập riêng ("được dùng ảnh catalog") |

Thứ tự đề xuất: A và B trước (rẻ, giúp chọn đúng hướng), rồi C để có so sánh công bằng, sau đó D hoặc E.

## 7. Lưu ý khi trích dẫn IncreACO

- **Hai giá trị α mâu thuẫn:** 0,05 ở mục 3.2.2 và Hình 8, nhưng 0,5 ở mục 4.1.2.
- **Số của DPNet:** IncreACO ghi DPNet đạt cAcc 0,7283 (Syn + Render), nhưng paper DPNet tự báo cáo **0,8051**. Vậy câu "vượt state-of-the-art" của IncreACO dựa trên bản DPNet họ tự chạy lại. Khi trích dẫn nên ghi cả hai số. Nguồn: [arXiv 1904.04978](https://arxiv.org/abs/1904.04978).
- **Thí nghiệm học tăng dần chỉ có 1 bước, 1 lần chạy**; không có đường cong độ quên qua nhiều bước.
- **mCCS ghi dạng %** (ví dụ 100,36%), còn bảng này ghi dạng tỉ lệ (1,004).

## 8. Nguồn số liệu và cách tính lại

- **Số liệu của nhóm:**
  - `results/experiments/<run>/metrics_count_test.json` và `metrics_count_test_nms0.5.json`;
  - khóa `stages.<t>.test_by_task` (từ commit `38040c7`) chứa mCCD/mCCS theo `task_<g>`, `new` và `old`;
  - mAP ở `metrics_cl_test.json`.
- **Tính lại trên VM** (chỉ dùng CPU, khoảng 10–40 giây mỗi run):
  ```bash
  python -m tools.eval_count --run-dir /data/runs/<run> --val-ann $TASKS/val_full.json \
    --test-ann $TASKS/test_full.json --task-config configs/tasks_100-4x25_seed0.json [--nms-iou 0.5]
  ```
- **Số liệu IncreACO:** Bảng 1 (Syn + Ren, Ours), Bảng 3 và Hình 8 của paper (file PDF ở thư mục gốc repo, không commit).
