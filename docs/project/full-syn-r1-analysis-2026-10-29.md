# Phân tích kết quả FULL-SYN vòng 1 — 29/10/2026

Người làm: さかな (với Claude).

Tài liệu này kiểm lại các giả thuyết trong [handoff họp 28/10 của An](../references/meeting-2026-10-28/HANDOFF_CAI_THIEN_KET_QUA_R1.md) bằng chính kết quả của vòng 1 ([plan mục 11](full-syn-train-plan-2026-10-22.md#11-kết-quả-vòng-1-2710)). Mọi phép kiểm chạy trên máy local (CPU và RTX 3060), không train lại, không chấm lại test để chọn cấu hình.

**Hướng của nhóm:** giữ phương pháp của paper (PDP), nhưng đưa kết quả lên mức dùng được cho production. Vì vậy tài liệu không xét hướng làm ảnh ghép "thật hơn" (bóng đổ, màu sắc, CycleGAN).

## Đọc nhanh

1. **Cách chấm điểm đúng.** Số đo trên val gần như trùng số đo trên test, nên chọn cấu hình dựa trên val là tin được.
2. **Chỉnh cách đếm không giúp thêm.** Đổi ngưỡng hay đổi cách lọc box trùng chỉ thay đổi kết quả khoảng ±0,3 điểm.
3. **Lỗi lớn nhất là nhận nhầm sản phẩm** (đúng vị trí nhưng sai SKU). Nếu hết lỗi này, tỉ lệ đếm đúng cả giỏ lên từ 43% đến khoảng 67%.
4. **Nguyên nhân chính của nhầm:** với khoảng 40% số SKU, bộ ảnh mẫu của RPC chỉ chụp một mặt sản phẩm. Model chưa bao giờ thấy mặt kia, nên khi sản phẩm nằm ngửa mặt đó trên quầy thì nhận nhầm. Nhóm SKU này nhầm nhiều gấp 2,3 lần nhóm có đủ hai mặt.
5. **Các giả thuyết khác của handoff không đứng vững:** SKU cũ và mới "tranh nhau", điểm tin cậy tụt làm model sai, và E4 đã đạt 42,4% trên cùng tập test.

**Việc nên làm tiếp:**
1. Train cùng phương pháp nhưng trên ảnh quầy thật (FULL-REAL), vì đây là cách duy nhất để model thấy đủ các mặt sản phẩm.
2. Ghép thêm hai chỉnh sửa rẻ: giảm learning rate ở cuối mỗi task, và bớt số epoch thừa.

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

### 3.3 Vì sao nhầm: bộ ảnh mẫu chỉ có một mặt sản phẩm

Ảnh ghép được tạo bằng cách cắt sản phẩm từ ảnh mẫu của RPC (mỗi ảnh một sản phẩm đặt trên bàn xoay) rồi dán lên nền khay. Xem [ảnh đối chiếu](../../runs/analysis_r1/confusions.png):
- Ảnh mẫu của 98_milk và 97_milk đều chỉ chụp **mặt sau** hộp (ống hút, bảng dinh dưỡng). Ở mặt này hai SKU gần như giống hệt nhau.
- Trên quầy thật, hộp thường nằm ngửa **mặt trước** (logo, tên sản phẩm). Model chưa từng thấy mặt này, nên đoán thành 97_milk, đôi khi thành cả kẹo gum.

RPC chụp thêm mặt còn lại cho 122 trong 201 SKU; 79 SKU chỉ có một mặt. So hai nhóm sau task 5:

| SKU | Số vật trên val | Nhầm SKU | Sót |
|---|---|---|---|
| Có ảnh cả hai mặt (122 SKU) | 43.497 | 3,2% | 2,1% |
| Chỉ một mặt (79 SKU) | 27.611 | **7,4%** | 2,9% |

13 trong 20 SKU tệ nhất thuộc nhóm một mặt, dù nhóm này chỉ chiếm 39% số SKU.

**Ý nghĩa:**
- Làm ảnh ghép thật hơn (bóng, màu, CycleGAN) không sửa được lỗi này.
- Cắt thêm ảnh mẫu từ RPC cũng không sửa được, vì RPC không hề có ảnh của mặt còn thiếu.

Có hai cách để model thấy mặt đó:
1. **Train trên ảnh quầy thật**, như B2 đã làm, nhưng thêm các bản sửa của vòng 1 (FSA, I3, F14). Gọi là FULL-REAL. Cách này giữ đúng phương pháp của paper và không cần làm thêm dữ liệu. Lưu ý: ảnh quầy của RPC chỉ có trong val2019 và test2019, nên FULL-REAL phải dùng cách chia dữ liệu của B2 (22.494 ảnh train) và so trên 6.003 ảnh test cũ.
2. Nhóm tự chụp mặt còn thiếu của 79 SKU rồi đưa vào ảnh ghép.

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

## 4. Cấu hình train

- **Learning rate không giảm lần nào** trong cả vòng 1, ở cả FSA lẫn PDP. Code gốc của paper cũng vậy. Deformable DETR thường giảm learning rate 10 lần ở gần cuối, và việc này hay giúp tăng điểm (R1).
- **PDP task 1 chạy 16 epoch là thừa:** AP50 trên val gần như không đổi từ epoch đầu (0,928 → 0,934). Task 3 và 5 dừng tăng ở khoảng epoch 10–12. Bớt epoch tiết kiệm được giờ GPU (R4).

## 5. Một chỗ sai trong handoff

Handoff viết E4 (tháng 9, ảnh thật) đạt 42,4% "trên cùng 6.003 ảnh test". Thực ra E4 được chấm trên tập test của cách chia dữ liệu tháng 9, là một file khác. Vì vậy chỉ nói được E4 cùng tầm với FULL-SYN, không so trực tiếp được.

## 6. Xếp lại các đề xuất của handoff

| Đề xuất | Sau khi kiểm | Ưu tiên |
|---|---|---|
| FULL-REAL: cùng phương pháp, train trên ảnh quầy thật | Cách duy nhất cho model thấy mặt sản phẩm còn thiếu | **Cao nhất** |
| R1 giảm learning rate, R4 bớt epoch | Đã xác nhận learning rate không giảm, task 1 thừa epoch. Chỉ cần đổi cờ, ghép được vào FULL-REAL | **Cao** |
| R3 ảnh 800px | Các cặp hay nhầm khác nhau ở chữ in, nên ảnh rõ hơn có lý do để giúp. Chưa kiểm được ở local | Trung bình |
| R2 bật augment | Chưa kiểm được ở local | Trung bình |
| I5 khoá phần dùng chung | Điểm tụt không làm model sai thêm (mục 3.4) | Thấp |
| C1 cân điểm SKU cũ/mới | Không có hiện tượng "tranh nhau" (mục 3.2) | Thấp |
| S1 cắt thêm ảnh mẫu | RPC không có mặt còn thiếu; chỉ có ích nếu nhóm tự chụp | Thấp |
| T1, D3 chỉnh ngưỡng, lọc box trùng | Không giúp (mục 2) | Bỏ |
| Làm ảnh ghép thật hơn (S2, S3, CycleGAN) | Theo hướng của nhóm, và không sửa được lỗi chính | Bỏ |

## Phụ lục: chi tiết kỹ thuật

- Dữ liệu: campaign `data/training/full-syn-r1/` (giải nén từ `training.zip`). Script ở `runs/analysis_r1/` (không vào git): `d1_errors.py`, `d1b_cross_task.py`, `d1c_per_object.py`, `d1d_ceilings.py`, `d1e_one_side.py` (mục 3), `d2_scores.py` (3.4), `t1_split_half.py` (2), `hybrid_predict.sh` và `hybrid_eval.py` (3.4), `sheet.py` (ảnh đối chiếu).
- Một vật tính là "khoanh đúng" khi box trùng với vật ít nhất 50% (IoU ≥ 0,5). Ngưỡng và cách lọc trùng lấy từ policy vòng 1, trừ mục 2 và mục 3.4 có chọn lại.
- Mục 3.4: chạy lại model gốc trên 3060 cho đúng dự đoán của 4090. Mức thay đổi trọng số từ task 1 đến task 5: bộ chọn prompt (`query_tf`) 130%, prompt dùng chung 33–160%, `input_proj` 14%, `bbox_embed` 5%; backbone, encoder, decoder và phần phân loại của SKU cũ không đổi. `query_tf` thay đổi nhiều vì nó khớp cờ `--new_params prompts` nên được train ở learning rate cao ở mọi task. Cờ `--freeze_shared_after_task1` có khoá `query_tf` nhưng không khoá prompt dùng chung.
- Mục 3.3: "có ảnh hai mặt" nghĩa là SKU có bộ ảnh `-back` trong `instances_train2019.json`. Train2019 có 53.739 ảnh, mỗi ảnh một vật.
- Mục 5: md5 file test của E4 là `8a508281…` (`/data/rpc/tasks/100-4x25_seed0`); tập 6.003 ảnh là `e9b5da5e…`.
