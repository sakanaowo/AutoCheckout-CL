# Quyết định ảnh ghép v2 và FULL-SYN — 21/10/2026

Nhánh `feat/synthetic-data-v2`. Tài liệu này là **nguồn quyết định hiện hành** cho dữ liệu ảnh ghép v2 và cổng dừng sau task 1. Nó thay thế các điểm tương ứng trong [handoff](../references/meeting-2026-10-20/HANDOFF_ANH_GHEP_VA_TRAIN.md) và [kế hoạch họp 20/10](../references/meeting-2026-10-20/KE_HOACH_ANH_GHEP_VA_BAN_FULL.md). [Thiết kế](synthetic-data-v2-design-2026-10-20.md) và [plan](synthetic-data-v2-plan-2026-10-20.md) đã được sửa theo tài liệu này.

Nguồn: bản phân tích của người dùng (`AN/QUYET_DINH_21-10.md`, bản đầu) và các lượt kiểm tra trực tiếp trên dữ liệu/code ngày 21/10 (mục 6). **Trạng thái:** quyết định đã chốt để sửa doc; chưa có code generator, chưa sinh dữ liệu, chưa train.

## 1. Tóm tắt

- Train **100% ảnh ghép**, sinh từ cutout của `train2019`. Val = `val2019` (bỏ 208 ảnh làm nguồn nền), test = toàn bộ `test2019`.
- Chỉ chạy **FULL-SYN** ở giai đoạn này. FULL-REAL hoãn.
- **71.642 ảnh ghép**, đúng số ảnh mỗi task của B2; 16 epoch/task nên số bước train bằng B2.
- Chọn góc cutout **theo hình dạng vật**: gói/hộp dẹt dùng góc trên (camera 2), chai/lon dùng góc ngang (camera 0), vì trên khay chúng nằm ngang. Cắt bổ sung góc ngang từ `train2019` cho nhóm dạng trụ.
- Cổng dừng sau task 1 đo bằng **cAcc-M2** trên val2019, so với B2 tính cùng cách: Δ ≥ −13 thì chạy tiếp, Δ < −28 sau 3 vòng thì dừng.

## 2. Phạm vi và giao thức

| # | Quyết định | Ghi chú |
|---|---|---|
| D1 | Train 100% ảnh ghép, cutout cắt từ `train2019` (ảnh sản phẩm đơn) | Đã xác minh 7.502/7.502 cutout có nguồn trong `instances_train2019.json` |
| D2 | Val = `val2019`, test = `test2019` của RPC gốc, dùng nguyên ảnh | Annotation vẫn phải chuyển: RPC ID → nhãn 0..199 theo `task_config.json`, tách theo task, giữ `level`. Không dùng lại val/test của release cũ cho bộ chấm chính |
| D3 | Bỏ **208 ảnh val2019** đã dùng làm nguồn nền; val còn **5.792 ảnh** (easy 1.933 / medium 1.921 / hard 1.938) | Lý do: nền được ước lượng từ chính các ảnh này. `03_make_backgrounds.py` lấy trung bình lưới 24×24, inpaint ô bị che, phóng to và cộng nhiễu, nên nền **không** giống từng pixel ảnh gốc; chỉ giữ trường sáng. Bỏ để không phải giải thích. Không nền nào lấy từ test2019 |
| D4 | Dùng val theo **cách 2**: được chỉnh ảnh ghép dựa trên val tối đa **3 vòng**; test2019 chỉ chấm **một lần** ở cấu hình cuối | Mỗi vòng ghi: thay đổi gì, config hash, kết quả val |
| D5 | Thống kê thiết kế (số vật/ảnh, kích thước, chồng lấn, màu/sáng) lấy từ **val2019 đã lọc**, không lấy từ test2019 | Báo cáo ghi: "train bằng ảnh ghép; phân bố bố cục ước lượng từ val2019". Val vì vậy được dùng cho thống kê, chỉnh ảnh, chọn ngưỡng và cổng dừng; phải nêu đủ |
| D6 | **Chỉ chạy FULL-SYN.** FULL-REAL hoãn | FULL-REAL vẫn khả thi về sau: train trên 22.494 ảnh thật của split cũ, chấm trên 6.003 ảnh test cũ. Không có nó, so FULL-SYN với B2 chỉ kết luận ở **mức toàn hệ thống**: hai bên khác nhau cùng lúc về dữ liệu, FSA, F14, I3, ngưỡng theo nhóm và mức nhãn task 1 |
| D7 | Task 1 có nhãn đầy đủ. Task ≥ 2: SKU mới có nhãn, SKU cũ không nhãn. Không có SKU tương lai. Không lật ảnh | Giữ nguyên |
| D8 | FULL-SYN = ConvNeXt-V2-Base + FSA + F14 (`--pseudo_dedup_iou 0.5`) + I3 (`--pseudo_gt_iou 0.5`) + NMS không phân biệt lớp 0,45 + ngưỡng đếm theo nhóm task chọn trên val | Ngưỡng theo nhóm cần viết code (`select_threshold` hiện chỉ chọn một ngưỡng chung) |

**Bảng chấm điểm:**

| Mục đích | Tập chấm | Model |
|---|---|---|
| Kết quả chính, đối chiếu IncreACO | Toàn bộ test2019 (24.000 ảnh) | FULL-SYN |
| So với B1/B2 | 6.003 ảnh test của split cũ (B1/B2 chưa thấy) | FULL-SYN, B1, B2 |

B1/B2 đã train trên 6.000 ảnh val2019 và 16.494 ảnh test2019, nên **không chấm B1/B2 trên val2019 hay toàn bộ test2019**. Khi so với IncreACO, chỉ ghi "cùng tập test": protocol khác (IncreACO dùng 183+17 trong một bước).

## 3. Dữ liệu ảnh ghép

| # | Quyết định | Lý do |
|---|---|---|
| D9 | **71.642 ảnh** = `[21752, 12534, 12129, 11850, 13377]` theo task, đúng số ảnh B2 đã dùng; train **16 epoch/task**, nên số bước bằng B2 | Thứ đắt là số bước GPU, không phải số ảnh. Với cùng số bước, 34.000 ảnh nghĩa là mỗi ảnh bị xem 32 lần, trong khi FULL-SYN tắt augmentation online (I2). Sinh gấp đôi gần như miễn phí và giữ được so sánh sạch với B2. Trên mức này không có lợi: độ đa dạng bị giới hạn bởi số cutout (mỗi cutout đã được dùng khoảng 160–200 lần) |
| D10 | Lưu JPEG ở **1850px gốc** như val/test; giữ owner mask PNG16 cho toàn bộ release | Ảnh ghép đi qua cùng chuỗi thu nhỏ 1850 → 640 như ảnh thật. Dung lượng khoảng 16 GB JPEG (đo 217 KB/ảnh val2019), mask thêm vài GB. Ước lượng cũ 50–100 GB là quá cao |
| D11 | Ánh xạ camera: **0 = ngang, 1 ≈ 45°, 2 = từ trên, 3 ≈ 30°**. Chọn góc theo hình dạng: **gói/hộp dẹt → camera 2; chai/lon/hộp trụ → camera 0 là chính**, camera 2 (thấy nắp, dạng vật đứng) chỉ dùng tỉ lệ nhỏ. **Bỏ camera 1 và 3** | Xác minh bằng gallery cùng SKU qua 4 camera. Trong ảnh val2019, chai và lon nằm ngang trên khay, nhìn từ trên thấy thân và nhãn, tức giống cutout camera 0. Tỉ lệ vật đứng/nằm của từng nhóm chốt ở gallery M0 |
| D12 | **Giữ 7.502 cutout hiện có**, không cắt lại toàn bộ. **Cắt bổ sung camera 0** cho khoảng 51 SKU dạng trụ (drink 15, canned_food 14, alcohol 11, milk 11; danh sách cuối chốt ở M0), mục tiêu khoảng 15–20 cutout/SKU | Cắt thêm camera 2 vô ích: bàn xoay chỉ tạo thêm góc xoay trong mặt phẳng, bộ sinh đã tự xoay; cutout hiện có đã phủ đủ 322/322 mặt (trước/sau). Camera 0 thì khác: mỗi góc quay cho thấy một phần nhãn khác. Hiện 22/51 SKU dạng trụ có dưới 5 cutout camera 0, có SKU 0 cutout; train2019 có khoảng 67 ảnh camera 0 mỗi SKU. Cần token Kaggle (mục 7) |
| D13 | Số vật mỗi ảnh lấy theo val2019: easy 4–10, medium 10–15, hard 15–20. Task ≥ 2: số vật mới trong `[ceil(0,4n), floor(0,6n)]`, kẹp vào `[1, n−1]`; kiểm tra tỉ lệ 40–60% **trên toàn task** | Với n ≥ 4 khoảng này luôn có nghiệm; phép kẹp chỉ để phòng thủ. 40–60% là lựa chọn có chủ đích: ảnh thật task 2–5 chỉ có khoảng 23% vật mới (2,7/ảnh có nhãn) |
| D14 | SKU cũ trong ảnh task ≥ 2 lấy **đều theo từng SKU cũ** | Để đo được chất lượng nhãn giả theo từng SKU |
| D15 | **Không** thêm vật lạ không nhãn vào ảnh task 1 | D7 cấm SKU tương lai, không có nguồn vật lạ khác. Việc chấm điểm xử lý bằng cAcc-M2 (D16) |
| — | SKU 88 (rượu, task 1, 3 cutout camera 2) và SKU 144 (kẹo, task 4, 1 cutout camera 2) | SKU 88 là chai, đã có 5 cutout camera 0 và được bổ sung theo D12. SKU 144 xem ở M0; nếu thiếu thì cắt bổ sung cùng đợt |

## 4. Cổng dừng sau task 1

| # | Quyết định |
|---|---|
| D16 | Chỉ số: **cAcc-M2** trên các ảnh val2019 đã lọc có chứa SKU task 1 (5.562 ảnh). Trước khi đếm, bỏ mọi dự đoán có IoU ≥ 0,5 với vật thuộc SKU chưa học. Ngưỡng đếm chọn trên chính tập này, như B2 đã làm với val của nó. Báo cáo kèm cAcc M1, AP50@A, cAcc theo level và bảng nhầm lẫn làm **chẩn đoán** |
| D17 | Mốc B = cAcc-M2 task 1 của B2, tính lại từ `pred_val.npz`. Δ = FULL-SYN − B. **🟢 Δ ≥ −13** (khoảng ≥ 60%): chạy task 2–5. **🟡 −28 ≤ Δ < −13**: chỉnh một vòng. **🔴 Δ < −28** (khoảng < 45%): chỉnh một vòng |
| D18 | Kiểm tra sớm: chấm model FSA trước khi chạy PDP task 1. Nếu cAcc-M2 của FSA < 40%, bỏ qua PDP task 1 của vòng đó và sang vòng chỉnh tiếp theo |
| D19 | Hết 3 vòng: ở 🟡 thì **vẫn chạy task 2–5** và ghi "kỳ vọng ngang B2"; ở 🔴 thì **dừng** hướng ảnh ghép thuần |
| D20 | Nút được chỉnh ở mỗi vòng **đăng ký trước**, tối đa 2 nút/vòng, chọn theo chẩn đoán: cân màu/sáng theo val2019, bóng đổ mềm, làm mờ mép cutout, mức che khuất, phân bố tỉ lệ vật, tỉ lệ vật đứng/nằm, cắt bổ sung cutout. **Không** đổi số ảnh, không thêm CycleGAN trong gói này |
| D21 | Số epoch FSA chọn theo đường val2019 với trần đặt trước; số cụ thể chốt trong plan gói train. Ghi giờ GPU của FSA riêng |

**Vì sao M2:** 48% số vật trong các ảnh val chứa SKU task 1 là SKU tương lai. B2 học chúng như nền, còn FULL-SYN chưa từng thấy. Theo câu chuyện của đồ án, SKU tương lai là hàng cửa hàng chưa bán, nên ngoài thực tế không xuất hiện trên quầy. M1 sẽ phạt FULL-SYN vì một tình huống không có thật. Ở task 5 không còn SKU chưa học nên M1 bằng M2; kết quả cuối không bị ảnh hưởng. Phương án chỉ chấm 218 ảnh toàn SKU task 1 bị loại: quá ít (sai số khoảng ±3 điểm) và lệch về ảnh dễ.

**Vì sao các con số này:**

| Nguồn | cAcc task 1 | Ghi chú |
|---|---:|---|
| B2 | 73,19 val / 73,16 test | ConvNeXt khóa, ảnh thật, không FSA |
| E4 tháng 9 | 75,98 val / 73,41 test | ResNet + FSA, 6.000 ảnh thật; cùng nhóm ảnh test 6.003 |
| IncreACO | 74,3 | Ảnh ghép + CycleGAN, 200 SKU, học một lần |
| Dán thô (theo IncreACO) | 26,5 | Rủi ro nếu ảnh ghép kém |

Mốc khoảng 73 được ba nguồn độc lập xác nhận, nên không phụ thuộc riêng vào các khuyết điểm của B2 (khuyết điểm của B2 chủ yếu thể hiện ở task 2–5).

Tỉ lệ giữ lại cAcc từ task 1 đến task 5: B2 là 0,30, E4 là 0,49. FULL-SYN có các bản sửa của E4 nên giả định 0,45–0,5. Để kết quả cuối hơn B2 (21,74) khoảng 5 điểm cần task 1 khoảng 53–59; để chỉ ngang B2 cần khoảng 44–48. Từ đó ra hai mốc 60 và 45.

**Giới hạn:**
- So khác tập: val cũ của B2 là một phần test2019, khác val2019 nhưng cùng phân bố.
- Ngưỡng chọn trên chính tập chấm nên số val lạc quan khoảng 0–2,6 điểm.
- Một seed.
- Nếu FULL-SYN chỉ giữ được như B2 (0,30), 60% ở task 1 chỉ cho khoảng 18 ở cuối. Không có điểm kiểm tra sớm nào phân biệt được điều này.
- Cổng tương đối dễ: ConvNeXt có FSA trên ảnh thật nhiều khả năng vượt 73.

## 5. Kết quả review nhánh (R1–R8)

| # | Trạng thái |
|---|---|
| R1 | Đóng bằng D13. Lỗi chỉ xảy ra khi n ≤ 3; val2019 có tối thiểu 4 vật/ảnh |
| R2 | Đóng bằng D11 |
| R3 | Đóng bằng D14 |
| R4 | Đóng bằng D9/D10; ước lượng dung lượng đã sửa |
| R5 | Mở: viết plan chi tiết cho gói train, làm song song với gói dữ liệu |
| R6 | Đóng bằng D2/D3; plan Task 6 đã sửa |
| R7 | Đóng bằng D5; plan Task 1 đã sửa |
| R8 | Mở: bản sao `REPORT_20-10.md` trong `docs/references/meeting-2026-10-20/` chưa tách Cũ/Mới. Bản mới chưa có trong repo; khi có thì thay và cập nhật SHA-256 trong README |

## 6. Bằng chứng đo ngày 21/10

| Kiểm tra | Kết quả |
|---|---|
| Nguồn cutout | 7.502/7.502 `source` nằm trong `instances_train2019.json`; IoU mask/bbox trung vị 0,946, 5% thấp nhất ≥ 0,906; loại tự động 2.098 (2.043 IoU, 55 fill) |
| Cutout theo camera | 0: 876; 1: 2.417; 2: 3.143; 3: 1.066. Camera 2 mỗi SKU: tối thiểu 1, trung vị 18; 30 SKU dưới 10, 18 SKU dưới 8 |
| Độ phủ mặt | train2019: 200 SKU có mặt trước, 122 SKU có thêm mặt sau; cutout hiện có phủ 322/322 |
| Nguồn còn lại trong train2019 | 13.438 ảnh camera 2 (khoảng 67/SKU), 13.434 ảnh camera 0 |
| SKU ít cutout camera 2 | Chủ yếu chai/lon/hộp đứng (nhìn từ trên chỉ thấy nắp); 11–19 lần cắt thử bị loại vì IoU |
| Camera 0 của nhóm dạng trụ | drink 0/4, canned_food 0/5, alcohol 3/5, milk 2/5 (tối thiểu/trung vị); 22/51 SKU dưới 5 |
| Nền | 208/208 khớp tên ảnh val2019, 0 ảnh test2019; mức: easy 67, medium 79, hard 62 |
| val2019 | 6.000 ảnh, 2.000 mỗi level; số vật/ảnh: easy 4–10, medium 10–15, hard 15–20 |
| Split cũ | train 22.494 = 6.000 val2019 + 16.494 test2019; val 1.503 và test 6.003 đều từ test2019. Nhóm ảnh và số đếm của split tháng 9 và tháng 10 giống nhau |
| Dữ liệu thật B2 | Vật có nhãn/SKU: task 1 1.382, task 2–5 1.292–1.537; vật có nhãn/ảnh: 6,4 (task 1), 2,7–2,9 (task 2–5) |
| E4 tháng 9 | 6.000 ảnh/task, 6 epoch; cAcc test task 1→5: 73,4 → 36,1 |
| Thời gian B2 | task 1 khoảng 9,7 giờ; tổng 5 task khoảng 42,5 giờ GPU |

## 7. Việc còn mở

1. **Token Kaggle** để tải `train2019`. Máy này chưa có Kaggle CLI hay token. `scripts/download_rpc.sh` đọc `~/.kaggle/access_token` hoặc `$KAGGLE_API_TOKEN` và tải zip đầy đủ (25,3 GB). API cũng cho tải từng file, nên có thể chỉ lấy khoảng 3.400 ảnh camera 0 cần dùng.
2. Tỉ lệ chai/lon đứng so với nằm trong val2019, theo nhóm hàng (gallery M0).
3. Trần số epoch FSA (D21) và plan chi tiết cho gói train (R5).
4. Người duyệt gallery M0 và pilot M2.
