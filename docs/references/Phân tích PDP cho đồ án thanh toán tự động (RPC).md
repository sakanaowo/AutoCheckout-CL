# Phân tích PDP cho đồ án thanh toán tự động (RPC)

Sep 27, 2026 · @An

PDP áp dụng được cho đề tài, nhưng phải đổi 3 điểm. Test trên 24.000 ảnh checkout thật của RPC, không test trên ảnh ghép. Thêm bước thích nghi miền ở task đầu, và sửa code đang hard-code 80 lớp COCO.

## 1. PDP làm gì

PDP là phương pháp Incremental Object Detection (IOD) không replay: detector học thêm lớp mới qua từng task mà không lưu ảnh cũ. Nó xây trên MD-DETR (ECCV 2024): Deformable DETR pretrained, đóng băng backbone, encoder, decoder, chỉ học prompt và đầu phân loại.

- **Decoupled Dual-Pool Prompting (DDP).** Pool dùng chung (100 prompt, học qua mọi task) giữ tri thức chung. Pool riêng (mỗi lớp mới 1 prompt, đóng băng sau task của nó) giữ đặc trưng phân biệt. Prompt được trộn theo cosine với query của ảnh, rồi chèn làm prefix vào self-attention của 6 lớp decoder, nên suy luận không cần task ID.
- **Prototypical Pseudo-Label Generation (PPG).** Ở task t, vật thuộc lớp cũ trong ảnh không có nhãn. Teacher (mô hình task t−1) đề xuất box: điểm > 0,5 nhận luôn; điểm 0,2–0,5 chỉ nhận nếu cosine với prototype của lớp ≥ 0,5. Prototype là trung bình embedding query của lớp, tính ở epoch cuối của task.
- **Loss L\_DDL** (Eq. 10) đẩy hai pool về hướng trực giao (ngưỡng 90°, λ = 0,15).

Kết quả paper (mAP@0,5): COCO 4 task đạt 59,4 mAP@A ở task cuối, hơn MD-DETR 9,2 điểm. VOC 10+10 / 15+5 / 19+1 đạt 78,7 / 78,0 / 79,4. Trong ablation, PPG đóng góp nhiều nhất (+13,9 mAP@P ở task 4), còn L\_DDL chỉ +0,4 mAP@A.

Ba giả định ngầm của paper ảnh hưởng trực tiếp tới đề tài:

- Detector gốc là Deformable DETR đã train trên đủ 80 lớp COCO rồi bị đóng băng. Trên benchmark COCO, phần đóng băng đã "thấy" mọi lớp tương lai; trên RPC không có lợi thế này.
- PPG chỉ có việc để làm khi ảnh của task mới chứa vật cũ không nhãn, như giao thức COCO/VOC.
- Paper chỉ thử 20–80 lớp vật thể thông thường, chưa thử lớp fine-grained như 200 SKU của RPC.

## 2. Source code PDP\_IOD

Repo [zyt95579/PDP\_IOD](https://github.com/zyt95579/PDP_IOD) là bản sửa nhỏ trên code [MD-DETR](https://github.com/GauravBh1010tt/MD-DETR): thêm hai pool prompt (`models/prompt.py`) và PPG (`engine.py`). Nền là Deformable DETR của HuggingFace (transformers 4.37.2, chép vào `models/`) cộng PyTorch Lightning 2.1.3. Dữ liệu đọc theo định dạng COCO JSON cho từng task.

Phần dùng lại được gần như nguyên: module prompt, logic PPG và prototype (có lưu trong checkpoint), vòng train Lightning, bộ đánh giá mAP@P / mAP@C / mAP@A. RPC vốn có annotation dạng COCO, nên chỉ cần viết hàm chia task và sinh JSON theo task.

Mình đã đọc toàn bộ code và dựng thử mô hình (CPU, trọng số ngẫu nhiên) để kiểm tra. Các điểm cần xử lý trước khi chạy RPC:

| # | Vấn đề | Vị trí | Ảnh hưởng tới đề tài | Cách xử lý |
| --- | --- | --- | --- | --- |
| 1 | Hard-code 80 lớp COCO trong pool riêng | `models/prompt.py` (`total_classes = 80`, `private_size = 80`) | Thử với 200 lớp: task đầu > 80 lớp thì crash; chia 4×50 thì task 3–4 nhận 0 prompt riêng mà không báo lỗi | Đặt `private_size = sum(task_num_classes)`, `pool_sizes = task_num_classes` |
| 2 | Chia task hard-code COCO 40+20+20 | `datasets/create_coco_instance.py`; `main.py` luôn gọi `task_info_coco` | Không chạy được RPC | Viết `task_info_rpc` và script sinh `train_task_t.json`, `test_task_*.json` |
| 3 | `run.sh` bắt đầu từ task 2, nạp checkpoint Task 1 có sẵn trên máy tác giả | `run.sh` (`start_task=2`) | README không kèm checkpoint, phải tự train task 1 | Chạy với `start_task=1` |
| 4 | Loss L\_DDL không chạy | `use_ddl_loss = False`, `ortho_mu = 0`; decoder bỏ qua loss trả về (`p_list, _, output = prompts.forward(...)`) | Code khác paper; theo ablation chỉ lệch khoảng 0,4 mAP@A | Bật và cộng loss vào tổng, ghi rõ trong báo cáo |
| 5 | Teacher chạy không có prompt | `engine.py` dòng 304: gọi `old_model(...)` không truyền `query` | Teacher yếu hơn Φ\_{t−1} trong paper; prototype tính từ đặc trưng có prompt nhưng đem so với đặc trưng không prompt | Cho teacher chạy 2 lượt: lấy query rồi forward có prompt |
| 6 | Lệch 1 chỉ số khi lọc nhãn giả | `engine.py` dòng 210: `labels_tensor <= PREV_INTRODUCED_CLS` | Lớp mới đầu tiên có thể nhận nhãn giả từ teacher (ảnh hưởng nhỏ) | Đổi thành `<` |
| 7 | Đóng băng backbone, encoder, decoder ngay từ task 1 | `run.sh` (`freeze='backbone,encoder,decoder'`) | Đặc trưng COCO có thể không phân biệt nổi 200 SKU rất giống nhau | Thêm bước thích nghi miền ở task 1 (mục 5) |
| 8 | Bộ phân loại cấp phát cố định `n_classes`, logit lớp tương lai bị che | `main.py`, `engine.py` | Phải biết trước số lớp tối đa | Cấp dư, ví dụ 256 slot cho 200 SKU cộng sản phẩm demo |
| 9 | Đường dẫn `/data/zyt/...`, tên `checkpoint07.pth`, 8 epoch/task đều hard-code | `run.sh`; `engine.py` dòng 271–278; `main.py` | Đổi số epoch là teacher nạp sai file | Đưa hết thành tham số |
| 10 | `requirement.txt` thiếu torch, torchvision, scikit-learn; đường dẫn CUDA kernel trỏ tới thư mục không có trong repo | `requirement.txt`, `models/load_custom.py` | Cài lỗi; deformable attention rơi về bản PyTorch chậm hơn | Cài torch khớp CUDA; lấy kernel từ Deformable-DETR gốc |

Mô hình có 66,4 triệu tham số; với cấu hình freeze của repo, 32,3 triệu vẫn được train ở mỗi task. Riêng `query_tf` (Linear 76.800→300) chiếm 23 triệu, pool chung 1,84 triệu, mỗi lớp mới thêm 18.432 tham số prompt (\~74 KB). Suy luận cần 2 lượt forward, train cần 3 lượt (thêm teacher).

## 3. PDP có hợp với đề tài không

PDP khớp phần lõi của đề tài: thêm lớp, không lưu ảnh cũ, nhiều vật trên một ảnh. Hai điểm paper chưa kiểm chứng là SKU fine-grained và dữ liệu tổng hợp.

| Yêu cầu của đề tài | PDP | Ghi chú |
| --- | --- | --- |
| Thêm sản phẩm mới, không train lại từ đầu | Đáp ứng | Backbone, encoder, decoder đóng băng; mỗi task vẫn train khoảng 32 triệu tham số (prompt, `query_tf`, `input_proj`, đầu phân loại/box) nhưng chỉ trên dữ liệu mới |
| Không lưu ảnh sản phẩm cũ | Đáp ứng | Mỗi lớp lưu \~74 KB prompt và tối đa 100 vector prototype (\~100 KB); RPC có trung bình \~270 ảnh gốc mỗi SKU |
| Nhiều sản phẩm trên quầy | Đáp ứng | 300 query; ảnh RPC có trung bình 12,3 vật, tối đa khoảng 20 |
| Không biết sản phẩm thuộc đợt nhập nào | Đáp ứng | Prompt trộn theo cosine, không cần task ID |
| Phân biệt 200 SKU rất giống nhau (cùng hãng, khác vị) | Chưa kiểm chứng, rủi ro cao | Decoder COCO bị đóng băng; paper chỉ thử 20–80 lớp thông thường |
| Train bằng ảnh cắt–ghép | Không được thiết kế cho | Ảnh ghép chỉ có sản phẩm mới thì PPG gần như không có việc |
| Tính tiền đúng (đếm đúng từng món) | Cần làm thêm | Hậu xử lý 1 nhãn mỗi query + ngưỡng; đo checkout accuracy (cAcc), không chỉ mAP |

## 4. Cách xử lý data cắt–ghép

Cắt nền rồi ghép để **train** là hợp lý và có tiền lệ: đó chính là cách làm của baseline RPC, IncreACO và DPNet. Dùng ảnh ghép để **test** thì không nên. RPC đã có 24.000 ảnh checkout thật có bounding box, sát thực tế hơn mọi ảnh ghép.

Bảng xếp hạng RPC cho thấy rõ khoảng cách miền. Tất cả đều train từ ảnh sản phẩm đơn và test trên ảnh checkout thật:

| Phương pháp | Dữ liệu train | cAcc | mAP50 |
| --- | --- | --- | --- |
| Baseline RPC: Syn | Chỉ ảnh cắt–ghép | 9,27% | 80,66% |
| Baseline RPC: Render | Ảnh ghép qua CycleGAN | 45,60% | 95,50% |
| Baseline RPC: Syn+Render | Cả hai | 56,68% | 96,57% |
| DPNet | Syn+Render, có lọc ảnh | 80,51% | 97,91% |
| CommNet v2 | Ảnh sản phẩm đơn, protocol chuẩn | 93,11% | 98,92% |

cAcc là tỉ lệ ảnh mà số lượng từng SKU đếm đúng hoàn toàn, tức hóa đơn đúng. mAP50 80% vẫn có thể đi kèm hóa đơn đúng chưa tới 10%. IncreACO ghi nhận cùng hiện tượng: chỉ ảnh ghép đạt 26,5% cAcc, thêm bước render lên 75,3%.

**Làm ảnh ghép sát thực tế hơn:**

1. Nguồn cắt: chỉ từ 53.739 ảnh sản phẩm đơn (split train), không bao giờ từ ảnh test. Ưu tiên góc trên xuống và 30°/45°, loại tư thế sản phẩm không thể nằm yên trên quầy.
2. Tách nền: BiRefNet/rembg, hoặc SAM với bbox làm prompt. Kiểm tra tay vài trăm mẫu; viền sót nền là lỗi hay gặp nhất.
3. Kích thước: giữ tỉ lệ thật của từng SKU. Lấy kích thước bbox trung vị của mỗi SKU trên 6.000 ảnh val để hiệu chỉnh, chỉ jitter khoảng ±10%.
4. Bố cục: xoay 0–360°, 3–20 vật mỗi ảnh theo phân bố easy/medium/hard, che khuất tối đa 30–50%. Tính lại bbox sau khi che và đối chiếu quy ước bbox với annotation RPC.
5. Ánh sáng: nền là ảnh mặt quầy trắng thật; thêm bóng đổ mờ, cân màu theo ảnh checkout, blur, noise, nén JPEG.
6. Nếu còn thời gian: render bằng CycleGAN hoặc mô hình image harmonization. Đây là bước tạo khác biệt lớn nhất trong các paper trên.

**Ảnh train của mỗi task chứa gì** quyết định PPG có tác dụng hay không:

| Phương án | Ảnh train của task t | Ưu | Nhược |
| --- | --- | --- | --- |
| A. Chỉ sản phẩm mới | Ảnh ghép từ các SKU mới | Đúng tinh thần không lưu dữ liệu cũ, dễ làm | PPG không có việc; lớp mới dễ nhận nhầm sản phẩm cũ giống nó vì chưa từng thấy chúng làm mẫu âm |
| B. Kiểu IOD (đề xuất) | Ảnh ghép SKU mới + ảnh checkout thật từ split val, chỉ giữ nhãn SKU mới | Đúng giao thức của paper nên PPG phát huy; giảm khoảng cách miền | Dùng ảnh thật khi train; cần giải thích kịch bản "nhân viên chỉ gán nhãn hàng mới" |
| C. Replay nhẹ | Ghép thêm vài crop đã lưu của SKU cũ, có nhãn | Thường chống quên tốt nhất | Không còn replay-free; chỉ nên là thí nghiệm đối chứng |

Nên lấy B làm giao thức chính, A làm ablation "chỉ dữ liệu tổng hợp", C làm cận trên có lưu dữ liệu.

## 5. Khó khăn và vấn đề cần giải quyết

Bốn rủi ro lớn nhất: đặc trưng COCO đóng băng, khoảng cách miền của ảnh ghép, cách đánh giá giữa các task, và demo không có sản phẩm RPC thật. Cả bốn đều có cách xử lý trong phạm vi đồ án.

| Nhóm | Khó khăn | Mức | Hướng giải quyết |
| --- | --- | --- | --- |
| Mô hình | Đặc trưng COCO đóng băng khó phân biệt các SKU cùng hãng, khác vị | Cao | Thích nghi miền ở task 1: fine-tune toàn bộ Deformable DETR trên dữ liệu task 1 rồi mới đóng băng, giống First Session Adaptation (ICCV 2023). Cách gọn: lưu thành checkpoint HF riêng rồi trỏ `repo_name` vào đó |
| Mô hình | Lớp mới nhận nhầm sản phẩm cũ; sigmoid tính độc lập từng lớp nên một query có thể ra 2 nhãn | Cao | Mỗi query chỉ lấy nhãn điểm cao nhất, ngưỡng chỉnh trên val; phương án B ở mục 4 cho lớp mới thấy sản phẩm cũ làm mẫu âm |
| Mô hình | Khoảng 32 triệu tham số dùng chung vẫn cập nhật mỗi task (pool chung, `query_tf`, `input_proj`, `bbox_embed`), có thể gây quên | Trung bình | Đo mAP@P sau từng task; thử đóng băng `query_tf` sau task 1 |
| Mô hình | Phải cố định trước số lớp tối đa | Thấp | Cấp dư slot cho sản phẩm tương lai |
| Dữ liệu | Khoảng cách miền từ ảnh ghép sang ảnh thật | Cao | Checklist ở mục 4; luôn báo cáo kết quả trên ảnh thật |
| Đánh giá | Ảnh test thật chứa cả sản phẩm chưa học; code hiện coi chúng là nền nên mọi dự đoán lên chúng bị tính là sai. Lọc ảnh chỉ chứa lớp đã học không khả thi: task đầu 50/200 lớp thì ảnh 4 SKU chỉ \~0,4% thỏa | Cao | Bỏ các dự đoán trùng (IoU ≥ 0,5) với sản phẩm chưa học trước khi tính mAP; đo cAcc đầy đủ sau task cuối |
| Dữ liệu | Tách nền lỗi viền, mất chi tiết nhãn mác | Trung bình | SAM với bbox làm prompt; kiểm tra tay theo mẫu |
| Đánh giá | mAP không phản ánh hóa đơn đúng | Trung bình | Dùng rpctool: cAcc, ACD, mCCD, mCIoU; thêm độ quên, thời gian cập nhật, dung lượng lưu |
| Tính toán | Google Cloud free trial không cho gắn GPU; phải nâng lên tài khoản trả phí (vẫn dùng credit $300) | Cao | Nâng cấp và xin quota GPU ngay tuần 1. Kaggle có sẵn bộ RPC 15 GB và GPU miễn phí theo tuần, hợp cho thí nghiệm nhỏ |
| Tính toán | Mỗi bước train 3 lượt forward; thiếu CUDA kernel thì deformable attention chạy bản PyTorch chậm hơn | Trung bình | Đo thời gian 1 epoch trên tập con trước khi lên lịch; build kernel |
| Demo | RPC là hàng Trung Quốc, nhóm không có sản phẩm thật để demo webcam | Cao | Biến demo thành 1 task incremental: tự chụp 10–15 sản phẩm Việt Nam, tách nền, ghép lên ảnh mặt quầy của nhóm. Đây cũng là minh họa rõ nhất cho "cửa hàng nhập hàng mới" |
| Demo | 2 lượt forward mỗi ảnh; trên CPU máy ảo của mình, ảnh 512×512 mất 2–3 giây mỗi lượt | Trung bình | Chụp khi bấm "Thanh toán" thay vì stream liên tục; đo lại trên máy demo, dùng GPU nếu có |
| Dự phòng | PDP có thể không đạt mức dùng được trên 200 SKU | Cao | Chuẩn bị baseline: detector không phân lớp + embedding (ví dụ DINOv2) + prototype mỗi SKU. Thêm sản phẩm chỉ cần tính prototype mới, không train |

## 6. Thiết kế thí nghiệm đề xuất

Giao thức chính: 5 task trên RPC, task đầu lớn, test trên ảnh checkout thật. Đây là kịch bản cửa hàng có sẵn catalog rồi nhập thêm hàng theo đợt.

| Hạng mục | Đề xuất |
| --- | --- |
| Chia lớp | 100 SKU ở task 1, rồi 4 task × 25 SKU; thứ tự lớp ngẫu nhiên, cố định seed. Tùy chọn thêm 183+17 như IncreACO để so với tài liệu |
| Dữ liệu train mỗi task | Phương án B (mục 4): ảnh ghép SKU mới + ảnh val thật chỉ giữ nhãn SKU mới. Giữ riêng khoảng 1.000 ảnh val để chỉnh ngưỡng |
| Test | 24.000 ảnh test thật. Sau mỗi task tính mAP@P/C/A, bỏ qua vùng SKU chưa học; sau task cuối tính đủ cAcc, ACD, mCCD, mCIoU |
| Task demo | Thêm 1 task 10–15 sản phẩm Việt Nam nhóm tự chụp, test bằng ảnh chụp thật tại quầy demo |
| Baseline | Joint training cả 200 lớp (cận trên); fine-tune tuần tự không chống quên (cận dưới); MD-DETR (repo sẵn); PDP; detector + prototype retrieval |
| Ablation | Có/không bước thích nghi task 1; chỉ ảnh ghép (A) so với B; PPG bật/tắt; pool chung bật/tắt |
| Chỉ số hệ thống | Thời gian cập nhật mỗi task, dung lượng lưu thêm mỗi task, độ trễ suy luận mỗi ảnh |

Những con số này là điểm khởi đầu; nếu thời gian train quá lâu, giảm trước ở số ablation chứ không cắt baseline.

## 7. Kế hoạch 8 tuần

&#91;embedded content: Kế hoạch 8 tuần · 7 hạng mục, 1 mốc quyết định\]

Tuần 1–2 chạy song song: một người dựng môi trường và sửa code PDP, hai người làm pipeline dữ liệu và đánh giá. Ở mốc quyết định, so PDP (đã thích nghi task 1) với baseline retrieval trên tập con 2 task bằng mAP@A và cAcc. Nếu PDP kém rõ rệt, đổi retrieval thành phương pháp chính và giữ PDP làm đối chứng.

## Nguồn

Zhang et al., *Beyond Prompt Degradation: Prototype-guided Dual-pool Prompting for Incremental Object Detection*, CVPR 2026 (PDF trong project)

- [ ] [PDP\_IOD — code chính thức](https://github.com/zyt95579/PDP_IOD) (commit 31/5/2026)
- [ ] [MD-DETR — code nền](https://github.com/GauravBh1010tt/MD-DETR)
- [ ] [RPC: A Large-Scale Retail Product Checkout Dataset](https://ar5iv.labs.arxiv.org/html/1901.07249)
- [ ] [Trang RPC dataset](https://rpc-dataset.github.io/)
- [ ] [RPC Leaderboard](https://github.com/RPC-Dataset/RPC-Leaderboard)
- [ ] [rpctool — công cụ tính cAcc, ACD, mCCD, mCIoU](https://github.com/DIYer22/retail_product_checkout_tools)
- [ ] [IncreACO, WACV 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Yang_IncreACO_Incrementally_Learned_Automatic_Check-Out_With_Photorealistic_Exemplar_Augmentation_WACV_2021_paper.pdf)
- [ ] [First Session Adaptation](https://arxiv.org/abs/2303.13199)
- [ ] [Google Cloud free trial — giới hạn GPU](https://docs.cloud.google.com/free/docs/free-cloud-features)
