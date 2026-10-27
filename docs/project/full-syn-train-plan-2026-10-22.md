# Plan gói train FULL-SYN — 22/10/2026

Nhánh `feat/synthetic-data-v2`. Gói này train **FULL-SYN** trên ảnh ghép v2 và chấm trên ảnh RPC thật, theo [quyết định D1–D21](synthetic-data-v2-decisions-2026-10-21.md). Đóng mục R5 và việc mở số 3 (trần epoch FSA, D21) của tài liệu quyết định.

**Trạng thái (27/10):** **vòng 1 chạy xong trên 4090 vast.ai, qua mọi cổng; kết quả ở [mục 11](#11-kết-quả-vòng-1-2710).** Trên 6.003 ảnh test cũ, cAcc sau task 5 là 43,15% so với B2 21,74%; AP50 94,94% so với 84,40%. Mới một seed. F0.7 đạt 26/10. Trước đó: dữ liệu xong và đã rà lại 22/10. Code F0.1–F0.8, workflow và hai notebook đã xong (bảng "Đã làm" ở mục 4.1). Q1–Q3 ở mục 9 đã chốt 22/10 (người dùng chọn theo đề xuất). Q4 (hình thức chạy: notebook mới, gói dữ liệu trên Drive, người dùng tự setup máy vast) chốt 25/10, xem mục 4.1. **F0.2 đạt 25/10:** `test_convnext_fsa_export_keeps_the_finetuned_backbone_in_pdp_task_1` dùng ConvNeXt-V2 atto, truyền `--backbone_pretrained_file` như B2. Sau khi PDP task 1 nạp `hf_model`, digest của backbone + encoder + decoder bằng đúng digest của FSA, nghĩa là trọng số ImageNet không ghi đè lên backbone đã fine-tune. Khóa thiếu khi nạp chỉ có prompt và `class_embed`. Gói dữ liệu: `data/packages/full_syn_rpc_synth_v2_v1_20261025.tar` (25 GB, 101.681 file dữ liệu, md5 `13066bcff65cc42047c969e127c08114`), chờ người dùng upload.

## 0. Đọc nhanh

**Ý tưởng:** dạy model đếm hàng trên quầy chỉ bằng ảnh ghép (dán ảnh sản phẩm đơn lên nền khay), rồi chấm trên ảnh quầy thật của RPC. Model học tăng dần qua 5 đợt: 100 SKU, rồi 4 đợt mỗi đợt 25 SKU mới.

**Dữ liệu chấm lấy nguyên từ RPC.** Val = val2019 bỏ 208 ảnh đã dùng làm nền ảnh ghép (còn 5.792 ảnh); test = toàn bộ test2019 (24.000 ảnh). Đã so từng ảnh và từng box với file gốc của RPC: khớp 100%. B2 thì khác: B2 gộp ảnh thật lại rồi chia lại (train 22.494, val 1.503, test 6.003), nên test của B2 chỉ có 6.003 ảnh.

**Quy trình gồm các trạm kiểm tra,** để không tốn tiền GPU cho một model hỏng từ đầu:

1. **FSA** (khoảng 6–9 giờ): fine-tune toàn bộ model trên ảnh ghép task 1. B2 không có bước này; B2 khoá phần lớn model ngay từ đầu.
2. **Kiểm tra sớm:** model FSA đếm thử trên val thật. Dưới 52,8% thì dừng vòng này, sửa ảnh ghép.
3. **PDP task 1** (khoảng 10 giờ): chạy phương pháp chính (học tăng dần bằng prompt) trên nền model FSA.
4. **Cổng task 1:** ≥ 72,8% thì đi tiếp. Mốc này bằng B2 trừ 13 điểm, chấp nhận kém B2 một chút vì không dùng ảnh thật để train.
5. **Task 2–5** (khoảng 33 giờ).
6. **Chấm test một lần** khi mọi thứ đã chốt, để con số không bị chỉnh theo test.

Trượt ở bước 2 hoặc 4 thì xem model sai ở đâu, sửa tối đa 2 điểm của bộ sinh ảnh, sinh lại ảnh, rồi chạy lại bước 1–4. Làm tối đa 3 vòng; vẫn trượt nặng thì dừng hướng "chỉ ảnh ghép".

**Model học gì ở từng bước:**

| Bước | Dữ liệu | Học gì | Mục đích |
|---|---|---|---|
| FSA | Ảnh ghép task 1, mọi vật (100 SKU) có nhãn | Deformable DETR thường, không prompt, không nhãn giả; **mở khoá toàn bộ** model | Dạy model "nhìn" sản phẩm RPC. B2 bỏ bước này: lõi model giữ nguyên từ COCO/ImageNet, chưa từng thấy sản phẩm RPC |
| PDP task 1 | Cùng dữ liệu task 1 | Khởi tạo từ FSA; khoá backbone/encoder/decoder, chỉ học **prompt** và bộ phân loại 100 SKU | Dựng bộ máy học tăng dần của PDP (prompt pool, prototype mỗi lớp) để task 2–5 dùng |
| Task 2–5 | Ảnh ghép: khoảng 50% vật là SKU mới (có nhãn), 50% là SKU cũ (**không nhãn**); không có SKU tương lai | Như paper: lõi vẫn khoá; thêm prompt và lớp cho 25 SKU mới; phần của SKU cũ chỉ chỉnh nhẹ (lr 1e-5). SKU cũ được **gán nhãn giả** | Học SKU mới mà không quên SKU cũ, không cần lưu ảnh cũ |

Nói gọn: **FSA dạy model nhìn, PDP task 1 dựng hệ thống prompt trên nền đó, task 2–5 thêm prompt cho SKU mới.**

**Nhãn giả (PPG, như paper):** model của task trước làm "thầy", quét ảnh mới và đề xuất các vật SKU cũ. Đề xuất có điểm > 0,5 thì được nhận; điểm từ 0,2 đến 0,5 chỉ nhận khi đặc trưng vật gần prototype của lớp đó. FULL-SYN thêm hai bản sửa mà B2 tắt: **F14** (mỗi vật chỉ một nhãn giả) và **I3** (bỏ nhãn giả "SKU cũ" đè lên vật đã có nhãn SKU mới). Code của nhóm đã sửa một số lỗi so với code gốc PDP (F1–F14, xem `docs/references/meeting-2026-10-20/REPORT_20-10.md`), nhưng ý tưởng phương pháp giữ nguyên.

Lợi thế riêng của ảnh ghép: biết chính xác mọi vật SKU cũ nằm đâu (`train_task_<t>_gt_full.json`), nên đo được nhãn giả của "thầy" đúng bao nhiêu, sót bao nhiêu (F0.6). Với ảnh thật của B2 thì không đo được.

**Chấm điểm thế nào:**

- **cAcc** (chỉ số chính): tỉ lệ ảnh mà model đếm **đúng hết** mọi món, giống hoá đơn phải đúng từng dòng. Kèm theo là **AP50** (khoanh box đúng chỗ, đúng loại), cAcc theo mức easy/medium/hard, và các cặp SKU hay nhầm.
- **M1 / M2:** ảnh val thật có cả SKU model chưa được học (thuộc task sau). M1 bỏ vật chưa học khỏi nhãn nhưng vẫn tính box model đoán lên chúng (dưới tên một SKU đã học), nên phạt model vì đếm thừa thứ chưa ai dạy. M2 bỏ luôn các box đó (IoU ≥ 0,5 với vật chưa học) trước khi đếm; ngoài đời, món cửa hàng chưa bán thì không nằm trên quầy. Cổng task 1 dùng M2. Ở task 5 model đã học đủ 200 SKU nên M1 = M2.
- **Ngưỡng đếm:** mỗi box model đưa ra có một điểm tự tin từ 0 đến 1, và chỉ box có điểm trên ngưỡng mới được đếm. Ngưỡng thấp thì đếm thừa, cao thì đếm sót. Ngưỡng luôn chọn trên val, rồi giữ nguyên khi chấm test.
- **Sau mỗi task:** dự đoán trên val, chọn ngưỡng, ghi cAcc, AP50, mức quên SKU cũ và chất lượng nhãn giả (box mà model task trước tự gán cho SKU cũ trong ảnh mới; đối chiếu được với nhãn đầy đủ của ảnh ghép).
- **Chấm cuối:** model sau task 5 chấm trên 24.000 ảnh test2019; đây là con số chính, cùng tập test với IncreACO. Bảng phụ so FULL-SYN với B1/B2 trên 6.003 ảnh test cũ, vì B1/B2 đã train trên phần còn lại của test2019 và val2019.

**Thuật ngữ:**

| Từ | Nghĩa |
|---|---|
| D1–D21 | Mã các quyết định đã chốt ngày 21/10 trong [tài liệu quyết định](synthetic-data-v2-decisions-2026-10-21.md) (ví dụ D17 = mức cổng task 1) |
| F14, I3 | Hai bản sửa lỗi nhãn giả. F14 bỏ nhãn giả trùng nhau; I3 bỏ nhãn giả "SKU cũ" đè lên vật có nhãn SKU mới. Tháng 9, bật I3 tăng điểm SKU mới từ 0,683 lên 0,925 |
| CycleGAN | Mạng AI biến ảnh ghép trông giống ảnh chụp thật. IncreACO dùng nó để tăng cAcc từ khoảng 26,5% (dán thô) lên 74,3%. Gói này **chưa dùng** (D20): ảnh ghép của mình đã làm kỹ hơn dán thô nhiều, CycleGAN có thể làm méo chữ trên bao bì (nhiều SKU chỉ khác nhau ở chữ), và tốn thêm 1–2 ngày GPU. Để dành làm phương án dự phòng nếu hết 3 vòng vẫn trượt |

## 1. Mục tiêu và tiêu chí xong

1. Chạy đủ 5 task FULL-SYN (protocol 100+4×25, seed 0) nếu qua cổng task 1 (D17–D19).
2. Chấm **một lần** trên toàn bộ test2019 (24.000 ảnh) và trên 6.003 ảnh test cũ để so B1/B2.
3. Mỗi task có log đủ theo [handoff mục 6](../references/meeting-2026-10-20/HANDOFF_ANH_GHEP_VA_TRAIN.md#6-log-bắt-buộc-sau-mỗi-task).

Không làm trong gói này: FULL-REAL (D6), CycleGAN, trộn ảnh thật, LoRA, `--augment`, `--prototype_nearest`, `--freeze_shared_after_task1`, thêm seed.

## 2. Đầu vào đã sẵn sàng

| Gói | Đường dẫn | Ghi chú |
|---|---|---|
| Ảnh ghép train | `data/processed/rpc_synth_v2_v1/` | 71.642 ảnh `[21752, 12534, 12129, 11850, 13377]`, verify 14/14, manifest `ad6d899e…`. [Báo cáo](../data_preprocessing/reports/synth-v2-2026-10-21.md) |
| Nhãn train | `rpc_synth_v2_v1/tasks/synth_v2/train_task_<t>.json` | Nhãn model 0..199; chỉ lớp của task t. `file_name` dạng `images/task_<t>/<id>.jpg`, tương đối với `rpc_synth_v2_v1/` |
| Nhãn đầy đủ (audit) | `rpc_synth_v2_v1/tasks/synth_v2/train_task_<t>_gt_full.json` | Có cả vật SKU cũ, cờ `labeled`. Chỉ để đo nhãn giả, **không** đưa vào train |
| Bộ đánh giá | `data/processed/rpc_eval_official_v1/` | `val_task_<t>.json`, `val_full.json` (5.792), `test_full.json` (24.000), `compare_b12/test_full.json` (6.003). `file_name` dạng `val2019/…`, `test2019/…` |
| Ảnh thật | `data/archive/val2019/`, `data/archive/test2019/` | 1,3 GB + 5,0 GB |
| Mốc B2 | `data/training/vast4090-seed0-640-v1/baseline/EXP-B2/` | cAcc-M2 task 1 = **85,79%** (`runs/data_preprocessing/synthetic_v2/b2_task1_cacc_m2.json`) |

Rà lại 22/10 (script độc lập, không dùng `verify`): hash của mọi file nhãn và 2.000 ảnh mẫu khớp manifest; không trùng id giữa các task; nhãn train của task t đúng và đủ các lớp của task t; mọi bbox nằm trong ảnh; 208 nền không xuất hiện trong val/test; `task_config.json` của hai gói giống nhau. 34 test dữ liệu/đếm đạt.

**Lên máy GPU chỉ cần khoảng 25 GB:** `rpc_synth_v2_v1/images` (17 GB) + `rpc_synth_v2_v1/tasks` (1,7 GB) + `rpc_eval_official_v1` + `data/archive/{val2019,test2019}`. Không cần `labels/`, `owner_masks/`, `annotations/`.

## 3. Cấu hình FULL-SYN

Giữ mọi thứ như B2 (`data/training/vast4090-seed0-640-v1/baseline/EXP-B2/resolved_cli.json`), trừ các dòng in đậm.

| Mục | B2 | FULL-SYN |
|---|---|---|
| Backbone | ConvNeXt-V2-Base `convnextv2_base.fcmae_ft_in22k_in1k` | Giữ |
| Khởi tạo task 1 | Deformable DETR COCO, khóa backbone/encoder/decoder | **FSA**: fine-tune toàn bộ trên ảnh ghép task 1 (`FINETUNE_ARGS`), `--save_hf 1`, rồi PDP task 1 dùng `--repo_name <FSA>/task_1/hf_model` |
| Ảnh train | `--train_img_dir data/archive` | **`--train_img_dir data/processed/rpc_synth_v2_v1`** |
| Ảnh val/test | `--test_img_dir data/archive` | Giữ |
| Thư mục nhãn | `tasks/real` | **Thư mục symlink** (mục 4, F0.1) |
| 640px, batch 8, eff. batch 8, 16 epoch/task, seed 0, TF32, `eval_epochs 1`, lr 1e-4 / lr_old 1e-5, prompt 100×10 | | Giữ (D9) |
| Batch của FSA | — | **4 × tích luỹ 2 = eff. batch 8** (`fsa_batch_size`). Đo 26/10 trên 4090: FSA mở khoá toàn bộ model nên batch 8 hết 24 GB ngay bước 0; batch 4 đạt đỉnh 15,8 GiB, khoảng 1,0 s/bước. PDP vẫn 8/8 như B2 |
| F14 `--pseudo_dedup_iou` | 0 | **0,5** |
| I3 `--pseudo_gt_iou` | 0 (tắt) | **0,5** |
| Nhãn giả τ_h / τ_l | 0,5 / 0,2 | Giữ |
| Đếm | NMS không phân biệt lớp 0,45, một ngưỡng chung | NMS giữ; **ngưỡng theo nhóm task** chọn trên val (D8) |
| Chỉ số cổng task 1 | — | **cAcc-M2** `--unlearned-iou 0.5` (D16) |

## 4. Giai đoạn 0 — code và kiểm tra trước khi thuê GPU

Làm trên máy local (CPU/3060); chỉ F0.7 cần GPU thuê, chạy ngắn.

| ID | Việc | Đầu ra | Tiêu chí đạt |
|---|---|---|---|
| F0.1 | Script `configs/exp/native/EXP-FULL-SYN.sh` (+ `EXP-FULL-SYN-FSA.sh`) dựa trên `native/common.sh`. Tạo thư mục nhãn symlink: `train_task_<t>.json` → ảnh ghép; `val_task_<t>.json`, `val_full.json`, `test_full.json` → bộ đánh giá. Đặt `--train_img_dir` và `--test_img_dir` khác nhau. Không đổi file của B1/B2 | 2 script + test trong `tests/test_native_exp_configs.py` | Test kiểm: args đúng bảng mục 3; symlink trỏ đúng file và hash khớp manifest; không file `gt_full` nào lọt vào train |
| F0.2 | **Kiểm chứng export FSA với ConvNeXt** (test hiện có `test_hf_export_loads_as_the_start_of_pdp_task_1` chỉ chạy ResNet). Rủi ro cụ thể: script B2 truyền `--backbone_pretrained_file`, nên khi PDP nạp `hf_model` thì `runtime.model_config` bật `use_pretrained_backbone`; phải chắc backbone đã fine-tune **không** bị trọng số ImageNet ghi đè | Test round-trip với backbone ConvNeXt nhỏ (timm) | Digest trọng số backbone + encoder + decoder sau khi PDP nạp = digest của FSA; `loading_report.json` không có khóa thiếu ngoài các phần mới của PDP |
| F0.3 | Cờ `--pred_splits` cho `main.py` (mặc định `val,test`); các vòng cổng task 1 dùng `val` | Sửa `write_task_predictions` + test | Vòng cổng không tạo `pred_test.npz` (không lỡ nhìn test, tiết kiệm khoảng 15–20 phút/task cho 24.000 ảnh); `run_exp.sh` vẫn bổ sung được `pred_test` sau bằng `--predict_only` |
| F0.4 | Chọn epoch FSA theo val (D21). Epoch cuối chưa chắc tốt nhất (model có thể quá quen ảnh ghép), mà code hiện chỉ lưu model cuối. Cách làm: lưu checkpoint nhẹ mỗi epoch, ghi AP50 val_task_1 mỗi epoch, export `hf_model` từ epoch tốt nhất trong trần 8 epoch (Q1) | Sửa nhỏ trong `main.py`/trainer + test | `hf_model` ghi rõ epoch đã chọn và đường val; chọn chỉ dựa trên val |
| F0.5 | **Ngưỡng theo nhóm task** (D8). SKU học sớm bị quên dần nên điểm tự tin thấp hơn SKU mới; một ngưỡng chung sẽ đếm sót nhóm này hoặc đếm thừa nhóm kia. `select_threshold` hiện chỉ chọn một ngưỡng. Chọn mỗi nhóm task một ngưỡng trên val, lưu vector ngưỡng vào policy; test chỉ nạp policy | `autocheckout/counting.py`, `tools/eval_count.py` + test | Policy cũ (một ngưỡng) vẫn chạy; B2 tính lại cho cùng số M1 cũ khi chỉ có một nhóm. Cần trước khi chấm task 2, không cần cho cổng task 1 |
| F0.6 | Log theo handoff mục 6, tối thiểu: hash dữ liệu và config ảnh ghép, số nhãn giả bị I3/F14 loại, **precision/recall nhãn giả so với `train_task_<t>_gt_full.json`**, top-20 cặp nhầm SKU mới→cũ, cAcc theo level, ngưỡng từng nhóm | `task_<t>/task_log.json`, `SUMMARY.md`; xem lại `pdp/ppg_audit.py` | Chạy được trên run smoke; số đếm khớp tay trên 1 batch |
| F0.7 | Smoke + benchmark trên GPU thuê (`--stop_after_steps`): FSA và PDP mỗi loại vài trăm bước | `runs/.../full_syn_benchmark/` | Đo ảnh/giây và VRAM tối đa của FSA (chưa từng đo) và PDP; chạy tiếp từ checkpoint được; ước lượng lại mục 7 |
| F0.8 | Script so B1/B2 trên 6.003 ảnh: lọc `pred_test.npz` của FULL-SYN (test 24.000) theo id của `compare_b12/test_full.json` | Tool nhỏ + test | Chạy với B2 trên tập này ra đúng số test cũ của B2. Chỉ cần ở giai đoạn 4 |

### 4.1 Hình thức triển khai (người dùng chốt 25/10)

Phần việc trên nhánh `feat/synthetic-data-v2`:

1. Viết code F0.1–F0.8.
2. Tạo **notebook training mới** trong `notebooks/training/` để chạy gói FULL-SYN. Notebook chỉ gọi code của gói, logic train nằm trong code. Notebook dùng lại `autocheckout/training_workflow.py`, thêm các phase FULL-SYN vào đó, và dùng lại `tools/execute_training_notebook.py`. Notebook cũ (01–03 của B1/B2) giữ nguyên làm lịch sử.
3. Cập nhật tài liệu: plan này, README của notebooks và runbook.
4. **Đóng gói dữ liệu cần trên GPU** để đẩy lên Drive bằng `scripts/pack_full_syn_data.sh`. Script ghi một file tar `data/packages/<tên>.tar` ở local; người dùng tự upload bằng các lệnh rclone mà script in ra. Lần stream thẳng lên Drive ngày 25/10 chỉ đạt khoảng 2 MB/s nên đã dừng. Gói gồm phần dữ liệu ở mục 2 và pretrained `data/pretrained/hf-cache`, kèm `SHA256SUMS` của từng file để kiểm lại sau khi giải nén, và `tar.md5` để so với MD5 mà Drive ghi nhận.

Người dùng tự push nhánh, tự ssh vào máy 4090 vast.ai và setup dữ liệu (tải gói từ Drive), code (lấy nhánh này từ git) và môi trường (theo [README môi trường](../environment/README.md)), rồi chạy notebook trên máy đó.

**Hai notebook:**

| Notebook | Khi nào chạy | Nội dung |
|---|---|---|
| `04_full_syn_smoke` | Một lần trên mỗi máy mới, khoảng 30–60 phút | F0.7: kiểm tra GPU, môi trường, gói dữ liệu so với `SHA256SUMS`; FSA và PDP mỗi loại vài trăm bước; đo ảnh/giây, VRAM; thử dừng rồi chạy tiếp. Người dùng xem kết quả rồi mới chạy notebook kế tiếp |
| `05_full_syn_r<N>` | Mỗi vòng một **file mới** và một campaign mới (vòng chỉnh sau tạo `_r2`, `_r3`), để giữ lịch sử đối chứng giữa các vòng. Bị ngắt giữa chừng thì chạy tiếp bằng chính file của vòng đó | G1 → G4 chạy liền một mạch, các cell cổng tự dừng (bảng dưới) |

Các cổng chỉ dùng ngưỡng số đã chốt, nên notebook chạy liền một mạch, không phải chờ người duyệt giữa chừng trong lúc GPU thuê vẫn tính tiền:

| Cổng | Điều kiện dừng |
|---|---|
| Sau G1 (D18) | cAcc-M2 của FSA < 52,8% |
| Sau G2 (D17, D19) | 🟡 hoặc 🔴 ở vòng 1–2. Ở vòng 3: 🟡 thì chạy tiếp và ghi "kỳ vọng ngang B2"; 🔴 thì dừng hướng ảnh ghép thuần |
| Sau task 2 (G3) | Precision nhãn giả so với `gt_full` < **0,5** (người dùng chốt 25/10) |
| Trước G4 | Chưa đủ 5 task hoặc chưa có policy chọn trên val; hoặc đã có kết quả test (chỉ chấm một lần) |

**Đã làm (26/10):**

| ID | Code | Bằng chứng |
|---|---|---|
| F0.1 | `configs/exp/native/EXP-FULL-SYN.sh`, `EXP-FULL-SYN-FSA.sh`, `full_syn.sh`; `full_syn.link_tasks` | `test_full_syn_configs_follow_the_plan_table`: khác B2 đúng 4 tham số (ảnh train, `repo_name`, F14, I3). Symlink khớp sha256 của cả hai manifest trên dữ liệu thật; thư mục có file `gt_full` thì bị từ chối |
| F0.2 | không cần sửa code | Xem dòng trạng thái đầu tài liệu |
| F0.3 | `--pred_splits` (`main.py`), `PRED_SPLITS` (`run_exp.sh`) | Vòng cổng không ghi `pred_test.npz`; sau đó predict-only thêm `pred_test` mà không train lại |
| F0.4 | `--select_best_epoch` (`main.py`, `checkpointing.record_epoch`) | `task_final.pth` và `hf_model` lấy từ epoch có AP50 val cao nhất; giữ epoch sớm nhất khi bằng điểm; vẫn đúng sau khi chạy tiếp từ checkpoint; đường val ghi trong `hf_model/epoch_selection.json` |
| F0.5 | `counting.select_group_thresholds`, `eval_count --group-thresholds` | Trên B2, chạy một ngưỡng ra đúng policy cũ ở cả 5 stage. Ngưỡng theo nhóm nâng val cAcc stage 2–5 từ 0,552/0,304/0,230/0,205 lên 0,598/0,403/0,274/0,228 (chọn và chấm trên cùng val nên lạc quan) |
| F0.6 | `ppg_audit` đếm số nhãn bị I3/F14 loại; `full_syn.score_val`, `task_<t>/task_log.json`, `SUMMARY.md` | Audit chạy trước khi train task t, vì nhãn giả của task t do model task t−1 sinh ra. Nhờ vậy cổng precision < 0,5 ở task 2 dừng trước khi tốn giờ GPU |
| F0.7 | Notebook `04_full_syn_smoke` trên 4090 vast.ai (26/10) | FSA batch 4 × tích luỹ 2: 0,776 s/bước, đỉnh 15,8 GiB. PDP batch 8: 0,406 s/bước, đỉnh 6,9 GiB. Dừng rồi chạy tiếp ra cùng trạng thái. Giờ đo thật của vòng 1 ở mục 7 |
| F0.8 | `full_syn.subset_run` | Chấm B2 qua đường lọc này ra đúng số test cũ ở cả 5 stage (stage 5 cAcc 0,2174) |
| Workflow | `autocheckout/full_syn.py`: `prepare`, `stage_smoke/fsa/task1/tasks/final_test` | Dry run trên CPU đi qua mọi stage, chạy lại thì không train lại; đổi settings trong campaign cũ thì bị từ chối. Hai notebook chạy hết qua `tools.execute_training_notebook --dry-run` |

Cách chạy trên máy vast: [runbook mục 7](TRAINING_NOTEBOOK_RUNBOOK.md#7-full-syn-trên-máy-vastai-26102026).

**Log:** notebook là **log chạy**, đi song song với log do code ghi. Cell output ghi cấu hình, hash, tiến độ và kết quả từng trạm kiểm tra; bản notebook có output được commit vào git. Log đầy đủ do code ghi (checkpoint, `train.log`, `step_metrics.jsonl`, predictions, policy, các bản notebook đã chạy) nằm dưới `data/training/<campaign>/`, thư mục này bị gitignore.

## 5. Giai đoạn 1–4 — chạy

```text
G1  FSA trên ảnh ghép task 1  ──►  kiểm tra sớm D18: cAcc-M2 của FSA trên val
         │ < 52,8% → bỏ PDP task 1, sang vòng chỉnh (mục 6)
         ▼
G2  PDP task 1 từ FSA  ──►  cổng D17: cAcc-M2 trên 5.562 ảnh val có SKU task 1
         │ 🟢 ≥ 72,8%                 🟡 57,8–72,8% / 🔴 < 57,8% → vòng chỉnh (tối đa 3)
         ▼
G3  PDP task 2–5 (dùng lại task 1, REUSE_TASK1)  ──►  chọn ngưỡng theo nhóm trên val
         ▼
G4  Chấm test một lần: test2019 24.000 ảnh + bảng so B1/B2 trên 6.003 ảnh ──► báo cáo
```

| Giai đoạn | Lệnh chính | Đầu ra | Kiểm tra |
|---|---|---|---|
| G1 FSA | `run_exp.sh configs/exp/native/EXP-FULL-SYN-FSA.sh` | `FSA/task_1/hf_model`, `pred_val.npz`, đường AP50 val theo epoch | `eval_count --run-dir <FSA> --val-ann rpc_eval_official_v1/val_full.json --unlearned-iou 0.5 --calibrate-only`; ghi giờ GPU của FSA riêng (D21) |
| G2 PDP task 1 | `EXP-FULL-SYN.sh` với `N_TASKS=1`, `--pred_splits val` | `task_1/pred_val.npz`, policy M2 | Δ so với B = 85,79%. Báo cáo kèm M1, AP50, cAcc theo level, bảng nhầm lẫn (D16) |
| G3 task 2–5 | `EXP-FULL-SYN.sh` với `N_TASKS=5`, `REUSE_TASK1=<run G2>` | `task_2..5`, log nhãn giả | Sau mỗi task xem precision/recall nhãn giả và số bị I3/F14 loại; nếu nhãn giả task 2 có precision thấp rõ (nhiều vật SKU mới bị gán nhãn SKU cũ; ngưỡng < 0,5, mục 4.1) thì dừng, báo nhóm trước khi chạy tiếp |
| G4 chấm cuối | `--predict_only 1` cho `pred_test` các task còn thiếu; `eval_count --policy-in <policy val>`; `eval_cl`; F0.8 | `metrics_count_test.md`, `metrics_cl_test.md`, bảng so B1/B2 | Test chỉ đọc policy đã khóa từ val; ghi config hash ảnh ghép, run id |

## 6. Vòng chỉnh (D4, D19, D20)

- Mỗi vòng **đăng ký trước** tối đa 2 nút, chọn theo chẩn đoán của vòng trước: cân màu/sáng, bóng mềm, làm mờ mép cutout, mức che khuất, phân bố tỉ lệ vật, tỉ lệ vật đứng/nằm, cắt bổ sung cutout. Không đổi số ảnh, không CycleGAN.
- Đổi nút nào thì sinh lại ảnh ghép thành release mới (`rpc_synth_v2_v2`, ...), có manifest và verify 14/14; không ghi đè `rpc_synth_v2_v1`.
- Mỗi vòng ghi: nút đã đổi, config hash, cAcc-M2 của FSA và PDP task 1, giờ GPU.
- Hết 3 vòng: 🟡 thì vẫn chạy task 2–5 và ghi "kỳ vọng ngang B2"; 🔴 thì dừng hướng ảnh ghép thuần.
- Ứng viên nút theo các giới hạn đã biết của release: giỏ ảnh ghép thưa hơn thật (mật độ cụm 0,77/0,79/0,87 so với 0,82/0,90/0,98), vật ghép phẳng, ảnh thật có vật lạ không nhãn.

## 7. GPU (4090): ước lượng và đo thật vòng 1

B2 trên 4090: task 1 = 9,7 giờ (21.752 ảnh × 16 epoch, kể cả val mỗi epoch), cả 5 task 42,3 giờ.

| Phần | Ước lượng | Đo vòng 1 (26–27/10) | Cơ sở ước lượng |
|---|---|---|---|
| FSA, trần 8 epoch | khoảng 6–9 giờ | 5,04 giờ train | 174.000 lượt ảnh; fine-tune toàn bộ ước chậm hơn PDP khoá 1,5–2 lần |
| PDP task 1 | khoảng 10–11 giờ | 5,96 giờ | Như B2; val_task_1 mới 5.562 ảnh (B2 nhỏ hơn) nên val mỗi epoch thêm khoảng 3 phút |
| Một vòng cổng (G1 + G2) | khoảng 16–20 giờ, khoảng 7–9 USD | khoảng 11 giờ | |
| Task 2–5 | khoảng 33–36 giờ | 21,67 giờ train (5,36 / 5,23 / 5,18 / 5,90) | Như B2, cộng log nhãn giả |
| Dự đoán test 24.000 ảnh × 5 task | khoảng 1,5 giờ | khoảng 1,5 giờ | |
| **Trọn gói nếu qua cổng ngay vòng 1** | **khoảng 52–58 giờ, khoảng 22–27 USD** | **khoảng 34,5 giờ wall (12:43 26/10 → 23:10 27/10), khoảng 14–16 USD** | 0,4–0,46 USD/giờ |
| Mỗi vòng chỉnh thêm | khoảng 16–20 giờ GPU + khoảng 2 giờ CPU sinh lại ảnh | theo đo: khoảng 11 giờ GPU | |

Ước lượng lấy giờ của B2 làm cơ sở, nhưng B2 đã train/resume qua 4090, L4 và 4080 SUPER ([so sánh backbone](FINAL_BACKBONE_COMPARISON_2026-10-20.md)), nên giờ của B2 không so trực tiếp với vòng này. Số bước khớp với F0.7: PDP task 1 có 43.504 bước × 0,406 s ≈ 4,9 giờ, cộng val mỗi epoch.

## 8. Bẫy riêng của gói này

| Bẫy | Cách tránh |
|---|---|
| Backbone FSA bị trọng số ImageNet ghi đè khi PDP nạp | F0.2 trước khi thuê GPU |
| `train_img_dir` và `test_img_dir` trỏ cùng một gốc như B2 | Ảnh ghép và ảnh thật ở hai gốc khác nhau; F0.1 test |
| `gt_full` lọt vào train (thành replay SKU cũ) | Thư mục symlink chỉ có `train_task_<t>.json`; F0.1 test |
| Nhìn test trong các vòng cổng | `--pred_splits val` (F0.3); test chỉ ở G4 |
| Chọn epoch FSA hay ngưỡng trên test | Chỉ dùng val (D4, D21) |
| So FULL-SYN với B2 trên val2019 hoặc toàn test2019 | B1/B2 đã train trên val2019 và 16.494 ảnh test2019; chỉ so trên 6.003 ảnh (F0.8) |
| Mốc B2 đo trên val cũ (1.464 ảnh test2019), FULL-SYN đo trên val2019 | Ghi rõ trong báo cáo; Δ có sai số vài điểm |
| Notebook có output quá nặng để commit (vòng 1: 69 MB, gần hết là dòng tiến độ tqdm) | Trước khi commit, bỏ các dòng thanh tiến độ khỏi stream output (còn 1,2 MB); log đầy đủ vẫn ở `data/training/<campaign>/` |

## 9. Đã chốt (22/10, 25/10)

| # | Câu hỏi | Quyết định |
|---|---|---|
| Q1 | Trần epoch FSA (D21) | **8 epoch** (chốt), chọn epoch tốt nhất theo AP50 val_task_1. Tháng 9 FSA chạy 6 epoch × 6.000 ảnh (36.000 lượt); 8 × 21.752 = 174.000 lượt, gấp khoảng 5 lần, đủ để đường val bão hòa mà vẫn trong khoảng 6–9 giờ |
| Q2 | Bỏ dự đoán test trong các vòng cổng (F0.3) | **Có** (chốt) |
| Q3 | Máy train | **4090 thuê (vast.ai) như B2** (chốt): cùng loại GPU nên so giờ và tốc độ trực tiếp, script chạy tiếp từ checkpoint đã kiểm trên đó |
| Q4 | Hình thức chạy (25/10) | **Notebook training mới + gói dữ liệu trên Drive; người dùng tự setup máy vast** (người dùng chốt). Notebook là log chạy commit vào git, song song với log của code dưới `data/`. Hai notebook: smoke `04` và `05_full_syn_r<N>`, mỗi vòng một file mới; tự dừng khi precision nhãn giả task 2 < 0,5 (mục 4.1) |

## 10. Bàn giao

Người làm phần model đọc theo thứ tự: tài liệu này → [quyết định D1–D21](synthetic-data-v2-decisions-2026-10-21.md) → [handoff mục 5–7](../references/meeting-2026-10-20/HANDOFF_ANH_GHEP_VA_TRAIN.md#5-gói-việc-2-train-full-syn) → [báo cáo ảnh ghép v2](../data_preprocessing/reports/synth-v2-2026-10-21.md). Làm F0.1–F0.6 trước, trình kết quả F0.2 và F0.7 rồi mới chạy G1.

## 11. Kết quả vòng 1 (27/10)

Campaign `full-syn-r1`, notebook [05_full_syn_r1](../../notebooks/training/05_full_syn_r1.ipynb), một seed (0). Số liệu lấy từ `data/training/full-syn-r1/SUMMARY.md` và `stages/*.json`. Thư mục này không vào git (7,6 GB); phải kéo về trước khi trả máy (runbook mục 7).

### 11.1 Val và các cổng

Ngưỡng chọn trên chính tập val nên số val hơi lạc quan.

| Stage | cAcc-M2 | cAcc-M1 (theo nhóm) | Ngưỡng theo nhóm | AP50 | Nhãn giả P / R | I3 / F14 loại | Giờ train | Cổng |
|---|---|---|---|---|---|---|---|---|
| FSA | 77,65% | 23,83% | 0,77 | 92,41% | — | — | 5,04 | D18 đạt (≥ 52,8%) |
| Task 1 | 79,38% | 27,18% | 0,83 | 93,38% | — | — | 5,96 | D17 🟢 (≥ 72,8%) |
| Task 2 | 73,64% | 24,43% | 0,68/0,65 | 94,44% | 99,72% / 99,61% | 5.413 / 68 | 5,36 | Precision ≥ 0,5 |
| Task 3 | 65,04% | 22,86% | 0,53/0,50/0,60 | 94,85% | 98,98% / 99,32% | 5.423 / 75 | 5,23 | |
| Task 4 | 56,33% | 26,05% | 0,37/0,33/0,39/0,46 | 94,59% | 98,66% / 99,32% | 5.128 / 78 | 5,18 | |
| Task 5 | 43,68% | 43,68% | 0,10/0,15/0,18/0,18/0,20 | 94,95% | 98,54% / 99,48% | 5.405 / 90 | 5,90 | |

- **FSA:** AP50 val theo epoch 0,902 / 0,914 / 0,920 / 0,919 / 0,923 / 0,916 / 0,919 / 0,924; chọn epoch 7. Loss train giảm từ 0,156 xuống 0,100 nhưng AP50 val gần như phẳng từ epoch 3. Như vậy giới hạn nằm ở khoảng cách ảnh ghép–ảnh thật chứ không ở số epoch; 3–4 epoch có thể đủ (Q1), tiết kiệm khoảng 2,5 giờ mỗi vòng.
- **Task 1:** Δ = 79,38 − 85,79 = −6,4 điểm so với B. Hai bên đo trên hai tập val khác nhau (mục 8). Cặp hay nhầm nhất của FSA: 41→59 (19 lần), 84→85 (9), 80→81 (7), 73→72 (5).
- **Nhãn giả:** precision ≥ 98,5% và recall ≥ 99,3% ở mọi task. I3 loại khoảng 5.100–5.400 nhãn giả mỗi task, F14 loại 68–90.
- **Quên:** mức quên SKU cũ trên val xấp xỉ 0 hoặc âm ở mọi stage.

### 11.2 Test (chấm một lần)

| Stage | test2019 AP50 | test2019 cAcc (một ngưỡng / theo nhóm) | 6.003 ảnh cũ AP50 | 6.003 ảnh cũ cAcc (một ngưỡng / theo nhóm) | B2 AP50 | B2 cAcc |
|---|---|---|---|---|---|---|
| 1 | 93,29% | 28,00% / 28,00% | 93,25% | 28,65% / 28,65% | 98,35% | 73,16% |
| 2 | 94,26% | 24,21% / 24,23% | 94,34% | 23,90% / 24,25% | 97,54% | 53,32% |
| 3 | 94,63% | 21,49% / 22,07% | 94,63% | 21,42% / 22,12% | 94,66% | 29,97% |
| 4 | 94,38% | 25,40% / 25,68% | 94,34% | 25,27% / 25,62% | 89,72% | 22,29% |
| 5 | **94,79%** | **43,53% / 44,00%** | **94,94%** | **43,15% / 43,68%** | 84,40% | 21,74% |

So B1/B2 trên 6.003 ảnh cũ sau task 5 (FULL-SYN một ngưỡng 0,15 / theo nhóm; B1/B2 một ngưỡng 0,42/0,35; số B1/B2 từ [so sánh backbone](FINAL_BACKBONE_COMPARISON_2026-10-20.md)):

| Chỉ số | FULL-SYN | B1 | B2 |
|---|---|---|---|
| AP50 | 94,94% | 62,33% | 84,40% |
| cAcc | 43,15% / 43,68% | 2,58% | 21,74% |
| cAcc easy / medium / hard (một ngưỡng) | 62,4 / 40,5 / 26,7 | 6,42 / 1,32 / 0,00 | 35,46 / 21,28 / 8,59 |
| ACD (thấp là tốt) | 1,674 / 1,653 | 8,831 | 3,919 |
| mCCD (thấp là tốt) | 0,143 / 0,141 | 0,743 | 0,317 |
| mCIoU | 0,876 / 0,877 | 0,453 | 0,747 |
| Quên trung bình (AP50) | −0,49 | 6,63 | 8,67 |

Trên toàn test2019, quên trung bình là −0,33.

**Vì sao cAcc ở stage 1–4 thấp hơn B2 mà task 5 lại cao hơn.** Khi đếm, vật của SKU chưa học bị bỏ khỏi nhãn, nhưng box model đoán lên các vật đó (dưới tên một SKU đã học) vẫn được tính. Đây là M1, chỉ số của bảng test. Ảnh ghép không có SKU tương lai, nên model FULL-SYN chưa từng học bỏ qua vật lạ và đếm thừa trên các vật này. Khoảng cách M1/M2 cho thấy điều đó: ở task 1 val, M1 là 27,18% còn M2 là 79,38%. B2 học trên ảnh thật, nơi SKU tương lai có mặt nhưng không có nhãn, nên ít đếm thừa hơn. Ở task 5 mọi SKU đã học, M1 = M2, và đây là stage duy nhất so trực tiếp được với B2.

### 11.3 Giới hạn

- Một seed.
- Ngưỡng task 5 rất thấp (0,10–0,20), và ngưỡng của mọi nhóm giảm dần qua các task. Điểm tự tin của model sau nhiều task thấp hơn hẳn, chưa xem phân bố điểm.
- Lý do FULL-SYN hơn B2 là **giả thuyết, chưa kiểm chứng riêng**: ảnh ghép không có SKU tương lai, nhãn giả sạch hơn (I3/F14, precision ≥ 98,5%), và có FSA. Vòng 1 không tách được đóng góp của từng yếu tố.
- Theo D6, đây chỉ là so sánh ở mức hệ thống (FULL-SYN so với B2). FULL-REAL chưa chạy.
- Kỳ vọng trong [tài liệu quyết định](synthetic-data-v2-decisions-2026-10-21.md) là hơn B2 khoảng 5 điểm nếu task 1 đạt khoảng 53–59. Kết quả vượt xa kỳ vọng đó: tỉ lệ giữ cAcc từ task 1 đến task 5 trên M2 val là 43,68 / 79,38 ≈ 0,55, so với giả định 0,45–0,5.

### 11.4 Hướng tiếp theo (chưa chốt)

- Thêm seed để biết chênh lệch 21 điểm so với B2 có ổn định không.
- Ablation để tách đóng góp, ví dụ chạy FULL-SYN với I3/F14 tắt như B2.
- Giảm trần epoch FSA xuống 3–4 epoch (Q1), dựa vào đường AP50 val phẳng.
- Cặp nhầm 41→59 và 84→85 là ứng viên cho nút "cắt bổ sung cutout" nếu chạy vòng chỉnh.
