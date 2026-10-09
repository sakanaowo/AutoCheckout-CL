# Kế hoạch triển khai nhánh investigation

Cập nhật: **10/10/2026**. [Kết luận nghiệm thu notebook foundation](PDP_FOUNDATION_ACCEPTANCE.md): foundation đã hoàn tất; [runtime CLI/runner](TRAINING_RUNTIME_ACCEPTANCE.md) và [adapter ConvNeXt](CONVNEXT_READINESS.md) có nghiệm thu local riêng; native kernel/Vast.ai còn chờ. Phạm vi hiện tại: **tái dựng PDP theo paper, sau đó thay backbone bằng ConvNeXt-V2-Base**. GPU train là **RTX 4090 trên Vast.ai**, không phải GPU local. Dữ liệu đã nhập vào `data/archive`, phải audit trước khi chuẩn bị split hoặc train. Notebook Kaggle của thành viên nhóm chỉ là tham khảo. Xem [bàn giao hệ thống](AGENT_HANDOFF.md) để biết nguồn quyết định và trạng thái repo.

Sau khi kiểm tra reset, đã khôi phục 189 file bị xóa từ HEAD: test/fixture, tools/CLI, runner/config, baseline truy xuất và báo cáo cũ. **Tái sử dụng nền tảng đã có**, không triển khai lại các phần đó từ đầu. Các thay đổi kiến trúc dưới đây vẫn cần thực hiện và kiểm chứng.

Thứ tự ưu tiên: **notebook audit dữ liệu → xác nhận protocol/split → notebook kiểm chứng PDP nền → notebook tích hợp backbone → pilot Vast.ai → đánh giá**. Đợt đầu giữ các thành phần PDP theo paper; LoRA, prototype K=3, freeze shared bổ sung, FSA và generator khay đầy đủ là mở rộng/ablation sau baseline, không tự bật cùng backbone.

**Đã chốt protocol 5 task và nghiệm thu release ảnh thật ngày 07/10.** [Notebook split/acceptance](../../notebooks/data_preprocessing/02_split_and_acceptance.ipynb) đã sinh 22.494 train / 1.503 val / 6.003 test và 5 task JSON, có checksum/test lock. [Notebook review nhãn](../../notebooks/data_preprocessing/03_annotation_review.ipynb) độc lập cho 222 case synth; nguồn đó chưa được thêm vào release. Notebook 04 đã xử lý riêng 202 case (9 giữ/193 tái ghép, task/geometry/CSV PASS); còn 20 pilot pending. Processor/loader/PDP nền đã nghiệm thu notebook 01 và full CUDA smoke notebook 02; tiếp theo CLI/runner explicit resolution và adapter ConvNeXt. Bản augmentation mới chưa tích hợp training. [Chỉ mục triển khai](IMPLEMENTATION_INDEX.md) và [runbook](../data_preprocessing/OPERATIONS.md) là điểm vào code/operations.

## 0. Quy trình notebook bắt buộc

Mỗi bước audit, tiền xử lý, kiểm chứng model, huấn luyện hoặc đánh giá phải có notebook trong `notebooks/`, ghi rõ:

- Mục đích, phạm vi, đầu vào, đầu ra và điều kiện trước khi chạy.
- Thời điểm tạo và thời điểm chạy thực tế **bắt đầu/kết thúc**, run ID, timezone `Asia/Bangkok`; trạng thái chưa chạy/đạt/lỗi/dừng.
- Git revision, config resolved, seed, dependency versions, host/GPU thực tế và checksum data/weights liên quan.
- Lệnh/code được thực hiện, output/log, kết quả và giới hạn kiểm chứng. Notebook chưa chạy không được ghi timestamp chạy giả.
- Các lần chạy mới phải có dấu vết riêng bằng notebook/output version hoặc run artifact; checkpoint/log trên Vast.ai cần được lưu vào vị trí bền vững và lấy về cùng kết quả trước khi kết thúc instance.

Notebook là điểm vào ghi chép và điều phối; thuật toán dùng lại vẫn ở modules/CLI trong repo. Không copy toàn bộ engine vào notebook. [notebooks/README.md](../../notebooks/README.md) mô tả thứ tự và quy ước. [01_data_audit.ipynb](../../notebooks/data_preprocessing/01_data_audit.ipynb) là bước đầu trên data đã nhập.

## 1. Ràng buộc và thiết kế chung

- **Protocol đã chốt: 100+4×25 SKU**, seed mapping 0, đủ 17 nhóm hàng mỗi task. Dùng `TaskConfig` ánh xạ RPC ID sang model label; giữ 224 slots/225 outputs để tương thích mã nền, 24 slot dự phòng không huấn luyện. Nguồn 4 task chỉ để tham khảo.
- Không thêm replay buffer/loader task cũ; teacher là snapshot model, memory lâu dài gồm tham số và prototype tổng hợp. Train JSON chỉ có GT của lớp hiện tại; full GT chỉ dùng audit. Release hiện tại cho phép ảnh nhiều SKU xuất hiện ở nhiều task với nhãn mới khác nhau, chưa bảo đảm điều kiện nghiêm ngặt “mỗi ảnh chỉ dùng ở một task”. Kiểm chứng và ghi rõ quy ước này trong notebook loader/PDP nền trước training.
- Một GPU RTX 4090 24 GB **thuê trên Vast.ai**. Kiểm tra GPU/driver/CUDA/kernel và dependency versions trên instance thực tế. Khởi đầu FP32; chỉ bật TF32/AMP sau khi kiểm tra finite loss và đo thực tế. Máy phát triển có thể audit và chạy test CPU; không mặc định có GPU tại đó. Không sử dụng hướng vận hành GCP cũ.
- Ảnh 640 cho cấu hình chính, 800 cho đối sánh; giữ toàn cảnh khay. Processor, bbox và dự đoán phải dùng cùng quy ước tọa độ.
- Mỗi thành phần có cờ bật/tắt để đo tác động. Không thay đồng thời backbone, LoRA và prototype rồi quy kết cải thiện cho một thành phần.
- Kết quả cũ không được nhập vào bảng nghiên cứu mới. Lưu git revision, config, seed, checksum dữ liệu, weight nguồn và phiên bản thư viện cho mỗi run.

## 2. Milestone và tiêu chí nghiệm thu

| ID | Công việc / kết quả bàn giao | Phụ thuộc | Kiểm chứng cần có | Làm khi chưa có data thật |
|---|---|---|---|---|
| D1 | Split/manifest và 5-task JSON ảnh thật: đã nghiệm thu; synth có gate riêng | 5 task đã chốt; release native v1 đã tạo | T8 đạt: partition/group/hash disjoint, đủ SKU, label đúng và split tái lập | Processor/loader notebook đã đạt; giữ holdout lock |
| S0 | Local Conda/CPU+GPU smoke đã đạt; full suite và môi trường/native kernel Vast.ai còn chờ | Source hiện tại | T0 local có evidence theo runs; cần kiểm chứng toolkit/kernel và suite còn thiếu trước train | Local 3060 đã có evidence |
| S1 | Foundation + CLI/runner với model/processor checkpoint và resolution explicit; [bằng chứng](TRAINING_RUNTIME_ACCEPTANCE.md) | S0, D1 cho data thật | T1: bbox native, loss/resume, batch và optimizer steps qua entrypoint thật | Local 3060 smoke; full training budget ở E1 |
| S2 | ConvNeXt adapter đã PASS notebook 03; cấu hình runtime ở notebook 04 | S1 | T2: features/masks, pretrained preservation, CUDA 640/800 và cold reload đạt | RTX 3060, batch 1 FP32 |
| S3 | Ablation freeze shared/private bổ sung, kiểm soát optimizer | E1; freeze nền kiểm tra ở S1 | T3: tham số/slice cũ bất biến, tham số mới có gradient | Có bằng fixture; thực nghiệm sau baseline |
| S4 | LoRA decoder cross-attention, config và vòng đời adapter | E1, S3 | T4: zero-init equivalence, đúng target, gradient và reload | Có bằng fixture; thực nghiệm sau baseline |
| S5 | Spherical K-Means K=3 và PPG multi-prototype | E1, S3; tương thích S4 nếu bật | T5: cluster, similarity, thiếu prototype, resume, ranh giới task | Có bằng fixture; thực nghiệm sau baseline |
| S6 | Mở rộng CLI mAP/đếm đã có: cấu hình grid/NMS, lưu/nạp policy val, tắt oracle mặc định | S0, S1 | T6: chỉ số biết trước, NMS chéo lớp, ngưỡng val được cố định cho test | Có bằng dự đoán giả |
| S7 | Generator đầy đủ/overlap; compositor resolution đã có trong tools/annotation_resolution.py | S0; data/masks thật cho sản xuất | T7: mask/bbox/area, overlap, seed, tách nguồn split | Chỉ phần ghép với mask giả |
| E1 | Notebook pilot EXP-B1 PDP ResNet / EXP-B2 ConvNeXt trên Vast.ai; đo steps/VRAM/convergence | S1 runtime, S2, S6, D1, GPU train | T9: metrics val, latency/VRAM, run artifacts tái lập | Local foundation/adapter/runtime có evidence; chờ evaluator/train GPU và pilot budget |
| E2 | Baseline đủ 5 task rồi ablation riêng S3/S4/S5/S7 | E1; S3–S5/S7 chỉ là phụ thuộc của ablation tương ứng | T10: metrics sau mỗi task, độ quên, policy calibration và ngân sách memory; full teacher/PPG transition smoke trước protocol | Chờ E1 |

Kết quả audit tài nguyên: [notebook 01b](../../notebooks/data_preprocessing/01b_preprocessing_acceptance.ipynb) full decode/hash 57.710 file đạt, đã phát hiện 222 mask/bbox case. Sau đó notebook 04 khép 202 case bằng bản riêng có owner masks/provenance/task mapping; còn 20 pilot và mục tiêu che khuất riêng. **Release ảnh thật đã nghiệm thu** trong notebook 02 sau quyết định 5 task, xem [báo cáo split](../data_preprocessing/reports/split-100-4x25-2026-10-07.md). Không chặn release real-only vì lỗi synth, và không đánh dấu synth/processor sẵn sàng chỉ vì split đạt.

## 3. S0–S1: nền tảng trước khi đổi mô hình

Tái dựng ở đây là kiểm chứng architecture, prompt pools, DDL/query losses, teacher và PPG so với paper trên dữ liệu RPC. Thay backbone và dataset là thay đổi điều kiện thực nghiệm; không gọi kết quả đó là tái lập nguyên bảng benchmark COCO/VOC của paper. Giữ run PDP nền cùng protocol để đối chiếu tác động backbone.

### Kiểm chứng và tái sử dụng nền tảng hiện có

Nền tảng đã khôi phục từ HEAD. Release split/task JSON và scope resolution 202 có nghiệm thu riêng; kết quả regression CPU hiện hành nằm trong [PROGRESS](PROGRESS.md). Conda pdp đã đạt pip check, limited CPU foundation và full pretrained Task 1 CUDA smoke 640/800 trên 3060; full suite/native kernel/Vast.ai chưa nghiệm thu. Dùng Conda này cho local validation, không suy ra trạng thái `.venv`. Chế độ `merge_annotations(..., size=None)` giữ nguyên geometry/path source để không phải resize/copy ảnh khi chia tập.

- Tái sử dụng `tests/coco_helpers.py`, `tests/synth_rpc.py`, `tests/pdp_helpers.py` và test suites taskcfg/predictions/counting/CL/runner/PDP. Thêm ca kiểm thử còn thiếu cho thay đổi mới; không viết lại fixture tương đương.
- Chạy regression nền trước khi đổi kiến trúc; thêm kiểm tra giao thức 100+4×25 với config hiện có, chỉ thay catalog/mapping khi data mới được xác nhận.
- `configs/tasks_100-4x25_seed0.json` đã đối chiếu catalog và seed, snapshot trong release. Dùng 5 task dữ liệu, 224 slots/225 outputs; bổ sung EXP-B1/EXP-B2 trỏ tới release hiện tại, không lấy paths/split configs GCP cũ.
- Điều chỉnh `scripts/run_exp.sh` thay vì viết lại: hiện hỗ trợ DATA/RUNS/PYTHON, skip task hoàn tất, chỉ chạy lại prediction còn thiếu và resume qua main. Gọi runner từ notebook trên Vast.ai; không bật `--shutdown` vì tùy chọn này không phải cơ chế kết thúc thuê instance Vast.ai. `configs/exp/common.sh` giữ paths/policy cũ để truy nguồn. Dùng `configs/exp/native/EXP-B1.sh` / `EXP-B2.sh` cho release native, resolution explicit và pretrained tương ứng; configs mới bỏ qua evaluator legacy cho đến khi S6 đạt.
- Giữ scripts GCP và `results/` cũ để truy nguồn; không chạy chúng hoặc nhập metrics cũ vào kết quả nghiên cứu mới.

### Processor và training budget

Đã có `--image_size` / `--max_image_size` và processor dùng chung train/val/predict, lưu trong cả hai định dạng checkpoint. `pdp/runtime.py` dựng lại kiến trúc/processor từ checkpoint khi resume hoặc predict-only, không cần weights bootstrap còn tồn tại. Predictions ghi `coordinates=native_pixels`; notebook 04 so bbox với kích thước annotation gốc. Xem [nghiệm thu runtime](TRAINING_RUNTIME_ACCEPTANCE.md).

Phân biệt rõ batch vật lý, accumulation và batch hiệu dụng:

```text
effective_batch = physical_batch × n_gpus × accumulate_grad_batches
steps_per_epoch = ceil(number_of_training_batches / accumulate_grad_batches)
total_optimizer_steps ≈ epochs × steps_per_epoch
```

Với `drop_last`, số batch cần lấy từ dataloader thực tế. CLI hiện từ chối effective batch không chia hết cho `n_gpus*batch_size`; `run_info.json` ghi accumulation, batch thực và optimizer steps toàn task/session. Resume từ chối thay batch/data/optimization contract trong task.

Batch vật lý 4/8/16 là các cấu hình để đo, không phải số đã chứng minh vừa VRAM. Mốc 1.100–2.200 steps/task từ handoff là mục tiêu pilot, phải tính lại từ số ảnh thật và đường cong val. Không mặc định mỗi task có 2.000–3.000 ảnh.

## 4. S2: tích hợp ConvNeXt-V2-Base

Các điểm sửa: `pdp/models/backbones.py` (mới), `configuration_deformable_detr.py`, `modeling_deformable_detr.py`, khởi tạo config trong `engine.py` và CLI trong `main.py`.

1. Tạo factory backbone qua `timm.create_model(..., features_only=True)`. Kiểm tra tên `convnextv2_base.fcmae_ft_in22k_in1k` trong phiên bản cài đặt; fail rõ khi không hỗ trợ. Kiểm tra khả năng tạo model khi `pretrained=False` để test không cần mạng.
2. Với đường ConvNeXt mục tiêu, lấy bốn stage stride 4/8/16/32 và kênh từ `feature_info`; tạo pixel masks theo feature shapes thực tế. Không hard-code `out_indices=(2,3,4)` của ResNet.
3. Tái sử dụng projection `Conv2d(1×1)+GroupNorm` về `d_model` đang có; tránh thêm một bộ projection thứ hai không cần thiết. Kiểm tra `num_feature_levels` khớp số stage.
4. Nạp pretrained backbone riêng. Từ checkpoint Deformable DETR, chỉ nạp tensor có tên/shape và ý nghĩa tương thích; bỏ backbone/projection/classifier không tương thích. Ghi danh sách tensor nạp, bỏ qua, khởi tạo mới; không dùng `ignore_mismatched_sizes=True` làm bằng chứng nạp đúng.
5. Checkpoint phải lưu backbone ID, stage indices, `d_model`, feature levels, class config, processor config và nguồn weights; reload phải dựng cùng kiến trúc trước khi nạp tensor.

T2 cần kiểm tra tensor 640/800, shape và mask từng stage, loss hữu hạn, gradient tới nhánh được train, reload cùng đầu ra ở chế độ eval. Smoke CPU dùng fake backbone để kiểm tra wiring; smoke model thật/GPU là bước kiểm chứng riêng, không suy ra từ test fake.

## 5. S3: freeze theo ranh giới task

Phân biệt kiểm chứng freeze của PDP nền với **thử nghiệm freeze shared thêm sau Task 1**. Chỉ bổ sung policy mới sau khi baseline PDP + backbone hoàn tất; không quy kết cải thiện của policy này cho riêng backbone.

- Thay logic dựa vào substring bằng policy có danh sách nhóm tham số và báo cáo số lượng train/frozen. Policy chạy trước khi tạo optimizer, không phụ thuộc việc người dùng có truyền `--freeze` hay không.
- Task 1: train thành phần theo config pilot. Task ≥2: khóa `shared_p_*`, `shared_k_*`, `shared_a_*`; private prompt của task hiện tại được học, phần task cũ giữ nguyên.
- `query_tf`, `input_proj`, bbox head và embedding chung có policy riêng; cờ hiện tại không đồng nghĩa “chỉ freeze shared pool”. Backbone/encoder/decoder base được kiểm soát tách khỏi adapter LoRA.
- Với tensor chứa nhiều private slice, bảo đảm optimizer momentum/weight decay không làm đổi slice cũ; có thể tách parameter theo task hoặc bảo vệ update theo mask. Chọn cách triển khai sau khi có test bắt lỗi.
- DDL chỉ cập nhật các tham số còn được phép học. So sánh trọng số trước/sau một optimizer step ở Task 2; kiểm tra đồng thời shared không đổi, private cũ không đổi và private mới có gradient.

## 6. S4: LoRA trên cross-attention thực tế

**Mở rộng sau baseline**, tắt trong đợt tái dựng PDP + thay backbone đầu tiên.

Tạo `pdp/models/lora.py`; adapter linear có công thức `W(x) + (alpha/r) B(A(x))`, `r=8`, `alpha=16`, khởi tạo nhánh bổ sung bằng 0. Khóa base weights theo policy S3, thêm adapter params vào optimizer với lr tường minh.

Target đề xuất ban đầu: `model.decoder.layers.*.encoder_attn.value_proj` và `.output_proj`. In danh sách target và fail nếu không match; không nhắm `self_attn.q_proj/v_proj` rồi báo là đã thay cross-attention. `sampling_offsets`/`attention_weights` để ngoài cấu hình khởi đầu, chỉ thêm trong ablation riêng.

**Vòng đời adapter cần chốt bằng thí nghiệm:** bắt đầu với một adapter có kích thước cố định cho đường smoke/Task 1. Trước E2, ghi policy Task ≥2 (adapter chung tiếp tục train hay adapter theo task có routing). Adapter chung có nguy cơ đổi không gian đặc trưng của lớp cũ; adapter theo task tăng bộ nhớ và cần routing khi inference không biết task. Không tuyên bố LoRA tự bảo đảm không quên. Teacher phải giữ snapshot adapter của task trước.

T4: khi bật adapter zero-init, đầu ra khớp model gốc; base frozen không có gradient, LoRA có gradient; save/load giữ config và tensor; teacher không đổi khi student update. Các bước merge weights, nếu hỗ trợ, cần test riêng và không merge vào teacher nhầm thời điểm.

## 7. S5: prototype K=3 xuyên suốt PPG/checkpoint

**Mở rộng sau baseline**. Đợt đầu giữ prototype một vector/lớp của PDP để đo riêng tác động backbone.

Tạo hàm thuần trong `pdp/prototypes.py`, sau đó nối vào `engine.py`, `ppg.py`, `ppg_audit.py` và checkpoint hooks.

1. Tích lũy feature của đối tượng được match với GT hiện tại và phân loại đúng; normalize L2; không dùng nhãn đầy đủ của lớp cũ cho train.
2. Thực hiện spherical clustering: gán cluster bằng cosine lớn nhất, cập nhật centroid từ mean rồi normalize, seed cố định. Không chỉ gọi Euclidean `KMeans` rồi đổi tên.
3. Lưu tensor `[num_classes, 3, d_model]` và mask hợp lệ `[num_classes, 3]`. Với ít mẫu, chỉ dùng số cluster đủ dữ liệu; vector 0 không phải prototype hợp lệ; empty cluster có xử lý xác định.
4. PPG dùng `max_k cosine(feature, prototype[class,k])` trên cluster hợp lệ. Nearest-prototype mode so sánh điểm lớp đã tổng hợp; giữ nguyên giới hạn label `< PREV` và các ngưỡng confidence của hai nhánh.
5. Giữ prototype lớp cũ qua task transition; clear cache đặc trưng task vừa xong sau khi tổng hợp. Khi backbone/adapter làm đổi không gian embedding, đo độ lệch feature và chất lượng PPG; centroid cũ không tự được bảo đảm tương thích.
6. Đặt version cho schema checkpoint. Checkpoint một prototype/lớp chỉ chuyển thành một cluster hợp lệ nếu policy cho phép; không tự nhân bản thành ba cluster độc lập.

T5: vector normalize, cluster có hướng biết trước, seed tái lập, ít mẫu/zero vector/empty cache, PPG max-cosine, checkpoint/resume và cache task cũ không được dùng lại để fit cluster task mới. Lưu số lớp/cluster thiếu prototype và dung lượng memory/task.

## 8. S6: evaluation và calibration độc lập

Mở rộng `tools/eval_count.py` và tái sử dụng `tools/eval_cl.py` hiện có. CLI counting đã chọn threshold trên val, áp sang test, hỗ trợ `--nms-iou` và ghi metadata, nhưng dùng grid cố định 0.05–0.95 và luôn tính oracle. Cần thêm cấu hình grid, artifact calibration riêng và khả năng nạp policy đã khóa; oracle phải tắt trên đường nghiên cứu chính.

- Input: dự đoán `.npz`, COCO GT, task config, `seen_classes`, hậu xử lý config. Kiểm tra image IDs/label range, annotation checksum và tọa độ trước khi tính chỉ số.
- Nhánh mAP: dùng output detector thô theo quy ước top-k hiện tại, không áp NMS đếm âm thầm.
- Nhánh đếm: `top1_per_query` → class-agnostic NMS theo từng ảnh (khởi đầu IoU 0.45) → score threshold → count vector → `cAcc`, `ACD`, `mCCD`, `mCIoU`.
- Calibration trên val: grid khởi đầu 0.10–0.90 bước 0.01, tối đa hóa cAcc, hòa thì chọn ngưỡng nhỏ nhất. Lưu `score_threshold`, `nms_iou`, grid/tie-break, task/checkpoint ID và checksum val thành artifact; test chỉ đọc policy này.
- Nếu tune cả NMS thì chỉ dùng val và lưu toàn bộ policy. Không chọn threshold hoặc policy bằng test/oracle; hàm oracle còn trong lõi không thuộc pipeline chính mới.
- Ghi rõ `K`, `K_eff`, lớp đã thấy và quy tắc bỏ lớp zero-GT khỏi trung bình mCCD/mCIoU. mCIoU là count IoU, không đánh giá định vị bbox.

T6 dùng dự đoán giả có kết quả đếm biết trước, box hai nhãn khác nhau cho cùng vật, hai vật thật chồng lấn và tie scores. Thay GT test phải không làm đổi policy calibration. Cần kiểm tra dedup trước score threshold không bỏ điểm thuộc grid cấu hình mới.

## 9. D1 trước training; S7 chỉ khi cần sinh thêm khay

Source giữ nguyên ở `data/archive`. Release `data/processed/rpc_100-4x25_seed0_native_v1` chỉ có annotation/split/task JSON **real-only**, giữ pixel gốc và mapping 5 task đã chốt. Validation/test độc lập theo group và checksum đã khóa; source val2019 và 208 nền thuộc train. Nguồn synth 4 task còn được review ở notebook 03, không đưa thẳng vào release mới.

Data nhập đã có pipeline tham khảo tại `data/archive/scripts/`: trích SAM2, tạo background và ghép COCO; có index cutout và ảnh synth hoàn tất. Đọc source để đối chiếu, không tự chạy script hoặc môi trường đi kèm archive. [Notebook nghiệm thu tài nguyên](../../notebooks/data_preprocessing/01b_preprocessing_acceptance.ipynb) kiểm tra vật liệu hiện có trước khi quyết định sinh lại; không mặc định phải tách mask/tạo nền từ đầu.

Compositor có giới hạn cho resolution đã có trong `tools/annotation_resolution.py`; `compose_replacement`/owner-mask export không tương đương generator đủ mục tiêu overlap 20–45%. Nếu mở rộng generator đầy đủ, tái sử dụng compositor và cutout/background đã đạt kiểm tra và bổ sung provenance, config, seed/transform cùng policy overlap/split. Chỉ cần ảnh `train2019` hoặc SAM weights khi tái trích cutout từ ảnh gốc; ghép lại từ cutout đã có không bắt buộc chúng. Test compositor bằng mask giả trước.

Kiểm tra hiện tại: 208 nền, 7.502 cutout/200 SKU và 20.000 synth đã tồn tại. Khép các nhãn mask/bbox lệch và truy vết source trước khi quyết định tái ghép. Source nhập có horizontal flip 50%; phải ghi policy và kiểm tra ảnh chữ/logo trong ablation, không bật lại âm thầm. GRN của ConvNeXt-V2 kiểm chứng ở S2/model, không đặt tiêu chí “có background thì hết bão hòa kênh”.

Mục tiêu overlap 20%–45% trong handoff cần được định nghĩa bằng diện tích mask bị che so với diện tích mask gốc, có xử lý giới hạn canvas và thất bại sau số lần thử hữu hạn. Xuất ảnh, bbox/area theo quy ước đã chọn, mask/visibility và provenance sản phẩm/nền/seed. Chia nguồn train/val/test **trước khi** sinh ảnh để một sản phẩm/nền nguồn không gây leakage giữa split. Chỉ ghép SKU/nguồn hợp lệ cho giao thức task hiện tại.

Audit trước khi train: checksum/version, catalog 200 SKU, đường dẫn và bbox bounds, ID duy nhất, split theo nhóm, nguồn ảnh thật/ghép, class coverage từng task và JSON train chỉ có nhãn hiện tại. Mapping 5 task đã được đối chiếu và khóa; bản resolved 202 đã xuất task JSON theo mapping này; toàn bộ nguồn synth chưa phát hành lại. Không dùng nguyên `tasks_pdp/group*` 4 task. Mọi nguồn bổ sung phải giữ nguyên holdout đã khóa và được nghiệm thu trước khi phát hành release mới.

## 10. E1–E2: thực nghiệm sau audit và chốt protocol

**EXP-B1 (baseline mới trên Vast.ai):** PDP với backbone hiện tại trên Task 1, cùng dữ liệu, resolution, số bước tối ưu và evaluator với EXP-B2; giữ protocol theo quyết định mới. Không tự bật FSA/LoRA/K=3/freeze bổ sung. Tên này không tham chiếu kết quả E0/E1 cũ.

**EXP-B2:** PDP + ConvNeXt-V2-Base trên Task 1 của protocol được chọn; LoRA tắt, prototype mean baseline giữ để đo riêng tác động backbone. Ghi rõ pretrained weights và thành phần train/frozen. Quét batch vật lý 4/8/16 theo VRAM, ghi batch hiệu dụng, số bước tối ưu, loss/val curves, peak VRAM, latency và tensor load report trong notebook có timestamp. Nếu không hội tụ, kiểm tra loader/classifier/optimizer/số bước; FSA nếu thử phải là run/ablation riêng.

Sau pilot, chạy toàn bộ protocol đã chọn khi checkpoint/resume, teacher và memory policy đã test. Sau mỗi task lưu `mAP@C/P/A`, ma trận theo task, chỉ số đếm tại lớp đã thấy và policy calibration. Sau task cuối báo cáo checkout trên 200 SKU bằng test cố định. Các cải tiến freeze/LoRA/K=3/FSA/dữ liệu ghép được đo trong run riêng sau baseline và có notebook ghi thời điểm chạy.

Không đặt trước AP/cAcc như kết quả đã đạt. Chỉ công nhận một milestone khi có lệnh/output kiểm chứng và artifacts của run tương ứng.

## 11. Các bước tiếp theo

Phạm vi notebook processor/loader/PDP nền đã hoàn tất; xem [kết luận nghiệm thu](PDP_FOUNDATION_ACCEPTANCE.md).

1. **S1/S2 local:** đọc [runtime acceptance](TRAINING_RUNTIME_ACCEPTANCE.md) và [ConvNeXt acceptance](CONVNEXT_READINESS.md); giữ artifacts/notebooks 01–04, không tạo lại adapter hoặc pipeline song song.
2. **Môi trường train + S6:** kiểm chứng toolchain/kernel hoặc fallback có đo throughput trên máy training; calibration policy chọn trên val, lưu/nạp cho test, oracle tắt ở pipeline chính.
3. **E1:** pilot EXP-B1/EXP-B2 Task 1 cùng real-only split/resolution/optimizer-step budget/evaluator; ghi loss/val curves, VRAM, latency và run artifacts. Budget smoke 4 steps không phải budget pilot.
4. **E2:** full-model Task 1→2 teacher/PPG/prototype/current-task-only loader smoke, sau đó baseline đủ 5 task và metrics/counting/forgetting trên holdout cố định.

Synthetic 202/20 pilot pending là track riêng; không chặn baseline real-only. LoRA/K=3/FSA/freeze bổ sung/generator overlap là ablation sau baseline, không là điều kiện bắt buộc để bắt đầu baseline E2.


Cập nhật S2 ngày 09/10: notebook 03 đã đạt CUDA smoke 640/800 (run aeeb770b); notebook 04 tiếp nối qua CLI/runner thật. Native kernel/Vast.ai, calibration và convergence chưa được suy ra từ các smoke local.
