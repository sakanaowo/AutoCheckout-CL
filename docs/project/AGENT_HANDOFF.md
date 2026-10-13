# Bàn giao AutoCheckout-CL

**DỪNG THEO NGƯỜI DÙNG — 13/10/2026 09:56 UTC+7, chuẩn bị chuyển L4 → 4080 SUPER.**
ConvNeXt Task 1 đã COMPLETED; Task 2 dừng ở observed step 6.420, checkpoint resume
**6.414**, backup 6.365. ResNet chưa bắt đầu. Đã đọc/audit checkpoints, tensors hữu hạn,
đủ states; GPU trống, không còn training. Giữ toàn bộ artifacts, không tự launch lại.
[Điểm dừng, manifest và paths cần chuyển](../environment/TRANSFER_TO_4080_2026-10-13.md).
Settings đích `data/training/settings_baseline_4080.json`; máy 4080 chưa kiểm chứng.
Các mô tả RUNNING bên dưới là snapshot trước điểm dừng này.

**Cập nhật máy đích 12/10/2026: NVIDIA L4 trên GCP**, thay cho phương án 4080 SUPER.
[Setup L4](../environment/GCP_L4_SETUP_2026-10-12.md) đã đạt Python/CUDA/native kernel,
58 regression + 4 kernel tests; data đã tải/giải nén bằng rclone. Checkpoint step
31.550 và backup khớp hashes nguồn, ConvNeXt strict restore/640px forward trên L4 đạt.
Smoke resume trên bản checkpoint riêng đã đạt 2 updates đến 31.552: Lightning
restore đủ model/optimizer/scheduler/PDP memory, batch 8, finite losses, peak allocated
6,847 GiB; checkpoint baseline gốc giữ nguyên ở 31.550.
Settings: `data/training/settings_baseline_l4.json`; plan chỉ đổi roots, campaign/budget
giữ nguyên. **Baseline RUNNING trên L4 từ 12/10/2026 23:46:00 UTC+7**,
tmux `ac-baseline-l4`, ConvNeXt Task 1 resume từ 31.550 đã có updates mới.
Notebook 03 và bản execution autosave 15 giây, console tổng `tmux_baseline_l4.log`
trong campaign; `baseline_launch_l4.json` giữ receipt/PID attempt paths.
Session detached vẫn train khi SSH ngắt. Snapshot Vast.ai bên dưới là lịch sử.

**DỪNG THEO YÊU CẦU NGƯỜI DÙNG — 11/10/2026 15:08:40 UTC+7.**
Baseline ConvNeXt dừng ở Task 1, completed 11 epochs; checkpoint step 31.550
có thể resume trên GPU tương thích. **Giữ artifact**, không tự khởi chạy lại.
[Báo cáo dừng và chuyển GPU](BASELINE_4090_STOP_REVIEW_2026-10-11.md) ghi validation,
audit, manifest và điều kiện tiếp tục. ResNet chưa bắt đầu, full baseline chưa hoàn tất.


Snapshot nền **11/10/2026**; trạng thái baseline/GPU mới nhất ở đầu tài liệu. Notebook modeling 01/02 đã đạt processor/loader và full pretrained ResNet CUDA smoke.
[ConvNeXt notebook 03](CONVNEXT_READINESS.md) đã đạt 8/8 gates, 29 tests trên RTX 3060 (run aeeb770b).
CLI/runner và notebook 04 có triển khai/kiểm chứng riêng tại [báo cáo runtime](TRAINING_RUNTIME_ACCEPTANCE.md).
Các run model local dùng PyTorch fallback. [S6 calibration/evaluation](COUNT_CALIBRATION_ACCEPTANCE.md) đã nghiệm thu kỹ thuật bằng CPU, policy val và test-only reload.
[Pilot Task 1 RTX 4090](TASK1_PILOT_4090_REVIEW.md) đã COMPLETED ngày 10/10 lúc 18:14:57 UTC+7: đủ hai models, 21.752 steps/model, 100/100 prototypes, policy val và test metrics. B1/B2 test M1 AP 26,88%/73,54%, cAcc 3,50%/30,57%.
[Notebook 02 transition smoke](TASK_TRANSITION_4090_REVIEW.md) đã COMPLETED ngày 11/10 lúc 09:13:40 UTC+7: 8 steps/model, teacher frozen/16 calls, PPG 62/46, old prototypes/private prompt rows giữ nguyên. Chưa có prototype lớp mới; scope 16 ảnh/7 lớp mới chỉ đạt kỹ thuật. [Benchmark GPU baseline](BASELINE_4090_CONFIGURATION.md) đã PASS 20 cases: batch/effective 8/8, workers 8, FP32 + TF32. Notebook 03 khởi chạy lúc 10:54:46 ngày 11/10 UTC+7, tmux `ac-baseline`, ConvNeXt Task 1–5 trước rồi ResNet Task 1–5. Budget khởi đầu 16 epochs/task; ưu tiên validation, chưa nghiệm thu hội tụ hoặc full baseline.

[Snapshot đầu ngày 09/10](HANDOFF_2026-10-09.md) giữ bằng chứng dependency/branch tại lúc viết, trước các run model đã đạt;
đọc cùng kết luận hiện hành, không dùng các đoạn “chưa chạy” trong snapshot làm trạng thái model mới nhất.
Đo hai model cùng train Task 2 ở batch 8 cũng PASS: GPU memory used 19,30 GiB,
không OOM, nhưng B1/B2 chỉ 6,77/5,44 ảnh/s so với 16,80/10,94 khi chạy riêng.
Giữ notebook 03 chạy tuần tự; xem [benchmark đồng thời](BASELINE_4090_CONFIGURATION.md).

[Khảo sát resource/cost 11/10](GPU_COST_REVIEW_2026-10-11.md): batch 8 peak GPU-wide
9,59/9,72 GiB B1/B2; current price 0,4153 USD/h, DLPerf 49,57. Shortlist host 4090
khác giá 0,422–0,459 (DLPerf 97–123), hoặc hai GPU riêng 2×3090 giá 0,323 tổng /
2×4070 Ti SUPER giá 0,425 tổng để nghiên cứu parallel. Chưa thuê/chuyển instance,
chưa benchmark PDP trên các offers mới; không coi DLPerf là tốc độ PDP đã kiểm chứng.
[Nhật ký model](../timelines/model-foundation-2026-10-09.md) giữ chi tiết runs PASS/FAIL và các sửa lỗi.
Các làm rõ của người dùng có ưu tiên hơn handoff gốc.
Quyết định 10/10/2026: bỏ notebook nghiệm thu môi trường Vast.ai riêng; bước tiếp là chuẩn bị pilot local và thực chạy pilot/chuyển Task 1→2/baseline 5 task trên RTX 4090. Ghi metadata/VRAM/thời gian trong các run đó; fixture/debug nhỏ tiếp tục dùng CPU/3060 local.
Yêu cầu tiếp nối: mọi artifacts của training notebooks ghi dưới `data/training/` để upload; logging từng step/epoch/task và resume phải có evidence dry run local. [Notebooks 01–03](../../notebooks/training/README.md) dùng [settings](../../configs/training/notebooks.json), headless executor/tmux theo [runbook](TRAINING_NOTEBOOK_RUNBOOK.md). Không tạo lại notebook environment riêng; real pilot 4090 đã hoàn tất trong campaign `vast4090-seed0-640-v1`, full training còn chờ. Dùng `data/training/settings_4090.json` cho notebook 02; notebook 03 mặc định đọc `data/training/settings_baseline_4090.json`, budget 16 đã khóa; checkpoint đang giữ để chuyển GPU.

[Conda cài sạch 10/10](CONDA_CLEAN_INSTALL_ACCEPTANCE.md) đã đạt trên local RTX 3060: Python 3.10.22,
105 tests, ResNet/ConvNeXt 640 step 1→4 và notebook pilot CPU dry run. Requirements bổ sung **setuptools==81.0.0**:
lần cài mới kéo setuptools 84 khiến Lightning thiếu pkg_resources dù pip check đạt.
Artifacts ở `data/training/validation/conda_py310_clean_install_20261010/`; env thử còn trong /tmp.
Không chuyển project sang Python 3.9/3.12; trên máy thuê dùng env Conda Python 3.10 riêng.

## 1. Phạm vi và thứ tự ưu tiên

- Bài toán: phát hiện và đếm sản phẩm trên khay checkout; mục tiêu RPC gồm 200 SKU.
- **Đã chốt 07/10: 5 task 100+4×25 SKU**, seed mapping 0, phân tầng đủ 17 nhóm hàng mỗi task. Nguồn 4 task `[48,68,47,37]` giữ làm tham khảo/đối chứng. 24 slot dự phòng hiện có không phải task dữ liệu. Ràng buộc replay-free giữ nguyên: loader train chỉ đọc task hiện tại, không bổ sung replay ảnh/GT task cũ; được giữ trọng số, prompt và prototype embedding.
- [Notebook so sánh protocol](../../notebooks/data_preprocessing/01c_protocol_comparison.ipynb) ghi phân tích trước quyết định; output lịch sử được giữ. Người dùng đã chọn 5 task sau lần chạy đó, không phải kết luận protocol này chắc chắn có cAcc cao hơn.
- GPU nguồn baseline là **L4 GCP**, đã dừng để chuyển sang **4080 SUPER**; target chưa kiểm chứng. Pilot trước đó chạy trên RTX 4090 Vast.ai. Dùng Python 3.10/Torch cu121 theo pins, build kernel mới trên GPU đích; không suy ra full 5-task convergence từ Task 1.
- Độ phân giải mục tiêu: **640×640** cho cấu hình chính; **800×800** cho cấu hình đối sánh. Cần cấu hình processor và kiểm tra biến đổi bbox, không chỉ đổi kích thước file ảnh.
- Mục tiêu đợt đầu: **tái dựng PDP theo paper rồi thay backbone ConvNeXt-V2-Base**. Giữ losses/prompt/teacher/PPG và prototype nền để đối chiếu; LoRA, K=3, FSA và freeze shared bổ sung là mở rộng/ablation sau baseline.
- Dữ liệu đã nhập tại **`data/archive`**. **Audit data trước** khi chuẩn bị split và train; không tiếp tục ghi là chờ dữ liệu.
- **Mỗi bước đều phải có notebook** mô tả làm gì, input/output, config và thời điểm chạy bắt đầu/kết thúc theo `Asia/Bangkok`; lưu output và run ID. Xem [quy trình notebooks](../../notebooks/README.md) và [notebook audit đầu tiên](../../notebooks/data_preprocessing/01_data_audit.ipynb).
- Xây baseline và kết quả mới có thể tái lập. **Không dùng bảng kết quả E0/E1/E4/E5 cũ làm kết quả của nhánh investigation.** Có thể tái sử dụng mã sau khi kiểm chứng.

## 2. Nguồn bàn giao và tài liệu tham khảo

| Nguồn | Vai trò hiện tại |
|---|---|
| [Bản handoff gốc từ vault](../references/agent-handoff-vault-2026-10-06.md) | Lưu nguồn các quyết định kiến trúc và mục tiêu; một số mô tả chưa khớp mã hiện tại |
| [Notebook của thành viên nhóm](../../notebooks/references/notebookac5e317914.ipynb) | Tham khảo cách chạy PDP trên Kaggle, tăng số optimizer steps và kiểm tra học/tổng quát hóa; không phải runner chính |
| Ảnh chụp Kaggle do người dùng cung cấp | Tham khảo dataset `hinvminh12/rpc-checkout-data`; giao diện hiển thị Version 1, 16.56 GB và các thư mục ở mục 5 |
| [Kế hoạch hiện tại](IMPLEMENTATION_PLAN.md), [tiến độ hiện tại](PROGRESS.md) | Tài liệu vận hành nhánh investigation; phần tháng 09 đã tách vào docs/archive |
| [GCP_TRAINING_GUIDE.md](../archive/GCP_TRAINING_GUIDE.md), [báo cáo tháng 09](../archive/status-2026-09-29.md) | Tham khảo vận hành và thí nghiệm của hướng cũ; không xác nhận trạng thái máy hoặc kết quả hiện tại |

Notebook dùng **4 task** với số lớp `[48, 68, 47, 37]`; cấu hình đang lưu là `MODE='A'`, `group_mini`, 10 epoch, batch 2, ảnh 512. Output lưu dừng ở epoch 6 và chưa có kết quả cuối. Không lấy split, checkpoint hay lịch train đó làm mặc định cho giao thức 5 task.

## 3. Trạng thái repo hiện tại

Code nền đã khôi phục sau reset: 189 file được đối chiếu Git blobs tại thời điểm khôi phục, gồm CLI/runner/tests/baseline/results. Không cần viết lại các phần đó. Kết quả tháng 09 giữ làm lịch sử.

Phần mới đã triển khai: native merge và split 5 task; visual review; OpenAI pilot/cache/content audit và Batch; API chốt rồi áp dụng scope 202 bằng tái ghép/owner masks/RLE/task JSON. [Chỉ mục triển khai](IMPLEMENTATION_INDEX.md) chỉ rõ code, config, notebook, tests và output. [Hướng dẫn vận hành](../data_preprocessing/OPERATIONS.md) mô tả cách chạy lại.

Code model nền nằm trong `pdp/main.py`, `engine.py`, `models/`, `ppg.py`, `checkpointing.py`, `inference.py`; metrics/mapping/provenance ở `autocheckout/`, evaluation CLI ở `tools/`, runner ở `scripts/`. Notebook 02 đã kiểm chứng full pretrained Task 1 CUDA smoke trên 3060; điều đó chưa nghiệm thu full 5-task training hoặc toolchain Vast.ai. ConvNeXt adapter đã đạt CUDA smoke; LoRA và K=3 bổ sung vẫn là phần sau baseline.

Notebook [01](../../notebooks/modeling/01_processor_loader_pdp_acceptance.ipynb) đạt limited foundation trong run 65687942; [02](../../notebooks/modeling/02_full_pdp_baseline_acceptance.ipynb) đạt full CUDA smoke trong run d967dc7b. Conda pdp Python 3.10/pip check đạt, Torch/vision 2.2.2/0.17.2, typing-extensions 4.13.2, packaging 24.2, Ninja 1.13.0. [Downloader public](../../tools/download_pdp_pretrained.py) tự nhận cache path và lưu revision/hash. [README môi trường](../environment/README.md) giữ hướng dẫn cài; không tự cài local. Full model smoke đạt với PyTorch fallback, chưa chứng nhận native kernel/convergence/Vast.ai.

## 4. Quyết định kiến trúc và điểm phải làm rõ khi code

### Backbone

**ConvNeXt-V2-Base**, tên `convnextv2_base.fcmae_ft_in22k_in1k`, đã có adapter metadata ở `pdp/models/backbones.py`. Factory chọn 3 stage cuối từ provider, nối masks/projections cho Deformable DETR; notebook 03 đã kiểm chứng 640/800, pretrained preservation và cold reload.

Giữ đường chạy backbone hiện tại để so sánh trên cùng dữ liệu mới. Kiểm tra tên model với bản `timm` cài đặt trước khi thay dependencies; không mặc định pretrained ResNet và ConvNeXt có thể nạp chung toàn bộ checkpoint.

### Shared/private pool

Đợt đầu đối chiếu freeze của PDP với paper. Đề xuất **freeze shared thêm sau Task 1** từ handoff trước được giữ cho ablation sau baseline. Code có `--freeze_shared_after_task1`, nhưng phần freeze đang xử lý `input_proj`, `query_tf`, query/reference embedding và bbox head; **chưa thấy freeze các tham số `shared_p_*`, `shared_k_*`, `shared_a_*` bằng cờ này**. Nhánh đó còn nằm trong điều kiện `args.freeze` không rỗng. Cần kiểm chứng trên tên tham số thực tế, không tự bật policy bổ sung khi chỉ đo tác động backbone.

Private pool lưu nhiều task trong cùng tensor. Việc `detach` slice cũ trong forward chưa đủ để kết luận trọng số cũ bất biến qua AdamW/weight decay; test phải so sánh slice cũ trước và sau optimizer step.

Giữ bối cảnh khay trong ảnh detector. Ý nghĩa “shared pool học nền” là giả thuyết nghiên cứu, cần ablation; DDL không tự chứng minh gradient giữa các task không xung đột.

### LoRA

**Mở rộng sau baseline**, không bật trong đợt PDP + ConvNeXt-V2-Base đầu tiên.

Giữ mục tiêu `r=8`, `alpha=16`. Cross-attention thực tế là `DeformableDetrMultiscaleDeformableAttention` với `sampling_offsets`, `attention_weights`, `value_proj`, `output_proj`; **không có `q_proj` trong nhánh này**. `q_proj/v_proj` của attention thông thường thuộc một nhánh khác, không được thay thế nhầm.

Kế hoạch đề xuất bắt đầu từ `decoder.layers.*.encoder_attn.value_proj` và `output_proj`. Đây là cách thích ứng đề xuất với mã hiện tại, không phải khẳng định đã triển khai đúng công thức `Wq/Wv` trong nguồn handoff. Phải chốt vòng đời adapter ở ranh giới task trước khi chạy học liên tục.

### Prototype

Hiện tại `compute_class_prototypes()` lấy trung bình cache để tạo một vector/lớp; **giữ cơ chế này cho baseline**. **Spherical K-Means, K=3** là mở rộng sau baseline, chỉ tích lũy đặc trưng của đối tượng task hiện tại được phân loại đúng, không lưu ảnh/bbox cũ. Chuẩn hóa vector; ghi mask cluster hợp lệ; dùng độ tương đồng lớn nhất với prototype hợp lệ của lớp.

Thay đổi ảnh hưởng cả PPG, teacher cache, audit và checkpoint. Chỉ giữ prototype tổng hợp của task đã xong; feature cache dùng trong task phải được dọn ở ranh giới task theo giao thức đã ghi.

### Đếm và đánh giá

Lõi `autocheckout/counting.py` và CLI `tools/eval_count.py` đã có class-agnostic dedup/NMS, bốn chỉ số đếm và chọn threshold trên **validation** rồi áp cho test trong cùng lần đánh giá. CLI hiện còn tự báo cáo oracle test, ghi NMS/grid/checksum vào metrics nhưng chưa có artifact calibration riêng để dùng lại. Cần mở rộng CLI hiện có để lưu/nạp policy và tắt oracle trên pipeline nghiên cứu chính.

Handoff đề xuất NMS IoU **0.45**, score grid **0.10–0.90**. Code hiện có grid **0.05–0.95**, bước 0.01; runner cũ xuất kết quả không NMS và NMS **0.5**. Cần cấu hình grid/NMS cho hướng mới và ghi lại trong kết quả. Không nhầm NMS nhãn giả của PPG với NMS đầu ra đếm. Giữ output mAP thô để so sánh detector; lưu hậu xử lý đếm riêng.

`mCIoU` trong mã là IoU trên **số lượng theo lớp**, không phải IoU hình học giữa bbox. Các vật có thể chồng lấn trong ảnh top-down; ngưỡng 0.45 là cấu hình khởi đầu cần kiểm chứng, không phải định luật bảo đảm một vật/box.

## 5. Dữ liệu và kết quả hiện hành

- Nguồn giữ nguyên tại `data/archive`: 6.000 val2019, 24.000 test2019, 20.000 synth; nguồn task `[48,68,47,37]` khác protocol mới. `instances_train2019.json` có metadata nhưng thiếu ảnh train2019. Không thực thi code/.venv đi kèm archive.
- Audit và nghiệm thu tài nguyên đã decode/hash 57.710 file: 208 background, 7.502 cutout cho 200 SKU; catalog dùng được. Original owner maps/transform từng ảnh synth không có; 222 case bbox/polygon cần xem không đồng nghĩa 222 nhãn chắc chắn sai.
- **Real-only release:** `data/processed/rpc_100-4x25_seed0_native_v1`, 22.494 train/1.503 val/6.003 test, pilot 3.002; native pixels, 5 task, 9 gate đạt. Toàn bộ val2019 thuộc train để giữ nguồn nền; một phần test2019 dùng train. Đây là split nghiên cứu, không phải official RPC test2019 benchmark. Giữ holdout/checksums đã khóa.
- **Resolution 202:** `data/processed/rpc_synth_resolved_202_v1`, 202 ảnh/2.846 object/200 SKU, 9 giữ/193 tái ghép thật. 2.721 object có owner-mask/RLE mới được xác minh. Có provenance và mapping 5 task; là bản augmentation riêng, chưa thêm vào release ảnh thật.
- **CSV hiện hành:** `data/processed/synthetic_annotation_review_v1/review_decisions.csv`: 193 regenerate đã áp dụng / 9 giữ / 20 pilot pending ngoài scope. JSON queue và notebook 03 là lịch sử; notebook 04/manifest/CSV là bằng chứng áp dụng mới.
- API pilots/Batch trước resolution có gate nội dung FAIL. Lượt reviewer cuối cho 202 chạy 08/10 13:16:30–13:24:42 UTC+7; apply/acceptance 13:31:19–13:34:20, không gọi lại API. [Báo cáo resolution](../data_preprocessing/reports/annotation-resolution-202-2026-10-08.md) ghi cost, IDs và giới hạn.

Nghiệm thu scope 202 không chứng nhận original mask của 9 ảnh giữ, accuracy GT của API, toàn bộ nguồn synth, overlap 20–45% hay GRN. Tái ghép tạo ảnh/pose mới; số 193 là action thận trọng, không phải số bbox gốc đã chứng minh sai.

Train task chỉ đọc nhãn lớp hiện tại; full GT dùng audit. Ảnh nhiều SKU có thể xuất hiện ở nhiều task trong release này; chưa bảo đảm điều kiện mỗi ảnh chỉ dùng một lần. Notebook loader phải ghi và kiểm tra quy ước này. [Schema](../data_preprocessing/formats.md) phân biệt RPC IDs, model labels và native geometry.

## 6. Điểm bắt đầu cho người tiếp quản

1. Đọc tài liệu này, [chỉ mục triển khai](IMPLEMENTATION_INDEX.md), [hướng dẫn vận hành](../data_preprocessing/OPERATIONS.md), rồi [kế hoạch hiện tại](IMPLEMENTATION_PLAN.md).
2. Kiểm tra `git status`, branch và file thực tế; bảo toàn code/tests/config đã khôi phục. Kết quả cũ chỉ để tham khảo, không vận hành lại GCP theo nhật ký tháng 09.
3. **D1 ảnh thật đã đạt:** dùng release đã khóa ở data/processed; không chia lại test. Bản synth 202 tách riêng đã đạt nghiệm thu; còn 20 pilot pending, chưa tích hợp augmentation.
4. **S1 runtime/S2:** đọc [báo cáo notebook 04](TRAINING_RUNTIME_ACCEPTANCE.md) để lấy trạng thái CLI/runner thật. Model/processor được lưu trong checkpoint; dùng configs native EXP-B1/B2, giữ configs tháng 09 làm lịch sử. Notebook 03 adapter đã PASS.
5. **Tiếp tục baseline trên GPU đích:** pilot/transition và ConvNeXt Task 1 đã hoàn tất. L4 dừng theo người dùng; tmux `ac-baseline-l4` pane dead. Không launch lại nếu chưa được yêu cầu. [Transfer](../environment/TRANSFER_TO_4080_2026-10-13.md): copy campaign hiện tại, kiểm chứng 4080, resume Task 2 từ **6.414**. Giữ 16 epochs/task, batch 8, TF32 và hashes; budget chưa chứng minh hội tụ. Source notebook/bản execution lưu outputs mỗi 15 giây khi chạy; console L4 `data/training/vast4090-seed0-640-v1/tmux_baseline_l4.log`. Bản đầy đủ tại commit được giữ trong data. LoRA/K=3/freeze bổ sung sau baseline trong run riêng.

## 7. Quy tắc tài liệu và giới hạn xác minh

- Tài liệu repo dùng Markdown link tương đối; bản nguồn từ Obsidian giữ nguyên wikilink. Quy tắc wikilink và `check_vault.py` áp dụng khi sửa vault, không phải khi chỉ sửa repo này.
- Viết tiếng Việt, không thêm emoji trang trí. Source data/archive giữ nguyên; derived data/processed và runs tách riêng. Người dùng đã yêu cầu tổ chức lại notebook/docs, các link được cập nhật và lịch sử chuyển vào archive. Training dùng Vast.ai, không phụ thuộc pipeline GCP cũ.
- Báo cáo tháng 09 là lịch sử của hướng cũ; mọi checkpoint/dataset mới cần provenance và kiểm tra tương thích trước khi dùng.
- CLI AI DevKit lint/memory chưa chạy được qua `npx --offline ai-devkit` do package không có trong npm cache. Việc bàn giao dựa vào source và đọc file trực tiếp; không coi đây là kết quả lint hay memory search thành công.
