# Bàn giao AutoCheckout-CL

Cập nhật: **10/10/2026**. Notebook 01/02 đã đạt processor/loader và full pretrained ResNet CUDA smoke.
[ConvNeXt notebook 03](CONVNEXT_READINESS.md) đã đạt 8/8 gates, 29 tests trên RTX 3060 (run aeeb770b).
CLI/runner và notebook 04 có triển khai/kiểm chứng riêng tại [báo cáo runtime](TRAINING_RUNTIME_ACCEPTANCE.md).
Các run model local dùng PyTorch fallback. [S6 calibration/evaluation](COUNT_CALIBRATION_ACCEPTANCE.md) đã nghiệm thu kỹ thuật bằng CPU, policy val và test-only reload. Native kernel, calibration checkpoint pilot, pilot và full 5-task training chưa nghiệm thu.

[Snapshot đầu ngày 09/10](HANDOFF_2026-10-09.md) giữ bằng chứng dependency/branch tại lúc viết, trước các run model đã đạt;
đọc cùng kết luận hiện hành, không dùng các đoạn “chưa chạy” trong snapshot làm trạng thái model mới nhất.
[Nhật ký model](../timelines/model-foundation-2026-10-09.md) giữ chi tiết runs PASS/FAIL và các sửa lỗi.
Các làm rõ của người dùng có ưu tiên hơn handoff gốc.

## 1. Phạm vi và thứ tự ưu tiên

- Bài toán: phát hiện và đếm sản phẩm trên khay checkout; mục tiêu RPC gồm 200 SKU.
- **Đã chốt 07/10: 5 task 100+4×25 SKU**, seed mapping 0, phân tầng đủ 17 nhóm hàng mỗi task. Nguồn 4 task `[48,68,47,37]` giữ làm tham khảo/đối chứng. 24 slot dự phòng hiện có không phải task dữ liệu. Ràng buộc replay-free giữ nguyên: loader train chỉ đọc task hiện tại, không bổ sung replay ảnh/GT task cũ; được giữ trọng số, prompt và prototype embedding.
- [Notebook so sánh protocol](../../notebooks/data_preprocessing/01c_protocol_comparison.ipynb) ghi phân tích trước quyết định; output lịch sử được giữ. Người dùng đã chọn 5 task sau lần chạy đó, không phải kết luận protocol này chắc chắn có cAcc cao hơn.
- GPU training mục tiêu là **RTX 4090 24 GB trên Vast.ai**, không phải GPU local. Máy phát triển và instance train là hai môi trường khác nhau; GPU/driver/runtime trên instance chưa được kiểm tra trong phiên này.
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
5. **Môi trường train → E1/E2:** S6 kỹ thuật đã PASS [evaluation 01](../../notebooks/evaluation/01_count_calibration_acceptance.ipynb); configs native có `EVALUATE=1`, tạo policy val lần đầu rồi giữ policy khi rerun. Kiểm chứng kernel hoặc fallback có đo throughput trên RTX 4090 Vast.ai; pilot ResNet/ConvNeXt cùng split/budget và policy val checkpoint thật trước full 5 task. Mọi bước có notebook/timestamps; LoRA/K=3/FSA/freeze bổ sung sau baseline trong run riêng.

## 7. Quy tắc tài liệu và giới hạn xác minh

- Tài liệu repo dùng Markdown link tương đối; bản nguồn từ Obsidian giữ nguyên wikilink. Quy tắc wikilink và `check_vault.py` áp dụng khi sửa vault, không phải khi chỉ sửa repo này.
- Viết tiếng Việt, không thêm emoji trang trí. Source data/archive giữ nguyên; derived data/processed và runs tách riêng. Người dùng đã yêu cầu tổ chức lại notebook/docs, các link được cập nhật và lịch sử chuyển vào archive. Training dùng Vast.ai, không phụ thuộc pipeline GCP cũ.
- Báo cáo tháng 09 là lịch sử của hướng cũ; mọi checkpoint/dataset mới cần provenance và kiểm tra tương thích trước khi dùng.
- CLI AI DevKit lint/memory chưa chạy được qua `npx --offline ai-devkit` do package không có trong npm cache. Việc bàn giao dựa vào source và đọc file trực tiếp; không coi đây là kết quả lint hay memory search thành công.
