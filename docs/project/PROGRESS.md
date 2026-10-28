# Tiến độ AutoCheckout-CL

**27/10/2026 — FULL-SYN vòng 1 xong trên 4090 vast.ai, qua mọi cổng:** [kết quả](full-syn-train-plan-2026-10-22.md#11-kết-quả-vòng-1-2710), notebook [05_full_syn_r1](../../notebooks/training/05_full_syn_r1.ipynb). Trên val: FSA cAcc-M2 77,65% (D18 đạt), task 1 79,38% (D17 🟢), precision nhãn giả ≥ 98,5% ở task 2–5. Trên 6.003 ảnh test cũ sau task 5: cAcc 43,15% (một ngưỡng) / 43,68% (theo nhóm), so với B2 21,74%; AP50 94,94% so với 84,40%; quên −0,49 điểm. Ở stage 1–4, cAcc-M1 trên test thấp hơn B2 (28,65% so với 73,16% ở stage 1): ảnh ghép không có SKU tương lai, nên model đếm thừa trên các vật đó. Mới một seed; nguyên nhân hơn B2 chưa tách riêng. Khoảng 34,5 giờ wall. Đối chiếu 28/10 với artifact trong `data/training/training.zip`: khớp.

**26/10/2026 — code gói FULL-SYN xong:** F0.1–F0.6, F0.8, `autocheckout/full_syn.py` và hai notebook ([04 smoke](../../notebooks/training/04_full_syn_smoke.ipynb), [05 vòng 1](../../notebooks/training/05_full_syn_r1.ipynb)); dry run của cả hai notebook đạt. Trên B2, F0.5 và F0.8 tái lập đúng số cũ. Còn F0.7 (smoke trên 4090) rồi G1. Cách chạy: [runbook mục 7](TRAINING_NOTEBOOK_RUNBOOK.md#7-full-syn-trên-máy-vastai-26102026).

**25/10/2026 — hình thức chạy gói FULL-SYN:** trên nhánh này viết code F0.1–F0.8, tạo notebook training mới, cập nhật tài liệu và đóng gói dữ liệu lên Drive; người dùng tự setup máy 4090 vast.ai rồi chạy notebook. Notebook là log chạy commit vào git, song song với log của code dưới `data/training/` ([plan mục 4.1](full-syn-train-plan-2026-10-22.md#41-hình-thức-triển-khai-người-dùng-chốt-2510)).

**22/10/2026 — plan gói train FULL-SYN:** [plan](full-syn-train-plan-2026-10-22.md). Chốt: FSA tối đa 8 epoch chọn theo val, không dự đoán test trong vòng cổng, train trên 4090 vast.ai như B2. Bước tiếp theo: F0.1–F0.6 (code, kiểm chứng export FSA ConvNeXt) trước khi thuê GPU. Ước lượng khoảng 52–58 giờ GPU nếu qua cổng ngay vòng 1. Bản A cũ (23 GB) đã xoá, giữ biên bản ở `runs/data_preprocessing/synthetic_v2/layoutA_record/`.

**22/10/2026 — ảnh ghép v2 sinh lại theo bố cục o15:** mỗi vật còn thấy ≥ 90%, 15% cảnh còn thấy ≥ 75%, vì bản A che gấp khoảng 4 lần ảnh thật (đo bằng mask SAM). `rpc_synth_v2_v1` 71.642 ảnh, 0 cảnh lỗi, verify 14/14; [báo cáo](../data_preprocessing/reports/synth-v2-2026-10-21.md) đã cập nhật.

**21/10/2026 — ảnh ghép v2 đã phát hành:** [báo cáo](../data_preprocessing/reports/synth-v2-2026-10-21.md). `rpc_synth_v2_v1` 71.642 ảnh, verify 14/14; bộ đánh giá `rpc_eval_official_v1` (val 5.792, test 24.000); loader PDP đọc được ở 640. cAcc-M2 có trong `eval_count --unlearned-iou 0.5`; mốc B2 task 1 = 85,79%, đi tiếp khi FULL-SYN ≥ 72,8%. Chưa train FULL-SYN; bước tiếp theo là plan gói train.

**21/10/2026 — chốt quyết định ảnh ghép v2:** [quyết định D1–D21](synthetic-data-v2-decisions-2026-10-21.md); spec/plan đã sửa theo.
- Train 100% ảnh ghép, 71.642 ảnh, 16 epoch/task.
- Val = val2019 bỏ 208 ảnh nguồn nền (5.792 ảnh); test = test2019 đầy đủ.
- Chỉ chạy FULL-SYN; FULL-REAL hoãn.
- Chọn camera theo hình dạng vật, cắt bổ sung camera 0 cho khoảng 51 SKU dạng trụ (cần token Kaggle).
- Cổng sau task 1 dùng cAcc-M2, ngưỡng Δ −13/−28 so với B2.

Đây là quyết định và tài liệu; chưa có code, dữ liệu hay train v2.

**21/10/2026 — quản lý dữ liệu:** synthetic cũ được đặt tên **v1**, người dùng xác nhận backup Drive và cho phép xóa `data/archive/synth/group/` local; đã xóa 20.000 JPEG + 4 JSON sau khi lưu inventory SHA-256. [Biên bản](../data_preprocessing/reports/synth-v1-retirement-2026-10-21.md). Cutout/nền/real release giữ nguyên; resolved 202 còn 193 ảnh mới nhưng 9 ảnh nguồn cần restore. Synthetic **v2** chưa sinh; không đổi trạng thái nghiệm thu lịch sử thành kết quả mới.

**Kế hoạch nhánh `feat/synthetic-data-v2`, 20/10/2026:** đã đọc quy tắc tài liệu, pipeline dữ liệu/loader và index cutout; lưu [thiết kế](synthetic-data-v2-design-2026-10-20.md), [kế hoạch](synthetic-data-v2-plan-2026-10-20.md) và [nguồn cuộc họp](../references/meeting-2026-10-20/README.md). Có metadata camera/source cho đủ 7.502 cutout; chưa review semantic hoặc tạo v2. Mốc kế tiếp đề xuất: inventory/gallery nguyên liệu, nhóm chốt trước pilot. Đây là kế hoạch, không phải nghiệm thu generator hay một run train mới.

**Cập nhật 20/10/2026: baseline EXP-B1/EXP-B2 hoàn tất 5 task ngày 15/10, 19:28:27 UTC+7.** [Báo cáo kết quả](FINAL_BACKBONE_COMPARISON_2026-10-20.md) thay trạng thái dừng/chuyển máy ngày 13/10. Pilot/transition, foundation/ConvNeXt adapter, CLI/runner và S6 đã có bằng chứng riêng.

Final test B1/B2: AP50 **62,33%/84,40%**, AP[.50:.95] **51,55%/71,37%**, cAcc **2,58%/21,74%**; forgetting **6,63/8,67 điểm**. Đã tính lại counting 10 stage, calibration validation Task 5 và AP cuối; hashes/checkpoints khớp. 63 tests evaluation PASS. Hoàn tất budget 16 epochs/task chưa chứng minh hội tụ; LoRA/K=3/FSA/freeze bổ sung vẫn là ablation chưa chạy.

[Bàn giao](AGENT_HANDOFF.md) · [Kế hoạch](IMPLEMENTATION_PLAN.md) · [Chỉ mục triển khai](IMPLEMENTATION_INDEX.md) · [Snapshot training đến 13/10](../timelines/training-handoff-through-2026-10-13.md).

| Phần | Trạng thái / bằng chứng | Việc tiếp theo |
|---|---|---|
| Tổ chức repo | Tài liệu vận hành, báo cáo có ngày, nguồn tham khảo và lịch sử tách riêng; notebook theo giai đoạn | Cập nhật chỉ mục cùng mỗi thay đổi |
| Khôi phục mã nền | 189 file khôi phục từ HEAD sau reset; code/runner/tests và kết quả cũ được giữ | Tái sử dụng code đã kiểm chứng; bổ sung tests theo thay đổi mới |
| Audit tài nguyên | 57.710 file decode/hash; 208 nền, 7.502 cutout/200 SKU | Tái sử dụng catalog đã kiểm tra |
| Real-only split | 22.494 train/1.503 val/6.003 test; native geometry; 9 gate đạt | Giữ test lock; processor/loader đã đạt trên phạm vi notebook |
| Visual và AI review | 222 card/gallery, Luna 10 + Sol 10 + Batch 200; gate API đạt, gate nội dung lịch sử FAIL | Đọc reports như lịch sử đề xuất |
| Resolution 202 | 202 ảnh/2.846 object/200 SKU; 9 giữ/193 tái ghép; owner-mask/RLE cho 2.721 object, task JSON và CSV acceptance PASS | Bản augmentation riêng; chưa tích hợp training |
| Scope còn lại | 20 pilot còn pending; toàn bộ 20.000 synth chưa phát hành lại | Xác định scope xử lý bổ sung riêng |
| Dependency/Conda | [Kiểm tra 20/10](FINAL_BACKBONE_COMPARISON_2026-10-20.md): Conda pdp Python 3.10.22, Torch cu121, 64 pins, pip check/imports và 63 tests evaluation PASS | Driver local chưa hoạt động; metrics dùng CPU, GPU acceptance cũ không thay kiểm tra máy hiện tại |
| Handoff 09/10 | [Snapshot trước kiểm chứng model](HANDOFF_2026-10-09.md) giữ làm lịch sử; [kết luận hiện hành](PDP_FOUNDATION_ACCEPTANCE.md) | Dùng kết luận mới khi tiếp quản trạng thái model |
| PDP nền/processor | [Notebook 01](../../notebooks/modeling/01_processor_loader_pdp_acceptance.ipynb) đạt PASS_LIMITED_PDP_FOUNDATION, 11 gate, CPU tiny smoke; regression 27/27, resume 1/1; [bằng chứng](../timelines/model-foundation-2026-10-09.md) | Giữ evidence foundation; native kernel máy training đã có nghiệm thu riêng |
| Full PDP baseline CUDA | [Notebook 02](../../notebooks/modeling/02_full_pdp_baseline_acceptance.ipynb), run d967dc7b: PASS_FULL_PDP_CUDA_SMOKE trên RTX 3060, 640/800 batch 1 FP32; resume/state/predictions/frozen hashes đạt; peak allocated 3,49/4,22 GiB | Native kernel chưa đạt (PyTorch fallback); ConvNeXt và CLI/runner đã có nghiệm thu riêng; không chứng nhận convergence |
| Pretrained public | Script tải + notebook tự nhận cache path đã có; 3 files SenseTime detector tải thật, ~161 MB, pinned revision/SHA256, offline reuse đạt | Tái sử dụng cache/revision đã khóa; không cần tải lại để bắt đầu adapter |
| ConvNeXt-V2-Base | [Notebook 03](../../notebooks/modeling/03_convnext_v2_base_acceptance.ipynb) PASS 8/8 gates, 29 tests; run aeeb770b trên 3060, 640/800 peak 3,683/4,416 GiB | Adapter đã được dùng trong full baseline; xem báo cáo cuối |
| CLI/runner native | [Notebook 04](../../notebooks/modeling/04_training_runtime_acceptance.ipynb) PASS 10/10 gates, 78 tests; ResNet/ConvNeXt 640/800, step 1→4, checkpoint metadata/resume/predict-only/bbox native | Runtime đã dùng cho pilot/transition/full baseline; giữ nghiệm thu trước thay đổi mới |
| S6 calibration/evaluation | Policy val-only/test-only reload đã dùng cho full baseline; final thresholds B1/B2 0,42/0,35, NMS 0,45; tính lại khớp | Giữ oracle tắt; thay đổi policy phải chọn trên validation trong run riêng |
| Training notebooks 01–03 | Pilot/transition và baseline **cả hai backbone đủ 5 task**; campaign COMPLETED 15/10 | Giữ artifacts và outputs lịch sử; notebook báo cáo evaluation 02 tổng hợp kết quả |
| GPU training/eval | B2 4090→L4→4080 SUPER; B1 4080 SUPER. [Kết quả cuối](FINAL_BACKBONE_COMPARISON_2026-10-20.md) đã đối chiếu checkpoints/predictions/policies | Phân tích validation, hội tụ, ablation và thêm seeds; không còn task baseline chờ train |
| LoRA/K=3/FSA/freeze/overlap | Chưa triển khai các ablation mới | Sau baseline, mỗi thay đổi một run |

Báo cáo hiện hành cho scope đã xử lý: [Resolution 202](../data_preprocessing/reports/annotation-resolution-202-2026-10-08.md). CSV có **193 regenerate đã áp dụng, 9 giữ, 20 pending**; không dùng hàng đợi JSON ban đầu hoặc output cũ để suy ra pending hiện tại. Bản ảnh thật và holdout không đổi.

Kiểm chứng 08/10 cho lần tổ chức/commit: **78 CPU tests đạt** trên 11 suites native/resize/audit/group/split/task-config/taskcfg/visual/API/Batch/resolution. Command dùng `pytest.main` và multiprocessing `fork` trong kernel Python 3.14 local (4 deprecation warnings); xem các suites trong [runbook](../data_preprocessing/OPERATIONS.md). `tests/test_make_task_json.py` không collect được do thiếu `pycocotools`, tách khỏi nhóm regression đã đạt; cần chạy trong environment model đã chuẩn bị. Đó là snapshot 08/10; ngày 09/10 đã đạt limited foundation và full CUDA Task 1 smoke, full suite vẫn chưa xác nhận. Tests API/schema không phải accuracy GT.

Snapshot kiểm tra tổ chức tài liệu 08–09/10: **284 liên kết nội bộ hợp lệ**, **49 cell code tiền xử lý compile**, **8 config JSON hợp lệ**, `git diff --check` đạt. Kiểm tra read-only dataset thực tế đạt 202 ảnh/2.846 object/2.721 regenerated mask objects và CSV 193/9/20; không gọi API mới hoặc chạy training. Preview đầy đủ giữ local ở `runs/notebook_versions/`; bản notebook trong git giữ output nghiệm thu chữ/bảng.


## Bước tiếp theo

1. Dùng báo cáo cuối và notebook evaluation 02 làm bảng kết quả baseline; giữ snapshot/tham số/hash để tái lập.
2. Phân tích trên validation: thiếu đếm lớp mới, lỗi ảnh hard, confidence theo nhóm task và tác động top-1/NMS. Không chọn threshold bằng test.
3. Đối chiếu learning curves để đánh giá hội tụ; 16 epochs/task hiện là budget đã chạy, không phải kết luận tối ưu.
4. Thêm seeds và ablation độc lập nếu tiếp tục nghiên cứu; giữ baseline gốc. LoRA/K=3/FSA/freeze/synthetic là các hướng riêng, chưa được triển khai trong baseline này.
5. Khôi phục driver/kiểm chứng CUDA local trước công việc cần GPU. Evaluation từ predictions có thể chạy CPU trong Conda pdp.
