# Chỉ mục triển khai và đầu ra

Cập nhật 08/10/2026. Bảng này ánh xạ các phần **đã triển khai** đến code, notebook, cấu hình và bằng chứng. [Vận hành](../data_preprocessing/OPERATIONS.md) mô tả cách chạy lại; [kế hoạch](IMPLEMENTATION_PLAN.md) dành cho phần chưa làm.

| Phần | Code / giao diện | Notebook và config | Tests / bằng chứng |
|---|---|---|---|
| Audit tài nguyên nhập | [audit_rpc.py](../../tools/audit_rpc.py); inventory, decode/hash/alpha và đối chiếu synth trong notebook | [01](../../notebooks/data_preprocessing/01_data_audit.ipynb), [01b](../../notebooks/data_preprocessing/01b_preprocessing_acceptance.ipynb), [01c](../../notebooks/data_preprocessing/01c_protocol_comparison.ipynb) | [test_audit_rpc.py](../../tests/test_audit_rpc.py); [nghiệm thu tài nguyên](../data_preprocessing/reports/preprocessing-acceptance-2026-10-07.md) |
| Split native và 5 task | [resize.py](../../tools/resize.py): `merge_annotations(size=None)` giữ bbox/area/source paths; [make_split.py](../../tools/make_split.py), [make_task_config.py](../../tools/make_task_config.py), [make_task_json.py](../../tools/make_task_json.py), [TaskConfig](../../autocheckout/taskcfg.py) là nền tái sử dụng | [02](../../notebooks/data_preprocessing/02_split_and_acceptance.ipynb); [release config](../../configs/data/rpc_100-4x25_seed0_native_v1.json), [mapping](../../configs/tasks_100-4x25_seed0.json) | [test_native_checkout.py](../../tests/test_native_checkout.py), resize/split/group/task-config tests; [split/test lock](../data_preprocessing/reports/split-100-4x25-2026-10-07.md) |
| Visual review | [annotation_review.py](../../tools/annotation_review.py): crop bao phủ bbox và polygon, full/crop/overlay card | [03 phần 1–5](../../notebooks/data_preprocessing/03_annotation_review.ipynb); `CASE_KEY=(source_task,annotation_id)` | [test_annotation_review.py](../../tests/test_annotation_review.py); [222 card/gallery](../data_preprocessing/reports/annotation-review-2026-10-07.md) |
| Responses API và pilot | [ai_annotation_review.py](../../tools/ai_annotation_review.py): key loader, prompt/schema/parser, receipt cache/fingerprint, usage/cost, content gate | Notebook 03 phần 6–12; [Luna config](../../configs/data/annotation_review_ai_v1.json), [Sol pilot](../../configs/data/annotation_review_sol_pilot_v1.json) và các `*_content_reference_v1.json` | [test_ai_annotation_review.py](../../tests/test_ai_annotation_review.py); [Luna](../data_preprocessing/reports/ai-review-pilot-2026-10-07.md), [content audit](../data_preprocessing/reports/content-test-2026-10-07.md), [Sol](../data_preprocessing/reports/sol-pilot-2026-10-07.md) |
| Batch submit/resume/parse | [ai_annotation_batch.py](../../tools/ai_annotation_batch.py): `prepare_bundle`, `submit_bundle`, `poll_bundle`, `parse_outputs`; JSONL sharding, persisted intent/state, terminal receipts, ID/model/schema/hash checks | Notebook 03 phần 13–16; [Batch config](../../configs/data/annotation_review_sol_batch_v1.json), [sample reference](../../configs/data/annotation_review_sol_batch_sample_reference_v1.json) | [test_ai_annotation_batch.py](../../tests/test_ai_annotation_batch.py); [Batch 200](../data_preprocessing/reports/sol-batch-2026-10-08.md) |
| Reviewer cuối và xử lý thật | [annotation_resolution.py](../../tools/annotation_resolution.py): `build_adjudication_request`/`parse_adjudication`/`final_action`; `compose_replacement`, `apply_resolution_dataset`, `verify_resolution_dataset`, `update_review_decisions` | [04](../../notebooks/data_preprocessing/04_annotation_resolution.ipynb); [resolution config](../../configs/data/annotation_resolution_202_v1.json); dùng Batch helper với request/parser riêng | [test_annotation_resolution.py](../../tests/test_annotation_resolution.py); [Resolution 202](../data_preprocessing/reports/annotation-resolution-202-2026-10-08.md): 9 giữ/193 tái ghép, owner-mask/RLE/task/CSV PASS |

Các module review/API/resolution là thư viện được notebook gọi, **chưa có CLI end-to-end riêng**. Audit/split/task tools có CLI kế thừa, nhưng pipeline native đã nghiệm thu đi qua notebook 02; `scripts/prepare_data.sh` là đường legacy resize 800, không phải lệnh phát hành native v1.

## Artifacts và dữ liệu hiện hành

| Đường dẫn local, không đưa vào git | Nội dung / vai trò |
|---|---|
| `data/archive/` | Nguồn ảnh/COCO/cutout/background giữ nguyên |
| `data/processed/rpc_100-4x25_seed0_native_v1/` | Real-only release, splits/task JSON, mapping và checksum/test lock |
| `data/processed/synthetic_annotation_review_v1/review_queue.json` | Queue gốc 222 case, lịch sử detector nghi vấn |
| `data/processed/synthetic_annotation_review_v1/review_decisions.csv` | Trạng thái hiện tại: 193 regenerate đã áp dụng, 9 giữ, 20 pending |
| `data/processed/synthetic_annotation_review_v1/ai/` | Request manifest/JSONL, submission state, output/error receipts, accepted/rejected, usage/cost |
| `data/processed/rpc_synth_resolved_202_v1/` | annotations/resolution manifest; 193 JPEG/NPZ/recipe mới; task_config và 10 task JSON; 9 ảnh giữ trỏ nguồn |
| `runs/data_preprocessing/<workflow>/<run_id>/` | Run metadata, gates, HTML QA và bằng chứng kiểm tra |
| `runs/notebook_versions/` | Snapshot notebook từng lần chạy, giữ timestamp/output lịch sử |

Real-only ảnh dùng root `data/archive`; resolved synth COCO dùng root repo. RPC IDs ở annotations nguồn/canonical, model labels ở task JSON. Loader phải chọn đúng root/schema, không áp mapping hai lần. Xem [formats](../data_preprocessing/formats.md).

## Mã nền và việc tương lai

PDP/teacher/PPG/checkpoint đã có trong [pdp/](../../pdp/); mapping/metrics/counting/provenance ở [autocheckout/](../../autocheckout/); evaluation ở [tools/eval_cl.py](../../tools/eval_cl.py), [tools/eval_count.py](../../tools/eval_count.py); runner ở [scripts/run_exp.sh](../../scripts/run_exp.sh). Kết quả cũ [results/](../../results/) là lịch sử.

[Notebook 01 kiểm chứng processor/loader/PDP foundation](../../notebooks/modeling/01_processor_loader_pdp_acceptance.ipynb) đã tạo, chưa chạy; [hướng dẫn](../../notebooks/modeling/README.md). Chưa có nghiệm thu full model/pretrained, adapter ConvNeXt-V2-Base, run train/eval mới trên Vast.ai, LoRA/K=3 hay generator đầy đủ đạt overlap 20–45%. Compositor resolution hiện có phục vụ scope 202; không đánh dấu toàn bộ S7 đã hoàn tất. Thứ tự tiếp theo: processor/loader/PDP nền → adapter ConvNeXt → pilot Vast.ai cùng split → 5 task và evaluation → ablation riêng.

Khi sinh code/docs mới: thuật toán dùng lại đặt vào module, config có version, notebook giải thích và ghi run thật, báo cáo có ngày trong `reports/`, tests kiểm tra hành vi, rồi cập nhật bảng này + [PROGRESS](PROGRESS.md) + [timeline](../timelines/README.md). Commit theo chức năng sau nghiệm thu, giữ key/data/receipts lớn ngoài git.
