# Vận hành pipeline tiền xử lý

Cập nhật 08/10/2026. [Chỉ mục triển khai](../project/IMPLEMENTATION_INDEX.md) chỉ code/config/tests; [báo cáo](reports/README.md) lưu kết quả từng run. Các bước audit/review/resolution chạy CPU; GPU train là RTX 4090 trên Vast.ai.

## Chuẩn bị và đọc kết quả

Mở notebook từ repo, chọn kernel có Python, NumPy và Pillow. Các phần API dùng standard library, không cần OpenAI SDK. Notebook 03 có cells torch/CUDA do người dùng bổ sung: chạy các phần API độc lập nếu kernel CPU không có torch; không dùng Run All làm nghiệm thu GPU. Môi trường model riêng theo Python 3.10 của `pyproject.toml`, cần kiểm tra PyTorch/CUDA/deformable-attention trên Vast.ai trước train.

Root input mặc định `data/archive`, có thể chỉ định `AUTOCHECKOUT_DATA_ROOT` trước setup cell. Root derived/output hiện cố định dưới repo. Copy `.env.example` thành `.env` local và đặt `OPENAI_API_KEY`, hoặc export key qua environment; environment được ưu tiên. Không in key hoặc đưa `.env` vào git. Không cần key khi chỉ đọc receipts đã có.

Kiểm tra trạng thái hiện tại bằng:

1. `data/processed/synthetic_annotation_review_v1/review_decisions.csv`: 193 regenerate đã áp dụng, 9 keep, 20 pending ngoài scope.
2. `data/processed/rpc_synth_resolved_202_v1/resolution_manifest.json`: ảnh/geometry/provenance đã áp dụng; phân biệt ảnh mới với ảnh giữ nguồn.
3. `runs/data_preprocessing/annotation_resolution/f3292ec6-ce43-47e5-a07f-9198d60b1e43/acceptance.json` và `resolution_report.html`: nghiệm thu và trước/sau.
4. `data/processed/synthetic_annotation_review_v1/ai/gpt-6.1-sol_resolution_202_v1/latest_summary.json`: API IDs, chi phí lịch sử và apply run cuối. Dùng cùng manifest/CSV; một lần chạy read-only mới có thể thay summary mà không đổi dataset.

`review_queue.json` và output notebook 03 ghi lịch sử trước khi áp dụng. Các giá trị “222 pending”, “2 deferred”, 78/93/29 là trạng thái cũ. Không sửa/xóa lịch sử để biến thành nghiệm thu mới. Timestamp/outputs đã lưu thuộc lần thực chạy; định dạng lại source không tạo ra lần chạy mới.

## Audit và real-only split

Để đọc bằng chứng đã có, mở output và báo cáo của 01/01b/01c/02. Chỉ chạy lại audit khi cần đối chiếu nguồn; full decode/hash duyệt 57.710 file. Notebook 02 dùng native merge, giữ geometry và nguồn ảnh, khóa protocol 100+4×25 seed0, phát hành release `rpc_100-4x25_seed0_native_v1`.

Khi chạy 02, kiểm tra input/catalog/config và chạy cells theo thứ tự; chỉ công nhận release khi các gate native/counts/coverage/mapping/disjoint/reproducibility/checksum/test lock đạt. Giữ holdout và release v1 đã khóa. Thay catalog/seed/chính sách split phải tạo version/config/output mới, có notebook và báo cáo riêng; không ghi đè v1 để “tối ưu” kết quả test.

`scripts/prepare_data.sh` và defaults `checkout_800` là legacy resize, không dùng thay notebook native. Processor 640/800 sẽ biến đổi ảnh/bbox ở bước model và cần gate round-trip riêng.

## Visual, pilots và Batch đề xuất — notebook 03

Phần 1–5 tạo/đọc queue, card và CSV; chọn case bằng `CASE_KEY`. Nếu duyệt manual, ghi decision/evidence/reviewer cho case đã xem, giữ header CSV và các dòng khác; không đặt lại 222 dòng về pending. Chạy cell nghiệm thu sau cập nhật. CSV manual có thể xung đột reviewer API khi áp dụng lại; pipeline từ chối ghi đè quyết định của người khác.

Phần 6–12 là pilots/content audit. Mặc định các cờ `AUTOCHECKOUT_REVIEW_AI_PILOT`, `AUTOCHECKOUT_REVIEW_AI_REMAINING`, `AUTOCHECKOUT_REVIEW_SOL_PILOT` bằng `0`. Gate nội dung các pilot lịch sử FAIL; không bật continuation Luna dựa trên API/schema PASS. Reference/content score được khóa với output hashes, không tái dùng điểm cho model/prompt mới.

Phần 13–16 là Batch **đúng 200 case**, loại 20 pilot và liệt kê 2 deferred ở thời điểm đó:

| Cờ environment, đọc khi chạy setup cell | Giá trị / tác dụng |
|---|---|
| `AUTOCHECKOUT_REVIEW_BATCH_SUBMIT` | `0` mặc định; `1` upload/create hoặc resume batch theo manifest |
| `AUTOCHECKOUT_REVIEW_BATCH_REFRESH` | `0` mặc định; `1` poll/tải receipt, không tạo batch mới |
| `AUTOCHECKOUT_REVIEW_BATCH_WAIT_SECONDS` | `0` mặc định; số giây chờ trạng thái có giới hạn |

Chạy setup → submit/refresh nếu cần → parse/API acceptance → content report. State/fingerprints giúp resume và tránh gửi trùng; không xóa `submission.json`/intent khi batch đang chạy hoặc khi POST trả lỗi chưa rõ đã tạo hay chưa. Nếu thiếu local state, kiểm tra remote batch metadata trước khi tạo lại; lỗi state không được giải quyết bằng xóa cache rồi submit mù.

Đề xuất Batch không tự sửa labels. Sample content audit là mẫu có chủ đích, không phải accuracy GT và không nghiệm thu 200 ảnh. Scope đã được xử lý tiếp bằng notebook 04.

## API reviewer cuối và áp dụng — notebook 04

Phụ thuộc: visual summary/queue/cards từ 03, accepted results của Batch 200, cutout/background catalog đã nghiệm thu, mapping và real-only lock. Scope là 200 case + `(4,57855)` + `(4,61176)`; giữ 20 pilot ngoài scope.

| Cờ environment, đọc khi chạy setup cell | Giá trị / tác dụng |
|---|---|
| `AUTOCHECKOUT_RESOLUTION_SUBMIT` | `0` mặc định; `1` submit/resume reviewer Batch 202, 4 views, medium |
| `AUTOCHECKOUT_RESOLUTION_REFRESH` | `0` mặc định; `1` poll/download, không tạo mới |
| `AUTOCHECKOUT_RESOLUTION_WAIT_SECONDS` | `0` mặc định; số giây chờ có giới hạn |
| `AUTOCHECKOUT_RESOLUTION_APPLY` | `0` mặc định; `1` materialize/reuse dataset rồi verify và cập nhật CSV phạm vi 202 |

Chạy lại **chỉ đọc/parse receipts**: để cả submit/refresh/apply bằng `0`, chạy setup và các cells API/parse/summary. Notebook vẫn tạo run artifact và cập nhật summary; chưa coi run này là apply acceptance.

Chạy lại **apply bằng cache, không gửi API mới**: giữ submit/refresh `0`, đặt apply `1`, chạy setup → cached state/parse (đủ 202 hợp lệ, không rejected) → apply → acceptance cuối. Dataset v1 chỉ được reuse nếu fingerprint trùng; không tự xóa output khi mismatch. Cell cuối backup CSV, verify geometry/source/task/holdout, rồi cập nhật đúng scope và ghi status `RESOLVED_202_ACCEPTANCE_PASS`.

Fingerprint dataset bao gồm hash **toàn bộ file** `tools/annotation_resolution.py` và assets/settings/decisions. Thay code, kể cả comment/format, có thể làm mismatch. Muốn chạy thuật toán/config mới phải tạo version output/config/AI cache tương ứng và điều chỉnh đường dẫn notebook; không ghi đè dataset đã nghiệm thu. Helper này là compositor có giới hạn, không phải CLI sinh lại toàn bộ 20.000 synth.

9 keep giữ JPEG/geometry nguồn với giải trình AI, không phục hồi original owner masks. 193 regenerate tạo ảnh mới từ cutouts/backgrounds, giữ multiset SKU/instance, không flip; lưu owner maps/recipes và bbox/area/RLE từ visible pixels. Bất định dẫn đến regenerate. Acceptance phải kiểm tra image paths/dimensions/IDs/bounds, owner↔RLE/bbox/area/visibility, counts, task ranges, SHA assets/raw và real-only lock, rồi CSV evidence/reviewer. Không đọc train GT-full thay nhãn task.

## Kiểm tra code và artifacts không gọi API

Từ repo root, chạy tests cơ học review/API/Batch/resolution:

```bash
python3 -m pytest tests/test_annotation_review.py tests/test_ai_annotation_review.py tests/test_ai_annotation_batch.py tests/test_annotation_resolution.py -q -o addopts='' --tb=short
```

Regression dữ liệu mở rộng trong kernel Python 3.14 local bị hạn chế forkserver socket (11 suites, 78 tests đạt ngày 08/10):

```bash
python3 - <<'CHECK'
import multiprocessing, pytest
multiprocessing.set_start_method('fork')
raise SystemExit(pytest.main([
    'tests/test_native_checkout.py', 'tests/test_resize.py', 'tests/test_audit_rpc.py',
    'tests/test_groups.py', 'tests/test_make_split.py', 'tests/test_make_task_config.py',
    'tests/test_taskcfg.py', 'tests/test_annotation_review.py',
    'tests/test_ai_annotation_review.py', 'tests/test_ai_annotation_batch.py',
    'tests/test_annotation_resolution.py', '-q', '-o', 'addopts=', '--tb=short']))
CHECK
```

Lượt này có 4 multiprocessing deprecation warnings. `tests/test_make_task_json.py` cần `pycocotools`, kernel local hiện thiếu; suite này và model/GPU/full suite chạy trong environment phù hợp, không tính vào 78 tests đã đạt.

Kiểm tra dataset hiện có, không submit/apply/cập nhật CSV:

```python
import json
from pathlib import Path
from tools.annotation_resolution import verify_resolution_dataset
repo = Path.cwd().resolve()  # chạy từ repo root
out = repo / "data/processed/rpc_synth_resolved_202_v1"
manifest = json.loads((out / "resolution_manifest.json").read_text())
print(verify_resolution_dataset(manifest, repo=repo,
    raw_root=repo / "data/archive", out_dir=out))
```

Không chạy full notebook 03 trong kernel thiếu torch. Toàn bộ suite/model/GPU cần môi trường model riêng; CPU tests/owner geometry không chứng minh convergence, chất lượng semantic GT hay GRN.

## Tiếp tục sau tiền xử lý

Tạo notebook modeling kiểm chứng processor 640/800, loader/collation, task-label masking/full-GT separation, loss/optimizer/resume và nguồn paths. Baseline dùng real-only release đã khóa; resolved 202 là config augmentation riêng sau gate loader, không đổi val/test. Xử lý 20 pilot còn lại theo scope/version riêng nếu tiếp tục; gate 202 không cho phép đánh dấu toàn bộ synth đã phát hành.

Sau PDP nền mới tích hợp ConvNeXt-V2-Base và pilot EXP-B1/B2 trên Vast.ai cùng split/budget. Ghi environment/GPU/weights/pretrained load report; lấy checkpoints/logs về storage bền vững trước kết thúc instance. Runner không dùng `--shutdown` để kết thúc thuê Vast.ai. LoRA/K=3/FSA/freeze/overlap là ablation riêng sau baseline.

Mỗi lần chạy: mục đích/input/output → run ID và started_at thật, git revision/dirty/config/seed/versions/checksums → execution/gates → finished_at/status/error → báo cáo có ngày và cập nhật chỉ mục/PROGRESS/timeline. Không tái ghi timestamp lịch sử thành thời điểm chạy mới.
