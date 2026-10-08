# Định dạng dữ liệu và thư mục

> Phạm vi 07/10/2026: protocol **5 task 100+4×25 đã chốt**. [Notebook split](../../notebooks/data_preprocessing/02_split_and_acceptance.ipynb) đã tạo release native v1 real-only và khóa validation/test. Các ví dụ `/data/rpc`/`checkout_800` phía dưới thuộc pipeline kế thừa; dùng hợp đồng native ở mục 0 cho run hiện tại. Xem [báo cáo split](reports/split-100-4x25-2026-10-07.md), [bàn giao](../project/AGENT_HANDOFF.md) và [kế hoạch](../project/IMPLEMENTATION_PLAN.md).

Tài liệu này là "hợp đồng" giữa các phần code: công cụ dữ liệu (DL1–DL6), code PDP (`pdp/`), công cụ đánh giá (V1–V6) và baseline. Muốn đổi định dạng nào thì sửa ở đây trước, rồi sửa code và test tương ứng.

Quy ước chung:

- **Nhãn model** (`label`) là số nguyên 0..`num_slots-1`, xếp liền nhau theo thứ tự task. **`category_id` RPC** là 1..200. Hai loại số này không bao giờ được dùng lẫn cho nhau; muốn đổi thì dùng `TaskConfig` (`autocheckout/taskcfg.py`).
- Tọa độ COCO theo **width/height của ảnh được annotation**: release hiện tại giữ pixel gốc, pipeline legacy dùng 800×800. Processor có thể resize 640/800 nhưng dự đoán phải scale lại về cùng hệ tọa độ GT khi đánh giá. COCO bbox là `[x,y,w,h]`; prediction là `[x1,y1,x2,y2]`.
- Mọi file JSON/npz đều ghi kiểu nguyên tử (ghi file tạm rồi đổi tên), dùng `autocheckout.io.save_json` và `autocheckout.predictions.save_predictions`.

## 0. Release hiện hành: native pixels, 5 task, real-only

```text
data/archive/                         source giữ nguyên, image root của loader
data/processed/rpc_100-4x25_seed0_native_v1/
├── ann/checkout_native.json           30.000 ảnh/367.935 object, RPC IDs 1..200
├── task_config.json                   snapshot 5 task + slot dự phòng
├── splits/{train,val,test,train_pilot}.json
├── split_config/rpc_checkout_seed0.json
├── tasks/real/                        model labels 0..199, native coordinates
└── release_manifest.json              checks, input hashes, test lock
```

`images.file_name` là `val2019/<name>` hoặc `test2019/<name>`; `width/height` giữ nguyên, ID mới không đụng nhau, `source/orig_id/orig_file_name/level` lưu nguồn. Annotation merged có `source/orig_id` để đối chiếu và bbox/area không bị scale/làm tròn. Task JSON remap category_id sang model label, các trường geometry/source giữ nguyên. Chỉ labels của task hiện tại có trong train; file `_gt_full` không được đưa vào loader training.

Split mới: train 22.494, val 1.503, test 6.003, pilot 3.002; mọi group chứa val2019 ở train. Phần test2019 không vào holdout được dùng train, nên đây là split nghiên cứu riêng, không phải official test2019 nguyên vẹn. Config lock nằm ở `configs/data/rpc_100-4x25_seed0_native_v1.json`; data/tasks JSON sinh mới nằm trong gitignore. Synth chưa được đưa vào release này.

## 1. Thư mục pipeline legacy trên VM (`$DATA=/data/rpc`)

```
/data/rpc/
├── raw/retail_product_checkout/        # T0.5: giải nén từ Kaggle, không sửa
│   ├── val2019/*.jpg, test2019/*.jpg   # ảnh quầy vuông, cạnh khoảng 1750–1890 px
│   └── instances_val2019.json, instances_test2019.json
├── checkout_800/*.jpg                  # DL2: toàn bộ 30.000 ảnh quầy, 800×800, một thư mục phẳng
├── ann/checkout_800.json               # DL2: annotation gộp val2019 + test2019 (mục 2)
├── splits/{train,val,test,train_pilot}.json   # DL3: tập con của ann/checkout_800.json, nhãn RPC đầy đủ
└── tasks/<tên>/                        # DL5/DL6: file JSON cho từng task (mục 4)
```

Trong repo (được commit): `configs/tasks_<tên>.json` (DL4, mục 3) và `configs/splits/<tên>.json` (DL3: các seed đã thử, số ảnh theo tập và mức, md5 của từng file split, danh sách ảnh của val/test/train_pilot; train là phần còn lại). Mỗi ảnh trong danh sách ghi dạng `<source>/<orig_file_name>` (ví dụ `test2019/20180827-13-42-20-204.jpg`), vì cùng một tên file có thể có ở cả val2019 lẫn test2019. Bản sao `manifest.json` của DL5 (mục 4, có md5 của `test_full.json`) cũng được commit vào `configs/splits/` để khóa tập test.

## 2. Annotation gộp `ann/checkout_800.json` (DL2)

Định dạng COCO. Các trường thêm so với COCO chuẩn:

| Mục | Trường | Ý nghĩa |
|---|---|---|
| `images[]` | `id` | Đánh số lại 1..30000 (không dùng `id` gốc vì val2019 và test2019 có thể trùng) |
| | `file_name` | Tên file trong `checkout_800/` (giữ tên gốc; nếu hai tập trùng tên thì thêm tiền tố nguồn) |
| | `width`, `height` | Kích thước sau khi thu nhỏ (800, 800) |
| | `source` | `"val2019"` hoặc `"test2019"` |
| | `orig_id`, `orig_file_name`, `orig_width`, `orig_height` | Thông tin gốc để truy ngược |
| | `level` | `"easy"`, `"medium"` hoặc `"hard"` (lấy từ file gốc; DL1 xác nhận trường này có) |
| `annotations[]` | `category_id` | `category_id` RPC 1..200 |
| | `bbox`, `area` | Đã co theo tỉ lệ ảnh; `area` = diện tích gốc × tỉ lệ² |
| | `iscrowd` | 0 |
| `categories[]` | `id`, `name`, `supercategory` | Giữ nguyên của RPC |

Các file split (DL3) giữ nguyên cấu trúc trên và thêm `images[].group` (khóa nhóm ảnh chụp liên tiếp, mục 4.2 của plan).

## 3. File cấu hình task `configs/tasks_<tên>.json` (DL4)

```json
{
  "name": "100-4x25_seed0",
  "seed": 0,
  "num_slots": 224,
  "num_classes": 225,
  "tasks": [
    {"task_id": 1, "offset": 0, "reserved": false, "classes": [
      {"label": 0, "name": "<tên RPC>", "rpc_category_id": 57, "supercategory": "puffed_food"}
    ]},
    {"task_id": 6, "offset": 200, "reserved": true, "classes": [
      {"label": 200, "name": "reserved_200", "rpc_category_id": null, "supercategory": null}
    ]}
  ]
}
```

- `label` = `offset` + vị trí trong danh sách `classes`; các task nối tiếp nhau, không hở.
- Task dự phòng (`reserved: true`) đứng cuối và không có `rpc_category_id`.
- `num_slots`, `num_classes` chỉ để đọc cho tiện; `TaskConfig` tự tính lại.
- Nhãn học xong sau task t là `range(seen_classes(t))`.

## 4. File JSON cho từng task (DL5, thư mục `tasks/<tên>/`)

Định dạng COCO mà code PDP đọc được (`pdp/datasets/coco_hug.py`). **`category_id` là nhãn model** (0-based, giống code gốc). `categories[]` chỉ liệt kê các lớp mà file đó có nhãn, mỗi mục có thêm `rpc_category_id`. Mỗi annotation phải có `area` và `iscrowd`. `images[]` giữ nguyên `id` và mọi trường của file split (`level`, `source`, `group`, `orig_file_name`, ...) để đánh giá tách được theo mức độ đông và so khớp được dự đoán giữa các file.

| File | Ảnh | Nhãn giữ lại |
|---|---|---|
| `train_task_<t>.json` | Ảnh train có ít nhất 1 vật thuộc task t | Chỉ lớp của task t |
| `train_task_<t>_capped.json` | Tối đa N ảnh của file trên, lấy mẫu cố định, phân tầng theo `level` | Chỉ lớp của task t |
| `train_task_<t>_gt_full.json` | Giống `train_task_<t>.json` | **Mọi** lớp (chỉ dùng cho V4, không bao giờ đưa vào train) |
| `val_task_<t>.json` | Ảnh val có vật thuộc task t | Chỉ lớp của task t |
| `val_full.json`, `test_full.json` | Toàn bộ val / test | Mọi lớp có dữ liệu (nhãn 0..199) |

Mỗi thư mục có thêm `manifest.json`: tên task config, nguồn ảnh train, số ảnh và số vật mỗi file, md5 mỗi file.

Ảnh train có thể đến từ nhiều nguồn (`--train-source TÊN=ĐƯỜNG_DẪN`, ví dụ `real` bây giờ, ảnh ghép sau này); mỗi ảnh train có thêm trường `train_source`. Khi thêm nguồn mới (giai đoạn 2), `id` của ảnh và annotation phải không trùng với ảnh RPC, và mỗi ảnh phải có `level` vì việc lấy mẫu `_capped` phân tầng theo trường này.

## 5. File dự đoán `pred_<split>.npz` (V1, B3)

Đọc/ghi bằng `autocheckout.predictions`. Mỗi dòng là một cặp (ảnh, query, lớp):

| Mảng | Kiểu | Ý nghĩa |
|---|---|---|
| `image_id` | int64 | `id` ảnh trong file annotation được đánh giá |
| `query` | int32 | Chỉ số query của detector (hoặc chỉ số hộp với baseline truy xuất) |
| `label` | int32 | Nhãn model |
| `score` | float32 | Xác suất (sigmoid) |
| `boxes` | float32 (N, 4) | `[x1, y1, x2, y2]`, pixel của ảnh 800×800 |
| `meta` | JSON | Ít nhất: `task_id`, `seen_classes`, `split`, `ann_file`, `ann_md5`, `producer`, `topk_per_image` |

- Detector: mỗi ảnh giữ top-100 cặp (query, lớp) trong các lớp đã học (`label < seen_classes`), giống hậu xử lý của Deformable DETR, để mAP so được với paper.
- Đếm sản phẩm (V3) và demo dùng `Predictions.top1_per_query()`: mỗi query một nhãn, rồi lọc theo ngưỡng.

## 6. Thư mục một lần chạy (R2/R3, `$RUNS=/data/runs`)

```
/data/runs/<thí nghiệm>/
├── config.sh                     # bản sao file cấu hình thí nghiệm (configs/exp/<tên>.sh)
├── metrics_cl_<split>.json       # V2: mAP theo task, ma trận, độ quên (các khóa mô tả trong autocheckout/cl_metrics.py và tools/eval_cl.py)
├── metrics_count_test.json       # V3: ngưỡng chọn trên val, chỉ số trên test, ngưỡng "oracle" (tools/eval_count.py)
└── task_<t>/
    ├── task_final.pth            # F8: trọng số + prototype, không có optimizer
    ├── last.ckpt                 # R1: checkpoint resume (xóa khi task xong)
    ├── pred_val.npz, pred_test.npz
    ├── run_info.json             # R3: tham số, git hash + diff, phiên bản thư viện, md5 dữ liệu, thời gian, bộ nhớ GPU
    ├── train.log, lightning_logs/
    └── (last.ckpt, last.ckpt.prev khi task đang dở; tự xóa khi task xong)

Task được coi là xong khi có `task_final.pth` và cả hai file dự đoán; `scripts/run_exp.sh` dựa vào đó để bỏ qua task hoặc chỉ ghi lại dự đoán.
```
