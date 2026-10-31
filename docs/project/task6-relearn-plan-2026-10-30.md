# Mở rộng model và học lại 50 SKU nhận dạng kém ở task 6–7 — kế hoạch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lấy model round 1 sau task 5 và mở rộng thêm slot mà không train lại từ đầu. Sau đó train hai task mới, task 6 và task 7, mỗi task 25 SKU nhận dạng kém, trên dữ liệu ảnh ghép làm lại sao cho đủ mặt và cân đối. Cuối cùng chấm model như một model 200 SKU, so với round 1.

**Architecture:** `task_config` mới giữ nguyên task 1–5 và thêm hai task gồm các *bí danh*: mỗi bí danh là một label mới (200–249) trỏ về label gốc của SKU. Task 8 là 24 slot dự phòng. Checkpoint task 5 được mở rộng ngoại tuyến: chép nguyên các hàng đã học, còn hàng mới khởi tạo như lúc dựng model. Sau đó PDP train task 6 rồi task 7 như mọi task khác. Khi chấm, dự đoán ở label bí danh được đổi về label gốc, nên model sau task 7 được chấm đúng như một model "stage 5" của round 1.

**Tech Stack:** Python 3.10, PyTorch 2.2.2, PDP (Deformable DETR + ConvNeXt-V2), pycocotools, SAM 2.1 (ultralytics, env `sam`), generator `tools/synth_v2.py`.

**Spec:** mục "Quyết định" ngay dưới đây, chốt với người dùng ngày 30/10. Yêu cầu sinh ảnh lấy từ tin nhắn của Minh cùng ngày. Số liệu nền nằm ở [phân tích round 1](full-syn-r1-analysis-2026-10-29.md); mục 9 của tài liệu đó được viết ở Task 11.

## Quyết định (spec)

| # | Quyết định | Giá trị |
|---|---|---|
| Q1 | SKU học lại | **50 SKU**: 32 SKU có recall hoặc precision < 85% trên val sau task 5 (policy `calibration_count_m2.json` stage 5); 16 SKU hay bị nhầm với chúng (cùng nhóm hàng với ≥ 10 lần nhầm cộng hai chiều, hoặc khác nhóm với ≥ 30 lần); thêm 123_chocolate và 130_chocolate, vì cutout cũ thiếu hẳn mặt có logo (phân tích d7) |
| Q2 | Chia task | Hai task, mỗi task 25 SKU. Các SKU nối với nhau bằng quan hệ "hay nhầm" ở Q1 thành một nhóm (20 nhóm, lớn nhất 8 SKU), và mỗi nhóm nằm trọn trong một task. Xếp nhóm theo cỡ giảm dần (bằng nhau thì theo label nhỏ nhất), rồi đưa từng nhóm vào task đang ít SKU hơn (bằng nhau thì task 6). Kết quả ở Task 1 |
| Q3 | Config | Task 1–5 y hệt round 1. Task 6: 25 bí danh, label 200–224. Task 7: 25 bí danh, label 225–249. Trong mỗi task, bí danh xếp theo label gốc tăng dần, tên `<tên gốc>@t<task>`, cùng RPC id và nhóm hàng, `alias_of` = label gốc. Task 8 dự phòng 24 slot (250–273). `n_classes` = 275 |
| Q4 | Mở rộng checkpoint | Chép nguyên hàng 0–199 và hàng cuối (index `n_classes − 1`). Prompt riêng của hàng mới = 0, giống slot chưa học hiện nay. Hàng mới của `class_embed`: weight ~ N(0, `init_std` = 0,02), bias = −log(99) ≈ −4,5951, giống hệt nhau ở mọi head. Bộ nhớ prototype thêm khoá rỗng |
| Q5 | Yêu cầu sinh ảnh (Minh, 30/10) | **(a) Đủ các mặt mà RPC có:** mọi kiểu đặt (trước/sau) × mọi camera được chọn cho SKU; cam0 phủ đủ 360°, khe hở lớn nhất ≤ 2 bước (18°). **(b) Góc thấy nhãn:** hàng đứng (lon, chai, hộp) dùng cam0 để lấy đủ các mặt bao quanh, nắp giữ 1% như v1; hàng dẹt dùng cam2 cả hai mặt; SKU dẹt mà mặt có nhãn chỉ chụp được từ ngang thì thêm cam0. **(c) Cân đối:** số vật có nhãn của mỗi SKU lệch ≤ 5% (v1 đã đạt ±0,07%); trong mỗi SKU và camera, các kiểu đặt dùng luân phiên; trong mỗi kiểu đặt, mọi góc dùng đều. Không dùng cam1/3, vì renderer chưa có `object_gain` cho hai camera này |
| Q6 | Nguồn cutout | Cắt **toàn bộ** ảnh cam0 và cam2 trong train2019 của 50 SKU, khoảng 2.640 + 2.640 ảnh (catalog hiện có 1.569 cutout cho 48 SKU). Nhóm hình dạng đề xuất: 119 và 121_canned_food → `upright`; 123 và 130_chocolate → `mixed`; 134 và 135_gum chốt khi duyệt |
| Q7 | Số ảnh | 12.330 ảnh mỗi task (trung vị task 2–5 của round 1), tổng 24.660; sinh mất khoảng 33 phút. Bố cục, scale, photometric và tỉ lệ vật mới 40–60% giữ như v1; vật của SKU khác không có nhãn (thầy gán nhãn giả) |
| Q8 | Train | Như round 1 (640 px, batch 8, 16 epoch, lr, F14, I3), chỉ khác `--n_classes 275` và dữ liệu. Khoảng 5,3 giờ mỗi task trên 4090. I3 (`--pseudo_gt_iou 0.5`) là bắt buộc: thầy sẽ gán nhãn giả theo label **gốc** cho chính các vật đang học lại, và I3 bỏ những nhãn giả trùng GT |
| Q9 | Chấm | Có hai cách đổi bí danh: `merge` giữ cả slot cũ lẫn bí danh; `alias_only` bỏ dự đoán ở slot cũ của các SKU đã học lại. Mỗi cách chấm như stage 5 của round 1 (`full_syn.score_val`). Chấm sau task 6 và sau task 7, lần nào cũng qua cổng ở Q10; chọn cách có cAcc-M2 val cao hơn, bằng nhau thì chọn `merge` |
| Q10 | Luật quyết định (viết trước khi chạy) | **Sau task 6:** train tiếp task 7 chỉ khi cAcc-M2 val ≥ 0,4368 + 0,0165 = 0,4533 **và** AP50 của 150 SKU không học lại giảm ≤ 0,01; ngược lại **stop**, không train task 7. Ngưỡng 0,0165 = 55% × 0,03: SKU của task 6 chiếm 55% số lỗi của 50 SKU trên val sau task 5 (2.639 so với 2.152 lần nhận sai, bỏ sót hoặc bị gọi nhầm sang), tương đương khoảng 2,5 lần sai số chuẩn. Quên mà đã vượt 1 điểm sau task 6 thì task 7 chỉ làm quên thêm. **Sau task 7:** **go** nếu cAcc-M2 val ≥ 0,4368 + 0,03 = 0,4668 **và** AP50 của 150 SKU giảm ≤ 0,01; ngược lại **stop**. Stop ở cổng nào thì nhóm cũng bàn hướng khác; hướng B (800 px) đã bỏ ngày 30/10 |
| Q11 | Test | Chỉ chấm một lần, sau khi `go` và người dùng xác nhận đây là cấu hình cuối |

**Giới hạn đã biết (ghi vào báo cáo):**
- Chọn SKU và chọn cách đổi bí danh đều dựa trên val, nên số trên val hơi lạc quan. Con số không thiên lệch là test (Q11).
- Học lại SKU đã học là **biến thể production**, không phải CL thuần như paper; round 1 vẫn là bản đúng paper.
- Mặt nào RPC không chụp (đáy hộp, góc xiên cam1/3) thì không sinh được.
- Khi chấm, ngưỡng điểm vẫn chia theo task gốc của SKU, nên dự đoán từ slot bí danh dùng chung ngưỡng với nhóm gốc. Ngưỡng vẫn được chọn tối ưu trên val, nên nếu có sai lệch thì chỉ làm kết quả thấp hơn thực tế, không làm ra `go` sai.
- Audit nhãn giả ở task 7 coi bí danh và label gốc của cùng SKU là hai nhãn khác nhau, nên precision báo ra thấp hơn thực tế. Audit chỉ để ghi lại, không làm cổng.

## Global Constraints

- Không sửa gì trong `data/training/full-syn-r1/`, `data/processed/rpc_synth_v2_v1/`, `data/processed/rpc_eval_official_v1/`. Mọi output mới vào thư mục mới: `rpc_synth_relearn_v1`, campaign `full-syn-relearn`.
- `TaskConfig.to_dict()` của config không có bí danh phải giữ nguyên từng byte như hiện nay, vì digest policy của round 1 dựa vào nó. Generator với config không bật `view_balance` phải cho đúng plan như hiện nay.
- Chỉ chọn trên val; test chấm đúng một lần (Q11).
- Chạy test bằng `python -m pytest <file> -q` từ repo root, trong env `pdp` **đã `conda activate`** (để `ninja` có trên PATH), chạy tuần tự (không dùng `-n`).
- Code theo phong cách file xung quanh; tài liệu viết tiếng Việt. Mỗi task một commit, kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Người dùng tự push.

## Review Focus

1. Nạp checkpoint **chưa mở rộng** vào model của config mới phải lỗi rõ ràng, không được âm thầm khởi tạo ngẫu nhiên → test ở Task 4.
2. Mở rộng một checkpoint không khớp config cũ, hoặc đã có slot sau `learned` được train, phải bị từ chối → test ở Task 4.
3. Bí danh trỏ sai (tới bí danh khác, tới class cùng task hay class sau, khác RPC id) phải bị `TaskConfig` từ chối → test ở Task 2.
4. SKU học lại ở task 6 xuất hiện trong ảnh task 7 phải mang nhãn bí danh task 6 trong `gt_full` (thầy của task 7 dự đoán bí danh), và không có nhãn trong file train → test ở Task 5.
5. Một vật có box ở cả slot cũ lẫn bí danh chỉ được đếm một lần, và AP không bị phạt vì box trùng → test ở Task 7.

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `configs/relearn/selection.json` (mới) | 50 SKU, quy tắc chọn và chia task, danh sách task 6/7 (Q1, Q2) |
| `autocheckout/taskcfg.py` (sửa) | Khái niệm bí danh: `ClassInfo.alias_of`, kiểm tra, ánh xạ RPC → label theo task |
| `tools/make_relearn_config.py` (mới) | Sinh config từ config gốc + các nhóm SKU (Q3) |
| `configs/relearn/task_config_relearn.json` (mới, sinh ra) | Config 275 lớp |
| `pdp/expand.py` (mới) | Mở rộng checkpoint (Q4), có CLI |
| `autocheckout/synthetic.py` (sửa) | Sinh ảnh/xuất nhãn cho task bí danh; cân đối kiểu đặt và góc; kiểm phủ mặt (Q5) |
| `tools/synth_v2.py` (sửa) | Subcommand `coverage`; `verify` đọc dung sai từ config |
| `autocheckout/full_syn.py` (sửa một dòng) | `_config` ghi đè `--n_classes` khi plan có số lớp riêng |
| `autocheckout/relearn.py` (mới) | File val bí danh, đổi bí danh trong dự đoán, luật quyết định, workflow campaign |
| `configs/exp/native/EXP-FULL-SYN-RL.sh` (mới) | EXP-FULL-SYN + `--n_classes 275` |
| `configs/data/synth_relearn.json` (mới) | Cấu hình generator (Q5–Q7) |
| `configs/training/full_syn_relearn.json` (mới) | Settings của campaign |
| `scripts/pack_relearn_data.sh` (mới) | Đóng gói release và file seed cho máy vast |
| `notebooks/data_preprocessing/06_synthetic_relearn.ipynb` (mới) | Log làm dữ liệu |
| `notebooks/training/07_full_syn_relearn.ipynb` (mới) | Log chạy campaign trên 4090 |
| Tests: `tests/test_taskcfg.py`, `tests/test_make_relearn_config.py`, `tests/test_pdp_expand.py`, `tests/test_synthetic_sampling.py`, `tests/test_synthetic_export.py`, `tests/test_synthetic_catalog.py`, `tests/test_relearn.py`, `tests/test_native_exp_configs.py` | |

---

### Task 1: Môi trường và danh sách SKU

**Files:**
- Create: `configs/relearn/selection.json`

**Interfaces:**
- Produces: `selection.json` với khoá `tasks` = `{"6": [...], "7": [...]}` (label gốc, tăng dần), dùng ở Task 3.

- [x] **Step 1: Sửa env** (đã làm ngày 30/10, người dùng đồng ý). Gỡ `numpy-base 2.2.5` mà conda cài đè (`conda remove --force`), rồi `pip install --force-reinstall --no-deps numpy==1.24.4`. Kết quả: numpy 1.24.4; pycocotools, pandas, torch import được. `pip check` chỉ còn báo `mkl-fft` và `mkl-random` (conda kéo theo khi cài pandas, project không dùng).

- [ ] **Step 2: Chạy toàn bộ test làm mốc, trong env đã activate**

Run: `conda activate pdp && python -m pytest tests -q`
Expected: PASS. Lần chạy ngày 30/10 không activate env thì 3 test trong `test_pdp_f11_kernel.py` báo "CUDA or ninja not available". Nếu vẫn lỗi khi đã activate, báo người dùng trước khi làm tiếp.

- [ ] **Step 3: Viết `configs/relearn/selection.json`**

```json
{
 "name": "relearn-v1",
 "source": {"run": "data/training/full-syn-r1/runs/EXP-FULL-SYN", "stage": 5, "split": "val_full.json",
            "policy": "calibration_count_m2.json"},
 "rule": "low: min(recall, precision) < 0.85; partners: same supercategory with >= 10 confusions (both directions) or any with >= 30; faces: SKUs whose cutouts miss the labelled face (d7)",
 "split": "groups = SKUs connected by the partner rule; by size desc (then smallest label), each group to the task with fewer SKUs (tie: task 6)",
 "low": [17, 18, 19, 39, 41, 46, 59, 60, 67, 68, 70, 98, 101, 102, 109, 111, 118, 129, 133, 141, 145, 153, 162, 163, 164, 172, 185, 186, 188, 191, 195, 199],
 "partners": [11, 49, 54, 55, 57, 58, 69, 84, 122, 134, 136, 138, 151, 178, 183, 190],
 "faces": [61, 166],
 "tasks": {
  "6": [19, 39, 49, 57, 58, 60, 61, 69, 70, 98, 102, 111, 118, 134, 141, 151, 162, 164, 178, 185, 186, 188, 190, 191, 199],
  "7": [11, 17, 18, 41, 46, 54, 55, 59, 67, 68, 84, 101, 109, 122, 129, 133, 136, 138, 145, 153, 163, 166, 172, 183, 195]
 }
}
```

- [ ] **Step 4: Kiểm tra file selection**
  - Hai task có 25 + 25 label, không trùng nhau.
  - Hợp của hai task bằng `low ∪ partners ∪ faces`.
  - Không cặp nhầm nào theo luật partner (đọc `pairs` trong `runs/analysis_r1/d6_selection.json`) bị chia sang hai task khác nhau.

Run: một lệnh `python -c` làm ba kiểm tra trên.
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add configs/relearn/selection.json
git commit -m "data(relearn): the 50 SKUs tasks 6 and 7 relearn"
```

---

### Task 2: Bí danh trong `TaskConfig`

**Files:**
- Modify: `autocheckout/taskcfg.py` (`ClassInfo`, `from_dict`, `to_dict`, `_validate`; thêm 2 method)
- Test: `tests/test_taskcfg.py`

**Interfaces:**
- Produces:
  - `ClassInfo.alias_of: int | None = None` (label gốc).
  - `TaskConfig.rpc_to_label()`: giữ nguyên nghĩa, chỉ label gốc, bỏ qua bí danh.
  - `TaskConfig.rpc_to_label_in(task_id: int) -> dict[int, int]`: như `rpc_to_label()`, nhưng bí danh của các task ≤ `task_id` đè lên label gốc; nếu một SKU có nhiều bí danh thì bí danh của task sau thắng.
  - `TaskConfig.alias_to_label() -> dict[int, int]`: label bí danh → label gốc.

- [ ] **Step 1: Viết test hỏng**

```python
def alias_config(**alias):
    d = task_config_dict(sizes=(2, 1), reserved=0)          # labels 0,1 | 2 ; rpc 1,2 | 3
    d["tasks"].append({"task_id": 3, "offset": 3, "reserved": False, "classes": [
        {"label": 3, "name": "sku_1@t3", "rpc_category_id": 1, "supercategory": "drink", "alias_of": 0, **alias}]})
    return d

def test_alias_class_shares_the_rpc_id_of_its_original():
    cfg = TaskConfig.from_dict(alias_config())
    assert cfg.rpc_to_label() == {1: 0, 2: 1, 3: 2}
    assert cfg.alias_to_label() == {3: 0}
    assert cfg.rpc_to_label_in(2)[1] == 0 and cfg.rpc_to_label_in(3)[1] == 3

@pytest.mark.parametrize("alias", [{"alias_of": 3}, {"alias_of": 7}, {"rpc_category_id": 2}])
def test_an_alias_must_point_to_an_existing_earlier_class_with_its_rpc_id(alias):   # itself, missing, other rpc id
    with pytest.raises(ValueError): TaskConfig.from_dict(alias_config(**alias))

def test_an_alias_of_an_alias_or_of_a_class_of_its_own_task_is_refused():
    # (a) task 4 = [{"label": 4, "name": "sku_1@t4", "rpc_category_id": 1, "alias_of": 3}]  (3 is an alias)
    # (b) task 3 = [{"label": 3, "rpc_category_id": 9}, {"label": 4, "rpc_category_id": 9, "alias_of": 3}]
    # both -> ValueError

def test_a_repeated_rpc_id_without_alias_is_still_refused(): ...   # alias_config() with "alias_of" removed -> ValueError
def test_alias_round_trips_and_plain_configs_serialise_as_before(): ...
```

Ghi chú:
- `test_alias_round_trips...`: `from_dict(to_dict(cfg))` phải giữ `alias_of`. Với config không có bí danh, `to_dict()` không ghi khoá `alias_of` ở class nào, để giữ digest cũ.
- Không cần test bí danh trỏ tới class reserved. `_validate` đã buộc task reserved đứng sau mọi task data, nên label đó luôn ở sau bí danh và rơi vào trường hợp "không phải class trước".

- [ ] **Step 2: Chạy, xác nhận hỏng** — `python -m pytest tests/test_taskcfg.py -q` → FAIL (`alias_of` chưa tồn tại).

- [ ] **Step 3: Cài đặt**
  - Trong `_validate`: một RPC id được xuất hiện lần hai **chỉ khi** class sau có `alias_of` = label của class gốc mang đúng RPC id đó, class gốc không phải bí danh và thuộc một task data đứng trước.
  - `to_dict` chỉ ghi `alias_of` khi giá trị khác `None`.

- [ ] **Step 4: Chạy test của file và các test dùng `TaskConfig`**

Run: `python -m pytest tests/test_taskcfg.py tests/test_pdp_f1_task_config.py tests/test_make_task_json.py tests/test_make_task_config.py tests/test_synthetic_export.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autocheckout/taskcfg.py tests/test_taskcfg.py
git commit -m "feat(taskcfg): alias classes relearn an already learned SKU in a later task"
```

---

### Task 3: Sinh config học lại

**Files:**
- Create: `tools/make_relearn_config.py`, `configs/relearn/task_config_relearn.json` (sinh ra)
- Test: `tests/test_make_relearn_config.py`

**Interfaces:**
- Consumes: `TaskConfig` có `alias_of` (Task 2); `selection.json` (Task 1).
- Produces: `relearn_config(base: TaskConfig, tasks: Sequence[Sequence[int]], *, reserved: int, name: str) -> TaskConfig`.
  - Giữ các task data của `base` (bỏ task reserved).
  - Mỗi phần tử của `tasks` thành một task bí danh, id nối tiếp, offset nối tiếp. Class thứ i ứng với `sorted(group)[i]`, tên `f"{gốc.name}@t{task_id}"`, cùng `rpc_category_id` và `supercategory`, `alias_of` = label gốc.
  - Sau cùng là một task reserved gồm `reserved` slot, tên `reserved_<label>` như config hiện nay.
  - Từ chối (`ValueError`): label không phải class data của `base`, label lặp lại (trong cùng nhóm hoặc giữa các nhóm), nhóm rỗng.
  - CLI: `python -m tools.make_relearn_config --base PATH --selection PATH --reserved 24 --name NAME --out PATH`, đọc `selection["tasks"]` theo thứ tự id.

- [ ] **Step 1: Viết test hỏng**

```python
def test_relearn_config_appends_alias_tasks_and_keeps_the_learned_tasks():
    base = TaskConfig.from_dict(task_config_dict(sizes=(3, 2), reserved=2))
    cfg = relearn_config(base, [[4, 0], [1]], reserved=2, name="toy+relearn")
    assert [t.num_classes for t in cfg.tasks] == [3, 2, 2, 1, 2] and cfg.num_classes == 11
    assert cfg.tasks[:2] == base.tasks[:2]
    assert [(c.label, c.alias_of, c.rpc_category_id) for c in cfg.task(3).classes] == [(5, 0, 1), (6, 4, 5)]
    assert cfg.task(3).classes[0].name == "sku_1@t3" and cfg.task(4).classes[0].name == "sku_2@t4"
    assert cfg.task(5).reserved

@pytest.mark.parametrize("tasks", [[[5]], [[0], [0]], [[0, 0]], [[0], []]])
def test_relearn_config_refuses_unlearned_repeated_or_empty(tasks):
    with pytest.raises(ValueError): relearn_config(base, tasks, reserved=2, name="x")
```

- [ ] **Step 2: Chạy, xác nhận hỏng** — `python -m pytest tests/test_make_relearn_config.py -q` → FAIL (module chưa có).

- [ ] **Step 3: Cài đặt `relearn_config` và CLI** (argparse theo mẫu `tools/make_task_config.py`).

- [ ] **Step 4: Chạy test** — PASS.

- [ ] **Step 5: Sinh config thật và kiểm**

Run: `python -m tools.make_relearn_config --base data/processed/rpc_eval_official_v1/task_config.json --selection configs/relearn/selection.json --reserved 24 --name 100-4x25_seed0_relearn2x25 --out configs/relearn/task_config_relearn.json && python -c "from autocheckout.taskcfg import TaskConfig as T;c=T.load('configs/relearn/task_config_relearn.json');print(c.num_classes,[t.num_classes for t in c.tasks],c.task(6).classes[0].name,c.task(7).classes[0].name)"`
Expected: `275 [100, 25, 25, 25, 25, 25, 25, 24] 38_instant_drink@t6 23_dried_food@t7`

- [ ] **Step 6: Commit**

```bash
git add tools/make_relearn_config.py tests/test_make_relearn_config.py configs/relearn/task_config_relearn.json
git commit -m "feat(relearn): task config with two 25-alias tasks and 24 reserved slots"
```

---

### Task 4: Mở rộng checkpoint

**Files:**
- Create: `pdp/expand.py`
- Test: `tests/test_pdp_expand.py`

**Interfaces:**
- Consumes: `TaskConfig` (Task 2); `relearn_config` (Task 3), dùng để dựng config toy B trong test.
- Produces: `expand_checkpoint(checkpoint: dict, old_cfg: TaskConfig, new_cfg: TaskConfig, learned: int, seed: int = 0) -> dict`.
  - Trả về checkpoint mới cùng định dạng với `save_task_final` (`runtime`, `model`, `task_id`, `class_query_cache`, `class_prototypes`, `class_cache_count`), thêm khoá `expanded_from = {old_classes, new_classes, learned, seed}`.
  - CLI: `python pdp/expand.py --checkpoint IN --old-config A --new-config B --learned N --out OUT`. Ghi qua file tạm rồi `os.replace`, như `save_task_final`.

Quy tắc (Q4), áp dụng theo tên khoá:
- Khoá chứa `class_embed` có chiều 0 = `old_cfg.num_classes`: `cat(old[:learned], mới, old[-1:])`. Phần mới sinh **một lần** bằng `torch.Generator().manual_seed(seed)`: weight ~ N(0, `runtime.model_config.init_std`), bias = `-log((1 - 0.01) / 0.01)`. Dùng cùng giá trị cho mọi head.
- Khoá `model.prompts.private_{p,k,a}_{l}` có chiều 0 = `old_cfg.num_slots`: `cat(old[:learned], zeros)`.
- Mọi khoá khác chép nguyên. Nếu khoá khác có chiều phụ thuộc số lớp thì raise.
- Bộ nhớ PDP: giữ các khoá `< learned`. Khoá `learned .. new_cfg.num_slots - 1` khởi tạo bằng `[]`, `zeros(d_model)`, `0`.
- `runtime.model_config`: `id2label` / `label2id` = `LABEL_i` với `i < new_cfg.num_classes`; `task_num_classes` = cỡ các task của `new_cfg`.
- Từ chối (`ValueError`) khi:
  - số hàng không khớp `old_cfg`;
  - prompt ở hàng `learned .. old_slots - 1` khác 0 (đã có task sau được train);
  - `new_cfg` khác `old_cfg` ở bất kỳ class nào có label `< learned`.

- [ ] **Step 1: Viết test hỏng**

```python
def test_learned_rows_are_copied_and_new_rows_start_untrained(tmp_path, monkeypatch):
    # tiny model; config A = sizes (3, 2) + 1 reserved (n_classes 7); A's task-2 checkpoint written by save_task_final
    # after init_task_prompts() at task 2 (prompts of tasks 1-2 non-zero); B = relearn_config(A, [[0, 3], [1]], reserved=1)
    # -> tasks (3, 2, 2, 1, 1), n_classes 10
    new = expand_checkpoint(old, cfg_a, cfg_b, learned=5)
    w_old, w_new = old["model"]["class_embed.0.weight"], new["model"]["class_embed.0.weight"]
    assert w_new.shape[0] == 10 and torch.equal(w_new[:5], w_old[:5]) and torch.equal(w_new[-1], w_old[-1])
    assert torch.allclose(new["model"]["class_embed.0.bias"][5:9], torch.full((4,), -math.log(99)))
    assert all(torch.equal(new["model"][f"class_embed.{i}.weight"], w_new) for i in heads)
    p = new["model"]["model.prompts.private_k_0"]
    assert torch.equal(p[:5], old["model"]["model.prompts.private_k_0"][:5]) and p[5:].abs().sum() == 0
    assert len(new["class_prototypes"]) == cfg_b.num_slots and new["class_query_cache"][8] == []
    runtime.validate_runtime(args_b, new["runtime"])                       # passes
    with pytest.raises(ValueError): runtime.validate_runtime(args_a, new["runtime"])

def test_expanded_model_predicts_the_learned_classes_exactly_as_before(tmp_path, monkeypatch):
    # same batch, prompts at task_count 1 in both models: logits[..., :5] and pred_boxes are torch.equal

def test_task_after_expansion_trains_on_alias_labels(tmp_path, monkeypatch):
    # trainer B task 3: resume(expanded), set_teacher(), init_task_prompts(); common_step on make_batch([[5, 6], [6]])
    # -> finite loss; after backward, model.prompts.private_p_<first e layer>.grad is 0 on rows < 5, non-zero on rows 5..6

def test_unexpanded_checkpoint_does_not_load_into_the_larger_model(tmp_path, monkeypatch):
    # trainer B .resume(path of the A checkpoint) raises RuntimeError (size mismatch in load_state_dict)

@pytest.mark.parametrize("break_it", ["rows", "trained_later_slot", "changed_learned_class"])
def test_expansion_refuses_a_checkpoint_or_config_that_does_not_match(break_it, ...):  # -> ValueError
```

Dựng model bằng `pdp_helpers.use_tiny_detr`, `pdp_args`, `make_trainer`. Config B được ghi ra file và parse như `pdp_args` làm, với `--n_classes` = `cfg_b.num_classes`.

- [ ] **Step 2: Chạy, xác nhận hỏng** — `python -m pytest tests/test_pdp_expand.py -q` → FAIL (module chưa có).

- [ ] **Step 3: Cài đặt `pdp/expand.py`.**

- [ ] **Step 4: Chạy test của task và các test PDP liên quan**

Run: `python -m pytest tests/test_pdp_expand.py tests/test_pdp_f2_private_pool.py tests/test_pdp_f8_paths.py tests/test_pdp_runtime.py tests/test_pdp_r1_resume.py -q`
Expected: PASS

- [ ] **Step 5: Kiểm trên checkpoint thật (CPU, không commit output)**

Run: `python pdp/expand.py --checkpoint data/training/full-syn-r1/runs/EXP-FULL-SYN/task_5/task_final.pth --old-config data/processed/rpc_eval_official_v1/task_config.json --new-config configs/relearn/task_config_relearn.json --learned 200 --out "${TMPDIR:-/tmp}/expand_check.pth"`, rồi in shape `class_embed.0.weight`, số khoá `class_prototypes` và `expanded_from`.
Expected: `torch.Size([275, 256]) 274 {...}`; xong thì xoá file.

- [ ] **Step 6: Commit**

```bash
git add pdp/expand.py tests/test_pdp_expand.py
git commit -m "feat(pdp): expand a trained checkpoint to more class slots, learned outputs unchanged"
```

---

### Task 5: Generator và export hiểu task bí danh

**Files:**
- Modify: `autocheckout/synthetic.py`: `plan_scenes` (dòng 208–209), `export_task_annotations` (dòng 470), `verify_dataset` (dòng 520)
- Test: `tests/test_synthetic_sampling.py`, `tests/test_synthetic_export.py`

**Interfaces:**
- Consumes: `rpc_to_label_in` (Task 2); `relearn_config` (Task 3), dùng để dựng config toy.
- Produces:
  - `plan_scenes` với task bí danh:
    - SKU "mới" (có nhãn) = RPC id các class của task.
    - SKU "cũ" (không nhãn) = RPC id **duy nhất** của các class gốc (không phải bí danh) thuộc task data đứng trước, **trừ** các SKU mới.
  - `export_task_annotations` và `verify_dataset` ánh xạ RPC id sang label bằng `task_config.rpc_to_label_in(task_id)`.
  - Config không có bí danh: kết quả y hệt hiện nay.

- [ ] **Step 1: Viết test hỏng** (dùng fixture sẵn có của từng file; config toy `cfg_rl = relearn_config(<toy>, [[0, 3], [1]], reserved=1)`, có task 3 và task 4 là bí danh)

```python
def test_relearn_task_scenes_label_the_relearned_skus_and_never_list_them_as_old():
    recipes = plan_scenes(3, 40, task_config=cfg_rl, ...)
    objs = [o for r in recipes for o in r["objects"]]
    relearned = {c.rpc_category_id for c in cfg_rl.task(3).classes}
    assert all(o["labeled"] == (o["rpc_category_id"] in relearned) for o in objs)
    assert {o["rpc_category_id"] for o in objs if not o["labeled"]} <= original_rpc_ids - relearned

def test_a_sku_relearned_in_task_3_appears_in_task_4_under_its_task_3_alias():
    files = export_task_annotations(coco_rpc_task4, 4, cfg_rl)      # scene contains RPC ids of task 3 and task 4
    assert {a["category_id"] for a in files["train_task_4.json"]["annotations"]} <= set(cfg_rl.task(4).labels)
    full = {a["category_id"] for a in files["train_task_4_gt_full.json"]["annotations"]}
    assert set(cfg_rl.task(3).labels) <= full and not ({0, 3} & full)    # originals of task-3 SKUs never used

def test_verify_accepts_a_relearn_release(): ...   # verify_dataset(...) on a toy release of tasks 3-4 has no failures
```

- [ ] **Step 2: Chạy, xác nhận hỏng** — `python -m pytest tests/test_synthetic_sampling.py tests/test_synthetic_export.py -q` → FAIL.

- [ ] **Step 3: Cài đặt** (ba chỗ sửa nêu trên).

- [ ] **Step 4: Chạy lại cùng lệnh, cộng thêm `tests/test_synthetic_catalog.py tests/test_synthetic_compositing.py tests/test_synthetic_loader.py`** → PASS.

- [ ] **Step 5: Commit**

```bash
git add autocheckout/synthetic.py tests/test_synthetic_sampling.py tests/test_synthetic_export.py
git commit -m "feat(synth): scenes and labels for tasks that relearn earlier SKUs"
```

---

### Task 6: Đủ mặt và cân đối mặt, góc (Q5)

**Files:**
- Modify: `autocheckout/synthetic.py` (thêm `source_view`, `view_coverage`; nhánh `view_balance` trong `plan_scenes`; tham số mới của `verify_dataset`), `tools/synth_v2.py` (subcommand `coverage`; `cmd_verify` truyền dung sai)
- Test: `tests/test_synthetic_sampling.py`, `tests/test_synthetic_catalog.py`, `tests/test_synthetic_export.py`

**Interfaces:**
- Produces:
  - `source_view(source: str) -> tuple[str, int, int]`: (kiểu đặt `'front'|'back'`, camera, bước xoay 1–40), đọc từ tên file RPC như `6909409012031-back_camera0-15.jpg`.
  - `view_coverage(cutouts: list[dict], skus: list[dict], train_coco: dict, rpc_ids: Iterable[int], *, max_gap: int = 2) -> dict[int, dict]`. Kết quả cho mỗi SKU gồm:
    - `placements`: các kiểu đặt RPC có cho SKU đó;
    - `views`: `{"front/c0": n, ...}`, chỉ tính camera có weight > 0;
    - `cam0_gap`: `{kiểu đặt: khe xoay lớn nhất, đơn vị bước}`;
    - `ok`: đúng khi mọi (kiểu đặt × camera có weight > 0) đều có cutout và mọi `cam0_gap` ≤ `max_gap`.
  - `plan_scenes` khi `layout["view_balance"]` đúng:
    - Vòng xáo cutout tách theo `(sku, camera, kiểu đặt)`.
    - Trong mỗi `(sku, camera)`, kiểu đặt được chọn luân phiên theo thứ tự tên.
    - Camera vẫn chọn theo `camera_weights` như cũ.
    - Khi khoá này vắng hoặc sai: code và thứ tự rng y như hiện nay.
  - `verify_dataset(..., quota_tolerance, cutouts=None, view_tolerance=None)`: khi có `cutouts` và `view_tolerance`, với mỗi SKU có nhãn và mỗi camera, số vật theo kiểu đặt phải lệch ≤ `view_tolerance` so với trung bình; nếu không thì ghi lỗi `view_balance`.
  - `cmd_verify`:
    - `quota_tolerance` lấy từ `layout.get("quota_tolerance", 0.2)`.
    - Khi `layout.get("view_balance")`, truyền thêm catalog và `view_tolerance = layout["quota_tolerance"]`.
  - CLI: `python -m tools.synth_v2 coverage --config C --tasks 6 7`. Ghi `<catalog dir>/coverage.json`, exit ≠ 0 nếu có SKU `ok = false`.

- [ ] **Step 1: Viết test hỏng**

```python
def test_source_view_reads_placement_camera_and_step():
    assert source_view("6909409012031-back_camera0-15.jpg") == ("back", 0, 15)
    assert source_view("038900004095_camera2-3.png") == ("front", 2, 3)

def test_view_coverage_flags_a_missing_placement_and_a_wide_cam0_gap():
    # toy: SKU 1 upright ({"0": 1.0}), RPC has front+back; cutouts front/c0 steps 1..40, back/c0 steps 1, 2, 3
    cov = view_coverage(cutouts, skus, train_coco, [1, 2])
    assert cov[1]["cam0_gap"]["back"] == 38 and not cov[1]["ok"]
    assert cov[2]["ok"]     # SKU 2 flat ({"2": 1.0}), cutouts on both placements

def test_view_balance_alternates_placements_and_spreads_cutouts():
    # SKU 1 flat with 6 front/c2 and 2 back/c2 cutouts; layout["view_balance"] = True
    objs = [o for r in plan_scenes(..., layout={**layout, "view_balance": True}) for o in r["objects"] if o["labeled"]]
    front, back = counts by placement
    assert abs(front - back) <= 1
    assert max(per-cutout uses within a placement) - min(...) <= 2

def test_without_view_balance_the_plan_is_unchanged():
    # same inputs, layout without the key: recipes == those of the current code (golden recipes saved in the test)

def test_verify_reports_unbalanced_views(): ...   # a release with 70/30 front/back -> failures["view_balance"]
```

- [ ] **Step 2: Chạy, xác nhận hỏng** — `python -m pytest tests/test_synthetic_sampling.py tests/test_synthetic_catalog.py tests/test_synthetic_export.py -q` → FAIL.

- [ ] **Step 3: Cài đặt.** Golden recipes của `test_without_view_balance...` phải được sinh từ code **trước** khi sửa: viết test đó trước, chạy, lưu kết quả vào test, rồi mới sửa `plan_scenes`.

- [ ] **Step 4: Chạy lại cùng lệnh** → PASS.

- [ ] **Step 5: Kiểm v2 không đổi.** Chạy `python -m tools.synth_v2 plan --config configs/data/synth_v2.json --out-dir <thư mục tạm> --tasks 2`, rồi `cmp <thư mục tạm>/plans/task_2.jsonl data/processed/rpc_synth_v2_v1/plans/task_2.jsonl`.
Expected: không có khác biệt.

- [ ] **Step 6: Commit**

```bash
git add autocheckout/synthetic.py tools/synth_v2.py tests/test_synthetic_sampling.py tests/test_synthetic_catalog.py tests/test_synthetic_export.py
git commit -m "feat(synth): every face of an SKU, placements and angles used evenly"
```

---

### Task 7: File val bí danh, đổi bí danh trong dự đoán, luật quyết định

**Files:**
- Create: `autocheckout/relearn.py` (phần helper)
- Test: `tests/test_relearn.py`

**Interfaces:**
- Consumes: `TaskConfig.alias_to_label`, `Predictions`, `save_predictions` / `load_predictions`.
- Produces:
  - `MODES = ('merge', 'alias_only')`; `GAIN = {6: 0.0165, 7: 0.03}` (Q10); `OTHER_DROP = 0.01`.
  - `alias_val_file(val_full: dict, cfg: TaskConfig, task_id: int) -> dict`:
    - Giữ các ảnh val có ít nhất một vật thuộc SKU của task bí danh `task_id`.
    - Chỉ giữ annotation của các SKU đó, `category_id` đổi sang bí danh.
    - `categories` là các bí danh (`id`, `name`, `supercategory`).
  - `resolve_predictions(preds: Predictions, cfg: TaskConfig, *, mode: str, as_stage: int) -> Predictions`. Gọi `t = preds.meta['task_id']`, rồi:
    - Từ chối label ≥ `cfg.seen_classes(t)`.
    - Chỉ dùng bí danh của các task ≤ t.
    - `alias_only` bỏ các hàng mang label gốc của SKU đã có bí danh (ở task ≤ t).
    - Đổi bí danh → label gốc.
    - Gộp các hàng trùng `(image_id, query, label)`, giữ điểm cao nhất.
    - `meta`: `task_id = as_stage`, `seen_classes = cfg.seen_classes(as_stage)`, `resolved_from = {'mode', 'task_id': t}`; các khoá meta khác giữ nguyên.
  - `resolve_run(run: Path, stage: int, cfg: TaskConfig, target: Path, *, mode: str, as_stage: int, splits: Sequence[str] = ('val',)) -> Path`:
    - Ghi `target/task_{as_stage}/pred_{split}.npz` đã đổi bí danh.
    - Symlink `task_final.pth` và `run_info.json` của `run/task_{stage}`.
    - **Chỉ** có stage `as_stage` trong `target`, vì `evaluate_policy` đòi tập stage của test bằng tập stage của policy, và kiểm md5 checkpoint từng stage. Nhờ vậy máy vast không cần checkpoint task 1–4 của seed.
  - `relearn_verdict(report: dict, seed: dict, stage: int) -> tuple[str, str]`: đọc khoá `cAcc_m2` và `AP50_other`, so độ tăng với `GAIN[stage]` và độ giảm với `OTHER_DROP`, làm tròn 6 chữ số rồi mới so sánh như `pilot_verdict`. Trả về `go` hoặc `stop`; lý do ghi rõ cổng nào hỏng.

- [ ] **Step 1: Viết test hỏng** (`cfg = relearn_config(task_config_dict(sizes=(2, 1), reserved=0), [[0], [2]], reserved=1)` → bí danh 3 của label 0 ở task 3, bí danh 4 của label 2 ở task 4)

```python
def test_alias_val_file_keeps_the_relearned_objects_under_their_alias_labels(): ...
def test_merge_counts_an_object_seen_by_the_old_slot_and_the_alias_once():
    # task-4 preds: (img 1, q0, label 0, .6), (img 1, q0, label 3, .9), (img 1, q1, label 4, .5), (img 1, q3, label 0, .7)
    out = resolve_predictions(preds, cfg, mode="merge", as_stage=2)
    assert sorted(zip(out.query.tolist(), out.label.tolist(), out.score.round(2).tolist())) == [(0, 0, .9), (1, 2, .5), (3, 0, .7)]
    assert out.meta["task_id"] == 2 and out.meta["seen_classes"] == cfg.seen_classes(2)
def test_alias_only_drops_the_old_slot_of_relearned_skus():          # same rows -> (3, 0, .7) is gone
def test_after_task_3_the_old_slot_of_a_task_4_sku_is_kept():       # task-3 preds with label 2 rows, alias_only -> kept
def test_resolve_refuses_labels_not_yet_seen(): ...                 # task-3 preds containing label 4 -> ValueError
def test_resolve_run_lays_out_a_stage_the_evaluators_read(tmp_path):
    # target/task_2/pred_val.npz passes tools.eval_count._validated_predictions for stage 2 of the base config;
    # discover_stage_predictions(target, 'val') == {2: ...}; task_final.pth and run_info.json are symlinks to the source
@pytest.mark.parametrize("stage, cacc, ap_other, decision", [
    (6, 0.4533, 0.94, "go"), (6, 0.4532, 0.95, "stop"), (6, 0.4533, 0.9399, "stop"),
    (7, 0.4668, 0.94, "go"), (7, 0.4667, 0.95, "stop"), (7, 0.4668, 0.9399, "stop"), (7, 0.50, 0.96, "go")])
def test_relearn_verdict_needs_the_gain_of_its_stage_without_hurting_the_other_skus(stage, cacc, ap_other, decision):
    seed = {"cAcc_m2": 0.4368, "AP50_other": 0.95}
    assert relearn_verdict({"cAcc_m2": cacc, "AP50_other": ap_other}, seed, stage)[0] == decision
```

- [ ] **Step 2: Chạy, xác nhận hỏng** — `python -m pytest tests/test_relearn.py -q` → FAIL.

- [ ] **Step 3: Cài đặt bốn hàm.**

- [ ] **Step 4: Chạy** — `python -m pytest tests/test_relearn.py tests/test_eval_count.py tests/test_counting.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add autocheckout/relearn.py tests/test_relearn.py
git commit -m "feat(relearn): alias val files, alias resolution of predictions and the verdict"
```

---

### Task 8: Workflow campaign học lại

**Files:**
- Modify: `autocheckout/relearn.py` (phần workflow), `autocheckout/full_syn.py` (`_config`, một dòng)
- Create: `configs/exp/native/EXP-FULL-SYN-RL.sh`, `configs/training/full_syn_relearn.json`
- Test: `tests/test_relearn.py` (dry run, đánh dấu chậm như `test_m5_pilot...`), `tests/test_native_exp_configs.py`

**Interfaces:**
- Consumes:
  - Task 3–7.
  - Từ `full_syn`: `_attempt`, `_environment`, `_config`, `_train`, `_audit`, `_run`, `_gate`, `_finish`, `_require`, `_fsa_hf`, `score_val`, `_toy_releases`, `link_tasks`, `GateStop`.
  - `training_workflow.campaign_root`, `resolve`; `cl_metrics.group_ap`; `tools.eval_cl`, `tools.eval_count`.
- Produces: các stage cùng kiểu với `full_syn`. Mỗi stage ghi `stages/<name>.json` và trả về dict; chạy lại thì đọc kết quả cũ, không train lại.
  - `RELEARN = 'EXP-FULL-SYN-RL'`; `RELEARN_TASKS = (6, 7)` (đọc từ config: các task mà mọi class đều là bí danh).
  - `full_syn._config`: khi plan có khoá `n_classes`, thêm dòng cuối `ARGS+=(--n_classes <n_classes>)`, đặt sau tham số model nhỏ của dry run (trong đó có `--n_classes 7`). Plan không có khoá này thì không đổi gì.
  - `prepare(settings: dict) -> dict`:
    - Settings là các khoá của `full_syn_m5_pilot.json`, bỏ `pdp_experiment` và `fsa_*`, thêm `task_config`.
    - Plan có **đủ các khoá** của plan `full_syn.prepare`, để `_environment`, `_config`, `_train`, `_audit`, `_attempt` chạy không đổi.
    - `resolution`, `batch_size`, `effective_batch`, `epochs`, `tf32`, `num_workers`, `checkpoint_every_minutes` lấy từ `plan.json` của seed; settings khác seed thì từ chối.
    - `pdp_experiment = RELEARN`, `task_config` = config học lại, `release_root`/`synth_root` = release học lại, `train_images = [số ảnh task 6, số ảnh task 7]`.
    - Thêm các khoá: `seed_root`; `seed = {campaign_id, round, dry_run, task5_md5}`; `seed_task_config`; `n_classes`; `relearn_tasks`; `relearned` (label gốc).
    - `task_dir` chứa:
      - `train_task_{6,7}.json` và `train_task_{6,7}_gt_full.json` (symlink, sha khớp manifest của release);
      - `val_task_{6,7}.json` (ghi bằng `alias_val_file`);
      - `val_full.json` và `test_full.json` (symlink, sha khớp manifest eval).
    - Dry run:
      - Seed là một campaign FULL-SYN dry.
      - Config = `relearn_config(<config toy 6 label>, [[1], [3]], reserved=1)` (`n_classes` 7 → 10), ghi vào fixture.
      - Release dựng từ fixture của `_toy_releases`. Với mỗi task bí danh t: `train_task_t.json` = ảnh của `train_full.json` toy, chỉ giữ vật của SKU task t, label đổi sang bí danh; `gt_full` = mọi vật, map theo `rpc_to_label_in(t)`.
    - Từ chối khi: task 1–5 của config khác `seed_task_config`; trộn dry run với chạy thật; đổi setting hoặc input trong campaign đã có (như `full_syn.prepare`).
  - `stage_expand(plan)`:
    - Kiểm md5 của `task_5/task_final.pth` trong seed.
    - Ghi `runs/EXP-FULL-SYN-RL/task_5/task_final.pth` bằng `expand_checkpoint(..., learned=seed_cfg.seen_classes(5))`.
    - Link `runs/EXP-FULL-SYN-FSA → seed/runs/EXP-FULL-SYN-FSA` như `full_syn.stage_seed`, vì `_fsa_hf(plan)` đọc từ `plan['runs']`.
  - `stage_task(plan, t)` với t ∈ `RELEARN_TASKS`; ghi `stages/task{t}.json`:
    - Với t = 7: chỉ chạy khi `stages/task6.json` có decision `go`. Dry run thì cổng không chặn, như `full_syn`.
    - `full_syn._audit(plan, record, attempt, t)` (thầy = checkpoint task t−1; chỉ ghi lại, không chặn).
    - `full_syn._train(plan, record, attempt, RELEARN, start=t, n=t, epochs=plan['epochs'], splits='val', fsa_hf=_fsa_hf(plan))`.
    - Chấm, với mỗi mode:
      - `resolve_run(run, t, cfg, root/'scored'/f'task{t}_{mode}', mode=mode, as_stage=5)`.
      - Rồi `full_syn.score_val({'task_config': plan['seed_task_config'], 'task_dir': plan['task_dir']}, target, 5)`. Hàm này cũng ghi policy val vào `target`.
      - Thêm `AP50_other` = `group_ap(coco val_full, dự đoán đã đổi bí danh, 150 label không học lại)['AP50']`, không NMS, như `eval_cl`.
    - Số của seed: `tasks.json['tasks']['5']` (cAcc-M2 0,4368), cộng `AP50_other` tính cùng cách trên `task_5/pred_val.npz` của seed.
    - Chọn mode (Q9), gọi `relearn_verdict(..., stage=t)`, rồi `_gate` (dry run thì không chặn; chạy thật thì `stop` raise `GateStop`).
  - `stage_final_test(plan, confirm: bool = False)`:
    - Chỉ chạy khi `stages/task7.json` có decision `go` và `confirm=True`. Chỉ chấm một lần, như `full_syn.stage_final_test`.
    - Dự đoán test của task 7 bằng `_train(..., start=7, n=7, splits='test')`.
    - `resolve_run(splits=('test',))` với mode đã chọn, vào `root/scored/task7_<mode>`.
    - Chấm bằng `eval_cl.evaluate_run` và `eval_count.evaluate_policy` với `calibration_count_groups.json` và `calibration_count.json` của thư mục đó. Chỉ test2019, không chấm `compare_b12`.
- `EXP-FULL-SYN-RL.sh`: `source EXP-FULL-SYN.sh` rồi `ARGS+=(--n_classes 275)  # configs/relearn/task_config_relearn.json`.

- [ ] **Step 1: Viết test hỏng**

```python
def test_full_syn_rl_config_only_sets_the_class_count():   # tests/test_native_exp_configs.py, like the M5 test
    _, pdp = resolved_args("EXP-FULL-SYN", FSA_HF="/fixture/fsa/task_1/hf_model")
    _, rl = resolved_args("EXP-FULL-SYN-RL", FSA_HF="/fixture/fsa/task_1/hf_model")
    assert {k for k in vars(pdp) if getattr(rl, k) != getattr(pdp, k)} == {"n_classes"} and rl.n_classes == 275

def test_prepare_refuses_a_config_whose_learned_tasks_differ_from_the_seed(tmp_path): ...   # ValueError

def test_task_7_is_refused_when_the_task_6_gate_said_stop(tmp_path):
    # real-run plan (dry_run False) with stages/task6.json gate decision "stop" -> stage_task(plan, 7) raises GateStop
    # before any training command is started

@pytest.mark.slow
def test_relearn_dry_run_expands_trains_two_tasks_and_scores_both_modes(tmp_path):
    # seed: a dry FULL-SYN round through full_syn.stage_tasks(plan, last=5); relearn config of the toy (n_classes 10)
    relearn.stage_expand(plan); r6 = relearn.stage_task(plan, 6); r7 = relearn.stage_task(plan, 7)
    assert torch.load(run / "task_5/task_final.pth")["model"]["class_embed.0.weight"].shape[0] == 10
    assert load_predictions(run / "task_7/pred_val.npz").label.max() < 8
    for r in (r6, r7):
        assert set(r["modes"]) == {"merge", "alias_only"}
        assert r["gate"]["decision"] in {"go", "stop"} and r["gate"]["enforced"] is False
    mtime = (run / "task_7/task_final.pth").stat().st_mtime
    relearn.stage_task(plan, 7); assert (run / "task_7/task_final.pth").stat().st_mtime == mtime
```

- [ ] **Step 2: Chạy, xác nhận hỏng** — `python -m pytest tests/test_relearn.py tests/test_native_exp_configs.py -q` → FAIL.

- [ ] **Step 3: Cài đặt workflow, script EXP và file settings.** Settings mặc định `dry_run: true` như `full_syn_m5_pilot.json`; `campaign_id: "full-syn-relearn"`, `seed_campaign: "data/training/full-syn-r1"`, `task_config: "configs/relearn/task_config_relearn.json"`, `release_root: "data/processed/rpc_synth_relearn_v1"`; các khoá còn lại chép từ settings M5.

- [ ] **Step 4: Chạy test của task và toàn bộ test FULL-SYN**

Run: `python -m pytest tests/test_relearn.py tests/test_native_exp_configs.py tests/test_full_syn.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autocheckout/relearn.py autocheckout/full_syn.py configs/exp/native/EXP-FULL-SYN-RL.sh configs/training/full_syn_relearn.json tests/test_relearn.py tests/test_native_exp_configs.py
git commit -m "feat(relearn): campaign that expands, trains tasks 6-7 with a gate after each, scores both alias modes"
```

---

### Task 9: Dữ liệu (a) — cutout và catalog cho 50 SKU

Đây là bước chạy dữ liệu, không làm theo TDD. Mọi thứ ghi log trong `notebooks/data_preprocessing/06_synthetic_relearn.ipynb`; mỗi bước có tiêu chí đạt. **Có hai điểm dừng chờ người dùng duyệt.**

**Files:**
- Create: `notebooks/data_preprocessing/06_synthetic_relearn.ipynb`
- Output (gitignored): `data/processed/cutouts_relearn_v1/`; `data/processed/rpc_synth_relearn_v1/catalog/` gồm `cutouts_relearn_v1.json`, `reviews.json`, `skus.json`, `cutouts.json`, `m0_review.json`, `coverage.json`

**Interfaces:**
- Consumes: `view_coverage` / subcommand `coverage` (Task 6).
- Produces: `rpc_synth_relearn_v1/catalog/cutouts.json` và `skus.json`, đúng định dạng v1, dùng ở Task 10.

- [ ] **Step 1: Giải nén ảnh cam2 còn thiếu của 50 SKU từ zip Kaggle.** Chỉ lấy các file `retail_product_checkout/train2019/<tên>` có camera 2, thuộc 50 RPC id và chưa có trong `data/archive/train2019`.
  Đạt khi đủ ảnh cam0 và cam2 của 50 SKU theo `instances_train2019.json` (khoảng 2.640 + 2.640).

- [ ] **Step 2: Cắt bằng SAM** (env `sam`; khoảng 20 phút trên 3060)

Run: `conda activate sam && python data/archive/scripts/02_extract_objects.py --root data/archive --out data/processed/cutouts_relearn_v1 --per-class 200 --camera-weights 0:1,1:0,2:1,3:0 --classes <50 RPC id> --device cuda`
Đạt khi `index.json` có cutout của đủ 50 SKU, và `rejected.json` có lý do cho phần bị loại.

- [ ] **Step 3: Kiểm kê** — `python -m tools.synth_v2 inventory --index data/processed/cutouts_relearn_v1/index.json --cutout-set relearn_v1 --root data/archive --out data/processed/rpc_synth_relearn_v1/catalog/cutouts_relearn_v1.json`. Đạt khi `issues` rỗng.

- [ ] **Step 4: Ghi `reviews.json`**
  - Bắt đầu từ `reviews.json` của v1.
  - Cutout mới có `source` đã có trong `rpc_synth_v2_v1/catalog/cutouts.json` thì `reject`, lý do `duplicate_source`, `reviewer: "rule"`.
  - Phần còn lại duyệt bằng mắt trên trang lưới của `python -m tools.synth_v2 cutout-sheets ...`. Mảnh khay, mask hỏng, hai mặt hộp dính nhau thì `reject`, theo tiêu chí M0.
  - **Dừng: người dùng xem gallery cutout bị loại và xác nhận.**

- [ ] **Step 5: Duyệt góc chụp, viết `skus.json` mới**
  - Chép `skus.json` của v1.
  - Với 50 SKU, xem trang `python -m tools.synth_v2 review-sheets` (val | camera 2 | camera 0) rồi chốt `shape_group`/`camera_weights` theo Q5(b): góc nào thấy nhãn thì dùng.
  - Đề xuất sẵn:
    - 119 và 121_canned_food → `upright` (`{"0": 0.99, "2": 0.01}`);
    - 123 và 130_chocolate → `mixed` (`{"0": 0.5, "2": 0.5}`);
    - 134 và 135_gum: chốt sau khi xem trang.
  - Mỗi SKU đổi ghi `proposed_by` / `accepted_by`.
  - **Dừng: người dùng chốt bảng thay đổi.**

- [ ] **Step 6: Chốt catalog và kiểm đủ mặt**
  - `python -m tools.synth_v2 finalize --catalogs <6 catalog v1> data/processed/rpc_synth_relearn_v1/catalog/cutouts_relearn_v1.json --reviews .../reviews.json --skus .../skus.json --out-dir data/processed/rpc_synth_relearn_v1/catalog --accepted-by "user (chat <ngày>)"`.
  - `python -m tools.synth_v2 coverage --config configs/data/synth_relearn.json --tasks 6 7` (config viết ở Task 10 Step 1; làm step đó trước nếu cần).
  - Đạt khi mọi SKU có ≥ 8 cutout và cả 50 SKU `ok` trong `coverage.json` (đủ kiểu đặt × camera, khe cam0 ≤ 18°).
  - SKU nào không đạt vì RPC thiếu ảnh hoặc vì mọi ảnh của một góc bị loại thì ghi rõ, rồi hỏi người dùng.

- [ ] **Step 7: Commit notebook** (dữ liệu nằm trong `data/`, không commit)

```bash
git add notebooks/data_preprocessing/06_synthetic_relearn.ipynb
git commit -m "data(relearn): cutouts, camera choices and face coverage of the 50 SKUs"
```

---

### Task 10: Dữ liệu (b) — sinh release học lại và đóng gói

**Files:**
- Create: `configs/data/synth_relearn.json`, `scripts/pack_relearn_data.sh`
- Modify: `notebooks/data_preprocessing/06_synthetic_relearn.ipynb` (thêm cell)

**Interfaces:**
- Consumes: catalog và `skus.json` (Task 9); generator (Task 5, 6); `task_config_relearn.json` (Task 3).
- Produces:
  - `data/processed/rpc_synth_relearn_v1/`, gồm:
    - `tasks/synth_v2/train_task_{6,7}.json` và `train_task_{6,7}_gt_full.json`;
    - `manifest.json`, `task_config.json`, `verify_report.json`;
    - `images/task_{6,7}/`.
  - Gói `data/packages/relearn_<ngày>.tar`.

- [ ] **Step 1: Viết `configs/data/synth_relearn.json`**
  - Chép `synth_v2.json`.
  - Đổi các trường:
    - `name: "rpc_synth_relearn"`, `version: "v1"`, `seed: 20261031`;
    - `task_config: "configs/relearn/task_config_relearn.json"`;
    - `catalog` và `skus` trỏ tới `rpc_synth_relearn_v1/catalog`;
    - `images_per_task: {"6": 12330, "7": 12330}`;
    - trong `layout`: `"view_balance": true`, `"quota_tolerance": 0.05`.
  - Còn lại giữ nguyên: `scale`, `photometric`, `design_val`, `backgrounds`, `image_root`, và các khoá khác của `layout`.

- [ ] **Step 2: Scales, plan, generate, export, verify**

Run (lần lượt):
1. `python -m tools.synth_v2 scales --config configs/data/synth_relearn.json`
2. `... plan --config ... --out-dir data/processed/rpc_synth_relearn_v1 --tasks 6 7`
3. `... generate ... --workers 16 --tasks 6 7` (khoảng 33 phút)
4. `... export ...`
5. `... verify ...`

Expected:
- generate không có lỗi;
- verify không có failure, gồm cả `sku_quota` ±5% và `view_balance`;
- tỉ lệ vật mới 40–60%.

In thêm số vật có nhãn mỗi SKU, so với round 1 (khoảng 3.070–3.280).

- [ ] **Step 3: Báo cáo phủ mặt so với ảnh quầy thật (không chặn).** Chạy lại `runs/analysis_r1/d7_faces.py` trên release mới: thêm tham số đường dẫn release và danh sách task. In bảng trước/sau cho 50 SKU: tỉ lệ vật val rơi vào mặt hoặc góc không dùng, và độ giống với cutout đã dùng.

- [ ] **Step 4: Duyệt gallery** — `python -m tools.synth_v2 gallery --config configs/data/synth_relearn.json --out-dir data/processed/rpc_synth_relearn_v1 --dest runs/data_preprocessing/synthetic_relearn/gallery`. **Dừng: người dùng xem khoảng 20 ảnh mỗi task và xác nhận.**

- [ ] **Step 5: Viết `scripts/pack_relearn_data.sh`**
  - Theo đúng mẫu receipts của `pack_full_syn_data.sh` (SHA256SUMS, PACKAGE.txt, tar.md5).
  - `PATHS` gồm:
    - release: `images`, `tasks`, `manifest.json`, `task_config.json`, `verify_report.json`;
    - từ seed `data/training/full-syn-r1/`: `plan.json`, `stages/`;
    - `runs/EXP-FULL-SYN/task_5/{task_final.pth, pred_val.npz, run_info.json}`;
    - `runs/EXP-FULL-SYN-FSA/task_1/hf_model`.
  - Không cần task 1–4 hay policy của seed (Task 7: chỉ chấm stage 5).
  - Eval set, ảnh val/test và pretrained không đóng gói lại. Nếu là máy vast mới, giải nén thêm gói `full_syn_rpc_synth_v2_v1_<ngày>.tar` của lần M5. Ghi điều này vào runbook.

- [ ] **Step 6: Đóng gói và kiểm**

Run: `bash scripts/pack_relearn_data.sh`, rồi giải nén thử vào thư mục tạm và chạy `sha256sum -c`.
Expected: mọi file `OK`. In kích thước và md5 để người dùng upload.

- [ ] **Step 7: Commit**

```bash
git add configs/data/synth_relearn.json scripts/pack_relearn_data.sh notebooks/data_preprocessing/06_synthetic_relearn.ipynb
git commit -m "data(relearn): synthetic tasks 6-7, 2 x 12,330 images with every face, and their package"
```

---

### Task 11: Notebook chạy, tài liệu, chạy thử dry

**Files:**
- Create: `notebooks/training/07_full_syn_relearn.ipynb`, `docs/project/HANDOFF_RELEARN_<ngày>.md`
- Modify: `notebooks/training/README.md`, `docs/project/TRAINING_NOTEBOOK_RUNBOOK.md` (mục 7.2), `docs/project/full-syn-r1-analysis-2026-10-29.md` (mục 9)

- [ ] **Step 1: Notebook 07**, theo cấu trúc notebook 06:
  1. setup: `relearn.prepare(settings)`, `dry_run` lấy từ settings;
  2. `stage_expand`;
  3. `stage_task(plan, 6)`: bảng hai mode so với round 1, cổng sau task 6. Cell hỏng (`GateStop`) nghĩa là dừng ở đây;
  4. `stage_task(plan, 7)`: như trên, cổng cuối; thêm recall/precision của 50 SKU và số SKU dưới 85% (trước là 32);
  5. cell test, có dòng `confirm=False` để người dùng tự đổi.

- [ ] **Step 2: Chạy notebook ở dry run trên máy local** — `python tools/execute_training_notebook.py notebooks/training/07_full_syn_relearn.ipynb` (settings dry). Expected: chạy hết, gate dry có `enforced: false`.

- [ ] **Step 3: Tài liệu**
  - Mục 9 của tài liệu phân tích gồm:
    - các phát hiện về dữ liệu ngày 30/10: lỗi góc chụp, SKU anh em, bảng 32 SKU dưới 85%, phân tích phủ mặt d7;
    - quy tắc chọn 50 SKU và chia task;
    - Q1–Q11 và các giới hạn;
    - ghi chú đã bỏ hướng B.
  - Runbook 7.2: lệnh tải gói, giải nén ở repo root, `sha256sum -c`, chạy notebook 07.
  - Handoff: trạng thái, đường dẫn, các lệnh.

- [ ] **Step 4: Chạy lại toàn bộ test** — `python -m pytest tests -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/training/07_full_syn_relearn.ipynb notebooks/training/README.md docs/project
git commit -m "docs(relearn): runbook, handoff and the data findings of 30/10"
```

---

### Task 12: Chạy trên 4090 và ghi kết quả (người dùng chạy, Claude đọc kết quả)

- [ ] **Step 1:** Người dùng checkout `feat/task6-relearn`, cài env từ lock, tải và giải nén gói ở repo root, rồi `sha256sum -c`.
- [ ] **Step 2:** Đặt `dry_run: false` trong `configs/training/full_syn_relearn.json`, rồi chạy notebook 07 trong tmux: khoảng 5,3 giờ train task 6, cộng audit và dự đoán. Nếu cổng sau task 6 ra `stop` thì dừng ở đó (sang Step 3); nếu `go` thì notebook train tiếp task 7, thêm khoảng 5,3 giờ.
- [ ] **Step 3:** Mang `data/training/full-syn-relearn/` về (zip lên Drive như lần M5). Kiểm số trong `stages/task6.json` (và `task7.json` nếu có) khớp với file dự đoán.
- [ ] **Step 4:** Ghi kết quả và quyết định vào mục 9 của tài liệu phân tích và handoff; commit.
- [ ] **Step 5:** Chỉ khi `go` **và** người dùng xác nhận đây là cấu hình cuối: chạy cell test (`confirm=True`), ghi kết quả, commit.
