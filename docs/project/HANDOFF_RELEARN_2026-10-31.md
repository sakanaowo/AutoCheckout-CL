# Handoff: học lại 50 SKU ở task 6–7 — 31/10/2026

Người làm: さかな (với Claude). Tài liệu này để người tiếp theo (hoặc chính mình ở phiên sau) chạy campaign học lại trên máy 4090 mà không phải đọc lại cả cuộc trao đổi.

## 1. Đang làm gì và vì sao

- Round 1 FULL-SYN: cAcc-M2 trên val sau task 5 là 43,68%. Chạy thử M5 (mở khoá đặc trưng) ngày 30/10 ra `stop`; hướng B (800px) bị bỏ cùng ngày.
- Kiểm dữ liệu ảnh ghép (phân tích mục 9): 32 SKU có recall hoặc precision dưới 85% trên val; nguyên nhân gồm lỗi góc chụp (119/121_canned_food, gum chỉ có ảnh nắp), SKU anh em (97/98_milk), và cutout thiếu mặt hoặc thiếu góc xoay.
- **Học lại**: mở rộng model task 5 của round 1 lên 275 ô class, rồi train **task 6** và **task 7**, mỗi task 25 SKU dưới class bí danh, trên release ảnh ghép làm lại đủ mặt (`rpc_synth_relearn_v1`). Khi chấm, dự đoán bí danh đổi về label gốc, nên model sau task 7 được chấm như một model 200 SKU của round 1. Đây là biến thể cho production, không phải học tăng dần thuần như paper.
- Chi tiết: [plan](task6-relearn-plan-2026-10-30.md) (spec Q1–Q11), [phân tích mục 9](full-syn-r1-analysis-2026-10-29.md#9-dữ-liệu-ảnh-ghép-và-học-lại-50-sku-303110).

## 2. Trạng thái lúc viết

| Hạng mục | Trạng thái |
|---|---|
| Code (nhánh `feat/task6-relearn`, tách từ `feat/full-syn-m5`) | Bí danh trong `TaskConfig`, config học lại, `pdp/expand.py`, generator đủ mặt, đổi bí danh khi chấm, workflow `autocheckout/relearn.py`, notebook 07. **Chưa push**: người dùng push trước khi `git pull` trên vast |
| Test | Toàn bộ suite qua trên máy local, trừ 3 test kernel CUDA của `tests/test_pdp_f11_kernel.py` (máy local không có `CUDA_HOME`; máy 4090 có). Dry run notebook 07 chạy hết, exit 0 |
| Dữ liệu | Release `data/processed/rpc_synth_relearn_v1`: 2 × 12.330 ảnh, verify 15/15 cổng (gồm quota mỗi SKU ±5% và cân bằng mặt). Notebook [06_synthetic_relearn](../../notebooks/data_preprocessing/06_synthetic_relearn.ipynb) của data preprocessing ghi lại cả quá trình |
| Gói | `data/packages/relearn_rpc_synth_relearn_v1_20261031.tar` trên máy local: 6,8 GB, 24.679 file, md5 `d4330b997cd31e52da4df36b4f4e3e14`; đã giải nén thử và `sha256sum -c` đủ. **Chưa upload** lên Drive |
| Train | Chưa chạy (Task 12 của plan) |

## 3. Việc tiếp theo

**3.1 Ở máy local:** push nhánh, upload gói và kiểm md5 trên Drive.

```bash
git push -u origin feat/task6-relearn
rclone copy -P --drive-chunk-size 64M data/packages/relearn_rpc_synth_relearn_v1_20261031.tar drive:AutoCheckout/2026-10-31/
rclone md5sum drive:AutoCheckout/2026-10-31/relearn_rpc_synth_relearn_v1_20261031.tar   # phải là d4330b997cd31e52da4df36b4f4e3e14
```

**3.2 Trên 4090:** code, dữ liệu, môi trường, settings. Máy vast mới cần thêm gói dữ liệu của round 1 (eval set, ảnh val/test, pretrained); máy đã chạy M5 thì đã có. Giải nén cả hai ở gốc repo.

```bash
cd ~/AutoCheckout-CL && git fetch && git checkout feat/task6-relearn && git pull
rclone cat drive:AutoCheckout/2026-10-25/full_syn_rpc_synth_v2_v1_20261025.tar | tar -x -C ~/AutoCheckout-CL   # chỉ máy mới
rclone cat drive:AutoCheckout/2026-10-31/relearn_rpc_synth_relearn_v1_20261031.tar | tar -x -C ~/AutoCheckout-CL
sha256sum -c --quiet data/packages/relearn_rpc_synth_relearn_v1_20261031/SHA256SUMS && echo RELEARN_OK
ls data/processed/rpc_eval_official_v1/val_full.json data/pretrained/hf-cache data/archive/val2019 | head -3
```

Môi trường như [handoff M5](HANDOFF_M5_PILOT_2026-10-30.md), mục 3.3. Settings:

```bash
cp configs/training/full_syn_relearn.json data/training/settings_full_syn_relearn.json
sed -i 's/"dry_run": true/"dry_run": false/' data/training/settings_full_syn_relearn.json
grep -E 'dry_run|campaign_id|seed_campaign|release_root|task_config' data/training/settings_full_syn_relearn.json
```

Phải thấy `"dry_run": false`, `"campaign_id": "full-syn-relearn"`, `"seed_campaign": "data/training/full-syn-r1"`, `"release_root": "data/processed/rpc_synth_relearn_v1"` và `"task_config": "configs/relearn/task_config_relearn.json"`. `prepare` từ chối nếu công thức train khác round 1 hoặc `task_config.json` của release khác config học lại; bước mở rộng từ chối nếu `task_5/task_final.pth` của seed đổi md5 so với lúc `prepare`.

**3.3 Chạy** trong tmux:

```bash
tmux new -s relearn zsh            # Ctrl+B rồi D để thoát ra; tmux attach -t relearn để vào lại
conda activate pdp && cd ~/AutoCheckout-CL
python -m tools.execute_training_notebook --notebook notebooks/training/07_full_syn_relearn.ipynb \
  --settings data/training/settings_full_syn_relearn.json --live-notebook notebooks/training/07_full_syn_relearn.ipynb
```

Khoảng 2 × 5,3 giờ train (như một task của round 1), cộng audit nhãn giả và chấm val hai cách đổi bí danh sau mỗi task. Bị ngắt thì chạy lại đúng lệnh: bước đã xong được bỏ qua, task dở chạy tiếp từ `last.ckpt`.

## 4. Trong lúc chạy, nên xem gì

- `data/training/full-syn-relearn/runs/EXP-FULL-SYN-RL/task_6/train.log`: model phải có `n_classes` 275 và nạp được `task_5/task_final.pth` đã mở rộng. Checkpoint chưa mở rộng sẽ báo lỗi rõ ràng, không âm thầm khởi tạo lại.
- `step_metrics.jsonl` cùng thư mục: `peak_allocated_gib` nên quanh mức task 5 của round 1 (8,9 GiB ở batch 8); `seconds_per_optimizer_step` khoảng 0,78.
- Sau mỗi task, notebook in dòng `GATE relearn task <t>: GO/STOP - …` và bảng hai cách đổi bí danh (`merge`, `alias_only`) cạnh task 5 của round 1.

## 5. Khi chạy xong

| Kết quả | Điều kiện (val) | Việc tiếp |
|---|---|---|
| `STOP` sau task 6 | cAcc-M2 < 0,4533 hoặc AP50 của 150 SKU không học lại giảm > 1 điểm | Không train task 7, không chấm test. Lấy kết quả về, nhóm bàn hướng khác |
| `STOP` sau task 7 | cAcc-M2 < 0,4668 hoặc AP50 của 150 SKU giảm > 1 điểm | Không chấm test. Xem bảng recall/precision của 50 SKU để biết SKU nào chưa học lại được |
| `GO` sau task 7 | cAcc-M2 ≥ 0,4668 và AP50 của 150 SKU giảm ≤ 1 điểm | Người dùng xác nhận đây là cấu hình cuối, sửa `CONFIRM_TEST = True` trong notebook 07 rồi chạy lại lệnh ở 3.3: chỉ cell test chạy, test chấm đúng một lần |

Dù kết quả thế nào, xem số SKU có recall hoặc precision dưới 85% sau task 7 (trước khi học lại: 32 trên 200), và các cặp nhầm nhiều nhất cạnh round 1.

**Lấy kết quả về máy local trước khi trả máy:**

```bash
# ở máy local
rsync -a --partial --info=progress2 -e 'ssh -p PORT' \
  root@HOST:~/AutoCheckout-CL/data/training/full-syn-relearn/ data/training/full-syn-relearn/
rsync -a -e 'ssh -p PORT' root@HOST:~/AutoCheckout-CL/notebooks/training/07_full_syn_relearn.ipynb notebooks/training/
# symlink mang đường dẫn tuyệt đối của vast; nối lại về bản local
ln -sfn "$PWD/data/training/full-syn-r1/runs/EXP-FULL-SYN-FSA" data/training/full-syn-relearn/runs/EXP-FULL-SYN-FSA
for l in data/training/full-syn-relearn/inputs/tasks/*.json; do
  t=$(readlink "$l"); ln -sfn "$PWD/data/processed/${t#*/data/processed/}" "$l"
done
```

Trước khi commit notebook 07, bỏ các dòng thanh tiến độ tqdm khỏi output (runbook mục 7), rồi ghi kết quả vào phân tích mục 9 và đánh dấu Task 12 của plan.

## 6. Việc còn mở

- **123 và 130_chocolate:** sau khi làm lại dữ liệu, phép đo phủ mặt d7 vẫn thấy 82,5% và 72,5% vật trên val gần với góc không dùng, trong khi lúc duyệt bảng camera (31/10) trang so sánh cho thấy val chỉ lộ mặt trên có nhãn. Kết quả task 6–7 của hai SKU này sẽ cho biết bên nào đúng.
- **Ảnh xiên cam1/cam3** chưa dùng (renderer chưa có `object_gain` cho hai camera này): ứng viên cho vòng sau.
- **Học lại 50 SKU không đủ cho production:** để đạt 80% cAcc, mỗi SKU cần khoảng 98%, mà 179/200 SKU đang dưới mức đó (phân tích 9.4).
- **Dọn đĩa ở máy local:** file tar 6,8 GB trong `data/packages/` xoá được khi md5 trên Drive đã khớp; giữ thư mục `SHA256SUMS` đi kèm.

## 7. Bẫy đã gặp

- Như [handoff M5](HANDOFF_M5_PILOT_2026-10-30.md), mục 7: không dùng `rclone cat --progress` khi pipe vào `tar`; giải nén ở gốc repo, không phải trong `data/`.
- Đổi bất kỳ setting nào trong một campaign đã có sẽ bị từ chối; muốn đổi thì dùng `campaign_id` mới.
- Chạy lại toàn bộ notebook 06 của data preprocessing sẽ ghi lại `catalog/cutouts.json` (trường `created_at`), làm lệch sha mà plan receipt của release đã ghim. Mục 5–7 của notebook đó chạy riêng vì lý do này.

## 8. File chính

| File | Vai trò |
|---|---|
| [task6-relearn-plan-2026-10-30.md](task6-relearn-plan-2026-10-30.md) | Spec Q1–Q11, các task và giới hạn |
| [full-syn-r1-analysis-2026-10-29.md](full-syn-r1-analysis-2026-10-29.md) mục 9 | Phát hiện về dữ liệu, 32 SKU dưới 85%, phủ mặt, chọn và chia 50 SKU |
| [TRAINING_NOTEBOOK_RUNBOOK.md](TRAINING_NOTEBOOK_RUNBOOK.md) mục 7.2 | Lệnh tải gói, giải nén, kiểm và chạy |
| [07_full_syn_relearn.ipynb](../../notebooks/training/07_full_syn_relearn.ipynb) | Notebook chạy, cũng là log |
| `configs/training/full_syn_relearn.json` | Settings mẫu |
| `configs/relearn/selection.json`, `configs/relearn/task_config_relearn.json` | 50 SKU và config 275 class |
| `configs/exp/native/EXP-FULL-SYN-RL.sh` | Cờ train: như round 1, thêm `--n_classes 275` |
| `configs/data/synth_relearn.json`, `scripts/pack_relearn_data.sh` | Config sinh release và script đóng gói |
| `autocheckout/relearn.py` | `prepare`, `stage_expand`, `stage_task`, `sku_accuracy`, `stage_final_test`, `relearn_verdict` |
| `pdp/expand.py` | Mở rộng checkpoint, giữ nguyên output đã học |
