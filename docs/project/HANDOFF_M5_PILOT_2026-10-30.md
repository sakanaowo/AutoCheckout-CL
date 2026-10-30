# Handoff: chạy thử M5 trên task 2 — 30/10/2026

Người làm: さかな (với Claude). Tài liệu này để người tiếp theo (hoặc chính mình ở phiên sau) chạy tiếp lần thử M5 trên máy 4090 mới mà không phải đọc lại cả cuộc trao đổi.

## 1. Đang làm gì và vì sao

- Round 1 FULL-SYN: cAcc-M2 trên val tụt từ 79% (task 1) xuống 44% (task 5). Lỗi chính là **nhầm giữa các SKU anh em** (cùng dòng, khác vị) được học ở đợt sau. 6 cách sửa không cần train đều thất bại, nên phải train lại. Chi tiết: [phân tích round 1](full-syn-r1-analysis-2026-10-29.md), mục 3.6 và 7.
- Giả thuyết: PDP khoá backbone, encoder và decoder sau FSA, mà FSA chỉ học SKU task 1, nên các SKU học sau không tách được.
- **M5** (hướng A): từ task 2, cho tầng cuối backbone (`stages_3`, 27,5M tham số) và encoder (4,5M) học tiếp ở `lr_old` = 1e-5, kèm loss giữ output encoder gần với model thầy (`--feature_kd 1`).
- Lệch so với paper: phần dùng chung bị sửa khi học task mới tăng từ 2,4% lên khoảng 27% model. Nhóm chấp nhận được nếu không quên SKU cũ (mục 8.1–8.2 của tài liệu phân tích).
- **Lần này chỉ chạy task 2**, dùng lại FSA và task 1 của round 1, đổi đúng một biến so với round 1 (learning rate, số epoch, batch, 640px giữ nguyên).

## 2. Trạng thái lúc viết

| Hạng mục | Trạng thái |
|---|---|
| Code (nhánh `feat/full-syn-m5`, tách từ `feat/synthetic-data-v2` tại `6d2cc96`) | `2c0ecf6` (cờ pdp), `61b919b` (workflow + notebook 06), `6d2cc96` (tài liệu), `0494202` (file lock môi trường) đã có trên GitHub; commit chứa file này phải push trước khi `git pull` trên vast |
| Test | Toàn bộ suite qua trên CPU; venv dựng từ file lock cũng qua. Dry run notebook 05 rồi 06 chạy hết, exit 0 |
| Gói dữ liệu trên Drive | `drive:AutoCheckout/2026-10-25/full_syn_rpc_synth_v2_v1_20261025.tar` (26,8 GB, md5 `13066bcff65cc42047c969e127c08114`); `drive:AutoCheckout/2026-10-29/full_syn_r1_seed_20261029.tar` (1,06 GB, md5 `2ae021116a7c0124018051766ba23ca4`) |
| Máy 4090 mới | Repo ở `~/AutoCheckout-CL`, shell zsh. **Đã tải xong 2 gói.** Lúc tải lần đầu repo đang ở nhánh `main` và có một lần giải nén lỗi vào `data/` (do `--progress` trên `rclone cat`), nên phải kiểm lại theo mục 3 |
| Train | Chưa chạy |

## 3. Việc tiếp theo trên 4090

Mọi lệnh chạy từ `~/AutoCheckout-CL`.

**3.1 Code đúng nhánh.**

```bash
cd ~/AutoCheckout-CL
git fetch && git checkout feat/full-syn-m5 && git pull
git log --oneline -6          # phải thấy 0494202 và commit handoff này
```

**3.2 Dữ liệu đúng chỗ.** Đường dẫn trong tar bắt đầu bằng `data/`, nên phải giải nén ở gốc repo, không phải trong `data/`.

```bash
ls data/data 2>/dev/null && echo "giải nén lỗi còn sót: rm -rf data/data"
ls data/processed/rpc_synth_v2_v1/images | head -3        # task_1 task_2 ...
ls data/processed/rpc_eval_official_v1/val_full.json data/archive/val2019 | head -2
ls data/pretrained/hf-cache
ls data/training/full-syn-r1/plan.json data/training/full-syn-r1/runs/EXP-FULL-SYN/task_1/task_final.pth
sha256sum -c --quiet data/packages/full_syn_r1_seed_20261029/SHA256SUMS && echo SEED_OK
```

Nếu thiếu thư mục nào thì giải nén lại gói tương ứng ở gốc repo (`tar -xf <file>.tar -C ~/AutoCheckout-CL`, hoặc `rclone cat … | pv -s <bytes> | tar -x -C ~/AutoCheckout-CL`). Gói 26 GB được notebook tự kiểm 300 file ngẫu nhiên và mọi ảnh mà nhãn liệt kê. Kiểm đủ 101.681 file thì chạy `sha256sum -c --quiet data/packages/full_syn_rpc_synth_v2_v1_20261025/SHA256SUMS` (mất khá lâu).

Giải nén xong thì xoá file tar đã tải (nếu dùng `rclone copy`) để lấy lại đĩa.

**3.3 Môi trường.**

```bash
conda env create -f configs/environment/pdp.yml && conda activate pdp
python -m pip install -r configs/environment/requirements-lock-cu121.txt
python -m pip install --no-deps -e . && python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"   # 2.2.2+cu121, RTX 4090
```

**3.4 Settings.**

```bash
cp configs/training/full_syn_m5_pilot.json data/training/settings_full_syn_m5_pilot.json
sed -i 's/"dry_run": true/"dry_run": false/' data/training/settings_full_syn_m5_pilot.json
grep -E 'dry_run|campaign_id|pdp_experiment|seed_campaign' data/training/settings_full_syn_m5_pilot.json
```

Phải thấy `"dry_run": false`, `"campaign_id": "full-syn-m5-pilot"`, `"pdp_experiment": "EXP-FULL-SYN-M5"` và `"seed_campaign": "data/training/full-syn-r1"`.

**3.5 Chạy.** zsh không giữ tiến trình khi SSH đứt, nên chạy trong tmux (hoặc nohup):

```bash
tmux new -s m5 zsh            # Ctrl+B rồi D để thoát ra; tmux attach -t m5 để vào lại
conda activate pdp && cd ~/AutoCheckout-CL
python -m tools.execute_training_notebook --notebook notebooks/training/06_full_syn_m5_pilot.ipynb \
  --settings data/training/settings_full_syn_m5_pilot.json --live-notebook notebooks/training/06_full_syn_m5_pilot.ipynb
```

Thời gian khoảng 6 giờ: audit nhãn giả vài phút, train task 2 khoảng 5,6 giờ (round 1 mất 5,36 giờ; M5 chậm hơn khoảng 4,5%, đo trên 3060), rồi chấm val. Bị ngắt thì chạy lại đúng lệnh trên: bước đã xong được bỏ qua, task dở chạy tiếp từ `last.ckpt`.

## 4. Trong lúc chạy, nên xem gì

File `data/training/full-syn-m5-pilot/runs/EXP-FULL-SYN-M5/task_2/step_metrics.jsonl`, mỗi dòng một bước:

```bash
tail -1 data/training/full-syn-m5-pilot/runs/EXP-FULL-SYN-M5/task_2/step_metrics.jsonl | python -m json.tool
```

- `peak_allocated_gib`: ước khoảng 11–12 GiB ở batch 8; vượt 22 GiB thì nguy cơ hết bộ nhớ. Nếu hết bộ nhớ thật, dùng campaign mới với `"batch_size": 4, "effective_batch": 8` (settings trong campaign cũ bị khoá).
- `losses.FKD`: khoảng 0,008 ngay từ đầu là nhiễu do dropout. Nếu tăng dần rõ rệt thì đặc trưng đang trôi khỏi model thầy.
- `seconds_per_optimizer_step`: round 1 khoảng 0,76 giây/bước ở task 2 (chỉ bước train); M5 nhỉnh hơn một chút.
- `train.log` có dòng `Trainable parameters: 64.18M of 130.53M`. Nếu là 32.18M thì M5 không được bật.

## 5. Khi chạy xong

Cell cuối của notebook in quyết định (`full_syn.pilot_verdict`), so với task 2 round 1 (val cAcc-M2 73,64%, AP50 SKU task 1 là 94,96%):

| Quyết định | Điều kiện | Việc tiếp |
|---|---|---|
| `GO` | cAcc-M2 tăng ≥ 3 điểm và AP50 SKU task 1 giảm ≤ 1 điểm | Viết phần làm lại prototype của SKU task 1 (mục 6), rồi chạy task 3–5 |
| `STOP` | cAcc-M2 tăng < 3 điểm | Giả thuyết "đặc trưng bị khoá" sai. Chuyển sang hướng B: train cả vòng ở 800px, giữ khoá như paper (khoảng 50 giờ) |
| `RETRY` | Tăng nhưng SKU task 1 giảm > 1 điểm | Thử lại một lần với `feature_kd` lớn hơn (ví dụ 10). Giá trị nằm trong `configs/exp/native/EXP-FULL-SYN-M5.sh`; chạy với `campaign_id` mới để giữ lịch sử, và ghi giá trị đã dùng |

Dù kết quả thế nào, cũng xem bảng nhầm lẫn mà notebook in ra cạnh số của round 1 (nhãn số; ở task 2 round 1, cặp nhiều nhất là 41→59 với 32 lần). Các cặp SKU anh em có giảm không là dấu hiệu trực tiếp nhất của giả thuyết.

**Lấy kết quả về máy local trước khi trả máy:**

```bash
# ở máy local
rsync -a --partial --info=progress2 -e 'ssh -p PORT' \
  root@HOST:~/AutoCheckout-CL/data/training/full-syn-m5-pilot/ data/training/full-syn-m5-pilot/
rsync -a -e 'ssh -p PORT' root@HOST:~/AutoCheckout-CL/notebooks/training/06_full_syn_m5_pilot.ipynb notebooks/training/
# hai symlink sang round 1 mang đường dẫn tuyệt đối của vast; nối lại về bản local
ln -sfn "$PWD/data/training/full-syn-r1/runs/EXP-FULL-SYN-FSA" data/training/full-syn-m5-pilot/runs/EXP-FULL-SYN-FSA
ln -sfn "$PWD/data/training/full-syn-r1/runs/EXP-FULL-SYN/task_1" data/training/full-syn-m5-pilot/runs/EXP-FULL-SYN-M5/task_1
```

Trước khi commit notebook 06, bỏ các dòng thanh tiến độ tqdm khỏi output như đã làm ở round 1 (runbook mục 7), rồi ghi kết quả vào mục 8 của tài liệu phân tích.

## 6. Việc còn mở

- **Làm lại prototype của SKU task 1 (chặn task 3–5).** Prototype của mỗi SKU được tính bằng đặc trưng lúc SKU đó được học. Từ task 3, model thầy đã có encoder bị sửa, nên so đặc trưng mới với prototype cũ là sai. `stage_tasks` từ chối M5 quá task 2 (`M5_LAST_TASK` trong `autocheckout/full_syn.py`). Hướng làm: ở epoch cuối của mỗi task, cập nhật lại prototype của SKU cũ từ các box nhãn giả đã xác minh, bằng đặc trưng hiện tại. Hiện code chỉ cập nhật SKU của task đang học (`cls_label >= PREV_INTRODUCED_CLS` trong `pdp/engine.py`). Chỉ làm khi lần thử ra `GO`.
- **Symlink tuyệt đối.** `stage_seed` tạo symlink bằng đường dẫn tuyệt đối, nên phải nối lại sau khi rsync. Có thể đổi sang đường dẫn tương đối.
- **R1 và R4** (giảm learning rate cuối task, bớt epoch thừa) chưa ghép, cố ý để lần thử chỉ đổi một biến.
- **Dọn đĩa ở máy local:** hai file tar trong `data/packages/` xoá được khi md5 trên Drive đã khớp. Giữ các thư mục `SHA256SUMS` đi kèm. `data/cache/kaggle/retail-product-checkout-dataset.zip` (27 GB) chưa xoá, vì có thể chứa ảnh camera 1 và 3 chưa giải nén.

## 7. Bẫy đã gặp

- `rclone cat … --progress | tar -x` làm hỏng tar: thanh tiến độ in ra stdout và trộn vào dữ liệu. Muốn xem tiến độ thì dùng `pv`, hoặc `--stats 10s --stats-one-line -v`, vì cờ này in ra stderr.
- `tar -x -C ~/AutoCheckout-CL/data/` sai thư mục: ra `data/data/...`.
- Tải bằng `rclone cat` chỉ khoảng 11 MB/s (một kết nối, khoảng 40 phút cho 26,8 GB). `rclone copy --multi-thread-streams 8` nhanh hơn, nhưng cần thêm 26 GB đĩa tạm.
- Đổi bất kỳ setting nào trong một campaign đã có sẽ bị từ chối; muốn đổi thì dùng `campaign_id` mới.
- Test chỉ chấm một lần, sau task 5. Mọi lựa chọn đều dựa trên val; code chặn chấm test khi chưa đủ 5 task.
- Trên RTX 3060 (12 GB), M5 ở batch 8 hết bộ nhớ. Đo trên máy local thì dùng batch 2.

## 8. File chính

| File | Vai trò |
|---|---|
| [full-syn-r1-analysis-2026-10-29.md](full-syn-r1-analysis-2026-10-29.md) | Phân tích round 1, các hướng train lại (mục 7), M5 (mục 8) |
| [TRAINING_NOTEBOOK_RUNBOOK.md](TRAINING_NOTEBOOK_RUNBOOK.md) mục 7 và 7.1 | Setup vast, chạy, lấy kết quả |
| [06_full_syn_m5_pilot.ipynb](../../notebooks/training/06_full_syn_m5_pilot.ipynb) | Notebook chạy thử, cũng là log |
| `configs/training/full_syn_m5_pilot.json` | Settings mẫu |
| `configs/exp/native/EXP-FULL-SYN-M5.sh` | Cờ M5: `--tune_after_task1 stages_3,encoder --feature_kd 1` |
| `autocheckout/full_syn.py` | `stage_seed`, `stage_tasks(plan, last=2)`, `pilot_verdict`, `M5_LAST_TASK` |
| `pdp/engine.py`, `pdp/main.py` | Mở khoá theo tên từ task 2; loss `loss_feature_kd` (ghi là `FKD` trong `step_metrics.jsonl`) |
| `configs/environment/requirements-lock-cu121.txt` | Bản chụp môi trường đã chạy |
| `tests/test_pdp_m5_tune_features.py`, `tests/test_full_syn.py` | Test của M5 |
