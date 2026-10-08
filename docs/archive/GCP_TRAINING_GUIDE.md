# Hướng dẫn máy ảo GCP cho AutoCheckout-CL

> Tài liệu lịch sử của hướng GCP tháng 09/2026. Theo làm rõ ngày 07/10, nhánh `investigation` train trên **RTX 4090 Vast.ai** theo [bàn giao](../project/AGENT_HANDOFF.md) và [kế hoạch mới](../project/IMPLEMENTATION_PLAN.md); không vận hành VM theo hướng dẫn này. Scripts/configs kế thừa đã được khôi phục; trạng thái VM GCP chưa được kiểm tra lại.

Tài liệu này hướng dẫn vận hành máy ảo `auto-cl` cho project AutoCheckout-CL (PDP trên bộ RPC). Các việc cần làm và lý do nằm trong [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN_2026-09.md); tài liệu này chỉ ghi cách thao tác.

Một số lệnh chỉ dùng được sau khi task tương ứng trong plan đã làm xong; các lệnh đó được ghi chú *(cần task X)*.

---

## 0. Thông tin nhanh

Thông tin dưới đây được kiểm tra bằng `gcloud` và SSH ngày 28/09/2026.

| Mục | Giá trị |
|---|---|
| Repo | https://github.com/AnNguyen05092004/AutoCheckout-CL (public, QĐ-7 chốt 28/09) |
| Project ID | `project-95a0d104-9d0f-4aa1-ba0` |
| Tên VM / zone | `auto-cl` / **`us-central1-c`** |
| Loại máy | `g2-standard-4` (4 vCPU, 16 GB RAM, trong VM thấy 15 GiB) |
| GPU | 1 × **NVIDIA L4**, 23 GB dùng được, compute capability 8.9 |
| Kiểu cấp phát | **STANDARD (on-demand)**, khoảng 18.400 VND/giờ. Chuyển sang Spot thì còn khoảng 11.100 VND/giờ (mục 2.5) |
| Ổ đĩa | 1 ổ boot 100 GB `pd-balanced` (NVMe), còn trống khoảng 80 GB, tốc độ ghi đo được khoảng 178 MB/s. Có snapshot tự động hằng ngày, giữ 14 ngày. Không có ổ dữ liệu riêng |
| Image | `common-cu129-ubuntu-2204-nvidia-580-v20260909`: Ubuntu 22.04.5, driver 580.178.04, CUDA 12.9 (`/usr/local/cuda`, có `nvcc`). **Không cài sẵn PyTorch** |
| Python | 3.10.12 (`/usr/bin/python3`). **Chưa có** `python3.10-venv` và `unzip` (cài ở mục 5.2) |
| User khi SSH bằng `gcloud` từ Mac | `an` |

Quota của project chỉ cho **1 GPU chạy cùng lúc**. Muốn bật VM `anmetarayban` (V100) thì phải tắt `auto-cl` trước.

Biến đường dẫn dùng trong tài liệu (nên thêm vào `~/.bashrc` trên VM):

```bash
export PROJ=~/AutoCheckout-CL        # code (git clone từ GitHub, mục 5.1)
export DATA=/data/rpc                # dữ liệu RPC
export RUNS=/data/runs               # kết quả train
export VENV=~/venvs/pdp              # môi trường Python (scripts/setup_vm.sh)
```

Để gõ lệnh ngắn hơn trên Mac (mảng dùng được cả với zsh lẫn bash; nếu gộp tên VM và các cờ vào một biến chuỗi thì zsh không tách chúng ra được):

```bash
GC=(--zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0)   # dùng: gcloud compute ... auto-cl "${GC[@]}"
```

---

## 1. Nguyên tắc

1. **Dataset chỉ tải và xử lý trên VM**, không tải về Mac. Từ VM về Mac chỉ lấy kết quả (chỉ số, log, file dự đoán nhỏ).
2. Không commit `kaggle.json`, dữ liệu, checkpoint (`*.pth`, `*.ckpt`), file dự đoán (`*.npz`) lên repo. Repo để **public**.
3. Không chạy `sudo do-release-upgrade`: giữ Ubuntu 22.04 để không làm hỏng driver và CUDA.
4. Job dài luôn chạy trong `tmux`.
5. **Không dùng thì tắt VM.** VM on-demand vẫn tính tiền khi GPU đứng yên.
6. **Credit hết hạn ngày 24/10/2026.** Trước ngày này phải tải kết quả về và xóa VM và ổ. Nếu tài khoản đã nâng cấp lên trả phí, sau ngày đó phí sẽ tính vào phương thức thanh toán.

---

## 2. Bật, SSH và tắt VM (từ Terminal trên Mac)

### 2.1 Bật VM

```bash
gcloud compute instances start auto-cl "${GC[@]}"
```

### 2.2 SSH

```bash
gcloud compute ssh auto-cl "${GC[@]}"
```

Có thể SSH từ Console (Compute Engine → VM instances → `auto-cl` → SSH). Lưu ý:

- SSH trên trình duyệt có thể đăng nhập bằng user khác (thường là tên tài khoản Google), tức là thư mục home khác.
- Dữ liệu và kết quả để ở `/data` nên không phụ thuộc vào user; code và venv ở home của user `an`.

### 2.3 Tắt VM

Thiếu `--preemptible` thì API báo `preemptible=false and provisioning_model=SPOT is contradicting` (gặp ngày 29/09).

```bash
gcloud compute instances stop auto-cl "${GC[@]}"
```

Hoặc trong VM: `sudo shutdown -h now`. Khi VM tắt, file trên ổ vẫn còn; chỉ còn tính tiền ổ (khoảng 261 nghìn VND/tháng cho 100 GB `pd-balanced`).

### 2.4 Xem trạng thái

```bash
gcloud compute instances describe auto-cl "${GC[@]}" --format="value(status,scheduling.provisioningModel)"
```

### 2.5 (Khuyên dùng) Chuyển sang Spot để giảm khoảng 40% chi phí

Nên chuyển khi code resume đã chạy được (task R1/R2), vì Spot có thể bị Google thu hồi giữa chừng. Lệnh chỉ chạy được khi VM **đang tắt**:

```bash
gcloud compute instances stop auto-cl "${GC[@]}"
gcloud compute instances set-scheduling auto-cl "${GC[@]}" \
  --provisioning-model=SPOT --preemptible --instance-termination-action=STOP --no-restart-on-failure
# Quay lại on-demand: --provisioning-model=STANDARD --no-preemptible --clear-instance-termination-action --restart-on-failure
gcloud compute instances start auto-cl "${GC[@]}"
```

Thiếu `--preemptible` thì API từ chối với lỗi `preemptible=false and provisioning_model=SPOT is contradicting` (gặp ngày 29/09).

Lưu ý khi chạy Spot (chi tiết ở mục 3.6 của plan):

- Tốc độ không đổi.
- Google có thể tắt VM bất cứ lúc nào, chỉ báo trước tối đa 30 giây, và VM **không tự bật lại**.
- Có lúc hết GPU Spot nên không bật được VM. Nếu đang gấp, chuyển tạm về on-demand (VM phải đang tắt):

```bash
gcloud compute instances set-scheduling auto-cl "${GC[@]}" --provisioning-model=STANDARD
```

Khi đã chạy Spot, xem VM có bị thu hồi không:

```bash
gcloud compute operations list --project=project-95a0d104-9d0f-4aa1-ba0 \
  --filter="operationType=compute.instances.preempted" --limit=5
```

---

## 3. Kiểm tra lần đầu trên VM

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
# mong đợi: NVIDIA L4, 23034 MiB, 580.178.04, 8.9
/usr/local/cuda/bin/nvcc --version | tail -1
python3 --version       # 3.10.12
df -h /                 # còn khoảng 80 GB
```

---

## 4. Ổ đĩa và thư mục dữ liệu

VM chỉ có một ổ 100 GB. Tạo thư mục dữ liệu dùng chung một lần:

```bash
sudo mkdir -p /data/rpc /data/runs && sudo chmod 1777 /data /data/rpc /data/runs
```

Dung lượng cần (mục 3.3 của plan): khoảng 50 GB nếu **xóa file zip RPC (25,3 GB) sau khi giải nén** (`download_rpc.sh` tự xóa). 80 GB trống là đủ, nhưng phải dọn checkpoint thường xuyên.

Nếu thiếu chỗ thì tăng dung lượng ổ. Việc này không mất dữ liệu; mỗi 50 GB thêm khoảng 130 nghìn VND/tháng:

```bash
gcloud compute disks resize auto-cl --size=150GB --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0
```

Sau đó khởi động lại VM để hệ thống tự mở rộng phân vùng, rồi kiểm tra bằng `df -h /`.

---

## 5. Đưa code lên VM và cài môi trường

### 5.1 Lấy code (trên VM)

Code viết và commit trên Mac, `git push` lên GitHub. VM chỉ lấy về:

```bash
# lần đầu
git clone https://github.com/AnNguyen05092004/AutoCheckout-CL.git ~/AutoCheckout-CL
# các lần sau: đặt code trên VM đúng bằng bản trên GitHub
cd ~/AutoCheckout-CL && git fetch -q origin && git reset -q --hard origin/main && git log --oneline -1
```

- Dùng `reset --hard` thay cho `git pull`, vì các file cấu hình dữ liệu sinh trên VM (mục 6.2) sau đó được commit từ Mac. Nếu dùng `git pull`, git sẽ từ chối ghi đè các file untracked trùng tên. `reset --hard` ghi đè đúng các file đó (nội dung giống nhau), và giữ nguyên mọi file untracked khác.
- Không sửa code trực tiếp trên VM; mọi sửa đổi đều làm trên Mac.
- Commit và diff của mỗi lần chạy được ghi vào `run_info.json` (R3).

### 5.2 Cài môi trường (chạy trên VM, một lần)

```bash
bash ~/AutoCheckout-CL/scripts/setup_vm.sh
```

Script làm các việc sau, và chạy lại lần nữa cũng không sao:

1. Tạo `/data/rpc` và `/data/runs`.
2. Cài `python3.10-venv`, `unzip`.
3. Tạo venv `~/venvs/pdp` với `torch 2.2.2 + cu121`, các thư viện đã khóa phiên bản trong `requirements.txt`, `kaggle`, và repo ở chế độ editable (`pip install -e .`).
4. Build kernel CUDA cho L4 (`TORCH_CUDA_ARCH_LIST=8.9`), in thời gian build.
5. Chạy toàn bộ test. Riêng test so sánh kernel với bản PyTorch (`tests/test_pdp_f11_kernel.py`) chỉ chạy được trên VM.

Kết quả mong đợi:
- dòng `2.2.2+cu121 True NVIDIA L4`;
- dòng `kernel: True`;
- test xanh hết, không còn test bị bỏ qua vì thiếu CUDA.

Mỗi lần SSH lại: `source ~/venvs/pdp/bin/activate`.

---

## 6. Dữ liệu (chỉ trên VM)

### 6.1 Kaggle API token

Dataset gốc chỉ có trên Kaggle (cần tài khoản miễn phí). Bản mirror trên HuggingFace không dùng được vì thiếu tên file gốc và trường `level` (plan mục 4.1).

1. Đăng nhập kaggle.com → *Settings* → mục *API* → *Generate New Token*. Kaggle hiện ra một chuỗi dạng `KGAT_...`, gọi là API token định dạng mới. File `kaggle.json` chỉ dùng cho định dạng cũ ("Legacy API Credentials"), không cần nữa.
2. Đưa token lên VM, vào `~/.kaggle/access_token`, quyền 600 (**không commit**, không ghi vào file nào trong repo):

```bash
printf '%s' 'KGAT_...' | gcloud compute ssh auto-cl "${GC[@]}" --command 'mkdir -p ~/.kaggle && chmod 700 ~/.kaggle && (umask 077 && cat > ~/.kaggle/access_token)'
```

`scripts/download_rpc.sh` gọi thẳng API tải của Kaggle bằng `curl` với token này, vì Kaggle CLI bản dành cho Python 3.10 (1.7.4.5) chưa đọc được token định dạng mới. Script vẫn nhận `~/.kaggle/kaggle.json` kiểu cũ nếu có. Nếu token bị lộ thì vào *Settings* → *API* để thu hồi và tạo token mới.

### 6.2 Tải và chuẩn bị dữ liệu (trên VM, trong tmux)

```bash
source ~/venvs/pdp/bin/activate && cd ~/AutoCheckout-CL
bash scripts/download_rpc.sh                               # T0.5: tải 25,3 GB, khoảng 4 phút (tải tiếp được nếu bị ngắt), giải nén ảnh quầy, xóa zip
bash scripts/prepare_data.sh 2>&1 | tee /data/rpc/prepare_data.log   # DL1 -> DL6
```

`prepare_data.sh` chạy lần lượt:
1. DL1: kiểm tra dữ liệu, ghi `results/data_audit/`.
2. DL2: thu nhỏ ảnh về 800×800 vào `/data/rpc/checkout_800`.
3. DL3: chia tập theo nhóm ảnh, ghi `configs/splits/`.
4. DL4: chia lớp theo task, ghi `configs/tasks_100-4x25_seed0.json`.
5. DL5: sinh các file JSON cho từng task, kể cả file joint cho E0 và file class-agnostic cho detector của E5.
6. DL6: sinh dữ liệu pilot.

**Trước khi dùng kết quả chia tập:**

1. Đọc `results/data_audit/audit.md`. Cần kiểm tra:
   - có trường `level` không;
   - có trường `area` không;
   - ảnh có vuông không;
   - thống kê nhóm ảnh: có nhóm nào quá lớn không, hậu tố tên file có trùng giữa `val2019` và `test2019` không.
2. Xem 20 ảnh đã vẽ bbox trong `/data/rpc/draw_check/`.
3. Đưa các file cấu hình sinh trên VM về Mac rồi **commit**, vì tập test phải được khóa trong repo:

```bash
# trên Mac
cd "/Users/an/Documents/Do An/AutoCheckout-CL"
for f in configs/tasks_100-4x25_seed0.json configs/splits results/data_audit; do
  gcloud compute scp --recurse "auto-cl:AutoCheckout-CL/$f" "$(dirname "$f")/" --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0
done
gcloud compute scp auto-cl:/data/rpc/tasks/100-4x25_seed0/manifest.json configs/splits/manifest_100-4x25_seed0.json --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0
git add configs results/data_audit && git commit -m "DL1-DL5: data audit, split and task config from the real data" && git push
# rồi trên VM: git fetch + reset như mục 5.1
```

---

## 7. Chạy thí nghiệm

### 7.1 Đo tốc độ và bộ nhớ trước khi chạy dài

Trên VM, lấy các tham số model của một thí nghiệm rồi chạy benchmark:

```bash
source ~/venvs/pdp/bin/activate && cd ~/AutoCheckout-CL/pdp
ARGS=$(bash -c 'REPO=~/AutoCheckout-CL; DATA=/data/rpc; RUNS=/data/runs; source ../configs/exp/E3.sh; printf "%q " "${ARGS[@]}"')
eval python benchmark.py $ARGS --bench_mode train --batch_size 2 --bench_steps 20
eval python benchmark.py $ARGS --bench_mode train --batch_size 4 --bench_steps 20
eval python benchmark.py $ARGS --bench_mode infer --bench_sizes 640 800
```

Mỗi lệnh in ra `seconds_per_image` và `peak_gpu_memory_gb`. Chọn batch size lớn nhất chạy được mà còn dư khoảng 20% bộ nhớ, rồi sửa giá trị mặc định `BATCH_SIZE` trong `configs/exp/common.sh`. Dùng số giây/ảnh đo được để tính lại số giờ GPU (plan mục 3.4).

### 7.2 Chạy một thí nghiệm

```bash
tmux new -s train
source ~/venvs/pdp/bin/activate && cd ~/AutoCheckout-CL
bash scripts/run_exp.sh configs/exp/P2.sh --shutdown 2>&1 | tee -a /data/runs/P2.log
```

- Rời tmux mà job vẫn chạy: `Ctrl+B`, rồi `D`. Quay lại: `tmux attach -t train`.
- `--shutdown` tắt VM khi script kết thúc, **kể cả khi lỗi**.
- Mỗi file trong `configs/exp/` là một thí nghiệm của plan mục 7. Thứ tự chạy:
  1. Pilot: P1, P2, rồi `FSA_pilot` trước P3.
  2. `FSA` trước E4 và các ablation A1–A9.
  3. E4 trước A1, A4, A7, A8, vì các ablation này dùng lại task 1 của E4.
  4. `DET` trước E5 (mục 7.4).
- Kết quả của từng thí nghiệm nằm ở `/data/runs/<tên>/`:
  - `metrics_cl_{val,test}.json` và `.md`;
  - `metrics_count_test.json` và `.md`;
  - thư mục `task_<t>/`, bố cục như `docs/formats.md` mục 6.

**Chạy nhiều thí nghiệm liên tiếp bằng hàng đợi** (`scripts/run_queue.sh`):

```bash
printf '%s\n' FSA E4 >> /data/runs/queue.txt     # mỗi dòng một tên trong configs/exp/
tmux new -d -s queue 'bash ~/AutoCheckout-CL/scripts/run_queue.sh'
```

- File hàng đợi được đọc lại trước mỗi thí nghiệm, nên có thể thêm tên trong lúc đang chạy.
- Mỗi thí nghiệm xong ghi một dòng `<tên> exit=<mã>` vào `/data/runs/queue.log` và không bao giờ chạy lại. Output ở `/data/runs/<tên>.log`.
- Thí nghiệm bị ngắt bởi tín hiệu (VM tắt, Spot bị thu hồi) được ghi `<tên> interrupted`, **không** tính là xong, và hàng đợi dừng lại.
- Muốn bỏ một thí nghiệm: xóa dòng của nó trong `queue.txt`.
- `queue.log` có dòng `queue empty` khi mọi thí nghiệm đã xong. Dòng `queue stopped` luôn được ghi ngay trước khi VM tắt.
- Hết hàng đợi thì VM tự tắt, kể cả khi có thí nghiệm lỗi.

### 7.3 Khi VM bị tắt giữa chừng (Spot, lỗi, bảo trì)

1. Bật VM lại (mục 2.1) và SSH vào.
2. Chạy lại **đúng lệnh cũ** (với hàng đợi: chạy lại `run_queue.sh`; thí nghiệm đang dở không có dòng `exit=` nên được chạy tiếp). Script tự xử lý:
   - task đã xong thì bỏ qua;
   - task thiếu file dự đoán thì chỉ dự đoán lại;
   - task đang dở thì nối tiếp từ `task_<t>/last.ckpt` (lưu mỗi 30 phút và mỗi cuối epoch).

### 7.4 E5 (truy xuất DINOv2)

Cần chạy xong `DET` trước:

```bash
cd ~/AutoCheckout-CL
CFG=configs/tasks_100-4x25_seed0.json; TASKS=/data/rpc/tasks/100-4x25_seed0
python -m baselines.retrieval run --det-dir /data/runs/DET/task_1 --task-dir $TASKS --task-config $CFG \
  --image-dir /data/rpc/checkout_800 --emb-dir /data/runs/e5_emb --out-dir /data/runs/E5 --capped
for split in val test; do
  python -m tools.eval_cl --run-dir /data/runs/E5 --split $split --ann $TASKS/${split}_full.json --task-config $CFG
done
python -m tools.eval_count --run-dir /data/runs/E5 --val-ann $TASKS/val_full.json --test-ann $TASKS/test_full.json --task-config $CFG
```

Muốn thử `--mode knn` hoặc một nhiệt độ khác: đổi `--out-dir`, dùng lại `--emb-dir` (embedding không phải tính lại), rồi so sánh trên **val**.

### 7.5 Bảng tổng hợp

```bash
python -m tools.summarize --runs /data/runs/E0 /data/runs/E1 /data/runs/E2 /data/runs/E3 /data/runs/E4 /data/runs/E5 --out /data/runs/summary
```

---

## 8. Theo dõi từ Mac (chỉ đọc, không ảnh hưởng job)

Tóm tắt nhanh bằng một lệnh: trạng thái VM, thí nghiệm nào đã xong, tiến độ hiện tại, GPU, dung lượng ổ.

```bash
bash scripts/vm_status.sh           # tóm tắt một lần
bash scripts/vm_status.sh follow    # xem trực tiếp output của thí nghiệm đang chạy; Ctrl+C chỉ dừng việc xem
```

`scripts/run_pilot.sh` ghi output của từng thí nghiệm vào `/data/runs/<tên>.log`, không in ra cửa sổ tmux, nên `tmux attach -t pilot` chỉ thấy màn hình trống. Muốn xem thì dùng `follow`. Khi chạy `run_exp.sh` trực tiếp trong tmux (mục 7.2) thì output hiện trong tmux.

Các lệnh riêng lẻ:

```bash
gcloud compute ssh auto-cl "${GC[@]}" --command "tail -n 30 /data/runs/<tên>.log"                          # log
gcloud compute ssh auto-cl "${GC[@]}" --command "nvidia-smi"                                               # GPU có đang chạy
gcloud compute ssh auto-cl "${GC[@]}" --command "tmux capture-pane -pt train:0.0 -S -120"                  # 120 dòng cuối của tmux
gcloud compute ssh auto-cl "${GC[@]}" --command "ls /data/runs/*/task_*/task_final.pth; df -h /"           # task đã xong, dung lượng
```

---

## 9. Lấy kết quả về Mac

Chỉ lấy chỉ số, log, `run_info.json` và file dự đoán (mỗi file dự đoán khoảng vài chục MB). Không lấy dữ liệu, không lấy checkpoint.

```bash
# trên VM: gói một thí nghiệm, bỏ checkpoint
tar czf /data/runs/<tên>.tgz -C /data/runs --exclude='*.pth' --exclude='*.ckpt*' --exclude='hf_model' <tên>
# trên Mac
mkdir -p "/Users/an/Documents/Do An/AutoCheckout-CL/results/runs"
gcloud compute scp auto-cl:/data/runs/<tên>.tgz "/Users/an/Documents/Do An/AutoCheckout-CL/results/runs/" --zone=us-central1-c --project=project-95a0d104-9d0f-4aa1-ba0
```

`results/runs/` không commit file `.npz` (đã có trong `.gitignore`); các file `metrics_*` và `.md` thì commit được.

---

## 10. Checklist trước khi chạy dài

```text
[ ] setup_vm.sh: "2.2.2+cu121 True NVIDIA L4", "kernel: True", pytest xanh
[ ] Dữ liệu: audit.md đã đọc; configs/splits và configs/tasks_*.json đã commit trên Mac và sync lại
[ ] Đã đo benchmark (mục 7.1) và đặt BATCH_SIZE trong configs/exp/common.sh
[ ] Đã chạy pilot P1-P3 và qua mốc G1 (plan mục 8)
[ ] df -h / còn đủ chỗ (mỗi lần chạy 5 task khoảng 2 GB, cộng 1 GB checkpoint resume của task đang chạy)
[ ] Chạy trong tmux, có --shutdown
[ ] (Nếu chạy dài) đã chuyển sang Spot, sau khi thử tắt VM giữa chừng và resume thành công
```

---

## 11. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `python3 -m venv` báo thiếu `ensurepip` | `sudo apt-get install -y python3.10-venv` |
| `CUDA out of memory` | Giảm `BATCH_SIZE` trong `configs/exp/common.sh` (có thể xuống 1). Số bước tích lũy gradient tự tăng để batch hiệu dụng vẫn là 32 (`--eff_batch_size`) |
| `kernel: False` / lỗi biên dịch | Kiểm tra `ninja --version` (trong venv), `echo $CUDA_HOME`, `/usr/local/cuda/bin/nvcc --version`; đặt `TORCH_CUDA_ARCH_LIST=8.9`; xóa cache `~/.cache/torch_extensions` rồi thử lại |
| `No space left on device` | `df -h /`, `du -sh $RUNS/*`; xóa `last.ckpt` của các task đã xong (vẫn giữ `task_final.pth`); xóa zip RPC; hoặc tăng dung lượng ổ (mục 4) |
| `gcloud` trên Mac báo `NameResolutionError ... compute.googleapis.com` | DNS của mạng đang dùng (ví dụ mạng trường) chập chờn. Thử lại sau vài giây, hoặc đổi mạng hoặc DNS (ví dụ 8.8.8.8) |
| VM tự tắt giữa chừng | Xem mục 2.4 và 2.5; làm theo mục 7.1 |
| Không bật được VM, báo `STOCKOUT` (zone tạm hết GPU L4, kể cả on-demand) | Đợi vài phút rồi thử lại. Ngày 28/09 gặp lỗi này một lần, lần thử thứ hai bật được. Nếu kéo dài: tạo snapshot ổ rồi tạo VM ở zone khác có L4 (ổ đĩa nằm cố định ở `us-central1-c`) |
| `git fetch`/`reset` trên VM báo lỗi | Thư mục `~/AutoCheckout-CL` trên VM có thể xóa rồi clone lại (mục 5.1); dữ liệu và kết quả nằm ở `/data` nên không mất |
| `RuntimeError: Multi-scale deformable attention: PyTorch fallback` | Cấu hình đặt `--require_kernel 1` mà kernel chưa build được: xem dòng `kernel: False` ở trên |
