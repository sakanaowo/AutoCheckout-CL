# Nghiệm thu CLI/runner ResNet và ConvNeXt

**Đã đạt `PASS_PDP_CLI_RUNNER_CUDA_SMOKE`, 10/10 gates trên RTX 3060.** [Notebook 04](../../notebooks/modeling/04_training_runtime_acceptance.ipynb) gọi runner và CLI bằng các tiến trình thật; không patch factory/model/loader.

Run `ac818116-ceb0-4ce8-9c1e-979c6f54788a`. Bắt đầu **2026-10-09T23:58:00+07:00**, kết thúc **2026-10-10T00:02:03+07:00** (UTC+7). **78 tests PASS**, 0 failures/errors/skips trong `regression.xml`. Chạy bằng Conda pdp hiện có, không cài dependencies.

## Kết quả thực chạy

| Model | Resolution | Peak training allocated VRAM | Optimizer steps qua resume | Kết quả |
|---|---:|---:|---|---|
| EXP-B1 | 640 | 3.61 GiB | 1 → 4 | PASS |
| EXP-B1 | 800 | 4.35 GiB | 1 → 4 | PASS |
| EXP-B2 | 640 | 3.79 GiB | 1 → 4 | PASS |
| EXP-B2 | 800 | 4.52 GiB | 1 → 4 | PASS |

Mỗi case: 4 ảnh train Task 1, 2 ảnh val_task và 2 ảnh mỗi tập prediction val/test; physical batch 1, effective batch 2, accumulation 2, 2 epochs, FP32, TF32 tắt, seed 0, shuffle=0. Source/release/holdout/weights giữ nguyên; subset JSON chỉ nằm trong artifact. Prediction ảnh ID 6114 được kiểm tra từ size 640/800 về native **1838×1838** bằng bbox chuẩn hóa nhân kích thước ảnh gốc.

- Shell runner dừng có chủ đích ở optimizer step 1 (exit 75), không tạo `task_final.pth` hoặc predictions trước khi hoàn tất.
- Tiến trình mới resume bằng metadata checkpoint dù paths bootstrap detector/backbone không tồn tại; hash model/optimizer/scheduler/PDP memory sau Lightning restore khớp checkpoint. Global step đạt 4, session tiếp nối thực hiện thêm 3 optimizer steps.
- Checkpoint/config dựng lại đúng ResNet hoặc ConvNeXt và processor. Hash backbone/encoder/decoder giữa interruption checkpoint và final giữ nguyên; loss hữu hạn.
- Khi thiếu một prediction, runner dùng predict-only; scores/labels/queries/boxes khớp lần xuất trước. Khi đủ final checkpoint và hai predictions, runner bỏ qua task đã hoàn tất.
- Runner hoạt động không cần `PYTHONPATH` hoặc cài project editable; CLI thiết lập đường import từ vị trí source.

Artifacts: `runs/modeling/training_runtime/ac818116-ceb0-4ce8-9c1e-979c6f54788a/`, gồm `summary.json`, `gates.json`, `regression.xml`, `notebook_executed.ipynb`, nguồn/config/weights manifest và bốn thư mục case. Mỗi case giữ initial loading report, interrupted state hashes, logs cho bốn lần gọi runner, `runtime.json`, `run_info.json`, `task_final.pth` và predictions. Checkpoint resume tạm được xóa theo lifecycle khi task hoàn tất.

## Phần đã triển khai

- [runtime.py](../../pdp/runtime.py): dựng model/processor từ config, selective detector transfer khi đổi backbone, kiểm tra loading report và metadata checkpoint.
- [main.py](../../pdp/main.py): `--backbone`, `--backbone_pretrained_file`, `--model_config`, `--image_size`, `--max_image_size`, `--stop_after_steps`, `--verify_resume`; train/val/predict cùng processor.
- [engine.py](../../pdp/engine.py): lưu model/processor/provenance trong cả hai định dạng checkpoint, dùng strict model state cho checkpoint mới; từ chối resume khác batch/data/optimization contract.
- [checkpointing.py](../../pdp/checkpointing.py): dừng ở ranh giới optimizer step và kiểm tra model/optimizer/scheduler/PDP memory sau restore. Hash state dùng chung ở [model_state.py](../../autocheckout/model_state.py).
- [EXP-B1](../../configs/exp/native/EXP-B1.sh) / [EXP-B2](../../configs/exp/native/EXP-B2.sh): release native, seed 0, 225 outputs, 100 prompts ×10, pretrained đúng backbone. Configs tháng 09 giữ nguyên để truy nguồn.
- [acceptance worker](../../tools/pdp_runtime_acceptance.py): gọi shell runner thật, kiểm chứng resume/predict/skip và đối chiếu bbox native.

Effective batch phải là bội số dương của physical batch × số GPU; CLI từ chối làm tròn. `run_info.json` ghi batch, accumulation, số train batches, optimizer steps toàn task/session, model và processor size. Model/config/processor trong checkpoint quyết định kiến trúc khi resume/predict; cờ backbone/resolution xung đột bị từ chối.

## Cách sử dụng

Mở notebook 04, chọn kernel Conda `pdp` Python 3.10, restart rồi Run All. Notebook tự gọi CLI/runner, ghi logs và kiểm tra kết quả; người dùng tiếp tục thao tác bằng notebook. Defaults đọc release/cache hiện có, `offline=True`; đặt roots bằng `AUTOCHECKOUT_DATA_ROOT` và `AUTOCHECKOUT_RELEASE_ROOT` nếu chuyển máy.

Để gọi runner trực tiếp sau khi đã chọn budget và môi trường training, các biến cấu hình gồm `RAW_ROOT`, `RELEASE_ROOT`, `RUNS`, `PYTHON`, `RESOLUTION`, `BATCH_SIZE`, `EFFECTIVE_BATCH`, `EPOCHS`, `N_TASKS`, `DETECTOR_DIR`, `BACKBONE_WEIGHTS`. Ví dụ chọn config với outputs local bằng `RUNS="$PWD/runs/experiments" bash scripts/run_exp.sh configs/exp/native/EXP-B2.sh`. Defaults là Task 1, 2 epochs, batch 1/effective 2; đây chưa phải budget pilot được chốt. `STOP_AFTER_STEPS=1` lưu checkpoint rồi trả exit 75; chạy lại cùng config với `STOP_AFTER_STEPS=0` để tiếp tục. Dùng thư mục run mới khi đổi cấu hình thí nghiệm; runner bỏ qua task đã hoàn tất.

Native configs đặt `SKIP_EVAL=1` để không gọi evaluator legacy trước khi S6 có calibration artifact. Các metrics trên subset trong notebook là diagnostics, không đưa vào bảng nghiên cứu. Không bật `--shutdown` như cơ chế kết thúc thuê Vast.ai.

## Giới hạn và bước kế tiếp

`full_training_ready=False`, `native_kernel_accepted=False`. Thực chạy dùng PyTorch fallback; notebook này không làm forward/backward equivalence cho native kernel. Chưa chứng nhận convergence, đủ prototypes 100 SKU, full teacher/PPG Task 2+, training trên Vast.ai hoặc metrics protocol mới. Resume state đã kiểm chứng với shuffle=0; không tuyên bố thứ tự minibatch ngẫu nhiên sau interruption giống hệt lượt không bị ngắt.

1. Hoàn thiện S6: threshold/NMS chọn trên val, lưu/nạp calibration policy cố định cho test; oracle tắt trong pipeline nghiên cứu.
2. Kiểm chứng GPU/toolchain trên môi trường training và native kernel hoặc fallback có đo throughput/VRAM.
3. Pilot EXP-B1/EXP-B2 Task 1 cùng split/resolution/optimizer-step budget/evaluator, rồi chọn budget từ loss/val curves.
4. Full-model Task 1→2 teacher/PPG/prototype smoke trước protocol 5 task. LoRA/K=3/FSA/freeze bổ sung và synthetic là tracks sau baseline.

## Lịch sử kiểm chứng

Tests được viết trước production: thiếu cờ CLI, effective batch bị làm tròn, resume cho phép thay batch và CLI thiếu đường import đã có tín hiệu FAIL rồi PASS. Regression gồm paths/teacher fixture, interruption, inference, parameter groups, pretrained classifier, CPU+CUDA acceptance và configs/runner cũ.

Run `dc9cddec-6ede-406e-bfce-1f550ad1a3ab` trước sửa import-path đạt 10/10 gates, 77 tests. Run `11a2b9a7-592f-4eef-9747-318b1dcaaa32` đạt 78 tests và 4 GPU cases nhưng **FAIL source preservation**: runtime/inference/worker được định dạng lại trong lúc chạy; so AST xác nhận không đổi logic. Giữ nguyên trạng thái FAIL của run này. Run hiện hành ở đầu báo cáo là lượt mới trên source đã ổn định.

`npx --offline ai-devkit@latest lint` và memory search báo `ENOTCACHED`; không coi đó là lint/memory thành công. Kiểm chứng dùng pytest, ruff, shell/Python syntax và artifacts thực chạy.
