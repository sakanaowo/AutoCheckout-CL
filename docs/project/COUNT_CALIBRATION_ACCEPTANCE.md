# S6: nghiệm thu calibration và evaluation

**Đạt `PASS_COUNT_CALIBRATION_ACCEPTANCE`, 7/7 gates, 71 tests PASS**, không failures/errors/skips. [Notebook evaluation 01](../../notebooks/evaluation/01_count_calibration_acceptance.ipynb) thực chạy **2026-10-10T00:36:13+07:00 → 00:36:24+07:00**, run `af79910c-a01e-4f63-83b7-8c84ac3e49b6`, Conda pdp Python 3.10. Chỉ dùng CPU; bước này không cần RTX 3060 hoặc 4090.

## Hành vi đã triển khai và kiểm chứng

- [eval_count.py](../../tools/eval_count.py): calibration val-only, grid score/NMS có cấu hình, lưu JSON policy riêng; test nạp policy mà không đọc val hoặc tìm threshold mới. Oracle mặc định tắt; `--oracle` bật diagnostic có nhãn và file `_oracle` riêng.
- Policy lưu từng stage: threshold, NMS IoU, val scores với K/K_eff, candidates theo NMS, grid/tie-break, task config SHA256, annotation/prediction val MD5 và checkpoint MD5 khi checkpoint tồn tại. Test từ chối mapping/checkpoint/stages không khớp; đọc policy không sửa artifact đã khóa.
- [counting.py](../../autocheckout/counting.py): NMS chéo lớp theo từng ảnh, không bỏ score dưới 0,05 trước khi áp grid; thứ tự tie theo query rồi label. Grid được kiểm tra hữu hạn trong [0,1], sắp tăng dần. Chọn maximum val cAcc, rồi threshold nhỏ nhất, rồi NMS IoU nhỏ nhất (0 = off).
- [predictions.py](../../autocheckout/predictions.py): top-1 cùng query/score chọn label nhỏ nhất, kết quả không phụ thuộc thứ tự dòng. Input counting kiểm tra annotation hash, split, task/seen classes, image IDs, label range, finite score/xyxy và metadata tọa độ. Legacy không có `coordinates` được hiểu là pixels của annotation; metadata chỉ rõ hệ tọa độ khác bị từ chối. Bbox có thể vượt rìa ảnh theo output detector; không clip âm thầm.
- [runner](../../scripts/run_exp.sh) với configs native: `EVALUATE=1` chạy mAP val/test trên predictions thô, tạo policy val nếu chưa có rồi evaluate test bằng `--policy-in`. Không gọi oracle hoặc chọn NMS từ test. Mặc định smoke vẫn bỏ evaluation. Test dùng runner thật với task đã hoàn tất và chứng minh rerun giữ policy dù predictions val thay đổi.

Nhánh mAP giữ nguyên quy ước top-k detector; NMS/threshold counting không sửa predictions hoặc áp sang mAP. Các metrics/formulas cũ được tái sử dụng, gồm old/new groups, level breakdown và quy tắc zero-GT của mCCD/mCIoU.

## Kết quả biết trước và predictions thật

Fixture val có duplicate hai labels cho cùng vật và decoy score 0,20: chọn NMS 0,45 và threshold **0,21**, cAcc val **1,0**. Test có một true detection score 0,15: policy vẫn 0,21, cAcc test **0,5**. Sau khi làm paths val không còn, test-only evaluation vẫn chạy. Thay GT test trong fixture làm cAcc thành **0,0**, hash policy không đổi.

Bốn cases EXP-B1/B2 × 640/800 từ [notebook modeling 04](TRAINING_RUNTIME_ACCEPTANCE.md) đều PASS các tiến trình calibration → locked test evaluation → raw mAP. Threshold val của bốn checkpoint smoke là 0,10, NMS 0,45. Counting diagnostics trên 2 ảnh test: cAcc 0,0, ACD 4,0, mCCD 1,0, mCIoU 0,0; K=100, K_eff=2. Model mới train 4 optimizer steps, nên kết quả này chỉ kiểm chứng kỹ thuật, không đánh giá chất lượng baseline. Source, subset annotations, mapping, predictions và checkpoints được đối chiếu hash trước/sau, không đổi.

Artifacts: `runs/evaluation/count_calibration/af79910c-a01e-4f63-83b7-8c84ac3e49b6/`, gồm config, input hashes, gates/summary, regression logs/JUnit, executed notebook, fixture và 4 thư mục case chứa policy/metrics. Các task directories trỏ bằng symlink tới notebook 04; không copy checkpoint. Run đầu `8d2521e8-aaa8-4d57-b7a7-df806f41c2b2` đạt 68 tests, run `5a8e3611-1dc1-4cfb-82ec-c88a5e749680` đạt 70. Run hiện hành thêm test tie score của cùng query; test đã FAIL khi bỏ correction, rồi PASS sau khi khôi phục correction.

## Vận hành từ notebook và runner

Mở notebook evaluation 01, chọn Conda pdp Python 3.10, restart rồi Run All. Mặc định đọc marker latest của modeling 04. Nếu chuyển artifact sang máy khác, đặt `AUTOCHECKOUT_RUNTIME_SUMMARY` trỏ tới `summary.json`; source run cần giữ config paths, subset và checkpoints có thể truy cập. Notebook mới có run ID riêng, không train lại.

Pilot dùng configs native với `EVALUATE=1`. `COUNT_NMS_GRID=0.45` và score grid mặc định **0.10–0.90, step 0.01**. Có thể đặt `COUNT_THRESHOLD_GRID=0.1,0.2,0.3` hoặc `COUNT_NMS_GRID=0,0.35,0.45,0.55` trước khi calibration lần đầu; lựa chọn chỉ dựa trên val. `CALIBRATION_POLICY` nhận đường dẫn policy riêng, mặc định `<run>/calibration_count.json`.

Policy hiện có được giữ nguyên khi runner chạy lại. Khi thay checkpoint/stages/task mapping hoặc chủ động đổi grid, dùng run hoặc policy output mới và calibration lại trên val; không sửa policy để khớp test. Lệnh test-only có thể không có val files; runner còn cần val để báo mAP val.

```bash
python -m tools.eval_count --run-dir RUN --task-config TASKS.json \
  --calibrate-only --val-ann VAL.json --nms-grid 0.45 --policy-out POLICY.json
python -m tools.eval_count --run-dir RUN --task-config TASKS.json \
  --policy-in POLICY.json --test-ann TEST.json
```

## Giới hạn và bước tiếp theo

S6 kỹ thuật đã nghiệm thu; `pilot_calibrated=False`, `full_training_ready=False`. Policy từ fixture/subset không dùng cho metrics nghiên cứu. Mỗi checkpoint pilot phải có policy được chọn trên toàn bộ val của release rồi áp cố định sang test. Test lock và protocol 5 task giữ nguyên.

Theo quyết định người dùng ngày 10/10/2026, bỏ notebook nghiệm thu môi trường Vast.ai riêng. Tiếp theo: chuẩn bị notebook/config local; thực chạy pilot EXP-B1/B2 cùng split/resolution/optimizer-step budget trên RTX 4090; full-model Task 1→2 teacher/PPG/prototype smoke trên cùng máy; baseline đủ 5 task. Metadata và throughput/VRAM ghi trong run pilot/training. Chưa có run Vast.ai hoặc convergence mới.

Validation: regression 71 tests trong notebook, 7 gates end-to-end, thêm 19 retrieval tests PASS, ruff cho các Python files thay đổi, shell syntax và `git diff --check`. AI DevKit offline lint/memory trả `ENOTCACHED`; không dùng kết quả đó làm bằng chứng lint/memory thành công.
