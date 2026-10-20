# Đánh giá

**20/10/2026:** [02_final_backbone_report.ipynb](02_final_backbone_report.ipynb) tổng hợp baseline hai backbone đủ 5 task, đọc metrics và receipts đã tính lại, xuất bảng/ma trận và timestamps thực chạy. [Báo cáo chi tiết](../../docs/project/FINAL_BACKBONE_COMPARISON_2026-10-20.md) chứa kết luận và đối chiếu paper. Notebook 02 chỉ tổng hợp kết quả; lần tính lại NPZ đầy đủ được ghi riêng trong receipts/script, không giả nhận đó là phép tính AP trong notebook.

[01_count_calibration_acceptance.ipynb](01_count_calibration_acceptance.ipynb) đã nghiệm thu S6 kỹ thuật: 7/7 gates trên CPU, grid/NMS chéo lớp, calibration val-only, lưu/nạp policy, oracle mặc định tắt và mAP dùng predictions thô. [Báo cáo](../../docs/project/COUNT_CALIBRATION_ACCEPTANCE.md) ghi run thực chạy và tests. Chọn Conda pdp Python 3.10, restart rồi Run All; không cần GPU.

Notebook tự đọc marker latest của modeling 04; đặt `AUTOCHECKOUT_RUNTIME_SUMMARY` nếu cần chọn source run khác. Chỉ đọc predictions/checkpoints, không train lại. Artifacts ở `runs/evaluation/count_calibration/<run_id>/`; nguồn và release được kiểm tra hash trước/sau.

Fixture và predictions 2 ảnh từ smoke chỉ xác nhận pipeline. Metrics nghiên cứu cần checkpoint pilot thật, val toàn bộ để chọn policy và test lock cố định. Configs native bật evaluation bằng `EVALUATE=1`; runner tạo policy val lần đầu rồi dùng lại. Không dùng policy subset cho pilot.

Protocol chính 5 task; không dùng kết quả E0/E1/E4/E5 cũ làm kết quả nhánh hiện tại.
