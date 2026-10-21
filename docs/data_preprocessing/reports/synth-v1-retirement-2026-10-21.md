# Synthetic v1: lưu trên Drive, xóa bản local — 21/10/2026

Theo yêu cầu người dùng ngày 21/10, gọi bộ ảnh ghép cũ là **synthetic v1**, bộ sinh mới là **synthetic v2**. Người dùng xác nhận v1 đã có bản sao lưu trên Google Drive và cho phép xóa bản local. Không tải lên hoặc xóa dữ liệu trên Drive trong thao tác này; chưa kiểm tra độc lập bản remote, chưa có URL/file ID Drive được cung cấp.

## Phạm vi và kết quả

Đã xóa đúng thư mục `data/archive/synth/group/`, gồm `task1`–`task4`: **20.000 JPEG + 4 annotations.json**. Tổng kích thước file **9.873.969.276 bytes**; allocated size trước xóa **9.915.019.264 bytes (~9,23 GiB)**. Không xóa các annotation tham chiếu ở `data/archive/tasks/` hoặc `tasks_pdp/`.

Trước xóa đã lưu danh mục path/size/SHA-256 của toàn bộ 20.004 file tại `runs/data_preprocessing/synth_v1_retirement/inventory.json`. SHA-256 danh mục: `2449768f994c3ae87526f9a095ae09991c31f0069c6d07046ca038642fe477d2`. Đây là inventory local trước xóa, không chứng minh nội dung trên Drive trùng khớp.

[Receipt kiểm tra](assets/synth-v1-retirement-2026-10-21.json) lưu thời gian bắt đầu/kết thúc thực tế UTC+7, counts, hashes và phạm vi bảo toàn. Sau thao tác:

- `data/archive/synth/group/` không còn; `data/archive/synth/README.md` đánh dấu v1 đã lưu trữ.
- Số file trong cutouts, backgrounds, scripts, val2019 và test2019 không đổi.
- Hash cutout index, real release manifest, annotations và manifest của resolved 202 không đổi. Không full-hash lại mọi ảnh thật; đây là kiểm tra metadata và file counts cùng phạm vi xóa chính xác.
- Resolved 202 còn **193 ảnh tái ghép**; **9 ảnh giữ nguồn** trỏ vào v1 đã xóa. Bản 202 hiện không đọc đủ ảnh local.

## Quy ước phiên bản

| Tên | Nội dung | Trạng thái |
|---|---|---|
| **Synthetic v1** | 20.000 ảnh ghép cũ, 4 task; đường dẫn gốc `data/archive/synth/group/` | Backup Drive do người dùng xác nhận; đã xóa local |
| **Resolved 202 v1** | Bản sửa riêng của v1 tại `data/processed/rpc_synth_resolved_202_v1/` | Giữ artifacts; 193 ảnh mới còn local, 9 ảnh nguồn cần restore |
| **Synthetic v2** | Bộ sẽ sinh mới theo kế hoạch 5 task | Chưa sinh; output riêng `data/processed/rpc_synth_v2_<version>/` |
| **Real release v1** | `rpc_100-4x25_seed0_native_v1` và holdout thật | Giữ nguyên; chữ v1 này không có nghĩa là bộ synthetic cũ |

## Ảnh hưởng đến công việc cũ

`tasks/group/*_train_real+synth.json`, `tasks_pdp/group/train_task_*.json`, `tasks_pdp/group_mini/train_task_*.json` vẫn chứa đường dẫn ảnh v1. Các file này là lịch sử, không dùng làm đầu vào v2. Không sửa/xóa annotation để che việc nguồn ảnh đã được lưu trữ.

Notebook audit/review/resolution cũ và `verify_resolution_dataset` có thể cần cả annotation và hashes ảnh v1. Các báo cáo PASS trước đây vẫn là bằng chứng tại thời điểm chạy; chúng không chứng nhận dataset hiện tại đủ file để rerun. 20 case pilot pending vẫn là lịch sử chưa xử lý, không được đánh dấu hoàn tất do xóa ảnh.

Baseline real-only B1/B2 và nguyên liệu tạo v2 không dùng ảnh ghép v1 làm đầu vào. V2 tiếp tục từ cutout, nền và thống kê train thật; không phục hồi v1 chỉ để sinh v2.

## Khôi phục khi cần

1. Lấy bản backup v1 trên Drive do người dùng quản lý; restore đúng cấu trúc `data/archive/synth/group/task{1..4}/images/` và `annotations.json`.
2. Đối chiếu số file, size và SHA-256 với inventory đã lưu. Nếu thiếu inventory local, receipt trong git giữ hash để kiểm tra bản inventory được khôi phục.
3. Kiểm tra lại paths của 9 ảnh trong receipt. Chỉ khôi phục 9 ảnh giúp đọc bản 202 nhưng **không đủ** để đảm bảo toàn bộ verifier/audit cũ chạy lại, vì chúng còn kiểm tra các nguồn khác.
4. Chạy lại gate tương ứng, lưu run mới; không sửa timestamp hoặc PASS của run lịch sử.

Không dùng đường dẫn v1 đã nghỉ để chứa ảnh v2. Xem [kế hoạch v2](../../project/synthetic-data-v2-plan-2026-10-20.md).
