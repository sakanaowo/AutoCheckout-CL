# Audit dữ liệu nhập — 07/10/2026

Notebook đã chạy: [01_data_audit.ipynb](../../../notebooks/data_preprocessing/01_data_audit.ipynb). Root kiểm tra: `data/archive`. Run ID: `1bfc5c33-6a4b-4145-9bc5-24d390362b6f`.

**Thời gian thực thi:** 07/10/2026, **10:46:26–10:47:54, UTC+7 (Asia/Bangkok)**, 88 giây. Chạy code cells tuần tự bằng Python 3.14.4 trên máy phát triển, không train hoặc kiểm tra GPU. Training dự kiến trên RTX 4090 Vast.ai.

**Trạng thái sau audit:** protocol đã chốt 5 task và split ảnh thật đã nghiệm thu trong [notebook 02](../../../notebooks/data_preprocessing/02_split_and_acceptance.ipynb). Xem [báo cáo release](split-100-4x25-2026-10-07.md); những mục chưa chốt dưới đây mô tả thời điểm audit ban đầu.

## 1. Inventory

| Nguồn | Số file ảnh |
|---|---:|
| `val2019` | 6.000 |
| `test2019` | 24.000 |
| `synth/group/task1..task4` | 5.000/task, tổng 20.000 |
| `backgrounds` | 209 |
| `cutouts` | 7.513 |

Số file trong `cutouts` gồm các ảnh tài nguyên/preview/review, chưa xác nhận toàn bộ đều là cutout sản phẩm hợp lệ. Bỏ qua metadata macOS và `.venv` nằm trong archive; không thực thi scripts hoặc môi trường được đóng gói cùng dataset.

## 2. Kết quả kiểm tra annotation và ảnh tham chiếu

- Đọc và audit **68 COCO JSON**: annotation RPC gốc, `tasks_pdp`, `tasks/group`, `tasks/random` và annotations synth.
- **9 JSON có ảnh tham chiếu bị thiếu**: `instances_train2019.json` và 8 file `task*_train_single.json` trong `tasks/group`/`tasks/random`.
- `instances_train2019.json` tham chiếu **53.739 ảnh**, tất cả chưa có tại root kiểm tra; không thấy thư mục `train2019`. Các file train_single là tập con, không cộng chúng để tính số ảnh thiếu duy nhất.
- Không ghi nhận lỗi ID trùng trong cùng JSON, ID ảnh/lớp không tồn tại, bbox không hợp lệ/vượt kích thước annotation, hoặc thiếu/không hợp lệ `area`/`iscrowd` trong các JSON đã audit. Đây là kiểm tra theo metadata annotation, chưa đối chiếu kích thước pixel của toàn bộ ảnh.
- Checksum SHA-256 từng JSON được lưu trong output notebook.
- Kiểm tra header/file bằng Pillow `verify()` trên **65 ảnh mẫu**, không báo lỗi. Chưa giải mã đầy đủ pixel của mọi ảnh hoặc xác nhận chất lượng thị giác/mask cho toàn bộ dataset.

## 3. Protocol và split

`tasks_pdp/{group,group_real,group_mini}/rpc_task_info.json` dùng **4 task `[48,68,47,37]`**, tổng 200 lớp. Repo có config **5 task `100+4×25`**, thêm 24 slot dự phòng. Hai cấu hình không thể dùng lẫn bằng cách chỉ đổi `n_tasks`.

| Split nhập | Ảnh train duy nhất | Ảnh test duy nhất | Trùng đường dẫn train/test |
|---|---:|---:|---:|
| `group` | 26.000 | 24.000 | 0 |
| `group_real` | 6.000 | 24.000 | 0 |
| `group_mini` | 1.947 | 1.000 | 0 |

“Ảnh duy nhất” được tính bằng đường dẫn, gộp các task. Không trùng đường dẫn chưa chứng minh không leakage: chưa hash/perceptual-hash toàn bộ ảnh, kiểm tra nhóm ảnh chụp liên tiếp hoặc truy nguồn nền/cutout của ảnh ghép sang split đánh giá.

## 4. Điều kiện còn thiếu trước training

1. Xác nhận dùng 4 task hiện có hay tạo lại 5 task theo handoff; sau đó khóa label mapping, class coverage và số slot.
2. Chọn train thật/ghép/mix, tạo **validation độc lập**, chia theo nhóm và khóa test. `val2019` đang là nguồn ảnh train của các JSON nhập; tên thư mục không chứng minh đó là validation độc lập cho run mới.
3. Kiểm tra ảnh gần trùng, nguồn ảnh ghép, bbox overlays và decode ảnh ở phạm vi đủ để nghiệm thu dữ liệu.
4. Nếu dùng train_single/generator từ ảnh gốc, cần bổ sung `train2019`; đường train checkout thật/ghép có ảnh tham chiếu hiện hữu và không mặc định bị chặn bởi việc thiếu ảnh đơn.
5. Tạo notebook split/preprocess có timestamp, config và manifest; notebook model-check/train chỉ chạy sau khi các điều kiện liên quan được đáp ứng.

**Trạng thái:** audit cấu trúc/tham chiếu đã chạy; chưa phê duyệt dataset là sẵn sàng train. Không sửa, clip bbox, đổi task hoặc xóa dữ liệu trong lần audit này.

**Kiểm tra sâu tiếp theo:** [notebook 01b](../../../notebooks/data_preprocessing/01b_preprocessing_acceptance.ipynb) đã full decode/checksum 57.710 file và kiểm tra polygon mask, không còn giới hạn chỉ verify 65 ảnh ở bước này. Phát hiện 222 sai khác mask/bbox sau tolerance dù bbox bounds/ID/class vẫn hợp lệ. Xem [báo cáo nghiệm thu](preprocessing-acceptance-2026-10-07.md); không diễn giải kết quả audit cấu trúc ở trên thành segmentation đã hoàn toàn đúng.
