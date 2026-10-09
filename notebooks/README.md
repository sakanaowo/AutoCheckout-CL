# Notebook thực nghiệm AutoCheckout-CL

Cập nhật **10/10/2026**. Protocol chính đã chốt **5 task 100+4×25**. Mỗi bước dữ liệu, model, train và đánh giá phải có notebook ghi rõ **làm gì và chạy lúc nào**. GPU training là **RTX 4090 trên Vast.ai**; local RTX 3060 đã đạt foundation CPU và full pretrained Task 1 CUDA smoke 640/800 bằng PyTorch fallback; native kernel/môi trường Vast.ai còn chờ.

## Cấu trúc và thứ tự

| Thư mục / notebook | Nội dung / trạng thái |
|---|---|
| [data_preprocessing/01_data_audit.ipynb](data_preprocessing/01_data_audit.ipynb) | Đã audit inventory/COCO, ảnh thiếu và protocol nguồn |
| [data_preprocessing/01b_preprocessing_acceptance.ipynb](data_preprocessing/01b_preprocessing_acceptance.ipynb) | Đã full decode/hash 57.710 file; nguồn synth còn gate riêng |
| [data_preprocessing/01c_protocol_comparison.ipynb](data_preprocessing/01c_protocol_comparison.ipynb) | Phân tích trước quyết định; giữ output lịch sử, người dùng đã chọn 5 task |
| [data_preprocessing/02_split_and_acceptance.ipynb](data_preprocessing/02_split_and_acceptance.ipynb) | Đã chạy; lần mới nhất 16:02:43–16:03:17 UTC+7, real-only split/task JSON, 9 gate đạt |
| [data_preprocessing/03_annotation_review.ipynb](data_preprocessing/03_annotation_review.ipynb) | Visual 222 case, pilots và Batch 200; lịch sử đề xuất, gate nội dung FAIL |
| [data_preprocessing/04_annotation_resolution.ipynb](data_preprocessing/04_annotation_resolution.ipynb) | Đã xử lý 202 case: 9 giữ/193 tái ghép; acceptance PASS ngày 08/10, 13:31:19–13:34:20 UTC+7 |
| [modeling/](modeling/README.md) | Notebook 01 đạt limited foundation (CPU, máy RTX 3060); notebook 02 full pretrained CUDA 640/800 smoke PASS trên 3060 (PyTorch fallback); notebook 03 ConvNeXt PASS; notebook 04 kiểm chứng CLI/runner thật |
| [training/](training/README.md) | Pilot và 5 task trên Vast.ai; chưa train |
| [evaluation/](evaluation/README.md) | Notebook 01 S6 PASS trên CPU: val-only policy, test-only reload và raw mAP; metrics pilot chưa có |
| [references/](references) | Notebook thành viên nhóm, chỉ dùng tham khảo |

[Báo cáo split và test lock](../docs/data_preprocessing/reports/split-100-4x25-2026-10-07.md), [tài liệu tiền xử lý](../docs/data_preprocessing/README.md). Release chính chỉ gồm ảnh thật; bản synth 202 riêng đã nghiệm thu geometry/provenance/mapping, còn 20 pilot pending. Kiểm chứng loader/processor trước khi tích hợp augmentation. Xem [hướng dẫn vận hành](../docs/data_preprocessing/OPERATIONS.md).

## Nội dung bắt buộc

1. Markdown mở đầu: mục tiêu, thay đổi so với run nền, input/output, phụ thuộc và ngày tạo.
2. Cell khởi động: run ID, `started_at` theo `Asia/Bangkok` với UTC offset; git revision/dirty, seed, config, paths/checksum, phiên bản môi trường và host/GPU thực tế.
3. Cells thực hiện: giải thích từng bước, giữ output/log và gọi modules/CLI dùng lại trong repo.
4. Cell kết thúc: `finished_at`, duration, status, kết quả, lỗi và công việc còn thiếu. Nếu lỗi/ngắt, lưu trạng thái đó; không ghi thời điểm như một lần chạy thành công.
5. Lưu dấu vết mỗi lần chạy bằng notebook có output hoặc notebook version/run artifact theo run ID. Khi config/mục tiêu thay đổi đáng kể, tạo version mới; giữ kết quả cũ trước khi rerun.

Ngày tạo khác thời điểm thực thi. Không đưa ảnh, checkpoint hoặc token vào git; notebook lưu thống kê/checksum. Các notebook audit cũ giữ nguyên output của thời điểm chạy; quyết định hiện hành nằm trong báo cáo split và docs/project.

## Vận hành

- Data root mặc định `data/archive`; notebook nhận `AUTOCHECKOUT_DATA_ROOT` khi dataset ở nơi khác. Release ở `data/processed`, log/QA ở `runs`.
- Split notebook dùng `AUTOCHECKOUT_SPLIT_OUTPUT` để tạo release riêng; rerun v1 phải giữ source/config/policy giống nhau. Khi thay input, tạo v2; không tự chia lại holdout.
- Trên Vast.ai, kiểm tra `nvidia-smi`, PyTorch/CUDA và kernel trước train. Lưu checkpoint/config/log/notebook vào vị trí bền vững trước khi kết thúc thuê instance.
- Đợt đầu PDP nền rồi PDP + ConvNeXt-V2-Base. LoRA, K=3, FSA, freeze bổ sung và generator khay thử riêng sau baseline.

Preview ảnh nhúng được lưu trong snapshot đầy đủ ở `runs/notebook_versions/` và gallery/HTML local. Bản notebook vào git giữ code, markdown, timestamps và output nghiệm thu dạng chữ/bảng, tránh chứa ảnh dataset trong lịch sử git.
