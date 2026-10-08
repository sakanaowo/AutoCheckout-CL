# Nghiệm thu tiền xử lý RPC — 07/10/2026

**Kết luận tại lần audit:** tài nguyên ảnh/background/cutout đạt kiểm tra kỹ thuật và có thể tái sử dụng; toàn bộ tiền xử lý chưa đạt vì mask–bbox còn lệch, protocol/validation/provenance chưa được khóa và mức che khuất chưa đạt mục tiêu handoff nếu giữ định nghĩa diện tích mask bị che.

**Cập nhật sau audit:** người dùng đã chốt 5 task; [notebook 02](../../../notebooks/data_preprocessing/02_split_and_acceptance.ipynb) chạy 15:49:50–15:50:21 UTC+7 và nghiệm thu split/task JSON **ảnh thật**. [Báo cáo release](split-100-4x25-2026-10-07.md) là trạng thái hiện hành về protocol/holdout. Ma trận bên dưới giữ kết quả lúc audit; các gate nhãn/provenance/overlap nguồn synth và processor vẫn chưa khép. [Notebook 03](../../../notebooks/data_preprocessing/03_annotation_review.ipynb) đã tạo queue 222 case, chưa áp sửa.

Notebook đã chạy: [01b_preprocessing_acceptance.ipynb](../../../notebooks/data_preprocessing/01b_preprocessing_acceptance.ipynb). Run ID `2e18812d-f053-4fad-83fc-67af1ac0b9b6`. Audit chính **11:00:42–11:06:15 ngày 07/10/2026, UTC+7**, 333 giây; chẩn đoán geometry **2026-10-07T11:07:31+07:00–2026-10-07T11:07:37+07:00**; QA/tolerance **2026-10-07T11:09:49+07:00–2026-10-07T11:09:52+07:00**. Máy phát triển, không train hoặc chạy GPU Vast.ai.

## 1. Ma trận nghiệm thu

| Tiêu chí | Kết quả đo | Trạng thái |
|---|---|---|
| Background tồn tại và đọc được | 208 ảnh 1840×1840; thêm 1 preview được loại khỏi danh sách nền | Đạt kỹ thuật |
| Nguồn background | 208 tên khớp val2019, 0 khớp test2019 | Đạt kiểm tra tên nguồn; chưa có manifest nguồn/config riêng |
| Cutout/mask SAM | 7.502 RGBA, alpha nhị phân không rỗng, đủ 200 SKU, 10–48 cutout/SKU | Đạt kỹ thuật và ánh xạ; chưa chứng minh mọi mask tách vật hoàn hảo |
| Ảnh dùng để train/test hiện hữu | Full pixel decode + SHA-256 của 57.710 file; 0 lỗi đọc/kích thước/alpha | Đạt phạm vi đã kiểm tra |
| Annotation synth | 20.000 ảnh, 238,013 đối tượng; đúng SKU task và source cutout, không thiếu ảnh | Đạt class/path/area/bbox bounds |
| Polygon mask và bbox nhất quán | 222 nhãn IoU <0.98 sau tolerance (0.093%) | Chưa đạt theo checker mask |
| Che khuất 20–45% diện tích mask | Visible 0.85–1.00; mức che tối đa 15%, không có vật ở khoảng 20–45% | Chưa đạt mục tiêu handoff nếu áp dụng |
| Trùng nội dung train/test | Không có SHA-256 file trùng giữa val2019/synth và test2019; không trùng suffix hoặc SKU multiset giữa hai tập ảnh thật | Đạt các phép đo này; không thay thế kiểm tra mọi near-duplicate |
| Validation độc lập và protocol | Import có 4 task; repo có 5 task. Chưa có validation manifest riêng của run mới | Chờ chốt và tạo split |
| Provenance ảnh ghép | Mỗi vật có cutout source, nhưng không có background/config/seed/transform từng ảnh | Chưa đủ truy vết/tái tạo run |
| GRN/feature collapse | Đây là thuộc tính model, không nghiệm thu bằng việc có ảnh background | Kiểm tra ở bước model |

57.710 file gồm 208 background, 7.502 cutout được index chấp nhận, 6.000 ảnh val2019, 24.000 ảnh test2019 và 20.000 ảnh synth. Không bao gồm ảnh preview/review hoặc ảnh train2019 đang thiếu.

## 2. Các sai khác mask/bbox cần xử lý

| Task | Ảnh synth | Đối tượng | Số cảnh báo theo checker nguồn |
|---|---:|---:|---:|
| 1 | 5,000 | 60,473 | 17 |
| 2 | 5,000 | 55,449 | 74 |
| 3 | 5,000 | 60,743 | 84 |
| 4 | 5,000 | 61,348 | 49 |

Checker nguồn báo 224 trường hợp. Kiểm tra lại với tolerance 1e-9 loại 2 trường hợp đúng sát ngưỡng 0.98, còn **222** sai khác thực sự theo tiêu chí đó; **52** có IoU <0.9. IoU nhỏ nhất **0.727**, độ lệch cạnh lớn nhất **137 px**. Không thể giải thích tất cả bằng làm tròn 1–2 pixel.

Source `rpc_common.py` tính bbox từ owner-mask trước khi lọc contour nhỏ/rút gọn polygon; đây là một nguyên nhân có thể gây lệch. Preview các trường hợp nặng cho thấy bbox phủ vùng rộng hơn phần polygon giữ lại. Vì owner-mask gốc không được lưu, chưa thể xác định bbox hay polygon là biểu diễn đúng cho từng trường hợp. Không tự sửa annotation nguồn bằng bbox polygon rồi coi lỗi đã hết.

Cách khép tiêu chí: lưu mask/RLE đầy đủ hoặc tính bbox/area từ cùng mask cuối, kiểm tra lại các nhãn bị lệch và cập nhật các JSON real+synth/tasks_pdp dùng cùng ảnh. Nếu chỉ dùng bbox detector, cần quyết định có căn cứ về annotation bbox và ghi ngoại lệ; không tuyên bố segmentation đã nghiệm thu.

## 3. Chất lượng nền, cutout và augmentation

Đã xem preview mẫu gồm 12 nền, 20 cutout, 12 ảnh synth và 5 trường hợp mask/bbox lệch nặng. Nền giữ biến thiên sáng của khay nhưng một số mẫu còn mảng màu/đốm mờ sau inpainting; cần QA thêm trước khi xem là khay trống hoàn hảo. Alpha nhị phân và SAM IoU metadata không chứng minh cutout không dính mâm hoặc cắt mất bao bì. `rejected.json` có 2.098 trường hợp lọc tự động (2.043 IoU, 55 fill), chưa thấy reason `manual`; điều này không chứng minh đã review tất cả cutout bằng mắt.

Source generator có horizontal flip xác suất 50%; đây là augmentation cần cân nhắc với chữ/logo SKU. Archive không lưu transform từng vật, nên không suy ra chắc chắn mọi ảnh được tạo đúng bằng source/config hiện có. Khi sinh lại, ghi policy flip và transform vào provenance; đối chiếu thật/ghép bằng notebook.

## 4. Background và bão hòa kênh là hai tiêu chí khác nhau

Background là vật liệu ghép ảnh và giữ bối cảnh môi trường. **GRN** trong ConvNeXt-V2 là layer tăng cạnh tranh giữa các kênh, được paper đưa vào để giảm feature collapse. Không có tiêu chí “đủ background thì GRN đã xử lý bão hòa kênh”. Cần kiểm tra backbone có GRN, thống kê activation/độ đa dạng feature và đối sánh khi train ở bước model. Nguồn chính: [paper ConvNeXt-V2](https://arxiv.org/abs/2301.00808), [implementation chính thức](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py).

## 5. Phần có thể tái sử dụng và phần còn thiếu

- Giữ 208 background, 7.502 cutout và 20.000 synth làm nguồn đã có; không cần mặc định chạy lại SAM hay tạo lại nền từ đầu.
- Thiếu `train2019` chỉ chặn việc tái trích cutout từ ảnh gốc hoặc dùng ảnh đơn; **ghép lại từ cutout đã có vẫn thực hiện được**.
- Khép các lỗi mask/bbox, chốt policy overlap/flip, lưu source background/config/seed/transform khi cần sinh lại.
- Chốt 4 hay 5 task, ánh xạ model label và tạo validation độc lập theo nhóm. Với holdout từ val2019, phải kiểm tra nguồn background/statistics dùng để sinh ảnh ghép, không chỉ kiểm tra tên file train/test.
- Kiểm tra resize/normalization/bbox round-trip 640/800 trong notebook processor trước train; audit này chưa chạy processor model.

Artifact chi tiết ở `runs/preprocessing_acceptance/2e18812d-f053-4fad-83fc-67af1ac0b9b6` (thư mục runs được gitignore): image manifest/checksum, summary, geometry diagnostic, danh sách nhãn lỗi và PNG QA. Notebook lưu output và preview để bàn giao. Dữ liệu nguồn không bị chỉnh sửa; đây là kết quả kiểm tra, chưa phải kết quả sửa lỗi hoặc training.
