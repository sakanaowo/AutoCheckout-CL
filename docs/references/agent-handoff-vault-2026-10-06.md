---
type: output
status: evergreen
domains:
  - "[[Computing]]"
tags:
  - handoff
  - context-transfer
  - autocheckout
  - continual-learning
created: 2026-10-06
updated: 2026-10-06
---

# Báo cáo Chuyển giao Tri thức Toàn diện — Dự án AutoCheckout-CL

Tài liệu này tổng hợp toàn bộ bối cảnh hệ thống, tôn chỉ nghiên cứu, cơ sở lý thuyết toán học, các quyết định kiến trúc đã chốt, bản đồ mã nguồn và lộ trình hành động của dự án **AutoCheckout-CL** nhằm phục vụ việc chuyển giao tri thức (*knowledge handoff*) liền mạch cho các agent tiếp quản.

---

## 1. Tôn chỉ Dự án & Bối cảnh Nghiên cứu (Core Identity & Scope)

- **Tên dự án:** AutoCheckout-CL (Smart Retail Checkout with Replay-Free Class-Incremental Object Detection).
- **Vấn đề cốt lõi:** Nhận diện và đếm số lượng 200 mã sản phẩm (SKU) tại quầy thanh toán tự động thông minh bằng camera góc nhìn từ trên xuống (*top-down view*) trên tập dữ liệu RPC (Retail Product Checkout).
- **Thiết lập Học liên tục (Continual Learning):**
  - **Replay-Free tuyệt đối:** Không lưu trữ bất kỳ hình ảnh hay bounding box nào của các đợt hàng cũ trong bộ nhớ đệm (tránh vi phạm quyền riêng tư và chi phí lưu trữ).
  - **Quy chuẩn chia Task (100 + 4x25):** Task 1 học 100 SKU nền tảng; 4 task tiếp theo mỗi task học tăng dần 25 SKU mới (tổng cộng 200 SKU).
- **Quy tắc Nghiên cứu Độc lập (Independent Research Branch):**
  - **Không phụ thuộc Cloud/GCP:** Toàn bộ quá trình tiền xử lý, huấn luyện và đánh giá được chuẩn hóa để chạy trực tiếp trên hạ tầng phần cứng cục bộ.
  - **Không sử dụng kết quả thí nghiệm cũ:** Bỏ qua các bảng kết quả cũ (E0, E1, E4, E5) của repository tác giả ban đầu; xây dựng lại đường cơ sở khoa học độc lập, minh bạch và có thể tái lập hoàn toàn.

---

## 2. Hồ sơ Hạ tầng Phần cứng & Tham số Huấn luyện (Hardware & Training Profile)

- **Hạ tầng cục bộ:** 1x NVIDIA GeForce RTX 4090 (24GB VRAM).
- **Độ phân giải hình ảnh:**
  - Khảo sát thực tế quầy POS: Camera thực tế thường có độ phân giải $720p - 1080p$, trường nhìn rộng, góc nhìn xiên nhẹ, ánh sáng lóa và mờ do chuyển động tay người.
  - Độ phân giải chuẩn huấn luyện thực tế: $640 \times 640$ (điểm cân bằng tối ưu giữa tốc độ, bộ nhớ VRAM và khả năng nhận diện).
  - Độ phân giải benchmark học thuật: $800 \times 800$ (phục vụ đối sánh công bằng với các bài báo CVPR/ICCV).
- **Chiến lược Batch Size & Động lực Học:**
  - Không cố định batch size 64: Tập dữ liệu RPC mỗi task chỉ có khoảng $2,000 - 3,000$ ảnh; nếu dùng batch size 64, mỗi epoch chỉ có khoảng 46 optimizer steps (~280 steps trên 6 epochs), khiến thuật toán AdamW bị underfitting nghiêm trọng.
  - Cấu hình linh hoạt: Sử dụng batch size thực tế $4, 8, 16$ kết hợp Gradient Accumulation để đạt $1,100 - 2,200$ optimizer steps mỗi task, đảm bảo các tham số PEFT và Prototype Memory hội tụ ổn định.

---

## 3. Kiến trúc Mô hình & Các Quyết định Kỹ thuật Đã Chốt

### 3.1 Backbone Thị giác (TODO 1.1)
- **Kiến trúc đã chốt:** **`ConvNeXt-V2-Base`** (`convnextv2_base.fcmae_ft_in22k_in1k`, ~89M tham số).
- **Cơ sở khoa học:**
  - 200 SKU bán lẻ có đặc trưng bao bì biến thiên rất nhỏ (hạt mịn), các backbone nhỏ (Nano/Tiny ~28M tham số) không đủ dung lượng biểu diễn (*representational capacity*), dễ dẫn đến sụp đổ đặc trưng (*feature collapse*) khi trải qua 5 task liên tục.
  - Kiến trúc tích chập 4 tầng phân cấp (4-stage hierarchy: $4\times, 8\times, 16\times, 32\times$) khớp tự nhiên với Deformable DETR Feature Pyramid Network (FPN).
  - Cơ chế chuẩn hóa phản hồi toàn cục (**Global Response Normalization - GRN**) triệt tiêu hiện tượng bão hòa kênh (*channel dead/saturation*), duy trì độ dẻo (*plasticity*) qua các task gia tăng.

### 3.2 Cơ chế Thích ứng Tham số & Vai trò Shared Pool (TODO 1.2)
- **Bản chất của Shared Pool ($P_s$):**
  - Shared Pool học hệ tọa độ không gian và phân bố nền của quầy thanh toán (viền khay, mặt đáy, điều kiện chiếu sáng).
  - **Quy tắc tiền xử lý:** Tuyệt đối không cắt bỏ bối cảnh môi trường xung quanh món hàng. Nếu loại bỏ môi trường, 285 query trống trong Deformable DETR sẽ mất điểm tựa nền, dẫn đến kích hoạt báo động giả (*false positive*) tràn lan trên khay thực tế và làm mất thông tin thứ tự xếp chồng (Z-order).
- **Cơ sở toán học của Prompt Tuning:**
  - **Tách biệt Không gian Con (Subspace Orthogonality):** Sử dụng hàm mất mát đa dạng hóa $L_{\text{DDL}}$ ép góc giữa Shared Pool và Private Pool đạt $\cos(P_s, P_p) \le 0$, ngăn chặn triệt để hiện tượng nhiễu gradient (*gradient interference*): $\langle \mathbf{g}_t, \mathbf{g}_{<t} \rangle \ge 0$.
  - **Lưỡng nan Động - Ổn (Plasticity-Stability Dilemma):** Đóng băng vĩnh viễn Shared Pool sau khi kết thúc Task 1 để neo giữ tri thức không gian quầy hàng, chỉ cho phép Private Pool thích ứng với các SKU mới.
  - **Low-Rank Adaptation (LoRA):** Mở rộng áp dụng LoRA ($r=8, \alpha=16$) lên các ma trận biến đổi $W_q, W_v$ của Cross-Attention trong Transformer Decoder, đảm bảo ổn định dung lượng khi số lớp tăng lên 200 SKU.

### 3.3 Khử Trùng lặp Dự đoán bằng Class-Agnostic NMS (TODO 4.1)
- **Sự sụp đổ của giả định "NMS-Free" của DETR:** Bipartite Matching (Hungarian Loss) chỉ ép ràng buộc $1-1$ trên tập huấn luyện; trong quầy hàng mật độ cao, các queries lân cận cùng rơi vào vùng hấp dẫn (*basin of attraction*) của cùng một món hàng và cùng kích hoạt điểm cao ($> 0.6 - 0.8$).
- **Bắt buộc dùng Class-Agnostic NMS:**
  - Phân loại hạt mịn khiến các queries lân cận dễ gán 2 nhãn SKU anh em khác nhau cho cùng một sản phẩm thực tế (ví dụ: Snack vị Muối vs Snack vị BBQ).
  - Per-Class NMS thông thường sẽ giữ lại cả 2 bounding box, khiến quầy thu ngân tính tiền khách 2 lần cho 1 món.
  - **Nguyên lý Bài trừ Không gian Vật lý (Physical Space Exclusion):** Hai vật thể rắn không thể chiếm cùng một không gian nhìn từ trên xuống; ép lọc **Class-Agnostic NMS** với ngưỡng $\text{IoU} \ge 0.45$, chỉ giữ duy nhất 1 box có điểm tự tin cao nhất.

### 3.4 Tối ưu Ngưỡng & Bộ Chỉ số Bán lẻ Chuẩn (TODO 4.2)
- **Lệch pha giữa mAP và Hàm Thiệt hại Hóa đơn:** mAP tính tích phân diện tích dưới đường cong Precision-Recall, trong khi thanh toán tự động vận hành theo hàm thiệt hại 0-1 tuyệt đối ($\mathcal{L}_{\text{checkout}}$) trên từng khay hàng.
- **Nguyên tắc Giảm thiểu Rủi ro Thực nghiệm (ERM) Chống Rò rỉ Dữ liệu:**
  - Quét lưới tìm ngưỡng cắt điểm tối ưu $s^*$ trên tập **Validation độc lập**:
    $$s^* = \arg\max_{s \in [0.10, 0.90]} cAcc_{\text{val}}(s)$$
  - Đóng băng $s^*$ và áp dụng cố định sang tập **Test**, ngăn chặn triệt để hành vi rò rỉ dữ liệu (*Oracle Thresholding*).
- **Bộ tứ chỉ số RPC:** Đánh giá đồng thời $cAcc$ (độ chính xác hóa đơn 100%), $ACD$ (khoảng cách sai số đếm trung bình $L_1$), $mCCD$ (sai số đếm theo từng SKU), và $mCIoU$ (tính toàn vẹn không gian và số lượng).

---

## 4. Bản đồ Tài nguyên & Đường dẫn Hệ thống

| Thành phần | Vị trí Đường dẫn | Vai trò & Trách nhiệm |
| :--- | :--- | :--- |
| **Knowledge Vault** | `/home/sakana/dianoia` | Hệ thống quản trị tri thức Obsidian của dự án |
| **Trạm Điều phối Chính** | `06 - Projects/AutoCheckout/TODO.md` | Chứa toàn bộ kế hoạch thực nghiệm, phân tích toán học, thuật toán và mã nguồn chi tiết |
| **Tổng quan Dự án** | `06 - Projects/AutoCheckout/overview.md` | Bức tranh toàn cảnh về Smart Retail Checkout và phân tích paper PDP |
| **README Hub** | `06 - Projects/AutoCheckout/README.md` | Trung tâm điều hướng liên kết và mục tiêu tổng quan |
| **Tài liệu Kỹ thuật** | `06 - Projects/AutoCheckout/docs/` | Thư mục chứa tài liệu định dạng dữ liệu, cẩm nang toán học PDP |
| **Mã nguồn Chính** | `/home/sakana/Code/AutoCheckout-CL` | Repository thực thi (Git branch: `investigation`) |
| **Kịch bản Đánh giá** | `tools/eval_count.py` | Mô-đun dò ngưỡng ERM và tính toán $cAcc, ACD, mCCD, mCIoU$ |
| **Sinh Khay Tổng hợp** | `tools/generate_synthetic_trays.py` | Pipeline copy-paste vật thể đơn lẻ lên khay trống với độ đè lấn $20\% - 45\%$ |

---

## 5. Quy chuẩn Tác vụ cho Agent Kế tiếp (Rules & Operational Constraints)

Bất kỳ agent nào tiếp quản dự án này phải tuân thủ nghiêm ngặt các quy tắc từ `AGENTS.md`:
1. **Liên kết Obsidian Wikilinks tuyệt đối:** Luôn sử dụng cú pháp wikilink hai ngoặc vuông (ví dụ: `[` `[`Tiêu đề ghi chú`]` `]`). Không dùng cú pháp Markdown link chuẩn `[text](path)` cho các liên kết nội bộ trong vault.
2. **Loại bỏ hoàn toàn Emoji và Biểu tượng Trang trí:** Không chèn emoji trong tiêu đề, danh sách, bảng biểu hay nội dung ghi chú (Quy tắc 13 trong `AGENTS.md`). Duy trì phong cách học thuật, tối giản và chuyên nghiệp.
3. **Ngôn ngữ:** Ưu tiên tiếng Việt học thuật, diễn đạt chính xác các thuật ngữ khoa học máy tính và toán học.
4. **Kiểm tra Toàn vẹn:** Sau mỗi lần chỉnh sửa file Markdown trong vault, luôn chạy script kiểm tra:
   ```bash
   python3 scripts/check_vault.py
   ```
5. **Quy tắc An toàn:** Không xóa file hàng loạt; không đổi tên file khi không có yêu cầu rõ ràng; giữ nguyên cấu trúc phân cấp tri thức hiện có.

---

## 6. Lộ trình Hành động Kế tiếp (Next Action Items)

1. **Mã nguồn Mô hình (`/home/sakana/Code/AutoCheckout-CL`):**
   - Tạo file `pdp/models/backbones.py` tích hợp `ConvNeXt-V2-Base` từ thư viện `timm` kèm các tầng chiếu kênh (Channel Projection) thích ứng với Deformable DETR FPN.
   - Triển khai lớp bọc LoRA cho Transformer Decoder Cross-Attention.
   - Cập nhật thuật toán Spherical K-Means ($K=3$) cho bộ nhớ Prototype trong `pdp/engine.py`.
2. **Kịch bản Sinh dữ liệu Khay hàng Tổng hợp:**
   - Triển khai `tools/generate_synthetic_trays.py` để trích xuất mặt nạ phân đoạn (SAM masks) từ ảnh vật thể đơn và tổng hợp lên khay trống nhằm giải quyết khoảng cách miền (*domain gap*).
3. **Thực nghiệm Khoa học:**
   - Tiến hành chạy thực nghiệm **EXP-B2** (ConvNeXt-V2-Base trên Task 1: 100 SKU đầu tiên) trên GPU RTX 4090 cục bộ với batch size 8/16.
   - Ghi nhận đường cong hội tụ, mức tiêu hao VRAM và đánh giá kết quả ban đầu bằng `tools/eval_count.py`.
