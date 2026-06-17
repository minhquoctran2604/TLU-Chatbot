# Đặc tả Thiết kế: Nhận diện Thương hiệu Đại học Thủy lợi

## Tổng quan
Tài liệu đặc tả thiết kế này hướng dẫn tái cấu trúc giao diện LightRAG WebUI mặc định để phù hợp với bản sắc thương hiệu của Trường Đại học Thủy lợi (TLU). Giao diện sẽ chuyển đổi từ phong cách mặc định dành cho lập trình viên (sử dụng tông màu xanh lá/emerald) sang ngôn ngữ thiết kế học thuật, tập trung vào tài nguyên nước và kỹ thuật bền vững (sử dụng màu xanh Cobalt, màu vàng lúa chín làm điểm nhấn, bố cục trang nhã).

## Đặc tả Giao diện hiển thị

### 1. Hệ thống Màu sắc Thương hiệu (TLU Colors)

Chúng ta ánh xạ các màu nhận diện chính thức của TLU sang các giá trị HSL để thay thế biến CSS thuận tiện:

| Vai trò màu | Tên màu | Mã Hex | Định dạng HSL | Ánh xạ biến CSS |
|---|---|---|---|---|
| **Chủ đạo (Primary)** | Xanh Cobalt TLU | `#004F95` | `208 100% 29%` | `--primary`, các điểm nhấn active |
| **Điểm nhấn (Accent)** | Vàng TLU | `#F2A900` | `42 100% 47%` | Nhãn làm nổi bật, dấu sao chỉ thị |
| **Cảnh báo (Alert)** | Đỏ cờ | `#D21F3C` | `350 74% 47%` | `--destructive`, cảnh báo lỗi |
| **Nền sáng** | Trắng tinh | `#FFFFFF` | `0 0% 100%` | `--background`, `--card` |
| **Nền tối (Dark Mode)**| Xanh đen đại dương | `#0A192F` | `217 64% 11%` | `--background` (chế độ tối) |

### 2. Đặc tả Header và Thanh Điều hướng (Navigation)

- **Biểu tượng Logo**: Thay thế `ZapIcon` (tia sét) bằng biểu tượng `Droplet` (Giọt nước - đại diện cho ngành Thủy lợi) hoặc một thành phần logo tùy chỉnh hiển thị đúng hình dáng logo tròn của TLU.
- **Tab điều hướng hoạt động (Active)**:
  - Thay đổi class: Đổi `!bg-emerald-400 !text-zinc-50` thành `!bg-blue-600 !text-white` hoặc sử dụng giá trị biến HSL tương đương của TLU.
  - Trạng thái hover: Hover vào các tab không hoạt động sẽ chuyển sang màu xanh lam nhạt (`bg-blue-50/50` ở chế độ sáng, `bg-slate-800/50` ở chế độ tối).

### 3. Đặc tả Trang Đăng nhập (Login Page)

- **Gradient Nền**:
  - Chế độ sáng: Đổi từ `from-emerald-50 to-teal-100` thành `from-blue-50 to-indigo-100`.
  - Chế độ tối: Đổi từ `from-gray-900 to-gray-800` thành `from-slate-950 to-slate-900`.
- **Phần thông tin thương hiệu**:
  - Tiêu đề chính: Thay đổi `"LightRAG"` thành `"TLU Chatbot"` hoặc `"Cổng Tư Vấn TLU"`.
  - Mô tả phụ: Thay đổi thành: `"Hệ thống Hỗ trợ Học tập & Tuyển sinh - Trường Đại học Thủy lợi"`.

### 4. Đặc tả Biểu đồ Đồ thị Kiến thức (Knowledge Graph)

Để đồng bộ với thương hiệu TLU:
- Cập nhật các màu sắc của nút và cạnh khi được chọn trong tệp tin [constants.ts](file:///home/tts/AI/aiQuoc/TLU-Chatbot/lightrag_webui/src/lib/constants.ts):
  - Thay đổi `nodeBorderColorSelected` từ `#F57F17` (Cam) thành màu vàng TLU (`#F2A900`).
  - Thay đổi màu viền cạnh khi được chọn thành màu xanh Cobalt TLU (`#004F95`) hoặc xanh dương sáng (`#3B82F6`).

## Bố cục Đáp ứng (Responsive)

| Kích thước màn hình | Hành vi |
|---|---|
| Điện thoại (<640px) | Thanh điều hướng thu gọn thành các tag rút gọn; tiêu đề hiển thị ngắn gọn `"TLU"` thay vì tên đầy đủ. |
| Máy tính (>1024px) | Hiển thị đầy đủ tiêu đề `"TLU Chatbot"`; thanh điều hướng hiển thị đầy đủ nhãn. |

## Khả năng Tiếp cận (Accessibility)

- **Kiểm tra độ tương phản**: Chữ màu xanh chủ đạo `#004F95` trên nền trắng đạt tỷ lệ tương phản `6.54:1`, đáp ứng tốt tiêu chuẩn WCAG AA (`>= 4.5:1`).
- **Chỉ thị Focus**: Các vòng focus mặc định khi điều hướng bằng bàn phím sẽ sử dụng màu xanh `ring-blue-500/50` thay cho màu xám hoặc màu xanh emerald mặc định.
