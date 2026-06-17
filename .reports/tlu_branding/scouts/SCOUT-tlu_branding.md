# Báo cáo Khảo sát: Tích hợp Nhận diện Thương hiệu Đại học Thủy lợi

## Phạm vi Khảo sát
- **Mục tiêu**: Các biến màu sắc, thành phần giao diện hiển thị, yếu tố thương hiệu, logo và các cấu hình định dạng tùy chỉnh.
- **Phạm vi**: `lightrag_webui/src/`, `lightrag_webui/tailwind.config.js`, `lightrag_webui/src/index.css`, `lightrag_webui/src/lib/constants.ts`.

## Các Mẫu Thiết Kế Phát Hiện Được

### Quy ước: Ánh xạ màu sắc qua biến CSS
- **Vị trí**: [index.css](file:///home/tts/AI/aiQuoc/TLU-Chatbot/lightrag_webui/src/index.css)
- **Cách sử dụng**: Được cấu hình bằng các biến HSL chuẩn trong Tailwind / cấu hình tùy chỉnh.
- **Bắt buộc tuân thủ**: Có (Các phần tử giao diện như đường viền, nền, popover, màu chủ đạo, màu phụ đều được liên kết với các giá trị HSL này).

### Quy ước: Màu sắc điểm nhấn và tương tác
- **Vị trí**: [SiteHeader.tsx](file:///home/tts/AI/aiQuoc/TLU-Chatbot/lightrag_webui/src/features/SiteHeader.tsx), [App.tsx](file:///home/tts/AI/aiQuoc/TLU-Chatbot/lightrag_webui/src/App.tsx), [LoginPage.tsx](file:///home/tts/AI/aiQuoc/TLU-Chatbot/lightrag_webui/src/features/LoginPage.tsx)
- **Cách sử dụng**: Các class được viết trực tiếp (hardcoded) sử dụng các màu `emerald-400` / `emerald-500` / `teal-100` của Tailwind cho các tab active, icon logo Zap, gradient nền và màu thông báo thành công.
- **Bắt buộc tuân thủ**: Không (có thể điều chỉnh để phù hợp với nhận diện thương hiệu của TLU).

### Quy ước: Logo & Tên ứng dụng
- **Vị trí**: [constants.ts](file:///home/tts/AI/aiQuoc/TLU-Chatbot/lightrag_webui/src/lib/constants.ts)
- **Cách sử dụng**: `SiteInfo.name` được xuất ra dưới dạng `'LightRAG'`, hiển thị trên tiêu đề header và tiêu đề trang.
- **Tài nguyên Logo**: `public/logo.svg` được hiển thị cùng với `ZapIcon` của Lucide.

## Các Điểm Tích Hợp Thay Đổi
| Điểm tích hợp | Tệp tin | Thành phần / Chức năng | Vị trí mã nguồn mới |
|---|---|---|---|
| Hệ thống màu chủ đạo | `index.css` | Các class `:root`, `.dark` | Định nghĩa các biến HSL cho màu xanh Cobalt của TLU |
| Điểm nhấn Tab hoạt động | `SiteHeader.tsx` | Class active của `NavigationTab` | Đổi `!bg-emerald-400` sang màu xanh dương tùy chỉnh |
| Logo / Icon thương hiệu | `SiteHeader.tsx`, `LoginPage.tsx` | Thẻ `img` của logo / `ZapIcon` | Thay thế bằng SVG logo thương hiệu TLU |
| Tiêu đề ứng dụng | `constants.ts` | Cấu hình `SiteInfo` | Thay đổi thành thông tin thương hiệu TLU Chatbot |
| Nền trang đăng nhập | `LoginPage.tsx` | Phần bao quanh card & nền | Đổi gradient `emerald` sang gradient xanh lam/trắng |

## Các Quy Ước Khác
- Giá trị màu sắc được khai báo bằng các thuộc tính CSS mở rộng của Tailwind.
- Bố cục giao diện sử dụng flexbox, hệ thống lưới chuẩn và các thành phần bao bọc UI dựa trên Radix.

## Cảnh báo
- ⚠️ Việc sửa đổi màu chủ đạo toàn cục trong `index.css` sẽ tự động ảnh hưởng đến trạng thái hiển thị của các nút, đường viền và đồ thị.
- ⚠️ Màu sắc của các nút và cạnh trên đồ thị được định nghĩa một phần trong `lightrag_webui/src/lib/constants.ts` qua mã hex (ví dụ: `nodeBorderColorSelected`, `edgeColorSelected`). Những mã này cũng cần được căn chỉnh đồng bộ với màu chủ đạo mới.
