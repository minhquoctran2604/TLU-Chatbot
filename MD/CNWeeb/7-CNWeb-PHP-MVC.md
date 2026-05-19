![Image](/content/drive/MyDrive/MD/CNWeeb/7-CNWeb-PHP-MVC_artifacts/image_000000_7077877430444fb4b190f710d6a0f2603f4e160e9d6e4974156e13cfdf22bc6d.png)

![Image](/content/drive/MyDrive/MD/CNWeeb/7-CNWeb-PHP-MVC_artifacts/image_000001_85d33cb7539ac4dac89a904f25a9aab71dc7ab0cb76c9b3f129fa450bbbe6414.png)

Giảng viên: Kiều Tuấn Dũng , Nguyễn Tu Trung BM HTTT, Khoa CNTT, Trường ĐH Thủy Lợi

Hà Nội , 2024

## Nôi dung

- ❖ Khái niệm

- ❖ Thành phần

- ❖ Luồng xử lý dữ liệu

- ❖ Cấu trúc thư mục

- ❖ Kiến trúc MVC biến thể

## Khái niệm

- ❖ Là mô hình kiến trúc phần mềm 3 lớp bao gồm Model, View, Controller
- ❖ Tách biệt ứng dụng web ra làm các thành phần riêng biệt, nên thuận lợi cho việc phát triển và bảo trì
- ❖ Phổ biến trong các framework hiện nay (Laravel, Zend, Cake, .v.v)
- ❖ Sử dụng OOP làm nền tảng để xây dựng và phát triển

## Thành phần

## ❖ M – Model:

- ❖ Nhận dữ liệu từ Controller gửi tới
- ❖ Thực hiện thao tác với database
- ❖ Gửi kết quả trả về Controller
- ❖ V – View:
- ❖ Hiển thị dữ liệu từ Controller gửi về cho trình duyệt tại client
- ❖ C – – Controller:
- ❖ Tầng trung gian nhận request từ client
- ❖ Gọi Model thực thi
- ❖ Trả dữ liệu lại cho client thông qua View
- ❖ View định dạng dữ liệu hiển thị cho client

## Luồng xử lý dữ liệu

![Image](/content/drive/MyDrive/MD/CNWeeb/7-CNWeb-PHP-MVC_artifacts/image_000002_ee721bf3537cc75bc3e1c0983053403889c4d017a6caad61c42fdce2b58073d4.png)

## Luồng xử lý dữ liệu

- ❖ Ví dụ: Client truy cập link http://cse.tlu.edu.vn/chi-tiet/3
- ❖ Controller nhận request chứa id của bài viết = 3
- ❖ Controller gọi Model truy vấn database lấy ra nội dung bài viết có id = 3
- ❖ Model truy vấn thành công, gửi nội dung bài viết lại cho Controller
- ❖ Controller gửi nội dung bài viết đó ra View
- ❖ View hiển thị nội dung bài viết tới client

## Cấu trúc thư mục

- ❖ CRUD tương ứng các chức năng: Hiển thị thông tin,

Thêm, Sửa, Xoá

![Image](/content/drive/MyDrive/MD/CNWeeb/7-CNWeb-PHP-MVC_artifacts/image_000003_477b5a7b2d996fd3862d5027a53ea214150b364e65e78a11e5e2b9f156de7eeb.png)

## Kiến trúc MVC biến thể

❖ Cấu trúc thư mục Code theo MVC

NTTrung

Bài giảng CNWeb

![Image](/content/drive/MyDrive/MD/CNWeeb/7-CNWeb-PHP-MVC_artifacts/image_000004_bb3997e3b7b97e86193bf90a1ebbb3621eee41701e7338b36c03aa3230237e23.png)

NTTrung