## Chương 5

Bộ nhớ trong

## Chương 5. Bộ nhớ trong

- 5.1. Bộ nhớ chính bán dẫn
- 5.2. Cơ chế sửa lỗi
- 5.3. Tổ chức bộ nhớ DRAM mở rộng

## 5.1 Bộ nhớ bán dẫn a. Tổ chức

- ◼ Các thành phần chính của BN bán dẫn là các ô nhớ (memory cell)
- ◼ Đặc điểm chính:
- ◼ Có 2 trạng thái biểu diễn 2 bit 0, 1
- ◼ Có khả năng ghi vào (ít nhất một lần)
- ◼ Có khả năng đọc ra
- ◼ Có 3 đầu cuối:
- ◼ Đường select để chọn ra ô nhớ để đọc hoặc ghi
- ◼ Đường điều khiển để chỉ thị thao tác đọc hoặc ghi
- ◼ Đường đưa dữ liệu vào hoặc đọc dữ liệu ra

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000000_570efad6091b3ad79dd271837faf7caa4d433a52b9f749bca46ca065a4ddc2a5.png)

## b. Các loại bộ nhớ bán dẫn

- ◼RAM -Ramdom Access Memory: Bộ nhớ truy cập ngẫu nhiên
- ◼ Bộ nhớ đọc ghi
- ◼ Cơ chế ghi sử dụng tín hiệu điện
- ◼ Cho phép xóa
- ◼ Bộ nhớ điện động
- ◼ROM -Read -only Memory: Bộ nhớ chỉ đọc
- ◼ Bộ nhớ chỉ đọc
- ◼ Trước đây không xóa được . Hiện nay một số loại xóa được nhưng phải sử dụng mạch điện chuyên biệt
- ◼ Bộ nhớ điện tĩnh
- ◼Đặc điểm chung:
- ◼ Phương thức truy cập: truy cập ngẫu nhiên , sử dụng địa chỉ

## c. Bộ nhớ RAM

- ◼Bộ nhớ random -access memory (RAM): cho phép đọc và ghi dữ liệu một cách nhanh chóng , cả đọc và ghi đều sử dụng các tín hiệu điện
- ◼Bộ nhớ RAM là bộ nhớ điện động , khi mất nguồn , dữ liệu bị mất → Chỉ sử dụng RAM với mục đích lưu trữ tạm m thời
- ◼Có hai công nghệ RAM :
- ◼ RAM động - Dynamic RAM (DRAM)
- ◼RAM tĩnh -Static RAM (SRAM)

## DRAM và SRAM

## RAM động - Dynamic RAM (DRAM)

## ◼ DRAM

- ◼ Có các ô nhớ lưu trữ dữ liệu bằng cách nạp cho các tụ điện
- ◼ Điện tích có hoặc không có trên mỗi tụ điện tương ứng với các bit 1 hoặc 0
- ◼ Cần phải nạp điện định kỳ để duy trì lưu trữ dữ liệu
- ◼ Điện tích trên tụ điên bị rò rỉ ngay cả khi có nguồi nuôi → cần có dòng làm tươi (định kỳ)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000001_2beee8ce22edbe7bccd854dbf3967a1acb91751a11bc8b3694ff27eb54718bf1.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000002_30981b64c33ba4e2ad1380dcfb74b831c81f7862042d0ba3cda6df121e225833.png)

## DRAM và SRAM

## RAM tĩnh -Static RAM (SRAM)

- Thiết bị số sử dụng các thành phần logic giống nhau trong bộ xử lý
- Các giá trị nhị phân được lưu trữ trong các cổng logic flip-flop truyền thống .
- Giữ được dữ liệu đến khi nào còn có nguồn cung cấp cho nó

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000003_7623ac4ca1337df4e01e707aa3fc75b73f5d62bff6f4f6a287397f62a69070f8.png)

## So sánh SRAM và DRAM

- ◼ Đều là bộ nhớ điện động
- ◼ Phải được nối với nguồn liên tục để ô nhớ lưu trữ được giá trị bit.
- ◼RAM động
- ◼ Dễ dàng chế tạo , kích thước nhỏ hơn
- ◼Mật độ lớn hơn(các ô nhớ nhỏ hơn → nhiều ô nhớ hơn trong một đơn vị diện tích)
- ◼Giá thành rẻ hơn
- ◼Đòi hỏi phải hỗ trợ dòng làm tươi
- ◼Thích hợp với các bộ nhớ dung lượng cao
- ◼Được sử dụng cho bộ nhớ chính
- ◼RAM Tĩnh
- ◼Nhanh hơn
- ◼Được sử dụng làm bộ nhớ đệm (cache) (cả trong và ngoài chip)

## d . Bộ nhớ ROM (Read Only Memory)

- ◼ Chứa dữ liệu vĩnh cửu , không thể thay đổi hay thêm vào
- ◼ Không cần cung cấp nguồn để duy trì giá trị bit trong bộ nhớ
- ◼ Dữ liệu hay chương trình lưu trữ vĩnh cửu trong bộ nhớ chính và không cần thiết phải tải từ thiết bị lưu trữ thứ hai
- ◼ Ứng dụng:
- ◼ Vi lập trình
- ◼ Lưu trữ các file hệ thống
- ◼ Dữ liệu thực tế được nạp với chip như một phần của chu trình sản xuất chip.
- ◼ Nhược điểm của điều này:
- ◼ Không có chỗ cho lỗi , nếu sai một bit thì toàn bộ lô ROM sẽ bị vứt đi
- ◼ Việc nạp dữ liệu vào ROM tốn một khoản chi phí cố định khá lớn

## Các loại BN ROM

## ◼ PROM (Programmable ROM) – ROM có thể lập trình được

- Việc thay thế ít tốn kém hơn
- Không xóa được và chỉ có thể ghi một lần duy nhất
- Quá trình ghi được thực hiện bằng điện do nhà cung cấp hoặc khách hàng thực hiện tại thời điểm sau thời điểm sản xuất chip
- Cần có thiết bị ghi đặc biệt để thực hiện quá trình ghi
- Linh hoạt và tiện lợi
- Thích hợp với chu trình sản xuất một số lượng lớn

## ◼ EPROM

- Bộ nhớ PROM có thể xóa được
- Quá trình xóa có thể thực hiện nhiều lần
- Đắt hơn so PROM nhưng có ưu điểm do khả năng cập nhật lại

## Các loại BN ROM

## ▪ EEPROM

- Bộ nhớ PROM xóa bằng điện
- Có thể ghi vào bất cứ thời điểm nào mà không cần phải xóa dữ liệu trước đó
- Kết hợp ưu điểm của việc không xóa được và sự linh hoạt của việc cập nhật tại chỗ
- Đắt hơn so với EPROM

## ▪ Flash Memory

- Trung gian giữa EPROM và EEPROM
- Sử dụng cộng nghệ xóa điện , không cho phép xóa cấp độ byte
- Tốc độ xóa nhanh hơn
- Mỗi cell chỉ sử dụng 1 transistor, mật độ cell lớn hơn các bộ nhớ trên

## Các loại bộ nhớ bán dẫn

Table 5.1 Semiconductor Memory Types

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000004_cb8138c956cb911382c5534d2100656c60db07d2ccc0635cbefbc51e02c7272d.png)

## e. Tổ chức chip bộ nhớ

- ◼ Cùng với công nghệ mạch tích hợp , bộ nhớ bán dẫn cũng được sản xuất dưới dạng chip đóng gói. Trong đó , các ô nhớ được tổ chức dưới dạng ma trận nhớ .
- ◼Một số vấn đề khi thiết kế Chip bộ nhớ:
- ◼Cân đối giá thành , tốc độ, dung lượng
- ◼ Số lượng bit được đọc , ghi cùng một lúc

+

## Tổ chức bộ nhớ một chiều

- ◼ Các đường địa chỉ: 𝐴 𝑛−1 ÷ 𝐴 0
- → có 2
𝑛
từ nhớ
- ◼ Các đường dữ liệu: 𝐷 𝑚−1 ÷ 𝐷 0
- → độ dài từ nhớ = m bit
- ◼ Dung lượng chip nhớ = 2 𝑛 × 𝑚 bit
- ◼ Các đường điều khiển:
- ◼ Tín hiệu chọn chip CS (Chip Select)
- ◼ Tín hiệu điều khiển đọc OE (Output Enable)
- ◼ Tín hiệu điều khiển ghi WE (Write Enable)
- ◼ Các tín hiệu điều khiển thường tích cực với mức 0)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000005_7102b195b7cfa0a7d426e3b2497b5946e74a241baa5eb424d40abdceb14f2004.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000006_f083bcddfe350e064b48f248e2a2475b3ff250f8ebadc403e4dabeb77aaf7ee0.png)

## + Ví dụ tổ chức BN RAM: DRAM 16Mb

- ◼VD: Tổ chức bộ nhớ DRAM 16Mb (mô hình tương tự với ROM và các bộ nhớ trong khác)
- ◼ Tổ chức thành 4 ma trận nhớ 2048 x 2048
- ◼11 đường địa chỉ cho hàng
- ◼11 đường đc cho cột , mỗi cột 4 bit
- ◼ Sử dụng ghép kênh để giảm số chân địa chỉ của DRAM → thêm 1 chân thì dung lượng DRAM tăng 4 lần
- ◼Mạch làm tươi: sử dụng bộ đệm , đọc dữ liệu ra sau đó ghi vào chính vị trí đó

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000007_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000008_7ea4ecc77deab3d042728305d4a5cd06ea7c2f3f6d30a8b84e43cc77e2c11b1c.png)

## f. f. Đóng gói chip

- ◼ Công nghệ mạch tích hợp: đóng gói chip thành các IC và có các chân để giao tiếp dữ liệu với bên ngoài
- ◼ Chip EPROM 8-Mbit: tổ chức thành 1M x 8, 32 chân:
- ◼ 20 chân địa chỉ (A0 – A19)
- ◼ 8 chân dữ liệu (D0 – D7)
- ◼ chân cấp nguồn Vcc
- ◼ chân nối đất Vss
- ◼ Chân cho phép hoạt động: Chip enable -CE
- ◼ Chân Vpp: cung cấp trong quá trình lập trình (hoạt động ghi)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000009_1b52d2bac62c790245d90c28fec99683e230156e7d9574d54fa1a3d93a3cb18b.png)

## Đóng gói chip

## ◼ Chip DRAM

- ◼ 11 chân địa chỉ (A0 – A10), 4 chân dữ liệu (D0 – D3)
- ◼ Chân cho phép ghi WE (write enable)
- ◼ Chân cho phép đọc OE (output enable)
- ◼ Chân chọn địa chỉ hàng RAS và cột CAS
- ◼ Chân không có tín hiệu NC

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000010_c15dc69355df344de4ce86fd5561ed0182c71192e23e24e9ad83b0467647c8a6.png)

## Tổ chức module 1MByte

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000011_666c47486933f5e212d3c427d37f2e93c848dfc7182f6a639d6547fcb7b107e6.png)

## Tổ chức bộ nhớ đan xen (interleaved memory)

- ◼ Bao gồm một tập hợp các chip DRAM
- ◼ Nhóm lại thành một dải bộ nhớ
- ◼ Mỗi dải có thể độc lập phục vụ một yêu cầu đọc, ghi bộ nhớ
- ◼ K dải có thể phục vụ K yêu cầu đồng thời, tốc độ đọc, ghi bộ nhớ tăng theo tỷ lệ của K
- ◼ Nếu các từ liên tiếp của bộ nhớ được lưu trữ ở các dải khác nhau, việc truyền một khối của bộ nhớ sẽ được đẩy nhanh .

## 5.2. Cơ chế sửa lỗi -Error Correction

- ◼ Việc lưu trữ và truyền dữ liệu trong hệ thống máy tính có thể xuất hiện các lỗi
- ◼ Lỗi cứng
- ◼ Lỗi vật lý vĩnh viễn
- ◼ Một hoặc nhiều ô nhớ không thể lưu trữ dữ liệu , bị kẹt ở giá trị 0 hoặc 1 hoặc không thể chuyển giữa 0 hoặc 1 bất cứ lúc nào .
- ◼ Nguyên nhân:
- ◼ Do tác động của môi trường
- ◼ Lỗi sản xuất
- ◼ Do hao mòn dần
- ◼ Lỗi mềm
- ◼ Sự kiện ngẫu nhiên làm thay đổi nội dung của một hay nhiều ô nhớ
- ◼ Không phá hủy bộ nhớ vĩnh viễn
- ◼ Nguyên nhân:
- ◼ Có vấn đề về nguồn điện
- ◼ Ảnh hưởng của các hạt phóng xạ
- ◼ Rõ ràng , cả lỗi cứng và lỗi mềm là không mong muốn , hầu hết bộ nhớ chính hiện đại đều có cơ chế phát hiện và sửa lỗi

## Chức năng mã sửa lỗi – Error -Correcting Code

- ◼ Hàm f tính toán M bit dữ liệu và sinh ra một mã K bit
- ◼ M bit dữ liệu và K bit mã cùng được lưu trữ
- ◼ Khi đọc bộ nhớ , sử dụng hàm f tính lại mã trên dữ liệu lấy ra , so sánh với K bit mã lưu trữ. 3 trường hợp xảy ra:
- ◼ Không phát hiện ra lỗi . Dữ liệu được gửi đi
- ◼ Phát hiện ra lỗi , có thể sửa lỗi . Dữ liệu và bit sửa lỗi được đưa vào bộ sửa lỗi sau đó được gửi đi
- ◼ Một lỗi được phát hiện nhưng không thể sửa . Lỗi sẽ được báo cáo
- ◼ Mã này được gọi là mã CRC

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000012_68d17084e17c7aa9042885965f4c71c1374babdd3f45c309d4ab162bb45446ec.png)

## Mã sửa lỗi Hamming

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000013_df4cbce6ae3cc0c1ac927fc9797cdfb90db4f60fb5d2f854011a68bd902f0674.png)

- ◼ Mã đơn giản nhất mã: Hamming do Richard Hamming đưa ra tại Bell Laboratories.
- ◼ Ví dụ việc tính toán mã này trên các từ nhớ 4 bit (M = 4).
- ◼ Với ba vòng tròn giao nhau ta có bảy khoang .
- ◼ Gán 4 bit dữ liệu cho các ngăn bên trong (hình 5.8a).
- ◼ Các ngăn còn lại được điền vào các bit 0 hoặc 1 với nguyên tắc làm sao để tổng các bit trong một vòng tròn là số chẵn .
- ◼ Phát hiện lỗi , tính tổng của mỗi vòng , ta thấy vòng A và C có tổng lẻ, trong7 khoanh chỉ có 1 khoanh nằm trong cả 2 vòng: A,C → phát hiện lỗi →sửa lỗi

## Số lượng bit kiểm tra

Với 8 bit dữ liệu cần ít nhất 4 bit check, sinh ra 4 bit syndrome (đầu ra của bộ compare) có đặc điểm như sau:

- Nếu syndrome toàn 0, không có lỗi .
- Nếu syndrome có 1 bit 1, lỗi nằm ở 4 bit check, ko cần sửa
- Nếu syndrome chứa hơn 1 bit 1, giá trị của syndrome chỉ ra vị trí của bit lỗi , sửa lỗi

## Bố cục các bit dữ liệu và các bit kiểm tra

## Tính toán bit kiểm tra

## Mã Hamming SEC-DED

- ◼ Mã Phát hiện 2 lỗi – – sửa một lỗi (Single Error Correcting – – Double Error Detecting)
- ◼ Sử dụng thêm 1 bit parity (bit chẵn lẻ)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000014_8c4d23fbc287aa7a9a97a8bd8ef60a34454289312f560ae778e877cd77f789af.png)

## + Bài tập

1. Giả sử một word 8b cần được lưu trữ trong bộ nhớ là: 11000010 . Sử dụng mã Hamming xác định các bit kiểm tra (check bits) được lưu trữ cùng từ trên. Viết từ được lưu trữ.
2. Dữ liệu được lấy ra từ bộ nhớ: 000101001111. Xác định xem dữ liệu trên có bị lỗi hay không. Nếu có thì sửa lỗi.
3. Dữ liệu được lấy ra từ bộ nhớ: 001101001110. Xác định xem dữ liệu trên có bị lỗi hay không. Nếu có thì sửa lỗi .
4. Nếu kích thước từ là 1024b. Tính số lượng check bit nếu sử dụng:
- a) Mã Hamming SEC
- b) Mã Hamming SEC-DED

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000015_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## 5.3. Tổ chức DRAM tiên tiến

- ◼ Vấn đề quan trọng nhất hệ thống gặp phải là tắc nghẽn (tình trạng nút cổ chai) khi bộ vi xử lý có hiệu năng cao hơn bộ nhớ chính .
- ◼ Chip DRAM truyền thống bị hạn chế bởi kiến trúc bên trong và giao diện bên ngoài đến bộ VXL.
- ◼ Giải pháp:
- ◼ Sử dụng các bộ nhớ cache nhiều cấp giữa VXL và DRAM
- ◼ Tuy nhiên, do SRAM đắt hơn DRAM rất nhiều nên nếu tăng dung lượng của SRAM thì lợi nhuận sẽ giảm .
- → Cải tiến kiến trúc DRAM: SDRAM, DDR -DRAM và RDRAM.

## a. SDRAM -DRAM đồng bộ (Synchronous DRAM)

- Một trong những loại DRAM được sử dụng nhiều nhất
- Việc trao đổi dữ liệu với bộ xử lý được đồng bộ với tín hiệu đồng hồ bên ngoài và chạy ở tốc độ tối đa của VXL/ bus bộ nhớ mà không cần thiết lập trạng thái đợi (wait state)

## DRAM truyền thống

- Truyền không đồng bộ
- Sau khi nhận được yêu cầu truy xuất của VXL, DRAM mất một khoảng thời gian trễ để chuẩn bị công việc truyền dữ liệu đi .
- Trong lúc đó , VXL không thể làm gì khác vì phải thiết lập trạng thái đợi (wait state) để đợi bộ nhớ trả về dữ liệu .

## SDRAM

- Truyền đồng bộ theo đồng hồ hệ thống
- Sau khi nhận được yêu cầu truy xuất , SDRAM sẽ trả lời sau một số chu kỳ đồng hồ trễ (theo đồng hồ hệ thống) .
- VXL có thể thực hiện các tác vụ khác một cách an toàn trong khi chờ đợi SDRAM đang xử lý

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000016_3b130de8f62863c222c9558a50d5717cec5fe679a2d4547ced734c9aac466108.png)

A

M

## SDRAM -DRAM đồng bộ

- ◼ Chế độ truyền liên tục (burst mode) cho phép truyền liên tục một chuỗi các bit sau lần truy cập đầu tiên
- ◼Phù hợp với việc truy xuất theo thứ tự và trong cùng một hàng với lần truy cập đầu tiên .
- ◼SDRAM hoạt động tốt nhất khi chuyển các khối dữ liệu lớn nối tiếp nhau chẳng hạn như cho các ứng dụng như xử lý văn bản, bảng tính và đa phương tiện .

## Các chân của SDRAM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000017_fc5ba065a795e59b26417e9d2f361b1a75ca0fcbfc7a39fe2171996c5f7fe7e4.png)

## Đồ thị thời gian quá trình đọc SDRAM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000018_0897619f38ae0321e72f01aaf71e85978ea51a1800cafc32045a7ad031743b71.png)

## b. SDRAM tốc độ dữ liệu gấp đôi – Double Data Rate SDRAM (DDR SDRAM)

- ◼ SDRAM chỉ có thể gửi dữ liệu một lần trong một chu kỳ xung nhịp bus
- ◼ DDR SDRAM có thể gửi dữ liệu hai lần trong một chu kỳ xung nhịp , một ở sườn lên của xung , một ở sườn xuống
- ◼ Được phát triển bởi JEDEC Solid State Technology Association
- ◼ DDR SDRAM được sử dụng nhiều trong các máy tính để bàn và server

## Định thời đọc DDR SDRAM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000019_5e6741c7ace3671d3aa4c00584d2bdf804dfc721e685d40f518ab1a7bfd72029.png)

## c. RDRAM -Rambus DRAM

- Phát triển bởi Rambus
- Được Intel chấp nhận cho bộ vi xử lý Pentium và Itanium
- Trở thành đối thủ cạnh tranh chính của SDRAM
- Chip RDRAM được đóng gói theo chiều dọc với tất cả các chân ở một mặt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000020_c8304a264e707702a79d06a6c1728f8eede2702cd3c89d52f775b0e462fac087.png)

## Cấu trúc RDRAM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH05-Main memory_artifacts/image_000021_032b329649dbe5ba48c1ea4e8f6655fcdbb3fa299245307ad9b206cce2a047e2.png)

- ◼ Gồm: 1 bộ điều khiển (controller) và một số module RDRAM nối song song với nhau và nối ra bus. Bus gồm:
- ◼ 18 đường dữ liệu (16 đường cho dữ liệu và 2đường chẵn/lẻ),truyền 2 lần một chu kỳ đồng hồ (2 sườn của xung). Tốc độ 800Mbps/mỗi đường
- ◼ 8 đường địa chỉ và điều khiển (RC)
- ◼ Đường tín hiệu đồng hồ phục vụ cho việc truyền đồng bộ
- ◼ Các đường điện áp tham chiếu, nối đất, và nguồn điện

## Cơ chế truy xuất dữ liệu RDRAM

- ◼ Trao đổi dữ liệu với bộ vi xử lý:
- ◼ Thông qua 28 dây trên 12 cm chiều dài
- ◼ Bus có thể cho phép đánh địa chỉ được 320 chip RDRAM và thiết lập tốc độ 1.6 GBps (sau 480ns truy cập)

## ◼ Cơ chế truy xuất dữ liệu

- ◼ Không sử dụng tín hiệu điều khiển RAS, CAS, R/W và CE như DRAM thông thường.
- ◼ Việc truyền thông tin địa chỉ và điều khiển sử dụng giao thức truyền khối không đồng bộ qua bus tốc độ cao (high speed bus)
- ◼ Các thông tin trên bao gồm: địa chỉ, loại hoạt động (đọc/ghi) và số lượng byte của hoạt động

## d. Bộ nhớ đệm DRAM – Cache DRAM (CDRAM)

- ◼Được phát triển bởi Mitsubishi
- ◼Tích hợp một bộ nhớ đệm SRAM vào một chip DRAM chung
- ◼SRAM trong CDRAM có thể được sử dụng theo hai cách:
- ◼ Có thể sử dụng như một bộ nhớ cache thực sự gồm các line 64 bit. Chế độ cache của CDRAM hiệu quả với việc truy cập bộ nhớ ngẫu nhiên
- ◼ Có thể được sử dụng như một bộ đệm để hỗ trợ truy cập liên tiếp vào một khối dữ liệu

## Tổng kết

## Chương 5

- ◼ Bộ nhớ bán dẫn
- ◼ Tổ chức
- ◼ DRAM và SRAM
- ◼ Các loại ROM
- ◼ Chip logic
- ◼ Đóng gói chip
- ◼ Tổ chức module
- ◼ Interleaved memory
- ◼ Sửa lỗi
- ◼ Lỗi cứng
- ◼ Lỗi mềm

## Bộ nhớ trong

- ◼ Mã Hamming
- ◼ Tổ chức DRAM cải tiến
- ◼ Synchronous DRAM
- ◼ DDR SDRAM
- ◼ Rambus DRAM
- ◼ Cache DRAM

## + Câu hỏi

1. Các tính chất chính của bộ nhớ bán dẫn?
2. Về mặt ứng dụng , sự khác nhau giữa SRAM và DRAM là gì?
3. Sự khác nhau giữa SRAM và DRAM về mặt đặc tính như tốc độ , giá thành và dung lượng là gì?
4. Một số ứng dụng của bộ nhớ ROM là gì
5. Sự khác nhau giữa EPROM, EEPROM và bộ nhớ flash là gì?
6. Giải thích chức năng các chân trong hình 5.4b
7. Trình bày cơ chế các mã Hamming SEC và mã Hamming SEC -DED
8. Bit chẵn lẻ là gì?
9. SDRAM khác gì so với DRAM truyền thống?