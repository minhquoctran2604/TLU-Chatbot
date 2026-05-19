## + Chương 3

Tổng quan về máy tính và hệ thống kết nối trong máy tính

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000000_14990bc9c0bf08b7bddeba733777f1fd531e6a27d461ee9918d15509b4a7b43e.png)

+

## Chương 3 . Tổng quan về máy tính và hệ thống kết nối trong máy tính

Phần I. Tổng quan về máy tính

- 3.1 Các thành phần của máy tính
- 3.2 Hoạt động của máy tính
- Phần II. Hệ thống kết nối
- 3.3 Cấu trúc kết nối
- 3.4 Hệ thống bus
- 3.5 Kết nối điểm -điểm (Point -To -Point)
- 3.6 PCI Express

+

## 3.1. Các thành phần của máy tính

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000001_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

- ◼ Máy tính hiện đại ngày nay được thiết kế dựa trên kiến trúc von Neumann (Viện nghiên cứu Princeton)
- ◼ Kiến trúc Von Neumann có 3 điểm chính:
- ◼ Dữ liệu và lệnh được lưu trữ trên cùng một bộ nhớ đọc-ghi (RAM)
- ◼ Nội dung của dữ liệu được định vị theo vị trí (địa chỉ) mà không phụ thuộc vào kiểu dữ liệu .
- ◼ Các lệnh được thực thi một cách tuần tự (trừ trong một số trường hợp yêu cầu gọi đến câu lệnh khác).

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000002_47de247f481d7fd2c2b76fdbb900f2d87c62d6796ec78546fa00ef575fd81f31.png)

+

## Các thành phần của máy tính (tiếp)

- ◼ Phần mềm
- ◼ Một chuỗi các lệnh
- ◼ Khối CU làm chức năng phiên dịch từng lệnh và tạo ra tín hiệu điều khiển
- ◼ Quá trình thực hiện chương trình là truy xuất lệnh từ bộ nhớ và thực thi lệnh của CPU
- ◼ Phần cứng (3 thành phần chính)
- ◼ CPU
- ◼ CU: Khối điều khiển thực hiện chức năng biên dịch và thực thi lệnh
- ◼ ALU: Khối tính toán số học và logic
- ◼ Các Module vào/ra (I/O module)
- ◼ Module vào: bao gồm các thành phần cơ bản cho việc nhận vào dữ liệu và lệnh; chuyển đổi chúng thành dạng tín hiệu sử dụng bên trong hệ thống
- ◼ Module ra: công cụ để hiện thị kết quả
- ◼ Bộ nhớ trong (bộ nhớ chính): bộ nhớ ROM, RAM: lưu trữ lệnh , dữ liệu
- ◼ Bộ nhớ Cache: cải thiện hiệu suất của hệ thống

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000003_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000004_35f4ab74e8ce7288a0a2affd4592bb59114abba4b19a7b9f9c95249887b50c64.png)

## Các thành phần của máy tính

## + Giải thích một số thanh ghi trong hình:

- Thanh ghi MAR (Memory Address Register) chứa địa chỉ trong bộ nhớ cho lần đọc hoặc ghi tiếp theo
- Thanh ghi MBR (Memory Buffer Register) dữ liệu được ghi vào bộ nhớ hoặc nhận dữ liệu được đọc từ bộ nhớ.
- Thanh ghi I/OAR (I/O Address Register) xác định một thiết bị I/O cụ thể.
- Thanh ghi I/O BR (I/O Buffer Register) được sử dụng để trao đổi dữ liệu giữa một mô -đun I/O và CPU .
- Thanh ghi PC (Program counter Register) chứa địa chỉ lệnh tiếp theo
- Thanh ghi IR (Instruction Register) chứa lệnh đang được thực thi

## + 3.2. Hoạt động của máy tính

- ◼ Hoạt động cơ bản của máy tính là thực hiện chương trình:
- a. Thực hiện lệnh: chu kỳ lệnh
- b. Thực hiện lệnh có xử lý ngắt
- c. Thực hiện các chức năng vào ra

+

## 3.2 . Hoạt động của máy tính

- a. Thực hiện lệnh: chu kỳ lệnh
2. ◼ Chức năng chính của máy tính là thực thi chương trình (một tập
ầ g y ựg (ộ
lệnh lưu trữ trong BN): VXL phải thực hiện lần lượt các lệnh
3. ◼ Quá trình VXL thực hiện 1 lệnh gồm 2 bước: lấy lệnh (truy
ấ Qựệệ
xuất) từ bộ nhớ và thực thi lệnh .
4. ◼ Việc thực thi một chương trình là quá trình lặp đi lặp lại việc
ấ ệựộg 
truy xuất và thực thi lệnh
5. ◼ Quá trình thực hiện một lệnh được gọi là chu kỳ lệnh (instruction cycle)
6. ◼ Quá trình truy xuất lệnh từ bộ nhớ được gọi là chu kỳ truy xuất (fetch cycle)
7. ◼ Quá trình thực thi lệnh được gọi là chu kỳ thực thi (execute cycle)

## + a. Truy xuất và thực thi lệnh

## Chu kỳ truy xuất

- ◼ Vào đầu mỗi chu kỳ lệnh , bộ xử lý truy xuất một lệnh từ bộ nhớ
- ◼ Thanh ghi PC (Program Counter) giữ địa chỉ của lệnh được truy xuất tiếp theo
- ◼ Bộ xử lý tăng PC sau mỗi lần truy xuất lệnh do đó nó sẽ truy xuất được lệnh tiếp theo vào lần sau .
- ◼ Lệnh vừa được truy xuất được tải vào thanh ghi IR (Instruction Register)
- ◼ Bộ xử lý biên dịch lệnh và thi hành những hành động cần thiết

Chu kỳ lệnh cơ bản

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000005_613a65124005c3c7fe9cb0f3d9bb8a3dc7face3361a7987f6f9520c107e0b56b.png)

+

## a. Truy xuất và thực thi lệnh

## Chu kỳ thực thi

- ◼ CPU giải mã và thực hiện các hoạt động (action) tương ứng được chỉ ra trong mã lệnh (Opcode)
- ◼ Có 4 nhóm hoạt động chính của một CPU:

Bộ xử lý – bộ nhớ

Bộ xử lý – I/O

Xử lý dữ liệu

Điều khiển

- Dữ liệu truyền từ bộ xử lý đến bộ nhớ hoặc ngược lại
- Dữ liệu truyền đến/đi từ thiết bị ngoại vi bằng cách truyền thông tin giữa bộ xử lý và module I/O
- Bộ xử lý có thể thực hiện một số phép toán số học hoặc logic trên dữ liệu
- Đưa ra lệnh chỉ rõ thứ tự thực hiện các lệnh bị thay đổi

+

## Ví dụ việc thực hiện lệnh

Máy giả thiết gồm một số thông tin cấu hình như sau:

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000006_5b608e525924f3603c63247e96e5e20497176b102b5c6d066c063b2c93aa1a21.png)

+

Ví dụ Thực hiện lệnh

Dữ liệu và lệnh được biểu diễn dưới dạng mã thập lục phân

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000007_f934c19938b5ef642c717029a877cd3746cc80503b57826189909f2440095dd8.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000008_8dbb175bda0eaff712f917d637eb86c908cd02688f48e47bee7002ac47b82570.png)

## + Ví dụ

- ◼ Máy giả thiết trong ví dụ trên có hai lệnh vào/ra sau:

0011 = Đọc dữ liệu từ module I/O vào thanh ghi AC

0111 = Ghi dữ liệu từ AC ra module I/O

Biết các thiết bị ngoại vi được đánh địa chỉ 12b. Giải thích hoạt động của chương trình sau (giống Ví dụ 3.5):

1. Đọc dữ liệu từ thiết bị 5 vào thanh ghi AC.
2. Cộng AC với địa chỉ 940 của bộ nhớ.
3. Ghi AC ra thiết bị 6.

Giả sử giá trị được lấy từ thiết bị 5 là 5 và địa chỉ bộ nhớ 940 có giá trị 7.

## +

## 3.2 . Hoạt động của máy tính

## b. Xử lý ngắt

- Ngắt là một cơ chế máy tính cho phép các module khác (I/O, bộ nhớ có thể ngắt quá trình xử lý thông thường của BXL . Một số ngắt:
- Ngắt chương trình: Sinh ra bởi lỗi thi hành lệnh , ví dụ như tràn số học , lỗi chia cho 0 , cố tình thực hiện các lệnh máy không hợp lệ , hoặc tham chiếu ngoài phạm vi bộ nhớ mà người sử dụng được phép
- Ngắt định thời: Sinh ra bởi đồng hồ nằm trong bộ xử lý . Nó cho phép hệ điều hành thực hiện các chức năng cơ bản nhất định .
- Ngắt I/O: Sinh ra bởi bộ điều khiển I/O, để báo hiệu hoàn thành một thao tác , yêu cầu dịch vụ từ bộ xử lý, ý, hoặc báo hiệu các trường hợp lỗi
- Gián đoạn lỗi phần cứng: Gây ra bởi một số lỗi như lỗi nguồn hay lỗi bộ nhớ

## + Quá trình xử lý ngắt

- ◼ Ngắt được đưa vào chủ yếu như là 1 cách để để cải thiện hiệu quả xử lý:
- ◼ Trong trường hợp VXL thực hiện chương trình có trao đổi dữ liệu với I/O
- ◼ Do tốc độ của I/O chậm hơn rất nhiều so với VXL → VXL phải đợi I/O
- Giải pháp: trong lúc chờ đợi I/O, VXL thực hiện tiếp các phần công việc khác → đến khi I/O xong, nó sẽ gửi tín hiệu đến VXL (tín hiệu y/c ngắt) → VXL dừng công việc đang làm (ngắt), phục vụ I/O → VXL tiếp tục cv đang thực hiện

## Điều khiển dòng chương trình

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000009_df15b11f35e77d1307997d42a2d0e27569bc03d957d6432c9d04327de7c75ffe.png)

+

## Điều khiển ngắt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000010_e38fc8b41ab301aa308131e80e7d5726ab64278d7ced0dfb6750ef8df64a0c82.png)

+

## Chu kỳ lệnh có ngắt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000011_e855feb2a8eb9029d8c1612337c0bb721309d0e9d7282b23f43287eebd694fd5.png)

## Sơ đồ trạng thái chu kỳ lệnh Có ngắt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000012_7f3e30dcbce2c8751c9d22aaec1ba29bdc3aa3107932bbded1c6c471c81ae5df.png)

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000013_d3b6829bcce16800fbbaf11ec6a0122ac80250b042569cf9172cf56ae96462e5.png)

Minh họa thời gian thực hiện chương trình

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000014_516b91f1d7aab6b0171b5430162a5fc54fc214b458da37ddb51314c832d1c200.png)

Định thời chương trình: Đợi I/O dài

## + Xử lý nhiều ngắt

Trong trường hợp có nhiều ngắt, hai phương pháp để xử lý:

- ◼ Tắt ngắt
- ◼ Bộ vi xử lý sẽ bỏ qua các tín hiệu ngắt khác trong khi xử lý một yêu cầu ngắt
- ◼ Các yêu cầu ngắt đó sẽ phải chờ đến khi bộ xử lý xử lý xong ngắt hiện tại
- ◼ Sau khi thực thi xong một ngắt, bộ xử lý sẽ kiểm tra xem có ngắt nào đang chờ không. Các ngắt sẽ được xử lý lần lượt
- ◼ Xác định ưu tiên
- ◼ Cho phép các ngắt có mức độ ưu tiên cao hơn được ngắt các ngắt có mức độ ưu tiên thấp hơn

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000015_663eed7e96bf7161dc2679abc87090f365eadecc47578452985aa099400b4dbd.png)

## Điều khiển ngắt

## Nhiều ngắt

## + Trình tự thời gian của xử lý nhiều ngắt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000016_1fb0118f753e623ac5fc8ff9558342d735722171133d28873aa4aa1fd21ba09e.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000017_a884e530ce7c407aec9a8039718159135e1cfeabbc498549e4ed9a9ed3c9108f.png)

## + c. Chức năng I/O

- ◼ Module I/O có thể chuyển dữ liệu trực tiếp với bộ xử lý
- ◼ Bộ xử lý có thể đọc dữ liệu từ hoặc ghi dữ liệu lên module I/O
- ◼ Bộ xử lý xác định thiết bị nào được điều khiển bởi module I/O nào
- ◼ Khi làm việc với module I/O, một chuỗi lệnh tương tự như P13 có thể được thực hiện chỉ khác ở các lệnh I/O chứ không phải là các lệnh tham chiếu đến bộ nhớ
- ◼ Cơ chế truy cập bộ nhớ trực tiếp (Direct Memory Access -DMA): cho phép I/O trao đổi dữ liệu trực tiếp với bộ nhớ
- ◼ Bộ xử lý cấp cho module I/O quyền đọc/ghi vào bộ nhớ do đó việc truyền tin giữa module I/O và bộ nhớ có thể diễn ra trực tiếp mà không cần thông qua bộ xử lý
- ◼ Giải phóng bộ XL khỏi nhiệm vụ điều khiển việc chuyển dữ liệu

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000018_5d21669f89ccc7a248d7c88066e7fb7d3daa6c8c6e6677018d389108fbf7cf53.png)

+

## 3.3 Cấu trúc kết nối

- ◼ Các thành phần chính (bộ vi xử lý, bộ nhớ, I /O) của máy tính cần được kết nối để trao đổi dữ liệu với nhau
- ◼ Một tập các đường kết nối tạo thành cấu trúc kết nối (Interconnection Structures)
- ◼ Cấu trúc này được thiết kế phụ thuộc vào cơ chế trao đổi dữ liệu giữa các thành phần máy tính.

## Cấu trúc kết nối hỗ trợ các hình thức truyền sau:

Bộ nhớ tới bộ xử lý

Bộ xử lý đọc 1 lệnh hoặc 1 đơn vị dữ liệu từ bộ nhớ

Bộ xử lý tới bộ nhớ

Bộ xử lý ghi 1 đơn vị dữ liệu vào bộ nhớ

I/O tới bộ xử lý

Bộ xử lý đọc dữ liệu từ thiết bị I/O thông qua I/O module

Bộ xử lý tới I/O

Bộ xử lý gửi dữ liệu tới thiết bị I/O

I/O tới/từ bộ nhớ

Module I/O được phép trao đổi dữ liệu trực tiếp với bộ nhớ mà không cần đi qua bộ xử lý nhờ DMA

## + Các dạng dữ liệu đến/đi từ các thành phần máy tính

## ◼ Bộ nhớ:

- ◼ T/h điều khiển đọc/ghi
- ◼ T/h địa chỉ
- ◼ Dữ liệu đi vào/ra

## ◼ Module I/O

- ◼ T/h điều khiển đọc/ghi
- ◼ T/h địa chỉ
- ◼ Dữ liệu bên trong (ngoài): dữ liệu đến từ CPU(thiết bị ngoại vi)
- ◼ Tín hiệu ngắt: phát ra từ module I/O gửi đến CPU

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000019_6da112da102a32658f0ab9812db605944fad78066e4d700faeb8d35c60710376.png)

Read

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000020_8952ae36eb5d1ef846c9ed1f4cfb02d1462580c7d9d0c48bcae6c532ad7d1b04.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000021_47f151a54d8c0156433887ec14b7a2d914b80d7da536135d10f7509ce5ba21bf.png)

Instructions

Data

Interrupt

Signals

Figure 3.15 Computer Modules

Address

Control

Signals

Data

CPU

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000022_a32e406547780dbd04faa5d4534b3e74fc2c0d672088e50288d3bf09bd374cc6.png)

## ◼ CPU

- ◼ Lệnh: truy xuất từ bộ nhớ
- ◼ Dữ liệu đến hoặc đi từ bộ nhớ hoặc I/O
- ◼ T/h ngắt: do I/O module gửi tới
- ◼ T/h địa chỉ: định vị một ô nhớ hoặc một tb ngoại vi
- ◼ T/h điều khiển

Read

Write

Address

Data

Memory

N Words

0

N – 1

## Các dạng dữ liệu đến/đi từ các thành 
I/O Module
W rite
Read phần máy tính
Addre
Intern
Dt hh 
I/O Module M Ports h
Address nh
Internal Data External Data Internal i t
W từ các thành 
Data rite Read

External

Data

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000023_16f2083e929b81bf6144f8177ab27d213fee6435a56e81b1b7afb5ad809c7ed4.png)

Figure 3.15 Computer Modules

Data

Interrupt

Signals

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000024_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## + Một số cấu trúc kết nối

- ◼ Cấu trúc kết nối phổ biến nhất: cấu trúc bus và cấu trúc đa bus (phần 3.4)
- ◼ Cấu trúc kết nối điểm – điểm: QPI (phần 3.5) và PCIe (phần 3.6)

+

## 3.4 Kết nối Bus (hệ thống Bus)

- ◼ Bus: đường thông tin kết nối giữa 2 hay nhiều thiết bị .
- ◼ Là đường truyền chia sẻ: Tín hiệu truyền bởi 1 thiết bị bất kì có thể 
ấếếố g yy
được nhận bởi tất cả các thiết bị khác kết nối với bus đó
- ◼ Nếu 2 thiết bị cùng truyền 1 lúc, tín hiệu của chúng sẽ bị chồng nhau và bị méo
- ◼ Một bus thường gồm nhiều đường , m ỗi đường có khả năng truyền tín hiệu dưới dạng bit 1 và bit 0
- ◼ Hệ thống máy tính có một số loại bus khác nhau cung cấp đường kết 
ốầấố g y g p g 
nối giữa các thành phần thuộc các cấp khác nhau của hệ thống máy tính
- ◼ Bus hệ thống
- ◼ Kết nối các thành phần chính của máy tính (bộ xử lý, bộ nhớ, module I/O)
- ◼ Gồm 50 đến 100 đường:
- ◼ Data bus (bus dữ liệu): gồm các đường truyền dữ liệu
- ◼ Address bus (bus địa chỉ): gồm các đường địa chỉ
- ◼ Control bus (bus điều khiển): các đường truyền tín hiệu điều khiển

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000025_88eb642e75094b84f383741ad042cdb7d2930f1b49ea098cb30dde8a8d9cb1c5.png)

+

## a. Cấu trúc bus

## Bus dữ liệu (data bus)

- ◼ Gồm các đường dữ liệu (data lines) để truyền dữ liệu giữa các module hệ thống: data bus
- ◼ Data bus bao gồm 32, 64, 128 đường hay nhiều hơn
- ◼ Số lượng đường được xem là độ rộng của bus dữ liệu
- ◼ Số lượng đường nối quyết định bao nhiêu bit có thể truyền đi cùng một lúc
- ◼ Độ rộng bus dữ liệu là yếu tố chính quyết định hiệu suất toàn hệ thống

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000026_0a66d7f2e31a3a789ed058015b56778707cbbc69f8e1cf871e02fff909066790.png)

## + Bus địa chỉ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000027_4b7a88f7f61934b34bb6d9ccefd1cacc63e1f791c60f65470f2608a7c56c293e.png)

- ◼ Được sử dụng để xác định địa chỉ nguồn/đích của dữ liệu trên bus dữ liệu .
- ◼ Nếu bộ xử lý muốn đọc 1 word từ bộ nhớ , nó sẽ đặt địa chỉ của word đó lên đường bus địa chỉ .
- ◼ Độ rộng bus xác định dung lượng nhớ tối đa của hệ thống
- ◼ Cũng được sử dụng để xác định cổng vào/ra (I/O port) trên module I/O.
- ◼ Các bit cao được sử dụng để lựa chọn module cụ thể trên bus còn bit thấp dùng để chọn vị trí bộ nhớ hoặc cổng vào/ra trong module.
- ◼ Được sử dụng để điều khiển việc truy nhập và sử dụng dữ liệu và bus địa chỉ .
- ◼ Bởi vì dữ liệu và bus địa chỉ được chia sẻ cho tất cả các thành phần nên cần phải có một công cụ kiểm soát việc sử dụng chúng .
- ◼ Các tín hiệu điều khiển truyền cả thông tin lệnh và định thời giữa các mô đun hệ thống .
- ◼ Tín hiệu định thời xác định tính hợp lệ của dữ liệu và thông tin địa chỉ .
- ◼ Tín hiệu lệnh chỉ ra thao tác (operation) cần được thực hiện .
- ◼ VD: t/h điều khiển read/write

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000028_0cec6dc1e30fa8d6c93bb29d5a7c3a3ca31bff354a8d151352236d185dbba892.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000029_a32e406547780dbd04faa5d4534b3e74fc2c0d672088e50288d3bf09bd374cc6.png)

## Ví dụ: đọc dữ liệu từ Bộ nhớ vào VXL

## ◼ CPU gửi:

- ◼ Địa chỉ qua bus địa chỉ
- ◼ Tín hiệu yêu cầu đọc (READ) qua bus điều khiển

## ◼ RAM:

- ◼ Nhận địa chỉ từ bus địa chỉ, giải mã địa chỉ
- ◼ Xác định yêu cầu: đọc dữ liệu
- ◼ Lấy dữ liệu từ ngăn nhớ đó đặt lên bus dữ liệu
- ◼ CPU: đọc dữ liệu từ bus dữ liệu, ghi vào thanh ghi. Loại bỏ các tín hiệu điều khiển và địa chỉ.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000030_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000031_0fc90c3f5033d8c64ab461a28f92a2e2e939ee0d34c0a4f456a7a9c89332e200.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000032_617630df514b23b46739ce05a6051aa2a8d220ecfe8eec5a60f46de818eebddc.png)

Hoạt động của bus như sau:

- Figure 3.16 Bus Interconnection Scheme
ai việc: (1) yêu cầu việc sử dụn · Nếu một module muốn gửi dữ liệu đến một module khác nó Figure 3.16 Bus Interconnection Scheme
phải làm hai việc: (1) yêu cầu việc sử dụng bus và (2) truyền dữ liệu qua bus.
- Nếu một module yêu cầu dữ liệu từ một module khác , nó phải (1) yêu cầu sử dụng bus và (2) chuyển yêu cầu tới module khác qua bus điều khiển và địa chỉ . Sau đó phải chờ cho module thứ hai gửi dữ liệu .

+

## b. Cấu trúc bus phân cấp

- ◼ Nếu một số lượng lớn các thiết bị được kết nối với bus, hiệu suất sẽ giảm. Hai nguyên nhân chính:
1. Nhiều thiết bị gắn vào bus, chiều dài bus càng dài và do đó trễ truyền càng lớn.
2. Hiện tượng nút cổ chai: lượng dl cần truyền quá lớn so với khả năng của bus.
- ◼Khắc phục nhược điểm trên: Cấu trúc bus phân cấp
- ◼ Bus địa phương kết nối bộ vi xử lý với bus hệ thống qua cache: cách li bộ XL và bộ nhớ → thực hiện DMA
- ◼ Bus hệ thống liên kết tất cả các module bộ nhớ chính
- ◼ Bus mở rộng kết nối các thiết bị ngoại vi: cho phép nối đc với nhiều tb ngoại vi hơn nhưng vẫn đảm bảo ko làm ảnh hưởng đến bus hệ thống khi dl truyền từ tb ngoại vi quá lớn

## + Kiến trúc truyền thống (Có cache)

(a) Traditional Bus Architecture

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000033_eb7acb62cc30b2ee85ee459ab1f90b7bd387446b75e8178832ea97c30f1aaef7.png)

Main

Memory

System Bus

Graphic

Cache

Local Bus

/Bridge

FireWire

High-Speed Bus

Expansion bus interface

Modem

Expansion Bus

(b) High-Performance Architecture

Figure 3.17 Example Bus Configurations

Video

Serial

Processor

SCSI

FAX

LAN

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000034_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

Network

Processor

Main

Memory

SCSI

Local Bus

Local I/O

controller

System Bus

Expansion bus interface

Modem

Expansion Bus

## (a) Traditional Bus Architecture
ệ () Tditil BAhitt
Kiến trúc hiệu suất cao

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000035_46f8718e090ab94a7c92efdf9ba3a1b0ea53892d17df41629b8ee74bc616b990.png)

(b) High-Performance Architecture

Figure 3.17 Example Bus Configurations

Cache

Serial

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000036_30981b64c33ba4e2ad1380dcfb74b831c81f7862042d0ba3cda6df121e225833.png)

## c. Các yếu tố trong thiết kế Bus

```
Loại
h Dành 
ài riêng
tr Ghép kênh Phương pháp trọng tài Tập trung Phân tán Định thời Đồng bộ Bất đồng bộ 1. Loại bus Chuyên dụng Lo
Ghép kênh Dành 
riêng
Ghé
2. Phương pháp trọng tài Tập trung Phân tán 3. Định thời Đồng bộ Bất đồng bộ
```

```
Độ rộng bus Địa chỉ
i Dữ liệu
oại Loại truyền dữ liệu
ạ Đọc Ghi Đọc thay 
G đổi ghi Đọc sau khi ghi Khối 4. Độ rộng bus Địa chỉ Dữ liệu Địa chỉ
Dữ liệu
ềdữ liệ
5. Loại truyền dữ liệu Đọc ay 
Ghi Đọc thay đổi ghi Đọc sau khi ghi Khối
```

+

## 1. Các loại bus: chuyên dụng và ghép kênh.

- ◼ Bus chuyên dụng sử dụng cho một chức năng cụ thể: vd: bus dữ liệu, bus địa chỉ, bus điều khiển
- ◼ Ưu điểm: nhanh hơn , ít có xung đột bus
- ◼ Nhược điểm: tăng kích thước và chi phí
- ◼ Bus ghép kênh: các thông tin (dữ liệu , địa chỉ) được truyền trên cùng một đường .
- ◼ Sử ử dụng đường điều khiển AV (Address Valid control line).
- ◼ Khi bắt đầu, đ/c được đưa vào bus và đường AV được kích hoạt .
- ◼ Các module nhận địa chỉ , kiểm tra xem có phải đ/c của nó không .
- ◼ Thông tin đ/c được loại bỏ và một kênh truyền được thiết lập để truyền dữ liệu đọc hoặc ghi
- ◼ Ưu điểm: ít đường hơn , tiết kiệm không gian và chi phí
- ◼ Nhược điểm: mạch phức tạp hơn

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000037_c68e5589c8f901553aa10cbd2b2e5e6d2e52f8628016f08cf1438233da3ababd.png)

+

## 2. Phương pháp phân xử (trọng tài)

- ◼ Đôi khi, tại một thời điểm có nhiều module cần chiếm bus → cần quyết định xem module nào có quyền sử dụng bus: phân xử (trọng tài) bus
- →Phương pháp phân xử bus: tập trung và phân tán
- ❑ Phân xử tập trung: bộ điều khiển (bộ phân xử) phân bổ thời gian trên bus. Bộ điều khiển này có thể là một thiết bị riêng hoặc một phần của bộ XL
- ❑ Phân xử phân tán: mỗi module chứa một access control logic và chúng làm việc cùng nhau để chia sẻ đường truyền

+

## 3. Định thời

- ◼ Định thời là cách các sự kiện được phối hợp truyền trên bus
- ◼ Hai loại: đồng bộ và không đồng bộ
- ◼ Định thời đồng bộ:
- ◼ Mỗi hoạt động truyền trên bus được thực hiện theo các xung đồng hồ
- ◼ Bus chứa một đường xung đồng hồ (clock line) truyền liên tiếp một chuỗi các bit 0, 1
R

Read

- ◼ Khoảng thời gian T được gọi là chu kỳ đồng hồ

cycle

- ◼ Tất cả các thiết bị trong máy tính đều đọc 
W được và đồng bộ các hoạt động truyền theo xung này Write cycle

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000038_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000039_dcda9a2e7d59c6fe649c88c364e2ef3dda36a602a0522ba493ce74da7da24b23.png)

Figure 3.18 Timing of Synchronous Bus Operations

+

## 3. Định thời (tiếp)

- ◼ Định thời không đồng bộ
- ◼ Không sử dụng tín hiệu đồng hồ.
- ◼ Sau khi dữ liệu được đưa vào bus, bộ nhớ gửi một tín hiệu ACK để báo cho VXL biết việc đọc hoặc ghi dữ liệu
- ◼ Truyền đồng bộ: thực hiện đơn giản tuy nhiên ít linh hoạt hơn truyền không đồng bộ
- ◼ Việc truyền theo xung đồng hồ đôi khi làm giảm hiệu suất hệ thống
- ◼ Truyền không đồng bộ: hiệu quả hơn trong trường hợp nhiều thiết bị có tốc độ xử lý khác nhau 
Ack chia sẻ chung bus Data lines au 
Acknowledge

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000040_fc26699d5fce0c8bef953c71742fac7562f5d46ae03c91eabfd13177be8dbd33.png)

Status lines

Address lines

Data lines

Write

Acknowledge

Status signals

Stable address

Valid data

(b) System bus write cycle

Figure 3.19 Timing of Asynchronous Bus Operations

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000041_420763b73e159c5784f652288101aa2483ef4db74a648d4e65773dde4cf070a0.png)

Status lines

Address lines

Read

Data lines

Acknowledge

Status signals

Stable address

Valid data

(a) System bus read cycle

(b) System bus write cycle

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000042_6ff1e91ce36cba358306b28ee3f3df20cf5714349e51e34cbaecd2e16ad4a15b.png)

Figure 3.19 Timing of Asynchronous Bus Operations

## + Bài tập

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000043_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

1. Xét một hệ thống máy tính có độ rộng bus địa chỉ là 16b, bus dữ liệu là 16b.
- a. Không gian địa chỉ bộ nhớ là bao nhiêu?
- b. Dung lượng tối đa của bộ nhớ là bao nhiêu nếu kích 
2. Xét VXL 32b, với bus dữ liệu có độ rộng 16b, 

+

## 3.5. Kết nối điểm -điểm

- ◼ Nhược điểm của hệ thống kết nối bus:
- ◼ Tốc độ của bus đồng bộ không cao do khó khăn trong việc tăng tần số tín hiệu đồng hồ.
- ◼ Khi tốc độ dữ liệu cao, việc thực hiện các chức năng đồng bộ và phân xử bus một cách kịp thời trở nên khó khăn hơn
- ◼ Với chip đa nhân, nếu sử dụng bus để kết nối, trao đổi dữ liệu giữa các nhân sẽ không đáp ứng được tốc độ VXL → giảm hiệu suất .
- ◼ Giải pháp: kết nối điểm -điểm: có độ trễ thấp , tốc độ dữ liệu cao , và khả năng mở rộng tốt hơn .
- ◼ 2 loại kết nối điểm – – điểm: QPI và PCIe

## +
Đường dẫn nhanh (Quick Path Interconnect)

- ◼ Được giới thiệu vào năm 2008
- ◼ Nhiều kết nối trực tiếp
- ◼ Các kết nối từng cặp trực tiếp tới các thành phần khác giúp loại bỏ việc phân xử thường thấy trong các hệ thống truyền dẫn chia sẻ .
- ◼ Kiến trúc giao thức phân lớp
- ◼ Những kết nối của bộ xử lý sử dụng kiến trúc giao thức phân lớp chứ không chỉ đơn giản sử dụng tín hiệu điều khiển thường thấy trong sắp xếp các bus chia sẻ .
- ◼ Truyền dữ liệu gói
- ◼ Dữ liệu được gửi thành 1 chuỗi các gói chứa tiêu đề điều khiển (header) và mã kiểm soát lỗi .

QPI

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000044_b935f20d3c493425038ea1193ec77732dee6d20364e0a15e738a7bc3988cdcc2.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000045_15c8d3bad66ae91f6c5a192ca368457528f19140bbdcd42df057c1c0b9a1df42.png)

Cấu hình chip đa nhân sử dụng QPI

Figure 3 . 20 Multicore Configuration Using QPI

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000046_a32e406547780dbd04faa5d4534b3e74fc2c0d672088e50288d3bf09bd374cc6.png)

## Các lớp QPI

- ◼ QPI được định nghĩa là một kiến trúc bốn lớp, bao gồm các lớp sau:
- ◼ Vật lý: Bao gồm dây dẫn mang tín hiệu, cũng như mạch và logic để hỗ trợ các tính năng truyền và nhận các bit 1 và 0. Đơn vị chuyển giao ở lớp này 20 bit, được gọi là Phit (physical unit).
- ◼ Liên kết: Chịu trách nhiệm truyền tin cậy và điều khiển luồng. Đơn vị dữ liệu của lớp Liên kết là một Flit 80 -bit (flow control unit)
- ◼ Định tuyến: Cung cấp một framework để chuyển các gói dữ liệu
- ◼ Giao thức: Bộ quy tắc để trao đổi các gói tin dữ liệu giữa các thiết bị. Một gói bao gồm một số không đổi các Flit.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000047_5ec909266433631abc1af2c092514ddc3289c98ed1a94712433ffaac98612cfe.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000048_30981b64c33ba4e2ad1380dcfb74b831c81f7862042d0ba3cda6df121e225833.png)

## a. Lớp vật lý

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000049_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000050_074ceacb2ea6a49985055ae94ceee766f2f1f6f2afc0431c87c4ed568963b750.png)

Figure 3

.

22 Physical Interface of the Intel QPI Interconnect

Figure 3.23 QPI Multilane Distribution

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000051_3e6a7157c04b6f60b5a11f2ca76f36fda13f771b46dce7c3fd2d1f4db552f4c5.png)

## + b. Lớp liên kết (tiếp)

- ◼ Thực hiện hai chức năng chính: điều khiển luồng và điều khiển lỗi .
- ◼ Vận hành trên cấp flit (flow control unit – – đơn vị điều khiển luồng)
- ◼ Mỗi flit gồm 1 bản tin 72 -bit và một mã kiểm soát lỗi 8 -bit được gọi là cyclic redundancy check (CRC)
- ◼ Chức năng điều khiển luồng
- ◼ Cần thiết để đảm bảo rằng 1 thực thể QPI gửi không áp đảo 1 thực thể QPI nhận bằng cách gửi dữ liệu nhanh hơn khả năng xử lý dữ liệu và xoá bộ đệm để nhiều dữ liệu mới đến của phía nhận
- ◼ Chức năng điều khiển lỗi
- ◼ Phát hiện và khắc phục lỗi bit, do đó tránh cho các lớp cao hơn gặp lỗi bit

+

## c. Lớp giao thức và lớp định tuyến

## Lớp Định tuyến

- ◼ Được sử dụng để xác định đường đi mà một gói sẽ đi qua các kết nối hệ thống có sẵn
- ◼ Xác định bởi phần sụn và mô tả các đường dẫn mà một gói tin có thể đi theo

## Lớp Giao thức

- ◼ Gói (packet) là đơn vị truyền
- ◼ Một chức năng quan trọng được thực hiện ở lớp này là giao thức liên kết bộ nhớ cache -đảm bảo rằng các giá trị bộ nhớ chính được giữ trong nhiều cache là phù hợp
- ◼ Một gói dữ liệu thông thường là một khối dữ liệu được gửi đến hoặc từ một bộ nhớ cache

+

## 3.6. Kết nối thiết bị ngoại vi Peripheral Component Interconnect (PCI)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000052_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

- ◼ Một bus băng thông cao , độc lập với bộ xử lý, có thể hoạt động như một bus ngoại vi
- ◼ PCI cung cấp hiệu suất cao hơn cho các thiết bị I/O tốc độ cao (vd: card mạng, card màn hình, card ổ cứng)
- ◼ Nhóm quan tâm đặc biệt PCI (Special Interest Group - SIG)
- ◼ Được tạo ra để phát triển và duy trì tính tương thích của các đặc tính PCI
- ◼ PCI Express (PCIe)
- ◼ Cơ chế kết nối điểm -điểm nhằm thay thế cơ chế dựa trên bus như PCI
- ◼ Yêu cầu chính là dung lượng cao để hỗ trợ nhu cầu của thiết bị I / O tốc độ dữ liệu cao hơn, như Gigabit Ethernet
- ◼ Một yêu cầu khác là phải hỗ trợ các ứng dụng với luồng dữ liệu thời gian thực

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000053_26a22ef3e06be53046782b02d44e3ef84a886c064b235630ae56c98f0ca1308d.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000054_b87faf987588516d59783ce04db939dfaef582ab6579d7904e8f934c1da07592.png)

Figure 3

24 Typical Configuration Using PCIe

.

## + Lớp giao thức PCIe

- ◼ Kiến trúc giao thức PCIe bao gồm các lớp sau:
- a. Vật lý (Physical): Bao gồm các dây dẫn thực tế mang tín hiệu , cũng như mạch và logic để hỗ trợ các tính năng cần thiết trong việc truyền và nhận các bit 1 và 0 .
- b. Liên kết dữ liệu (Data link layer – – DLL): Chịu trách nhiệm truyền tin cậy và điều khiển luồng. Các gói dữ liệu được tạo ra và được xử lý bởi DLL được gọi là gói DLLP.
- c. Giao dịch (Transaction Layer): Tạo ra và xử lý các gói dữ liệu được sử dụng để thực hiện các cơ chế truyền dữ liệu được tải/lưu trữ và cũng quản lý điều khiển luồng của các gói tin giữa hai thiết bị. Các gói dữ liệu của lớp này được gọi là gói TLP. P.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000055_9509540b11e940a88f8c921a60ff1237258e9d2ed8ee709cbf75bae8b0d3e62e.png)

+

## a. Lớp vật lý

Phân phối đa tuyến PCIe

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000056_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

Figure 3.26 PCIe Multilane Distribution

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000057_616b099ce0a3bb9d74a681363851cd66a015b870b31a93bc9ba94f240b354bd8.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000058_0fe048da7dec00ec7cb5818c191b6c9482bf4498beed9ae72175aa9078609cf9.png)

Figure 3.27 PCIe Transmit and Receive Block Diagrams

Sơ đồ khối Truyền và nhận PCIe

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000059_fb7ba1109f78ecb10501c4f29f329a04e6cbe994f9cc32ec6062f6c4aa7fa468.png)

- ◼ Nhận các yêu cầu đọc và ghi từ phần mềm phía trên TL và tạo ra các gói tin yêu cầu để truyền tới đích qua lớp liên kết (link layer)
- ◼ Hầu hết các giao dịch sử dụng kỹ thuật giao dịch phân chia
- ◼ Một thiết bị PCIe nguồn gửi 1 gói tin yêu cầu đi , sau đó đợi 1 phản hồi gọi là gói hoàn thành (completion packet)
- ◼ Bản tin TL và một số giao dịch ghi là giao dịch gửi (nghĩa là không cần có phản hồi)
- ◼ Định dạng gói TL hỗ trợ địa chỉ bộ nhớ 32 -bit và địa chỉ bộ nhớ 64-bit mở rộng

## + TL hỗ trợ bốn không gian địa chỉ:

- ◼ Bộ nhớ
- ◼ Không gian bộ nhớ bao gồm bộ nhớ chính của hệ thống và thiết bị I/O PCIe
- ◼ Các khoảng địa chỉ bộ nhớ nhất định được ánh xạ vào các thiết bị I/O
- ◼ Cấu hình
- ◼ Không gian địa chỉ này cho phép TL đọc/ghi các thanh ghi cấu hình kết hợp với các thiết bị I/O

## ◼ I/O

- ◼ Không gian địa chỉ này được sử dụng cho thiết bị PCI kế thừa, với dải địa chỉ dành riêng dùng để xác định các thiết bị I/O kế thừa
- ◼ Message
- ◼ Không gian địa chỉ này dành cho các tín hiệu điều khiển liên quan đến gián đoạn , x xử lý lỗi, và quản lý năng lượng

## Các kiểu giao dịch TLP PCIe

| Address Space               | TLP Type                    | Purpose                                                                           |
|-----------------------------|-----------------------------|-----------------------------------------------------------------------------------|
| Memory                      | Memory Read Request         | Transfer data to or from a location in the                                        |
| Memory                      | Memory Read Lock Request    | system memory map.                                                                |
| Memory                      | Memory Write Request        |                                                                                   |
| I/O                         | I/O Read Request            | Transfer data to or from a location in the  system memory map for legacy devices. |
| I/O                         | I/O Write Request           | Transfer data to or from a location in the  system memory map for legacy devices. |
| Configuration               | Config Type 0 Read Request  | Transfer data to or from a location in the  configuration space of a PCIe device. |
| Configuration               | Config Type 0 Write Request | Transfer data to or from a location in the  configuration space of a PCIe device. |
| Configuration               | Config Type 1 Read Request  | Transfer data to or from a location in the  configuration space of a PCIe device. |
| Configuration               | Config Type 1 Write Request | Transfer data to or from a location in the  configuration space of a PCIe device. |
| Message                     | Message Request             | band messaging and event                                                          |
| Message                     | Message Request with Data   |                                                                                   |
| Memory, I/O,  Configuration | Completion                  | Returned for certain requests.                                                    |
| Memory, I/O,  Configuration | Completion with Data        | Returned for certain requests.                                                    |
| Memory, I/O,  Configuration | Completion Locked           | Returned for certain requests.                                                    |
| Memory, I/O,  Configuration | Completion Locked with Data | Returned for certain requests.                                                    |

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000060_f7dbda9908d967d5db706d3a2ae05519a5789496ea126f4a4046ca0d103528ec.png)

(a) Transaction Layer Packet

Figure 3

.

28 PCIe Protocol Data Unit Format

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000061_6f27b8e5e6a8a00da1dda7e5bd12d872467a2cad1ac73f2174ca4ad2c5837d2d.png)

Định dạng Đơn vị dữ liệu Giao thức PCIe

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000062_52ec9173f17e5d7c33271dc201fd6b3f433690c62f33ba346db5b0c7cf991160.png)

32 bits

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000063_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

Figure 3.29 TLP Memory Request Format

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000064_cc25d20f6dc2d4d9b144a2a5e54655fe06a575f5a9807e685cf40f84f7915110.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000065_30981b64c33ba4e2ad1380dcfb74b831c81f7862042d0ba3cda6df121e225833.png)

## Tổng kết

## Chương 3

- ◼ Thành phần máy tính
- ◼ Chức năng máy tính
- ◼ Lệnh truy xuất và thi hành
- ◼ Gián đoạn
- ◼ Chức năng I / O
- ◼ Cấu trúc kết nối
- ◼ Kết nối bus
- ◼ Cấu trúc bus
- ◼ Nhiều phân cấp bus
- ◼ Các yếu tố thiết kế bus

## Chức năng máy tính và kết nối

- ◼ Kết nối điểm -điểm
- ◼ Lớp vật lý QPI
- ◼ Lớp liên kết QPI
- ◼ Lớp định tuyến QPI
- ◼ Lớp giao thức QPI
- ◼ PCI Express
- ◼ Kiến trúc vật lý và logic PCI
- ◼ Lớp vật lý PCIe
- ◼ Lớp giao dịch PCIe
- ◼ Lớp liên kết dữ liệu PCIe
1. Nêu các nhóm chức năng chính của một hệ thống máy tính
2. Các bước thực hiện một lệnh được diễn ra như thế nào?
3. Hai phương pháp xử lý đa ngắt là gì?
4. Các loại tín hiệu truyền nào cần được hệ thống bus hỗ trợ?
5. Các ưu điểm của kiến trúc đa bus so với kiến trúc đơn bus là gì?
6. Các đặc điểm của kết nối điểm – – điểm là gì?
7. Liệt kê các lớp của QPI.
8. Liệt kê các lớp của PCIe.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH03-Chuc nang va ket noi_artifacts/image_000066_049cf578d9488942545ccf98bf9cfce65fa3422f568d51abc816c57a846dbfb4.png)