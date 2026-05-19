## + Chương 7

Thiết bị ngoại vi

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000000_b8eb0923bd17d7163e0d6bec6d87c16bc580eedd820ebf12c6c183e0a64dc0b6.png)

+

## Chương 7. Thiết bị ngoại vi

- 7.1 Các thiết bị ngoại vi
- 7.2 Module vào/ra
- 7.3 Các kỹ thuật I/O
- a. I/O chương trình
- b. Điều khiển ngắt vào/ra
- c. Truy xuất bộ nhớ trực tiếp
- 7.4 Các bộ xử lý và kênh vào/ra

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000001_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## 7.1 Thiết bị ngoại vi

- ◼ Một trong ba thành phần cơ bản của hệ thống máy tính: CPU, bộ nhớ và thiết bị ngoại vi (thông qua module I/O)
- ◼ Chức năng: trao đổi dữ liệu giữa máy tính với bên ngoài
- ◼ Kết nối với máy tính qua module vào/ra (module I/O)
- ◼ Module I/O: Truyền các thông tin điều khiển, dữ liệu và địa chỉ giữa CPU và thiết bị ngoại vi
- ◼ Có ba loại
- ◼ Con người đọc được: màn hình, máy in,...
- ◼ Máy đọc được: ổ cứng, cảm biến, băng từ,...
- ◼ Truyền thông: modem, card mạng,...

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000002_c1e00cd1d0e120de6dc1a5377c8858ba7c786cee37fde9b92521322e14e7eaa8.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000003_0910bf85e6d345e6a0015005f61b4e3e400c7f6f6c4d93bcecc2131bd352ab82.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000004_573db454693c72acb9e93dd7b8ee089d53fa48fbc4d6094877e749e5a8c04736.png)

## + a. Sơ đồ khối thiết bị ngoại vi

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000005_3a76de5cf9bba25a56cb680c1b0390073314d1eaf62f1be244169e9fe0ef16eb.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000006_b6d497fe1fe7d5a86ef6ba23f1d43ee6e87cfe0c90824a5992e68b320b501261.png)

+

## a . Sơ đồ khối thiết bị ngoài (tiếp)

## ◼ Giao diện với module I/O:

- ◼ Tín hiệu điều khiển -Control signal: xác định chức năng mà thiết bị sẽ thực hiện:
- ◼ READ: yêu cầu thiết bị gửi dữ liệu vào module I/O (INPUT)
- ◼ WRITE: yều cầu thiết bị nhận dữ liệu từ module I/O (OUTPUT)
- ◼ Các tín hiệu điều khiển đặc biệt
- ◼ Dữ liệu – Data: một tập các bit được gửi đến hoặc đi từ module I/O.
- ◼ Tín hiệu trạng thái - Status signal: cho biết trạng thái của thiết bị . Ví dụ:
- ◼ READY: thiết bị sẵn sàng cho việc truyền dữ liệu .
- ◼ NOT -READY: không sẵn sàng truyền dữ liệu
- ◼ Logic điều khiển 
ề – Control logic: nhận các tín hiệu điều khiển từ
ểủế module I/O , gậ
điều khiển hoạt động của thiết bị .
- ◼ Bộ chuyển đổi -Transducer: chuyển đổi dữ liệu (đang ở dạng t/h 
ểả ộ yyệ(g ạg 
điện) sang các dạng khác (vd: điểm ảnh trên màn hình,...) và ngược lại .
- ◼ Bộ đệm (buffer) để lưu trữ tạm dữ liệu đang được chuyển giao giữa module I/O và môi trường bên ngoài; kích thước bộ đệm thường từ 8 
ế đến 16 bit.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000007_be882b8b19858f10f09af0d5d0cec0c5ea9c08bb6b84f25a4f61f141d3d1dbfa.png)

+

## b. Bàn phím/Màn hình

## Bảng chữ cái tham khảo quốc tế (IRA)

- ◼ Ký tự
- ◼ Gắn với mỗi ký tự là một mã
- ◼ Mỗi ký tự được biểu diễn bởi một mã nhị phân 7-bit : biểu diễn 128 ký tự
- ◼ Hai loại ký tự:
- ◼ In được
- ◼ Các ký tự chữ cái, số và ký tự đặc biệt có thể được in trên giấy hoặc hiển thị trên màn hình
- ◼ Điều khiển
- ◼ Điều khiển việc in/hiển thị các ký tự
- ◼ VD: carriage return
- ◼ Các ký tự điều khiển khác liên quan đến các thủ tục truyền tin

Công cụ tương tác máy tính/ người dùng phổ biến nhất Người dùng cung cấp đầu vào thông qua bàn phím Màn hình hiển thị dữ liệu được cung cấp bởi máy tính Đơn vị chuyển đổi cơ bản là ký tự

## Mã bàn phím

- ◼ Khi người dùng bấm một phím, một tín hiệu điện tử được tạo ra bởi một bộ chuyển đổi trong bàn phím và dịch sang mẫu bit của mã IRA tương ứng
- ◼ Mẫu bit này được truyền đến mô -đun I/O trong máy tính
- ◼ Trên đầu ra, các ký tự mã IRA được truyền đến một thiết bị ngoại vi từ module I/O
- ◼ Bộ chuyển đổi giải mã và gửi các tín hiệu điện tử yêu cầu đến thiết bị đầu ra để hiển thị ký tự được chỉ định hoặc thực hiện chức năng điều khiển yêu

cầu

- ▪

## 7.2 Module I/O

## a. Chức năng

Các chức năng chính của một module I/O gồm:

- Điều khiển và định thời: phối hợp luồng lưu lượng truy cập giữa thành phần thiết bị bên trong (main memory, bus) và thiết bị ngoại vi
- Trao đổi thông tin với VXL: gồm giải mã lệnh, dữ liệu, báo cáo trạng thái (trạng thái của thiết bị I/O có sẵn sàng hay không), nhận dạng địa chỉ (địa chỉ các cổng mà TBNV được nối vào)
- Trao đổi thông tin với TBNV: gồm các lệnh, thông tin trạng
- thái và dữ liệu
- Đệm dữ liệu: thực hiện các hoạt động đệm cần thiết để cân bằng
- tốc độ TBNV và bộ nhớ
- Phát hiện và báo cáo lỗi

## b. Cấu trúc Module I/O

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000008_bcdede0320693a2ff7974cd11fd9f586cee2b3ef9688271f0d8fd86d4137d4a8.png)

## + b. Cấu trúc Module I/O (tiếp)

Các module I/O thay đổi khác nhau theo sự phức tạp và số lượng các thiết bị ngoài mà nó điều khiển . Cấu trúc chung nhất:

- -Dữ liệu được truyền đến và đi từ module được đệm qua một hoặc nhiều thanh ghi dữ liệu (data register) .
- -Thanh ghi trạng thái/ điều khiển (status/control register): lưu trữ thông tin trạng thái của thiết bị hoặc thông tin điều khiển của bộ VXL
- -Khối logic điều khiển -I/O logic: tương tác với VXL qua một tập các đường điều khiển (control line). VXL sử dụng các đường điều khiển để ra lệnh cho module I/O. Module I/O cũng có thể sử dụng một số đường điều khiển để gửi các tín hiệu phân xử bus hoặc tín hiệu trạng thái .
- -Module cũng có khả năng nhận diện và sinh ra các địa chỉ với mỗi thiết bị được nối đến nó (địa chỉ cổng). Mỗi module I/O có một (nếu chỉ nối với một TBNV) hoặc một tập địa chỉ (nếu module nối với nhiều TBNV)
- -Cổng nối ghép vào/ra (External Device Interface Logic): giao tiếp với thiết bị ngoại vi

+

## c. Địa chỉ cổng vào/ra

- ◼ Cũng giống như bộ nhớ, cácTBNV được gắn vào module I/O qua các cổng. Để CPU giao tiếp được với các TBNV, các cổng này phải được gán một giá trị địa chỉ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000009_053543511c30b6a33b397b5b3f6c635ace1ce483cedaa2fcb91c7eb288937f5d.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000010_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## Không gian địa chỉ I/O

- ◼ Có hai phương thức thực hiện không gian địa chỉ cho các TBNV:
- ◼ I/O ánh xạ bộ nhớ (memory-mapped I/O):
- ◼ Bộ nhớ và TBNV chia sẻ chung không gian địa chỉ. VXL coi các thanh ghi dữ liệu và trạng thái như các ô nhớ và sử dụng cùng các lệnh để truy cập cả bộ nhớ và thiết bị ngoại vi
- ◼ Chỉ sử dụng một đường đọc và ghi, do đó bus phải sắp xếp giữa việc đọc/ghi bộ nhớ và vào/ra TBNV
- ◼ I/O riêng biệt (isolated I/O):
- ◼ Sử dụng một đường command line để xác định: địa chỉ BN hay địa chỉ TBNV
- ◼ Toàn bộ dải địa chỉ dùng cho cả hai. VD: 10 đường địa chỉ cho phép đánh địa chỉ 1024 ô nhớ và 1024 TBNV
- ◼ Tập các chỉ lệnh đến BN và TBNV khác nhau

## + Ví dụ

- ◼ Bộ xử lý 680x0 của Motorola : quản lý một không gian địa chỉ chung cho cả bộ nhớ và I/O.
- ◼ Bộ xử lý Intel Pentium:
- ◼ Không gian địa chỉ bộ nhớ = 2 32 địa chỉ
- ◼ Không gian địa chỉ vào-ra = 2 16 địa chỉ
- ◼ Tín hiệu điều khiển: M/IO
- ◼ Lệnh vào-ra chuyên dụng: IN, OUT

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000011_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## 7.3. Các kỹ thuật vào/ra

Hoạt động của module I/O theo 3 kỹ thuật sau:

## ◼ I/O chương trình

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000012_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

- ◼ CPU thực thi một chương trình trực tiếp điều khiển các hoạt động vào/ra
- ◼ Khi bộ xử lý ra lệnh , nó phải đợi cho đến khi hoạt động vào/ra hoàn tất
- ◼ Bộ xử lý chạy nhanh hơn module I/O sẽ gây lãng phí thời gian xử lý

## ◼ I/O điều khiển ngắt

- ◼ Bộ vi xử lý ra lệnh I/O sau đó tiếp tục thi hành các lệnh tiếp theo trong chương trình .
- ◼ Khi module I/O hoàn thành công việc , nó sẽ gửi tín hiệu yêu cầu ngắt đến VXL.

## ◼ Truy cập bộ nhớ trực tiếp (DMA)

- ◼ Module I/O và bộ nhớ chính trực tiếp trao đổi dữ liệu mà không có sự tham gia của bộ vi xử lý

| +   |
|-----|

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000013_f10eace682af738f0cb3e9366a7d4dd0b21e8730d77f749cbd03fbd3a0d8c1e1.png)

## + a. Kỹ thuật I/O chương trình

- ◼ Khi cần thực hiện một tác vụ vào/ra:
- ◼ VXL thực thi một chương trình và gửi lệnh đến module I/O tương ứng
- ◼ Module I/O nhận yêu cầu, thiết lập các bit trạng thái trên thanh ghi trạng thái
- ◼ CPU định kỳ kiểm tra trạng thái của module I/O
- ◼ Chưa sẵn sàng thì tiếp tục định kỳ kiểm tra
- ◼ Đã sẵn sàng, thiết lập việc truyền dl đến module I/O

## + a. Các lệnh I/O – I/O command (từ VXL đến module I/O)

Để thực thi một lệnh vào/ra, VXL thực hiện công việc sau:

- -Đặt địa chỉ lên bus địa chỉ: định ra module I/O và TBNV cụ thể
- -Đưa các mệnh lệnh vào/ra: Thiết lập các đường điều khiển trong bus điều khiển. Có 4 loại mệnh lệnh vào/ra:
- 1) Control: kích hoạt một thiết bị ngoại vi và chỉ định nó phải làm gì
- 2) Test: kiểm tra các điều kiện trạng thái liên quan đến một module I/O và các thiết bị ngoại vi: TBNV bật hay tắt, hoạt động I/O đang thực hiện đã xong chưa, có lỗi gì
- 3) Read: yêu cầu đọc dữ liệu từ TBNV vào VXL
- -Module I/O lấy dữ liệu từ thiết bị ngoại vi và đặt nó vào bộ đệm bên trong → đặt dữ liệu vào bus cho CPU
- 4) Write: yêu cầu ghi dữ liệu ra TBNV
- -Module I/O lấy dữ liệu từ bus dữ liệu rồi chuyển dữ liệu đó đến thiết bị ngoại vi

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000014_cd67a624d174f53708daa1ccd12ce22c186be31d8405628ca445c261cf9494e9.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000015_dc16362573da432045e1e718f8bf0277bff863b016d993f56fbb829dad51067e.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000016_dd64dafc9d555cf68dae32daa437a0b12975405d7f987865d46d7b583b41cf89.png)

## + b. I/O điều khiển ngắt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000017_e985b46e01d7364a631c7a0e954279fc594c3bcf6c1bd8f43b99bcaf0ccd4fd4.png)

- ◼ Vấn đề với I/O chương trình là bộ xử lý phải đợi một thời gian dài để module I/O sẵn sàng cho việc nhận hoặc truyền dữ liệu
- ◼ Giải pháp thay thế là bộ vi xử lý ra lệnh I/O cho module , sau đó thực hiện các việc khác .
- ◼ Khi module I/O sẵn sàng trao đổi dữ liệu với VXL, nó sẽ gửi tín hiệu ngắt đến VXL
- ◼ Bộ xử lý thực hiện việc truyền dữ liệu và tiếp tục quá trình xử lý trước đó

+

## Cơ chế làm việc

## Từ phía VXL

- ◼ VXL đưa ra lệnh READ.
- ◼ Sau đó thực hiện các công việc khác (vd: trong trường hợp có
ề (g g p 
nhiều CT đang chạy tại một thời
ể điểm)
- ◼ Sau mỗi chu kỳ lệnh, VXL sẽ
ể ỳ 
kiểm tra xem có tín hiệu yêu
ầắ cầu ngắt được gửi tới
- ◼ Nếu có, VXL lưu trữ nội dung 
ắ g 
đang thực hiện và xử lý ngắt
- ◼ VXL nhận dữ liệu từ bus lưu trữ vào bộ nhớ và tiếp tục chương trình

## Từ phía module I/O

- ◼ Nhận lệnh READ từ VXL
- ◼ Đọc dữ liệu vào từ TBNV tương ứng
- ◼ Khi dữ liệu được đưa vào thanh ghi dữ liệu, module gửi tín hiệu yêu cầu ngắt đến VXL và chờ đợi tín hiệu yêu cầu dữ liệu từ VXL.
- ◼ Khi có tín hiệu đó, module đặt dữ liệu vào bus và sẵn sàng để thực hiện các hoạt động I/O khác

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000018_48e64f166fc66589ab201e3e9cf4bcb7d75b34a766e05f616e0f1ccb72a47227.png)

+

## Cơ chế xử lý ngắt

## VXL nhận được yêu cầu ngắt

- ◼ Thiết bị phát tín hiệu ngắt cho bộ xử lý
- ◼ Bộ xử lý hoàn thành lệnh hiện tại → Kiểm tra thấy có y/c ngắt → gửi ACK báo đã nhận ngắt.
- ◼ Chuyển sang chế độ phục vụ ngắt: lưu trữ nội dung các thanh ghi vào vùng ngăn xếp của RAM (hình trang sau)
- ◼ Tải trình điều khiển ngắt: đặt địa chỉ đầu tiên của trình này vào thanh ghi PC → thực hiện hoạt động vào/ra

## VXL thực hiện xong yêu cầu ngắt

- ◼ Sau khi thực hiện xong hoạt động vào/ra, CPU khôi phục lại công việc đang thực hiện
- ◼ Nạp lại nội dung từ vùng ngăn xếp vào các thanh ghi: PSW, PC
- ◼ Khôi phục lại luồng điều khiển

+

- a. Quá trình chuyển sang chế độ phục vụ ngắt b. Khôi phục lại luồng điều khiển sau khi thực hiên yêu cầu ngắt xong

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000019_9654433961d659c05299b896d9e337572b297ce0403a0593f1261b2757020c87.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000020_b7dbd0307096e0de5421853a91c70bb4d90047f94a8800c22209cffc139ebfe6.png)

## Hai vấn đề phát sinh

Hai vấn đề thiết kế phát sinh khi thực hiện I/O điều khiển ngắt:

1. Nhận diện thiết bị: Bởi vì sẽ có nhiều module I/O, khi có một yêu cầu ngắt gửi tới, bộ vi xử lý sẽ xác định thiết bị đưa ra yêu cầu ngắt bằng cách nào?
2. Xác định ưu tiên . N Nếu xảy ra nhiều ngắt cùng một thời điểm , V VXL lựa chọn ngắt nào để xử lý?

## + 1. Nhận diện thiết bị

## Bốn loại kỹ thuật chung được sử dụng phổ biến:

- ◼ Nhiều đường ngắt
- ◼ Sử dụng nhiều đường ngắt giữa VXL và các module I/O → dễ dàng xác định thiết bị
- ◼ Không thực tế do kỹ thuật này làm tăng số đường bus và các chân của VXL. Thêm vào đó, vẫn phải có nhiều module I/O nối với một đường → vẫn cần một trong ba kỹ thuật còn lại

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000021_94af88ee75c7ba5895e75d3f2e365b5fa879a027d324154991bd78584dfa20ac.png)

+

## ◼ Thăm dò phần mềm

- ◼ Khi bộ xử lý phát hiện ra một ngắt , nó thực thi một phần mềm thăm dò:
- ◼ Thực hiện lệnh thăm dò (vd: lệnh TEST I/O) tới từng module I/O (thông qua địa chỉ) → Module I/O sẽ trả lời nếu nó đưa ra y/c ngắt.
- ◼ Hoặc thực hiện lệnh đọc thanh ghi trạng thái của từng module để phát hiện module y/c ngắt
- ◼ Nhược điểm: Tốn thời gian

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000022_f7706564b92cf6e8e62aa74dd112da0bce1e49f982755fba9a083aabcacb0a03.png)

+

## ◼ Chuỗi Daisy (thăm dò phần cứng, vector)

- ◼ Tất cả các module I/O sử dụng chung một đường yêu cầu ngắt (INTR)
- ◼ Đường nhận biết ngắt (INTA) được nối chuỗi qua các module
- ◼ Khi VXL nhận được y/c ngắt, nó sẽ gửi lại một tín hiệu ACK qua đường INTA
- ◼ T/h này truyền qua các module I/O đến khi gặp module y/c ngắt. Module này trả lời bằng cách đặt một word lên bus dữ liệu: được gọi là vector ngắt (chứa thông tin địa chỉ của module I/O hoặc mã nhận dạng thiết bị khác).
- ◼ VXL sử dụng vector này trỏ tới trình phục vụ ngắt tương ứng của thiết bị → ngắt vector

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000023_cc8203093c78ed2a05f30d6d17bbaea692c28b9eddd9f211aae4ec2a598111cf.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000024_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000025_30981b64c33ba4e2ad1380dcfb74b831c81f7862042d0ba3cda6df121e225833.png)

## ◼ Phân xử bus (cũng sử dụng vector ngắt)

- ◼ Sử dụng cơ chế cho phép một module I/O chiếm quyền sử dụng bus rồi mới gửi yêu cầu ngắt
- ◼ Khi bộ xử lý phát hiện ra ngắt , nó trả lời trên đường ACK.
- ◼ Module I/O đặt vector ngắt của nó lên các đường dữ liệu
- ◼ VXL sử dụng vector này trỏ tới trình phục vụ ngắt tương ứng của TB

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000026_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## 2. Xác định ưu tiên

- ◼ Các phương pháp nhận diện thiết bị đồng thời cho phép xác định độ ưu tiên của các TB khi có nhiều yêu cầu ngắt cũng một thời điểm:
- ◼ Nhiều đường ngắt: VXL sẽ chọn đường có độ ưu tiên cao hơn để xử lý trước
- ◼ Thăm dò phần mềm: thứ tự thăm dò các thiết bị được sắp xếp theo độ ưu tiên
- ◼ Chuỗi daisy: tương tự thăm dò phần mềm
- ◼ Phân xử bus: cơ chế phân xử bus đã có phân xử theo độ ưu tiên

## + Ví dụ

- ◼ Kiến trúc dòng máy sử dụng chip Intel 80386: thực hiện việc điều khiển ngắt thông qua vi điều khiển Intel 82C59A

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000027_dbca561b45d0756d879ea80a44034b43ab5faf76a0ef139d61dd43c8c7518dc1.png)

## + c . Truy cập bộ nhớ trực tiếp -DMA

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000028_a64f5f2c2c619db0172f65fb9de7c8ad0d016fab6559fe511a31bffc7b441e9f.png)

- ◼ Nhược điểm của I/O chương trình và I/O điều khiển ngắt: VXL thay gia vào hầu hết chu trình truyền/nhận dữ liệu giữa TBNV và BN
- 1) Tốc độ truyền I/O bị giới hạn bởi tốc độ kiểm tra và phục vụ thiết bị của bộ xử lý
- 2) Bộ xử lý gắn với việc quản lý truyền I/O; Một số chỉ lệnh phải được thực hiện cho mỗi lần truyền I/O

Khi khối lượng dữ liệu lớn được di chuyển , một kỹ thuật hiệu quả hơn là truy cập bộ nhớ trực tiếp (DMA)

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000029_7d8cd4dcdcf6aae1b864f69891962feb42154bff91f4124a15b3056efd540720.png)

+

+

+

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000030_29c48ef992c6137cafc21ab8917317165dbe9b031548d7eb16a7f88b0308441c.png)

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000031_275e6b47cabd531cfeb544ae1b5cc4d949ca34e709eff46bbfa2b29334b82805.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000032_a63aed3fe1de1162268bbe7e6ce3969f60b43c16be4d9f5934b29dac43019d1a.png)

+

+

## Chức năng DMA

- ◼ DMA bao gồm một module bổ sung trên hệ thống bus .
- ◼ Module DMA có khả năng thực hiện việc điều khiển việc truyền/nhận dữ liệu thay cho VXL
- ◼ Module DMA chỉ sử dụng bus khi VXL không cần đến nó hoặc buộc VXL phải tạm ngừng hoạt động để chiếm bus . Kỹ thuật thứ hai là phổ biến hơn .

## Quá trình đọc/ghi dữ liệu sử dụng DMA

- Khi cần đọc/ghi dữ liệu, VXL gửi đến module DMA các thông tin sau:
- Yêu cầu đọc/ghi: đường điều khiển
- Địa chỉ TBNV: đường dữ liệu
- Vị trí bộ nhớ bắt đầu để lưu trữ dữ liệu: đường dữ liệu và lưu trữ trên thanh ghi địa chỉ
- Số lượng word được đọc/ghi: đường dữ liệu, lưu trữ trên thanh ghi data count
- VXL thực hiện công việc khác, DMA thực hiện việc truyền giữa BN và TBNV
- Sau khi hoàn thành, DMA gửi tín hiệu ngắt cho VXL
- →VXL chỉ tham gia vào thời điểm bắt đầu và kết thúc việc truyền tin

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000033_1fb0118f753e623ac5fc8ff9558342d735722171133d28873aa4aa1fd21ba09e.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000034_c93b65818f117ed581c6cf0f23f7efe1df11336505eee0714dc4a7eef024ec83.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000035_ddf36c269c607734070203da69148b91406603c44d2e6184b5bd5ab9c0c318e2.png)

+

## Chu kỳ DMA

- Khi có yêu cầu bus từ DMA, VXL phải tạm thời "treo" để nhường bus cho DMA.
- Do không phải là ngắt nên VXL chỉ tạm dừng trong một chu kỳ bus
- DMA ▪ DMA cũng làm cho VXL bị chậm hơn, tuy nhiên, khi truyền một lượng lớn dữ liệu thì DMA hiệu quả hơn nhiều so với 2 kỹ thuật còn lại

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000036_5b15722431a7d8f5347b9ee9b7caa24f86dacffdb4a3e28afb80d0e24afcd3bf.png)

## + c. Cấu hình DMA

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000037_7f016d7ec56e62fabd12139775a797530471957544273238c4b30724d1b6c382.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000038_9c943a77f446ca823d22e0c7dc98174f9b579f88a51e07db6a77d0b5b630527c.png)

- ◼ Cấu hình DMA: tất cả các module chia sẻ chung bus hệ thống
- ◼ Module DMA điều khiển việc truyền dữ liệu giữa I/O và memory trong hai chu kỳ bus
- ◼ Cấu hình có chi phí thấp nhưng rõ ràng không hiệu quả
- ◼ DMA giao tiếp với I/O không qua bus hệ thống: giảm được chu kỳ bus
- ◼ DMA chỉ chiếm bus hệ thống khi cần trao đổi dữ liệu với bộ nhớ chính

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000039_dc4bb4598c1b37f7ef79583a48a13f079499fbf06eabddc491126ddeed9b3f4c.png)

## + d. Bộ điều khiển DMA Intel 8237

## 8237 DMA Cách sử dụng Bus Hệ thống

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000040_03cefd0f0455f7a5eb009ec0659fe8d4e133e8659d74e235f5dc594dfd373cfa.png)

## + d. Bộ điều khiển DMA Fly -By

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000041_be6c5ca5c0af67694590904b270e641975c9fa6f0c26b6b28d8a22977ce3a79a.png)

Bảng 7.2

Thanh ghi

Intel

8237A

## + 6. VXL và các kênh I/O

## a. Sự phát triển của chức năng I/O qua các thời kỳ

1. CPU trực tiếp điều khiển một thiết bị ngoại vi .
2. Thêm vào Một bộ điều khiển hoặc module I/O. CPU sử dụng I/O lập trình , không có ngắt .
3. Cấu hình tương tự như bước 2, nhưng có sử dụng ngắt . CPU không phải tốn nhiều thời gian chờ đợi một hoạt động I/O được thực hiện, do đó tăng hiệu quả .
4. Module I/O được truy cập trực tiếp tới bộ nhớ qua DMA. Nó có thể di chuyển một khối dữ liệu đến/từ bộ nhớ mà không liên quan đến CPU , ngoại trừ khi bắt đầu và kết thúc quá trình truyền.
5. Module I/O được tăng cường để trở thành một bộ xử lý theo quyền riêng của nó, với một tập hợp chỉ lệnh dành riêng cho I/O
6. Module I/O có bộ nhớ cục bộ riêng và trên thực tế là một máy tính theo quyền riêng của nó. Với kiến trúc này , có thể kiểm soát một tập hợp lớn các thiết bị I/O với sự tham gia tối thiểu của CPU.

I/O channel

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000042_71a4d28f99343ea5cf72cc3f258266524c4f389976c251019182359fbd2ee310.png)

+

## Review Questions

1. Liệt kê ba loại thiết bị ngoại vi (thiết bị ngoài).
2. IRA -International Reference Alphabet là gì?
3. Các chức năng chính của module I/O là gì?
4. Trình bày ba kỹ thuật để thực hiện I/O .
5. Sự khác nhau giữa I/O ánh xạ bộ nhớ và I/O riêng biệt là gì?
6. Khi một ngắt được gửi đến VXL , bộ xử lý xác định thiết bị đã yêu cầu ngắt như thế nào?
7. Trong khi một module DMA chiếm quyền điều khiển bus VXL làm gì?

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000043_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000044_1f43cde12b69071c10fe22ac36becd1cdf2d4b4c76f5ab6543f03eb2768ea611.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000045_85044c011b456138269c61720037d4d255147f5b8848c0779a2224485a4f76e3.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000046_686112a9635e869d1e14ea1a27fa92a9a320137bd73e4004649bca4b98bb8be0.png)

Ví dụ: EIA-232

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000047_a60e605006bd26abd9313478786a1a8e6bee4800a7c3a5f2e3723a8857de5910.png)

+

## Thunderbolt

- ◼ Công nghệ kết nối ngoại vi nhanh nhất và mới nhất sẵn sàng cho mục đích sử dụng chung
- ◼ Phát triển bởi Intel với sự hợp tác của Apple
- ◼ Kết hợp dữ liệu, video, âm thanh và năng lượng vào một kết nối tốc độ cao cho nhiều thiết bị ngoại vi như ổ cứng, mảng RAID, các hộp thu video và giao diện mạng
- ◼ Cung cấp thông lượng 10 Gbps theo từng hướng, công suất tối đa 10 W cho các thiết bị ngoại vi đã kết nối
- ◼ 1 giao diện ngoại vi tương thích Thunderbolt thì phức tạp hơn nhiều so với 1 thiết bị USB đơn giản
- ◼ Các sản phẩm thế hệ đầu tiên chủ yếu nhằm vào thị trường tiêu dùng chuyên nghiệp , VD như biên tập viên nghe nhìn cần di chuyển nhanh khối lượng lớn dữ liệu giữa các thiết bị lưu trữ và máy tính xách tay
- ◼ Thunderbolt là một tính năng chuẩn của máy tính xách tay Apple MacBook Pro và máy tính bàn iMac

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000048_63237f194c7734c9b6965cb70715f68549ba794d548251b54f32545521b39873.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000049_40fd9fb43a03e66df364a687fb7040fe0a2ce94b08d963bdd0f1a1cfc24ea3ff.png)

http://www.intel.com/content/www/us/en/io/thunderbolt/thunderbolt-technology-developer.html

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000050_882643380d4ee05194ee8f8c565812cbfdca3b445fce86d07080f282741dfb8c.png)

Cấu hình máy tính với Thunderbolt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000051_a03da0d5d639adfa8c4e2e9fc906fa5692a1063d085ff2bac6ad3683923dd0e7.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000052_5f511930aa0f2ee17df4df6982e48cc36bc22af51c8c7c4eb2f0e052b59ab538.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000053_c861900b807afa2f507786f0142e137ab9c442f2d7b1d856a6496f563ccb9405.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000054_2840dc5048404b43d11e0784a40d2751e66f0a1ccb72530c73c127d99830c107.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000055_641cb49dd22ddcb23fe52238d74e158a46c40e1f55293a3cec5d378c56e9ed34.png)

## + InfiniBand

- ◼ Công nghệ I/O gần đây nhằm vào thị trường máy chủ cao cấp
- ◼ Phiên bản đầu tiên được phát hành vào đầu năm 2001
- ◼ Chuẩn này mô tả 1 kiến trúc và thông số kỹ thuật cho luồng dữ liệu giữa các bộ xử lý và thiết bị I/O thông minh
- ◼ Trở thành giao diện phổ biến cho các khu vực lưu trữ mạng và các cấu hình lưu trữ lớn khác
- ◼ Cho phép các máy chủ, bộ lưu trữ từ xa và các thiết bị mạng khác được gắn trên một lưới tập trung các switch và link
- ◼ Kiến trúc switch có thể kết nối tới 64.000 máy chủ, hệ thống lưu trữ và thiết bị mạng

## Lưới Switch InfiniBand

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000056_42d1a26150b7ff77934c6adcca8d384abf4646c3526a1f79cb07fd0489114589.png)

## +

## Hoạt động của InfiniBand

- ◼ Mỗi liên kết vật lý giữa một switch và một giao diện đính kèm có thể hỗ trợ tối đa 16 kênh logic, được gọi là làn đường ảo (virtual lane)
- ◼ Một đường dành cho việc quản lý lưới và các đường khác để vận chuyển dữ liệu
- ◼ Một làn ảo tạm thời được dành riêng cho việc truyền dữ liệu từ nút này sang nút khác trên lưới InfiniBand
- ◼ Switch InfiniBand ánh xạ lưu lượng từ làn đường đến sang làn đường đi để định tuyến dữ liệu giữa các điểm đến mong muốn
- ◼ Sử dụng kiến trúc giao thức phân lớp, gồm 4 lớp:
- ◼ Physical - Vật lý
- ◼ Link -Liên kết
- ◼ Network -Mạng
- ◼ Transport - Vận chuyển

+

## Bảng 7.3 Liên kết InfiniBand và Tốc độ truyền dữ liệu

## InfiniBand Communication Protocol Stack

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000057_644db5f663b8d09d69455a4fa8756aebb7ab27861dcb3645c2587bccd9ab5e4f.png)

+

## zEnterprise 196

- ◼ Máy tính lớn mới nhất của IBM. Ra mắt năm 2010
- ◼ Hệ thống sử dụng chip z196
- ◼ Chip đa lõi 5.2 GHz với 4 lõi
- ◼ Có tối đa 24 chip xử lý (96 lõi)
- ◼ Có một hệ thống con I/O dành riêng để quản lý tất cả các hoạt động I/O
- ◼ Tối đa 4 trong số 96 bộ xử lý lõi có thể được dành riêng cho I/O sử dụng, tạo ra 4 hệ thống con kênh (CSS -channel subsystems)
- ◼ Mỗi CSS được tạo thành từ các yếu tố sau:
- ◼ Bộ xử lý hỗ trợ hệ thống (SAP)
- ◼ Vùng hệ thống phần cứng (HSA)
- ◼ Phân vùng logic
- ◼ Kênh con -Subchannels
- ◼ Đường dẫn kênh
- ◼ Kênh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000058_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000059_82529f25d31581d4bbe4a2f026c0ec24201c4761e9954a56d31135f88c6c07e8.png)

## Tổ chức Hệ thống I/O

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000060_52db9a7b45458a28da0f02f87fd6f71e941bd5946e274724e479d6b448774864.png)

Khung I/O IBM z196 – mặt trước

## Cấu trúc hệ thống I/O IBM z196

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH07-Input_Output_artifacts/image_000061_c05923540b79458cd0f38c8e87742762e13577f6f9dff2fe211f3a89ec93fad0.png)

+

## Tổng kết

## Chương 7

- ◼ Thiết bị ngoại vi
- ◼ Bàn phím / màn hình
- ◼ Ổ đĩa
- ◼ Mô đun I / O
- ◼ Chức năng mô-đun
- ◼ Cấu trúc module I / O
- ◼ I/O lập trình
- ◼ Tổng quan về I/O lập trình
- ◼ Lệnh I/O
- ◼ Chỉ thị I/O
- ◼ I/O định hướng gián đoạn
- ◼ Xử lý gián đoạn
- ◼ Các vấn đề thiết kế
- ◼ Bộ điều khiển gián đoạn Intel 82C59A
- ◼ Giao diện ngoại vi lập trình được Intel 82C55A

## Input/Output

- ◼ Truy cập bộ nhớ trực tiếp
- ◼ Nhược điểm của I/O lập trình và I/O định hướng gián đoạn
- ◼ Chức năng DMA
- ◼ Bộ điều khiển Intel 8237A DMA
- ◼ Các kênh I / O và bộ xử lý
- ◼ Sự phát triển của chức năng I / O
- ◼ Đặc điểm của các kênh I / O
- ◼ Giao diện ngoài
- ◼ Các loại giao diện
- ◼ Cấu hình điểm -điểm và đa điểm
- ◼ Thunderbolt
- ◼ InfiniBand
- ◼ Cấu trúc I/O IBM zEnterprise 196