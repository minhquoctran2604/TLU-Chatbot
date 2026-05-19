## + Chương 2

Lịch sử phát triển của máy tính

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000000_b8eb0923bd17d7163e0d6bec6d87c16bc580eedd820ebf12c6c183e0a64dc0b6.png)

+ Chương 2. Lịch sử phát triển của máy tính
1. Sơ lược lịch sử phát triển máy tính
2. Các đặc tính thiết kế máy tính
3. Chip đa nhân
4. Kiến trúc x86
5. Hệ thống nhúng
6. Đánh giá hiệu suất máy

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000001_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## + 1. Sơ lược lịch sử phát triển máy tính

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000002_4b5ead736b6afb07e97c0bace27ffa140ecf60daf18c2031d4b4047bbdb2f82e.png)

+

## 1. Sơ lược lịch sử phát triển máy tính a. Máy tính thế hệ đầu tiên: Ống chân không

## 1. ENIAC:

- ◼ Electronic Numerical Integrator And Computer
- ◼ Được thiết kế và xây dựng tại trường Đại Học Pennsylvania
- ◼ Bắt đầu xây dựng từ năm 1943 – – hoàn thành vào năm 1946
- ◼ Bởi giáo sư John Mauchly và người học trò John Eckert
- ◼ Là máy tính điện tử số đầu tiên trên thế giới
- ◼ Phòng thí nghiệm đạn đạo quân đội (BRL) cần thiết bị có thể cung cấp bảng quỹ đạo chính xác cho một loại vũ khí mới trong khoảng thời gian cho phép .
- ◼ Đã không kịp hoàn thành cho nỗ lực phục vụ chiến tranh . Được tháo rời vào năm 1955
- ◼ Nhiệm vụ đầu tiên của nó là thực hiện một loạt các tính toán giúp xác định tính khả thi của bomb hydrogen.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000003_7bbc640f00c0b3ada5da89df93dcbc47d9541cea6e0575411172b6a7f1c8ac59.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000004_14090959b4d924cc2f996f62124106f2700738d4384d02f0ce8ed5f0236c5768.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000005_3dbb5a5402f971ecd7ad55370698b4f9e41e30c04b6e1f57ee57db2f45218780.png)

+

## 2. Các máy Von Neumann

## EDVAC (Electronic Discrete Variable Computer)

- ◼ Dưới dạng bản thảo: ý tưởng thiết kế được đưa ra vào năm 1945
- ◼ Khái niệm chương trình lưu trữ (stored-program)
- ◼ Được đưa ra bởi các nhà thiết kế ENIAC, đặc biệt là nhà toán học John von Neumann
- ◼ Chương trình được biểu diễn dưới dạng thích hợp để lưu vào bộ nhớ cùng với dữ liệu

## IAS computer

- ◼ Viện nghiên cứu Princeton (Princeton Institute for Advanced Studies)
- ◼ Là nền tảng cho các máy tính hiện đại ngày nay.
- ◼ Hoàn thiện vào năm 1952

## IAS computer Cấu trúc

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000006_f39e93b68bd5145a9c84249e874f27cab37563e5e3d11d4463bf739c229ab175.png)

## + IAS computer (tiếp)

## Format bộ nhớ của máy IAS

- ◼ Bộ nhớ của máy IAS gồm 1000 ô nhớ (gọi là các từ (word)). Mỗi ô chứa 40 bit nhị phân .
- ◼ Cả dữ liệu và lệnh đều được lưu trữ trong đây
- ◼ Chữ số được biểu diễn dưới dạng
ố ỗ số nhị phân . g
Mỗi lệnh là một mã nhị phân

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000007_30ab14c7e417ef8668bcd482788b131ba0d6775f84504504a8d67321fa818e17.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000008_6e09f25d3bc20343018105b9416f67df12dddf71c0411468eb2105d02c9f71f4.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000009_e11e9ef661389f29f094c8ebacbb194067045c01f71a72623fa8f13e62187542.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000010_77fd9cea245913f64acbb52675dd8b9c95ba74b679c878b87a3c1ee5e4026781.png)

+

## IAS computer (tiếp)

## Các loại thanh ghi

Thanh ghi bộ nhớ đệm (Memory buffer register MBR)

Thanh ghi địa chỉ bộ nhớ (Memory address register MAR)

Thanh ghi tập lệnh (Instruction register - IR)

Thanh ghi bộ nhớ đệm chứa tập lệnh (Instruction buffer register -IBR)

Bộ đếm chương trình (Program counter - PC)

Bộ cộng tích luỹ (AC) và bộ Nhân chia (MQ)

- Chứa từ (word) sắp lưu vào trong bộ nhớ hoặc sắp được gửi ra cac cổng I/O. · Hoặc được sử dụng để nhận một từ từ trong bộ nhớ hoặc từ các cổng I/O.
- Chỉ định địa chỉ bộ nhớ của từ (word) chuẩn bị được đọc hoặc ghi vào MBR.
- Chứa mã tác vụ 8 bit của lệnh đang được thực thi .
- Được sử dụng để tạm thời lưu trữ lệnh nằm bên tay phải của 1 từ (word) trong bộ nhớ .
- Lưu giữ địa chỉ bộ nhớ của cặp lệnh tiếp theo.
- Được sử dụng để tạm thời giữ các toán hạng và kết quả của các phép tính trong ALU.

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000011_98b77b74f0e45e1e65d037f3ae264bd997b73f419a138ca25b1cf5e16b68a054.png)

## Hoạt động IAS

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000012_6e09f25d3bc20343018105b9416f67df12dddf71c0411468eb2105d02c9f71f4.png)

## Bảng 2.1

## Tập lệnh trong IAS

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000013_6e09f25d3bc20343018105b9416f67df12dddf71c0411468eb2105d02c9f71f4.png)

## Bảng 2.1

## Tập lệnh trong IAS (tiếp)

+

## 3. Máy tính thương mại UNIVAC

- ◼ 1947 – – Eckert and Mauchly thành lập Công ty máy tính EckertMauchly để sản xuất máy tính thương mại
- ◼ UNIVAC I (Universal Automatic Computer)
- ◼ Là máy tính thương mại thành công đầu tiên
- ◼ Được dùng cho cả các ứng dụng khoa học và thương mại
- ◼ Uỷ quyền bởi Cục điều tra dân số Mỹ để tính toán vào năm 1950
- ◼ UNIVAC II – – hoàn thành vào cuối những năm 1950
- ◼ Có dung lượng bộ nhớ lớn hơn và hiệu suất cao hơn
- ◼ Tương thích ngược: các chương trình viết cho các máy cũ có thể được thực hiện trên máy mới

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000014_047841ba05d05a2a0ebd70704722b0bdf878f89491db857e1b1979e4e3ab4f2e.png)

+

◼

◼

◼

◼

## 3. Máy tính thương mại (tiếp) IBM

Từng là hãng sản xuất thiết bị đục lỗ thẻ.

Chế tạo máy tính điện tử lưu trữ -chương trình đầu tiên (701) vào năm 1953: chủ yếu dành cho các ứng dụng khoa học

Dòng sản phẩm 702 được giới thiệu vào năm 1955: tính năng phần cứng làm nó phù hợp với các ứng dụng kinh doanh

Loạt máy tính thế hệ 700/7000 đã giúp IBM là nhà sản xuất máy tính 
ố y 
thống trị áp đảo

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000015_771520543b59d8856439d5713772ad1184970b888defe2964c0d7fe66e6f6c2f.png)

## + 1 .

## Sơ lược lịch sử phát triển máy tính

## b. Máy tính thế hệ thứ hai: transistor

Sự ra đời của transistor (linh kiện bán dẫn): là một thiết bị rắn làm từ silicon

Đặc điểm:

- ◼ Nhỏ gọn
- ◼ Giá thành rẻ
- ◼ Tản nhiệt ít hơn ống Vacuum
- ◼ Được phát minh bởi Bell Labs vào năm 1947
- ◼ Mãi đến cuối những năm 1950, máy tính bán dẫn hoàn toàn mới chính thức đưa vào thị trường thương mại

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000016_7f824eaf0495501ec5dc3e396ee448fd80d690ea5a2c310ec5d333754d13bc45.png)

◼

+

## Đặc điểm máy tính thế hệ thứ hai

- ◼ Sử dụng transistor:
- ◼ Hiệu năng xử lý tốt hơn, dung lượng bộ nhớ lớn hơn, kích thước nhỏ hơn
- ◼ Một số thay đổi khác:
- ◼ CU và ALU phức tạp hơn
- ◼ Sử dụng các ngôn ngữ lập trình bậc cao
- ◼ Xuất hiện các phần mềm hệ thống (giống như các hệ điều hành hiện đại như Window hay Linux) cho phép:
- ◼ Tải chương trình
- ◼ Di chuyển dữ liệu tới các thiết bị ngoại vi và thư viện
- ◼ Thực hiện các tính toán thông thường
- ◼ Giai đoạn này cũng đánh dấu sự xuất hiện của công ty DEC (Digital Equipment Corporation -DEC) vào năm 1957.
- ◼ PDP -1 là máy tính đầu tiên của DEC: máy tính mini đầu tiên 
ế – dòng máy y 
thống trị ở máy tính thế hệ thứ ba.
- ◼ Máy tính nổi bật ở thế hệ này là dòng máy IBM 7000 của hãng IBM (slide sau)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000017_266763015e989de1625c5e8f1cd144f2818fe507a21c9d086718919a78f7f29a.png)

## Bảng 2.3 Một số thông số của các dòng máy tính IBM 700/7000 Series

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000018_ec3ab6f0c4c469e42d5f9738c8f14bfaac24cfca988c6538600ee9a68f27e85e.png)

Figure 2.5 An IBM 7094 Configuration

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000019_0bc107dc1d60f8fb61c6595d6bd2f46b6f338a2b0c963946b68d7f1cb9cd1318.png)

## Cấu hình IBM 7094

## 1. Sơ lược lịch sử phát triển máy tính

- c. Máy tính thế hệ thứ ba: Vi mạch – – Mạch tích hợp (Intergated Circuit – IC)
2. ◼ Từ năm 1950 đến 1960, máy tính được chế tạo từ các linh kiện rời (transistor, điện trở, tụ điện)
3. ◼ Các linh kiện sản xuất đơn lẻ , độc lập , đóng gói riêng.
4. ◼ Sau đó được hàn lại hoặc nối với nhau lên trên một bảng mạch masonite .
5. ◼ Quá trình sản xuất tốn kém và cồng kềnh .
6. ◼ Các máy tính thế hệ thứ hai gồm khoảng 10000 transistor, sau đó con số này lên tới hàng trăm nghìn
7. ◼ 1958 – – phát minh ra mạch tích hợp → công nghệ nền tảng cho máy tính thế hệ thứ 3
8. ◼ Hai thành viên quan trọng nhất của thế hệ máy tính thứ ba là IBM System/360 và DEC PDP-8

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000020_484c12bd2a0531a9522597a1be06b1cba115400972d1ec3980152337ce13af8c.png)

+

## Khái niệm mạch tích hợp (IC)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000021_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

- ◼ Các thành phần cơ bản của máy tính số được chia thành 2 loại: gate (cổng logic) và memory cell (ô nhớ)
- ◼ Các thành phần này thực hiện bốn chức năng cơ bản của máy tính: lưu trữ, 
ềềể py 
truyền dữ liệu, xử lý dữ liệu và điều khiển.
- ◼ Trong đó:
- ◼ Gate: là thiết bị thực hiện các hàm logic và Boolean đơn giản
- ◼ Memory cell: là thiết bị lưu trữ một giá trị 0 hoặc 1
- ◼ Máy tính gồm các gate, memory cell và sự liên kết giữa chúng
- ◼ Lưu trữ dữ liệu: sử dụng các memory cell
- ◼ Xử lý dữ liệu: sử dụng các gate
- ◼ Truyền dữ liệu: Di chuyển dữ liệu – 
ẫ – Dữ liệu được di chuyển vào và ra bộ nhớ trên các yyy
đường dẫn giữa các bộ phận của máy tính và thông qua các gate.
- ◼ Điều khiển: các tín hiệu điểu khiển hoạt động của gate và memory cell

## Khái niệm mạch tích hợp (IC) + Sơ đồ gate và memory cell

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000022_8e750a45f21038fe17469e185df38750389f3057975d0c9687b1e7e78e1b7ebd.png)

+

## Khái niệm mạch tích hợp (IC)

- ◼ Mạch tích hợp: một tấm wafer silicon mỏng, chia thành ma trận các vùng nhỏ, mỗi vùng chứa một mạch giống hệt nhau được gọi là chip.
- ◼ Một chip chứa nhiều gate hoặc memory cell
- ◼ Chip được đóng gói (packaged)
- ◼ Ban đầu, số lượng gate/memory cell trong một chip còn ít → công nghệ này được gọi là SSI - smallscale integration: mạch tích hợp kích thước nhỏ
- ◼ Về sau, số lượng G/C trong một chip ngày càng nhiều

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000023_a64f5f2c2c619db0172f65fb9de7c8ad0d016fab6559fe511a31bffc7b441e9f.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000024_04049c59ee180b44125dcbfa14335a97c4483cb78fb294c31d1adc0b198652aa.png)

+

## Sự phát triển của Chip

uit

Figure 2 . 8 Growth in Transistor Count on Integrated Circuits (DRAM memory)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000025_cce0f1f3a5082ffd53f54199deaba5b02129c541836a3e9fea31f20058cf634d.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000026_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## Moore's Law

## 1965; Gordon Moore – đồng sáng lập Intel

Số lượng transistor trên mỗi chip sẽ tăng lên gấp đôi sau mỗi năm với giá thành không đổi

Tốc độ sau đó chậm lại thành gấp đôi sau m ỗi 18 tháng vào những năm 1970 và duy trì cho đến ngày nay

## Hệ quả của quy luật Moore:

Giá thành c ủa mạch bộ nhớ và logic máy tính đã giảm r ất mạnh

Chiều dài đường dẫn điện được rút ng ắn , tốc độ hoạt động tăng

Máy tính trở nên nhỏ gọn hơn và thuận tiện cho sử dụng ở các môi trường khác nhau

Giảm tiêu thụ điện năng và yêu cầu bộ làm mát

Kết nối giữa các chip ít hơn

+

## Table 2.4 Đặc tính của dòng máy System/360

## Table 2.5 Sự phát triển của dòng máy PDP -8

+

## Cấu trúc bus của DEC -PDP -8

Figure 2.9 PDP-8 Bus Structure

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000027_049ec3f44b90b216c09471eb3dc62e6621334e5188e19c5db81c518af51d24ba.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000028_c8d82c71f0e31410b9060362b7b7c84e51baa44b735ebc5d9f1b23b67a37b530.png)

## d. Các thế hệ tiếp theo

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000029_e4132a624bba463c66c7ba57bbc129dfd845ee7d1f0fda467998fe797d36e85a.png)

Bộ nhớ bán dẫn Bộ vi xử lý

VLSI Very Large Scale Integration

LSI Large Scale Integration

ULSI Ultra Large Scale Integration

+

## d. Các thế hệ tiếp theo

- ◼ Các thế hệ sau dựa trên sự phát triển của các công nghệ mạch tích hợp:
- ◼ Large-scale integration (LSI) - mạch tích hợp cỡ lớn: hơn 1000 thiết bị tích hợp trong một chip
- ◼ Very-large-scale integration (VLSI): 10000 thiết bị/chip
- ◼ Hiện nay, Ultra-large-scale integration (ULSI): hơn 1 tỉ thành phần/chip
- ◼ Các công nghệ này là nền tảng cho sự phát triển của các thế hệ máy 
ấế g gy g py 
tính và là công nghệ cơ bản cho việc sản xuất và chế tạo các linh kiện cơ bản:
- ◼ Bộ nhớ bán dẫn: mạch tích hợp ban đầu được sử dụng để chế tạo Bộ xử lý, 
ể ế p g ý
tuy nhiên, sau này người ta cũng sử dụng công nghệ đó để chế tạo bộ nhớ máy tính
- ◼ Vi xử lý: các bộ xử lý có kích thước nhỏ

## + a. Bộ nhớ bán dẫn

- ◼ Vào năm 1970 Fairchild giới thiệu bộ nhớ bán dẫn dung lượng tương đối lớn đầu tiên
- ◼ Chip đơn nhân
- ◼ Có thể chứa 256 bits bộ nhớ
- ◼ Không xoá được
- ◼ Tốc độ nhanh hơn lõi nhiều
- ◼ Vào năm 1974, giá thành trên 1 bit của bộ nhớ bán dẫn thấp hơn giá thành của bộ nhớ lõi
- ◼ Giá thành bộ nhớ tiếp tục giảm mạnh cùng với sự tăng nhanh của mật độ bộ nhớ vật lý
- ◼ Sự phát triển công nghệ bộ nhớ và xử lý đã làm thay đổi bản chất của máy tính trong suốt cả thập kỉ
- ◼ Kể từ năm 1970 bộ nhớ bán dẫn đã trả qua 13 thế hệ phát triển
- ◼ Mỗi thế hệ sau lại tăng mật độ bộ nhớ lên gấp 4 lần so với thế hệ trước cùng với giảm giá thành và thời gian truy câp .

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000030_9cbd931c77c437d4c2601ba98df0b759e5efd0aee503783d1111ff5a50ac445a.png)

+

## b. Vi xử lý

- ◼ Mật độ các thành phần trên chip xử lý tiếp tục tăng
- ◼ Ngày càng nhiều thành phần đặt trên chip dẫn đến càng ít chip cần thiết để xây dựng một bộ xử lý máy tính
- ◼ 1971 Intel phát triển dòng 4004
- ◼ Chip đầu tiên chứa được tất cả các thành phần của CPU trên một chip đơn
- ◼ Sự ra đời của bộ vi xử lý
- ◼ 1972 Intel phát triển dòng 8008
- ◼ Vi xử lý 8 bit đầu tiên
- ◼ 1974 Intel phát triển dòng 8080
- ◼ Vi xử lý đa năng đầu tiên
- ◼ Nhanh hơn , có một tập lệnh phong phú hơn , có khả năng định vị mạnh hơn

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000031_db11ca7686f1b3f3803bb1dcbaa17c76d8c871588b7db00daac22461832d51c3.png)

## Quá trình phát triển của vi xử lý Intel

## a. 1970s Processors

## b. 1980s Processors

## Quá trình phát triển của vi xử lý Intel

## c. 1990s Processors

## d. Recent Processors

+

## Key terms

- ◼ accumulator (AC) (bộ cộng tích lũy): thanh ghi AC
- ◼ arithmetic and logic unit (ALU): khối tính toán số học và logic
- ◼ Chip
- ◼ clock cycle: chu kỳ đồng hồ
- ◼ clock rate: tốc độ đồng hồ
- ◼ embedded system: hệ thống nhúng
- ◼ execute cycle: chu kỳ thực thi
- ◼ fetch cycle: chu kỳ truy xuất
- ◼ instruction buffer register (IBR): thanh ghi đệm lệnh
1. Khái niệm chương trình lưu trữ là gì?
2. Bốn thành phần chính của các máy tính đa nhiệm là gì?
3. Trình bày luật Moore.
4. Nêu các đặc điểm chính của máy IAS
5. Với công nghệ mạch tích hợp, các hệ thống máy tính có đặc điểm gì.
6. Mạch tích hợp được xây dựng dựa trên các thành phần nào?

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH02-Computer_Evolution_and_Performance_artifacts/image_000032_61c9c21759a96f516a40bf81e3410ac3504b3a17b7ab06a3391c81b8c2ba5cf0.png)