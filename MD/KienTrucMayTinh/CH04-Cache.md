## Chương 4. Bộ nhớ cache

- 4.1 Tổng quan về bộ nhớ máy tính
- 4.2 Nguyên lý của bộ nhớ cache
- 4.3 Các thành phần trong thiết kế bộ nhớ cache
- 4.4 Tổ chức cache của Pentium 4
- 4.5 Tổ chức cache trong ARM

## Một số khái niệm

- Từ (word): đơn vị " tự nhiên" của bộ nhớ . Kích thước từ thường bằng số bit biểu diễn một số nguyên và kích thước lệnh. Intel x86 có kích thước từ là 32b.
- Đơn vị đánh địa chỉ: ở các hệ thống khác nhau , đơn vị đánh địa chỉ có thể là byte hoặc word. Trong bất cứ trường hợp nào , mối quan hệ giữa số lượng các đơn vị đánh địa chỉ N và số bit địa chỉ A là 2 𝐴 = 𝑁
- Đơn vị truyền:
- Với bộ nhớ chính , đơn vị truyền bằng số lượng các bit được gửi đến hoặc đi từ bộ nhớ .
- Với bộ nhớ ngoài , đơn vị truyền thường lớn hơn rất nhiều , thường được gọi là các khối (block)

## Ví dụ

1. VXL Intel x86 -32b, kết nối bus (32 đường địa chỉ, 16 đường dữ liệu) với bộ nhớ tổ chức dưới dạng các ngăn nhớ 8b. Hãy cho biết:
- a. Kích thước word của BN trên
- b. Dung lượng tối đa của bộ nhớ mà VXL có thể quản lý được.
- c. Đơn vị truyền của BN trên. Để thực hiện một lệnh: cộng 2 số (trong bộ nhớ) và ghi kết quả vào 1 ngăn nhớ khác thì VXL sẽ phải thực hiện bao nhiêu thao tác đọc, ghi BN

## 4.1 Tổng quan về bộ nhớ máy tính Phân loại bộ nhớ máy tính

| Vị trí Bên trong (vd: thanh ghi ,  cache, bộ nhớ chính Bên ngoài (vd: đĩa quang ,  đĩa từ ,  băng từ) Dung lượng Số lượng từ Số lượng byte Đơn vị truyền Từ Khối Phương pháp truy cập Tuần tự Trực tiếp Ngẫu nhiên Kết hợp   | Hiệu suất Thời gian truy cập Chu kỳ xung nhịp Tốc độ truyền tải Loại vật lý Bán dẫn Từ Quang học Quang từ Tính chất vật lý Điện động/điện tĩnh (Dữ liệu có bị mất khi mất điện) Có thể xóa/không xóa được Tổ chức Module bộ nhớ   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## a. Vị trí

- Bộ nhớ có thể ở trong và ngoài máy tính
- Bộ nhớ chính là bộ nhớ trong
- Bộ xử lý cần có bộ nhớ cục bộ riêng của nó: thanh ghi
- Cache là một dạng khác của bộ nhớ trong
- Bộ nhớ ngoài bao gồm các thiết bị lưu trữ ngoại vi có thể truy cập vào bộ xử lý thông qua bộ điều khiển I/O

## b. Dung lượng

- Bộ nhớ thường được biểu diễn dưới dạng byte

## c. Đơn vị truyền

- Đối với bộ nhớ trong, đơn vị truyền bằng số lượng đường điện đi vào và ra khỏi module bộ nhớ

## Phân loại bộ nhớ

## Phân loại bộ nhớ (tiếp)

## d. Phương pháp truy cập các khối dữ liệu

| Truy cập tuần tự                                                                                                                                                      | Truy cập trực tiếp                                                                                                                                            | Truy cập ngẫu nhiên                                                                                                                                                                                                                                                                                             | Kết hợp                                                                                                                                                                                                                                                                |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| •  Bộ nhớ được tổ  chức thành các  đơn vị dữ liệu  được gọi là bản ghi (record) •  Truy cập được  thực hiện tuần tự •  Thời gian truy  cập biến đổi •  Ví dụ: băng từ | •  Có một cơ chế  đọc-ghi chia sẻ •  Mỗi khối hoặc  bản ghi có một địa  chỉ duy nhất dựa  trên vị trí vật lý •  Thời gian truy  cập biến đổi •  Ví dụ: đĩa từ | •  Mỗi vị trí trong bộ nhớ  có một cơ chế định địa  chỉ riêng •  Thời gian truy cập vào  một vị trí nhất định  không đổi và phụ thuộc  vào chuỗi các truy cập  trước đó  •  Một vị trí bất kỳ có thể  được chọn ngẫu nhiên , định địa chỉ và truy cập trực tiếp  •  Ví dụ: bộ nhớ chính và  một số bộ nhớ cache | •  Một word được  truy xuất dựa trên  một phần nội dung  thay vì địa chỉ của nó •  Mỗi vị trí có cơ chế  định địa chỉ riêng .  Thời gian truy xuất là  không đổi ,  phụ thuộc  vào vị trí hoặc các  truy cập trước đó •  Bộ nhớ Cache có  thể sử dụng truy cập kết hợp |

## e. Hiệu năng

## Hai đặc điểm quan trọng nhất của bộ nhớ: dung lượng và hiệu năng

## Ba tham số hiệu năng được sử dụng:

## Chu kỳ bộ nhớ

- Với bộ nhớ truy cập ngẫu nhiên: Thời gian truy cập cộng với thời gian cần trước khi truy cập thứ hai có thể bắt đầu
- Có thể cần thêm thời gian để các transients chết trên đường tín hiệu hoặc để khôi phục lại dữ liệu bị hỏng
- Liên quan đến hệ thống bus, không liên quan bộ xử lý

## Thời gian truy cập (độ trễ)

- Đối với bộ nhớ truy cập ngẫu nhiên, nó là thời gian cần để thực hiện 1 thao tác đọc hoặc ghi
- Đối với bộ nhớ truy cập không ngẫu nhiên , nó là thời gian cần để đặt cơ chế đọc -ghi vào vị trí mong muốn

## Tốc độ truyền tải

- Tốc độ truyền dữ liệu vào hoặc ra khỏi bộ nhớ
- Đối với bộ nhớ truy cập ngẫu nhiên, tốc độ truyền tải bằng 1/(chu kỳ)

## f. Đặc tính vật lý của bộ nhớ

- -Các dạng phổ biến nhất là: Bộ nhớ bán dẫn , Bộ nhớ bề mặt từ , Bộ nhớ quang , Bộ nhớ quang từ
- -Một số đặc điểm vật lý quan trọng:
1. Đặc điểm lưu trữ dữ liệu
- Bộ nhớ điện động (Volatile memory): thông tin bị suy yếu hoặc bị mất khi nguồn điện tắt
- Bộ nhớ điện tĩnh (Non-volatile memory): thông tin một khi đã được ghi thì sẽ không bị mất trừ khi cố tình thay đổi kể cả không có nguồn cung cấp VD: ROM, USB, HDD,…
2. Công nghệ sản xuất:
- Bộ nhớ bề mặt từ (Magnetic -surface memories): HDD, Tape
- Bộ nhớ bán dẫn (Semiconductor memory): RAM, ROM, Cache,…
- Bộ nhớ không xoá được (Nonerasable memory): Không thể thay đổi, trừ khi phá hủy các khối lưu trữ. VD: ROM
- ·

VD: RAM, Cache

## g. Tổ chức bộ nhớ: mô hình phân cấp bộ nhớ

- Thiết kế bộ nhớ của máy tính cần trả lời ba câu hỏi:
- How much? How fast? How expensive?
- Cần có sự cân đối giữa dung lượng, thời gian truy cập và chi phí
- Thời gian truy cập nhanh hơn, chi phí lớn hơn cho mỗi bit
- ·
- Dung lượng lớn hơn, chi phí nhỏ hơn cho mỗi bit
- ·
- Dung lượng lớn hơn, thời gian truy cập chậm hơn
- Giải pháp:
- Không dựa hoàn toàn vào một thành phần hoặc công nghệ bộ nhớ
- Sử dụng một hệ thống phân cấp bộ nhớ

## Bộ nhớ phân cấp -Sơ đồ

- Chi phí trên bit giảm
- Dung lượng tăng
- Thời gian truy cập tăng
- Tần suất truy cập bộ nhớ của VXL giảm

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000000_4102787e8f1e444b1767b6cc66d3c493a0b88a7b92d6afa74e4944b13689116b.png)

## 4.2. Nguyên lý bộ nhớ cache

## Bộ nhớ cache và bộ nhớ chính

- BXL truy cập xuất lệnh/dữ liệu từ BN chính theo đơn vị byte hoặc word → tốc độ chậm (do tốc độ BN chính, bus chậm hơn VXL)
- Bộ nhớ cache được thiết kế để cải thiện thời gian truy cập bộ nhớ:
- Dựa vào tính cục bộ của dữ liệu và lệnh lưu trữ trong BN chính
- BN cache có tốc độ cao nhưng dung lượng thấp hơn bộ nhớ chính
- Bộ nhớ cache chứa bản sao của một phần của bộ nhớ chính .

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000001_3234e089d9d727fb9a51f68898d680b8613aab352be7573d7da7ce9cb15ab2f7.png)

## Nguyên lý

- BN chính gồm 𝟐 𝒏 từ nhớ (word) được đánh địa chỉ: n bit địa chỉ
- BN chính được chia thành các khối (block) có kích thước cố định: K word .
- ➢Như vậy, BN chính có 2 𝑛 𝐾 = 𝑴 khối
- BN cache được chia thành các đường (line) , mỗi đường có K word.
- Mỗi block của BN chính được ánh xạ vào một line của Cache
- Khi bộ xử lý muốn đọc một word của bộ nhớ nó sẽ kiểm tra xem word đó có nằm trong bộ nhớ cache hay không .
- ➢Nếu có: word này được gửi đến bộ vi xử lý .
- ➢Nếu không: một khối dữ liệu từ bộ nhớ chính (chứa từ mà VXL đang muốn truy cập), được đọc vào bộ nhớ cache và sau đó từ được gửi đến bộ VXL.

## · Tổ chức cache

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000002_079d31afa90b073c36ecf6dda4611c47fc97231d78bc01a19dd64802ddf726ef.png)

## Cấu trúc bộ nhớ chính/cache

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000003_3f1fcdd47ddb12f07f9005ae302ec65312326a8dfbc5a00d9edb2ef7d951091b.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000004_73389a94d6fe872d43d568b01a7866dd712284cf47cae403c0e65b8b79d6d544.png)

## Tổ chức bộ nhớ cache điển hình

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000005_36d3b59d766f6c74424ec15244eb1b75ec3a46aadaa49fb4d9931b0c2f8c04fc.png)

## 4.3. Các yếu tố khi thiết kế Cache

- e. Chính sách ghi Ghi xuôi Ghi ngược
- f. Kích thước line
- g. Cache nhiều cấp Một hoặc hai cấp

```
Thống nhất hoặc phân chia
```

```
Table 4.2 Elements of Cache Design a. Địa chỉ bộ nhớ cache Logic Vật lý b. Kích thước bộ nhớ cache c. Ánh xạ bộ nhớ Trực tiếp Kết hợp Tập kết hợp d. Thuật toán thay thế Least recently used (LRU) First in first out (FIFO) Least frequently used (LFU) Random
```

## a. Địa chỉ bộ nhớ cache

- Địa chỉ ảo: bộ xử lý hỗ trợ bộ nhớ ảo:
- ✓Quản lý bộ nhớ thông qua địa chỉ logic
- ✓Các trường địa chỉ trong lệnh là các địa chỉ ảo
- ✓Để thực hiện các thao tác đọc/ghi vào bộ nhớ chính, khối quản lý bộ nhớ (MMU – Memory Management Unit) sẽ dịch từng địa chỉ ảo sang địa chỉ vật lý trong bộ nhớ chính
- Cache ảo (cache logic): bn cache đặt giữa BXL và MMU
- Địa chỉ được sử dụng là địa chỉ ảo
- Cache vật lý: bn cache đặt giữa MMU và bộ nhớ chính
- Địa chỉ được sử dụng là địa chỉ vật lý

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000006_3eec5c3357e6a959ba8946e4b92da13f98c693aa0064141897fdd4de364f95d5.png)

## b. Kích thước cache (cache size)

- Kích thước cache phải đủ nhỏ để không làm giá thành tăng cao
- Kích thước cache phải đủ lớn để giảm thời gian truy cập , tăng hiệu suất hệ thống
- Ngoài ra , kích thước cache quá lớn sẽ làm tăng số cổng để định địa chỉ cho các vị trí nhớ trong cache
- → giảm hiệu quả truy cập ngay cả khi cache nằm trong cùng chip hoặc board với VXL

b. Kích thước cache trong một số bộ xử lý

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000007_0dd6227393c579780a3d85ee0d9b2bc42fa9baae7f4737d9062ce30852ef9e0a.png)

a, Hai giá trị cách nhau bằng dấu / là cache chỉ thị và cache dữ liệu.

b, Cả hai cache đều là cache chỉ thị; Không có cache dữ liệu.

## c. Ánh xạ bộ nhớ

- Bởi vì số đường cache ít hơn số khối bộ nhớ chính, cần có một thuật toán ánh xạ các khối bộ nhớ chính vào các đường bộ nhớ cache
- Ba kỹ thuật có thể được sử dụng:

## Trực tiếp

- Mỗi khối của bộ nhớ chính được ánh xạ vào một đường cache duy nhất
- Đơn giản nhất

## Kết hợp

- Cho phép một khối nhớ chính được nạp vào bất kỳ đường cache nào
- Logic điều khiển cache diễn giải địa chỉ bộ nhớ bằng một trường Tag và trường Word
- Để xác định một khối có ở trong một cache không , logic điều khiển cache phải cùng lúc kiểm tra Tag của tất cả các đường

## Set Associative

- Kết hợp hai phương pháp trên
- Thể hiện ưu điểm của cả phương pháp trực tiếp và kết hợp , đồng thời giảm nhược điểm

## Ánh xạ trực tiếp (Direct mapping)

- Mỗi khối (block) của bộ nhớ chính được ánh xạ vào một đường (line) nhất định của bộ nhớ cache.
- Cách xác định: giả sử BN cache có m line. Vậy, block thứ j trong BN chính sẽ được ánh xạ vào line nào trong BN cache?

Với i là số thứ tự line mà block đó được ánh xạ vào, ta có

𝒊 = 𝒋 𝒎𝒐𝒅𝒖𝒍𝒐 𝒎 (phép chia lấy dư)

- Do vậy, y, nhiều block sẽ được ánh xạ vào một line. Để xác định block nào đang được ánh xạ vào cache: sử dụng trường tag

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000008_b2c62f8ff6371acac6dd4aa3197f386313c8a5709e8cb40cae27c8f1a5140d10.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000009_799af832711763d75e63c69d768c24d5fe1624740fc72efabdaa24cf15ac420f.png)

Ví dụ ánh xạ trực tiếp

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000010_58995039932a7dc6945222d11a7931c64d1a843bd6801f0ee0206f71cd7bba58.png)

## Tổng kết ánh xạ trực tiếp

- Độ dài địa chỉ = (𝑠 + 𝑤) bit
- Số ô nhớ trong bộ nhớ chính = 2 𝑠+𝑤 word hoặc byte
- Kích thước khối = kích thước đường = 2 𝑤 word hoặc byte
- Số khối trong bộ nhớ chính = 2 𝑠+ 𝑤 /2 𝑤 = 2 𝑠
- Số đường trong bộ nhớ cache = 𝑚 = 2 𝑟
- Kích thước của tag = (𝑠 – 𝑟) bit

Nhược điểm: các khối lưu cố định tại 1 đường trong bộ nhớ cache. Vậy nếu chương trình tham chiếu các từ lặp lại từ hai khối mà cùng ánh xạ đến 1 đường thì cache liên tục phải đổi dl từ memory vào , làm giảm hiệu suất (hiện tượng thrashing)

## Bài tập ví dụ

Bộ nhớ chính: 2 16 byte, kích thước khối 8 byte, ánh xạ trực tiếp vào cache 32 đường (kích thước ngăn nhớ = 1B).

- a. 16 bit địa chỉ được chia thành các trường Tag, Line và Word như thế nào?
- b. Các địa chỉ sau sẽ được lưu ở đường nào của cache?
- c. Giả sử byte có địa chỉ 0001 1010 0001 1010 được lưu ở cache, các byte nào của bộ nhớ chính cũng được lưu trên đường đó?
- d. Có bao nhiêu byte có thể được lưu trên cache?

```
0001 0001 0001 1011 1101 0000 0001 1101 1100 0011 0011 0100 1010 1010 1010 1010
```

## Ánh xạ kết hợp (Associative Mapping)

- Ánh xạ kết hợp khắc phục nhược điểm của ánh xạ trực tiếp bằng cách cho phép mỗi khối được nạp vào bất kỳ đường nào của bộ nhớ cache
- Trong trường hợp này, bộ logic điều khiển bộ nhớ cache (cache control logic) tách địa chỉ bộ nhớ thành hai trường: Tag và Word. Trường Tag hiển thị duy nhất một khối bộ nhớ chính.
- Để xác định liệu một khối có trong bộ nhớ cache, bộ logic điều khiển bộ nhớ cache phải cùng lúc kiểm tra mỗi Tag của một đường để so sánh

## Tổ chức ánh xạ kết hợp

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000011_c28891dd31310add0cbd7de6bf8407d4f53a416c2b1d2d8802f2013cfdeee7f9.png)

Ví dụ

ánh xạ

kết hợp

## Tổng hợp ánh xạ kết hợp

- Chiều dài địa chỉ = (s + w) bit
- Số ô nhớ được đánh địa chỉ = 2
- Kích thước khối = kích thước đường = 2 w word hoặc byte
- Số lượng khối trong bộ nhớ chính = 2s+ w/2 w = 2s
- Số đường trong cache = không xác định
- Chiều dài trường tag = s bit

Ưu điểm: linh hoạt khi thay thế một khối và đọc một khối mới vào cache.

Các thuật toán thay thế được xây dựng để tối ưu hóa tỷ lệ truy cập.

Nhược điểm: mạch phức tạp để thực hiện việc kiểm tra tất cả trường Tag của các đường trong cache một cách song song.

## Ánh xạ tập kết hợp (Set Associative Mapping)

- Tận dụng các ưu điểm của cả phương pháp trên đồng thời giảm nhược điểm của chúng
- Chia cache thành một số Tập (set) -Mỗi Tập chứa một số đường
- Một khối sẽ được ánh xạ vào một đường bất kỳ trong một Tập nhất định

## · Quan hệ

<!-- formula-not-decoded -->

i = j/v (chia lấy phần dư)

Trong đó

i = số thứ tự Tập trong cache j = số thứ tự khối trong bộ nhớ chính

m = số lượng đường trong cache v = số lượng Tập có trong cache

k = số lượng đường trong mỗi Tập

- Ví dụ: 1 Tập có 2 đường
- ·
- Ánh xạ kết hợp 2 chiều
- Một khối có thể nằm trong 1 trong 2 đường trong một Tập

Ánh xạ từ bộ nhớ chính đến bộ nhớ Cache:

k -Way Set Associative

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000012_7746ddb0437e8adb18e3ff4793956fa526f4d05d1ea84ee0ef58e6b8a06b15fe.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000013_3ddf0d2831a36e18a3da598acbe8b26c944d814e008e6c00f5567d7d69a7c53e.png)

## Tổ chức cache k -Way Set Associative

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000014_6c29b60df69030f213a720b86b0b150eab4725cb3635d5517d2c0d604b5ae662.png)

## Tổng kết ánh xạ Set Associative

- Chiều dài địa chỉ = (s + w) bit
- Số lượng ô nhớ được đánh địa chỉ = 2
- Kích thước khối (hoặc đường) = 2 w word hoặc byte
- Số khối trong BN chính = 2s+w/2
- Số đường trong 1 Tập= k · Số lượng Tập = v = 2d · Số lượng đường trong cache = m= kv = k * 2 d · Kích thước cache = k * 2
- Độ rộng trường Tag = (s – d) bits

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000015_e1d6c93f8615ae6584db7f8629423ad3309d9867b48f548669a21d6342faa32f.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000016_fd06fa32e306ef26d81fb285875bf033c0fff55e298aa468a54eb0259fa56f25.png)

## So sánh hiệu suất của các PP ánh xạ

- •

- Cache hit: số lần truy cập cache thành công

- •

- Cache miss: số lần truy cập cache không thành công

- •

- Hit ratio: tỷ lệ truy cập

<!-- formula-not-decoded -->

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000017_d720a780aab346b2e73fb28cba2aca366febbcedc77634b33f83f9afba973951.png)

Bài tập ví dụ

1. Xét VXL 32 bit, địa chỉ 32b. Bộ nhớ cache 16 -Kbyte, ánh xạ tập kết hợp 4 -way. Giả sử một đường gồm 4 từ 32 -bit (mỗi từ nhớ 32 bit, 1 đường có 4 từ). Xác định các trường của địa chỉ được sử dụng để ánh xạ cache. Từ nhớ có địa chỉ ABCDE8F8 được ánh xạ vào vị trí nào trong cache.
2. Bộ nhớ Cache 64 đường sử dụng ánh xạ tập kết hợp 4 đường . Bộ nhớ chính có 4K khối , mỗi khối có kích thước 128 từ . Xác định định dạng địa chỉ bộ nhớ . Khối 304 được ánh xạ vào đường nào trong cache

## d. Thuật toán thay thế

- Khi bộ nhớ cache đã đầy, nếu một khối mới được đưa vào cache, một trong những khối hiện có phải được thay thế
- Đối với ax trực tiếp: một khối bất kỳ chỉ có thể ánh xạ vào 1 đường cụ thể. Nên khi cần đưa 1 khối mới vào cache buộc phải xóa dữ liệu cũ trên đường tương ứng đi.
- Đối với các kỹ thuật kết hợp và tập kết hợp, một khối có thể được ánh xạ vào 1 số đường. Vậy khi đưa 1 khối mới vào cache, ta cần xác định xem khối đó sẽ được ánh xạ vào đường nào: thuật toán thay thế
- Để đạt được tốc độ cao, thuật toán phải được thực hiện trong phần cứng

## 4 thuật toán thay thế phổ biến nhất

## · Least recently used (LRU)

- Hiệu quả nhất
- Thay thế khối nằm trong cache lâu nhất mà không có tham chiếu đến nó
- Do triển khai đơn giản, LRU là thuật toán thay thế phổ biến nhất

## · First -in -first -out (FIFO)

- Thay thế khối đã nằm trong cache lâu nhất
- Dễ dàng thực hiện như một kỹ thuật vòng đệm hoặc round -robin

## · Least frequently used (LFU)

- Thay thế khối có ít tham chiếu đến nó nhất
- Ở mỗi line thêm vào một bộ đếm, mỗi khi có tham chiếu đến line nào, bộ đếm của line đó tăng thêm 1 đơn vị

## · Ngẫu nhiên

- Có thể thay thế bất cứ khối nào
- Các nghiên cứu đã chỉ ra: thay thế ngẫu nhiên chỉ làm giảm hiệu suất của hệ thống đi một chút so với các thuật toán thay thế ở trên

## e. Chính sách ghi

Khi một khối trong cache được thay thế, có 2 trường hợp cần xem xét:

- Nếu dữ liệu trong cache không thay đổi, có thể ghi đè khối mới lên mà không cần ghi khối cũ ra trước
- Nếu ít nhất 1 thao tác ghi đã được thực hiện trên 1 word trong đường của cache thì bộ nhớ chính phải được cập nhật bằng cách ghi dữ liệu từ cache ra bộ nhớ trước khi đưa khối mới vào

## Có hai vấn đề phải đối mặt:

- Nhiều thiết bị có thể có quyền truy cập vào bộ nhớ chính
- Trong trường hợp VXL đa nhân: nhiều cache tương ứng với
- các nhân → gây khó khăn trong việc quản lý dữ liệu
- Có hai chính sách ghi: write though và write back

## Write Through

## và Write Back

- Write through: khi VXL gửi lệnh ghi dữ liệu ra bộ nhớ, dữ liệu sẽ được ghi đồng thời ra cả cache và BN
- Kỹ thuật đơn giản nhất
- Tất cả các thao tác ghi được thực hiện cho bộ nhớ chính cũng như cache
- Nhược điểm: tạo ra lưu lượng bộ nhớ đáng kể và có thể tạo ra nút cổ chai
- Write back: khi VXL gửi lệnh ghi, dữ liệu chỉ được cập nhật trên cache, sử dụng 1 bit (gọi là dirty bit) thiết lập giá trị để đánh dấu là dữ liệu đã bị thay đổi. Việc cập nhật dữ liệu lên BNC chỉ xảy ra khi thay thế khối mới vào cache
- Giảm thao tác ghi bộ nhớ
- Dữ liệu trên BNC không có hiệu lực. Cơ chế DMA bắt buộc phải thực hiệu qua cache
- Nhược điểm: mạch phức tạp và khả năng có nút cổ chai

## f. Kích thước Line

- Khi khối dữ liệu được lấy ra và đặt trong cache, sẽ thu được không chỉ word mong muốn mà còn 1 số word liền kề
- Khi kích thước khối tăng, ban đầu tỷ lệ truy cập sẽ tăng do nguyên tắc cục bộ → dữ liệu hữu ích (dữ liệu có khả năng lớn được truy cập trong câu lệnh tiếp theo) được đưa vào cache
- Tuy nhiên, khi kích thước khối quá lớn, tỷ lệ này giảm đi do:
- 1, Các khối lớn hơn làm giảm số lượng đường trong một cache
- 2, Khi kích thước khối lớn, mỗi word thêm vào lại càng xa word yêu cầu → ít có khả năng truy xuất

## g. Cache nhiều cấp

- Với các hệ thống ngày nay, người ta thường sử dụng nhiều cache trong kiến trúc
- Cache trên chip (cache on chip) làm giảm hoạt động bus ngoài của bộ xử lý; tăng tốc thời gian xử lý và tăng hiệu năng toàn hệ thống
- Khi lệnh hoặc dữ liệu được tìm thấy trong cache → không cần truy cập bus
- Truy cập bộ nhớ cache trên chip nhanh hơn đáng kể
- Trong giai đoạn này, bus tự do hỗ trợ các lượt truyền khác
- Cache 2 cấp (cache 2 level) : sử dụng một cache on chip (cache L1) và một cache bên ngoài (cache L2)
- Cải thiện hiệu suất
- Việc sử dụng cache nhiều cấp làm cho các vấn đề thiết kế liên quan đến cache phức tạp hơn, gồm kích thước, thuật toán thay thế, chính sách ghi

## Tỉ lệ truy cập (L1 &amp; L2) cho 8 Kbyte và 16 Kbyte L1

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000018_766382d3c97900b41e1ddc3867b16e58b409d9111ae64369914575f8e31887f1.png)

## Cache thống nhất / cache phân chia

- Cache thống nhất: lệnh và dữ liệu được lưu trên cùng một cache
- Ưu điểm: Tốc độ truy cập cao hơn
- Tự động cân bằng giữa việc nạp dữ liệu và lệnh
- Chỉ cần thiết kế và thực hiện một bộ nhớ cache
- Cache phân chia: lệnh và dữ liệu được lưu trên hai cache khác nhau , thường là cache L1
- Ưu điểm:Loại bỏ sự cạnh tranh cache giữa khối tìm nạp/giải mã lệnh và khối thực hiện
- Xu hướng: cache phân chia ở L1 và cache thống nhất ở các level cao hơn

## 4 . 4 . Tổ chức Cache Pentium 4

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000019_13d020d46c14f3922843254074af24ecb61b526319d024500cf2cfa54c08104a.png)

Bảng 4.4 Intel Cache Evolution

## Sơ đồ khối Pentium 4

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000020_a6b1820e1d11f50c7bc8387b16c2f8dbca65d28d9f46ade10904439fe6281fd7.png)

## Các chế độ hoạt động Cache Pentium 4

Note: CD = 0; NW = 1 là kết hợp không hợp lệ.

## 4.5. Tổ chức cache ARM Đặc tính Cache ARM

## ARM Cache và tổ chức bộ đệm ghi

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH04-Cache_artifacts/image_000021_06291655e4cabfc13a7bd66ddfa7495abd15e8943976845078ec0eeb15f25e95.png)

## Tổng kết

## Chương 4: Bộ nhớ Cache

- Đặc điểm của hệ thống bộ nhớ
- Vị trí
- Dung lượng
- Đơn vị truyền
- Bộ nhớ phân cấp
- How much?
- How fast?
- How expensive?
- Nguyên lý bộ nhớ Cache
- Các yếu tố trong thiết kế cache
- Địa chỉ bộ nhớ cache
- Kích thước bộ nhớ cache
- Ánh xạ bộ nhớ
- Thuật toán thay thế
- Chính sách ghi
- Kích thước line
- Cache nhiều cấp
- Tổ chức cache Pentium 4
- Tổ chức cache ARM

## Từ khóa

- Cache: bộ nhớ cache

- Cache hit: việc truy cập thành công vào bộ nhớ cache

- Cache miss: truy cập BN cache không thành công

- Hit ratio: tỷ lệ truy cập BN cache

- Cache mapping: ánh xạ BN cache

- Write policy: chính sách ghi

- Cache L1, L2, L3: BN cache level 1, 2,3

- Line: đường trong cache

- Block: khối

- Set: tập

## Review questions

1. Sự khác nhau giữa truy cập tuần tự , trực tiếp và ngẫu nhiên là gì?
2. Nêu mối quan hệ giữa thời gian truy cập , giá thành và dung lượng bộ nhớ .
3. Sự khác nhau giữa ánh xạ trực tiếp , kết hợp và tập kết hợp là gì?
4. Khái niệm cache phân chia và cache thống nhất. Ứng dụng?
5. Các yếu tố chính trong thiết kế cache là gì
6. Write though và write back khác nhau như thế nào? Kỹ thuật nào giảm số lần truy cập bus hệ thống nhiều hơn?
7. Trình bày các kỹ thuật thay thế .
8. Kích thước của đường có ảnh hưởng như thế nào đến hiệu suất cache (cache hit)?
9. Dung lượng BN cache có ảnh hưởng như thế nào đến hiệu suất cache (cache hit)?