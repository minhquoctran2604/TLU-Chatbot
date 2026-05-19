![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000000_45b4415caa5e0e7d5ca85acebdbbb08c592894feb5ccbc1360ac5f4ed6d44a6c.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000001_4aca9640898fb31cf1590328b5be155873419007652a80d92f7211e7afc9638a.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000002_93c9dfa7ddbb75f7ec2892d9f5831d85346132f2abd4890feb031ed21541981a.png)

Kiến trúc máy tính Bộ môn Kỹ thuật máy tính và mạng

## + Chương 10

Tập lệnh:

Đặc điểm và chức năng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000003_b8eb0923bd17d7163e0d6bec6d87c16bc580eedd820ebf12c6c183e0a64dc0b6.png)

## + Chương 10. Tập lệnh

1. Các đặc điểm của lệnh máy
- a. Thành phần của lệnh máy
- b. Biểu diễn lệnh
- c. Các loại lệnh
- d. Số lượng địa chỉ
- e. Thiết kế tập lệnh
2. Các kiểu toán hạng
- a. Số
- b. Ký tự
- c. Dữ liệu logic
3. Các kiểu dữ liệu Intel x86 và ARM
- a. Các kiểu dữ liệu x86
- b. Các kiểu dữ liệu ARM
4. Các loại hoạt động
- a. Truyền dữ liệu
- b. Số học
- c. Logic
- d. Chuyển đổi
- e. Vào/ra
- f. Điều khiển hệ thống
- g. Truyền điều khiển
5. Các loại hoạt động Intel x86 và ARM
- a. Các loại hoạt động x86
- b. Các loại hoạt động ARM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000004_36b4cddd087fd72e076a25f439fd6688b1898eb87491b67ff9b2240522d6ff62.png)

+

## 10.1 Đặc điểm tập lệnh

- ◼ Hoạt động (operation) của VXL được quyết định bởi các lệnh nó thực hiện , đó là các lệnh máy tính (machine instructions hay computer instructions)
- ◼ VD: Lệnh STORE: lưu trữ dữ liệu vào bộ nhớ
- ◼ Tập hợp các lệnh khác nhau mà VXL có thể thực hiện được gọi là tập lệnh (instruction set) của VXL
- ◼ Mỗi lệnh phải có những thông tin cần thiết cho bộ xử lý thực hiện hoạt động
- ◼ VD: Lệnh STORE ở trên phải đi kèm địa chỉ ngăn nhớ mà dữ liệu được ghi vào

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000005_e101736e81e3062893a300637478fdb79db4c04de88ff2709faef9535fd44ecf.png)

## Biểu đồ chu kỳ lệnh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000006_a01af81bf6bf6cc088984587c019a7635fae81257b5e969777ebeb9d9d2a7294.png)

+

## a. Các thành phần của lệnh

- ◼ Mã lệnh - Operation code (opcode): chỉ ra hoạt động (operation: hoạt động/phép toán) được thực hiện thông qua một mã nhị phân được gọi là mã lệnh (opcode)
- ◼ Tham chiếu toán hạng nguồn: Mỗi hoạt động có thể tham chiếu đến một hoặc 
ềể ấầ ạg gạộg ộặ
nhiều toán hạng để lấy dữ liệu đầu vào cho hoạt động: các toán hạng này được gọi 
ồ ạg 
là toán hạng nguồn
- ◼ Tham chiếu toán hạng kết quả (toán hạng đích): Hoạt động có thể đưa ra một kết quả
- ◼ Tham chiếu lệnh tiếp theo: Một số hoạt động có thể rẽ nhánh đến một câu lệnh ở vị 
ể ấế ệp ộạộg ộệị 
trí khác → nói cho VXL nơi để lấy lệnh tiếp theo sau khi việc thực thi lệnh hiện tại hoàn thành
- ◼ VD: lệnh của máy IAS (20b: 8b opcode và 12b địa chỉ toán hạng)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000007_c277d5d89cc938f0ffcd124f19102e0dcc6fe3e80dd3897fc4ea302e0ec58681.png)

| Mã lệnh (Opcode)             | Mô tả hợp ngữ   | Công việc                                                                                                           |
|------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------------|
| 00100001                     | STOR M(X)       | Truyền dữ liệu từ thanh ghi AC vào ngăn  nhớ có địa chỉ X trong bộ nhớ (M)                                          |
| VD: 00100001 0001  1010 0000 | STOR M(1A0)     | Truyền dữ liệu từ AC vào ngăn nhớ có  địa chỉ 1A0 trong bộ nhớ. AC: toán hạng nguồn (ngầm định) 1A0: toán hạng đích |

## Các toán hạng nguồn và kết quả có thể ở một trong bốn vùng sau:

- 1) Bộ nhớ chính hoặc bộ nhớ ảo
2. ◼ Toán hạng nguồn/kết quả có thể là một vị trí bộ nhớ chính hoặc bộ nhớ ảo. Trong lệnh phải chứa địa chỉ bộ nhớ chính hoặc bộ nhớ ảo của toán hạng đó
- 3) Tức thì
4. ◼ Giá trị của toán hạng có thể được đưa trực tiếp vào trong câu lệnh
- 2) Thanh ghi
6. ◼ Toán hạng nguồn hoặc kết quả có thể là các thanh ghi. Một VXL chứa một hoặc nhiều thanh ghi, mỗi thanh ghi được gán cho một tên hoặc số riêng.
7. ◼ Một lệnh có thể tham chiếu đến các thanh ghi này.
- 4) Thiết bị vào/ra (I/O)
9. ◼ Dữ liệu có thể lấy từ (hoặc ghi ra) một thiết bị I/O → lệnh phải chỉ ra địa chỉ thiết bị và module vào/ra tương ứng

## + b. Biểu diễn lệnh

- ◼ Trong máy tính , mỗi câu lệnh được biểu diễn bằng một chuỗi bit nhị phân
- ◼ Câu lệnh được chia ra thành các trường tương ứng với các
ầấ thành phần cấu thành của lệnh
- ◼ Ví dụ
- ◼ Trong các tài liệu, để dễ hiểu, lệnh thường được biểu diễn dưới dạng các ký hiệu thay vì các bit nhị phân. Opcode được viết tắt, mô tả hoạt động (phép toán).

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000008_a4d88fde64d9d5d6679bd3a38f6d4425f07728cccf3cf51bbb93e67f4526101e.png)

```
Ví dụ: Tập lệnh IAS ADD SUB MUL DIV LOAD STOR Cộng Trừ Nhân Chia Tải dữ liệu từ bộ nhớ Lưu dữ liệu vào bộ nhớ
```

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000009_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## + Ví dụ:

NNLT bậc cao được đưa ra để giúp công việc của lập trình viên thuận lợi hơn

Ví dụ câu lệnh: X=X+Y viết bằng NN C++ nếu dịch sang tập lệnh IAS sẽ gồm các lệnh như sau

| 0010 0000 0001 0000   | LOAD X   | Đọc X từ bộ nhớ vào thanh ghi AC                    |
|-----------------------|----------|-----------------------------------------------------|
| 0101 0000 0001 0001   | ADD Y    | Đọc Y từ bộ nhớ, cộng Y với AC, kết quả  ghi vào AC |
| 1000 0000 0001 0000   | STOR X   | Ghi AC vào X trong bộ nhớ                           |

Trong đó: X, Y là biến, có địa chỉ BN: X: 0000 0001 0000

Y: 0000 0001 0001

+

## c. Các loại lệnh: chia thành 4 nhóm

- ◼ Xử lý dữ liệu: các lệnh số học và logic
- ◼ Các lệnh số học cung cấp khả năng tính toán để xử lý dữ liệu số
- ◼ Các lệnh logic (Boolean) hoạt động trên các bit
ế , cung cấp khả năng xử lý bất kỳ loại g() g g p g 
dữ liệu nào. Các lệnh này chủ yếu thực thi với các bit trên thanh ghi
- ◼ Lưu trữ dữ liệu
- ◼ Các lệnh đọc/ghi dữ liệu từ/vào thanh ghi hoặc bộ nhớ
- ◼ Di chuyển dữ liệu:
- Gồm các lệnh vào/ra: được sử dụng để truyền chương trình và dữ liệu vào bộ nhớ và 
ế
- ◼ g y
các kết quả tính toán được trở lại cho người dùng

Ví dụ: chương trình user và dữ liệu (lưu trữ ở ổ cứng) được nạp vào RAM

- ◼ Điều khiển: gồm các lệnh kiểm tra và rẽ nhánh
- ◼ Các lệnh kiểm tra được sử dụng để kiểm tra giá trị của dữ liệu hoặc trạng thái của một phép toán
- ◼ Các lệnh rẽ nhánh được dùng để rẽ nhánh tập lệnh khác nhau tùy thuộc vào điều 
ể
- kiện cụ thể

+

## d. Số lượng các địa chỉ

- ◼ Một thuộc tính quan trọng của tập lệnh là số lượng địa chỉ
- ◼ Tùy thuộc vào các lệnh khác nhau sẽ có số lượng toán hạng khác nhau
- ◼ Như phần trên đã đề cập, các toán hạng có thể là các vị trí nhớ (trong bộ nhớ chính ) hoặc I/O (I/O port), được đặc trưng bởi một địa chỉ logic
- ◼ Vậy, số lượng địa chỉ tối đa trong một lệnh là bao nhiêu:
- ◼ Các lệnh số học và logic: cần tối đa 4 địa chỉ: 2 đ/c toán hạng nguồn, 1đ/c toán hạng đích, 1 đ/c toán hạng truy xuất câu lệnh tiếp theo.
- ◼ Số lượng địa chỉ càng nhiều thì kích thước lệnh càng lớn
- ◼ Với hầu hết các hệ VXL, số lượng địa chỉ là 1, 2 hoặc 3. Lệnh tiếp theo được ngầm định truy xuất thông qua thanh ghi PC (program counter register)

## + Số lượng các địa chỉ (tiếp)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000010_af95778ebae64e7f0f9835e0e1a073bd2fd63f799ba366c3442214cb50340741.png)

## Ví dụ các lệnh tính toán biểu thức 𝑌=(𝐴−𝐵)/(𝐶+(𝐷×𝐸)) trong 3 trường hợp:

- ◼ Trường hợp lệnh 3 địa chỉ:
- ◼ Kích thước lệnh dài
- ◼ Toán hạng 1, toán hạng 2, kết quả
- ◼ T: vị trí bộ nhớ tạm thời để lưu trữ kết quả
- ◼ Trường hợp lệnh 2 địa chỉ:
- ◼ Kết quả phép toán được ghi vào một địa chỉ

Ví dụ các lệnh tính toán biểu thức 𝑌 = (𝐴 − 𝐵)/(𝐶 + 𝐷 × 𝐸 ) trong 3 trường hợp

- Trường hợp lệnh 1 địa chỉ:
- Một toán hạng ngầm định là thanh ghi AC
- Phổ biến ở các hệ VXL đơn giản, đời đầu
- Trường hợp lệnh 0 địa chỉ:
- Ngầm định 2 ngăn nhớ ở đỉnh vùng ngăn xếp (stack) của BN
- Số lượng các địa chỉ càng ít thì số lượng các câu lệnh để tính toán biểu thức càng nhiều

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000011_49c78e065289796204d7ee2cb36c91c0cca2ef075a38c1e508e5b9fbb15d6a26.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000012_39e8dc1ab9f18dc04f5bc0275e1a75bafcdda8117bf4d278643795a82482815a.png)

## Bảng 10.1 Mô tả các lệnh 0, 1, 2, 3 địa chỉ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000013_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+ ◼ Số lượng địa chỉ trong mỗi lệnh là một yếu tố cơ bản đối với thiết kế VXL .
- ◼ Lệnh càng ít địa chỉ → kích thước lệnh ngắn hơn →VXL ít phức tạp hơn → chương trình cần nhiều lệnh hơn để thực hiện một công việc → mất thời gian hơn
- ◼ Lệnh một địa chỉ, lập trình viên thường chỉ có sẵn một thanh ghi đa năng: thanh ghi AC .
- ◼ Với các hệ VXL cho phép lệnh nhiều địa chỉ thường có nhiều thanh ghi đa năng. Điều này cho phép một số hoạt động được thực hiện chỉ trong các thanh ghi → không cần truy xuất BNC → tốc độ nhanh hơn.
- → Hầu hết các hệ VXL hiện đại sử dụng kết hợp các cấu trúc lệnh hai và ba địa chỉ.

## e. Thiết kế tập lệnh

- Tập lệnh định nghĩa các chức năng được thực hiện bởi VXL
- Là phương tiện của người lập trình trong việc điều khiển VXL
- Các vấn đề thiết kế cơ bản:
- o Danh sách các hoạt động: bao nhiêu hoạt động và hoạt động nào được đưa ra? Độ phức tạp của các hoạt động như thế nào?
- o Các kiểu dữ liệu: các kiểu dữ liệu mà các hoạt động tham chiếu đến
- o Cấu trúc lệnh: độ dài lệnh theo bit, số lượng địa chỉ, kích thước của các trường khác nhau, v.v ...
- o Các thanh ghi: số lượng các thanh ghi của VXL có thể được tham chiếu đến bởi lệnh và chức năng của chúng
- o Chế độ địa chỉ: các cách để định ra địa chỉ của toán hạng

## + Ví dụ

- ◼ Viết chương trình tính giá trị biểu thức sau sử dụng các tập lệnh 0, 1, 2, 3 địa chỉ cho ở bảng dưới

<!-- formula-not-decoded -->

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000014_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## 10.2 Các kiểu toán hạng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000015_8080df29d2438e160e9385c9d448cf9c1861f7a9eb20ed9bccf7a9d56d82df49.png)

+

## Dữ liệu kiểu số

- ◼ Tất cả các ngôn ngữ máy đều có dữ liệu dạng số
- ◼ Các số được lưu trữ trong máy tính đều hữu hạn:
- ◼ Hữu hạn về độ lớn của các số biểu diễn trên máy
- ◼ Hữu hạn về độ chính xác đối với số dấu phẩy động
- ◼ Ba kiểu dữ liệu số thông thường trong máy tính:
1. Số nguyên nhị phân hoặc số nhị phân dấu chấm tĩnh
2. Số nhị phân dấu chấm động
3. Số thập phân đóng
- ◼ Mỗi chữ số thập phân được biểu diễn bởi một mã 4 bit

<!-- formula-not-decoded -->

Dấu dương (+): 1100, dấu âm ( -

- ◼ Chiều dài mã thường là bội của 8b

<!-- formula-not-decoded -->

+

## Dữ liệu kiểu ký tự

- ◼ Một trong những dạng dữ liệu cơ bản là văn bản (text) hoặc xâu ký tự (character strings)
- ◼ Dữ liệu văn bản dưới dạng ký tự không thể lưu trữ hoặc truyền qua hệ thống xử lý dữ liệu và truyền thông vì các hệ thống này được thiết kế cho dữ liệu nhị phân → sử dụng bảng mã
- ◼ Bảng mã mã hóa ký tự được sử dụng phổ biến nhất là bảng mã IRA (International Reference Alphabet)
- ◼ Còn được gọi ở Mỹ là bảng mã ASCII (American Standard Code for Information Interchange)
- ◼ Một bảng mã khác được sử dụng để mã hoá các ký tự bảng mã EBCDIC (Extended Binary Coded Decimal Interchange Code) được sử dụng trong các máy mainframe của IBM

+

## Dữ liệu logic

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000016_915cfb37f866732d202107075870954c6eb0a9c8fedaa8b02a17956642a0cc36.png)

- ◼ Một khối n -bit gồm n phần tử dữ liệu 1 bit, mỗi item có giá trị 0 hoặc 1
- ◼ Hai ưu điểm của view theo hướng bit:
- ◼ Đôi khi, ta muốn lưu trữ một mảng các bit nhị phân hoặc dữ liệu Boolean/nhị phân, trong đó mỗi phần tử chỉ nhận giá trị 1 (đúng) hoặc 0 (sai). Với kiểu dữ liệu logic, bộ nhớ lưu trữ điều này hiệu quả nhất
- ◼ Trong một số trường hợp chúng ta cần thao tác với các bit
- ◼ Trường hợp phép toán dấu chấm động: dịch các bit có nghĩa
- ◼ Trường hợp chuyển đổi từ mã IRA thành mã thập phân đóng gói: trích xuất 4 bit bên phải của mỗi byte

+

## 10.3 Các kiểu dữ liệu Intel x86 và ARM

## a. Intel x86

| Kiểu dữ liệu                                     | Mô tả                                                                                                                                                                                                                      |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| General                                          | Các vị trí bộ nhớ kích thước byte, word (16 bits), doubleword (32 bits), quadword (64 bits), và double  quadword (128 bits) với dữ liệu nhị phân bất kỳ                                                                    |
| Integer                                          | Giá trị nhị phân có dấu lưu trữ trong một byte, word, hoặc doubleword, sử dụng dạng biểu diễn bù 2                                                                                                                         |
| Ordinal                                          | Một số nguyên không dấu lưu trữ trong một byte, word, hoặc doubleword.                                                                                                                                                     |
| Unpacked binary coded decimal  (BCD)             | Biểu diễn một ký tự BCD trong khoảng từ 0 đến 9, với mỗi ký tự dùng một byte                                                                                                                                               |
| Packed BCD                                       | Biểu diễn 2 ký tự BCD trong 1 byte, một packed BCD có dải giá trị từ 0 đến 99                                                                                                                                              |
| Near pointer                                     | Địa chỉ hiệu dụng 16-bit, 32-bit, hoặc 64-bit biểu diễn độ lệch (offset) trong một phân đoạn. Được sử  dụng cho tất cả các con trỏ trong bộ nhớ không phân đoạn và cho các tham chiếu trong một đoạn của  bộ nhớ phân đoạn |
| Far pointer                                      | Địa chỉ logic gồm 16 - bit trỏ tới một đoạn (segment) và một địa chỉ lệch offset 16, 32, hoặc 64 bits. Far  pointers được sử dụng để tham chiếu bộ nhớ trong mô hình bộ nhớ phân đoạn                                      |
| Bit field                                        | Một chuỗi bit liên tục trong đó mỗi bit được coi như một đơn vị độc lập. Chuỗi bit có thể bắt đầu tại bất  cứ vị trí nào trong bất cứ byte nào và có thể chứa tới 32 bit                                                   |
| Bit string                                       | Một dãy bit liên tục, gồm từ 0 đến 232  -  1 bit.                                                                                                                                                                          |
| Byte string                                      | Một dãy byte, word hoặc doublewords liên tục gồm từ 0 đến 232  -  1 byte.                                                                                                                                                  |
| Floating point                                   | Xem hình 10.4.                                                                                                                                                                                                             |
| Packed SIMD (single instruction,  multiple data) | Các kiểu dữ liệu Packed 64 - bit and 128 - bit                                                                                                                                                                             |

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000017_4792282bd8d80fa2c9ade9309a4d7b5fee17d0acb620a763af53a0e80a4fbfdb.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000018_552f8189385c5c248ab8cca42a55d9ab839b877ad849179d862deed37eec645c.png)

+

## Các kiểu dữ liệu SIMD (Single-Instruction-Multiple-Data)

- ◼ Dùng cho kiến trúc x86 như là 1 phần của mở rộng tập lệnh để tối ưu hóa hiệu năng của các ứng dụng đa phương tiện
- ◼ Các mở rộng tập lệnh gồm MMX (multimedia extensions) và SSE (streaming SIMD extensions)
- ◼ Các kiểu dữ liệu SMID:
- ◼ Byte đóng gói và số nguyên byte đóng gói
- ◼ Word đóng gói và số nguyên word đóng gói
- ◼ Doubleword đóng gói và số nguyên doubleword đóng gói
- ◼ Quadword đóng gói và số nguyên quadword đóng gói
- ◼ Packed single-precision floating-point and packed doubleprecision floating-point

+

## b. Các kiểu dữ liệu của ARM

- ◼ Vi xử lý ARM hỗ trợ các kiểu dữ liệu có kích thước:
- ◼ 8b (byte)
- ◼ 16b (halfword)
- ◼ 32b (word)
- ◼ Đối với tất cả ba kiểu dữ liệu, có một kiểu tương ứng 
ố ấố g 
dành cho số nguyên không dấu (số nguyên dương)
- ◼ Tất cả ba kiểu dữ liệu cũng có thể được sử dụng cho số 
ể nguyên biểu diễn bù 2
- ◼ VXL ARM không hỗ trợ phần cứng cho biểu diễn dấu 
ấấấ g pg 
chấm động. Các phép toán cho dấu chấm động phải 
ằầề gpp 
được thực hiện bằng phần mềm

## + Hỗ trợ chuyển đổi Endian trong ARM

- ◼ Khái niệm Endian: cách tổ chức dữ liệu trong bộ nhớ máy tính
- ◼ Có 2 loại endian:
- ◼ Little Endian: byte có giá trị nhỏ nhất lưu trữ ở vị trí nhớ có địa chỉ nhỏ nhất
- ◼ Big Endian: byte có giá trị lớn nhất lưu trữ ở vị trí nhớ có địa chỉ nhỏ nhất
- ◼ ARM cho phép chuyển đổi giữa hai dạng endian: sử dụng E -bit trong thanh ghi PS( thanh ghi trạng thái chương trình)
- ◼ E -bit = 1: Big endian
- ◼ E -bit = 0: Little endian

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000019_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000020_d2b0eac54658bae0472be5382772588ec488cdedb4ae05cb67e02a7048b0b403.png)

+

## 10.4 Các loại operation (hoạt động, phép toán)

- ◼ Có rất nhiều các loại lệnh khác nhau đối với mỗi thế hệ máy tính. Tuy nhiên, một số loại chung đối với tất cả các máy tính như sau:
- a. Các lệnh truyền dữ liệu
- b. Các lệnh tính toán số học
- c. Các lệnh logic
- d. Các lệnh chuyển đổi
- e. Các lệnh vào/ra
- f. Các lệnh điều khiển hệ thống
- g. Các lệnh truyền điều khiển

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000021_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## a. Truyền dữ liệu

- Đây là lệnh cơ bản nhất của tất cả các hệ máy tính
- Cần phải định rõ:
- o Địa chỉ của các toán hạng nguồn và đích, loại vị trí: bộ nhớ, thanh ghi, ngăn xếp
- o Kích thước của dữ liệu được truyền
- o Chế độ định địa chỉ đối với mỗi toán hạng
- Các lệnh cơ bản

| Truyền  dữ liệu   | MOVE     | Copy dữ liệu từ nguồn đến đích                    |
|-------------------|----------|---------------------------------------------------|
| Truyền  dữ liệu   | LOAD     | Nạp dữ liệu từ bộ nhớ đến bộ xử lý                |
| Truyền  dữ liệu   | STORE    | Cất dữ liệu từ bộ xử lý đến bộ nhớ                |
| Truyền  dữ liệu   | EXCHANGE | Trao đổi nội dung của nguồn và đích               |
| Truyền  dữ liệu   | CLEAR    | Chuyển các bit 0 vào toán hạng đích               |
| Truyền  dữ liệu   | SET      | Chuyển các bit 1 vào toán hạng đích               |
| Truyền  dữ liệu   | PUSH     | Cất nội dung toán hạng nguồn vào ngăn xếp         |
| Truyền  dữ liệu   | POP      | Lấy nội dung đỉnh ngăn xếp đưa đến toán hạng đích |

## +

## b . Số học

- ◼ Hầu hết các máy đều cung cấp các phép toán số học cơ bản: cộng, trừ, nhân, chia
- ◼ Cố định với toán hạng là số nguyên có dấu. Một số máy có các phép toán trên với:
- ◼ Số thực dấu chấm động
- ◼ Số thập phân đóng
- ◼ Một số phép toán khác có dạng lệnh một toán hạng:
- ◼ Tuyệt đối: Tính giá trị tuyệt đối của một toán hạng
- ◼ Phép đảo: Đổi dấu một toán hạng
- ◼ Phép tăng: Cộng toán hạng thêm 1 đơn vị
- ◼ Phép giảm: Trừ toán hạng đi 1 đơn vị

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000022_16c05e2b94a0795e8d337e0ee025a1004579b8fe25e714a7064c1beff752a82a.png)

## + Một số lệnh số học

| Xử lý  số học   | ADD       | Cộng hai toán hạng           |
|-----------------|-----------|------------------------------|
| Xử lý  số học   | SUBTRACT  | Trừ hai toán hạng            |
| Xử lý  số học   | MULTIPLY  | Nhân hai toán hạng           |
| Xử lý  số học   | DIVIDE    | Chia hai toán hạng           |
| Xử lý  số học   | ABSOLUTE  | Lấy trị tuyệt đối toán hạng  |
| Xử lý  số học   | NEGATE    | Đổi dấu toán hạng (lấy bù 2) |
| Xử lý  số học   | INCREMENT | Tăng toán hạng thêm 1        |
| Xử lý  số học   | DECREMENT | Giảm toán hạng đi 1          |

## c . Phép toán logic (luận lý)

- ◼Các phép toán logic cơ bản

| Xử lý logic   | AND     | Thực hiện phép AND hai toán hạng                                                      |
|---------------|---------|---------------------------------------------------------------------------------------|
| Xử lý logic   | OR      | Thực hiện phép OR hai toán hạng                                                       |
| Xử lý logic   | NOT     | Đảo bit của toán hạng (lấy bù 1)                                                      |
| Xử lý logic   | XOR     | Thực hiện phép XOR hai toán hạng                                                      |
| Xử lý logic   | TEST    | Kiểm tra điều kiện cụ thể; thiết lập cờ dựa trên kết quả                              |
| Xử lý logic   | COMPARE | So sánh logic hoặc số học của hai hoặc nhiều toán hạng; thiết lập cờ dựa trên kết quả |
| Xử lý logic   | SHIFT   | Dịch trái (phải) toán hạng                                                            |
| Xử lý logic   | ROTATE  | Quay vòng trái (phải) toán hạng                                                       |

+

- ◼ Cho phép thực hiện với các khối n -bit

- ◼ Ngoài ra, gồm có một số phép toán dịch và xoay vòng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000023_7ecfe3fc8701bb66487663065a6d8810f6610e1374ce03efdcfe38b072eb177f.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000024_6362fca6cc9e37e53262a65153619782c907ba3fe3ca8c078776c6a7b067f13c.png)

## Ví dụ về các phép toán dịch và xoay vòng

+

## d. Chuyển đổi

- ◼ Lệnh chuyển đổi: thay đổi định dạng hoặc tác động vào định dạng của dữ liệu
- ◼ Ví dụ 1: chuyển đổi từ nhị phân sang mã thập phân đóng
- ◼ Ví dụ 2: chuyển đổi từ mã IRA sang mã EBCDIC qua một bảng gồm 256 byte trong bộ nhớ chính.

## +e. Vào/ra

- ◼ Các lệnh vào/ra liên quan đến:
- ◼ Cơ chế địa chỉ:
- ◼ I/O chương trình, ánh xạ riêng biệt - Isolated programmed I/O
- ◼ I/O chương trình, ánh xạ bộ nhớ - Memory-mapped programmed I/O
- ◼ Cơ chế DMA
- ◼ Cơ chế điều khiển I/O sử dụng bộ xử lý vào ra
- ◼ Nhiều tập lệnh của các hệ VXL chỉ cung cấp một vài lệnh I/O với các hoạt động cụ thể được xác định bởi các tham số, các mã hoặc các từ lệnh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000025_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000026_af36877dfdccd78f5c4267946ae0d8d442933ab6d38b30d7f12663f9ede693b2.png)

| Điều khiển vào/ra   | INPUT     | Truyền dữ liệu từ một cổng hoặc thiết bị I/O xác định đến đích (VD: Bộ nhớ chính hoặc thanh ghi bộ xử lý)   |
|---------------------|-----------|-------------------------------------------------------------------------------------------------------------|
| Điều khiển vào/ra   | OUTPUT    | Truyền dữ liệu từ nguồn xác định đến cổng hoặc thiết bị I/O                                                 |
| Điều khiển vào/ra   | START I/O | Truyền lệnh đến bộ xử lý I/O để bắt đầu hoạt động I/O                                                       |
| Điều khiển vào/ra   | TEST I/O  | Truyền thông tin trạng thái từ hệ thống I/O đến đích xác định                                               |

## f. Điều khiển hệ thống

- Các câu lệnh có thể được thực hiện chỉ khi VXL trong trạng thái đặc quyền hoặc đang thực hiện một chương trình trong vùng đặc quyền đặc biệt của bộ nhớ.
- Thông thường các lệnh này được dành riêng cho hệ điều hành
- Ví dụ về các hoạt động điều khiển hệ thống:
- Một câu lệnh điều khiển hệ thống có thể đọc hoặc thay đổi thanh ghi điều khiển
- Câu lệnh để đọc hoặc thay đổi khóa bảo vệ bộ nhớ
- Truy cập vào các khối điều khiển tiến trình trong hệ thống đa chương trình

## + g. Truyền điều khiển

- ◼ Các hoạt động truyền điều khiển là cần thiết:
- ◼ Cần thiết để có thể thực thi mỗi câu lệnh nhiều hơn một lần
- ◼ Hầu như mọi chương trình đều gồm có việc ra quyết định
- ◼ Cơ chế phân tách các nhiệm vụ ra thành các công việc nhỏ hơn có thể thực hiện tại các thời điểm khác nhau
- ◼ Các hoạt động truyền điều khiển nói chung:
- ◼ Rẽ nhánh:
- ◼ Rẽ nhánh có điều kiện: chỉ rẽ nhánh (thiết lập địa chỉ (toán hạng của lệnh) vào thanh ghi PC) khi một điều kiện nhất định được thỏa mãn
- ◼ Rẽ nhánh không điều kiện: việc rẽ nhánh luôn được thực hiện.
- ◼ Lệnh nhảy
- ◼ Lệnh gọi thủ tục

## + Truyền điều khiển

- ◼ Các lệnh nhảy và gọi thủ tục

| Truyền  điều khiển   |                  | JUMP (BRANCH) Truyền không điều kiện; tải địa chỉ xác định vào PC                                                   |
|----------------------|------------------|---------------------------------------------------------------------------------------------------------------------|
| Truyền  điều khiển   | JUMP CONDITIONAL | Kiểm tra điều kiện cụ thể; tải địa chỉ xác định vào PC hoặc không làm gì tùy thuộc vào điều kiện                    |
| Truyền  điều khiển   | CALL             | Cất nội dung của PC (địa chỉ trở về) vào một vị trí xác định; Nạp địa chỉ lệnh đầu tiên của chương trình con vào PC |
| Truyền  điều khiển   | RETURN           | Đặt địa chỉ trở về trả lại cho PC để trở về chương trình chính                                                      |
| Truyền  điều khiển   | SKIP             | Tăng PC để bỏ qua lệnh tiếp theo                                                                                    |
| Truyền  điều khiển   | SKIP CONDITIONAL | Kiểm tra điều kiện cụ thể; bỏ qua lệnh hoặc không làm gì tùy thuộc vào điều kiện                                    |
| Truyền  điều khiển   | HALT             | Dừng thực thi chương trình                                                                                          |
| Truyền  điều khiển   | WAIT (HOLD)      | Dừng thực thi chương trình; liên tục kiểm tra điều kiện cụ thể; quay lại thực thi tiếp khi điều kiện được thỏa mãn  |
| Truyền  điều khiển   | NO OPERATION     | Không có hành động nào được thực hiện, nhưng việc                                                                   |

thực thi chương trình vẫn được tiếp tục

## + Ví dụ một số lệnh rẽ nhánh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000027_a230ed6960d3592eff7cf22c515f048e694aa625d928fde681da8d0aab0baaa0.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000028_c2ff3bbf035a4817b7c1bbb25c792be58c1a130ad6abb02a3469f5c8b14e0a2a.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000029_75482835919980611227e6f3e054ad54a58a947bfd78b1828770534ff22ca56d.png)

+

## Hành động của VXL đối với các loại hoạt động

- ◼ Truyền dữ liệu:
- ◼ Truyền dữ liệu từ một vị trí đến vị trí khác
- ◼ Nếu gồm bộ nhớ:
- ◼ Xác định địa chỉ bộ nhớ
- ◼ Thực hiện việc chuyển địa chỉ ảo sang đ/c thực tế
- ◼ Kiểm tra cache
- ◼ Bắt đầu hoạt động đọc/ghi
- ◼ Tính toán số học
- ◼ Có thể bao gồm cả hoạt động truyền dữ liệu (trước hoặc sau khi tính toán)
- ◼ Các phép toán được thực hiện trong ALU
- ◼ Thiết lập các mã điều kiện và các cờ

+

## Hành động của VXL đối với các loại hoạt động (tiếp)

- ◼ Tính toán logic: giống tính toán số học
- ◼ Chuyển đổi: tương tự như tính toán số học và logic. Có thể gồm một logic đặc biệt để thực hiện chuyển đổi
- ◼ Truyền điều khiển:
- ◼ Cập nhật thanh ghi PC. Với lời gọi/trả về chương trình con, quản lý các thông số đi qua hoặc liên kết
- ◼ Vào/ra
- ◼ Đưa ra các lệnh cho module I/O
- ◼ Trong chế đó memory -mapped I/O, xác định địa chỉ I/O

## + 10.5 Các loại hoạt động Intel x86 và

## ARM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000030_6362fca6cc9e37e53262a65153619782c907ba3fe3ca8c078776c6a7b067f13c.png)

| +   |
|-----|

## Bảng12.8

Các loại hoạt động x86 (Với các ví dụ của các hoạt động thông thường)

(page 1 of 2)

## Bảng12.8

Các loại hoạt

động x86

(Với các ví dụ của các hoạt động

thông thường)

(page 2 of 2)

## Các cờ trạng thái x86

## Bảng 12.10

Các mã điều kiện x86 cho lệnh SETcc và lệnh nhảy có điều kiện

## + Ví dụ

1. Với mỗi số được mã hóa thập phân đóng như dưới đây, xác định giá trị số thập phân tương ứng:
- a. 0111 0011 0000 1001
- b. 0101 1000 0010
- c. 0100 1010 0110
2. VXL có kích thước từ nhớ là 1 byte. Giá trị số nguyên lớn nhất và nhỏ nhất biểu diễn theo các dạng dưới đây là bao nhiêu
- a. Số nguyên không dấu
- b. Số nguyên dạng dấu – – độ lớn
- c. Số bù hai
- d. Số thập phân đóng không dấu
- e. Số thập phân đóng có dấu
3. Thực hiện dịch trái logic, dịch phải logic, dịch trái số học, dịch phải số học , vòng trái, vòng phải 4b với word 16b sau: 1001 1101 1100 0001

## +Các lệnh Đơn-lệnh, nhiều -dữ liệu x86 (Single Instruction – Multiple Data -SIMD Instructions)

- ◼ Năm 1996 Intel giới thiệu công nghệ MMX cho dòng VXL Pentium
- ◼ MMX là một tập các lệnh được tối ưu hóa cao cho các chức năng đa phương tiện
- ◼ Các dữ liệu video và audio thường bao gồm các mảng lớn các kiểu dữ liệu nhỏ
- ◼ Ba kiểu dữ liệu mới được định nghĩa trong MMX
- ◼ Packed Byte
- ◼ Packed word
- ◼ Packed doubleword
- ◼ Mỗi kiểu dữ liệu có kích thước 64 bit và gồm có nhiều trường dữ liệu 
ố ấẩ gg 
nhỏ hơn, các trường này chứa dữ liệu số nguyên dấu phẩy tĩnh

Note: If an instruction supports multiple data types [byte (B), word (W), doubleword (D), quadword (Q)], the data types are indicated in brackets.

## Tập lệnh MMX

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000031_67e2f32352ac3ed389417a116229cdceea8bdf976fe318eb04584e631ff20ba1.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH10-Instruction set_artifacts/image_000032_a28da672d98b2f1e656ee72cf212ac15c6beca4b489282d3137a2dd9b3694309.png)

## Các loại câu lệnh ARM

Các lệnh tải và lưu trữ dữ liệu

Các lệnh nhân

Các lệnh rẽ nhánh

Các lệnh cộng trừ song song

Các lệnh truy cập vào thanh ghi trạng thái

Các lệnh xử lý dữ liệu

Các lệnh mở rộng

## Các điều kiện ARM cho thực thi câu lệnh điều kiện

+

## Tổng kết

## Chương 10

- ◼ Đặc điểm lệnh máy
- ◼ Các thành phần của lệnh máy
- ◼ Biểu diễn lệnh
- ◼ Các loại lệnh
- ◼ Number of addresses
- ◼ Thiết kế tập lệnh
- ◼ Các loại toán hạng
- ◼ Số
- ◼ Ký tự
- ◼ Dữ liệu logic

## Tập lệnh: Đặc điểm và chức năng

- ◼ Các kiểu dữ liệu Intel x86 và ARM
- ◼ Các loại hoạt động
- ◼ Truyền dữ liệu
- ◼ Tính toán số học
- ◼ Logical
- ◼ Chuyển đổi
- ◼ Vào/ra
- ◼ Điều khiển hệ thống
- ◼ Truyền điều khiển
- ◼ Các loại hoạt động trong Intel x86 và ARM

+

## Câu hỏi chương 10

1 Các thành phần điển hình của một lệnh máy?

- 2 Toán hạng nguồn và đích có thể được đặt ở đâu?
- 3 Nếu một lệnh có 4 địa chỉ, mục đích của từng địa chỉ là gì?
- 4 Trình bày ngắn gọn 5 vấn đề thiết kế tập lệnh quan trọng.
- 5 Các loại toán hạng điển hình trong tập lệnh máy?
- 6 Mối quan hệ giữa mã ký tự IRA và biểu diễn packed decimal?
- 7 Phân biệt dịch số học và dịch logic?
- 8 Tại sao phải truyền lệnh điều khiển?
- 9 Trình bày hai cách tạo điều kiện để kiểm tra một lệnh rẽ nhánh
- 10 Ý nghĩa của thuật ngữ làm thủ tục lồng nhau là gì?
- 11 Liệt kê ba vị trí có thể lưu trữ địa chỉ trả về của thủ tục có trả về.
- 12 Phân biệt endian lớn và endian nhỏ?