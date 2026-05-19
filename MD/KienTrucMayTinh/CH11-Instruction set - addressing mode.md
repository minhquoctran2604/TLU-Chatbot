![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000000_e57e83a910a6ecf7dc43ce6fd8e498af4b4666bd74aeabf1e74cda8342d76b07.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000001_61769a713dc9c7fb5025f6b99d89ae4db3694e64ea380fc4bc9189b83832e3ba.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000002_a47b8bc6c15fd76c66f23999fd37339d852c3c2a425a320b35d5c39fc9b4c654.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000003_db882e604fe1bc9d437a0b487a9d4a247ada3bd804c5caf2ae8e4077e39d92b5.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000004_059e364c396dc1212b5e9f28c9c15210b03005d683356feefd5f3dfba6c74ebc.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000005_93c9dfa7ddbb75f7ec2892d9f5831d85346132f2abd4890feb031ed21541981a.png)

Kiến trúc máy tính

+

## Chương 11

Tập lệnh: Các chế độ định địa chỉ và định dạng lệnh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000006_b8eb0923bd17d7163e0d6bec6d87c16bc580eedd820ebf12c6c183e0a64dc0b6.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000007_2e61313693f0e0669d1aabbebe3543210d03fe54f78e365b722fb8a3a1d0760a.png)

## Chương 11. Chế độ định địa chỉ và định dạng lệnh

- ◼ 11.1 Các chế độ địa chỉ
- ◼ 11.2 Các chế độ địa chỉ của x86 và ARM
- ◼ 11.3 Định dạng lệnh
- ◼ 11.4 Định dạng lệnh của x86 và ARM
- ◼ 11.5 Hợp ngữ (Assembly Language)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000008_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## + 11.1 Các chế độ định địa chỉ

- a. Tức thì
- b. Trực tiếp
- c. Gián tiếp
- d. Thanh ghi
- e. Gián tiếp thanh ghi
- f. Dịch chuyển
- g. Ngăn xếp

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000009_1023a40a7eac88842c8fc8b773e13c9026fa545fd0cf4ccb482e71da269bedf6.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000010_b76c3afd2fb45c3d01b9e3fda4ab69bed346f20a8427dbc974248afa32f52c11.png)

+

## a. Định địa chỉ tức thì

- ◼ Dạng đơn giản nhất của định địa chỉ
- ◼ Toán hạng = A
- ◼ Chế độ này có thể được sử dụng để định nghĩa và sử dụng các hằng số và thiết lập các giá trị ban đầu của biến
- ◼ Các số thường được lưu trữ dưới dạng số bù hai
- ◼ Bit ngoài cùng bên trái của trường toán hạng được sử dụng như bit dấu
- ◼ Chỉ cần truy xuất bộ nhớ một lần (để lấy lệnh), do vậy tiết kiệm một chu kỳ cache hoặc bộ nhớ trong chu kỳ lệnh .
- ◼ Nhược điểm:
- ◼ Kích thước của số bị giới hạn bởi kích thước của trường địa chỉ vì thông thường kích thước của trường này nhỏ hơn kích thước từ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000011_0ae0d6a5e608e244b5863d153daaff9c7cbe2ad555e9319cca16f6f95722cb92.png)

- ◼ Ưu điểm: ADD 3: acc+3 -- &gt; acc

Ví dụ:

<!-- formula-not-decoded -->

Trong đó, 5 là một toán hạng nguồn được tham chiếu trực tiếp trong câu lệnh: địa chỉ tức thì

+

## b. Định địa chỉ trực tiếp

- ◼ Trường địa chỉ chứa địa chỉ hiệu dụng của toán hạng
- ◼ Địa chỉ hiệu dụng (Effective address (EA)) = trường địa chỉ (Address field (A)) 0011 0001 1010 0000: EA
- ◼ Phổ biến trong các thế hệ máy tính trước đây
- ◼ Tham chiếu bộ nhớ một lần để lấy dữ liệu toán hạng
- ◼ Hạn chế: chỉ cung cấp một không gian địa chỉ hạn chế

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000012_8df4ff5da6f626b38773ed923a74499b276b3b6c1c26ed2c0451a30b8ed88448.png)

Ví dụ:

<!-- formula-not-decoded -->

Trong đó, A là địa chỉ một vị trí bộ nhớ, (A) là nội dung của vị trí đó. CPU tham chiếu đến toán hạng có địa chỉ A trong bộ nhớ .

+

## c. Định địa chỉ gián tiếp

- ◼ Tham chiếu đến địa chỉ của một từ trong bộ nhớ chứa địa chỉ đầy đủ của toán hạng
- ◼ EA = (A)
- ◼ Dấu ngoặc đơn được hiểu như là nội dung của
- ◼ Ưu điểm:
- ◼ Với một từ có kích thước N cho phép một không gian địa chỉ là 2 N
- ◼ Nhược điểm:
- ◼ Thực thi câu lệnh đòi hỏi hai lần tham chiếu bộ nhớ để truy xuất toán hạng
- ◼ Một để lấy ra địa chỉ, hai là để lấy ra giá trị của nó
- ◼ Một biến thể hiếm gặp của địa chỉ gián tiếp là địa chỉ gián tiếp nhiều cấp hoặc nhiều tầng
- ◼ EA = ( . . . (A) . . . )
- ◼ Nhược điểm là cần ba hoặc nhiều hơn tham chiếu bộ nhớ để truy xuất toán hạng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000013_725810484e364ffd69c7f191305dd02b4fe58649f0ecd0e9c5a90cc1c8bad144.png)

+

## d. Định địa chỉ thanh ghi

- ◼ Trường địa chỉ dùng để tham chiếu thanh ghi chứ không phải địa chỉ bộ nhớ chính
- ◼ EA = R
- ◼ Ưu điểm:
- ◼ Chỉ cần một trường địa chỉ nhỏ trong lệnh (do số lượng thanh ghi ít)
- ◼ Không cần tham chiếu bộ nhớ (tốn nhiều thời gian)
- ◼ Nhược điểm:
- ◼ Không gian địa chỉ giới hạn

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000014_2fff11098f8996a78fafb02b7a76cb4f2a7dcce2f8fd0810cc2ec6eb691f2b62.png)

Ví dụ:

<!-- formula-not-decoded -->

Trong đó, R1 là một thanh ghi trong bộ xử lý, câu lệnh trên tham chiếu giá trị (A) trong bộ nhớ cộng với giá trị lưu trữ trong thanh ghi R1, kết quả được ghi vào R1.

## ❖Ví dụ:

- ❖MOV BX, DX ; Copy nội dung DX vào BX
- ❖MOV AL, BL ; Copy nội dung BL vào AL
- ❖MOV AL, BX ; không hợp lệ vì các thanh ghi có kích thước khác nhau
- ❖MOV ES, DS ; không hợp lệ (segment to segment)
- ❖MOV CS, AX ; không hợp lệ vì CS không được dùng làm thanh ghi đích
- ❖ADD AL, DL ; Cộng nội dung AL

+

## e. Định địa chỉ gián tiếp thanh ghi

- ◼ Tương tự như địa chỉ gián tiếp
- ◼ Sự khác biệt duy nhất là trường địa chỉ tham chiếu đến thanh ghi
- ◼ EA = (R)
- ◼ Không gian địa chỉ lớn hơn (trường địa chỉ tham chiếu đến vị trí chứa địa chỉ có độ dài bằng một từ )
- ◼ Tham chiếu bộ nhớ ít hơn định địa chỉ gián tiếp
- ❖Ví dụ:
- ❖MOV AL, [BX] ; Copy nội dung ô nhớ có địa chỉ DS:BX vào AL
- ❖MOV [ SI ], CL ; Copy nội dung của CL vào ô nhớ có địa chỉ DS:SI
- ❖MOV [ DI ], AX ; copy nội dung của AX vào 2 ô nhớ liên tiếp DS: DI và DS: (DI +1)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000015_c11bb5efb8b4813081161c8b3bea3a3ad6466bf1261b0e3951533a6980f98f97.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000016_c96d35ca42ddbe5550839a75afa746ccf61db79324790c71e916da24846e9ae4.png)

+

## f. Định địa chỉ dịch chuyển -Displacement Addressing

- ◼ Kết hợp chế độ định địa chỉ trực tiếp và định địa chỉ trực tiếp thanh ghi
- ◼ EA = A + (R)
- ◼ Yêu cầu lệnh phải có hai trường địa chỉ, ít nhất một trong hai phải 
ể ệp
có giá trị cụ thể
- ◼ Một giá trị trong một trường địa chỉ (giá trị = A) được sử dụng trực 
ế tiếp
- ◼ Một trường địa chỉ khác tham chiếu đến thanh ghi trong đó nội dung 
ể ộg ị
được cộng với A để tạo ra địa chỉ hiệu dụng
- ◼ Hầu hết sử dụng:
- ◼ Định địa chỉ tương đối
- ◼ Định địa chỉ thanh ghi cơ sở
- ◼ Định địa chỉ chỉ mục
- ❖MOV AL, [BP]+5 ; copy nội dung của ô nhớ SS:BP+5 vào thanh ghi AL

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000017_6536af971e31a2804328ce878ee23f838823c385029820b259b518fc999d8864.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000018_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000019_8bf8181b68728491c57daa43478b8007304510dde6f829370472926782f720ad.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000020_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## Định địa chỉ tương đối

- ◼ Thanh ghi được tham chiếu ngầm là thanh ghi PC (program counter)
- ◼ Địa chỉ lệnh tiếp theo được cộng vào trường địa chỉ để tạo ra EA
- ◼ Thông thường trường địa chỉ được coi là dữ liệu dạng số bù 2 của hoạt động này
- ◼ Do đó, địa chỉ hiệu dụng là quan hệ dịch chuyển so với địa chỉ của lệnh
- ◼ Khai thác tính cục bộ của bộ nhớ
- ◼ Lưu các bit địa chỉ trong lệnh nếu hầu hết tham chiếu bộ nhớ tương đối gần lệnh đang được thực thi

+

## Định địa chỉ thanh ghi cơ sở

- ◼ Thanh ghi được tham chiếu chứa một địa chỉ bộ nhớ chính và trường địa chỉ chứa một giá trị dịch chuyển so với địa chỉ này
- ◼ Tham chiếu thanh ghi có thể rõ ràng hoặc ngầm định
- ◼ Khai thác tính cục bộ của tham chiếu bộ nhớ
- ◼ Phương tiện hữu ích để thực hiện phân đoạn
- ◼ Trong một số trường hợp , một thanh ghi cơ sở duy nhất được sử dụng và được ngầm định
- ◼ Các trường hợp khác, người lập trình có thể chọn một thanh ghi để lưu địa chỉ cơ sở của một đoạn và lệnh phải tham chiếu nó một cách rõ ràng

+

## Định địa chỉ chỉ số -Indexed Addressing

- ◼ Trường địa chỉ tham chiếu địa chỉ bộ nhớ chính và thanh ghi được tham chiêu 
ể g ịịộ 
chứa giá trị dịch chuyển dương từ địa chỉ này
- ◼ Phương pháp tính toán EA giống như với định địa chỉ thanh ghi cơ sở
- ◼ Một ứng dụng quan trọng: cung cấp một cơ chế hiệu quả để thực hiện các hoạt động lặp
- ◼ Autoindexing
- ◼ Tự động tăng hoặc giảm thanh ghi chỉ mục sau mỗi tham chiếu đến nó
- ◼ EA = A + (R)
- ◼ (R) (R) + 1
- ◼ Postindexing
- ◼ Indexing is performed after the indirection
- ◼ EA = (A) + (R)
- ◼ Preindexing
- ◼ Indexing is performed before the indirection
- ◼ EA = (A + (R))

+

## Định địa chỉ ngăn xếp

- ◼ Một ngăn xếp là một mảng liên tiếp các ô nhớ: danh sách dạng vào trước ra sau
- ◼ Lệnh có chế độ địa chỉ này là các lệnh thực hiện trực tiếp với đỉnh ngăn xếp → sử dụng thanh ghi ngầm định là thanh ghi SP
- ◼ Thanh ghi SP (stack pointer – con trỏ ngăn xếp) chứa địa chỉ đỉnh ngăn xếp (trỏ vào đỉnh ngăn xếp)
- ◼ Do đó, chế độ địa chỉ ngăn xếp thực chất là chế độ địa chỉ gián tiếp thanh ghi
- ◼ Các lệnh máy không cần tham chiếu bộ nhớ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000021_ec338ea18143cc9de9a0d975124501af13ca02c3fe4c82b1ad4411bc8cc67c16.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000022_200794f29767e8481e356bc8810b6c22eae37ab675bda714a1dcac2249469a7f.png)

+

## Các chế độ định địa chỉ cơ bản

| Chế độ               | Thuật toán         | Ưu điểm                      | Nhược điểm                        |
|----------------------|--------------------|------------------------------|-----------------------------------|
| Tức thì              | Toán hạng = A      | Không cần tham chiếu  bộ nhớ | Hạn chế về giá trị của toán  hạng |
| Trực tiếp            | EA = A             | Đơn giản                     | Không gian địa chỉ hạn chế        |
| Gián tiếp            | EA = (A)           | Không gian địa chỉ lớn       | Tham chiếu bộ nhớ nhiều  lần      |
| Thanh ghi            | EA = R             | Không cần tham chiếu  bộ nhớ | Không gian địa chỉ hạn chế        |
| Gián tiếp  thanh ghi | EA = A + (R)       | Linh hoạt                    | Phức tạp                          |
| Ngăn xếp             | EA = đỉnh ngăn xếp | Không cần tham chiếu  bộ nhớ | Khả năng ứng dụng ít              |

## +

## Ví dụ 1

Câu lệnh LOAD: nạp dữ liệu vào thanh ghi ngầm định AC. Xác định giá trị của AC trong các trường hợp sau. Tính số lần truy xuất bộ nhớ của các lệnh trên

- ◼ Trong đó:

- ◼ 1: chế độ tức thì

- ◼ 2: chế độ trực tiếp

- ◼ 3: chế độ gián tiếp c. LOAD 3 20 vào M[20]=40

- a. LOAD 1 20 AC=20

- b. LOAD 2 20 AC=40

- d. 40: địa chỉ toán hạng --&gt; Ac=60
- e. LOAD 1 30
- f. LOAD 2 30
- g. LOAD 3 30

| Nội dung các ngăn nhớ   | Nội dung các ngăn nhớ   |
|-------------------------|-------------------------|
| Địa chỉ                 | Dữ liệu                 |
| 20                      | 40                      |
| 30                      | 50                      |
| 40                      | 60                      |
| 50                      | 70                      |

## +

## Ví dụ 2

Trường địa chỉ của một câu lệnh là 14 (hệ thập phân). Toán hạng của lệnh nằm ở đâu trong các trường hợp sau

- a) Chế độ địa chỉ tức thì
- b) Chế độ địa chỉ trực tiếp
- c) Chế độ địa chỉ gián tiếp
- d) Chế độ địa chỉ thanh ghi
- e) Chế độ địa chỉ gián tiếp thanh ghi

## 11.3 Định dạng lệnh

- Định nghĩa cách bố trí các trường bên trong lệnh
- Gồm có các thông tin sau cần được thể hiện:
- Opcode: mã lệnh
- Chế độ địa chỉ: ngầm định hoặc cần được chỉ rõ, cho biết chế độ định địa chỉ cho mỗi toán hạng
- Địa chỉ các toán hạng: ngầm định hoặc được chỉ rõ
- Với mỗi tập lệnh, có thể có nhiều định dạng được sử dụng
- Ví dụ: máy giả thuyết CPUSIM

| read       | 0011 0000 0000 0000   |
|------------|-----------------------|
| jmpn  Done | 1011 0000 0000 1010   |

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000023_16dcc93f398c8fccd9f5daf16e7811a8576c5226044cfaa7d9873656eb4b8655.png)

+

## Kích thước từ lệnh

- ◼ Phụ thuộc vào:
- ◼ Kích thước bộ nhớ: vd: cho phép quản lý BN dung lượng bao nhiêu → số bit trường địa chỉ
- ◼ Tổ chức bộ nhớ: kích thước ngăn nhớ, kích thước từ
- ◼ Cấu trúc bus: số đường địa chỉ, số đường dữ liệu → đơn vị truyền
- ◼ Sự phức tạp của bộ vi xử lý: lệnh càng nhiều địa chỉ (kích thước từ lệnh càng lớn) → VXL càng phức tạp
- ◼ Tốc độ bộ vi xử lý
- ◼ Kích thước từ lệnh thường bằng hoặc là bội của đơn vị truyền
- ◼ VD: bus dữ liệu có kích thước 16b, kích thước từ lệnh có thể là 16b hoặc 32b tùy kiến trúc

+

## Phân bổ bit

- ◼ Việc phân bổ các bit (phân chia các trường) trong một từ lệnh phụ thuộc vào các yếu tố sau:
- ◼ Số lượng của các chế độ định địa chỉ: một lệnh có thể có một hoặc nhiều chế độ địa chỉ , các chế độ này có thể ngầm định hoặc
- ◼ Số lượng các toán hạng
- ◼ Thanh ghi so sánh với bộ nhớ
- ◼ Số lượng tập thanh ghi
- ◼ Dải địa chỉ
- ◼ Địa chỉ chi tiết

Số bit của trường mã lệnh (opcode) quyết định số lệnh tối đa của tập lệnh (quyết định số hoạt động tối đa mà vxl có thể thực hiện được)

CPUSIM – 16b: 4b mã lệnh + 12b địa chỉ

- --&gt; 4b mã lệnh: tối đa 16 lệnh = 2^4

## Định dạng lệnh PDP-8

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000024_d98d91f721ec479dab066c3b876b4a322083e20da14af942922c617fe6a0856a.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000025_39e8dc1ab9f18dc04f5bc0275e1a75bafcdda8117bf4d278643795a82482815a.png)

## Định dạng lệnh PDP-10

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000026_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## Các lệnh chiều dài thay đổi

- ◼ Sự thay đổi có thể được cung cấp hiệu quả và gọn nhẹ
- ◼ Tăng sự phức tạp của VXL
- ◼ Không loại bỏ mong muốn làm cho tất cả các chiều dài lệnh tương đồng với chiều dài từ
- ◼ Vì VXL không biết chiều dài của lệnh tiếp theo sẽ được truy xuất. Một chiến lược điển hình là truy xuất một số byte hoặc word bằng ít nhất lệnh dài nhất có thể
- ◼ Đôi khi nhiều lệnh được truy xuất

## Định dạng lệnh PDP-11

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000027_9cc37628b59d89ae7b4af16d38233aa949fa2bf583c8aacd7db03350a6716157.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000028_1e9835d1d1ef4727e1a941f1ae8bbabdd9e8db9c6fd7bbb06272d762e6e3961b.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000029_5c3f5c28762663ad0b6ec549b0d638fe9b8f85a40c87eeb391881845f3d75726.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000030_352473587890a482c95fe575b8b08f4bdbe27d840264aa9e7e89e29ff925e811.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000031_c97fb150f49326d596bb3fa3dbeece4bc206e5ae468f07b29a39a87ffd46beb9.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000032_2ad74cb68d4bdfa941f256b1dad8f6bb4192bc345a9990f08f609f4fef28b829.png)

## Định dạng lệnh x86

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000033_cdceea5405539f360c89f43c3f5235a552028877b207fc27f05b280d77a74aa2.png)

## Các định dạng lệnh ARM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000034_d7224713470e8b7c19d1a5417ab6124996a426ebc9a615894fa90b967c7d9e9b.png)

## Ví dụ về sử dụng ARM Immediate Constants (hằng số tức thì)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000035_34543c3c46c303fb13ffce878ce0b44e7cdefddc608c96fac59847f7fa133ec4.png)

## Tập lệnh Thumb

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000036_ac8dc17e93a3dddfe88e556014d72b6a3eba5ff3383f83b64f3051414f92e82c.png)

## Hợp ngữ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH11-Instruction set - addressing mode_artifacts/image_000037_39e8dc1ab9f18dc04f5bc0275e1a75bafcdda8117bf4d278643795a82482815a.png)

## Tổng kết

## Chapter 11

- ◼ Các chế độ định địa chỉ
- ◼ Định địa chỉ tức thì
- ◼ Định địa chỉ trực tiếp
- ◼ Định địa chỉ gián tiếp
- ◼ Định địa chỉ thanh ghi
- ◼ Định địa chỉ gián tiếp thanh ghi
- ◼ Định địa chỉ dịch chuyển
- ◼ Định địa chỉ ngăn xếp

Tập lệnh: Các chế độ định địa chỉ và định dạng

- ◼ Chế độ định địa chỉ x86
- ◼ Các chế độ định địa chỉ ARM
- ◼ Định dạng lênh
- ◼ Kích thước lệnh
- ◼ Phân bố các bit
- ◼ Các lệnh có kích thước thay đổi
- ◼ Định dạng lênh X86
- ◼ Định dạng lệnh ARM