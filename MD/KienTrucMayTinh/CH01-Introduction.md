![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000000_352311d92e7b1acbc1d437076507b0e3dc5bc9dd60c28c6a766ac457aac7f002.png)

KIẾN TRÚC MÁY TÍNH

## + Giới thiệu môn học Kiến trúc máy tính

- ◼ Tên môn học: Kiến trúc máy tính

- ◼ Phân loại môn học: Môn bắt buộc .

- ◼ Mã số môn học: CSE370

- ◼ Số tín chỉ: 3 (39 t LT – – 6t TH/BT)
- ◼ Tài liệu học tập:
- ◼ Computer Organization and Architecture, William Stallings, 9th Edition, 2012 – Tiê ́ ng Việ ̣ t – Gia ́ o tri ̀ nh KTMT
- ◼ https://sites.google.com/a/wru.vn/thaont/kien-truc-may-tinh
- ◼ Tổ chức đánh giá môn học

| TT   | Các hình thức đánh giá                          | Trọng số   |
|------|-------------------------------------------------|------------|
| 1    | Điểm quá trình (điểm danh + thảo luận + thi GK) | 50%        |
| 2    | Thi trắc nghiệm hết môn                         | 50%        |
|      | Điểm môn học = ĐQT x 50% + THM x 50%            |            |

+

## Nội dung môn học

Chương 1 – Giới thiệu Chương 2 – Sự phát triển của máy tính và hiệu năng Chương 3 – Tổng quan về chức năng và kết nối trong máy tính Chương 4 – Bộ nhớ Cache Chương 5 – Bộ nhớ trong Chương 6 – Bộ nhớ ngoài Chương 7 – Vào/Ra Chương 8 – Hệ thống số Chương 9 – Bộ xử lý số học Chương 10 – Tập lệnh: Các đặc tính và chức năng Chương 11 – Tập lệnh: Chế độ địa chỉ và khuôn dạng Chương 12 – Tổ chức và chức năng bộ vi xử

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000001_6362fca6cc9e37e53262a65153619782c907ba3fe3ca8c078776c6a7b067f13c.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000002_6f7976667d08406edd7cde14f52a103889702f57ecb2933fbf4341e218d71ae2.png)

Giới thiệu

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000003_b8eb0923bd17d7163e0d6bec6d87c16bc580eedd820ebf12c6c183e0a64dc0b6.png)

+

## Chương 1 – Giới thiệu

- 1.1 Cơ sở toán học và các hệ đếm
- 1.2 Tổ chức và kiến trúc
- 1.3 Cấu trúc và chức năng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000004_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## 1.1 Cơ sở toán học và các hệ đếm

- ◼ Hệ đếm là một tập các ký hiệu (bảng chữ số) để biểu diễn các số và xác định giá trị của các biểu diễn số .
- ◼ Phân loại:
- ◼ Hệ đếm không vị trí
- ◼ Hệ đếm có vị trí
- ◼ Các hệ đếm thông dụng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000005_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## Hệ đếm có vị trí

- ◼ Nguyên tắc chung
- ◼ Cơ số của hệ đếm 𝑟 là số ký hiệu được dùng
- ◼ Trọng số bất kỳ của một hệ đếm là 𝑟 𝑖 (i là số nguyên âm hoặc dương) giúp phân biệt giá trị biểu diễn của các chữ số khác nhau
- ◼ Mỗi số được biểu diễn bằng một chuỗi các chữ số, trong đó số ở vị trí thứ 𝑖 có trọng số 𝑟 𝑖
- ◼ Dạng tổng quát của một số trong hệ đếm có cơ số r là

<!-- formula-not-decoded -->

- 𝑟
- ◼ Giá trị của chữ số a i là 1 số nguyên trong khoảng 0 &lt; a i &lt; r.
- ◼ Dấu chấm giữa a 0 và a -1 được gọi là "d "dấu phẩy "

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000006_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## Biểu diễn số

Số . . . 𝑎3𝑎2𝑎1𝑎0 . 𝑎 − 1𝑎 − 2𝑎 − 3 . . . 𝑟 biểu diễn giá trị:

- ◼ Trong một số trường hợp, ta phải thêm chỉ số để tránh nhầm lẫn giữa các hệ đếm .

<!-- formula-not-decoded -->

- ◼ Chũ số quan trọng nhất (MSB): Chữ số ngoài cùng bên trái (mang giá trị lớn nhất)
- ◼ Chữ số ít quan trọng nhất (LSB): Chữ số ngoài cùng bên phải

+

## 1. Hệ đếm

- a. Hệ thập phân
2. ◼ Dựa trên 10 chữ số thập phân (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) để biểu diễn các s ố. Cơ số = 10
3. ◼ Phân bố trọng số:

- ◼ Ví dụ: 83 10 , 4728 10,

| Vị trí     | …   | 3    | 2    | 1    | 0    | - 1    | - 2    | - 3    | - 4    | …   |
|------------|-----|------|------|------|------|--------|--------|--------|--------|-----|
| Trọng  s ố | …   | 10 3 | 10 2 | 10 1 | 10 0 | 10 − 1 | 10 − 2 | 10 − 3 | 10 − 4 | …   |

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

+

## 1. Hệ đếm

- b. Hệ nhị phân
2. ◼ Hai chữ số, 1 và 0
3. ◼ Cơ số 2
4. ◼ Chữ số 1 và 0 trong ký hiệu nhị phân có cùng ý nghĩa như trong ký hiệu thập phân:

<!-- formula-not-decoded -->

- ◼ Để biểu diễn các số lớn hơn , m ỗi chữ số trong một số nhị phân có giá trị phụ thuộc vào vị trí của nó :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Các giá trị phân số được biểu diễn bằng số mũ âm của cơ số:

<!-- formula-not-decoded -->

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000007_4f94bf6706a306b023a1d6b74071b5f325b0f46fcde5d5d3afc1abde884e6e7b.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000008_8bc768d4d54e26be6531477a930fe383b41c39bbef5040ea912ff61cbb97780d.png)

## 2 . Chuyển đổi hệ thập phân và nhị phân

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000009_e55dc1308383ae77f2e67eac9dba2b98efe6fe41ec5ee68a4bbda9fa0b3f7632.png)

+

## a. Phần nguyên:

Bài toán: Đổi số nguyên thập phân N thành dạng nhị phân.

Đầu tiên chia N cho 2 được N1 N1 và phần dư R 0 :

<!-- formula-not-decoded -->

Tiếp theo, chia N1 N1 cho 2 thu được số mới là N2 N2 và số dư mới R 1 :

<!-- formula-not-decoded -->

Sao cho

<!-- formula-not-decoded -->

Nếu tiếp tục

<!-- formula-not-decoded -->

Ta có

<!-- formula-not-decoded -->

## Phần nguyên

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000010_5eaba62b082de08755e184e038948f41375d06979a327517d062b2761ed1a03b.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000011_596cf4836a1ae76f7665dc0987c8166a51c3641e3d1645e7741a98f97e01bf1f.png)

Do N &gt;N1 N1 &gt; N2 N2 . . . , tiếp tục chia thì cuối cùng sẽ tạo ra thương số Nm Nm-1 = 1 và phần dư R m-2 bằng 0 hoặc 1 .

Khi đó

<!-- formula-not-decoded -->

là dạng nhị phân của N.

Kết luận: Chuyển đổi phần nguyên từ cơ số 10 sang cơ số 2 bằng cách chia lặp đi lặp lại số đó cho 2. Phép chia dừng lại khi kết quả lần chia cu ối cùng bằng 0 .

+ 
L + 
Lấy các số dư theo chiều đảo ngược cho ta số + 
nhị phân cần tìm .

## Phần nguyên

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000012_b1ab5a0671defb6721082d088cc90dbb2c73d3f9aa98c9aa1f486a04878cc6ec.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000013_4f36716c285386cbbcd2b6009cfb18e721c79e3fe9e634c0ef9cf06d467079e5.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000014_a5b3daf1365f9ccfaffd4c95224acd0837495328da377266e5f2dbc4817e1205.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000015_2a0f3cdd91561f9031d88fe00f39cf52cfd6e734e53ad39a8f9392df65f3f645.png)

Số nhị phân 0.b -1 b -2 -2 b -3 . . . với b i = 0 or 1 có giá trị

<!-- formula-not-decoded -->

Có thể viết lại thành

<!-- formula-not-decoded -->

Bài toán: Đổi số F (0 &lt; F &lt; 1) từ thập phân sang nhị phân. Biết rằng F có thể được biểu diễn dưới dạng

<!-- formula-not-decoded -->

Nếu nhân F với 2, thu được,

<!-- formula-not-decoded -->

+ 
c Tư biểu thức đó, ta thấy rằng phần nguyên của (2 * F), phải bằng 0 hoặc 1 vì 0 &lt; F &lt; 1, đơn giản là b -1 . Vì thế ta có thể nói (2 * F) = b -1 + F1 F1, v ới 0 &lt; F1 F1 &lt; 1 và trong đó F1 F1 = 2 -1 * (b -2 -2 + 221 * (b -3 + 221 * (b -4 + . . . ) . . . )) Để tìm b − 2 , ta lặp lại quá trình này. y. Tại mỗi bước, phần phân số của kết quả bước trước được nhân với 2.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000016_c5cb06944e3e8d4fc2c38611921a4bb2746f6831883413d4f531de9c1d4739c9.png)

## Phần thập phân

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000017_825fc2c60c755dd86866c4db57aa5ab1d18240c8723076483140c0c756f5fab3.png)

Kết luận: Nhân liên tiếp phần phân số của số thập phân với 2. Lấy tuần tự phần nguyên c ủa tích thu được sau mỗi lần nhân là kết quả c ần tìm . Phần phân số của tích được sử dụng làm số bị nhân trong bước tiếp theo.

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000018_95ef65907e685f3faf5c6cf08a5a66638b5bea41ba573c044725cf1171e15a53.png)

## Phần thập phân

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000019_c96c87d513b70a80c55c0016ed48583fd466e8e827318be9b13f58e986955f3a.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000020_f83468d0e775d0c79755813c52f511e6627506eacc89371cd7d1ac45eb39b7c9.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000021_ff0ccae44b24a24bf503f9e38bd085099049ec9fbcc22bb7ff3987007655548e.png)

+

## 5. Hệ thập lục phân (Hexadecimal)

- ◼ Gồm 16 chữ số ́ : 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 𝐴 , 𝐵 , 𝐶 , 𝐷 , 𝐸 , 𝐹
- ◼ Các chữ số nhị phân được nhóm thành các nhóm bốn bit được gọi là nibble
- ◼ Mỗi tổ hợp có thể có của bốn chữ số nhị phân được biểu diễn bằng 1 chư ̃ sô ́ trong hệ ̣ 16, như sau :

<!-- formula-not-decoded -->

- ◼ Bởi vì 16 ký tự được sử dụng, biểu diễn này được gọi là hệ thập lục phân và 16 ký tự đó là chữ số thập lục phân
- ◼ Ví dụ

<!-- formula-not-decoded -->

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000022_0a7db9b53fe6fe250ae76d90cc6934ed62fdc8bf508736c10a1cf403e0cce230.png)

## Bảng 8.3

Thập phân, nhị phân, và thập lục phân

## Biểu diễn thập lục phân

Không chỉ được dùng để biểu diễn các số nguyên mà còn là một biểu diễn ngắn gọn để biểu diễn dãy số nhị phân bất kỳ

Ngắn gọn hơn ký hiệu nhị phân

Lý do sử dụng biểu diễn thập lục phân:

Trong hầu hết máy tính, dữ liệu nhị phân chiếm theo bội của 4 bit, tương đương với bội của một số thập lục phân duy nhất

Rất dễ dàng chuyển đổi giữa nhị phân và thập lục phân

## Bài tập (1)

1/ Sắp xếp các số theo thứ tự tăng dần: (1.1) 2 , (1.4) 10 , (1.5) 16

- 2/ Đổi giá trị biểu diễn
- a) 54 8 sang hệ cơ số 5 b) 312 4 sang hệ cơ số 7 3/ Đổi các số nhị phân sau ra số trong hệ thập phân: a) 001100 b) 011100 c) 101010 d)11100.011 e) 110011.10011 4/ Đổi các số thập phân sau ra số trong hệ nhị phân: a) 64 b) 100 c) 255
- d) 34.75
- e) 25.25
- f) 1010101010.1
- f) 27.1875

## Bài tập (2)

5/ Đổi các số thập lục phân sau ra số trong hệ thập phân:

- a) B52

- b) ABCD

- c) D3.E

- d) 1111.1

- e) EBA.C

- 6/ Đổi các số thập phân sau ra số trong hệ thập lục phân:

- a) 2560

- b) 6250

- c) 16245

- d) 204.125

- e) 255.875

- f) 631.25

- 7/ Đổi các số thập lục phân sau ra số trong hệ nhị phân:

- a) 568

- b) A74

- c) 1F.C

- d) 239.4

- 8/ Đổi các số nhị phân sau ra số trong hệ thập lục phân:

- a) 1001.1111

- b) 110101.011001

- c) 101001111.111011

## + 1.2 Kiến trúc máy tính

## Tổ chức máy tính

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000023_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## ◼ Tổ chức máy tính

- ◼ Các khối của máy tính và sự kết nối giữa chúng để thực hiện các đặc điểm của kiến trúc

## ◼ Các thuộc tính của tổ chức

- ◼ Chi tiết đặc tính phần cứng: Tín hiệu điều khiển, giao diện giữa máy tính và thiết bị ngoại vi, công nghệ bộ nhớ được sử dụng

## ◼ Kiến trúc máy tính

- ◼ Kiến trúc máy tính đề cập đến những thuộc tính của một hệ thống lập trình viên có thể nhìn thấy được
- ◼ Các thuộc tính có tác động trực tiếp đến việc thực hiện chính xác một chương trình .

## ◼ Các thuộc tính của kiến trúc

- ◼ Tập lệnh: là tập hợp các lệnh mã máy hoàn chỉnh có thể hiểu và xử lý bởi bộ xử lý trung tâm .
- ◼ Số bit dùng để biểu diễn dữ liệu
- ◼ Cơ chế I/O
- ◼ Kỹ thuật định địa chỉ bộ nhớ

+

## Kiến trúc hệ thống

## IBM 370

## ◼Kiến trúc hệ thống IBM 370

- ◼ Được giới thiệu vào năm 1970
- ◼ Bao gồm môt số model
- ◼ Có thể nâng cấp lên model đắt tiền và tốc độ nhanh hơn mà không cần bỏ đi các phần mềm gốc
- ◼ Mỗi mẫu model mới tung ra được cải tiến kĩ thuật nhưng giữ nguyên kiến trúc do đó khách hàng không cần mua phần mềm m ới
- ◼ Kiến trúc này được duy trì đến ngày nay trên các dòng máy tính IBM lớn

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000024_543740b77e646c336cade7d58cfc8af9c5f0ca7f7b08bc1cbdfadaa937d586b2.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000025_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## 1.3 Cấu trúc và chức năng

- ◼ Máy tính là một hệ thống phức tạp . Để mô tả , người ta dựa trên tính phân cấp của máy tính
- ◼ Hệ thống phân cấp
- ◼ Là tập hợp các hệ thống con có liên kết với nhau
- ◼ Tính phân cấp của hệ thống phức tạp là cần thiết cho cả thiết kế và mô tả của nó .
- ◼ Nhà thiết kế chỉ cần làm việc với một cấp cụ thể của hệ thống tại m ột thời điểm
- ◼ Tại mỗi cấp: hệ thống có các bộ phận và sự kết nối giữa chúng
- ◼ Mỗi cấp có cấu trúc và chức năng riêng

## Cấu trúc

Cách thức các bộ phận liên quan đến nhau

## Chức năng

Hoạt động của từng bộ phận trong cấu trúc

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000026_d1ad154fec969a1ae4afe31ddd3f44f872616decafded78ec2f11416b8c4d769.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000027_5f39264182ccbd3d2bc7f4b77be10c497a267cf6b29e926a0efe2e43b4bb240e.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000028_2e7c057ef5e391926460f9e54612574d53ddb3e91958d3f5f6facdd42f1dee59.png)

Figure 1.2 Possible Computer Operations

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000029_ef84d774a9d485c5bea006cd289c4ea0bf5935f6ee941cf1c17f8307cb40f74a.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000030_be0fbe8ba10ad001b077aa8e10c854d53792757f022414458c56578d35509c9c.png)

Movement

Control

(c)

Movement

Control

(b)

Movement

Control

(d)

Figure 1.2 Possible Computer Operations

Storage

Processing

Storage

Storage

Processing

Processing

Storage

Storage

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000031_17ed6db2d630577532ea275b76c7be68d4aebca932ee96345db49bd87840f175.png)

Movement

Control

(c)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000032_388ed552a19035d98082e5b038f69dbeccce35f5bb2729545c36a4a75f4ff6e0.png)

Movement

Control

(d)

Figure 1.2 Possible Computer Operations

Processing

Storage

Storage

Storage

Processing

Storage

Movement

Control

(a)

Figure 1.2 Possible Computer Operations

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000033_2c1b115d181d20e760090feaae48b990a60cb4e6c9a53d610f1b53f1c20d449a.png)

Figure 1.2 Possible Computer Operations

Processing

Storage

Storage

Movement

Control

(b)

Movement

Control

(d)

Processing

Processing

Storage

Storage

Movement

Control

(a)

+

+
Movement

## Hoạt động

Control

(c)

(d) Điều khiển (Control)

Processing

Processing

Storage

Movement

Control

(b)

Figure 1.2 Possible Computer Operations

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000034_a45282543d009bb022825791159d0baec340dd07c3fb334f4694957dcd98d950.png)

Figure 1.2 Possible Computer Operations

Processing

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000035_7b99533b9c131ce397ae9c37cc6bc0da187c71fda875ba9ce71f808fb47dec89.png)

## Máy tính

1.2.2 Cấu trúc

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000036_4b6224c81c1daf119eaa6fd2b2c00a8b38cbe36553d4299b1b807db140c03ba2.png)

-  CPU – – bộ xử lý trung tâm điều khiển hoạt động của máy tính và thực hiện chức năng xử lý dữ liệu
-  Bộ nhớ chính: lưu trữ dữ liệu . Là tập hợp các ô nhớ , m ỗi ô nhớ có một số bit nhất định và chứa thông tin mã hoá số nhị phân .
-  I/O – bộ phận nhập xuất thông tin – thực hiện giao tiếp giữa máy tính và người dùng hay giữa các máy tính trong cùng mạng ,
- Hệ thống kết nối (bus) – – một số cơ chế cung cấp cho việc truyền đạt thông tin giữa CPU, bộ nhớ chính và I/O
- ◼ Bộ điều khiển (Control Unit -CU)
- ◼ Điều khiển hoạt động của CPU và cả máy tính
- ◼ Bộ số học và logic (Arithmetic and Logic Unit -ALU)
- ◼ Thực hiện chức năng xử lý dữ liệu
- ◼ Thanh ghi (Registers)
- ◼ Cung cấp lưu trữ nội bộ cho CPU
- ◼ Các kết nối trong CPU
- ◼ Một số cơ chế dùng để cung cấp thông tin liên lạc giữa các khối CU, ALU và các thanh ghi .

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000037_efcb50d84c9fcfda017b2bb1971010b8391970d6ce36c134a8424628d8eff6db.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000038_30981b64c33ba4e2ad1380dcfb74b831c81f7862042d0ba3cda6df121e225833.png)

## Tổng kết

## Chương 1

- ◼ Hệ đếm và cơ sở toán học
- ◼ Tổ chức máy tính
- ◼ Kiến trúc máy tính
- ◼ Chức năng
- ◼ Xử lý dữ liệu
- ◼ Lưu trữ dữ liệu
- ◼ Di chuyển dữ liệu
- ◼ Điều khiển

## Introduction

- ◼ Cấu trúc
- ◼ CPU
- ◼ Bộ nhớ chính
- ◼ I/O
- ◼ Kết nối hệ thống
- ◼ Thành phần cấu trúc CPU
- ◼ Bộ điều khiển CU
- ◼ Bộ làm toán và logic ALU
- ◼ Thanh ghi
- ◼ Kết nối CPU

## + Từ khóa

- ◼ Arithmetic and logic unit (ALU): khối (đơn vị) số học và logic
- ◼ Central processing unit (CPU): khối (đơn vị) xử lý trung tâm
- ◼ Computer architecture: Kiến trúc máy tính
- ◼ Computer organization: Tổ chức máy tính
- ◼ Control unit: Khối (đơn vị) điều khiển
- ◼ Input–output (I/O): Vào-ra
- ◼ Main memory: Bộ nhớ chính (ROM, RAM)
- ◼ Processor: Vi xử lý
- ◼ Register: Thanh ghi
- ◼ System bus: Bus hệ thống
- ◼ Sự khác nhau giữa kiến trúc và tổ chức máy tính
- ◼ Sự khác nhau giữa chức năng và cấu trúc máy tính
- ◼ Bốn chức năng chính của máy tính là gì
- ◼ Liệt kê và định nghĩa tóm tắt bốn thành phần chính của máy tính
- ◼ Liệt kê và định nghĩa tóm tắt bốn thành phần chính của VXL

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000039_14ffcaabf263070de97ffa1772bc0c882067c0c87e89306922f3722c4cd84506.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH01-Introduction_artifacts/image_000040_7e49940254487746e3b6df51eb67994329a4b85791c9d9a2c7cd3f93e6252b43.png)