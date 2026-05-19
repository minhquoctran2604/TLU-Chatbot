## + Chương 9

Bộ xử lý số học

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000000_b8eb0923bd17d7163e0d6bec6d87c16bc580eedd820ebf12c6c183e0a64dc0b6.png)

## + Chương 9. Bộ xử lý số học

1. Đơn vị số học và logic
2. Biểu diễn số nguyên
3. Phép toán số học với số nguyên
4. Biểu diễn dấu chấm động
5. Phép toán với dấu chấm động

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000001_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000002_2e61313693f0e0669d1aabbebe3543210d03fe54f78e365b722fb8a3a1d0760a.png)

## 1. Đơn vị số học &amp; logic (ALU)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000003_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

- ◼ Phần của máy tính thực hiện phép toán số học và lôgíc trên dữ liệu
- ◼ Tất cả các bộ phận khác trong hệ thống máy tính (CU, thanh ghi, bộ nhớ, I/O) đưa dữ liệu tới ALU để xử lý, sau đó gửi lại kết quả
- ◼ Khối ALU được thực hiện sử dụng các linh kiện logic số có thể lưu trữ các số nhị phân và thực hiện các phép toán logic Boolean đơn giản

## Đầu vào và đầu ra ALU

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000004_a4be7365fd6d6fa92b7c41519eae8b04e4816792d7f9ef2da89ba3bcc0747b7a.png)

- Control Signals: các tín hiệu điều khiển được gửi đến từ CU, điều khiển hoạt động của ALU
- Operand registers: các thanh ghi lưu trữ giá trị toán hạng của phép toán
- Result registers: các thanh ghi lưu trữ kết quả phép toán
- Flags: các cờ. Vd: cờ tràn để đánh dấu kết quả tính toán vượt quá kích thước thanh ghi đang lưu trữ

+

## 2. Biểu diễn số nguyên

Biểu diễn dữ liệu trong máy tính

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000005_356a4579686189c8dbd7bf8de0de62a0f5b96d70c7be2d2ee60a31a9c0ec674f.png)

- ◼ Dữ liệu cần được mã hóa để lưu trữ và xử lý trong máy tính: dưới dạng nhị phân
- ◼ Dữ liệu kiểu số: độ lớn biểu diễn dưới dạng số nhị phân. Quy ước biểu diễn dấu âm (với số âm), dấu phẩy (với số có phần thập phân).
- ◼ Dữ liệu ký tự: sử dụng bảng mã
- ◼ Dữ liệu âm thanh, hình ảnh: lấy mẫu, lượng tử, mã hóa theo quy ước.

+

## Quy ước biểu diễn số trong máy tính

- ◼ Số nguyên: có hai dạng biểu diễn
- ◼ Biểu diễn dấu – – độ lớn
- ◼ Biểu diễn bù 2
- ◼ Đặc điểm: cả hai dạng biểu diễn đều sử dụng bit quan trọng nhất (MSB -most significant bit) làm bit dấu
- ◼ Số thực:
- ◼ Biểu diễn dấu chấm tĩnh
- ◼ Biểu diễn dấu chấm động

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000006_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## a. Biểu diễn dấu – độ lớn

- -Sign-magnitude representation
- Đây là dạng biểu diễn đơn giản nhất
- Trong một từ n bit, sử dụng (n-1) bên phải biểu diễn độ lớn của số
- Bit ngoài cùng bên trái làm bit dấu: 0 (+), 1( -)
- Nhược điểm:
- Thực hiện phép toán cộng, trừ đòi hỏi phải xét cả dấu của các số và độ lớn của chúng
- Có hai dạng biểu diễn của số 0: gây khó khăn khi thực hiện việc kiểm tra 0 trong một số phép toán

- Do những nhược điểm này, biểu diễn dấu – độ lớn hiếm khi được sử dụng trong việc mã hóa phần số nguyên trong ALU

## + b. Biểu diễn bù 2

- ◼ Bit ngoài cùng bên trái làm bit dấu: 0 (+), 1( -)
- ◼ Khác với biểu diễn dấu – độ lớn ở cách biểu diễn số âm
- ◼ Biểu diễn bù 2 số dương: giống dấu – độ lớn
- ◼ Biểu diễn bù 2 số âm:
- ◼ Lấy bù 1 của số dương tương ứng (đảo 0 → 1 và 1 → 0)
- ◼ Cộng thêm 1
- ◼ (Cách 2: đọc ngược từ dưới lên, gặp bit 1 đầu tiên, các bit sau đó sẽ đảo ngược 0 →1, 1→0)

## + Ví dụ 1

- ◼ Biểu diễn các số sau sang dạng dấu -độ lớn 8b và bù 2-8b
- a) -54
- b) 11
- c) -13
- d) 145

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000007_6362fca6cc9e37e53262a65153619782c907ba3fe3ca8c078776c6a7b067f13c.png)

+

## b. Biểu diễn bù 2 (tiếp)

Miền giá trị của từ mã n bit: − 2 𝑛 − 1 đến 2
𝑛 − 1 − 1

Tính toán giá trị mã bù 2:

- ◼ Một số nguyên A, biểu diễn dưới dạng bù 2, n -bit:

<!-- formula-not-decoded -->

- ◼ Nếu A là số dương
- ◼ Bit dấu 𝑎 𝑛 − 1 có giá trị 0
- ◼ Nếu A là số âm (A&lt;0)
- ◼ Bit dấu 𝑎 𝑛−1 có giá trị 1

- ❖ Trong đó, 𝒂𝒊là giá trị bit tại vị trí i

## + Hộp giá trị

## Bảng 10.2 Biểu diễn số nguyên 4 -Bit

| Biểu diễn thập  phân   | Biểu diễn dấu –  –  độ  lớn   | Biểu diễn bù 2   |
|------------------------|-------------------------------|------------------|

+

## Mở rộng phạm vi

- ◼ Trong một số trường hợp, ta muốn biểu diễn một số n -bit sang dạng biểu diễn m -bit (𝑚 &gt; 𝑛): mở rộng phạm vi biểu diễn
- ◼ Trong biểu diễn dấu – độ lớn: di chuyển bit dấu tới vị trí mới ngoài cùng bên trái và điền (m -n) bit 0 vào sau bit dấu
- ◼ Biểu diễn số bù 2:
- ◼ Quy tắc: di chuyển bit dấu sang vị trí ngoài cùng bên trái và điền vào bằng bản sao bit dấu
- ◼ Đối với số dương, điền 0 vào , và số âm thì điền vào số 1
- ◼ Đây được gọi là phần mở rộng dấu

## + Mở rộng phạm vi Ví dụ số bù 2

+

## Một số đặc điểm của biểu diễn bù 2

| Miền giá trị (n bit)         | − 2 𝑛−1  đến 2 𝑛 − 1  −  1                                                                |
|------------------------------|----------------------------------------------------------------|
| Biểu diễn số 0               | 1 cách                                                         |
| Biểu diễn số âm              | Lấy bù của số dương tương ứng sau đó cộng thêm  1              |
| Mở rộng chiều dài  chuỗi bit | Điền giá trị dấu vào bên trái                                  |
| Luật tràn                    | Khi cộng hai số cùng dấu, nếu kết quả có dấu  ngược lại → tràn |
| Luật trừ                     | Khi trừ A cho B, lấy bù 2 của B sau đó cộng với A              |

## + 2. Phép toán trên số nguyên

- a. Phép đảo (phép phủ định)
- b. Phép cộng
- c. Phép trừ - luật trừ
- d. Phép nhân
- e. Phép chia

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000008_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## a. Phép đổi dấu (phép lấy âm)

- ◼ Quy tắc:
- ◼ Biểu diễn dấu – độ lớn: đảo bit dấu
- ◼ Biểu diễn bù 2: thực hiện phép toán bù 2
- ◼ Đảo từng bit (kể cả bit dấu)
- ◼ Cộng 1

- ◼ Đảo của đảo một số là chính nó

## +

## Bài tập

1. Tính giá trị các số bù 2 sau:
2. Thực hiện phép đảo dấu của các số sau theo 2 dạng biểu diễn: dấu -độ lớn và bù 2

a. 1100 1110 b. 0001 0001

- c. 1000 0010 d. 0111 0000

a. 1000 1001 b. 1000 0000

c. 1000 1110

d. 0000 1110

## + Số bù 2: xét hai trường hợp đặc biệt

## b. Phép cộng (bù 2-4b)

- Phép cộng được thực hiện bình thường như cộng hai số nhị phân
- Trong một số trường hợp, xuất hiện thêm 1b (bit bôi đen) → bỏ qua các bit này
- Tràn ô nhớ: khi kết quả của một phép toán quá lớn vượt qua phạm vi biểu diễn của ô nhớ
- Số bù 2: Tràn ô nhớ xảy ra nếu hai số cùng dấu cộng với nhau mà kết quả thu được lại là một số có dấu ngược với dấu của hai số đó
- Khi phát hiện tràn, ALU cần phải báo hiệu việc này để CPU không sử dụng kết quả

|    |
|----|

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000009_3fb20db70f20f58f6a94f4aef1c212823884c80f14cbcb03559d410fcd031549.png)

+

Nguyên tắc

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000010_394445dfcb9a7b48e0a771c5dab33042f1089739b58feaaadfa58885ffb7290c.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000011_cbad7bcc0261059242ff590fe59964b831c119d82cc5fd49f7fb9967d61cd67e.png)

## c. Phép trừ

## NGUYÊN TẮC TRỪ:

Lấy bù 2 (đảo dấu) của số trừ và cộng với số bị trừ.

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000012_bd0e52ccf70be9e48b2eac42a87c3c0ece885d319c5f43a48b3d08bcef46afd8.png)

## Phép trừ

## Mô tả hình học của số nguyên bù 2

Hình 10.5 Mô tả hình học của số nguyên bù 2

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000013_35db2863136b06ce7f19a68cfd3229a2da46ad5aefc4a41f37d7a634b309a84c.png)

## Phần cứng thực hiện cộng và trừ

- Thanh ghi A, B (A, B Register): lưu trữ hai số
- Với phép trừ: thanh ghi B lưu trữ số trừ
- Complementer: bộ lấy bù 2
- SW (switch): lựa chọn cộng hoặc trừ
- Bộ cộng (Adder): thực hiện phép toán và đưa ra cờ tràn (nếu có)
- Cờ tràn (OF-overflow bit):
- 0: không tràn
- 1: tràn
- Kết quả có thể được lưu trữ ở thanh ghi thứ 3 hoặc một trong 2 thanh ghi A, B
- So với phép cộng và phép trừ, phép nhân phức tạp hơn
- Nhiều thuật toán tính toán phép nhân khác nhau đã được sử dụng trong các máy tính khác nhau
- Trong phần này:
- i. Phép nhân giữa hai số nguyên không dấu
- ii. Phép nhân hai số biểu diễn bù 2

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000014_62b046f6f6dc5e5c44f12a94758d120d89a7a0ba4ebd733e507751cd8296650c.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000015_4af3985d89fe101cd5a409b1c34786aab7c20d18d073a78d4cdf7f2b7dd69086.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000016_97746ccfdc02962defc689fc73f015bfe96c49ca82f4cbacda8c9d2e78677f0e.png)

+

## i. Phép nhân giữa hai số nguyên không dấu

## Các bước bằng tay:

- Tính các tích thành phần
- Nếu số nhân là bit 0, tích thành phần bằng 0. Nếu số nhân là bit 1, tích thành phần là số bị nhân (multiplicand)
- Tính tổng các tích thành phần (mỗi tích dịch trái đơn vị so với tích trước đó)
- Tích của hai số nbit là một số có kích thước 2n -bit

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000017_22e247238ccd158a99563097b875d4eee0646cf597898b163752c640309d6505.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000018_be75ee9ea676a73bb36a1a11836b3763fcdcb9e0a47f47b0d98ea171897d71e3.png)

## Mô tả:

- ◼ Thanh ghi M: số bị nhân
- ◼ Thanh ghi Q: số nhân
- ◼ Thanh ghi 1bit C: ghi giá trị tràn của A+M (nếu có)
- ◼ Kết quả 2n -bit: thanh ghi A, Q

Thuật toán phép nhân nhị phân không dấu theo hướng máy tính

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000019_236ea5c85abb9a5de8d4d120ef81ec17efcd7ed807397cdcc6b5d58e1361c4dd.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000020_0a7db9b53fe6fe250ae76d90cc6934ed62fdc8bf508736c10a1cf403e0cce230.png)

## Ví dụ phép nhân nhị phân không dấu

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000021_fd9d44deb2ad4a7ec6a05e374b6da2df12d99cc75a2f50841473e0bf6b1cc04e.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000022_6185c607bfdeb38889cf638b07c564e4ef3656a71fc9e24c408f19faa83a22c7.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000023_e8ebf6968e1304e47b024ac62cd7beb946d7675fc359cd1f1fe3984ea0b07358.png)

## +ii +ii. Phép nhân bù 2

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000024_ba26d287f40e43c1fe9c57223dea2e6f18efd98adef7eb11dbc69ee71f209739.png)

- Số nguyên dương: nguyên tắc nhân giống số nguyên không dấu
- Số nguyên âm: do có sự xuất hiện của bit dấu nên nguyên tắc trên không còn đúng nữa

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000025_1781b38f315ce5359753556db8c68a9fb397c873c45a812dbbbae4f5c9b8bab7.png)

+

## Giải pháp 1

Sử dụng thuật toán nhân hai số không dấu:

- Bước 1: Chuyển đổi số nhân và số bị nhân thành số dương tương ứng.
- Bước 2: Nhân 2 số bằng thuật giải nhân số nguyên không dấu → được tích 2 số dương.
- Bước 3: Hiệu chỉnh dấu của tích:
- Nếu 2 thừa số ban đầu cùng dấu thì tích nhận được ở bước 2 là kết quả cần tính.
- Nếu 2 thừa số ban đầu khác dấu nhau thì kết quả là số bù 2 của tích nhận được ở bước 2.

## Giải pháp 2: thuật toán Booth

- Tốc độ tính toán nhanh hơn do số lượng phép toán ít hơn
- Thuật toán chung cho cả số nguyên dương và âm

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000026_39e8dc1ab9f18dc04f5bc0275e1a75bafcdda8117bf4d278643795a82482815a.png)

## Thuật toán Booth

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000027_977591bebc4ccd0b4721cd5506e10ef8ca1c46c753b2daef2eaeb646029128cc.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000028_0dfe2eb865a6cab9887eea9c1bb4a53f5c0afee5f28ed73b496ca2e91140f0bd.png)

## + Giải thích thuật toán

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- ◼ Quy tắc: duyệt từ trái sang phải:
- Nếu gặp 10 thì trừ A đi M rồi dịch phải
- Nếu gặp 01 thì cộng A với M rồi dịch phải
- Nếu gặp 00 hay 11 thì chỉ dịch phải

## Ví dụ thuật toán Booth

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000029_94a3ccc9f059dee90f3f6ed40b731f7b4a5fd8fa75e7971109aed56056f83226.png)

+

## e. Phép chia

Phức tạp hơn phép nhân nhưng cũng sử dụng chung nguyên lý

## Bằng tay

Thuật toán thực hiện phép chia hai số nguyên không dấu trong máy tính

- ◼ Số chia đặt trong thanh ghi M, số bị chia trong thanh ghi Q. A=0. Bộ đếm = n
- ◼ Tại mỗi bước:
- ◼ A và Q dịch trái 1 đơn vị
- ◼ Thực hiện 𝐴 − 𝑀, nếu
- ◼ 𝐴 − 𝑀 ≥ 0 thì 𝑄 0 = 1 , A A = 𝐴 − 𝑀
- ◼ 𝐴 − 𝑀 &lt; 0 thì 𝑄 0 = 0
- ◼ Giảm bộ đếm và quay trở lại thực hiện vòng lặp đến khi Bộ đếm = 0

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000030_8dbec63bb91078effb2703700d87df9dd97a81683ff6598d184af6e6b7691c46.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000031_5a2ac39d729ebf138a5915a5c3456876c2e82aede999bcd68d9f52902abfc831.png)

+

## Chia số nguyên có dấu

- ◼ Bước 1: Chuyển đổi số chia và số bị chia thành số dương tương ứng
- ◼ Bước 2: Sử dụng thuật giải chia số nguyên không dấu để chia 2 số dương, kết quả nhận được là thương Q và phần dư R đều dương
- ◼ Bước 3: Hiệu chỉnh dấu kết quả theo quy tắc sau:

+

## + Phép chia số bù 2

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000032_4c492c61e1c6f4bf5859bcbe36e8c79da73e627cf2832861622d6050ec8c748a.png)

- ◼ Tương tự như phép nhân, do có bit dấu nên phải có thuật toán khác:
- ◼ Giả sử số chia V và số bị chia D dương và 𝑉 &lt; 𝐷
- ◼ Nếu 𝑉 = 𝐷 : thương = 1, dư = 0.
- ◼ Nếu 𝑉 &gt; 𝐷 : thương = 0, dư = D
- ◼ Thuật toán như sau:
- ◼ B1: Ghi số bù 2 của V vào thanh ghi M (M chứa số âm của V), ghi D vào thanh ghi 
ế A, Q, bộ đếm = n
- ◼ B2: Dịch A,Q sang trái 1 đơn vị
- ◼ B3: Tính A+M →A
- ◼ B4: Kiểm tra:
- ◼ Nếu A dương (bit dấu = 0), 𝑄 0 = 1
- ◼ Nếu A âm (bit dấu =1), 𝑄 0 = 0, khôi phục A lại giá trị trước đó
- ◼ B5: Giảm bộ đếm đi 1 đơn vị
- ◼ Lặp lại các bước từ 2 đến 5 cho đến khi bộ đếm = 0
- ◼ Với các trường hợp V, D không dương, hiệu chỉnh kết quả dựa theo bảng ở trên

+

## Ví dụ phép chia bù 2

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000033_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## + 4. Biểu diễn dấu chấm động

## a. Nguyên lý

- ◼ Quy ước: "dấu chấm" (point) được hiểu là kí hiệu ngăn 
ầầố y (p) 
cách giữa phần nguyên và phần lẻ của 1 số thực.
- ◼ Có 2 cách biểu diễn số thực trong máy tính:
- ◼ Số dấu chấm tĩnh (fixed -point number):
- ◼ Dấu chấm là cố định (số bit dành cho phần nguyên và phần 
ố lẻ là cố định)
- ◼ Hạn chế: không thể biểu diễn số rất lớn hoặc số thập phân 
ấầ rất nhỏ . g p p
Phần thập phân trong thương của một phép chia
ể ấ p p
hai số lớn có thể bị mất
- ◼ Dùng trong các bộ vi xử lý hay vi điều khiển thế hệ cũ.
- ◼ Số dấu chấm động (floating -point number):
- ◼ Dấu chấm không cố định
- ◼ Dùng trong các bộ vi xử lý hiện nay, có độ chính xác cao hơn.

## +
B +
Biểu diễn số nhị phân dấu chấm động

- ◼ Một số nhị phân có thể được biểu diễn dưới dạng như sau:

- ◼ Trong đó:
- ◼ Nếu B là một số định sẵn, ta chỉ cần lưu trữ S và E
- ◼ Vậy một số nhị phân có thể được lưu trữ trong máy tính với 3 trường sau:
- ◼ Dấu
- ◼ Giá trị S
- ◼ Số mũ E

- ◼ S: phần định trị

- ◼ B: cơ số

- ◼ E: số mũ

## Định dạng dấu chấm động 32 -bit

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000034_ed129db8b377f9d5ee0a3192b2bce0fb856b7f75847c63366b9b9cbc9503f5bd.png)

Ví dụ: các số trên được lưu trữ như sau:

B: ngầm định = 2

1 bit dấu: 0 nếu là số dương, 1 biểu diễn số âm

8 bit biased exponent: số mũ lệch

- →Số mũ thực tế = số mũ lệch -độ lệch (độ lệnh = 2 𝑘 − 1 − 1, k: số bit phần mũ). Trong trường hợp này, độ lệch = 127
- →Số mũ thực tế nằm trong khoảng -127 đến +128
- 23 bit còn lại: phần định trị, được quy ước dạng 1.bbbbb...... Trong đó, số 1 đầu tiên là ngầm định

## + Biểu diễn số nhị phân dấu chấm động 32b

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000035_7ce81789edf692cd3d6c3efc3887ef971557f8abcb184a8f5d30a5743a4849fe.png)

- ◼ Hầu hết các hệ VXL có 3 loại định dạng nhị phân dấu chấm động (quy định theo chuẩn 754)
- ◼ 32b: 1b dấu (S), 8b phần mũ (E), 23b phần định trị (M)
- ◼ 64b: 1b dấu (S), 11b phần mũ (E), 52b phần định trị (M)
- ◼ 128b: 1b dấu, 15b phần mũ (E), 112b phần định trị (M)

## ◼ Trong đó:

- ◼ S: 0 nếu là số dương, 1 nếu là số âm
- ◼ E = số mũ thực tế + độ lệch (độ lệch = 2 𝑘 − 1 − 1, k là số bit phần mũ)
- ◼ M: phần định trị được chuẩn hóa dạng 1.bbbbb.bbbb

## + Chú ý: chuẩn hóa phần định trị

- ◼ Phần định trị có thể được biểu diễn thành nhiều dạng

Các cách viết sau đây là tương đương, trong đó phần định trị được biểu diễn dưới dạng nhị phân:

0.110 * 25

110 * 22

0.0110 * 26

- ◼ Quy ước biểu diễn: đưa về dạng 1.bbbbbbb....
- ◼ Khi lưu trữ không cần lưu trữ phần nguyên, chỉ cần lưu trữ phần thập phân. Dấu chấm ngầm định ngay sau 8 bit phần số mũ

## + Ví dụ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000036_7fa2a0efef1789b4a9b7012873203d8daef200429e6340358ee67f30437b675d.png)

Chuyển số 23.5436 ra số nhị phân biểu diễn dạng dấu chấm động 32b

- ◼ B1: chuyển 23.5436 ra số nhị phân: 10111.1001101 (xấp xỉ)
- ◼ B2: chuẩn hóa: 𝟏 . 𝟎𝟏𝟏𝟏𝟏𝟎𝟎𝟏𝟏𝟎𝟏 𝒙 𝟐 𝟒
- ◼ B3: Biểu diễn dạng dấu chấm động:

0100 0001 1011 1100 1101 0000 0000 0000

## +

## Ví dụ 1

Biểu diễn số thực X = 0 . 375 về dạng số dấu chấm động theo chuẩn IEEE 754 dạng 32 bit

## Ví dụ 2

Có một số thực X có dạng biểu diễn nhị phân theo chuẩn IEEE 754 dạng 32 bit như sau: 1100 0001 0101 0110 0000 0000 0000 0000 . Xác định giá trị thập phân của số thực đó.

## Ví dụ 3

Xác định giá trị thập phân của số thực X có dạng biểu diễn theo chuẩn IEEE 754 dạng 32 bit như sau:

0011 1111 1000 0000 0000 0000 0000 0000

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000037_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000038_ce2a4c9b18ef61f7eb6ca7d6c02e35dbba7713fc1242657be528007b4af824b5.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000039_58ad382c6cf33c012f479c6b5b460b83c390753c1f64400167a0e5c7f932b2b1.png)

+

## Mật độ số dấu phẩy động

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000040_a5a8a1d2c8cd998d1a125c87ee61b88976bb55c8ca899733d0cf78706be435c1.png)

## b. Chuẩn IEEE 754

- ❑ Hiệp hội IEEE đã chuẩn hóa cho việc biểu diễn số dấu phẩy động nhị phân trong máy tính
- ❑ Mục đích:
- Hỗ trợ tính di động của chương trình từ bộ xử lý này sang bộ xử lý khác
- Khuyến khích phát triển các chương trình định hướng số học tinh vi hơn
- ❑ Chuẩn được công nhận rộng rãi và được sử dụng trên hầu hết các bộ VXL và bộ tính toán số học hiện đại
- ❑ IEEE 754 -2008 quy định các định dạng biểu diễn dấu phẩy động nhị phân và thập phân
- ❑ Trong phần này chỉ đề cập đến dạng biểu diễn dấu phẩy động nhị phân

+

## IEEE 754 -2008

- ◼ IEEE 754 -2008 định nghĩa 3 định dạng dấu chấm động sau:
- ◼ Định dạng số học
- ◼ Được sử dụng để biểu diễn các toán hạng hoặc kết quả phép toán dưới dạng dấu chấm động .
- ◼ Định dạng cơ bản: quy định 5 dạng biểu diễn dấu chấm động:
- ◼ Ba cho số nhị phân: chiều dài 32b, 64b và 128b
- ◼ Hai cho thập phân: chiều dài 64b và 128b
- ◼ Định dạng chuyển đổi
- ◼ Đưa ra dạng mã hoá nhị phân độ dài cố định cho phép trao đổi dữ liệu giữa các nền tảng khác nhau và có thể được sử dụng để lưu trữ .

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000041_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000042_a696796ffec8df94060ae00e81e0a6a9dad4044fce44a91e213b71c79229c441.png)

## Bảng 10.3 Thông số định dạng chính trong chuẩn IEEE 754

*không bao gồm bit ngầm định và bit dấu

+

## IEEE754: Một số quy ước đặc biệt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000043_58b8c200836c0ad173ee74acb1e23076a1af98e184e096f071b30e5c7e2b5201.png)

- ◼ Nếu tất cả các bit của phần số mũ đều bằng 0, các bit của phần định trị đều bằng 0, thì X = ± 0
- ◼
- Ví dụ: định dạng 32b:

+0 10 = 0 00000000 00000000000000000000000

− 0 10 = 1 00000000 00000000000000000000000

- ◼ Nếu tất cả các bit của phần mũ đều bằng 1, các bit của phần định trị đều bằng 0, thì X = ± ∞
- ◼ Ví dụ: định dạng 32b:

+∞ = 0 11111111 00000000000000000000000

```
− ∞ = 1 11111111 00000000000000000000000
```

- ◼ Nếu tất cả các bit của phần mũ đều bằng 1, phần định trị có ít nhất một bit bằng 1, thì X biểu diễn một giá trị quy ước NaN (not a number)
- ◼ NaN sinh ra bởi một số phép toán dấu chấm động đặc biệt, vd: ±∞/±∞
- ◼ Có hai loại NaN: quiet NaN và signalling NaN

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000044_39e8dc1ab9f18dc04f5bc0275e1a75bafcdda8117bf4d278643795a82482815a.png)

## Các định dạng bổ sung

## Extended Precision Formats

- ◼ Cung cấp các bit bổ sung trong số mũ (phạm vi mở rộng) và trong phần định trị (độ chính xác mở rộng)
- ◼ Giảm khả năng kết quả cuối cùng bị tồi đi do sai số làm tròn quá mức
- ◼ Giảm bớt khả năng tràn trung gian làm hủy bỏ phép tính có kết quả cuối cùng có thể biểu diễn được dưới định dạng cơ bản
- ◼ Có một số lợi ích của định dạng cơ bản rộng hơn mà không phải chịu chi phí thời gian khi muốn độ chính xác cao hơn

## Extendable Precision Format

- ◼ Độ chính xác và phạm vi được xác định dưới sự kiểm soát của người dùng
- ◼ Có thể được sử dụng để tính toán trung gian nhưng chuẩn sẽ không có ràng buộc hoặc định dạng hoặc chiều dài

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000045_f3ece18c302a67f29ac1570e7af845f2c2affab63dff44590218554f2be61bc5.png)

+

## 5 Các phép toán với số dấu chấm động

## a. Phép toán cộng và trừ

## Cộng và trừ dấu phẩy động

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000046_506f74104e617f15135ee8b04cb1aeb431b4e989479871e989e028c9ec2491a7.png)

+

## Bốn bước cơ bản của thuật toán cộng và trừ

- ◆Kiểm tra các số hạng có bằng 0 hay không
- ◼ Nếu có thì gán kết quả dựa trên số còn lại.
- ◆Hiệu chỉnh phần định trị
- ◼ Sao cho 2 số có phần mũ giống nhau: tăng số mũ nhỏ và dịch phải phần định trị tương ứng (dịch phải để hạn chế sai số nếu có).
- ◼ VD: 1.01 * 23 + 1.11 = 1.01 * 23 + 0.00111 * 23
- ◆Cộng hoặc trừ phần định trị
- ◼ Nếu tràn thì dịch phải và tăng số mũ, nếu bị tràn số mũ thì báo lỗi tràn số.
- ◆Chuẩn hóa kết quả
- ◼ Dịch trái phần định trị để bit trái nhất (bit MSB) khác 0.
- ◼ Tương ứng với việc giảm số mũ nên có thể dẫn đến hiện tượng tràn dưới số mũ.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000047_624b3170d4324a8f3d74332bf7eea1813372eff88f42984e5956bcd20b5a9147.png)

+

## Các khả năng tràn số

Phép toán cộng hoặc trừ có thể gây ra một số khả năng tràn như sau:

- ◼ Tràn trên số mũ (Exponent Overflow): mũ dương vượt ra khỏi giá trị cực đại của số mũ dương có thể.
- ◼ Tràn dưới số mũ (Exponent Underflow): mũ âm vượt ra khỏi giá trị cực đại của số mũ âm có thể.
- ◼ Tràn trên phần định trị (Mantissa Overflow): cộng hai phần định trị có cùng dấu, kết quả bị nhớ ra ngoài bit cao nhất.
- ◼ Tràn dưới phần định trị (Mantissa Underflow): Khi hiệu chỉnh phần định trị, các số bị mất ở bên phải phần định trị.

+

## b. Phép nhân dấu chấm động

- Bộ nhân phần định trị được thực hiện giống như thuật toán nhân hai số nhị phân thông thường.
- Kết quả sẽ có kích thước lớn gấp đôi, tuy nhiên giá trị sẽ được chuẩn hóa theo dạng biểu diễn

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000048_25e1645890a3bf20dc659d2b03dfd5ef9c5e38e3a4cde2f1f806b86586687e8a.png)

+

c. Phép chia dấu chấm động

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000049_c408637a3eccaee554cd4a9bd909ca1ed962f138b2fb572d678dddd3e1c53b02.png)

## +

## Nhân và chia dấu chấm động

Các bước cơ bản

1. Kiểm tra 0
2. Cộng/trừ số mũ (có thể xảy ra tràn → kiểm tra tràn mũ)
3. Nhân/chia các định trị
4. Chuẩn hóa
5. Làm tròn

## +

## Câu hỏi chương 9

1. Giải thích ngắn gọn về các biểu diễn: dấu -độ lớn, bù 2.
2. Cách xác định một số là âm hay dương trong các biểu diễn: dấu -độ lớn, bù 2.
3. Nguyên tắc mở rộng phạm vi biểu diễn số cho biểu diễn bù 2 là gì?
4. Cách đảo một số nguyên trong biểu diễn bù 2?
5. Phân biệt biểu diễn bù 2 của một số và bù 2 của một số?
6. Nếu coi 2 số bù 2 như là số nguyên không dấu khi thực hiện cộng, kết quả hiểu theo số bù 2 là chính xác. Điều này không đúng với phép nhân. Tại sao?
7. Bốn thành phần của một số trong biểu diễn dấu chấm động là gì?
8. Vì sao sử dụng biểu diễn bias cho số mũ của một số dấu chấm động?
9. Phân biệt tràn số mũ, và tràn định trị?
10. Các yếu tố cơ bản của phép cộng và trừ dấu chấm động là gì?

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH09-ALU_artifacts/image_000050_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)