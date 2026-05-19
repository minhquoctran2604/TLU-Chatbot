+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000000_d755dd23eae31a9f62798c3a9b9c229b8445e850aedde30589b9b0ca74089b29.png)

## Chương 12

Chức năng và cấu trúc Vi xử lý

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000001_b8eb0923bd17d7163e0d6bec6d87c16bc580eedd820ebf12c6c183e0a64dc0b6.png)

## + Nội dung

1. Tổ chức của Bộ xử lý
2. Tổ chức thanh ghi
3. Chu kỳ lệnh
4. Kỹ thuật đường ống lệnh (Pipelining)
5. Kiến trúc VXL tiên tiến

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000002_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## 12.1 Tổ chức bộ vi xử lý

## Các yêu cầu xử lý:

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000003_44e1f972994e528dfe86d8113ae27e139d2ac92b6c7e60f8efba138cc09b2aea.png)

- ◼ Truy xuất lệnh: Bộ xử lý đọc lệnh từ bộ nhớ (thanh ghi, bộ nhớ cache, bộ nhớ chính).
- ◼ Giải mã lệnh: Lệnh được giải mã để xác định hành động nào được yêu cầu.
- ◼ Truy xuất dữ liệu: Việc thực thi một lệnh có thể yêu cầu đọc dữ liệu từ bộ nhớ hoặc một module vào/ra
- ◼ Xử lý dữ liệu: Việc thực thi một lệnh có thể yêu cầu thực hiện một số phép toán số học hoặc logic trên dữ liệu.
- ◼ Ghi dữ liệu: Kết thúc việc thực hiện có thể yêu cầu ghi dữ liệu vào bộ nhớ hoặc một module vào/ra.

Để thực hiện những việc này, bộ vi xử lý cần lưu tạm thời một số dữ liệu → cần một bộ nhớ nhỏ bên trong → thanh ghi

## + Tổ chức VXL

- ◼ ALU: khối tính toán số học và logic
- ◼ CU: khối điều khiển: kiểm soát việc di chuyển dữ liệu và lệnh vào và ra khỏi bộ xử lý
ềể và điều khiển hoạt động của ALU
- ◼ Các thanh ghi: lưu trữ dữ liệu tạm thời trong quá trình lệnh được thực hiện

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000004_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000005_753c09671fae8b1bf7878d89b1bf9429c845e4d99dd2aceb9275e1bce56c4f2c.png)

## Cấu trúc bên trong CPU

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000006_b1194883dd45364377ea09ea129b0bd7e562946e9144a8d45c4661d97f92f3ce.png)

+

## 12.2 Tổ chức thanh ghi

- ◼ Các thanh ghi là một loại bộ nhớ.
- ◼ Vai trò của thanh ghi:

## Thanh ghi hiển thị với người dùng

- ◼ Cho phép người lập trình ngôn ngữ assembly hoặc ngôn ngữ máy sử dụng trong các câu lệnh
- ◼ VD: ADD AX, 300: lấy dữ liệu ở ngăn nhớ 300 cộng với AX 
ế gg 
và ghi kết quả vào AX
- ◼ Giảm thiểu các tham chiếu bộ nhớ chính bằng cách sử dụng thanh ghi

## Thanh ghi điều khiển và trạng thái

- ◼ Được sử dụng bởi CU để điều khiển hoạt động của bộ vi xử lý và bởi các chương trình hệ điều hành được đặc quyền (privileged) để kiểm soát việc thực thi chương trình
- ◼ VD: thanh ghi PC chứa địa chỉ lệnh tiếp theo của chương trình

## a. Thanh ghi hiển thị với người dùng

- Là các thanh ghi lập trình viên có thể sử dụng trong các lệnh để phục vụ cho mục đích viết chương trình của mình
- Phân loại
- Thanh ghi đa năng: lập trình viên có thể sử dụng các thanh ghi nhóm này cho nhiều mục đích khác nhau
- Thanh ghi dữ liệu: sử dụng để chứa dữ liệu và không dùng để tính toán địa chỉ toán hạng .
- Thanh ghi địa chỉ: có thể là thanh ghi đa năng hoặc là thanh ghi dành riêng cho một chế độ địa chỉ cụ thể .
- VD: thanh ghi SP (con trỏ đoạn), thanh ghi index, thanh ghi SP (con trỏ ngăn xếp)
- Mã điều kiện
- Còn gọi là bit cờ
- Là các bit do phần cứng của bộ xử lý đặt theo kết quả của hoạt động

+

## b. Thanh ghi điều khiển và trạng thái

## Bốn thanh ghi cần thiết để thực thi lệnh:

- ◼ Thanh ghi PC - Bộ đếm chương trình
- ◼ Chứa địa chỉ của lệnh sắp được truy xuất
- ◼ Thanh ghi IR – thanh ghi lệnh
- ◼ Chứa lệnh đang được truy xuất
- ◼ Thanh ghi MAR – thanh ghi địa chỉ bộ nhớ
- ◼ Chứa địa chỉ của một vị trí trong bộ nhớ
- ◼ Thanh ghi MBR (hoặc MDR) – thanh ghi đệm bộ nhớ
- ◼ Chứa một từ dữ liệu sắp được ghi vào bộ nhớ hoặc từ vừa được đọc ra từ BN
- ◼ Một số BXL còn có một hoặc nhiều thanh ghi PSW (program status 
ủ word -gpg
từ trạng thái chương trình): chứa thông tin trạng thái của chương trình đang được thực hiện

+

## b. Thanh ghi điều khiển và trạng thái

## Thanh ghi PSW – Thanh ghi trạng thái chương trình

- ◼ Thanh ghi hoặc tập hợp thanh ghi chứa thông tin trạng thái và mã 
ề g
điều kiện
- ◼ Các trường hoặc cờ phổ biến gồm:
- ◼ Sign: Chứa bit dấu của kết quả của phép tính số học cuối cùng.
- ◼ Zero: Thiết lập khi kết quả bằng 0.
- ◼ Carry: Thiết lập nếu một phép tính có nhớ (phép cộng) hoặc vay 
ử yp pp (pp g) y 
(phép trừ) vào bit có bậc lớn hơn. Được sử dụng cho các phép tính 
ố ề (pp ) 
số học nhiều từ.
- ◼ Equal: Thiết lập nếu kết quả so sánh logic là bằng nhau.
- ◼ Overflow: Được sử dụng để chỉ định sự tràn số học.
- ◼ Interrupt Enable/Disable: Được sử dụng để cho phép hoặc vô hiệu 
ắ p
hoá ngắt .
- ◼ Supervisor: Cho biết bộ xử lý đang thực hiện trong chế độ giám sát 
ế ố ỉ ể pộ ý g ựệg ộ g
hay chế độ người dùng. Một số lệnh privileged chỉ có thể được thực 
ế ố ỉ ể y ộ ggộệpgợự
hiện trong chế độ giám sát, và một số vùng bộ nhớ chỉ có thể được 
ế ệg ộ g, 
truy cập trong chế độ giám sát.

## + c. Ví dụ tổ chức thanh ghi MC68000, Intel 8086, Intel 80386

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000007_c17f058a815f370fac0107dd2022490cfd5a3b06066ba14f0a3d82850b2bbaee.png)

## 12.3 Chu kỳ lệnh

## Bao gồm các tầng sau:

- Truy xuất
- Đọc lệnh tiếp theo từ bộ nhớ vào bộ vi xử lý
- Thực thi
- Giải mã opcode và thực hiện hoạt động được chỉ định
- Ngắt
- Nếu có yêu cầu ngắt được gửi đến, VXL lưu trạng thái hiện tại của chương trình và chuyển sang phục vụ ngắt

## Sơ đồ trạng thái chu kỳ lệnh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000008_218ce2a398d9d016e70cf45dc65bc48551e9522320138bbdfd697abce65fd8fe.png)

## Luồng dữ liệu, chu kỳ truy xuất

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000009_a43cdf00727ba0ab001bfe8ea632ef3a34bd384ba876611f6b852f48c4baaa36.png)

## Luồng dữ liệu chu kỳ ngắt

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000010_441a0bba455eb09938004f1c31601c36ee32759758abfc89b895eb4a1f66fa38.png)

Sơ đồ trạng thái chu kỳ lệnh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000011_4551b83a1736715d27380f473c71cff7bf1f059b0347f7808000a918a01e92a1.png)

## Bài 3. Luồng xử lý (Pipelining) lệnh

- ❖ Chu kỳ lệnh được chia thành 6 giai đoạn:
- Truy xuất lệnh (FI – – Fetch instruction): Đọc lệnh tiếp theo vào bộ đệm.
- Giải mã lệnh (DI – Decode instruction): Giải mã opcode và nhận diện toán hạng.
- Tính toán địa chỉ toán hạng (CO – Calculate operands): Tính toán địa chỉ hiệu dụng của từng toán hạng nguồn: địa chỉ dịch chuyển, gián tiếp thanh ghi, gián tiếp .v.v... .
- Truy xuất toán hạng (FO – – Fetch operands): Truy xuất từng toán hạng từ bộ nhớ. Không cần truy xuất toán hạng từ thanh ghi
- Thực thi lệnh (EI – Execute instruction): Thực hiện hành động và lưu trữ kết quả (nếu có) trong vị trí toán hạng đích đã định.
- Ghi toán hạng (WO – Write operand): Lưu kết quả vào bộ nhớ.

## Biểu đồ thời gian của pipeline lệnh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000012_2e1efbef80d0f877ff99d24abbde85e49e48e6520671b73bdf1d8d2ccb85763d.png)

Kỹ

-

Chia

-

Thực

## Ví dụ: kỹ thuật đường ống lệnh trong trường hợp câu lệnh rẽ nhánh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000013_a488e69dc2ca88d0615e19dc5ba654dfc36b4fbb70debcb7d6a61de6265d161a.png)

Vấn

ống

Xung a=a+3

b=a+1

## Mô tả khác về Pipeline

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000014_cf4efa41cab1217455be73d4067c2dc2a281f4dbb2cc8207d09efc9964facda4.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000015_f49d81d8f7dc97436ae3ed20ecb6106c4c1636bb6a1a30d317f0395cc16c11a1.png)

## Xung đột trong kỹ thuật đường ống (Pipeline Hazard)

- Trong một số trường hợp , kỹ thuật đường ống bị đình trệ do một số xung đột như sau:
- Xung đột tài nguyên: do nhiều công đoạn dùng chung một tài nguyên .
- Xung đột dữ liệu: lệnh sau sử dụng kết quả của lệnh trước
- (một bộ phận phần cứng được dùng để đưa kết quả từ đầu ra ALU trực tiếp vào một trong các thanh ghi đầu vào)
- Xung đột điều khiển: do rẽ nhánh gây ra (đóng băng kỹ thuật ống dẫn trong một chu kỳ)

Vấn

ống

Xung a=a+3

b=a+1

cin&gt;&gt;a:

Cin&gt;&gt;b:

## Xung đột tài nguyên

- ◼ Hazard tài nguyên xảy ra khi hai hoặc nhiều lệnh đã ở trong đường ống cần dùng cùng một tài nguyên
- ◼ VD: Lệnh I1 truy xuất toán hạng từ bộ nhớ → xung đột với việc truy xuất lệnh I3 → việc truy xuất lệnh I3 phải chậm lại 1 chu kỳ (hình b)
- ◼ Còn được gọi là Hazard cấu trúc

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000016_adb01fd851ad01fcb7dddd9da693b923b4a40c75f494f03c5b7834b6fddc38c6.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000017_ababf719695691cf7e90ff6e3f30f4c5c358eabaf4740d185b61aafac970a1a8.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000018_cca50b8959e437f8619b5325f4c43d9c58a8464467aad5a8cf386b02c5ea8ec1.png)

+

## Xung đột dữ liệu

Câu lệnh I2 sử dụng kết quả của câu lệnh I1 (EAX): việc truy xuất toán hạng 
ồồ ể g q()y 
phải chậm lại 2 chu kỳ đồng hồ để đợi câu lệnh I1 thực hiện xong việc ghi toán hạng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000019_a9bc3ecbe929dd1432789eef727d7fdb309b213672fbea8ddb07adc333d02b78.png)

## Hazard

## + Các loại Hazard dữ liệu

- ❖Đọc sau khi ghi (RAW)
- ❖ Một lệnh sửa đổi một thanh ghi hoặc vị trí bộ nhớ
- ❖ Lệnh tiếp theo đọc dữ liệu trong bộ nhớ hoặc vị trí thanh ghi
- ❖ Hazard xảy ra nếu việc đọc diễn ra trước khi hoạt động ghi hoàn tất
- ❖Ghi sau khi đọc (WAR)
- ❖ Một lệnh đọc một thanh ghi hoặc vị trí bộ nhớ
- ❖ Lệnh tiếp theo ghi vào vị trí đó
- ❖ Hazard xảy ra nếu thao tác ghi hoàn thành trước khi có thao tác đọc
- ❖Ghi sau khi ghi (WAW)
- ❖ Hai lệnh cùng ghi vào 1 vị trí
- ❖ Hazard xảy ra nếu các thao tác ghi diễn ra theo thứ tự ngược với trình tự dự định

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000020_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## + Xung đột điều khiển

- ❖Còn được gọi là Hazard rẽ nhánh
- ❖Xảy ra khi kỹ thuật đường ống đưa ra dự báo nhánh bị sai so với nhánh thực tế
- ❖ Các lệnh được truy xuất sẽ bị loại bỏ.
- ❖ Các biện pháp đối phó với Hazard rẽ nhánh
- Sử dụng nhiều luồng
- Truy xuất trước mục tiêu rẽ nhánh
- Bộ đệm vòng lặp
- Dự báo rẽ nhánh
- Rẽ nhánh chậm

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000021_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000022_39e8dc1ab9f18dc04f5bc0275e1a75bafcdda8117bf4d278643795a82482815a.png)

## Bài 4. Kiến trúc VXL tiên tiến (đọc thêm)

1. Cấu trúc chung của các bộ xử lý tiên tiến
2. Các kiến trúc song song mức lệnh
3. Kiến trúc RISC

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000023_19c72cce06478e295e2200b6bbe9d3dee681282726b587705399af520098f635.png)

## 1. Cấu trúc chung của các BXL tiên tiến

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000024_f8392443b6c3c0c134bfaba913e6b3710bbf873c3dc82285d9fb24cdfc6c81e6.png)

## Các đơn vị xử lý dữ liệu

- ❖ Các đơn vị số nguyên (ALU -Integer Unit – IU)
- ❖ Các đơn vị số dấu chấm động (Floating Point Unit – FPU)
- ❖ Các đơn vị chức năng đặc biệt (SFU):
- ❖ Đơn vị xử lý dữ liệu âm thanh
- ❖ Đơn vị xử lý dữ liệu hình ảnh
- ❖ Đơn vị xử lý dữ liệu vector

## Bộ nhớ cache

- ◼ Được tích hợp trên chip vi xử lý
- ◼ Thường bao gồm 2 mức Cache:
- ◼ Cache L1 gồm 2 phần tách rời:
- ◼ Cache lệnh
- ◼ Cache dữ liệu
- → Giải quyết xung đột khi nhận lệnh và dữ liệu
- ◼ Cache L2: dùng chung cho lệnh và dữ liệu

## 2. Kiến trúc song song mức lệnh

- ❖ Siêu đường ống (Superpipeline và Hyperpipeline)
- ❖ Siêu vô hướng (Superscalar)
- ❖ Từ lệnh dài – VLIW (Very Long Instruction Word)

## Đơn vị quản lý bộ nhớ

- ❖ Chuyển đổi địa chỉ ảo thành địa chỉ vật lý
- ❖ Cung cấp cơ chế phân trang hoặc phân đoạn
- ❖ Cung cấp chế độ bảo vệ bộ nhớ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000025_e311e5e5dc38621bb5155eee4393cf4c2d76e7db4dac9ffc53cefb88eaa873d2.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000026_04727c988669b91a68f84473310f32ab535f98db93c152b579f8b0f9c65e3a3d.png)

## 3. Kiến trúc RISC

- ◼ CISC và RISC:
- ◼ CISC – – Complex Instruction Set Computer:
- ◼ Máy tính có tập lệnh phức tạp
- ◼ VD: các bộ xử lý 80x86 ...
- ◼ RISC – – Reduced Instruction Set Computer:
- ◼ Máy tính có tập lệnh rút gọn
- ◼ VD: các bộ xử lý Sun SPARC, Power PC, ...

## Các đặc trưng của RISC

- ❖ Số lượng lệnh ít
- ❖Các lệnh có thời gian thực hiện là 1 chu kỳ máy
- ❖ Độ dài của các lệnh bằng nhau (32 bit)
- ❖Có ít khuôn dạng lệnh (≤ 4)
- ❖ Có ít chế độ địa chỉ hóa toán hạng (≤ 4)
- ❖ Có nhiều thanh ghi
- ❖ Các lệnh chủ yếu là thao tác giữa thanh ghi với thanh ghi
- ❖Truy cập bộ nhớ thông qua 2 lệnh LOAD và STORE

## + Vi xử lý ARM

## ARM cơ bản là hệ thống RISC với các đặc điểm sau:

- ❖ Moderate array of uniform registers
- ❖ Mô hình tải / lưu trữ của quá trình xử lý dữ liệu trong đó các phép toán chỉ thực hiện trên toán hạng trong thanh ghi mà không thực hiện trực tiếp trong bộ nhớ
- ❖ Lệnh có độ dài cố định 32 bit cho tập lệnh tiêu chuẩn và 16 bit cho tập lệnh Thumb
- ❖ ALU và shifter riêng biệt
- ❖ Có ít chế độ định địa chỉ. Tất cả các địa chỉ tải / lưu trữ được xác định từ thanh ghi và trường lệnh
- ❖ Các chế độ định địa chỉ tự động tăng và tự động giảm được sử dụng để cải thiện hoạt động của vòng lặp của chương trình
- ❖ Việc thực hiện lệnh có điều kiện giúp giảm thiểu nhu cầu lệnh rẽ nhánh điều kiện, do đó nâng cao hiệu quả pipeline

## Tổ chức ARM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000027_4c897a502a15fd97b0b1f72ed3fdb2fda3b903394078076b575e1017f5cea5e1.png)

## Chế độ bộ xử lý

- ❖ Kiến trúc ARM hỗ trợ bảy chế độ thực hiện
- ❖ Hầu hết chương trình ứng dụng thực thi trong chế độ người dùng
- ❖ Trong khi bộ xử lý đang ở chế độ người dùng , chương trình đang được thực thi không thể truy cập vào tài nguyên hệ thống được bảo vệ hoặc thay đổi chế độ, chỉ có thể tạo ra một exception
- ❖ Sáu chế độ thực hiện còn lại được coi là chế độ ưu đãi (privileged mode)
- ❖ Các chế độ này được sử dụng để chạy phần mềm hệ thống
- ❖ Thuận lợi khi có nhiều chế độ privileged khác nhau
- ❖ Hệ điều hành có thể điều chỉnh việc sử dụng phần mềm hệ thống cho nhiều tình huống
- ❖ Một số thanh ghi được dành riêng để dùng trong mỗi chế độ privileged, cho phép thay đổi nhanh hơn trong

ngữ cảnh

## Chế độ Exception

- ❖ Có quyền truy cập đầy đủ vào tài nguyên hệ thống và có thể thoải mái thay đổi chế độ
- ❖ Khi xảy ra các exception
- ❖ Các chế độ Exception:
- ❖ Supervisor mode – giám sát
- ❖ Abort mode – huỷ bỏ
- ❖ Undefined mode – không xác định
- ❖ Fast interrupt mode – gián đoạn nhanh
- ❖ Interrupt mode – gián đoạn
- ❖ Chế độ hệ thống:
- ❖ Không vào được bởi bất kỳ exception nào và sử dụng cùng các thanh ghi có sẵn trong Chế độ người dùng
- ❖ Được dùng để chạy một số tác vụ hệ điều hành privileged
- ❖ Có thể bị gián đoạn bởi một trong năm loại chế độ Exception

## Tổ chức thanh ghi ARM

## Format of ARM CPSR and SPSR

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000028_bdc29ff346227a2d6a91d1e8aae6109c0ab11dab18f7f735dc5ad525a0429f9a.png)

## Table 14.4

## ARM Interrupt Vector

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000029_39e8dc1ab9f18dc04f5bc0275e1a75bafcdda8117bf4d278643795a82482815a.png)

## Tổng kết chương

- ◼ Tổ chức bộ xử lý
- ◼ Tổ chức thanh ghi
- ◼ Thanh ghi hiển thị người dùng
- ◼ Thanh ghi điều khiển và trạng thái
- ◼ Chu kỳ lệnh
- ◼ Chu kỳ gián tiếp
- ◼ Luồng dữ liệu
- ◼ Họ vi xử lý x86
- ◼ Tổ chức thanh ghi
- ◼ Xử lý gián đoạn

## Cấu trúc và Chức năng Bộ xử lý

- ◼ Pipelining lệnh
- ◼ Chiến lược Pipelining
- ◼ Hiêu suất Pipeline
- ◼ Pipeline hazard
- ◼ Xử lý rẽ nhánh
- ◼ Pipelining Intel 80486
- ◼ Bộ xử lý ARM
- ◼ Tổ chức bộ xử lý
- ◼ Chế độ bộ xử lý
- ◼ Tổ chức thanh ghi
- ◼ Xử lý gián đoạn

## 12.4 Kỹ thuật đường ống Pipelining

- Chu kỳ lệnh được chia thành 6 giai đoạn:
- Truy xuất lệnh (FI – Fetch instruction): Đọc lệnh tiếp theo vào bộ đệm.
- Giải mã lệnh (DI – Decode instruction): Giải mã opcode và nhận diện toán hạng.
- Tính toán địa chỉ toán hạng (CO – Calculate operands): Tính toán địa chỉ hiệu dụng của từng toán hạng nguồn: địa chỉ dịch chuyển, gián tiếp thanh ghi, gián tiếp .v.v... .
- Truy xuất toán hạng (FO – Fetch operands): Truy xuất từng toán hạng từ bộ nhớ. Không cần truy xuất toán hạng từ thanh ghi
- Thực thi lệnh (EI – Execute instruction): Thực hiện hành động và lưu trữ kết quả (nếu có) trong vị trí toán hạng đích đã định.
- Ghi toán hạng (WO – Write operand): Lưu kết quả vào bộ nhớ.

## Biểu đồ thời gian của pipeline lệnh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000030_ddd84dda0c57bacebb5a41807f54acef3181f27469dc1c01e30ddf2274e3fcfc.png)

## Ví dụ: kỹ thuật đường ống lệnh trong trường hợp câu lệnh rẽ nhánh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000031_d39eeaafcb1de21b47a24f17252d4836cbbb99af74473a9d4abcdfa137a30282.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000032_98a47ca546851b0e9b09d1e020da3b05f453d5110e06a6be8a74e5beb46dde94.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000033_93a2cee3a6e77ef54814a70701ca9fb5fb47bd619767e42bee4e1ce71f7ca32a.png)

## Mô tả khác về Pipeline

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000034_cf4efa41cab1217455be73d4067c2dc2a281f4dbb2cc8207d09efc9964facda4.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000035_f49d81d8f7dc97436ae3ed20ecb6106c4c1636bb6a1a30d317f0395cc16c11a1.png)

## Xung đột trong kỹ thuật đường ống (Pipeline Hazard)

- Trong một số trường hợp, kỹ thuật đường ống bị đình trệ do một số xung đột như sau:
- Xung đột tài nguyên: do nhiều công đoạn dùng chung một tài nguyên.
- Xung đột dữ liệu: lệnh sau sử dụng kết quả của lệnh trước (một bộ phận phần cứng được dùng để đưa kết quả từ đầu ra ALU trực tiếp vào một trong các thanh ghi đầu vào)
- Xung đột điều khiển: do rẽ nhánh gây ra (đóng băng kỹ thuật ống dẫn trong một chu kỳ)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000036_3567977f062b619e3e606f0d1fa090534a2ac2f61f992ed8b4196086a5f57910.png)

## Xung đột tài nguyên

- ◼ Hazard tài nguyên xảy ra khi hai hoặc nhiều lệnh đã ở trong đường ống cần dùng cùng một tài nguyên
- ◼ VD: Lệnh I1 truy xuất toán hạng từ bộ nhớ → xung đột với việc truy xuất lệnh I3 → việc truy xuất lệnh I3 phai chậm lại 1 chu kỳ (hình b)
- ◼ Còn được gọi là Hazard cấu trúc

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000037_9f5a6e26211f5d6ec514842338d71c8f4e046ccbe451682dfeecb78f5435cd57.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000038_75ffb6114b5a7bdb7f0e6f6e89a0f0e9f39d348878ffaec26347a90ce1da6a42.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000039_cca50b8959e437f8619b5325f4c43d9c58a8464467aad5a8cf386b02c5ea8ec1.png)

+

## Xung đột dữ liệu

Câu lệnh I2 sử dụng kết quả của câu lệnh I1 (EAX): việc truy xuất toán hạng 
ồồ ể g q()y 
phải chậm lại 2 chu kỳ đồng hồ để đợi câu lệnh I1 thực hiện xong việc ghi toán hạng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000040_a9bc3ecbe929dd1432789eef727d7fdb309b213672fbea8ddb07adc333d02b78.png)

## Hazard

+

## Các loại Hazard dữ liệu

- ◼ Đọc sau khi ghi (RAW)
- ◼ Một lệnh sửa đổi một thanh ghi hoặc vị trí bộ nhớ
- ◼ Lệnh tiếp theo đọc dữ liệu trong bộ nhớ hoặc vị trí thanh ghi
- ◼ Hazard xảy ra nếu việc đọc diễn ra trước khi hoạt động ghi hoàn tất
- ◼ Ghi sau khi đọc (WAR)
- ◼ Một lệnh đọc một thanh ghi hoặc vị trí bộ nhớ
- ◼ Lệnh tiếp theo ghi vào vị trí đó
- ◼ Hazard xảy ra nếu thao tác ghi hoàn thành trước khi có thao tác đọc
- ◼ Ghi sau khi ghi (WAW)
- ◼ Hai lệnh cùng ghi vào 1 vị trí
- ◼ Hazard xảy ra nếu các thao tác ghi diễn ra theo thứ tự ngược với trình tự dự định

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000041_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

## + Xung đột điều khiển

- ◼ Còn được gọi là Hazard nhánh
- ◼ Xảy ra khi kỹ thuật đường ống đưa ra dự báo nhánh bị sai so với nhánh thực tế
- ◼ Các lệnh được truy xuất sẽ bị loại bỏ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000042_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000043_39e8dc1ab9f18dc04f5bc0275e1a75bafcdda8117bf4d278643795a82482815a.png)

## + 12.5 Kiến trúc VXL tiên tiến

- a. Cấu trúc chung của các bộ xử lý tiên tiến
- b. Các kiến trúc song song mức lệnh
- c. Kiến trúc RISC

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000044_78e2a609f0f350844ddcacb7621fc61af005363411abeffb1add2c703d9d7f9a.png)

## 12.5 Kiến trúc VXL tiên tiến

## a. Cấu trúc chung của các BXL tiên tiến

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH12-CPU_artifacts/image_000045_e6404dde1e2be3754f65d5cbc8cf8137b3faeabed84c14ccf4e6d9ba7ffec770.png)

## 12.5 Kiến trúc VXL tiên tiến

## Các đơn vị xử lý dữ liệu

- ◼ Các đơn vị số nguyên (Integer Unit – – IU)
- ◼ Các đơn vị số dấu chấm động (Floating Point Unit – FPU)
- ◼ Các đơn vị chức năng đặc biệt:
- ◼ Đơn vị xử lý dữ liệu âm thanh
- ◼ Đơn vị xử lý dữ liệu hình ảnh
- ◼ Đơn vị xử lý dữ liệu vector

## 12.5 Kiến trúc VXL tiên tiến

## Bộ nhớ cache

- ◼ Được tích hợp trên chip vi xử lý
- ◼ Thường bao gồm 2 mức Cache:
- ◼ Cache L1 gồm 2 phần tách rời:
- ◼ Cache lệnh
- ◼ Cache dữ liệu
- → Giải quyết xung đột khi nhận lệnh và dữ liệu
- ◼ Cache L2: dùng chung cho lệnh và dữ liệu

## 12.5 Kiến trúc VXL tiên tiến

## Đơn vị quản lý bộ nhớ

- ◼ Chuyển đổi địa chỉ ảo thành địa chỉ vật lý
- ◼ Cung cấp cơ chế phân trang hoặc phân đoạn
- ◼ Cung cấp chế độ bảo vệ bộ nhớ