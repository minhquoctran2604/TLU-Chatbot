![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000000_45b4415caa5e0e7d5ca85acebdbbb08c592894feb5ccbc1360ac5f4ed6d44a6c.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000001_4aca9640898fb31cf1590328b5be155873419007652a80d92f7211e7afc9638a.png)

Kiến trúc máy tính Bộ môn Kỹ thuật máy tính và mạng

+

## Chương 6. Bộ nhớ ngoài

- 6.1. Đĩa từ
- 6.2. RAID
- 6.3. Ổ cứng trạng thái rắn (ổ cứng bán dẫn)
- 6.4. Bộ nhớ quang
- 6.5. Băng từ

+

## Các loại bộ nhớ ngoài

- ◼ Đĩa từ:
- ◼ Ổ cứng
- ◼ Có thể được tổ chức dưới dạng một mảng nhiều đĩa từ để có hiệu quả lưu trữ tốt hơn: RAID
- ◼ Ổ cứng trạng thái rắn
- ◼ Ổ cứng SSD
- ◼ Đĩa quang
- ◼ Băng từ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000002_6362fca6cc9e37e53262a65153619782c907ba3fe3ca8c078776c6a7b067f13c.png)

+

## 6.1. Đĩa từ

- ◼ Đĩa từ là một tấm platter tròn chế tạo bằng vật liệu không từ tính, được gọi là chất nền (substrate), được phủ một lớp vật liệu từ tính
- ◼ Chất nền thường là vật liệu nhôm hoặc hợp kim nhôm
- ◼ Gần đây người ta đưa chất nền thủy tinh
- ◼ Ưu điểm của chất nền thủy tinh:
- ◼ Cải thiện tính đồng nhất của bề mặt phim từ để tăng độ tin cậy của đĩa
- ◼ Giảm đáng kể các khiếm khuyết bề mặt để giúp giảm lỗi đọc -ghi
- ◼ Cho phép khoảng cách đầu đọc và bề mặt gần hơn
- ◼ Độ cứng tốt hơn nên giảm động lực đĩa
- ◼ Khả năng chống sóc và hư hỏng lớn hơn

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000003_6b50ba6d47d2f6f8b774e8412f65cf46b30ebb31d7e033fc80a50191a1c965db.png)

## + a. Cơ chế đọc – – ghi từ

- Dữ liệu được ghi vào, sau đó được lấy ra thông qua một cuộn dây dẫn gọi là đầu (head)
- Nhiều ổ đĩa thường thiết kế dạng 2 đầu: đầu đọc và đầu ghi riêng
- Trong quá trình đọc hoặc ghi, đầu đứng yên trong khi đĩa xoay bên dưới

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000004_7e5c362f3a1506c3bd15ef4651906c3d5955d4c098a390ade61181f36e31e1a6.png)

## + ▪ Cơ chế ghi

- o Dựa trên hiện tượng dòng điện chạy trong vòng dây tạo ra từ trường trên khoảng trống, từ đó từ hoá một vùng nhỏ của bề mặt ghi (bề mặt từ tính)
- o Đảo chiều dòng điện sẽ đảo chiều hướng từ hóa trên bề mặt ghi (bit 0 hoặc 1)
- Đầu ghi được làm bằng vật liệu từ hoá và có dạng hình chữ
- o nhật với một khoảng trống dọc một cạnh và một vài vòng dây dẫn ở dọc cạnh đối diện

## ▪ Cơ chế đọc

- o Khi bề mặt đĩa đi qua đầu (head), từ trường được phân cực ở bề mặt đĩa sinh ra dòng điện . Hướng từ hóa (ứng với các bit 0 hoặc 1) khác nhau sinh ra dòng điện có chiều khác nhau
- o Đầu đọc giống với đầu ghi nên chúng có thể sử dung chung (vd: đĩa mềm)
- o Tuy nhiên, một số ổ cứng người ta dùng đầu đọc – – ghi riêng cho phép hoạt động với tần số cao hơn và mật độ dữ liệu lớn hơn

## + b. Bố trí dữ liệu và định dạng dữ liệu trên đĩa

- ◼ Dữ liệu được bố trí thành các vòng trên platter (gọi là các track). Độ rộng của 
ằ p(gọ)ộ 
track bằng độ rộng của head
- ◼ Các track ngăn cách bởi một rãnh (gap) 
ể ế gộ(gp) 
để sự ảnh hưởng của track này đến track 
ỗ ự 
khác gây là lỗi
- ◼ Các track được chia ra thành các sector. Có hàng trăm sector trên một track.
- ◼ Dữ liệu được ghi vào và đọc ra từ các 
ể ố ệợgọ
sector. Sector có thể có kích thước cố định hoặc thay đổi. Tuy nhiên, thông 
ố ịặy y , g 
thường sector có kích thước cố định và 
ằ g 
bằng 512 byte.
- ◼ Giữa các sector cũng được ngăn cách bởi các rãnh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000005_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000006_2a2e28ae678be9f684a9c8f86aba8bee76c740fb457a266eba65802334fc7c4a.png)

+

## Sơ đồ phương pháp bố trí đĩa

- ◼ Có hai phương pháp bố trí đĩa
- ◼ Vận tốc góc không đổi (CAV -constant angular velocity)
- ◼ Ghi nhiều vùng (multiple zone recording)
- ◼ Vận tốc góc không đổi
- ◼ Số lượng sector trên các track bằng nhau
- ◼ Đĩa quay với vận tốc góc không đổi
- ◼ Truy cập dữ liệu: đầu đọc di chuyển 
ế y p y
đến track chứa dữ liệu và chờ cho đến khi sector đó quay đến
- ◼ Nhược điểm: số lượng sector ở các 
ằ g 
track bên ngoài (dài hơn) bằng các 
ắ g() 
track bên trong (ngắn hơn)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000007_ae7c0696aae2a3bae64ddb93a6602c55120506d546d31e3be5b390c6d157372d.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000008_655ed00188cb4cb0dfbcc2819fac183aa69121fd7489bce2988f145b0fcb7951.png)

## + Sơ đồ phương pháp bố trí đĩa (tiếp)

- ◼ Ghi nhiều vùng
- ◼ Chia bề mặt thành nhiều vùng (zone) vành khăn (vùng đồng tâm) (thường là 16 vùng)
- ◼ Trong một zone, số sector trên mỗi track bằng nhau
- ◼ Các zone càng xa thì càng có nhiều sector hơn zone trung tâm.
- ◼ Dung lượng lưu trữ lớn hơn CAV
- ◼ Nhược điểm:
- ◼ Mạch điện phức tạp hơn.
- ◼ Thời gian đọc/ghi dữ liệu trên các track nằm trong zone khác nhau thì khác nhau

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000009_444e65acc6e146139d3e076ca6d6a46204d8d3a084330e9b1c257804ea98db98.png)

## + Định dạng dữ liệu trên đĩa

- ◼ Một track thường có định dạng như sau (ví dụ với track có 30 sector):

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000010_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000011_b4a70f53734110b34821e16c58452a6bb95a86b588160002ea8dcf045e3530e8.png)

## + Định dạng dữ liệu trên đĩa (tiếp)

- ◼ Khi truy cập (đọc hoặc ghi) dữ liệu trên đĩa: đầu đọc sẽ được đặt ở vị trí đầu tiên của track
- ◼ Mỗi sector (kích thước 600 byte) chứa:
- ◼ 512 byte dữ liệu, còn lại là thông tin điều khiển.
- ◼ Trường ID: địa chỉ hoặc các thông tin để xác định 1 sector duy nhất
- ◼ Synch byte: đánh dấu điểm bắt đầu một trường
- ◼ Track number: xác định một track
- ◼ Sector number: xác định một sector
- ◼ CRC: mã sửa lỗi
- ◼ Các thông tin điều khiển chỉ được đọc và sử dụng bởi ổ đĩa, không được gửi ra ngoài.

+

## c. Đặc tính vật lý của hệ thống đĩa

- ◼ Chuyển động đầu
- -Đầu cố định
- -Đầu di chuyển
- ◼ Tính di động của đĩa
- -Đĩa không tháo được
- -Đĩa tháo được
- ◼ Mặt
- -1 mặt
- -2 mặt

## ◼ Tấm platter

- -Đơn tấm
- -Đa tấm
- ◼ Cơ chế đầu đọc/ghi
- -Tiếp xúc (đĩa mềm)
- -Rãnh cố định
- -Rãnh khí động học (Winchester)

+

## Chuyển động đầu

- ◼ Đĩa có đầu cố định
- ◼ Một đầu đọc -ghi cho mỗi track
- ◼ Tất cả các đầu được gắn trên một cánh tay cố định kéo dài trên toàn bộ các track
- ◼ Đĩa có đầu di chuyển
- ◼ Một đầu đọc -ghi
- ◼ Đầu được gắn trên một cánh tay
- ◼ Cánh tay có thể kéo dài hoặc rút ngắn được để đặt vào tất cả các track

## Tính di động của đĩa

- ◼ Đĩa không tháo được
- ◼ Gắn cố định vào ổ đĩa
- ◼ Đĩa cứng trong máy tính cá nhân là đĩa không tháo được
- ◼ Đĩa tháo được
ể
- ◼ Ưu điểm:
- ◼ ợ
Có thể được gỡ ra và thay thế bằng một đĩa khác
ể
- ◼ Dữ liệu không giới hạn 
ể
- ◼ y
Ví dụ: đĩa mềm, đĩa cartridge ZIP
- ◼ g g
Đĩa có thể được di chuyển từ hệ thống máy tính này sang hệ thống khác
ề

## ◼ Đĩa hai mặt

- ◼ Lớp phủ từ tính được phủ lên cả hai mặt của tấm platter

## Đặc tính vật lý

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000012_a32e406547780dbd04faa5d4534b3e74fc2c0d672088e50288d3bf09bd374cc6.png)

## Đặc tính vật lý (tiếp)

Platter: đơn tấm hoặc đa tấm

- ◼ Đơn tấm: một ổ đĩa chỉ gồm một tấm platter
- ◼ Đa tấm: một số ổ đĩa gồm nhiều tấm platter xếp chồng lên nhau. Một hệ thống gồm nhiều cánh tay có các đầu đọc/ghi cho mỗi tấm.
- ◼ Tất cả các head có cơ chế di chuyển cố định, cùng nhau. Tại cùng một thời điểm các head sẽ được đặt vào các track có cùng khoảng cách với tâm đĩa
- ◼ Tập các track như vậy được gọi là cylinder

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000013_7b5815d3776306b35a280df871bcbad2269c7afa6231751a77cdb85ea358c7e1.png)

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000014_70da13d54c8402e3e587ce972f0594232a5e4e4e0d6af5b8bcb2b8efa945f497.png)

Tracks

Cylinders

+

## Đặc tính vật lý (tiếp)

## Cơ chế đầu đọc/ghi

- ◼ Đầu phải tạo ra hoặc cảm nhận một trường điện từ đủ lớn để ghi và đọc đúng
- ◼ Đầu càng hẹp thì càng phải đặt gần bề mặt tấm platter để đảm bảo chức năng đọc/ghi. Đầu hẹp hơn nghĩa là các đường track hẹp hơn , do đó mật độ dữ liệu lớn hơn
- ◼ Đầu càng gần đĩa thì càng nhiều nguy cơ lỗi do tạp chất hoặc không hoàn hảo

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000015_6a83ccd029ae38da3a71634e75b9020b74be6d0b9b62f2a110350dd13a5a5af1.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000016_13b7b2f909b6cd048a4648531062c2c04a6129dd68976ffa7bdcb54924d047c5.png)

+

## ◼Cơ chế hoạt động của đầu đọc/ghi: 3 loại đĩa

- ◼ Loại đĩa thứ nhất có đầu đặt cách platter một khoảng nhỏ (air gap)
- ◼ Loại thứ hai: đầu tiếp xúc với bề mặt đĩa . Đĩa mềm là loại này: dung lượng nhỏ , giá thành rẻ
- ◼ Loại thứ ba: đĩa Winchester
- ◼ Được đóng gói kín , hầu như không có chất gây ô nhiễm
- ◼ Head được thiết kế để hoạt động gần bề mặt đĩa hơn so với các đầu đĩa cứng thông thường, do đó mật độ dữ liệu lớn hơn
- ◼ Thực chất head là một tấm foil khí động học đặt trên bề mặt tấm platter
- ◼ Khi đĩa quay: áp suất không khí sinh ra sẽ nâng tấm foil lên khỏi bề mặt giúp tạo ra một khoảng cách đủ nhỏ giữa bề mặt và foil tránh tiếp xúc nhưng cho khả năng đọc/ghi tốt hơn

## Các thông số đĩa cứng điển hình

Table 6.2 Typical Hard Disk Drive Parameters

+

## d. Các tham số hiệu năng

Các tham số để đánh giá hiệu năng của ổ đĩa gồm có:

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000017_aa38652e620952bcc772e4af992209bdee03b259f3bca3fd4caa9001b1224a08.png)

- ◼ Thời gian truy nhập (access time): khoảng thời gian cần thiết để đầu (head) vào vị trí đọc/ghi (sector được đọc/ghi)
- ◼ Tổng: Thời gian tìm kiếm (seek time) và Trễ quay (rotational delay)
- ◼ Thời gian tìm kiếm (seek time): khoảng thời gian đầu đọc/ghi di chuyển đến vị trí track mong muốn (với đĩa có đầu di chuyển) hoặc lựa chọn một đầu trên cánh tay (với đĩa có đầu cố định)
- ◼ Trễ quay (rotational delay): khoảng thời gian sau khi đầu đặt ở track mong muốn đến khi đĩa quay đến sector mong muốn
- ◼ Thời gian truyền (transfer time)
- ◼ Khi đầu vào vị trí, thao tác đọc/ghi được thực hiện bằng cách đầu sẽ đọc/ghi dữ liệu vào sector quay dưới nó. Khoảng thời gian truyền dữ liệu để thực hiện thao tác này được gọi là thời gian truyền

+

## Thời gian truyền I/O của đĩa

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000018_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000019_e3deabc29a3b7991a5f5de1c87277dc4d5b385de6bef0206fd3c253dfe7079ee.png)

## + Bài tập

1. Một ổ đĩa cứng 255GB có 65.536 cylinder với 255 sector/track và 512 B/sector.
- a. Tính số lượng đĩa và số đầu đọc mà ổ đĩa này cần (đĩa hai mặt).
- b. Giả sử thời gian tìm kiếm trung bình là 11ms , thời gian trễ quay trung bình là 7ms , tốc độ đọc là 100 MBps (thời gian truyền). Tính thời gian trung bình để đọc 400KB từ đĩa .
2. Một ổ đĩa từ có 8 mặt , mỗi mặt có 512 track, 64 sector trên mỗi track. Kích thước sector là 1KB . Dung lượng của ổ đĩa là bao nhiêu?

## + 6.2. RAID

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000020_a24da61123ed0bf89d6d811e9248d0202844cc5cdb185f696145cad663161a60.png)

- ◼ RAID (Redundant Arrays of Independent Disks) là hình thức ghép nhiều đĩa cứng vật lý thành một mảng đĩa cứng
- ◼ Gồm 7 mức: từ 0 đến 6
- ◼ Các mức không thể hiện mối quan hệ thứ bậc mà là các kiến trúc thiết kế khác nhau có chung ba đặc điểm:
- 1) Mảng đĩa cứng vật lý được hệ điều hành coi như một ổ logic duy nhất → có dung lượng lớn
- 2) Dữ liệu được lưu trữ phân tán trên các ổ đĩa vật lý → cho phép truy cập song song → tốc độ nhanh
- 3) Có thể sử dụng dung lượng dự phòng (redundant capacity) để lưu trữ các thông tin kiểm tra chẵn lẻ, cho phép khôi phục lại thông tin trong trường hợp đĩa bị hỏng → an toàn thông tin

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000021_4c22d140ffb13c334cf20ec7098f2856d1286058ae113f4e28e8de46d886ca8d.png)

| Loại         | p                            | Mô tả   | ab Số ab đĩa cần e 6.3 R Độ sẵn sàng dữ liệu                                                   | Table 6.3 RAID Levels Cấ Mô tả Số đĩa Độ sẵn sàng dữ Khả năng truyền dữ liệu vào/ra cỡ Tốc  AID Leve Khả năng truyền AID Leve dữ liệu vào/ra cỡ lớn                                                                           | s  Tốc độ yêu cầu vào/ra cỡ nhỏ                                                                          |
|--------------|------------------------------|---------|---------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 0            | Không dư thừa                | N       | Thấp hơn đĩa đơn                                  | Rất cao  Rất cao đối với cả đọc  và ghi                                   | Phân dải                                                                 |
| Nhân đôi 1   | Nhân đôi                     | 2 N     | Cao hơn RAID  2, 3, 4, hoặc 5;  thấp hơn RAID  6  | Cao hơn đĩa đơn  đối với đọc; tương  tự đĩa đơn đối với  ghi              | Gấp đôi đĩa đơn đối với  đọc; tương tự đĩa đơn  đối với ghi              |
| Truy cập     | 2  Dư thừa  nhờ mã  Hammin g | N+m     | Cao hơn đĩa đơn; tương đương RAID 3,  4, hoặc 5   | Cao nhất trong các cấp được nêu                                           | Xấp xỉ gấp đôi đĩa đơn                                                   |
| song  song 3 | Parity  xen kẽ  mức bit      | N+1     | Cao hơn đĩa  đơn; tương  đương RAID 2,  4, hoặc 5 | Cao nhất trong các cấp được nêu                                           | Xấp xỉ gấp đôi đĩa đơn                                                   |
| Truy cập độc N =  lập 4              | N = number of data disks;  ập block Parity  xen kẽ  mức  f data d block                              | N+1     | m proportional to log N 3, hoặc 5  Cao hơn đĩa  đơn; tương  đương RAID 2,  rtional to log  3, hoặc 5                                                   | Tương tự RAID 0  đối với đọc; thấp hơn đáng kể so với đĩa đơn đối với ghi | Tương tự RAID 0 đối với đọc; thấp hơn đáng kể so với đĩa đơn đối với ghi |

|   5 | Parity  phân tán xen kẽ mức block       | N+1   | Cao hơn đĩa đơn; tương đương RAID 2,  3, hoặc 4   | Tương tự RAID 0  đối với đọc; thấp hơn đĩa đơn đối với ghi   | Tương tự RAID 0 đối với đọc; thấp hơn đĩa đơn đối với ghi   |
|-----|-----------------------------------------|-------|---------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------------|
|   6 | Parity  nhân đôi  phân tán  xen kẽ  mức | N+2   | Cao nhất trong  các cấp được  nêu                 | Tương tự RAID 0  đối với đọc; thấp  hơn RAID 5 đối  với ghi  | Tương tự RAID 0 đối với đọc; thấp hơn RAID 5 đối với ghi    |

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000022_f71e71cee0f12008f705f36fab656775a2cf7c6984d76c937852d74585e34595.png)

## RAID Levels 0, 1, 2

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000023_2df83766d21c8b10e5ad0b08011bf46997ca5c12b16450b200fc8eb027121793.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000024_df16258a6a872089157caf841f038ec95c51d42bfe12423a38e6cc445e2a950f.png)

## RAID Levels

3, 4, 5, 6

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000025_20c1865e6771d5255b07fea46a4963d8c2eb44a84caa7c1095576186dae2c3de.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000026_30981b64c33ba4e2ad1380dcfb74b831c81f7862042d0ba3cda6df121e225833.png)

## RAID Level 0

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000027_44c9323a0e60716dccbd4791c5d46d5f242626d6403262a39218b5338a592764.png)

## RAID 0 cho Dung lượng truyền dữ liệu cao

- ◼ Các ứng dụng muốn có tốc độ truyền tải cao, phải đáp ứng hai yêu cầu:
1. Phải có dung lượng truyền tải cao trên toàn bộ đường dẫn giữa bộ nhớ máy chủ và các ổ đĩa riêng lẻ
2. Ứng dụng phải tạo ra các yêu cầu I/O để điều khiển mảng đĩa một cách hiệu quả

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000028_8a422aa28f4b638a7d569180788fafdf3e7eb48672c96cb08abc31e7a3dc745a.png)

## RAID 0 cho Tốc độ yêu cầu I/O cao

- ◼ Đối với yêu cầu I/O riêng lẻ yêu cầu lượng nhỏ dữ liệu, thời gian I/O phụ thuộc vào thời gian tìm kiếm và độ trễ quay
- ◼ Mảng đĩa có thể cung cấp tốc độ thực thi I/O cao bằng cách cân bằng tải I/O trên nhiều đĩa
- ◼ Nếu kích thước strip lớn , có thể xử lý song song nhiều yêu cầu I/O đợi, để giảm thời gian xếp hàng cho mỗi yêu cầu

## + b. RAID Level 1

R
ể a
ó i
c d
ệu 1 R
Dự phòng (redundancy): dữ liệu thêm vào để a
i
đảm bảo việc lưu trữ tin cậy trong bộ nhớ (có d
khả năng phát hiện, sửa lỗi, khôi phục dữ liệu khi bị lỗi)

## Đặc điểm

- ◼ RAID 1 khác với RAID 2 đến 6 trong cách thức dự phòng
- ◼ Khả năng dự phòng đạt được bằng cách đơn giản sao chép tất cả dữ liệu
- ◼ Data striping được sử dụng nhưng mỗi dải logic được ánh xạ tới hai đĩa vật lý riêng biệt sao cho mỗi đĩa trong mảng đều có một đĩa nhân bản có chứa cùng một dữ liệu
- ◼ RAID 1 có thể được thực hiện mà không cần data striping (không phổ biến)

## Hiệu quả

- ◼ Một yêu cầu đọc có thể được phục vụ bởi một trong hai đĩa có chứa dữ liệu yêu cầu
- ◼ Không có "write penalty"
- ◼ Dễ khắc phục sai sót. Khi một ổ đĩa hỏng, dữ liệu có thể được truy cập từ ổ đĩa thứ hai
- ◼ Cung cấp bản sao thời gian thực của tất cả dữ liệu
- ◼ Có thể đạt được tốc độ yêu cầu I/O cao nếu hầu hết các yêu cầu là Đọc
- ◼ Nhược điểm chủ yếu là chi phí

+

## c. RAID Level 2

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000029_9146f2b5d43fa6d8e056292a05d4d488933971b0c057cb5d3a37fdb3a8590fd6.png)

## Đặc điểm

- ◼ Sử dụng kỹ thuật truy nhập song song: tất cả các đĩa đều cùng tham gia vào việc xử lý một yêu cầu đọc/ghi.
- ◼ Trục của các ổ đĩa được đồng bộ sao cho các đầu đĩa ở vị trí như nhau trên đĩa vào bất kỳ thời điểm nào
- ◼ Sử dụng data striping
- ◼ Strip rất nhỏ, thường bằng 1 byte hoặc 1 word

## Hiệu quả

- ◼ Mã sửa lỗi được tính trên các bit tương ứng trên mỗi đĩa dữ liệu và các bit mã được lưu trữ trong các vị trí bit tương ứng trên các đĩa parity
- ◼ Sử dụng Hamming SEC-DEC
- ◼ Số lượng đĩa dự phòng tỷ lệ thuận với log của số đĩa dữ liệu
- ◼ Khi đọc/ghi dữ liệu, các mã CRC được tính toán – – ghi cùng lúc với dữ liệu
- ◼ Chỉ hiệu quả trong môi trường xảy ra nhiều lỗi đĩa

## + d. RAID Level 3

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000030_fbdbbcce87deac6a4805856823adbb26d4b08a0ce7e04b0f04bc0a2c2785afa6.png)

## Đặc điểm

- ◼ Chỉ cần 1 đĩa dự phòng, không cần quan tâm độ lớn mảng đĩa
- ◼ Sử dụng truy cập song song, kỹ thuật strip, tuy nhiên kích thước strip nhỏ
- ◼ Thay vì dùng mã sửa lỗi, một bit chẵn lẻ đơn giản được tính toán cho một tập các bit riêng ở cùng vị trí trên tất cả các đĩa dữ liệu
- Trong trường hợp ổ đĩa bị hỏng, toàn bộ mảng đc thiết lập chế độ reduced mode, ổ đĩa dự phòng được truy cập để phục hồi dữ liệu.
- Đĩa hỏng đc thay thế và được ghi dữ liệu đã phục hồi lên đó

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000031_66850b14a9e8ed9f6d005a54365772a6ded6ad7f3462ba12890fed575f3bda57.png)

## Hiệu năng

- ◼ Có thể đạt được tốc độ truyền dữ liệu rất cao
- ◼ Một yêu cầu truy cập vào/ra có thể được đáp ứng bằng việc truyền dữ liệu song song
- ◼ Trong một môi trường định hướng giao dịch, hiệu suất bị ảnh hưởng

+

## e. RAID Level 4

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000032_97b159c056b41d95806ceb2c027e3399708336b8c606bdbea0908fc4eda4bb47.png)

## Đặc điểm

- ◼ Sử dụng kỹ thuật truy cập độc lập: cho phép nhiều yêu cầu I/O riêng biệt có thể được đáp ứng song song
- ◼ Sử dụng data striping
- ◼ Strip có kích thước khá lớn
- ◼ Dự phòng: Dải chẵn lẻ -parity strip tương tự cách tính RAID 3 →đặt trên đĩa dự phòng .

## Strip – Dải

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000033_70b95894e121e6894acffc24c192cbcdaac712211322821ec1d9b1c53790a314.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000034_f8a91c2d7694da3f9158a22cf42ceeba8e873136d7444be6d4ec9bca9e4fbba5.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000035_97b159c056b41d95806ceb2c027e3399708336b8c606bdbea0908fc4eda4bb47.png)

## Dự phòng

- ◼ Do strip khá lớn và truy cập độc lập nên trong trường hợp yêu cầu I/O có kích thước nhỏ chỉ ghi trên một đĩa (strip write) cần phải tính toán lại dải chẵn lẻ (parity strip): 2 thao tác đọc và 2 thao tác ghi: hiện tượng write penalty – Số lần truy cập đĩa trong một hoạt động đọc/ghi)
- ◼ Trường hợp yêu cầu ghi I/O lớn phải thực hiện trên nhiều đĩa, Dải chẵn lẻ sẽ được tính toán lại toàn bộ.
- ◼ Do các hoạt động Ghi đĩa đều cần phải ghi lại Đĩa dự phòng →dễ gây hiện tượng nút cổ chai

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000036_28c72cd3a43b07f8680bb855fd68702100c038769d0236ab602269cfcda9c714.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000037_a32e406547780dbd04faa5d4534b3e74fc2c0d672088e50288d3bf09bd374cc6.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000038_523aeea3dabf11b2ed686ea312f0c412862d474a0781289929619ad5f843c2d4.png)

## f. RAID Level 5

## Đặc điểm

- ◼ Được tổ chức theo cách tương tự như RAID 4
- ◼ Chỉ khác ở sự phân bố Dải chẵn lẻ trên tất cả các đĩa
- ◼ Một phân bổ điển hình là cơ chế điều phối xoay vòng round -robin
- ◼ Việc phân phối Dải chẵn lẻ trên tất cả các ổ đĩa tránh được khả năng nút cổ chai I/O của RAID 4

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000039_2dc2e2b5aafbf9120030cbaeba48635a9a6ec5aa0dce0711b50a21f724a4039c.png)

## g. RAID Level 6

## Đặc điểm

- ◼ Hai thuật toán tính Dải chẵn lẻ (P và Q) riêng được thực hiện và được lưu trữ trong các khối riêng biệt trên các đĩa khác nhau
- ◼ Ưu điểm: tính sẵn sàng dữ liệu cực cao (khả năng khôi phục lại dữ liệu cao)
- ◼ Dữ liệu chỉ bị mất nếu ba ổ đĩa bị hỏng cùng lúc trong khoảng thời gian cần thiết để sửa chữa (MTTR -mean time to repair)
- ◼ Chịu một write penalty đáng kể do mỗi lần ghi đều tính toán và ghi lại hai Dải chẵn lẻ

## So sánh RAID

|   Le vel | Ưu điểm                                                                                                                                                                                               | Nhược điểm                                                                                 | Ứng dụng                                                                                      |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
|        0 | Cải thiện hiệu suất truy cập I/O bằng  cách phân phối tải I/O ra nhiều kênh và  đĩa Không tính parity (ko có Dự phòng) Thiết kế đơn giản Dễ thực hiện                                                 | Nếu dữ liệu trên 1 ổ  đĩa hỏng --> toàn bộ  dữ liệu sẽ hỏng hết                            | Sản xuất và biên tập  video Chỉnh sửa ảnh Các ứng dụng yêu cầu  băng thông cao                |
|        1 | Dữ liệu được dự phòng 100%: không  cần phải tính toán lại dữ liệu trong  trường hợp lỗi, chỉ cần sao lưu từ đĩa  dự phòng Chịu được nhiều lỗi ổ đĩa Thiết kế hệ thống đơn giản                        | Số lượng đĩa dự  phòng nhiều nhất                                                          | Kế toán Tính toán lương Tài chính Bất kỳ ứng dụng nào  yêu cầu tính sẵn sàng  dữ liệu rất cao |
|        2 | Tốc độ truyền dữ liệu cực kỳ cao Tốc độ truyền dữ liệu càng cao thì tỷ lệ  giữa số lượng đĩa dữ liệu/số lượng đĩa  ECC càng lớn Thiết kế bộ điều khiển tương đối đơn  giản so với mức RAID 3, 4, và 5 | Nếu kích thước strip  nhỏ  -- > tỷ lệ số đĩa  ECC/số đĩa dữ liệu  cao -- > không hiệu  quả | Không còn được sử  dụng do không hiệu  quả về mặt thương mại                                  |

| So sánh RAID   | So sánh RAID                                                                    | So sánh RAID                                                                                                                                                     | So sánh RAID                                                                                                                                             |
|----------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Le vel         | Ưu điểm                                                                         | Nhược điểm                                                                                                                                                       | Ứng dụng                                                                                                                                                 |
| 3              | Tốc độ đọc /ghi rất cao Tỷ lệ số đĩa ECC/số  đĩa dữ liệu thấp -- > Hiệu quả cao | Tốc độ tối đa của một  transaction bằng tốc độ của  một ổ đĩa đơn (nếu trục đĩa  được đồng bộ) Thiết kế bộ điều khiển khá  phức tạp                              | Sản xuất video và live  streaming Chỉnh sửa hình ảnh Chỉnh sửa video Ứng dụng chế bản in  (Illustrator, ...) Bất kỳ ứng dụng nào yêu  cầu băng thông cao |
| 4              | Tốc độ đọc rất cao Tỷ lệ số đĩa ECC/số  đĩa dữ liệu thấp -- > Hiệu quả cao      | Thiết kế bộ điều khiển phức K tạp Tốc độ ghi thấp nhất và số  lần ghi (write penalty) cao Phục hồi dữ liệu khó khăn  và không hiệu quả trong  trường hợp đĩa lỗi | Không còn được sử  dụng do không hiệu quả  về mặt thương mại                                                                                             |

|   Le vel | Ưu điểm                                                                                                                                                              | Nhược điểm                                                                                                           | Ứng dụng                                                                                                                    |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
|        5 | Tốc độ đọc rất cao Tỷ lệ số đĩa ECC/số  đĩa dữ liệu thấp -- > Hiệu quả cao Tốc độ đường truyền  cao                                                                  | Thiết kế bộ điều khiển phức F tạp nhất Khó khăn để khôi phục dữ  liệu trong trường hợp đĩa lỗi W (so với mức RAID 1) | File and application  servers Database servers  Web, e - mail, and news servers Intranet servers Most versatile RAID  level |
|        6 | Khả năng sửa lỗi và  phục hồi dữ liệu rất  cao, có thể phục hồi  trong trường hợp nhiều  đĩa bị lỗi (chỉ không  phục hồi được nếu có 3  đĩa lỗi cùng một thời  điểm) | Thiết kế bộ điều khiển phức G tạp nhất Việc tính toán parity phức  tạp                                               | Giải pháp hoàn hảo cho  các ứng dụng quan trọng                                                                             |

## + Bài tập

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000040_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

1. Xét mảng đĩa RAID gồm 4 ổ đĩa, mỗi ổ đĩa có dung lượng 200GB. Tính dung lượng lưu trữ của mảng đĩa với các level 0, 1, 3, 4, 5 và 6?
2. Một hệ thống máy tính cần dung lượng lưu trữ là 400GB. Nếu sử dụng mảng đĩa RAID thì cần bao nhiêu ổ đĩa (dung lượng mỗi ổ đĩa là 80GB) với các level 0, 1, 3, 4, 5, 6

## + 6.3 Ổ cứng trạng thái rắn – – SSD -Solid State Drives

- ◼ Công nghệ SSD dần thay thế HHD trong những năm gần đây
- ◼ Mạch điện tử được xây dựng dựa vào công nghệ bán dẫn
- ◼ Các bộ nhớ loại này được gọi là Flash memory

## Flash memory

- Được sử dụng trong điện thoại thông minh, thiết bị GPS, máy nghe nhạc MP3, máy ảnh kỹ thuật số và thiết bị USB
- Gần đây, dung lượng, tốc độ ngày càng cao với giá thành rẻ → thay thế HDD.

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000041_3f29f991c7b0a9284551b7937d15c2c34733e14d0f5b3e2e1a136cc4f5d835de.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000042_bab2a29f23c3a51075f3b923156dede02ac86b9d211d9d2e6ea6791eebfdcb7f.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000043_f5f2974fdf8b83c1d2ffc75ce92dc76fe3118893c79bf4a355dcd83fa931f1a2.png)

+

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000044_dd129508c16019001e049f92b6aede0b6dd1486e6b08e221b825390d45875c02.png)

Flash Memory Operation

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000045_53b3eb95641d6b55d32115705fa3974de3b934b6cecd235c67bbdae5328611c3.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000046_e9e5c2886412a9140fb8042b1af680dbd84d9c585f6752f14c5f3b87f922242a.png)

+

## Flash memory (tiếp)

- Có hai loại: NOR và NAND
- NOR
- Đơn vị truy cập cơ bản là bit.
- Cho phép truy cập ngẫu nhiên tốc độ cao.
- Sử dụng để lưu trữ HĐH của smart phone và chương trình BIOS để khởi động máy tính Window
- NAND:
- Đơn vị cơ bản là 16 hoặc 32 bit.
- Đọc/ghi theo khối.
- Sử dụng để sản xuất USB, thẻ nhớ, ổ cứng SSD,...
- Không cho phép truy cập ngẫu nhiên theo đ/c bus
- Sử dụng cơ chế truy cập trang (page access)

## SSD so sánh với HDD

SSD có các ưu điểm hơn HDD như sau:

- ◼ Số thao tác đọc/ghi trong một giây (IOPS) cao hơn
- ◼ Độ bền: ít chịu ảnh hưởng khi va đập vật lý
- ◼ Tuổi thọ dài hơn
- ◼ Tiêu thụ ít năng lượng hơn
- ◼ Khả năng chạy êm và mát hơn
- ◼ Thời gian truy cập ngắn hơn, thời gian trễ ít hơn

| +   |
|-----|

## Bảng 6.5

## So sánh

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000047_a32e406547780dbd04faa5d4534b3e74fc2c0d672088e50288d3bf09bd374cc6.png)

## Tổ chức SSD

- ◼ Tổ chức hệ thống SDD:
- ◼ Trên host , hệ điều hành khi có yêu cầu truy cập dữ liệu sử dụng một driver I/O của SSD: (driver: chương trình điều khiển các thiết bị ngoại vi)
- ◼ Nếu SSD là ổ cứng gắn trong: sử dụng kết nối PCIe.
- ◼ Nếu SSD thiết bị ngoại vi: giao tiếp theo chuẩn USB.
- ◼ SSD có chứa các thành phần sau:
- Controller: điều khiển kết nối bên ngoài và hoạt động bên trong SSD
- Addressing: Logic lựa chọn các Flash memory component
- Data buffer/Cache: chế tạo từ SRAM đệm dữ liệu và tăng tốc độ .
- Error Correction: cơ chế sửa lỗi .
- Flash memory component: các chip NAND .

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000048_d5778973baaeab505c393d86f5a57fcc0e41e2656849af048159bd5a094801ad.png)

+

## Vấn đề thực tế

## Có hai vấn đề thực tế xảy ra đối với SSD mà không xảy ra với HDDs

1. Hiệu năng SSD có khuynh hướng giảm dần khi thiết bị được sử dụng . Nguyên nhân:
2. ◼ Các file được lưu trữ dưới dạng các page 4KB, thường không được sắp xếp liên tục trong BN
3. ◼ Truy cập (thao tác đọc/ghi dữ liệu) BN flash được thực hiện theo các block 512 KB (vậy 1 block có 128 page)
4. ◼ Cơ chế Ghi vào dữ liệu vào một page:
5. ◼ Cả block (chứa page) phải được đọc từ BN ra bộ đệm RAM. Việc ghi dữ liệu vào page đó được thực hiện tại đây
6. ◼ Trước khi block được ghi lại vào bộ nhớ flash, toàn bộ nội dung block đó trong bộ nhớ flash phải được xoá
7. ◼ Khi đó dữ liệu từ bộ đệm mới được ghi vào flash memory

Chính vì cơ chế này gây ra một hiện tượng phân mảnh bộ nhớ:do quá trình sử dụng lâu, ghi đi ghi lại nhiều lần các page của một file được ghi rải rác ở nhiều block khác nhau → việc đọc/ghi bộ nhớ trở nên chậm

+

## Vấn đề thực tế (tiếp)

2. Flash memory không thể sử dụng được sau một số lần ghi (thông thường là 100.000 lần)
2. ◼ Kỹ thuật kéo dài tuổi thọ:
3. ◼ Sử dụng cache để giữ chậm và gộp các hoạt động ghi
4. ◼ Dùng thuật toán wear-leveling: phân bố đều các lần ghi lên các block
5. ◼ Quản lý các bad-block
6. ◼ Hầu hết các thiết bị flash có cơ chế ước tính thời gian hoạt động còn lại của chúng để hệ thống có thể dự đoán lỗi và có cơ chế dự phòng

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000049_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## 6.4 Đĩa quang

- ◼ Ra đời năm 1983
- ◼ Là sản phẩm tiêu dùng thành công nhất mọi thời đại
- ◼ Đĩa CD không thể xóa, lưu trữ 60 phút âm thanh một mặt
- ◼ Cách mạng trong công nghệ lưu trữ quang giá thành thấp

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000050_6362fca6cc9e37e53262a65153619782c907ba3fe3ca8c078776c6a7b067f13c.png)

+◼

- +◼ CD – Compact Disk: đĩa không thể xóa lưu trữ file âm thanh số. Kích thước tiêu chuẩn: 12 cm và có thể ghi hơn 60 phút không bị gián đoạn thời gian phát.
- ◼ CD -ROM – – Đĩa CD chỉ đọc: đĩa không thể xóa lưu trữ dữ liệu máy tính. Kích thước tiêu chuẩn: 12 cm và có thể chứa hơn 650 Mbytes.
- ◼ CD -R – Đĩa CD có thể ghi: tương tự như đĩa CD -ROM, người dùng chỉ có thể ghi vào đĩa một lần.
- ◼ CD -RW – Đĩa CD ghi lại được: tương tự như đĩa CD-ROM nhưng người dùng có thể xóa và ghi đè lên đĩa nhiều lần.
- ◼ DVD – Đĩa đa năng kỹ thuật số: một công nghệ lưu trữ dữ liệu video nén, số hóa cũng như các dữ liệu số khác. Kích thước: 8 hoặc 12 cm đều được sử dụng, lưu trữ dữ liệu cả hai mặt lên đến 17GB. DVD -ROM – DVD chỉ đọc
- ◼ DVD -R – – Đĩa DVD có thể ghi: tương tự như đĩa DVD -ROM. Người dùng có thể ghi vào đĩa chỉ một lần. Chỉ có thể sử dụng đĩa một mặt .
- ◼ DVD -RW – Đĩa DVD ghi lại được: tương tự như đĩa DVD-ROM, tuy nhiên n gười dùng có thể xóa và ghi đè lên đĩa nhiều lần. Chỉ có thể sử dụng đĩa một mặt .
- ◼ Blu -ray DVD – Đĩa video độ nét cao: cung cấp mật độ lưu trữ dữ liệu lớn hơn nhiều so với đĩa DVD, sử dụng laser 405 nm (xanh tím). Một lớp duy nhất trên một mặt có thể lưu trữ 25 Gbytes.

+

## Compact Disk Read-Only Memory (CD-ROM)

- ◼ Audio CD và CD -ROM dùng công nghệ tương tự nhau
- ◼ Điểm khác biệt chính: đầu đọc CD -ROM có độ gồ ghề hơn và có thiết bị sửa lỗi để đảm bảo cho dữ liệu được truyền đúng
- ◼ Quá trình sản xuất:
- ◼ Đĩa được chế tạo từ nhựa polycarbonate
- ◼ Thông tin ghi lại bằng kỹ thuật số được in dưới dạng một chuỗi các lỗ cực nhỏ trên bề mặt polycarbonate
- ◼ Được thực hiện bằng laser cường độ cao tập trung tạo ra đĩa master
- ◼ Đĩa master được dùng làm khuôn để tạo ra các bản sao trên polycarbonate
- ◼ Bề mặt lỗ sau đó được phủ 1 lớp phản xạ tốt, thường là nhôm/vàng
- ◼ Tiếp tục phủ lên 1 lớp sơn acrylic trong suốt để chống bụi và trầy xước
- ◼ Cuối cùng có thể dùng kĩ thuật in lụa để in nhãn hiệu lên bề mặt acrylic
- Tổ chức thông tin theo đường xoắn ốc
- Thông tin được ghi dưới dạng các pit và land:
- 0: khoảng không thay đổi độ cao
- 1: khoảng có thay đổi độ cao
- Khác với đĩa từ, trục quay quay theo vận tốc góc không đổi (CAV), CD quay với vận tốc tuyến tính không đổi (CLV): Thông tin được quét cùng tốc độ bằng cách quay đĩa ở tốc độ khác nhau
- ◼ Phù hợp để phân phối số lượng lớn dữ liệu cho một 
ố p pp
số lượng lớn người dùng
- ◼ Không phù hợp cho các ứng dụng cá nhân do chi 
ầ g pp g g 
phí lớn cho quá trình ghi ban đầu
- ◼ 2 ưu điểm:
- ◼ Đĩa quang chứa thông tin có thể được nhân bản 
ố qg g 
rộng rãi một cách không tốn kém
- ◼ Đĩa quang tháo ra được, cho phép đĩa được sử 
ể qg 
dụng để lưu trữ
- ◼ Nhược điểm:
- ◼ Chỉ đọc, không updated được
- ◼ Thời gian truy cập lâu hơn so với ổ đĩa từ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000051_c9cb2bf9fe18e97138d78eaf768f67ff51729a3863595eb15af9aad1150dbbc7.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000052_3da953f71c0184e7cc6038af9518eb17b2d2ffda3c2c53a2bfc0a04dc1a7af17.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000053_604849532dec42d5d85267953f5129e01f7e861b6ae6cc3ee9bd5e76b65f5d3c.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000054_c8d82c71f0e31410b9060362b7b7c84e51baa44b735ebc5d9f1b23b67a37b530.png)

## CD -ROM

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000055_2a964edcf31041a6ed49ba09eed331a5fa4be9c668af05a41358c8db96f3c9ec.png)

+

## CD Recordable (CD-R)

- ◼ Ghi 1 lần đọc nhiều lần
- ◼ Thích hợp với các ứng dụng chỉ cần một hoặc một số ít bản sao của một bộ dữ liệu
- ◼ Đĩa được chuẩn bị để có thể được ghi một lần bằng một tia laser có cường độ vừa phải
- ◼ Medium bao gồm 1 lớp khuôn được dùng để thay đổi độ phản xạ và được kích hoạt bởi 1 tia laser cường độ cao
- ◼ Cung cấp một bản ghi vĩnh viễn của khối lượng lớn dữ liệu người dùng

## CD Rewritable (CD-RW)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000056_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

- ◼ Có thể ghi lại nhiều lần
- ◼ Đĩa thay đổi pha sử dụng vật liệu có hai độ phản xạ khác nhau ở hai trạng thái pha khác nhau
- ◼ Trạng thái vô định hình
- ◼ Các phân tử có hướng ngẫu nhiên phản xạ ánh sáng kém
- ◼ Trạng thái tinh thể
- ◼ Có bề mặt nhẵn phản xạ ánh sáng tốt
- ◼ Một chùm tia laser có thể thay đổi vật liệu từ pha này sang pha kia
- ◼ Nhược điểm: cuối cùng vật liệu mất đi đặc tính mong muốn vĩnh viễn
- ◼ Ưu điểm: có thể ghi lại được

+

## Digital Versatile Disk (DVD)

Là đĩa đa năng kỹ thuật số, có một số ưu điểm sau:

- Chất luợng hình ảnh ấn tượng
- Dung lượng rất cao (4.7G mỗi lớp), có thể ghi 2 lớp, 2 mặt (17GB)
- Trọn 1 bộ phim dài trên 1 đĩa đơn
- Sử dụng ghi video nén theo chuẩn MPEG

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000057_4b9ba1f1c57516e18995af712cb02798066fb3573b6a408520e52007ab9fbc99.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000058_0d9e73e021827a05facfbb8c9e036543ce93ed105d2a5c62136756908d84b601.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000059_1aacb4ea4e58d7aaeb8f5be639b038740ab2c60d306f92729c17f5134ec395e6.png)

+

## Đĩa quang độ phân giải cao – HDDVD, Blue-ray

- ◼ Được thiết kế cho video độ nét cao
- ◼ Dung lượng lớn hơn nhiều so với DVD
- ◼ Laser dải màu xanh tím có bước sóng ngắn hơn:
- ◼ Pit nhỏ hơn
- ◼ HD -DVD: 15GB 1 lớp 1 mặt
- ◼ Blue -ray:
- ◼ Lớp dữ liệu gần với laser hơn: Tập trung cao, ít biến dạng hơn, pit nhỏ hơn
- ◼ 25GB trên một lớp

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000060_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000061_960c02a45736c7d60b6e6b43f80b6409680483e457f4725227e9d8b4a1f29d82.png)

## Đĩa quang độ phân giải cao

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000062_af4c0732dcf12f4286c8800e062109cd995cb50967fc467c911a93c06e66eba6.png)

+

## 5. Băng từ

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000063_06fc2d6fc1777f43ee91243db7072b94f5b36847547d293d8afc10a843e69c28.png)

- ◼ Hệ thống băng sử dụng kỹ thuật đọc và ghi giống như các hệ thống đĩa
- ◼ Vật liệu là băng polyester mềm dẻo được phủ bởi chất liệu từ hoá
- ◼ Lớp phủ có thể bao gồm lượng nhỏ kim loại tinh khiết trong binders đặc biệt hoặc cuộn phim kim loại mạ hơi
- ◼ Dữ liệu trên băng được cấu trúc theo các track song song chạy dọc
- ◼ Ghi nối tiếp
- ◼ Dữ liệu được trải ra theo một dãy bit dọc trên mỗi track
- ◼ Dữ liệu được đọc và ghi trong các block liền kề được gọi là bản ghi vật lý physical records
- ◼ Các block trên băng được phân cách bằng các khoảng trống được gọi là khoảng trống giữa các bản ghi inter -record gaps

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000064_76e6d1bbc02e1929c89f3fcf8904bdd3c5f0d9d183c08b7a9c62a0f7fb823d0d.png)

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000065_30981b64c33ba4e2ad1380dcfb74b831c81f7862042d0ba3cda6df121e225833.png)

## Bảng 6.7 LTO Tape Drives

![Image](/content/drive/MyDrive/MD/KienTrucMayTinh/CH06-Storage_artifacts/image_000066_4f696117a39d390d5104b28025c3d4578173c12cb729674c500a75196244ef09.png)

+

## Tổng kết

## Chương 6

- ◼ Đĩa từ
- ◼ Cơ chế đọc và ghi từ
- ◼ Tổ chức và định dạng dữ liệu
- ◼ Đặc tính vật lý
- ◼ Tham số hiệu suất đĩa
- ◼ Solid state drives
- ◼ Flash memory
- ◼ SSD so với HDD
- ◼ Tổ chức SSD
- ◼ Vấn đề thực tế
- ◼ Băng từ

## Bộ nhớ ngoài

- ◼ RAID
- ◼ RAID level 0
- ◼ RAID level 1
- ◼ RAID level 2
- ◼ RAID level 3
- ◼ RAID level 4
- ◼ RAID level 5
- ◼ RAID level 6
- ◼ Bộ nhớ quang
- ◼ Đĩa Compact
- ◼ Đĩa DVD
- ◼ High-definition optical disks

## + Câu hỏi

## +

## Bài tập ứng dụng

Xây dựng chiến lược sao lưu dữ liệu cho một hệ thống máy tính gồm các lựa chọn sau:

1. Sử dụng các ổ cứng ngoài dung lượng 500GB, chi phí 150$/ổ
2. Sử dụng ổ băng từ giá 2500$, với các băng từ dung lượng 400GB, giá 50$.

Dữ liệu được sao lưu thành 2 bản trên media onsite và 1 bản trên media offsite để khôi phục dữ liệu khi cần

- a. Giả sử cần back up 1TB ta nên chọn option nào?
- b. Với dung lượng tối thiểu là bao nhiêu thì giải pháp băng từ sẽ cho giá thành rẻ hơn?