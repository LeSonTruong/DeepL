# Yêu cầu hai thực hành — đối chiếu đề

Nguồn yêu cầu là `baitap.txt`, khớp với `Slides_Hoc_sau_B2.pdf`,
trang PDF **26–27**, và hai script gợi ý.
Thư mục đặt tên Buoi3 theo repository, slide có tiêu đề Buổi 2 theo tài liệu gốc.

| Nội dung | Thực hành 1 | Thực hành 2 |
|---|---|---|
| Mục tiêu | Viết training loop, khảo sát LR | So sánh optimizer |
| Train | 6000 ảnh đầu train chính thức | 8000 ảnh đầu train chính thức |
| Validation | 1000 ảnh đầu test chính thức | 1500 ảnh đầu test chính thức |
| MLP | Flatten → Linear(784,256) → ReLU → Linear(256,10) | Flatten → Linear(784,128) → ReLU → Linear(128,10) |
| Epoch | **5**, theo mã mẫu | **5**, theo mã mẫu |
| Batch train / val | 64 / 256 | 128 / 256 |
| Optimizer, LR | Adam: **.01, .001, .0001**, theo slide | SGD .05; Momentum .03, momentum=.9; Adam .001, theo mã mẫu |
| Loss | CrossEntropyLoss trên logits | CrossEntropyLoss trên logits |
| Phải in | Device, số tham số, train loss, val accuracy mỗi epoch | Cấu hình và kết quả mỗi epoch |
| Sản phẩm | Code, bảng 3 LR, nhận xét tốc độ hội tụ | Code, bảng cấu hình, learning curves, ưu/nhược, lỗi và cải thiện |
| Nâng cao | Không có yêu cầu riêng | Early Stopping khi val loss không cải thiện |

Các bổ sung phục vụ báo cáo: train/val accuracy, Macro-F1, confusion matrix,
ảnh dự đoán đúng/sai, thời gian, checkpoint và notebook có output thật.
Không đổi số mẫu, kiến trúc, LR hay tăng epoch.

Đã đối chiếu các yêu cầu: vòng học đủ 5 bước; in device, số tham số,
train loss và validation accuracy **mỗi epoch**; bảng ba LR và nhận xét hội tụ;
cùng dữ liệu/kiến trúc/epoch khi so sánh SGD–Adam; learning curves;
Early Stopping; ưu/nhược điểm và phân tích lỗi/hướng cải thiện.
Các cột bắt buộc từng epoch có trong log notebook, `history.csv`, báo cáo
và ảnh tổng hợp. Early Stopping đã cài đặt nhưng chưa kích hoạt dừng trước
giới hạn 5 epoch trong lần chạy này.

Hai khác biệt thực hiện có ghi rõ:

- Đọc trực tiếp IDX thay torchvision do môi trường chưa có torchvision.
  Dùng đúng dữ liệu gốc và phép ToTensor tương đương: uint8 → float32 /255,
  shape (1,28,28), nhãn long; không thêm chuẩn hóa hoặc augmentation.
- Reset seed và generator cho mỗi cấu hình để cùng trọng số và thứ tự batch.
  Mã mẫu BTVN chỉ seed một lần nên các optimizer khởi tạo khác nhau.
  Cách làm này kiểm soát yếu tố ngẫu nhiên khi so sánh.

Validation giữ nguyên cách lấy từ test chính thức của mã gốc. Đã dùng validation
để so sánh nên không gọi điểm này là kết quả test độc lập.
Early Stopping là nhóm riêng, giới hạn 5 epoch, patience2, min_delta0;
bảng cơ bản luôn dùng epoch cuối.
