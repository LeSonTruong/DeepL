# Giải thích ngắn khi trả bài

**Bài 1 làm gì?** Dùng MLP phân loại 10 loại trang phục Fashion-MNIST,
viết đủ các bước học và so sánh learning rate .01, .001, .0001 trong **5 epoch**.

**Một lượt học diễn ra thế nào?**

1. `zero_grad()`: xóa gradient cũ, vì PyTorch cộng dồn gradient.
2. `model(x)`: dự đoán 10 điểm số thô (logits) cho mỗi ảnh.
3. `CrossEntropyLoss(logits, y)`: đo sai số so với nhãn đúng.
4. `loss.backward()`: tính gradient, cho biết tham số ảnh hưởng loss thế nào.
5. `optimizer.step()`: dùng gradient để cập nhật trọng số.

**Tại sao không softmax trước loss?** CrossEntropyLoss đã xử lý log-softmax
ổn định số bên trong. Chỉ dùng softmax riêng khi muốn hiển thị xác suất/độ tự tin.

**Epoch và batch khác nhau thế nào?** Epoch là một lượt qua toàn bộ train;
batch là một nhóm ảnh trong một bước cập nhật. Bài 1 có 94 batch/epoch
(6000 ảnh, batch64); bài 2 có 63 batch/epoch (8000 ảnh, batch128).
Batch cuối vẫn được học, không bỏ đi.

**Vì sao cần train/eval và no_grad?** `train()` bật chế độ học;
`eval()` đổi hành vi của lớp như Dropout/BatchNorm; `no_grad()` ngừng tạo đồ thị
gradient khi đánh giá. MLP hiện tại chỉ có Linear/ReLU nên eval chưa làm đổi
công thức dự đoán, nhưng vẫn cần viết đúng quy trình.

**Kết quả bài 1:** LR .001 đạt **85.10%**, .01 đạt **84.40%**, .0001 đạt **76.50%**.
LR .01 học nhanh ở epoch đầu nhưng loss dao động ở epoch 3–4.
LR .0001 vẫn giảm loss nhưng học chậm trong ngân sách 5 epoch.
LR .001 cân bằng tốt hơn trong lần chạy này.

**Bài 2 làm gì?** Giữ dữ liệu, kiến trúc và số epoch; thay cấu hình tối ưu.
SGD dùng gradient hiện tại; Momentum tích lũy hướng cập nhật; Adam điều chỉnh
bước theo các thống kê gradient. LR giữ theo mã mẫu, không bằng nhau.

**Kết quả bài 2 tại epoch 5:** SGD **70.33%**, Momentum **80.53%**, Adam **83.20%**.
Đây là kết quả của các cấu hình LR .05/.03/.001, không chứng minh Adam luôn tốt nhất.
SGD và Momentum đều có train/val loss tăng từ epoch 4 sang 5, có thể là dao động
cập nhật. Chưa nên gọi đó là overfitting chỉ vì validation giảm điểm.

**Early Stopping là gì?** Theo dõi val loss, dừng nếu 2 epoch liên tiếp không
cải thiện và khôi phục trọng số tốt nhất. Trong 5 epoch hiện tại **chưa dừng sớm**;
SGD/Momentum khôi phục epoch 4, Adam giữ epoch 5. Accuracy sau khôi phục:
SGD **75.67%**, Momentum **83.20%**, Adam **83.20%**.
Bảng cơ bản vẫn báo cáo epoch 5; không thay ngầm bằng điểm epoch tốt nhất.

**Train loss có hai cột vì sao?** `online_train_loss` lấy trung bình trong khi
mô hình đang cập nhật qua từng batch. `train_loss` trên biểu đồ đo lại toàn train
bằng mô hình cuối epoch, để so sánh cùng cách với validation loss.

**Validation có phải test cuối không?** Mã mẫu dùng một phần test chính thức
làm validation. Đã dùng để khảo sát siêu tham số nên điểm này không phải kết quả
test độc lập. Chỉ chạy một seed; kết luận giới hạn ở thí nghiệm hiện tại.
