# Ôn tập Buổi 2 — Lê Sơn Trường

## 1. Loss là gì?

Loss đo mức sai của dự đoán, để mô hình biết cần điều chỉnh trọng số thế nào.

- Hồi quy: MSE hoặc half-MSE, ví dụ L=(yhat−y)²/2. Sai càng xa thì phạt càng mạnh.
- Phân loại: cross-entropy, L=−log p(lớp đúng). Đoán sai mà rất tự tin bị phạt mạnh.
- Loss khác accuracy: accuracy chỉ đếm đúng/sai; loss còn xét xác suất tự tin.

Với cross-entropy, đưa **logits thô** vào `nn.CrossEntropyLoss`, không softmax
trước một lần nữa. Mục 4.1/4.4 giải thích ổn định số và gradient p−y.

## 2. Backpropagation hoạt động thế nào?

Forward: dữ liệu → Linear → activation → logits → loss.
Backward: đi ngược đồ thị, dùng **chain rule**, tính đạo hàm loss theo từng trọng số.
Gradient cho biết chiều làm loss tăng; optimizer thường bước theo chiều ngược lại.

Ví dụ yhat=wx+b, L=(yhat−y)²/2:
∂L/∂w=(yhat−y)x; ∂L/∂b=yhat−y.
`loss.backward()` tính gradient, **chưa tự cập nhật trọng số**.

## 3. Tối ưu hóa

SGD cập nhật `w ← w − learning_rate × gradient` trên minibatch.
Adam dùng các trung bình trượt của gradient và bình phương gradient để điều chỉnh bước.
Learning rate nhỏ có thể học chậm; quá lớn có thể dao động hoặc phân kỳ.
Không có optimizer/LR tốt nhất cho mọi dữ liệu và ngân sách epoch.

## 4. Quy trình huấn luyện MLP

1. Đọc, xử lý dữ liệu; phân biệt train, validation và test.
2. Chọn mạng, loss, optimizer, batch size và số epoch.
3. Với mỗi epoch, đưa từng batch train qua mạng; zero gradient → forward → loss
   → backward → optimizer.step.
4. Đánh giá validation bằng `model.eval()` và `torch.no_grad()`; không cập nhật từ validation.
5. Xem learning curve, chọn mô hình/siêu tham số theo validation.
6. Sau khi chốt, mới đánh giá trên test độc lập nếu có.

```python
for epoch in range(epochs):
    model.train()
    for X, y in train_loader:
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Tính validation loss/accuracy, cân theo số mẫu.
        pass
```

Một epoch là một lượt qua toàn bộ train, không phải một batch hay một bước update.
Dropout hoạt động lúc train và tắt lúc eval; no_grad tắt xây graph, không thay thế eval.
Trong ví dụ Fashion-MNIST theo sách, test được dùng như validation, nên kết quả
tuning ở bài này không được quảng bá như đánh giá trên test còn nguyên độc lập.

## 5. Các câu có thể được hỏi khi trả bài

- **Vì sao cần activation?** Nếu chỉ ghép Linear thì toàn mạng vẫn là affine.
- **Vì sao train tốt nhưng validation kém?** Mạng có thể ghi nhớ nhiễu/chi tiết train (overfitting).
- **Vì sao zero init MLP không học?** Neuron ẩn đối xứng; với ReLU, gradient có thể chết tại 0.
- **Vì sao không reset gradient?** PyTorch cộng dồn `.grad`; bỏ reset khiến update không còn đúng minibatch SGD thông thường.
- **Dropout có luôn tốt?** Không; phụ thuộc mức nhiễu, kiến trúc và thời gian học. Báo đúng kết quả đã chạy.
- **Vì sao cần validation?** Để chọn mô hình mà không tối ưu trực tiếp trên test độc lập.
- **Vì sao chuẩn hóa House Prices?** Tránh chênh lệch scale gây tối ưu kém điều kiện và regularization lệch.
