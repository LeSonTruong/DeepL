# Thực hành 1 — Lê Sơn Trường

| Mô hình | Tham số | Val loss | Val accuracy | Giây/epoch | Epoch báo cáo |
|---|---:|---:|---:|---:|---:|
| MLP | 203,530 | 0.4766 | 82.83% | 0.100 | 5 |
| CNN | 206,922 | 0.4319 | 86.17% | 0.687 | 5 |

Bảng dùng epoch cuối (5); cùng seed42, split6000/1200, Adam, lr0.001, batch64.

<!-- BEGIN EPOCH_DETAILS -->

### Kết quả đầy đủ từng epoch

**Train loss**: trung bình theo số mẫu trong lúc cập nhật. **Val loss/accuracy**: đánh giá ở cuối epoch. **Δ accuracy**: thay đổi so với epoch trước, tính bằng điểm phần trăm. **Giây**: thời gian train + validation của epoch đó.

Epoch **in đậm** có validation accuracy cao nhất của cấu hình; nếu hòa, chọn validation loss nhỏ hơn. Bảng tổng kết bài 1 vẫn dùng epoch cuối để so sánh cùng ngân sách.

[Dữ liệu từng epoch: history.csv](history.csv)

#### MLP

| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.9494 | 0.5953 | 80.08% | — | 0.109 | — |
| 2 | 0.5999 | 0.5241 | 82.25% | +2.17 | 0.097 | — |
| **3** | 0.5264 | 0.4831 | 83.33% | +1.08 | 0.097 | Val tốt nhất |
| 4 | 0.4840 | 0.4918 | 81.92% | -1.42 | 0.097 | — |
| 5 | 0.4501 | 0.4766 | 82.83% | +0.92 | 0.098 | Epoch cuối |

Từ epoch 1 → 5: train loss 0.9494 → 0.4501; val loss 0.5953 → 0.4766; val accuracy 80.08% → 82.83%. Accuracy cao nhất là **83.33% tại epoch 3**.

Val loss thấp nhất lại ở **epoch 5** (0.4766). Loss và accuracy đo hai khía cạnh khác nhau, nên epoch có loss thấp nhất không nhất thiết có accuracy cao nhất.

#### CNN

| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.1296 | 0.6584 | 76.92% | — | 0.650 | — |
| 2 | 0.6478 | 0.5777 | 78.75% | +1.83 | 0.673 | — |
| 3 | 0.5521 | 0.4893 | 81.75% | +3.00 | 0.680 | — |
| 4 | 0.4973 | 0.4830 | 82.50% | +0.75 | 0.733 | — |
| **5** | 0.4617 | 0.4319 | 86.17% | +3.67 | 0.699 | Val tốt nhất; Epoch cuối |

Từ epoch 1 → 5: train loss 1.1296 → 0.4617; val loss 0.6584 → 0.4319; val accuracy 76.92% → 86.17%. Accuracy cao nhất là **86.17% tại epoch 5**.

<!-- END EPOCH_DETAILS -->

![Learning curves](learning_curves.png)

```text
input: (1, 1, 28, 28)
conv1: (1, 16, 28, 28)
relu1: (1, 16, 28, 28)
pool1: (1, 16, 14, 14)
conv2: (1, 32, 14, 14)
relu2: (1, 32, 14, 14)
pool2: (1, 32, 7, 7)
before Linear: (1, 1568)
Linear + ReLU: (1, 128)
output: (1, 10)

input: (64, 1, 28, 28)
conv1: (64, 16, 28, 28)
relu1: (64, 16, 28, 28)
pool1: (64, 16, 14, 14)
conv2: (64, 32, 14, 14)
relu2: (64, 32, 14, 14)
pool2: (64, 32, 7, 7)
before Linear: (64, 1568)
Linear + ReLU: (64, 128)
output: (64, 10)
```

Tensor CNN đi từ (N,1,28,28) qua (N,16,28,28), (N,16,14,14), (N,32,14,14), (N,32,7,7), rồi flatten thành (N,1568); chiều N là batch và được giữ nguyên.

MLP có 203,530 tham số và CNN có 206,922 tham số; convolution dùng chung trọng số theo vị trí, nhưng lớp Linear 1568→128 vẫn chiếm phần lớn tham số CNN.

Sau 5 epoch, MLP đạt 82.83% và CNN đạt 86.17% validation accuracy, tức CNN cao hơn 3.33 điểm phần trăm.

Thời gian trung bình gồm train và validation là 0.100 s/epoch cho MLP và 0.687 s/epoch cho CNN trên cpu; số tham số gần nhau không có nghĩa số phép tính giống nhau.

Kết quả chỉ phản ánh một seed và 5 epoch trên split này; train loss đo trong lúc cập nhật, còn validation loss đo cuối epoch nên hai đường loss không hoàn toàn cùng thời điểm.
