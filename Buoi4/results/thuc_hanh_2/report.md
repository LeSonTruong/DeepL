# Thực hành 2 — Lê Sơn Trường

<!-- BEGIN VARIANT_DETAILS -->

### Mục tiêu, đầu vào và đầu ra của bài 2

Dùng CNN của bài 1 làm baseline, rồi khảo sát riêng từng thay đổi: kernel, số kênh, BatchNorm và Dropout. Bài 1 so sánh MLP–CNN tại epoch cuối; bài 2 chọn checkpoint tốt nhất bằng validation rồi mới đánh giá test.

| Thành phần | Thông tin |
|---|---|
| Dữ liệu | FashionMNIST; ảnh xám 28×28, 10 lớp trang phục |
| Train | 6,000 ảnh từ official train; dùng cập nhật trọng số |
| Validation | 1,200 ảnh khác từ official train; chọn epoch/cấu hình |
| Test | 10,000 ảnh official test độc lập; đánh giá cuối và phân tích ma trận từng cấu hình |
| Tiền xử lý | Pixel float32 /255 thuộc [0,1], không augmentation; nhãn long 0–9 |
| Đầu vào | `(N,1,28,28)` |
| Đầu ra | `(N,10)` logits; argmax cho nhãn; CrossEntropyLoss nhận trực tiếp logits |

### Thiết lập và quy trình huấn luyện

| Thành phần | Giá trị |
|---|---|
| Thiết bị / môi trường | cpu; PyTorch 2.14.1+cpu; 2 threads |
| Seed | 42; cùng split và thứ tự batch cho mọi cấu hình |
| Optimizer / learning rate | Adam / 0.001 |
| Epoch | 5 cho mỗi cấu hình; 6 cấu hình, 30 dòng log |
| Batch | Train 64; validation/test 256 |
| Số batch train/epoch | 94; batch cuối 48 ảnh, vẫn được sử dụng |
| Loss | CrossEntropyLoss; lấy trung bình theo số mẫu |

Mỗi epoch: **train() → zero_grad → forward → loss → backward → optimizer.step** trên các batch train; sau đó **eval() + inference_mode()** trên validation. Chạy đủ số epoch, không Early Stopping. Thời gian/epoch gồm train + validation, không gồm kiểm tra shape, ghi checkpoint hay vẽ biểu đồ. Train loss đo trong lúc cập nhật; validation loss đo ở cuối epoch.

### Kiến trúc và shape của đủ 6 cấu hình

Shape có thứ tự `(N,C,H,W)`; N là số ảnh trong batch. Các shape dưới đây được ghi bằng hook trên `forward()` thật, với input giả bằng 0, chế độ eval trên CPU; đây là kiểm tra kiến trúc.

| Cấu hình | Sau Conv1 | Sau Pool1 | Sau Conv2 | Sau Pool2 | Sau Flatten | Đầu ra |
|---|---|---|---|---|---|---|
| baseline | `(N,16,28,28)` | `(N,16,14,14)` | `(N,32,14,14)` | `(N,32,7,7)` | `(N,1568)` | `(N,10)` |
| kernel5 | `(N,16,28,28)` | `(N,16,14,14)` | `(N,32,14,14)` | `(N,32,7,7)` | `(N,1568)` | `(N,10)` |
| channels8 | `(N,8,28,28)` | `(N,8,14,14)` | `(N,16,14,14)` | `(N,16,7,7)` | `(N,784)` | `(N,10)` |
| channels32 | `(N,32,28,28)` | `(N,32,14,14)` | `(N,64,14,14)` | `(N,64,7,7)` | `(N,3136)` | `(N,10)` |
| batchnorm | `(N,16,28,28)` | `(N,16,14,14)` | `(N,32,14,14)` | `(N,32,7,7)` | `(N,1568)` | `(N,10)` |
| dropout03 | `(N,16,28,28)` | `(N,16,14,14)` | `(N,32,14,14)` | `(N,32,7,7)` | `(N,1568)` | `(N,10)` |

ReLU giữ shape. Kernel3 dùng padding1, kernel5 dùng padding2 nên convolution giữ H,W; MaxPool2d(2) mới giảm 28→14→7. BatchNorm giữ shape và nằm giữa Conv–ReLU trong mỗi block. Dropout0.3 nằm sau ReLU của Linear128; các cấu hình khác có Dropout p=0. Phần phân loại: Flatten → Linear128 → ReLU → Dropout → Linear10. `argmax` trên 10 logits cho nhãn dự đoán.

### Phân bố số tham số

| Cấu hình | Block 1 (Conv + BN) | Block 2 (Conv + BN) | Phân loại (2 Linear) | Tổng |
|---|---:|---:|---:|---:|
| baseline | 160 | 4,640 | 202,122 | 206,922 |
| kernel5 | 416 | 12,832 | 202,122 | 215,370 |
| channels8 | 80 | 1,168 | 101,770 | 103,018 |
| channels32 | 320 | 18,496 | 402,826 | 421,642 |
| batchnorm | 192 | 4,704 | 202,122 | 207,018 |
| dropout03 | 160 | 4,640 | 202,122 | 206,922 |

Conv có `C_out × (C_in × k × k + 1)` tham số; Linear có `n_out × (n_in + 1)`, đã tính bias. BatchNorm có 2 tham số học được cho mỗi kênh (scale và bias); running statistics không tính vào tham số học được. ReLU, MaxPool, Flatten, Dropout và Identity không có tham số học được.

Ở baseline, phần phân loại chiếm **97.68%** tham số. Tăng số kênh còn làm tăng đầu vào của Linear đầu tiên. Dropout không thêm tham số; BatchNorm thêm một lượng nhỏ tham số.

[Nhật ký shape từng lớp, batch 1 và 64](shapes.txt) · [Kiến trúc và tham số dạng JSON](architectures.json)

### Mức thay đổi so với baseline

So sánh accuracy tại checkpoint validation tốt nhất của từng cấu hình; Δ accuracy tính bằng điểm phần trăm. Tỷ lệ thời gian >1 nghĩa là chậm hơn baseline trong lần chạy này; thời gian phụ thuộc máy và tải hệ thống.

| Cấu hình | Thay đổi duy nhất | Δ tham số | Δ val accuracy (điểm %) | Thời gian / baseline |
|---|---|---:|---:|---:|
| baseline | Mốc: k3, kênh16/32 | +0 | +0.00 | 1.00× |
| kernel5 | Kernel3→5, padding1→2 | +8,448 | -0.42 | 1.30× |
| channels8 | Kênh16/32→8/16 | -103,904 | -2.50 | 0.56× |
| channels32 | Kênh16/32→32/64 | +214,720 | +0.42 | 2.24× |
| batchnorm | Thêm BatchNorm sau mỗi Conv | +96 | +2.42 | 1.27× |
| dropout03 | Dropout p=0→0.3 sau Linear128 + ReLU | +0 | -2.42 | 1.04× |

### Chọn checkpoint, đánh giá test và đọc lỗi

1. Trong mỗi cấu hình, giữ epoch có validation accuracy lớn nhất; hòa thì chọn validation loss nhỏ hơn; hòa cả hai giữ epoch gặp trước.
2. So sánh checkpoint tốt nhất của 6 cấu hình theo cùng quy tắc; lưu lựa chọn vào `selection.json` trước khi đọc test.
3. Nạp [batchnorm_best.pt](batchnorm_best.pt) ở epoch **4**, có validation accuracy **88.58%**, validation loss **0.3297**.
4. Đánh giá đủ **10,000** ảnh test: accuracy **86.44%**, loss **0.3762**. Phần ma trận bổ sung đánh giá cả 6 checkpoint trên test để phân tích lỗi, giữ nguyên lựa chọn đã khóa bằng validation.

Confusion matrix: **hàng là nhãn thật, cột là dự đoán**. Tổng đường chéo **8,644/10,000** là số ảnh đúng; còn **1,356** ảnh sai. Lớp có tỷ lệ nhận đúng thấp nhất là **Shirt**: 623/1000 ảnh (62.30%).

- Nhầm Coat → Pullover: 127 ảnh.
- Nhầm Shirt → T-shirt/top: 119 ảnh.
- Nhầm Shirt → Pullover: 117 ảnh.

Ảnh minh họa gồm 5 ảnh đúng và 5 ảnh sai đầu tiên theo thứ tự test, có nhãn thật/dự đoán; đây là ví dụ để quan sát lỗi. Độ chính xác và confusion matrix được tính trên toàn bộ test.

### Gợi ý trình bày với giảng viên

Trình bày theo thứ tự: **mục tiêu → dữ liệu/đầu ra → hai block và shape → thiết lập/vòng học → so sánh 6 cấu hình → chọn mô hình/test → lỗi và kết luận**.

> Em dùng FashionMNIST, chia 6,000 ảnh train và 1,200 ảnh validation từ tập train chính thức. Đầu vào là ảnh xám 1×28×28; đầu ra gồm 10 logits. CNN baseline có hai block Conv–ReLU–MaxPool với 16/32 kênh; sau hai lần pooling, tensor thành 32×7×7, Flatten thành 1568 rồi đưa qua Linear128 và Linear10.

> Em giữ Adam, learning rate 0.001, batch 64, seed 42 và 5 epoch cho mỗi cấu hình. Em thay riêng kernel, số kênh, thêm BatchNorm hoặc Dropout để so sánh với baseline. Mô hình được chọn là batchnorm tại epoch 4, validation accuracy 88.58%. Sau khi khóa lựa chọn, em đánh giá 10,000 ảnh test độc lập, đạt accuracy 86.44%.

> Confusion matrix cho thấy lớp Shirt còn khó phân biệt. Kết quả mới có một seed và 5 epoch nên em chưa kết luận cấu hình thắng luôn tốt nhất, hoặc khẳng định overfitting chỉ từ một epoch giảm accuracy. Hướng cải thiện có thể thử là thêm dữ liệu/augmentation, khảo sát learning rate và thời gian học dựa trên validation.

<!-- END VARIANT_DETAILS -->

| Cấu hình | Channels | Kernel | BatchNorm | Dropout |
|---|---|---|---|---|
| baseline | 16/32 | 3 | False | 0.0 |
| kernel5 | 16/32 | 5 | False | 0.0 |
| channels8 | 8/16 | 3 | False | 0.0 |
| channels32 | 32/64 | 3 | False | 0.0 |
| batchnorm | 16/32 | 3 | True | 0.0 |
| dropout03 | 16/32 | 3 | False | 0.3 |

Baseline đại diện kernel3 và channel16 ở block1. Các biến thể bao phủ kernel **3/5** và channel block1 **8/16/32**; block2 có số kênh gấp đôi block1. Mỗi biến thể chỉ đổi một thuộc tính so với baseline để phân tích riêng ảnh hưởng của nó.

| Mô hình | Tham số | Val loss | Val accuracy | Giây/epoch | Epoch báo cáo |
|---|---:|---:|---:|---:|---:|
| baseline | 206,922 | 0.4319 | 86.17% | 0.691 | 5 |
| kernel5 | 215,370 | 0.4003 | 85.75% | 0.895 | 5 |
| channels8 | 103,018 | 0.4650 | 83.67% | 0.389 | 5 |
| channels32 | 421,642 | 0.4130 | 86.58% | 1.547 | 3 |
| batchnorm | 207,018 | 0.3297 | 88.58% | 0.877 | 4 |
| dropout03 | 206,922 | 0.4428 | 83.75% | 0.716 | 5 |

Bảng báo cáo checkpoint validation tốt nhất của mỗi cấu hình; history.csv lưu mọi epoch.

<!-- BEGIN EPOCH_DETAILS -->

### Kết quả đầy đủ từng epoch

**Train loss**: trung bình theo số mẫu trong lúc cập nhật. **Val loss/accuracy**: đánh giá ở cuối epoch. **Δ accuracy**: thay đổi so với epoch trước, tính bằng điểm phần trăm. **Giây**: thời gian train + validation của epoch đó.

Epoch **in đậm** có validation accuracy cao nhất của cấu hình; nếu hòa, chọn validation loss nhỏ hơn. Dấu **Mô hình được chọn** chỉ checkpoint thắng giữa các cấu hình theo validation. Phần bổ sung có ma trận test của từng cấu hình; không dùng test để chọn epoch/mô hình.

[Dữ liệu từng epoch: history.csv](history.csv)

#### baseline

| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.1296 | 0.6584 | 76.92% | — | 0.691 | — |
| 2 | 0.6478 | 0.5777 | 78.75% | +1.83 | 0.685 | — |
| 3 | 0.5521 | 0.4893 | 81.75% | +3.00 | 0.692 | — |
| 4 | 0.4973 | 0.4830 | 82.50% | +0.75 | 0.695 | — |
| **5** | 0.4617 | 0.4319 | 86.17% | +3.67 | 0.689 | Val tốt nhất; Epoch cuối |

Từ epoch 1 → 5: train loss 1.1296 → 0.4617; val loss 0.6584 → 0.4319; val accuracy 76.92% → 86.17%. Accuracy cao nhất là **86.17% tại epoch 5**.

#### kernel5

| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.0543 | 0.5938 | 78.92% | — | 0.875 | — |
| 2 | 0.5893 | 0.5049 | 81.83% | +2.92 | 0.881 | — |
| 3 | 0.4897 | 0.4623 | 84.33% | +2.50 | 0.910 | — |
| 4 | 0.4421 | 0.4371 | 83.83% | -0.50 | 0.891 | — |
| **5** | 0.3991 | 0.4003 | 85.75% | +1.92 | 0.917 | Val tốt nhất; Epoch cuối |

Từ epoch 1 → 5: train loss 1.0543 → 0.3991; val loss 0.5938 → 0.4003; val accuracy 78.92% → 85.75%. Accuracy cao nhất là **85.75% tại epoch 5**.

#### channels8

| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.2316 | 0.7160 | 73.08% | — | 0.387 | — |
| 2 | 0.7038 | 0.6112 | 78.42% | +5.33 | 0.383 | — |
| 3 | 0.6093 | 0.5304 | 80.00% | +1.58 | 0.390 | — |
| 4 | 0.5516 | 0.5207 | 81.92% | +1.92 | 0.397 | — |
| **5** | 0.5141 | 0.4650 | 83.67% | +1.75 | 0.389 | Val tốt nhất; Epoch cuối |

Từ epoch 1 → 5: train loss 1.2316 → 0.5141; val loss 0.7160 → 0.4650; val accuracy 73.08% → 83.67%. Accuracy cao nhất là **83.67% tại epoch 5**.

#### channels32

| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.0012 | 0.5632 | 79.92% | — | 1.542 | — |
| 2 | 0.5681 | 0.5110 | 81.25% | +1.33 | 1.552 | — |
| **3** | 0.4725 | 0.4130 | 86.58% | +5.33 | 1.536 | Val tốt nhất |
| 4 | 0.4133 | 0.3930 | 86.17% | -0.42 | 1.543 | — |
| 5 | 0.3767 | 0.3780 | 86.50% | +0.33 | 1.560 | Epoch cuối |

Từ epoch 1 → 5: train loss 1.0012 → 0.3767; val loss 0.5632 → 0.3780; val accuracy 79.92% → 86.50%. Accuracy cao nhất là **86.58% tại epoch 3**.

Val loss thấp nhất lại ở **epoch 5** (0.3780). Loss và accuracy đo hai khía cạnh khác nhau, nên epoch có loss thấp nhất không nhất thiết có accuracy cao nhất.

#### batchnorm

| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.6996 | 0.4256 | 85.17% | — | 0.870 | — |
| 2 | 0.4287 | 0.5018 | 82.08% | -3.08 | 0.858 | — |
| 3 | 0.3578 | 0.3457 | 88.17% | +6.08 | 0.867 | — |
| **4** | 0.2966 | 0.3297 | 88.58% | +0.42 | 0.866 | Val tốt nhất; **Mô hình được chọn** |
| 5 | 0.2737 | 0.3192 | 87.67% | -0.92 | 0.925 | Epoch cuối |

Từ epoch 1 → 5: train loss 0.6996 → 0.2737; val loss 0.4256 → 0.3192; val accuracy 85.17% → 87.67%. Accuracy cao nhất là **88.58% tại epoch 4**.

Val loss thấp nhất lại ở **epoch 5** (0.3192). Loss và accuracy đo hai khía cạnh khác nhau, nên epoch có loss thấp nhất không nhất thiết có accuracy cao nhất.

#### dropout03

| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.2215 | 0.6842 | 73.92% | — | 0.734 | — |
| 2 | 0.7081 | 0.5576 | 79.42% | +5.50 | 0.705 | — |
| 3 | 0.6114 | 0.5219 | 79.50% | +0.08 | 0.728 | — |
| 4 | 0.5553 | 0.5016 | 81.17% | +1.67 | 0.708 | — |
| **5** | 0.5220 | 0.4428 | 83.75% | +2.58 | 0.707 | Val tốt nhất; Epoch cuối |

Từ epoch 1 → 5: train loss 1.2215 → 0.5220; val loss 0.6842 → 0.4428; val accuracy 73.92% → 83.75%. Accuracy cao nhất là **83.75% tại epoch 5**.

<!-- END EPOCH_DETAILS -->

![Learning curves](learning_curves.png)

Dùng 6.000 ảnh train và 1.200 ảnh validation không giao nhau từ official train; 10.000 ảnh official test chỉ được đọc sau khi đã lưu lựa chọn mô hình.

Mỗi cấu hình giữ seed42, split, Adam, lr0.001, batch64 và 5 epoch; mỗi biến thể chỉ thay một thuộc tính so với baseline.

Baseline đạt validation accuracy tốt nhất 86.17% ở epoch 5, với 206,922 tham số.

Kernel 5×5 mở rộng vùng quan sát mỗi convolution và tăng số phép tính; đạt 85.75% validation accuracy (giảm 0.42 điểm phần trăm so với baseline), 215,370 tham số và 0.895 s/epoch.

Giảm số kênh từ 16/32 xuống 8/16 làm giảm tham số và khả năng biểu diễn; đạt 83.67% validation accuracy (giảm 2.50 điểm phần trăm so với baseline), 103,018 tham số và 0.389 s/epoch.

Tăng số kênh từ 16/32 lên 32/64 làm tăng số tham số và khả năng biểu diễn; đạt 86.58% validation accuracy (tăng 0.42 điểm phần trăm so với baseline), 421,642 tham số và 1.547 s/epoch.

BatchNorm dùng thống kê batch khi train và running statistics khi eval; đạt 88.58% validation accuracy (tăng 2.42 điểm phần trăm so với baseline), 207,018 tham số và 0.877 s/epoch.

Dropout tắt ngẫu nhiên 30% activation sau fc1 khi train và tắt cơ chế này khi eval; đạt 83.75% validation accuracy (giảm 2.42 điểm phần trăm so với baseline), 206,922 tham số và 0.716 s/epoch.

Chọn batchnorm tại epoch 4 theo validation; test accuracy cuối cùng là 86.44%, test loss 0.3762.

Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán; ảnh minh họa là 5 mẫu đúng và 5 mẫu sai đầu tiên theo thứ tự test, không chọn theo mức độ đẹp của kết quả.

Đây là kết quả một seed với 5 epoch, chưa chứng minh biến thể nào luôn tốt hơn; train loss là trung bình khi cập nhật, validation loss dùng eval ở cuối epoch.

![Confusion matrix](confusion_matrix.png)

![5 ảnh đúng](correct_examples.png)

![5 ảnh sai](incorrect_examples.png)

<!-- BEGIN TEST_COMPARISON -->

### Ma trận nhầm lẫn trên test của cả 6 cấu hình

Mỗi ma trận dùng **checkpoint validation tốt nhất của cấu hình đó**, đánh giá trên cùng **10.000 ảnh official test**. Hàng là nhãn thật, cột là dự đoán. Đây là phần bổ sung để đối chiếu lỗi của các cấu hình; mô hình chính vẫn được chọn bằng validation trong `selection.json`.

| Cấu hình | Epoch | Val accuracy | Test accuracy | Đúng / Sai | Ma trận riêng |
|---|---:|---:|---:|---:|---|
| baseline | 5 | 86.17% | 83.11% | 8,311 / 1,689 | [Ảnh](test_all_configs/baseline/confusion_matrix.png) · [CSV](test_all_configs/baseline/confusion_matrix.csv) |
| kernel5 | 5 | 85.75% | 84.59% | 8,459 / 1,541 | [Ảnh](test_all_configs/kernel5/confusion_matrix.png) · [CSV](test_all_configs/kernel5/confusion_matrix.csv) |
| channels8 | 5 | 83.67% | 81.31% | 8,131 / 1,869 | [Ảnh](test_all_configs/channels8/confusion_matrix.png) · [CSV](test_all_configs/channels8/confusion_matrix.csv) |
| channels32 | 3 | 86.58% | 84.04% | 8,404 / 1,596 | [Ảnh](test_all_configs/channels32/confusion_matrix.png) · [CSV](test_all_configs/channels32/confusion_matrix.csv) |
| **batchnorm (được chọn)** | 4 | 88.58% | 86.44% | 8,644 / 1,356 | [Ảnh](test_all_configs/batchnorm/confusion_matrix.png) · [CSV](test_all_configs/batchnorm/confusion_matrix.csv) |
| dropout03 | 5 | 83.75% | 82.23% | 8,223 / 1,777 | [Ảnh](test_all_configs/dropout03/confusion_matrix.png) · [CSV](test_all_configs/dropout03/confusion_matrix.csv) |

Các ô trong hình tổng hợp dùng cùng thang màu 0–1.000; mỗi hàng có 1.000 ảnh thật của lớp tương ứng. Tổng đường chéo chia cho 10.000 bằng test accuracy của cấu hình.

![Ma trận nhầm lẫn của 6 cấu hình](test_all_configs/confusion_matrices.png)

[Bảng test đầy đủ, có test loss](test_all_configs/summary.csv)

<!-- END TEST_COMPARISON -->
