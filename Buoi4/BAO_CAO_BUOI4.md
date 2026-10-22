# Báo cáo hai bài thực hành CNN — Buổi 4

**Sinh viên: Lê Sơn Trường**

Nguồn: [Đề bài](De_bai/baitap.txt) và hai file gợi ý trong `De_bai/`.

## 1. Mục tiêu, đầu vào và đầu ra

Phân loại ảnh Fashion-MNIST thành 10 loại trang phục. Thực hành 1 so sánh MLP với CNN; thực hành 2 khảo sát kernel, số channel, BatchNorm và Dropout.

- Đầu vào: ảnh xám `(N,1,28,28)`, pixel float32 /255; nhãn long 0–9.
- Đầu ra: `(N,10)` logits; lấy argmax để xác định lớp dự đoán. CrossEntropyLoss nhận logits thô, không thêm Softmax trước loss.
- Train/validation: 6.000/1.200 mẫu không giao nhau từ official train, chia bằng hoán vị seed42. Test cuối: toàn bộ 10.000 ảnh official test, chỉ dùng ở bài 2.

## 2. Quy trình và cấu hình chung

Adam, LR=0.001, batch=64, 5 epoch, seed=42; thiết bị **cpu**, 2 threads; PyTorch 2.14.1+cpu.

Vòng học: zero_grad → forward → loss → backward → step. Đánh giá dùng eval và inference_mode. Loss được tính trung bình theo số mẫu; train loss ghi trong lúc cập nhật, validation loss đo cuối epoch. Thời gian/epoch gồm train + validation, không gồm ghi checkpoint/vẽ hình.

Mỗi mô hình reset seed và thứ tự batch. Mô hình khác kiến trúc có số lượng/kích thước trọng số khác nhau, nên cùng seed không có nghĩa toàn bộ trọng số khởi tạo giống nhau.

### Flow Train → Validation bên trong mỗi epoch

![Flow Train → Validation bên trong mỗi epoch](figures/flow_train_validation.png)

<details>
<summary>Mã Mermaid để chỉnh sửa sơ đồ</summary>

```mermaid
flowchart TD
    N0["Bắt đầu epoch: TRAIN<br/>model.train()"]
    N1["Với mỗi batch train<br/>zero_grad → forward → CrossEntropyLoss<br/>→ backward → optimizer.step"]
    N0 --> N1
    N2["Hết các batch train: VALIDATION<br/>model.eval() + inference_mode()"]
    N1 --> N2
    N3["Duyệt mọi batch validation<br/>Tính val loss trung bình và val accuracy"]
    N2 --> N3
    N4["Ghi train loss, val loss/accuracy, giây/epoch<br/>Lưu checkpoint nếu validation tốt hơn"]
    N3 --> N4
    N5["Đủ 5 epoch: kết thúc mô hình/cấu hình"]
    N4 --> N5
    N4 -->|"Chưa đủ 5 epoch"| N0
```

</details>

**Train cập nhật trọng số. Validation chỉ đo chất lượng và chọn checkpoint.** Bước cập nhật được thực hiện sau mỗi batch train; validation chạy ở cuối epoch. Cả hai bài đều dùng vòng học này. Test riêng của bài 2 diễn ra sau khi đã chọn mô hình.


### CNN hai block: dữ liệu đi qua mạng thế nào?

CNN **hai block** nghĩa là lặp hai lần cụm
**Conv2d → ReLU → MaxPool2d(2)**. Sau hai block mới đến phần phân loại.
Trong `common.py`, xem các chú thích `BLOCK 1`, `BLOCK 2`, `PHÂN LOẠI`
ở hàm `CNN2Blocks.forward()`.

Ví dụ dưới đây dùng **baseline**: kernel3, channels16/32, không BatchNorm/Dropout.
`N` là số ảnh trong batch (thường là 64), không phải số kênh.
Shape luôn có thứ tự **(số ảnh, số kênh, chiều cao, chiều rộng)**.

```text
Ảnh xám (N,1,28,28)
  │
  ├─ BLOCK 1
  │   Conv2d(1,16,3,padding=1) → (N,16,28,28)
  │   ReLU                    → (N,16,28,28)
  │   MaxPool2d(2)            → (N,16,14,14)
  │
  ├─ BLOCK 2
  │   Conv2d(16,32,3,padding=1)→ (N,32,14,14)
  │   ReLU                    → (N,32,14,14)
  │   MaxPool2d(2)            → (N,32,7,7)
  │
  └─ PHÂN LOẠI
      Flatten → (N,1568) → Linear → (N,128) → ReLU
      → Linear → (N,10) logits → argmax → nhãn dự đoán
```

**Bên trong từng phép tính:**

| Phép tính | Thực hiện thế nào? | Ý nghĩa |
|---|---|---|
| Conv2d | Trượt bộ lọc trên ảnh/đặc trưng. Tại mỗi vị trí, nhân từng giá trị với trọng số tương ứng, cộng lại trên cả vùng và các kênh đầu vào, rồi cộng bias. | Học đặc trưng cục bộ; dùng cùng trọng số ở các vị trí. |
| ReLU | Tính `max(0,x)`: số âm thành 0, số dương giữ nguyên. Không đổi shape. | Tạo tính phi tuyến, giúp mạng học quan hệ phức tạp. |
| MaxPool2d(2) | Mỗi kênh chia thành vùng 2×2, lấy giá trị lớn nhất; mặc định stride=2. | Giảm chiều cao/rộng còn một nửa, giữ số kênh. |

**Cụ thể với hai block:** Conv1 có 16 bộ lọc, mỗi bộ kích thước
`1×3×3`, tạo 16 bản đồ đặc trưng. Conv2 có 32 bộ lọc, mỗi bộ
`16×3×3`: mỗi giá trị đầu ra tổng hợp thông tin từ **cả 16 kênh**
của block 1, tạo 32 bản đồ đặc trưng mới. Các bộ lọc do mạng học
trong lúc huấn luyện, không phải bộ dò cạnh được đặt sẵn.

`padding=1` thêm viền 0 để convolution kernel3, stride1 giữ nguyên kích thước.
Công thức với dilation=1 là `H_out = floor((H + 2p - k)/s) + 1`;
ở conv1: `(28 + 2×1 - 3)/1 + 1 = 28`. Pooling mới giảm
`28→14` rồi `14→7`. Số ảnh trong batch luôn được giữ nguyên.

**Ví dụ nhỏ cho ReLU và MaxPool** (số minh họa, không phải kết quả huấn luyện):

```text
Một vùng đặc trưng    Sau ReLU        Sau MaxPool2d(2)
[-2,  3]             [0, 3]
[ 1, -4]       →     [1, 0]      →    3
```

**Sau hai block:** `32×7×7 = 1568` giá trị được trải phẳng cho mỗi ảnh.
`Linear(1568,128) → ReLU → Linear(128,10)` tạo 10 điểm số **logits**.
`argmax` chọn lớp có điểm cao nhất. Khi học, đưa logits trực tiếp vào
`CrossEntropyLoss`; không cần thêm Softmax trước loss.

**Các biến thể trong bài 2:**

- Kernel5 dùng padding2 nên vẫn giữ nguyên H,W sau convolution.
- Channels8 nghĩa là hai block có 8/16 kênh; channels32 nghĩa là 32/64 kênh.
  H,W vẫn đi theo `28→14→7`; số chiều sau Flatten thay đổi theo số kênh.
- BatchNorm: mỗi block thành `Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)`.
  Ở baseline, `bn1/bn2` là `Identity`, cho dữ liệu đi qua nguyên vẹn.
- Dropout0.3 nằm **sau ReLU của Linear128**, ngoài hai block convolution.
  Nó ngẫu nhiên đặt một phần giá trị về 0 khi train; tắt khi eval.

**Liên hệ với quá trình học:** luồng trên là **forward**. Tiếp theo,
`loss → backward → optimizer.step()` tính lỗi, truyền gradient ngược
và cập nhật trọng số. Conv2d và Linear có tham số học được; ReLU,
MaxPool và Flatten không có trọng số học được. Với biến thể BatchNorm,
các tham số scale/bias của BatchNorm cũng được học.


## 3. Thực hành 1

### Quy trình chạy — Thực hành 1: so sánh MLP–CNN

![Flow bài 1 — So sánh MLP và CNN](figures/flow_thuc_hanh_1.png)

<details>
<summary>Mã Mermaid để chỉnh sửa sơ đồ</summary>

```mermaid
flowchart TD
    N0["Nạp FashionMNIST từ official train"]
    N1["Chia 6.000 train + 1.200 validation"]
    N0 --> N1
    N2["In shape CNN từng lớp với batch 1 và 64"]
    N1 --> N2
    N3["Lần lượt tạo MLP và CNN<br/>Cùng seed, split, Adam, LR, batch, số epoch"]
    N2 --> N3
    N4["Mỗi mô hình: Train → Validation<br/>Lặp đủ 5 epoch; lưu log và checkpoint tốt nhất"]
    N3 --> N4
    N5["So sánh MLP–CNN tại epoch 5<br/>Số tham số, val accuracy, thời gian/epoch"]
    N4 --> N5
    N6["Xuất bảng kết quả, learning curve và nhận xét"]
    N5 --> N6
```

</details>


**Luồng đọc nhanh:**

```text
Nạp official train → Chia 6000 train / 1200 validation
→ In shape → Lần lượt tạo MLP và CNN cùng optimizer
→ Với mỗi mô hình: 5 × [Train 1 epoch → Validation → Ghi log/checkpoint]
→ So sánh hai mô hình tại epoch 5
→ Lưu bảng kết quả, learning curve và nhận xét
```

| Bước | Công việc | Hàm thực hiện | Kết quả / ý nghĩa |
|---:|---|---|---|
| 1 | Khởi tạo | `main()` → `set_seed()` | Đặt seed42, xác định CPU/GPU; tạo thư mục `results/thuc_hanh_1/`. |
| 2 | Nạp và chia dữ liệu | `make_data()` → `read_official(train=True)` | Kiểm SHA-256; lấy 6.000 train và 1.200 validation từ hoán vị official train; pixel float32 /255, nhãn long. Lưu `split.json`. |
| 3 | Theo dõi tensor | `print_shapes()` | Chạy input giả với batch 1 và 64 qua conv1, pool1, conv2, pool2, Flatten và đầu ra; lưu `shapes.txt`. |
| 4 | Tạo từng mô hình | Vòng lặp MLP/CNN trong `main()` → `fit()` | Reset seed và thứ tự batch; tạo model trên device, Adam LR0.001, CrossEntropyLoss; đếm tham số. |
| 5 | Huấn luyện một epoch | `fit()` → `train_one_epoch()` | Bật train(); với mỗi batch64: zero_grad → forward → loss → backward → step. Cộng loss theo số mẫu. |
| 6 | Đánh giá và lặp | `fit()` → `evaluate()` | Bật eval(), dùng inference_mode trên 1.200 ảnh validation; ghi val loss/accuracy và giây/epoch; lưu checkpoint nếu validation tốt hơn. Lặp đủ 5 epoch rồi chuyển sang mô hình tiếp theo. |
| 7 | So sánh và xuất kết quả | Phần cuối `main()` | So sánh MLP–CNN tại epoch cuối bằng tham số, val accuracy và thời gian trung bình/epoch; xuất CSV, learning curve và báo cáo. |

Checkpoint tốt nhất vẫn được lưu để tham khảo. **Bảng so sánh chính của bài 1 dùng epoch 5**, đồng thời bảng chi tiết giữ đủ các epoch 1–5. Bài 1 chỉ dùng train/validation.

**File kết quả trong `results/thuc_hanh_1/`:** `split.json`, `shapes.txt`, `history.csv`, `summary.csv`, `metadata.json`, `MLP_best.pt`, `CNN_best.pt`, `learning_curves.png`, `report.md`.


| Mô hình | Tham số | Val loss | Val accuracy | Giây/epoch | Epoch báo cáo |
|---|---:|---:|---:|---:|---:|
| MLP | 203,530 | 0.4766 | 82.83% | 0.100 | 5 |
| CNN | 206,922 | 0.4319 | 86.17% | 0.687 | 5 |

Bảng dùng epoch cuối (5); cùng seed42, split6000/1200, Adam, lr0.001, batch64.

<!-- BEGIN EPOCH_DETAILS -->

### Kết quả đầy đủ từng epoch

**Train loss**: trung bình theo số mẫu trong lúc cập nhật. **Val loss/accuracy**: đánh giá ở cuối epoch. **Δ accuracy**: thay đổi so với epoch trước, tính bằng điểm phần trăm. **Giây**: thời gian train + validation của epoch đó.

Epoch **in đậm** có validation accuracy cao nhất của cấu hình; nếu hòa, chọn validation loss nhỏ hơn. Bảng tổng kết bài 1 vẫn dùng epoch cuối để so sánh cùng ngân sách.

[Dữ liệu từng epoch: history.csv](results/thuc_hanh_1/history.csv)

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

![Learning curves](results/thuc_hanh_1/learning_curves.png)

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


## 4. Thực hành 2

### Quy trình chạy — Thực hành 2: khảo sát các biến thể CNN

![Flow bài 2 — Các biến thể CNN](figures/flow_thuc_hanh_2.png)

<details>
<summary>Mã Mermaid để chỉnh sửa sơ đồ</summary>

```mermaid
flowchart TD
    N0["Nạp official train<br/>Chia 6.000 train + 1.200 validation như bài 1"]
    N1["Tạo 6 cấu hình CNN; kiểm tra shape và tham số<br/>Baseline, kernel5, channels8, channels32, BatchNorm, Dropout0.3"]
    N0 --> N1
    N2["Mỗi cấu hình: Train → Validation, đủ 5 epoch<br/>Giữ checkpoint validation tốt nhất của từng cấu hình"]
    N1 --> N2
    N3["Chọn cấu hình thắng bằng validation<br/>Accuracy cao nhất; hòa thì loss thấp hơn"]
    N2 --> N3
    N4["Lưu selection.json rồi nạp checkpoint đã chọn"]
    N3 --> N4
    N5["Đánh giá mô hình đã chọn và các checkpoint còn lại<br/>Cùng 10.000 ảnh official test; eval() + inference_mode()"]
    N4 --> N5
    N6["Xuất ma trận nhầm lẫn của 6 cấu hình<br/>5 ảnh đúng/sai của mô hình chính, learning curve và nhận xét"]
    N5 --> N6
```

</details>


**Luồng đọc nhanh:**

```text
Nạp official train → Chia 6000 train / 1200 validation
→ Với mỗi cấu hình: tạo CNN/optimizer rồi chạy
  5 × [Train 1 epoch → Validation → Cập nhật checkpoint tốt nhất]
→ Chọn cấu hình thắng và lưu selection.json → Nạp official test
→ Đánh giá 10000 ảnh test → Ma trận mô hình chính và cả 6 cấu hình
→ 5 ảnh đúng + 5 ảnh sai của mô hình chính
→ Xuất bảng so sánh, learning curve và báo cáo
```

| Bước | Công việc | Hàm thực hiện | Kết quả / ý nghĩa |
|---:|---|---|---|
| 1 | Khởi tạo và chia dữ liệu | `main()` → `set_seed()` → `make_data()` | Cùng split6000/1200 từ official train như bài 1; xác định device. Chưa nạp test ở bước này. |
| 2 | Khai báo và xem kiến trúc | `experiments()` → `export_architectures()` → `describe_cnn()` | Baseline; kernel5; channels8; channels32; BatchNorm; Dropout0.3. Mỗi biến thể chỉ đổi một thuộc tính; lưu cấu hình, shape từng lớp với batch 1/64 và tham số theo block/phân loại. Kiểm tra shape bằng input giả ở eval, trước khi tính thời gian huấn luyện. |
| 3 | Học từng cấu hình | Vòng lặp trong `main()` → `fit()` | Reset seed và batch order, dựng CNN và Adam LR0.001; chạy đủ 5 epoch. Mỗi epoch học trên train rồi đo val loss/accuracy, ghi thời gian. |
| 4 | Giữ checkpoint từng cấu hình | Trong `fit()` | Chọn epoch có val accuracy lớn nhất; hòa thì dùng val loss nhỏ hơn, hòa cả hai giữ epoch gặp trước. Lưu `<tên>_best.pt`. |
| 5 | Chọn cấu hình thắng | Sau vòng lặp cấu hình trong `main()` | So sánh các checkpoint tốt nhất bằng cùng quy tắc validation. Lưu quyết định vào `selection.json` trước khi gọi đánh giá test. |
| 6 | Đánh giá cuối và đối chiếu lỗi | Nạp checkpoint → `final_test()` → `evaluate_all_configurations()` | Nạp đủ 10.000 ảnh official test; eval() và inference_mode(), không cập nhật trọng số. Xuất test loss/accuracy và ma trận của mô hình đã chọn; bổ sung ma trận cả 6 checkpoint tốt nhất theo validation để đối chiếu lỗi. |
| 7 | Xuất kết quả | `final_test()` và phần cuối `main()` | Lưu đủ dự đoán test, ma trận CSV/PNG, 5 ảnh đúng và 5 ảnh sai đầu tiên có nhãn thật/dự đoán; hoàn thiện bảng cấu hình, learning curve, kết quả từng epoch và nhận xét. |

**Chọn epoch và cấu hình hoàn toàn bằng validation.** Các ma trận test của 6 cấu hình là phân tích bổ sung; giữ nguyên mô hình chính đã chọn trong `selection.json`.

**File kết quả trong `results/thuc_hanh_2/`:** `split.json`, `configurations.json`, `architectures.json`, `shapes.txt`, `history.csv`, `summary.csv`, `metadata.json`, các checkpoint `*_best.pt`, `selection.json`, `test_metrics.json`, `test_predictions.csv`, `confusion_matrix.csv/png`, `example_indices.csv`, `correct_examples.png`, `incorrect_examples.png`, `learning_curves.png`, `report.md`, `test_all_configs/summary.csv`, `test_all_configs/confusion_matrices.png` và thư mục ma trận/dự đoán riêng cho mỗi cấu hình.


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

[Nhật ký shape từng lớp, batch 1 và 64](results/thuc_hanh_2/shapes.txt) · [Kiến trúc và tham số dạng JSON](results/thuc_hanh_2/architectures.json)

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
3. Nạp [batchnorm_best.pt](results/thuc_hanh_2/batchnorm_best.pt) ở epoch **4**, có validation accuracy **88.58%**, validation loss **0.3297**.
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

[Dữ liệu từng epoch: history.csv](results/thuc_hanh_2/history.csv)

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

![Learning curves](results/thuc_hanh_2/learning_curves.png)

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

![Confusion matrix](results/thuc_hanh_2/confusion_matrix.png)

![5 ảnh đúng](results/thuc_hanh_2/correct_examples.png)

![5 ảnh sai](results/thuc_hanh_2/incorrect_examples.png)

<!-- BEGIN TEST_COMPARISON -->

### Ma trận nhầm lẫn trên test của cả 6 cấu hình

Mỗi ma trận dùng **checkpoint validation tốt nhất của cấu hình đó**, đánh giá trên cùng **10.000 ảnh official test**. Hàng là nhãn thật, cột là dự đoán. Đây là phần bổ sung để đối chiếu lỗi của các cấu hình; mô hình chính vẫn được chọn bằng validation trong `selection.json`.

| Cấu hình | Epoch | Val accuracy | Test accuracy | Đúng / Sai | Ma trận riêng |
|---|---:|---:|---:|---:|---|
| baseline | 5 | 86.17% | 83.11% | 8,311 / 1,689 | [Ảnh](results/thuc_hanh_2/test_all_configs/baseline/confusion_matrix.png) · [CSV](results/thuc_hanh_2/test_all_configs/baseline/confusion_matrix.csv) |
| kernel5 | 5 | 85.75% | 84.59% | 8,459 / 1,541 | [Ảnh](results/thuc_hanh_2/test_all_configs/kernel5/confusion_matrix.png) · [CSV](results/thuc_hanh_2/test_all_configs/kernel5/confusion_matrix.csv) |
| channels8 | 5 | 83.67% | 81.31% | 8,131 / 1,869 | [Ảnh](results/thuc_hanh_2/test_all_configs/channels8/confusion_matrix.png) · [CSV](results/thuc_hanh_2/test_all_configs/channels8/confusion_matrix.csv) |
| channels32 | 3 | 86.58% | 84.04% | 8,404 / 1,596 | [Ảnh](results/thuc_hanh_2/test_all_configs/channels32/confusion_matrix.png) · [CSV](results/thuc_hanh_2/test_all_configs/channels32/confusion_matrix.csv) |
| **batchnorm (được chọn)** | 4 | 88.58% | 86.44% | 8,644 / 1,356 | [Ảnh](results/thuc_hanh_2/test_all_configs/batchnorm/confusion_matrix.png) · [CSV](results/thuc_hanh_2/test_all_configs/batchnorm/confusion_matrix.csv) |
| dropout03 | 5 | 83.75% | 82.23% | 8,223 / 1,777 | [Ảnh](results/thuc_hanh_2/test_all_configs/dropout03/confusion_matrix.png) · [CSV](results/thuc_hanh_2/test_all_configs/dropout03/confusion_matrix.csv) |

Các ô trong hình tổng hợp dùng cùng thang màu 0–1.000; mỗi hàng có 1.000 ảnh thật của lớp tương ứng. Tổng đường chéo chia cho 10.000 bằng test accuracy của cấu hình.

![Ma trận nhầm lẫn của 6 cấu hình](results/thuc_hanh_2/test_all_configs/confusion_matrices.png)

[Bảng test đầy đủ, có test loss](results/thuc_hanh_2/test_all_configs/summary.csv)

<!-- END TEST_COMPARISON -->


## 5. Phân tích confusion matrix trên test

Hàng là nhãn thật, cột là nhãn dự đoán. Đường chéo chính đếm dự đoán đúng; các ô ngoài đường chéo chỉ ra những lớp bị nhầm lẫn.

Tổng đường chéo là **8,644/10,000** ảnh, tương ứng accuracy **86.44%**.

- Lớp khó nhất trong lần chạy: **Shirt**, nhận đúng 623/1000 ảnh (**62.30%**).
- Lớp nhận đúng cao nhất: **Ankle boot**, 978/1000 ảnh (**97.80%**).
- Nhầm **Coat → Pullover**: 127 ảnh.
- Nhầm **Shirt → T-shirt/top**: 119 ảnh.
- Nhầm **Shirt → Pullover**: 117 ảnh.

Accuracy tổng thể chưa thể hiện hết chênh lệch giữa các lớp; nên xem thêm các ảnh dự đoán sai để tìm nguyên nhân trước khi đề xuất cải thiện.

## 6. Kết luận và giới hạn

Mô hình được chọn là **batchnorm tại epoch 4**, validation accuracy **88.58%**. Đánh giá cuối trên test đạt **86.44%**, test loss **0.3762**.

CNN khai thác thông tin cục bộ và dùng chung trọng số theo vị trí; số tham số gần MLP không có nghĩa số phép tính hoặc thời gian tương đương. Các biến thể được so sánh trong cùng điều kiện; chỉ một seed và 5 epoch nên kết luận giới hạn ở thí nghiệm này. Các ma trận test của cả 6 cấu hình được bổ sung để đối chiếu lỗi; mô hình chính vẫn được chọn bằng validation trong selection.json.

## 7. Mã nguồn và notebook

- [Quy trình chạy notebook/terminal và các bước xử lý](QUY_TRINH_CHAY.md).
- [Notebook thực hành 1](bai03_class_cnn_fashionmnist.ipynb) — [mã Python](bai03_class_cnn_fashionmnist.py).
- [Notebook thực hành 2](bai03_homework_cnn_variants.ipynb) — [mã Python](bai03_homework_cnn_variants.py).
- [Code dùng chung](common.py); [kiểm tra checkpoint/kết quả](verify_results.py).
