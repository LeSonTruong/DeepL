# Thí nghiệm MLP – Buổi 1

Hai bài dùng scikit-learn, chạy bằng CPU, không cần tải dữ liệu từ Internet.

## Đọc và chạy notebook

Hai notebook là bản chính để học và trình bày, đã lưu sẵn bảng và biểu đồ:

- [Bài 1 – MLP trên make_moons](bai01_class_mlp.ipynb).
- [Bài 2 – Khảo sát MLP trên digits](bai01_homework_mlp_ablation.ipynb).

Mỗi notebook chia thành các phần: nạp thư viện, chọn siêu tham số, xem dữ liệu,
chọn cấu hình, đọc thuật toán huấn luyện, bảng so sánh, biểu đồ và nhận xét.
Phần huấn luyện hiện ngay trong ô code; `common.py` cung cấp các tiện ích dùng chung.
Giữ `common.py` cùng thư mục với notebook khi sao chép hoặc nộp bài.

Để chạy lại, dùng Python 3.10 trở lên. Cài thư viện từ thư mục gốc của repository:

```powershell
python -m pip install -r Buoi1/requirements.txt
```

Trong VS Code, mở file `.ipynb`, chọn **Select Kernel → Python Environments**
và chọn Python đã cài thư viện, rồi nhấn **Run All**. Có thể chạy từng ô từ trên
xuống dưới bằng **Shift+Enter**. VS Code cần extension Python và Jupyter để chạy notebook.

Hoặc mở giao diện Jupyter từ thư mục gốc:

```powershell
python -m notebook Buoi1
```

Muốn chạy thử nhanh, sửa ô siêu tham số thành `EPOCHS = 20`, `SEEDS = [42]`.
Sau khi chỉnh dữ liệu hoặc cấu hình, chạy lại toàn bộ notebook và lưu để cập nhật output.
Kết quả mặc định được ghi đè trong `results/bai1` hoặc `results/bai2`;
có thể sửa `args.output_dir` để lưu một lần chạy riêng.

## Chạy bằng script

Các file `.py` vẫn chạy được từ thư mục gốc:

```powershell
python Buoi1/bai01_class_mlp.py
python Buoi1/bai01_homework_mlp_ablation.py
```

Lệnh mặc định chạy 200 epoch cho mỗi cấu hình, với ba seed 42, 43, 44.
Kết quả được ghi đè trong thư mục tương ứng mỗi lần chạy. Để giữ một lần chạy
riêng hoặc chạy thử nhanh:

```powershell
python Buoi1/bai01_class_mlp.py --epochs 20 --seeds 42 --output-dir Buoi1/results/quick_bai1
python Buoi1/bai01_homework_mlp_ablation.py --epochs 20 --seeds 42 --output-dir Buoi1/results/quick_bai2
```

Có thể thay `--lr 0.001` và `--batch-size 64`. Khi so sánh các cấu hình,
giữ nguyên số epoch, learning rate và batch size.

## Nội dung

- **Bài 1**: `make_moons`, 1.200 mẫu, noise 0.25; MLP hai tầng ẩn cùng độ rộng
  8, 32, 64 hoặc 128 neuron, activation ReLU.
- **Bài 2**: bộ dữ liệu `digits` có sẵn trong scikit-learn, 1.797 ảnh chữ số
  8×8, 10 lớp. Baseline hai tầng ẩn 32 neuron, ReLU. Thay riêng từng yếu tố:
  số tầng 1/2/3, độ rộng 8/32/64/128, ReLU/tanh/sigmoid và L2 alpha 0/0.01/0.1.
  Bài này chọn L2 làm thí nghiệm nâng cao. L2 với Adam khác với weight decay
  tách rời của AdamW.
- **`common.py`**: chia dữ liệu, chuẩn hóa, huấn luyện từng epoch, chọn checkpoint,
  đo loss/accuracy/Macro-F1/số tham số/thời gian và tạo báo cáo.

Train/validation/test được chia phân tầng 60/20/20 với seed cố định 42.
Scaler chỉ fit trên train để tránh rò rỉ dữ liệu. Validation loss quyết định
checkpoint và cấu hình đề xuất; không dùng test để chọn mô hình.
Mỗi lần `partial_fit` là một epoch. Huấn luyện đủ số epoch để quan sát xu hướng
overfitting; số đo trong bảng là số đo tại checkpoint validation tốt nhất.
Learning curve vẫn bao gồm toàn bộ quá trình huấn luyện.

## Sản phẩm nộp

- [Báo cáo Bài 1](results/bai1/report.md): bảng kết quả, nhận xét và biểu đồ.
- [Báo cáo Bài 2](results/bai2/report.md): bảng so sánh, learning curve và nhận xét
  underfitting/overfitting.
- Trong mỗi thư mục kết quả: `summary.csv`, `runs.csv`, `history.csv`,
  `metadata.json`, `comparison.png`, `learning_loss.png`, `learning_accuracy.png`.
  Bài 1 có thêm `decision_boundaries.png`.
- Mã nguồn chính: hai notebook `.ipynb`, `common.py` và `requirements.txt`.
  Các file bài tập `.py` là cách chạy tương đương bằng dòng lệnh.

CSV dùng UTF-8 BOM để mở thuận tiện bằng Excel. Accuracy và Macro-F1 trong
CSV là tỷ lệ [0, 1]. Các biểu đồ có dải/thanh sai số thể hiện độ lệch chuẩn
giữa các seed. Đây là biến động huấn luyện trên cùng một tập dữ liệu, không
phải khoảng tin cậy hay kết quả cross-validation.

MLP học trọng số qua lan truyền ngược: mỗi tầng biến đổi tuyến tính rồi áp dụng
activation, đầu ra dự đoán xác suất lớp. Adam cập nhật trọng số để giảm
cross-entropy. Tăng số tầng/neuron có thể học ranh giới phức tạp hơn nhưng
cũng tăng số tham số và nguy cơ ghi nhớ train. L2 phạt trọng số lớn;
checkpoint theo validation giúp giữ thời điểm mô hình tổng quát hóa tốt hơn.
