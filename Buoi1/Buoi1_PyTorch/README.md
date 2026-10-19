# Buổi 1 – Bản PyTorch bám theo mã mẫu và bai1.txt

Bản này nằm trong thư mục riêng. Bản sao nguyên văn của đề nằm ở `bai1.txt`,
được chép từ `../Buoi1/bai1.txt`.

## Những thiết lập giữ từ mã ban đầu

| Thiết lập | Bài 1 – trên lớp | Bài 2 – BTVN |
|---|---|---|
| Framework | PyTorch | PyTorch |
| Dữ liệu | make_moons, 800 mẫu, noise=0.25 | make_moons, 1.200 mẫu, noise=0.28 |
| Seed dữ liệu/chia tập | 42 | 42 |
| Train/validation | random_split 640/160 | random_split 960/240 |
| Epoch | **20** | **15** |
| Batch train/validation | 32/128 | 32/256 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.01 và 0.001 |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Mạng cơ sở | 2 → 32 → 32 → 2, ReLU | 2 → 32 → 32 → 2, ReLU |

Không chuẩn hóa dữ liệu, không đổi sang digits, không tăng lên 200 epoch.
Đánh giá mô hình ở **epoch cuối**, không chọn checkpoint hay thêm test riêng.
Các cấu hình cùng seed 42; đặt lại seed trước mỗi cấu hình để so sánh công bằng.
TrainLoader dùng generator riêng seed 42 để thứ tự batch có thể tái lập.

## Bổ sung để hoàn thành đề

- **Bài 1:** thay hai tầng ẩn cùng độ rộng 8, 32, 64, 128; so sánh validation
  loss, accuracy, Macro-F1, tham số và thời gian.
- **Bài 2:** giữ khảo sát hidden size × learning rate của BTVN; bổ sung 128
  neuron, số tầng 1/2/3, ReLU/tanh/sigmoid và mức nâng cao Dropout 0.2 hoặc
  Weight Decay 0.0001. Cả hai cách regularization được thử riêng.
- Các khảo sát tầng/activation/regularization chỉ đổi một yếu tố so với
  baseline hai tầng 32 neuron, ReLU, LR=0.001. Không dùng tích đầy đủ mọi yếu tố.
- `common.py` mới định nghĩa `seed_everything`, `train_classifier` và các hàm
  tạo mô hình, đánh giá, xuất CSV, biểu đồ, nhận xét.

## Chạy code

Từ thư mục gốc repository trên Windows:

```powershell
python -m venv Buoi1_PyTorch/.venv
Buoi1_PyTorch/.venv/Scripts/python.exe -m pip install -r Buoi1_PyTorch/requirements.txt
Buoi1_PyTorch/.venv/Scripts/python.exe Buoi1_PyTorch/bai01_class_mlp.py
Buoi1_PyTorch/.venv/Scripts/python.exe Buoi1_PyTorch/bai01_homework_mlp_ablation.py
```

Môi trường `.venv` hiện tại đã được tạo với `--system-site-packages` để dùng
các thư viện khoa học có sẵn, và cài riêng PyTorch CPU. Nếu tạo lại trên máy
khác, dùng các lệnh trên để cài đầy đủ thư viện vào môi trường mới.

## Notebook và sản phẩm nộp

- `bai01_class_mlp.ipynb` và `bai01_homework_mlp_ablation.ipynb`: bản notebook
  cùng cấu hình với script, có giải thích và hiển thị kết quả trực tiếp.
  Chọn kernel `Buoi1_PyTorch/.venv/Scripts/python.exe`, rồi **Run All**.
- [Báo cáo Bài 1](results/bai1/report.md).
- [Bốn hình điểm và ranh giới phân loại Bài 1](results/bai1/decision_boundaries.png):
  cùng dữ liệu, lần lượt 8/32/64/128 neuron mỗi tầng; đường đen là ranh giới,
  dấu × là điểm validation dự đoán sai. Hiển thị trực tiếp trong notebook Bài 1.
- [Biểu đồ so sánh 8/32/64/128 neuron Bài 1](results/bai1/comparison_neurons.png):
  Train/Validation loss, accuracy, Macro-F1 ở epoch cuối; số tham số, thời gian
  và bốn đường validation loss trên cùng một hình. Hình cũng hiển thị trong notebook Bài 1.
- [Báo cáo Bài 2](results/bai2/report.md).
- Mỗi thư mục kết quả có `results.csv`, `history.csv`, `metadata.json`,
  `comparison.png`, `learning_loss.png`, `learning_accuracy.png`, `report.md`.

Nộp source `.py`, notebook `.ipynb`, `requirements.txt` và `results/`;
không cần nộp `.venv`.

## Cách hiểu kết quả

Loss càng thấp càng tốt; accuracy và Macro-F1 càng cao càng tốt.
Macro-F1 là trung bình F1 của từng lớp. Số tham số gồm cả trọng số và bias.
Learning curve đo train/validation ở chế độ eval để tắt Dropout khi so sánh.

Nếu train và validation đều chưa tốt, có thể underfitting hoặc chưa hội tụ.
Nếu train tốt lên nhưng validation xấu đi thì có dấu hiệu overfitting.
Với 15/20 epoch, có thể chưa quan sát được overfitting: chỉ nhận xét theo
số đo thực tế. Chỉ có một seed và không có test riêng nên không khẳng định
kết quả tốt nhất trên mọi lần chạy hoặc trên dữ liệu mới độc lập.
