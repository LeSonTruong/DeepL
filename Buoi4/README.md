# Bài tập Buổi 4 — Lê Sơn Trường

Bài làm theo `De_bai/baitap.txt`; giữ đề và hai mã mẫu trong `De_bai/`.

- **[Báo cáo tổng hợp hai bài](BAO_CAO_BUOI4.md)**: mục tiêu, đầu vào/đầu ra,
  cấu hình, bảng kết quả, biểu đồ, phân tích confusion matrix và kết luận.
  Có đầy đủ **40 hàng kết quả epoch**: MLP/CNN và 6 cấu hình CNN, mỗi cấu hình
  5 epoch. Bảng ghi train loss, val loss, val accuracy, mức thay đổi accuracy,
  thời gian; đánh dấu checkpoint val tốt nhất, epoch cuối và mô hình chính được chọn.
- [Notebook thực hành 1](bai03_class_cnn_fashionmnist.ipynb).
- [Notebook thực hành 2](bai03_homework_cnn_variants.ipynb).
- **[Giải thích CNN hai block](GIAI_THICH_CNN_HAI_BLOCK.md)**: Conv2d, ReLU,
  MaxPool hoạt động thế nào, shape từng bước và phần phân loại sau hai block.
  Trong `common.py`, hàm `CNN2Blocks.forward()` được tách rõ từng phép tính.
- **[Quy trình chạy hai bài](QUY_TRINH_CHAY.md)**: chọn kernel, thứ tự chạy
  notebook/terminal, sơ đồ luồng xử lý, từng hàm được gọi và file đầu ra.
  Mỗi notebook và báo cáo tổng hợp có **sơ đồ Flow riêng của từng bài** và
  **sơ đồ Train → Validation**, kèm mã Mermaid để chỉnh sửa.
  Ảnh sơ đồ được nhúng vào notebook để mở là xem được.

- [Thực hành 1](bai03_class_cnn_fashionmnist.py): MLP 784→256→10, CNN hai block, nhật ký shape với batch 1 và 64, bảng so sánh, learning curves và 5 câu nhận xét.
- [Thực hành 2](bai03_homework_cnn_variants.py): baseline, kernel5, channels8, channels32,
  BatchNorm, Dropout0.3; chọn checkpoint/cấu hình bằng validation rồi đánh giá test,
  confusion matrix và 5 ảnh đúng + 5 ảnh sai, có nhận xét từng biến thể.
- [Kết quả thực hành 1](results/thuc_hanh_1/report.md).
- [Kết quả thực hành 2](results/thuc_hanh_2/report.md).
  Báo cáo bài 2 có đầu vào/đầu ra, thiết lập học, bảng shape của đủ 6 cấu hình,
  tham số theo từng block và phần Linear, mức thay đổi so với baseline,
  cách chọn checkpoint, phân tích lỗi test và đoạn báo cáo mẫu để trình bày.
  [Nhật ký shape từng lớp với batch 1/64](results/thuc_hanh_2/shapes.txt) và
  [kiến trúc dạng JSON](results/thuc_hanh_2/architectures.json) được sinh từ model thật.
  Có [hình tổng hợp ma trận của đủ 6 cấu hình](results/thuc_hanh_2/test_all_configs/confusion_matrices.png),
  [bảng kết quả test](results/thuc_hanh_2/test_all_configs/summary.csv) và ảnh/CSV riêng từng cấu hình.

## Chạy từ gốc repository

Khi mở notebook, chọn **Select Kernel → DeepL (PyTorch)** rồi **Run All**.
Notebook đã có output; Run All huấn luyện lại thật, giữ 5 epoch mỗi cấu hình.
Các hàm được lấy trực tiếp từ `.py` và nạp mới khi chạy, tránh dùng nhầm
`common.py` của buổi khác hoặc giữ bản cũ trong kernel. Giữ các file `.py`
cùng thư mục notebook. Nếu vừa cập nhật file, mở lại notebook và Restart Kernel.

```powershell
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe -u Buoi4/bai03_class_cnn_fashionmnist.py
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe -u Buoi4/bai03_homework_cnn_variants.py
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/verify_results.py
```

Nếu dùng môi trường mới, cài `python -m pip install -r Buoi4/requirements.txt` rồi chạy hai file bằng Python của môi trường đó. Không dùng Python hệ thống nếu chưa cài torch.

Tạo/chạy lại notebook và cập nhật báo cáo từ gốc repository:

```powershell
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/build_notebooks.py --execute
# Chỉ cập nhật Markdown từ các kết quả đã lưu, không học lại
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/build_notebooks.py --report-only
# Bổ sung ma trận từ checkpoint rồi cập nhật hiển thị notebook, dùng lại output train
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/bai03_homework_cnn_variants.py --evaluate-only
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/build_notebooks.py --refresh
```

Mở `BAO_CAO_BUOI4.md` bằng Markdown Preview để xem hình. Khi chia sẻ báo cáo,
giữ kèm thư mục `results/` và `figures/` vì đường dẫn ảnh là đường dẫn tương đối.

## Dữ liệu và phép so sánh

FashionMNIST phù hợp máy CPU, được đề cho phép. Đọc IDX trực tiếp từ dữ liệu chính thức, kiểm SHA-256, chuyển float32 /255 giống `ToTensor`; không cần torchvision. Tự dùng cache Buổi 2/3 nếu có, nếu chưa có sẽ tải từ kho chính thức Zalando vào `Buoi4/data/`. Không phụ thuộc mã nguồn các buổi trước.

Hai bài dùng cùng hoán vị `torch.randperm(60000)` seed42: 6.000 ảnh đầu để train, 1.200 ảnh tiếp theo để validation. Chỉ thực hành 2 dùng official test 10.000 ảnh, sau khi đã chọn xong mô hình. Không dùng cách chia train/test của bài Buổi 3.

Bài 2 so sánh kernel **3/5** và channel block1 **8/16/32** (block2 tương ứng
**16/32/64**). Baseline là kernel3, channels16/32; mỗi biến thể chỉ thay một
thuộc tính. BatchNorm và Dropout0.3 được khảo sát riêng trên kiến trúc baseline.

Giữ Adam, lr0.001, batch64, 5 epoch, CPU 2 threads; tự dùng CUDA nếu khả dụng. Reset seed và generator thứ tự batch trước mỗi mô hình. Kiến trúc khác nhau không thể có toàn bộ trọng số khởi tạo giống nhau. Thời gian/epoch bao gồm train và validation, có synchronize CUDA, không gồm lưu checkpoint/vẽ hình. Số đo thời gian phụ thuộc máy và tải hệ thống.

Train loss là trung bình theo số mẫu trong khi cập nhật trọng số; validation dùng `eval()` và `inference_mode()` cuối epoch. Thực hành 1 so sánh epoch cuối. Thực hành 2 chọn epoch tốt nhất cho từng cấu hình và chọn cấu hình tốt nhất bằng validation accuracy; hòa thì dùng validation loss thấp hơn; hòa cả hai giữ kết quả gặp trước. Ma trận test của cả 6 cấu hình được bổ sung để phân tích lỗi; giữ nguyên mô hình chính đã chọn bằng validation.

Mỗi thư mục kết quả có `history.csv`, `summary.csv`, `metadata.json`, `split.json`, learning curves và checkpoint tốt nhất. Thực hành 2 có cấu hình, quyết định chọn mô hình trước test, dự đoán đủ 10.000 ảnh, confusion matrix CSV/PNG và chỉ số 10 ảnh minh họa để đối chiếu. Các file kết quả được tạo từ chạy huấn luyện thật.
