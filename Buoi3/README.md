# Hai bài thực hành — Lê Sơn Trường

Theo `baitap.txt`, tương ứng `Slides_Hoc_sau_B2.pdf` trang **26–27**, và hai file mẫu.
Xem [bảng đối chiếu yêu cầu](yeu_cau_thuc_hanh.md).
Các file mẫu trước khi sửa được giữ trong `ma_nguon_goc/`.

Cell thiết lập nạp lại `common.py`, `data.py` và module vòng học khi Run All,
để kernel không giữ mã cũ sau khi cập nhật. Phần xuất kết quả cũng hỗ trợ
history của loop cũ chưa có các cột device/parameters/lr: lấy metadata từ
summary và cấu hình CPU của bài, giữ nguyên các số đo loss/accuracy.

## Mở bài và xem kết quả

- [Notebook thực hành 1](bai02_class_training_loop.ipynb): vòng lặp học rõ từng bước;
  ba learning rate **.01, .001, .0001**, Adam, MLP 784–256–10,
  6000/1000 ảnh, batch64, **5 epoch**.
- [Notebook thực hành 2](bai02_homework_optimizer_compare.ipynb): SGD **.05**,
  Momentum **.03, momentum=.9**, Adam **.001**; MLP 784–128–10,
  8000/1500 ảnh, batch128, **5 epoch**. Có learning curves và Early Stopping riêng.
- [Kết quả và nhận xét](KET_QUA_VA_NHAN_XET.md): bảng thực đo và nhận xét.
- [Chuẩn bị trả bài](CHUAN_BI_TRA_BAI.md): giải thích ngắn về vòng học, LR,
  optimizer, kết quả và Early Stopping.
- `results/`: CSV từng epoch, bảng so sánh, dự đoán, confusion matrix,
  ảnh đúng/sai và checkpoint `.pt` của từng cấu hình.
- `anh_nop_bai/`: **2 ảnh tổng hợp**, mỗi bài 1 ảnh, tên có Lê Sơn Trường.
  Mở ảnh rồi phóng to, đọc từ trên xuống. Bài 2 có cả kết quả nâng cao.
- `Le_Son_Truong_Buoi3.pdf`: bản ảnh tổng hợp trong một PDF.
- `Le_Son_Truong_Buoi3_Anh_Nop.zip`: chỉ 2 ảnh để giải nén và upload Drive.
- `Le_Son_Truong_Buoi3.zip`: hồ sơ gồm code, notebook, kết quả và ảnh, không đóng gói dữ liệu.

Notebook có code và output huấn luyện thật. **Run All sẽ chạy lại huấn luyện**,
không phải chỉ tải cache. Nếu chỉ xem, mở output đã lưu sẵn.

## Chạy lại

Từ gốc repository có thể dùng môi trường hiện có:

```powershell
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi3/bai02_class_training_loop.py
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi3/bai02_homework_optimizer_compare.py
```

Khi mở notebook trong VS Code, bấm **Select Kernel** ở góc phải, chọn
**DeepL (PyTorch)**, rồi **Restart Kernel → Run All**. Không chọn Python hệ thống
nếu môi trường đó chưa cài torch. Cell đầu in đường dẫn Python thực sự đang dùng.
Đường dẫn môi trường có sẵn là `Buoi1/Buoi1_PyTorch/.venv/Scripts/python.exe`.

Nếu cần đăng ký lại kernel từ gốc repository:

```powershell
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe -m ipykernel install --user --name deepl-pytorch --display-name "DeepL (PyTorch)"
```

Hoặc tạo môi trường mới, cài `python -m pip install -r Buoi3/requirements.txt`,
rồi chọn đúng môi trường đó làm kernel. File `.vscode/settings.json` ở gốc
repository đặt môi trường có PyTorch làm Python mặc định; notebook đã mở vẫn
có thể giữ lựa chọn kernel cũ cho tới khi bạn đổi bằng Select Kernel.

```powershell
# Tạo và chạy thật hai notebook, xuất báo cáo/ảnh/ZIP
python Buoi3/build_submission.py --execute
# Đối chiếu loss, metric, checkpoint, dữ liệu và output
python Buoi3/verify_results.py
# Xuất lại ảnh và cập nhật ZIP từ kết quả đã lưu, không học lại
python Buoi3/build_submission.py --package-only
```

Thay đổi siêu tham số hoặc code huấn luyện phải chạy lại thí nghiệm/notebook trước
khi đóng gói. Chạy lại script chỉ cập nhật CSV/ảnh trong results, không tự làm mới
output notebook; dùng `build_submission.py --execute` để đồng bộ toàn bộ.

## Các lựa chọn được ghi rõ

- Dữ liệu Fashion-MNIST thật, SHA-256 ở `data_manifest.json`. Dùng lại bản tải
  từ Buổi 2 nếu có; nếu không, tải từ kho chính thức của Fashion-MNIST.
  Đọc IDX trực tiếp để không cần torchvision: đúng shape (1,28,28), float32 /255
  như ToTensor. Không thêm normalization, augmentation hay đổi thứ tự ảnh.
- Giữ đúng `range(6000)/range(1000)` và `range(8000)/range(1500)` trong mã mẫu.
  Validation lấy từ test chính thức; sau khi dùng so sánh LR/optimizer, không gọi
  là điểm test độc lập. Không tự đổi split để làm kết quả đẹp hơn.
- Chạy CPU, 2 threads, seed42. Reset khởi tạo và generator mỗi cấu hình để
  cùng trọng số và thứ tự batch; metadata lưu hash đối chiếu. Có in device
  và số tham số, train loss và validation accuracy sau **từng epoch**.
  Các thông tin này cũng có trong `history.csv`, báo cáo và ảnh tổng hợp.
  Không cần CUDA để chạy bài.
- Train loss trên biểu đồ được đo lại ở cuối epoch bằng eval, so sánh cùng cách
  với val loss. `online_train_loss` là trung bình batch trong lúc cập nhật,
  ghi riêng vì mô hình thay đổi giữa các batch.
- Bảng chính lấy epoch cuối. Early Stopping là nhóm nâng cao riêng:
  patience2, min_delta0, vẫn tối đa5 epoch và khôi phục trọng số có val loss tốt nhất.
  Nếu chưa dừng thì ghi đúng chưa dừng, không tăng epoch để tạo hiện tượng.
- Thời gian chính gồm vòng học và đánh giá train/validation từng epoch; không
  gồm nạp dữ liệu, dựng optimizer và vẽ hình. Thao tác restore/đánh giá lại
  checkpoint Early Stopping diễn ra sau phần bấm giờ này.
- LR SGD/Momentum/Adam khác nhau theo mã gốc: so sánh các cấu hình này,
  không suy ra thuật toán nào tốt nhất cho mọi dữ liệu. Chỉ chạy một seed.
- `verification.json` ghi các kiểm tra thực tế; đây không phải bảo đảm kết luận
  thống kê từ nhiều seed. Ảnh nộp là bản xuất kết quả, không giả lập ảnh giao diện tài khoản.

Bộ bài được chuẩn bị cục bộ; bạn giải nén ảnh hoặc chọn hồ sơ rồi upload Drive
và chia sẻ link theo yêu cầu lớp.
