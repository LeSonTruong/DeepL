# Bài tập Buổi 2 — Lê Sơn Trường

Theo `../baitap.txt`, đọc D2L và làm **toàn bộ Exercises chương 1–5**.
Lời giải cho **162 câu chính thuộc 28 nhóm Exercises** đối chiếu
**PDF `../d2l-en.pdf`** bạn cung cấp (D2L 1.0.3), không tự
đổi thành năm ví dụ minh họa. Mục 2.7 không có Exercises đánh số.

Đã rà soát lại câu hỏi/công thức/kết quả; xem [Biên bản kiểm tra](KIEM_TRA_LAI.md).
Notebook có khung đề gốc để đối chiếu. `verify_results.py` kiểm tra lại PDF,
các đồng nhất thức toán và tính nhất quán của kết quả; kết quả ở `audit_checks.json`.

## Mở bài và xem kết quả

- [Notebook Chương 1](Le_Son_Truong_Chuong_1.ipynb): Introduction.
- [Notebook Chương 2](Le_Son_Truong_Chuong_2.ipynb): tensor, pandas, đại số, giải tích, autograd, xác suất.
- [Notebook Chương 3](Le_Son_Truong_Chuong_3.ipynb): hồi quy, SGD, loss bền vững, generalization, weight decay.
- [Notebook Chương 4](Le_Son_Truong_Chuong_4.ipynb): softmax, Fashion-MNIST, ổn định số, distribution shift.
- [Notebook Chương 5](Le_Son_Truong_Chuong_5.ipynb): MLP, khởi tạo, dropout, backprop, House Prices.
- `ON_TAP_LOSS_BACKPROP_MLP.md`: phần kiến thức cần chuẩn bị để trả bài.
- `Le_Son_Truong_Buoi2.pdf`, `anh_nop_bai/`: bản lời giải/kết quả xuất để đọc hoặc đưa lên Drive.
- **`anh_nop_bai_gop/`**: bộ gọn để upload Drive, **10 ảnh** thay cho 137 ảnh lẻ.
  Chương 1 có 1 ảnh, chương 2–5 mỗi chương 2 ảnh; phần ôn tập có 1 ảnh.
  Ảnh giữ nguyên cỡ chữ, cắt lề trắng; phóng to rồi đọc từ trên xuống mỗi cột,
  sau đó sang cột bên phải. File ZIP riêng: `../Le_Son_Truong_Buoi2_Anh_Gop.zip`.
- `coverage.csv`: đối chiếu số câu từng mục và các phần còn cần thao tác cá nhân.

Notebook chứa lời giải từng câu và output thật của các thí nghiệm. Nguồn chi tiết
ở `experiments.py`, `house_prices.py`, `data.py` và `loi_giai/chuong*.md`.
Khi chỉ đọc, không cần tải lại dữ liệu hoặc huấn luyện lại.

## Chạy lại

Hiện tại có thể chọn kernel đã có PyTorch:
`Buoi1/Buoi1_PyTorch/.venv/Scripts/python.exe` từ gốc repository.
Hoặc tạo môi trường mới và cài `requirements.txt` trong thư mục bài làm.
Mở notebook từ thư mục này hoặc từ gốc repository rồi chọn **Run All**.

Mặc định `RECOMPUTE=False`: đọc các bảng/ảnh đã tính và lưu ở `results/`.
Để tính lại thực sự, đổi thành `True`; chuẩn bị dữ liệu trước bằng:

```powershell
# Chạy từ gốc repository, dùng Python đã có các requirements.
python Buoi2/bai_lam_Le_Son_Truong/data.py
python Buoi2/bai_lam_Le_Son_Truong/experiments.py --recompute
```

Hoặc chạy riêng mục, ví dụ `--sections 2.5 5.6 --recompute`.
`build_submission.py --execute` tạo lại notebook/bản PDF/ảnh; cần kernel cục bộ
được phép khởi chạy Jupyter. Thay đổi code hoặc siêu tham số phải tính lại mục
tương ứng trước khi xuất, vì cache không tự phát hiện mọi thay đổi tham số.

Chỉ gộp lại ảnh sẵn có, không chạy notebook hay huấn luyện lại:
`python Buoi2/bai_lam_Le_Son_Truong/merge_submission_images.py` (từ gốc repository).
`build_submission.py` cũng tự gộp ảnh trước khi đóng gói. `anh_nop_bai_gop/manifest.json`
ghi đối chiếu từng trang; chương trình kiểm tra đủ trang và pixel nội dung sau khi lưu PNG.

## Các thiết lập và giới hạn được ghi rõ

- Fashion-MNIST dùng **đúng dữ liệu gốc**, đủ 60.000 train và 10.000 test như
  validation theo ví dụ sách; vì đã dùng tuning, đây không còn là test độc lập.
- Baseline MLP: 256 neuron, SGD LR=0.1, batch256,10 epoch. Các lần tăng epoch,
  đổi neuron/LR/activation được đánh dấu là khảo sát Exercises, không thay baseline ngầm.
- Dropout có baseline 10 epoch theo sách và khảo sát 20 epoch để trả lời câu tăng thời gian học.
- Hồi quy dùng dữ liệu tổng hợp có tham số thật để kiểm tra sai số. Mục 3.5 có
  đường SGD theo cỡ mẫu và OLS tham chiếu riêng. Mục 3.7 dùng đúng bias=0.05,
  batch5, LR=0.01,10 epoch của sách; ridge chính xác chỉ là bảng tham chiếu riêng.
- Phép nhân ma trận cực lớn 2.3 dùng chi phí tính chính xác và benchmark thu nhỏ
  có ghi tỷ lệ, tránh tạo trung gian 4 GiB. Không gọi số đo thu nhỏ là thời gian đề gốc.
- Câu 5.2.7 đã có benchmark CPU và thông tin bus nhớ từ phần cứng/tài liệu hãng.
  **Chưa đo GPU** vì môi trường là PyTorch CPU; `benchmark_gpu.py` sẵn để chạy
  bằng kernel có CUDA, không bịa số liệu GPU.
- House Prices dùng 5-fold CV với tiền xử lý fit riêng từng train fold; thử
  baseline 10 epoch rồi các cải tiến có ghi cấu hình. `Le_Son_Truong_submission.csv`
  sẵn để nộp Kaggle, nhưng **chưa được upload và chưa có điểm leaderboard**.
- Các ảnh trong `anh_nop_bai` là **ảnh xuất từ lời giải/kết quả**, không giả làm
  ảnh chụp màn hình tài khoản Kaggle/Drive. Nếu giảng viên bắt buộc ảnh giao diện
  notebook, mở notebook đã có output rồi dùng Win+Shift+S để chụp.

## Các bước còn lại khi nộp

1. Đọc lại notebook/ôn tập để hiểu phần trình bày và xác nhận tên **Lê Sơn Trường**.
2. Dùng tài khoản Kaggle của bạn chấp nhận quy tắc cuộc thi House Prices,
   upload `Le_Son_Truong_submission.csv`, chụp điểm thật rồi bổ sung cho 5.7 câu 1.
3. Nếu phải có số đo GPU, chọn môi trường CUDA và chạy `benchmark_gpu.py`,
   bổ sung bảng/ảnh; kết quả CPU và GPU phải ghi đúng môi trường và precision.
4. Upload PDF/ảnh hoặc ZIP bài nộp lên Drive của bạn, đặt tên có **Lê Sơn Trường**,
   chọn quyền xem theo yêu cầu lớp và gửi link. Bài làm chưa được tự đăng lên Drive.

## Nguồn

- [D2L 1.0.3](https://d2l.ai/), PDF gốc đi kèm trong Buoi2.
- Nguồn d2l đúng phiên bản nằm trong `source_reference/d2l_torch.py`, trích từ
  wheel d2l 1.0.3 để đọc, **không cài đè** các dependency cũ của d2l.
- `data_manifest.json` ghi URL và SHA-256 của từng file dữ liệu tải thật.
- `hardware.json` ghi phần cứng và việc chưa có CUDA trong môi trường chạy.
- [Torchvision datasets](https://docs.pytorch.org/vision/stable/datasets.html),
  [LARS](https://arxiv.org/abs/1708.03888),
  [AMD CPU](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7945hx.html),
  [NVIDIA laptop GPU](https://www.nvidia.com/en-ph/geforce/laptops/compare/).
