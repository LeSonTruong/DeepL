# Kiểm tra lại bài Buổi 2 — Lê Sơn Trường

Đối chiếu với PDF `Buoi2/d2l-en.pdf`, không chỉ kiểm tra notebook có chạy không.
**Bản trước có các sai lệch/thiếu sót dưới đây; đã sửa và tính lại phần bị ảnh hưởng.**

## Những chỗ đã sửa

| Mục/câu | Vấn đề bản trước | Cách sửa |
|---|---|---|
| 3.4.1 | Chưa tách rõ variance 0.01 với std 0.01 | Thử riêng std 0.01, std 0.1 và sqrt(1000); CSV có cả std/variance |
| 3.4 | Bản forward thủ công vẫn cập nhật bằng optimizer framework | Bản manual cập nhật SGD trực tiếp bằng gradient; chạy lại thí nghiệm |
| 3.5.5 | Đường theo cỡ mẫu chỉ dùng least-squares chính xác | Bổ sung đường framework SGD LR=0.03,3 epoch,batch32; OLS giữ làm tham chiếu riêng |
| 3.7.1–2 | Hàm sinh dữ liệu dùng bias 4.2 thay vì 0.05 của mục này; đường chính là ridge chính xác | Dùng đúng w=0.01,b=0.05,n_train20/n_val100,d200; SGD batch5,LR=.01,10 epoch; ridge chỉ là tham chiếu |
| 3.7.3 | Hệ số L1 có thể bị hiểu khác quy ước λ/2 của sách | Ghi cả hai quy ước và ngưỡng proximal tương ứng, không bỏ sót hệ số 1/2 |
| 4.1.2 | Chưa khai thác mã Huffman khi ghép hai quan sát | Thêm 7 mã dài3 và 2 mã dài4, trung bình 29/18 bit/ký hiệu, tốt hơn 5/3 |
| 4.5.1 | Chỉ có mốc lý thuyết từ finfo | Thêm ngưỡng exp đo thật trên CPU và kiểm tra giá trị representable kề trước/sau |
| 5.3.2–3 | Hình thiếu cạnh W1/W2→regularizer; chưa tách bộ nhớ mô hình gốc và mô hình thêm bias | Bổ sung cạnh, công thức P₀/P, ví dụ byte FP32 và bảng bộ nhớ |
| 5.6.3 | Phương sai “eval” tầng 2 thực ra đã chịu mask của tầng 1; phép đo tiêu thụ RNG ảnh hưởng mask lần học sau | Chạy eval/train riêng, lưu/khôi phục RNG và mode, kiểm tra bằng hook độc lập; học lại toàn bộ khảo sát 5.6 |
| Đối chiếu câu hỏi | Notebook chủ yếu hiển thị lời giải, khó nhìn lại đề | Thêm khung đề tiếng Anh trích PDF tại mỗi mục, ghi trang và cảnh báo mất định dạng chỉ số khi trích text |
| Xuất ảnh | Đổi số trang có thể để sót ảnh cũ cùng số trang nhưng khác tên | Khi xuất lại, chỉ thay bộ ảnh có mẫu tên do chương trình sinh; không trộn phiên bản |

## Kết quả sau sửa

- Weight decay đúng thiết lập sách: λ=0 cho validation MSE khoảng **0.032820**;
  **λ=3** thấp nhất trong lưới SGD đã thử, khoảng **0.021553**. Đây là MSE không
  gồm hạng phạt, không phải classification accuracy và không phải nghiệm ridge chính xác.
- Dropout ở 20 epoch: baseline không dropout **87.30%**;
  p=(0.2,0.5) **87.12%**; p=(0.5,0.2) **87.01%**;
  dropout+decay **87.06%**. Kết luận chưa vượt baseline vẫn đúng, nhưng số đo cũ
  của các cấu hình có dropout đã được thay bằng kết quả chạy lại.
- House Prices không bị các lỗi trên ảnh hưởng: mean 5-fold log RMSE của cấu hình
  được chọn vẫn khoảng **0.26146**. Đó là CV local, không phải điểm Kaggle.

## Kiểm tra có thể chạy lại

`python verify_results.py` tạo `audit_checks.json` và `audit_pdf_questions.csv`:

- Đọc lại PDF gốc, đối chiếu nguyên văn trích text với inventory và số thứ tự câu
  ở 28 nhóm Exercises: **162 câu chính**. Kiểm tra này không thay thế việc đọc các ý con.
- Đối chiếu Hessian cross-entropy với covariance, tiếp tuyến tại x=1, công thức
  backward MLP có bias/L2, mã Huffman và cận Hoeffding bằng các phép tính độc lập.
- Kiểm tra phương sai eval tầng 1/2 với forward hooks, bảo đảm phép đo không đổi RNG/mode.
- Kiểm tra summary bằng epoch cuối history; variance=std²; tổng hợp CV từ 5 fold;
  schema/ID/giá dương của submission và SHA-256 của dữ liệu tải về.
- `build_submission.py --execute` chạy lại các ô notebook để hiển thị cache đã tính
  lại, rồi xuất PDF/ảnh/ZIP đồng bộ. Nó không học lại toàn bộ thí nghiệm nếu RECOMPUTE=False.

## Những chỗ dễ nhầm nhưng lời giải đã phân biệt đúng

- 2.4.6 là **x³−1/x**, tiếp tuyến tại 1 là **y=4(x−1)**.
- 2.4.8 là **norm L2**, không phải bình phương norm: gradient x/‖x‖ khi x≠0,
  không khả vi tại 0.
- 4.1.6.2 trong sách ghi bất biến của log-sum-exp là sai: đúng là
  **g(x+b·1)=g(x)+b**; softmax mới bất biến khi cộng cùng một hằng số.
- 4.4.2 cần dấu âm trong cross-entropy; 3.1.5 có mật độ Laplace hai phía;
  khoảng 1…255 ở 4.5.2 không phải khoảng signed INT8 chuẩn.
- Các giá trị finfo chỉ là mốc normal/subnormal lý thuyết, không đồng nhất với
  ngưỡng làm tròn thực tế của phép exp trên mọi phần cứng.

## Vẫn chưa hoàn thành bằng thao tác thực tế

1. **5.2.7.1:** có số đo CPU, chưa có số đo GPU vì môi trường PyTorch CPU;
   đã có `benchmark_gpu.py` để chạy bằng CUDA. Thông số bus GPU tra tài liệu
   không phải benchmark GPU.
2. **5.7.1:** đã tạo CSV dự đoán nhưng chưa upload Kaggle, chưa có điểm leaderboard.
3. **Nộp bài:** PDF/ảnh/ZIP chưa upload lên Drive; ảnh xuất không giả làm ảnh chụp tài khoản.

Vì vậy không nên ghi “hoàn thành 100% toàn bộ yêu cầu nộp bài”. Các công thức
và phần chạy cục bộ đã được rà soát/sửa, nhưng kết quả chỉ thuộc các cấu hình,
seed và ngân sách đã đo; không có kiểm tra tự động nào chứng minh mọi kết luận
thực nghiệm đúng cho mọi dữ liệu hoặc mọi lần chạy.
