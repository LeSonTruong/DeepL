# Chương 3 — Linear Neural Networks for Regression

**Người làm: Lê Sơn Trường.** Đối chiếu PDF D2L 1.0.3 trang 132–164.

## 3.1 — Linear Regression (8 câu)

### Câu 1

1. J(b)=Σ(xᵢ−b)²; J'(b)=2(nb−Σxᵢ)=0 ⇒ **b=mean(x)**;
   J''=2n>0 nên cực tiểu duy nhất khi n>0.
2. Nếu xᵢ IID Gaussian cùng mean b, variance cố định, cực đại likelihood tương
   đương cực tiểu bình phương sai số; mean mẫu là MLE của b.
3. Với Σ|xᵢ−b|, nghiệm là median; n chẵn có thể là cả khoảng giữa hai giá trị giữa.

### Câu 2

Đặt x̃=[x;1], w̃=[w;b] thì x̃ᵀw̃=xᵀw+b. Affine trên x là tuyến tính trên
không gian đặc trưng mở rộng; không có nghĩa nó tuyến tính thuần túy trên x.

### Câu 3

Tạo tầng đặc trưng `[1,x₁,…,x_d,x₁²,x₁x₂,…,x_d²]`, rồi tầng tuyến tính cho
đầu ra. Đây là mô hình tuyến tính theo tham số nhưng phi tuyến theo x.
Chỉ ghép các Linear không có tương tác/activation thì không tạo được xᵢxⱼ chính xác.

### Câu 4

1. XᵀX thiếu hạng: nghịch đảo không tồn tại; nghiệm bình phương tối thiểu vẫn có
   nhưng hệ số có thể không duy nhất.
2. Dùng pseudoinverse/SVD, loại cột phụ thuộc hoặc ridge. Nhiễu Gaussian độc lập
   có thể phá phụ thuộc chính xác khi n≥d; nếu n<d thì hạng vẫn≤n nên không thể đầy hạng cột.
3. Với X cố định và Eᵢⱼ IID mean 0, variance σ²:
   E[(X+E)ᵀ(X+E)]=XᵀX+nσ²I. Đó là kỳ vọng, không bảo đảm một mẫu có điều kiện tốt.
4. SGD có thể hội tụ tới một nghiệm tùy khởi tạo, LR và thành phần null space;
   dự đoán trên train có thể duy nhất dù hệ số không duy nhất. Với zero init và điều
   kiện thích hợp, gradient descent tuyến tính có thể chọn nghiệm norm nhỏ nhất.

### Câu 5

Mật độ (1/2)exp(−|ε|) trong đề thực ra là **Laplace hai phía**, không phải
exponential một phía. NLL = n log 2 + Σ|yᵢ−xᵢᵀw−b|. Intercept-only có
nghiệm median; hồi quy tổng quát không có công thức nghịch đảo đơn giản, có thể
giải bằng linear programming hoặc subgradient SGD. Một minibatch dùng gradient
`sign(yhat−y)*x`, trung bình theo batch. Gần nghiệm, gradient có thể đổi dấu và
dao động; giảm LR, averaging hoặc Huber loss giúp ổn định.

### Câu 6

Hai tầng affine ghép nhau vẫn affine:
`W₂(W₁x+b₁)+b₂=(W₂W₁)x+(W₂b₁+b₂)`. Không tăng khả năng biểu diễn phi tuyến;
nếu tầng giữa hẹp còn hạn chế hạng của phép biến đổi.

### Câu 7

1. Gaussian cộng có thể cho giá âm và variance tuyệt đối cố định, không phù hợp
   khi sai số thường mang tính tỷ lệ.
2. Học log(price) với giá dương giúp mô hình hóa biến động tương đối; exp đầu ra
   cho giá dương. Exp của dự đoán mean log không mặc nhiên là mean giá gốc
   (cần xét bias khi back-transform).
3. Giá giao dịch có bước giá rời rạc; cùng một tick là tỷ lệ thay đổi lớn hơn
   ở giá rất nhỏ. Đây là phân tích mô hình toán học theo đề, không phải tư vấn đầu tư.

### Câu 8

1. Số táo là số nguyên không âm; Gaussian liên tục có thể cho số âm/phân số.
2. Với K~Poisson(λ): E[K]=Σk·λᵏe⁻λ/k! = λe⁻λΣ[j≥0]λʲ/j! = λ.
3. NLL mỗi mẫu là `λ−k log λ+log(k!)`, λ>0; phần factorial không phụ thuộc tham số.
4. Đặt η=log λ thì loss=`exp η−kη+log(k!)`, gradient theo η=`exp η−k`.

## 3.2 — Object-Oriented Design (2 câu)

### Câu 1

Mã đầy đủ của HyperParameters, Module, DataModule, Trainer có ở
https://d2l.ai/_modules/d2l/torch.html và nguồn `d2l/torch.py` của D2L 1.0.3.
Bài làm lưu bản nguồn tham khảo tại `source_reference/d2l_torch.py`.
HyperParameters giữ cấu hình; Module mô tả mô hình/loss/optimizer;
DataModule cung cấp loader; Trainer điều phối forward/backward/update/validation.
Các phương thức được thêm dần bằng `add_to_class`, nên bản trong một đoạn sách
có thể chưa đầy đủ như thư viện.

### Câu 2

Nếu `B.__init__` chỉ nhận a,b mà không gán `self.a`, `self.b`, bỏ
`save_hyperparameters()` thì không còn hai thuộc tính đó, truy cập sẽ AttributeError.
Hàm tiện ích đọc biến cục bộ của frame gọi rồi gán lên self; tham số hàm không tự
trở thành thuộc tính đối tượng. Nếu superclass hoặc code khác đã gán chúng thì cần
xét riêng; ví dụ tối thiểu trong notebook không có các phép gán ấy.

## 3.3 — Synthetic Regression Data (4 câu)

### Câu 1

Nếu N không chia hết B, DataLoader mặc định giữ minibatch cuối nhỏ hơn.
Ví dụ N=101,B=32 → 32,32,32,5; `drop_last=True` → ba batch 32, bỏ 5 mẫu.
Khi tính loss trung bình cả epoch, phải cân theo số mẫu chứ không trung bình đều batch.

### Câu 2

Không đủ RAM: dùng streaming/IterableDataset, memory map hoặc shard trên đĩa.
Shuffle hiệu quả: shuffle thứ tự shard/block, đọc tuần tự từng block và shuffle
trong buffer RAM; hoặc dùng phép hoán vị giả ngẫu nhiên trên chỉ số và gom các
truy cập theo block để giảm seek. Block shuffle chỉ gần tương đương shuffle toàn bộ;
nếu cần hoán vị toàn cục chính xác thì dùng external shuffle nhiều lượt hoặc
khóa ngẫu nhiên + external sort, tốn I/O nhưng không cần giữ toàn bộ bảng hoán vị.

### Câu 3

Notebook định nghĩa IterableDataset Fresh: mỗi lần `__iter__` sinh x mới bằng
`torch.randn`, rồi y=xᵀw+b. Nhãn và đặc trưng được sinh theo từng mẫu, không lưu toàn bộ.

### Câu 4

Trong `__iter__`, tạo generator cục bộ cùng seed cố định rồi truyền vào mọi lần
sinh số. Hai iterator sẽ tạo lại cùng dữ liệu. Khi chạy nhiều worker cần thiết kế
seed/sharding để không vô tình lặp mẫu giữa các worker.

## 3.4 — Linear Regression from Scratch (9 câu)

### Câu 1

Hồi quy tuyến tính không có các neuron ẩn đồng vai trò nên zero init vẫn học được.
Variance khởi tạo 1000 nghĩa là std=√1000, không phải std=1000: dự đoán/loss
đầu lớn hơn, có thể mất nhiều bước hoặc mất ổn định với LR lớn. Notebook so sánh
zero, std=0.01 (variance=0.0001, cách sinh trong code sách), std=0.1
(variance=0.01 đúng theo chữ “variance” của câu hỏi), và std=√1000 trên cùng dữ liệu.
Không được coi variance và standard deviation là một đại lượng.

### Câu 2

Định luật Ohm V=RI. Tạo tham số R requires_grad; tối thiểu mean(Vhat−V)² bằng
autograd và SGD. Notebook khôi phục R=7 từ dữ liệu giả lập; kết quả đo được in ra,
không phải thí nghiệm điện vật lý.

### Câu 3

Giữ λ và h,c,k đúng đơn vị SI; tham số hóa T=exp θ để T>0. Fit phổ giả lập
Planck ở nhiều bước sóng bằng loss trên log B; tính expm1 để ổn định khi số mũ
nhỏ. Notebook học T từ khởi tạo 3000 K đến gần nhiệt độ thật 4500 K.
Với dữ liệu đo thực, cần thêm nhiễu, hiệu chuẩn và có thể hệ số phát xạ;
không khẳng định một phép fit đơn giản xác định mọi vật thể.

### Câu 4

Backward mặc định không giữ graph của gradient. Dùng `autograd.grad(...,
create_graph=True)` để đạo hàm tiếp. Hessian n×n tốn bộ nhớ và thời gian; HVP,
autograd.functional.hessian, hoặc xấp xỉ bậc hai phù hợp hơn tùy mục tiêu.
ReLU tại điểm gãy không có đạo hàm cổ điển; không coi mọi Hessian là tồn tại trơn.

### Câu 5

yhat shape `(B,1)`, y shape `(B,)` nếu trừ trực tiếp sẽ broadcast thành `(B,B)`.
`y.reshape(yhat.shape)` bảo đảm mỗi dự đoán so đúng một nhãn.

### Câu 6

Notebook vẽ loss với LR 0.003, 0.03, 0.3 trong 10 epoch. LR nhỏ học chậm;
LR lớn có thể nhanh nhưng vượt ngưỡng ổn định thì dao động/phân kỳ.
Thêm epoch có thể giảm lỗi tối ưu, không bảo đảm giảm lỗi validation mãi.
Các giá trị này là khảo sát theo yêu cầu, không đổi baseline 3 epoch, LR=0.03 của sách.

### Câu 7

Iterator giữ batch cuối nhỏ hơn nếu không drop_last. Nếu loss dùng tổng và chia
cho batch_size cố định, batch cuối bị scale sai; dùng trung bình theo số mẫu thực.

### Câu 8

Notebook chạy squared/L1/Huber trên cùng dữ liệu sạch và dữ liệu có y[5]=10000,
so sánh sai số trọng số và bias, không so trực tiếp trị số giữa các loss khác đơn vị.
Squared loss có gradient theo residual nên outlier chi phối; L1 có gradient biên độ
không đổi nhưng khó tối ưu gần 0. Huber bậc hai gần 0, tuyến tính khi residual lớn,
giúp kết hợp tính trơn gần nghiệm và độ bền trước outlier.

### Câu 9

Shuffle làm minibatch đại diện hơn, giảm thiên lệch do thứ tự. Ví dụ x=1,
nhãn gồm một block +1 rồi một block −1; SGD LR lớn có xu hướng chạy theo block
cuối ở mỗi epoch. Không phải mọi dữ liệu không shuffle đều thất bại, nhưng thứ tự
đối nghịch có thể gây dao động hoặc hội tụ chậm; shuffle không thay thế việc chọn LR hợp lý.

## 3.5 — Concise Linear Regression (5 câu)

### Câu 1

Nếu cập nhật cũ dùng **gradient tổng trực tiếp** với LR η, đổi loss thành mean
thì gradient giảm B lần; muốn cùng bước phải đặt η_mean=Bη_sum.
Nếu code cũ đã chia gradient tổng cho B trong optimizer thì mean loss dùng cùng η,
không chia thêm lần nữa. Phải kiểm tra reduction và bước update cùng nhau.

### Câu 2

Công thức của sách là scaled Huber: `e²/(2σ)` nếu |e|≤σ, và `|e|−σ/2` nếu ngoài.
`torch.nn.HuberLoss(delta=σ)` cho `e²/2` và `σ(|e|−σ/2)`, nên chia cho σ
để khớp sách; `SmoothL1Loss(beta=σ)` cũng khớp. Với σ=1 chúng trùng nhau.
Notebook dùng delta=1 và so sánh trong mục 3.4, đồng thời chạy bản framework ở mục này.

### Câu 3

Sau backward, đọc `model.weight.grad` hoặc `model.net[...].weight.grad`.
Gradient chưa có trước backward, bị reset khi zero_grad, và cộng dồn nếu không reset.

### Câu 4

LR/epoch quyết định mức hội tụ và ổn định. Thêm epoch không cải thiện vô hạn:
có đáy nhiễu, độ chính xác số, và có thể overfitting. Xem đường học của 3.4;
không dùng training loss làm bằng chứng duy nhất về tổng quát hóa.

### Câu 5

Notebook vẽ ‖ŵ−w‖₂ và |b̂−b| theo n=5,10,20,50,…,10000, lấy trung bình
10 seed dữ liệu. Đường chính chạy mô hình framework với **SGD LR=0.03, 3 epoch,
batch32** như ví dụ; có thêm least-squares chính xác làm tham chiếu để tách
tác động thống kê khỏi lỗi chưa hội tụ. Cùng số epoch nhưng n lớn hơn cũng có
nhiều bước SGD hơn. Với mô hình đúng và đủ điều kiện, lỗi thống kê thường giảm cỡ 1/√n,
nhưng từng mẫu có thể không giảm đều. Dải n theo log bao phủ nhiều bậc độ lớn,
giúp quan sát hiệu quả tăng dữ liệu thay vì chỉ một vùng hẹp.

## 3.6 — Generalization (7 câu)

### Câu 1

Hồi quy đa thức bậc d giải bình phương tối thiểu bằng ma trận Vandermonde.
Nếu n=d+1 điểm x phân biệt thì có đa thức nội suy duy nhất; n≤d+1 có thể
nội suy nhưng chưa chắc duy nhất. Khi n>d+1 chỉ khớp chính xác nếu nhãn nằm trong
không gian cột; thiếu hạng dùng pseudoinverse. Khớp train chính xác không bảo đảm tổng quát hóa.

### Câu 2

Năm ví dụ không nên chia ngẫu nhiên như IID: chuỗi thời gian liền kề; ảnh nhiều
khung của cùng video; nhiều lần đo một bệnh nhân; văn bản của cùng tác giả/tài liệu;
đo đạc địa lý gần nhau. Dùng time split, group split hoặc spatial split tương ứng.

### Câu 3

Có thể zero train error khi mô hình nội suy/ghi nhớ, hoặc dữ liệu không nhiễu và
mô hình đúng. Zero generalization error đòi hỏi nhiệm vụ xác định và hàm học đúng
trên phân phối đánh giá; nhiễu không loại bỏ tạo ra Bayes error dương. Với classification,
zero 0–1 error khác với cross-entropy đúng bằng 0.

### Câu 4

K-fold huấn luyện K mô hình cho mỗi cấu hình; H cấu hình cần khoảng K·H lần fit.
Tiền xử lý cũng phải fit riêng từng fold. Đó là chi phí lớn khi mạng sâu học lâu.

### Câu 5

CV đánh giá mô hình học trên (K−1)/K dữ liệu, không phải mô hình cuối học toàn bộ;
cỡ train nhỏ hơn thường gây thiên lệch bi quan, nhưng dấu bias không tuyệt đối trong
mọi thuật toán. Thử nhiều cấu hình và chọn min CV còn có selection bias lạc quan.

### Câu 6

VC dimension đếm khả năng shatter nhãn, bỏ qua norm, margin, độ trơn và thuật toán
chọn nghiệm. Hai lớp cùng VC dimension có thể có mức tổng quát hóa thực tế khác;
với hồi quy thực liên tục còn cần các khái niệm như pseudo-dimension/Rademacher complexity.

### Câu 7

Giảm tập train thành các kích thước khác nhau, giữ validation cố định và vẽ learning
curve. Nếu val error còn giảm theo n trong khi train/val có khoảng cách, đó là bằng
chứng thêm dữ liệu có thể giúp. Nếu cả hai lỗi cao và phẳng, cần cải thiện mô hình/
đặc trưng/tối ưu, chưa thể chỉ yêu cầu thêm dữ liệu. Notebook thực hiện đường học này.

## 3.7 — Weight Decay (6 câu)

### Câu 1

Đề viết “accuracy” trong bài hồi quy; ở đây báo cáo train/validation **MSE** cho
đúng đại lượng. Dữ liệu đúng mục 3.7: y=0.05+Σ0.01xᵢ+ε, std(ε)=0.01,
20 train/100 validation, d=200. Đường chính dùng SGD, **batch5, LR=0.01,
10 epoch**, khảo sát λ=0/0.01/0.1/1/3/10; không phạt bias.
Ridge quá yếu có thể overfit, quá mạnh có thể underfit. Bảng ridge giải chính xác
ở lưới 0…100 chỉ là tham chiếu riêng, không thay cho kết quả SGD của mục này.

### Câu 2

Chọn λ có validation MSE thấp nhất trong lưới đã thử. Đó không bảo đảm tối ưu
trên mọi λ hoặc dữ liệu mới vì validation hữu hạn và đã dùng chọn mô hình.
Giá trị đủ tốt/ổn định thường quan trọng hơn λ chính xác tuyệt đối; cần test độc lập
hoặc nested CV nếu cần ước lượng tổng quát hóa sau chọn tham số.

### Câu 3

Nếu giữ hệ số λ/2 như biểu thức sách và thay bình phương norm bằng Σ|wᵢ|,
J=L+(λ/2)Σ|wᵢ|: cập nhật `w←w−η(∇L+(λ/2)sign(w))`, chọn
subgradient trong [−1,1] tại 0. Proximal gradient cho nghiệm thưa rõ hơn:
`w←sign(z) max(|z|−ηλ/2,0)`, z=w−η∇L. Nếu định nghĩa lại J=L+λΣ|wᵢ|,
hệ số trong gradient và ngưỡng là λ thay vì λ/2. Bias có thể không phạt.

### Câu 4

‖W‖²_F=ΣᵢⱼWᵢⱼ²=tr(WᵀW)=tr(WWᵀ), tương tự wᵀw cho vector.

### Câu 5

Có thể thêm dữ liệu phù hợp, augmentation bảo toàn nhãn, early stopping, dropout,
giảm leakage và chọn đặc trưng. Tăng thời gian train tự nó không phải thuốc chống
overfit; phải theo dõi validation. Những biện pháp có thể không cộng hiệu quả tuyến tính.

### Câu 6

MAP cực đại log likelihood + log prior, tương đương NLL − log prior.
Prior Gaussian mean 0 tạo L2; Laplace tạo L1. Hệ số λ phụ thuộc variance prior,
quy ước tổng/trung bình loss và cỡ mẫu; không đồng nhất mọi optimizer weight decay
với cùng một MAP (AdamW tách decay là trường hợp cần phân biệt).
