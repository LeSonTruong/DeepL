# Chương 4 — Linear Neural Networks for Classification

**Người làm: Lê Sơn Trường.** Đối chiếu PDF D2L 1.0.3 trang 172–205.
Fashion-MNIST thật: 60.000 train, 10.000 test được dùng làm validation như ví dụ
của sách; vì đã dùng tập này để so sánh siêu tham số, không gọi kết quả là test độc lập.

## 4.1 — Softmax Regression (7 câu)

### Câu 1

Với one-hot y, L=logΣexp(oᵢ)−yᵀo, p=softmax(o): ∇L=p−y.
Hessian **H=diag(p)−ppᵀ**; Hᵢⱼ=pᵢ(δᵢⱼ−pⱼ).
Nếu Z là vector one-hot ngẫu nhiên theo p thì E[ZZᵀ]=diag(p), nên
Cov(Z)=diag(p)−ppᵀ=H. Với vector v, vᵀHv=Var(vᵀZ)≥0;
notebook kiểm chứng Hessian autograd và covariance trùng nhau.

### Câu 2

Ba lớp đều nhau có entropy log₂3≈1,585 bit/ký hiệu, không gán được số bit nguyên
cố định bằng entropy. Mã nhị phân cố định cần 2 bit, lãng phí một mã; mã prefix
0,10,11 có kỳ vọng 5/3 bit. Ghép hai quan sát có 9 khả năng: Huffman dùng bảy mã
dài 3 bit và hai mã dài 4 bit (000,…,110,1110,1111), trung bình 29/9 bit/cặp,
tức **29/18≈1,611 bit/ký hiệu**, tốt hơn 5/3. Nếu bắt buộc mã độ dài cố định,
cần ceil(log₂9)=4 bit/cặp, chưa tốt hơn fixed một ký hiệu; cần phân biệt hai loại mã.
Ghép n quan sát cần ceil(n log₂3) bit, trung bình → log₂3; ví dụ n=5 cần 8 bit,
tức 1,6 bit/ký hiệu. Có thể dùng arithmetic coding để khai thác xác suất.

### Câu 3

Hai trit đủ vì 3²=9≥8; một trit chỉ có 3 trạng thái. Ba mức điện áp mang
log₂3 bit mỗi symbol, có thể giảm symbol rate hoặc số đường truyền cho cùng dữ liệu.
Đổi lại phân biệt nhiều mức làm biên nhiễu nhỏ hơn; không khẳng định luôn tốt hơn nhị phân.

### Câu 4

Softmax pᵢ=eᵒⁱ/Σeᵒʲ tăng theo oᵢ khi giữ các điểm khác cố định:
∂pᵢ/∂oᵢ=pᵢ(1−pᵢ)>0 nếu có hơn một lựa chọn xác suất dương.
oᵢ>oⱼ ⇔ pᵢ>pⱼ. Thêm lựa chọn “không chọn” như lớp thứ ba với score riêng
(có thể đặt 0 làm mốc); chuẩn hóa lại trên cả ba lớp, thay vì ép hai lựa chọn có tổng 1.

### Câu 5

Đặt m=max(a,b), d=|a−b|. LSE=m+log(1+e⁻ᵈ)>m với a,b hữu hạn.
Chênh lệch trong (0,log2], bằng log2 khi a=b và tiến về 0 khi d→∞;
0 là infimum, không đạt với hai số hữu hạn.
Với λ>0: LSE(λa,λb)/λ=m+log(1+e⁻λᵈ)/λ;
sai số≤log2/λ→0. Softmin = −log(e⁻ᵃ+e⁻ᵇ), hoặc phiên bản nhiệt độ
`−logΣexp(−λxᵢ)/λ`. Với k số, sai số softmax≤log k/λ.

### Câu 6

1. Gradient log-partition là softmax, Hessian là covariance PSD ở câu 1, nên g convex.
2. **Phát biểu đề có lỗi:** g(x+b·1)=b+g(x), không phải g(x).
   Softmax(x+b·1)=softmax(x) mới là bất biến. Notebook đưa phản ví dụ b=3.
3. Logits dương lớn có thể overflow exp; cùng âm lớn có thể underflow về 0,
   dẫn tới chia 0/log 0. Chỉ trừ max trước exp không thay đổi softmax.
4. Đặt m=max x: `g(x)=m+logΣexp(xᵢ−m)`; số mũ≤0 và có ít nhất một số bằng 1.
   Khi tính cross-entropy dùng log_softmax/logsumexp, tránh tính p rồi log p quá nhỏ.

### Câu 7

Qᵢ∝Pᵢ^α tương ứng temperature T=1/α so với T=1. Gấp đôi nhiệt độ → α=1/2;
giảm một nửa → α=2. T→0 tập trung đều trên những lớp đồng cực đại (nếu chỉ
một cực đại thì one-hot). T→∞ đều trên support Pᵢ>0; phần tử Pᵢ=0 luôn bằng 0
với mọi α>0. Với logits hữu hạn thì mọi xác suất dương nên giới hạn đều trên mọi lớp.

## 4.2 — Image Classification Dataset (3 câu)

### Câu 1

Notebook đọc đủ 60.000 mẫu với batch 1/32/256/1024, đo samples/s.
Batch=1 thường có overhead Python/collate cao trên mỗi mẫu; kết luận định lượng
lấy từ bảng thực nghiệm, không chỉ suy đoán. Dữ liệu đã giải nén trong RAM, nên
phép đo này không đại diện tốc độ tải Internet hay đọc ảnh JPEG từ đĩa.

### Câu 2

Notebook dùng cProfile để tìm các hàm tốn thời gian, lưu top cumulative time.
Có thể batch fetch/collate, cache biến đổi, dùng worker hoặc pin_memory khi chuyển GPU.
Với tensor đã nằm RAM, worker/process overhead có thể lớn hơn lợi ích; cần đo
thay vì mặc định càng nhiều worker càng nhanh. Trên Windows phải đặt entry point
trong `if __name__=='__main__'` khi tạo worker ngoài notebook.

### Câu 3

API torchvision có MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, ImageFolder,
ImageNet… Theo [tài liệu torchvision](https://docs.pytorch.org/vision/stable/datasets.html).
Trong bài làm đọc trực tiếp file IDX chính thức để không phụ thuộc cặp phiên bản
torch/torchvision; dữ liệu vẫn là Fashion-MNIST gốc, không thay bằng digits hay make_moons.

## 4.3 — Base Classification Model (3 câu)

### Câu 1

Gọi N số mẫu, B số batch, m batch size chuẩn, b=N−(B−1)m là cỡ batch cuối.
Lq là trung bình không trọng số các batch mean; l_last là mean batch cuối.
Tổng loss các batch đầy là `m(B·Lq−l_last)`; tổng cuối `b·l_last`.
Do đó **Lv=[mB·Lq−(m−b)l_last]/N**. Khi b=m thì Lv=Lq.
Notebook kiểm tra với N=11, m=4, b=3.

### Câu 2

Nếu thứ tự mẫu là hoán vị ngẫu nhiên độc lập với loss, mọi batch mean đều có
kỳ vọng bằng mean toàn tập. Vậy E[Lq]=E[Lv]. Nhưng với một thứ tự cố định,
Lq có thể lệch và phương sai lớn hơn vì vài mẫu cuối có trọng số cao hơn.
Lv cân theo số mẫu cho giá trị chính xác của tập đó và không phụ thuộc cách chia batch.
Nếu dữ liệu không shuffle và batch cuối khác phân phối, không mặc định Lq unbiased.

### Câu 3

Chọn `yhat=argmin[a] Σ_y p(y|x) l(y,a)` để tối thiểu expected loss.
Với 0–1 loss là argmax xác suất. Khi chi phí sai khác nhau, quyết định có thể
khác lớp nhiều khả năng nhất; cần loss/cost matrix phù hợp bài toán.

## 4.4 — Softmax from Scratch (5 câu)

### Câu 1

Notebook chạy logits [100,101,102] và [−101,−102,−103] ở FP32. Exp trực tiếp
có thể tràn hoặc underflow, sinh NaN/đáp án sai do subnormal. Sửa bằng trừ max
mỗi hàng rồi exp/normalize; để tính loss dùng logsumexp cho ổn định hơn nữa.

### Câu 2

Cross-entropy đầy đủ là **−Σᵢ yᵢ log pᵢ** (cần dấu âm; bản in câu hỏi bỏ dấu âm).
Notebook thực hiện bản one-hot và đối chiếu với fused/index CE trên cùng logits.
Dense one-hot cần thao tác trên mọi lớp, thường tốn bộ nhớ/thao tác hơn gather nhãn
nguyên, nhưng tốc độ cụ thể phải đo; bảng có benchmark sau warm-up.
Dense CE hữu ích với soft labels/label smoothing/distillation. Cẩn thận p=0,
log 0 và biểu thức 0·(−∞) gây NaN; dùng logits và log_softmax thay vì nhân trực tiếp.

### Câu 3

Không phải lúc nào argmax cũng tối ưu: có bài toán chi phí false negative và false
positive khác nhau. Dùng expected cost, ngưỡng theo chi phí, hiệu chuẩn xác suất
hoặc cơ chế từ chối/chuyển kiểm tra. Đây là trả lời nguyên tắc phân loại, không
đưa ra ngưỡng chẩn đoán y khoa cụ thể.

### Câu 4

Vocabulary lớn làm logits/normalization và ma trận trọng số đầu ra lớn, lớp hiếm
ít dữ liệu, tăng thời gian và RAM. Có thể dùng subword, tying embeddings,
sampled/hierarchical/adaptive softmax tùy mục tiêu. Sampling khi train không luôn
cho đúng normalization tại inference; phải phân biệt xác suất chuẩn hóa và điểm xếp hạng.

### Câu 5

Notebook huấn luyện bản scratch (W,b và phép tính logits/CE rõ ràng) với
LR=0.01/0.1/1, batch mặc định 256, 10 epoch; sau đó batch=32/1024 với LR=0.1.
Vẽ validation loss theo epoch và ghi cả train loss/accuracy. Batch nhỏ làm nhiều
bước cập nhật hơn mỗi epoch; không coi so sánh cùng epoch là cùng số bước hay thời gian.
Kết luận lấy từ bảng kết quả đo, không đặt trước một LR luôn tốt nhất.

## 4.5 — Concise Softmax (4 câu)

### Câu 1

Nếu exp trả về số chuẩn hóa hữu hạn, khoảng x nằm xấp xỉ
`[log(finfo.tiny), log(finfo.max)]`. Khi tính cả subnormal, cận dưới lý thuyết
là `log(tiny*eps)`. Các mốc không phải ngưỡng làm tròn tuyệt đối chính xác;
hardware có thể flush subnormal thành 0. Notebook xuất bảng FP64/FP32/BF16/FP16,
đồng thời tìm ngưỡng bằng `torch.exp` thật trên CPU và kiểm tra số biểu diễn kề
trước/kề sau bằng `nextafter`. Bảng “measured” là ngưỡng của môi trường chạy,
không nhầm log(min_normal) với ngưỡng bắt đầu underflow về đúng 0.
TF32 giữ exponent FP32, giảm mantissa cho một số CUDA matmul, không phải dtype
CPU exp riêng. Số nguyên 8 bit không có khoảng exp dấu phẩy động tương đương;
phải quy định quantization/scale và accumulator.

### Câu 2

Khoảng 1…255 trong đề tương ứng mã unsigned 8 bit bỏ 0, không phải signed INT8
thông thường (−128…127). Mở rộng dynamic range bằng scale/zero-point,
block floating point hoặc log quantization; đánh đổi độ phân giải, không tạo thêm
thông tin. Với log coding, nhân gần thành cộng mã, nhưng cộng giá trị cần log-add;
phép cộng/nhân chuẩn trên mã không còn trực tiếp đúng với giá trị biểu diễn.

### Câu 3

Notebook tăng bản concise lên 20 epoch từ baseline 10. Validation accuracy có thể
giảm do overfitting hoặc dao động tối ưu; không khẳng định chắc chắn sẽ giảm ở
mọi lần chạy. Dùng early stopping theo validation, regularization/augmentation,
LR schedule. Không dùng nhãn test độc lập để quyết định dừng.

### Câu 4

Đối chiếu các LR curve tại 4.4: LR nhỏ có thể tốt khi học đủ lâu nhưng chậm;
LR quá lớn dễ dao động/phân kỳ. Lưới tối ưu phụ thuộc epoch, batch và mô hình.
Để tránh lặp cùng thí nghiệm, mục này dùng lại kết quả LR đã lưu ở 4.4 và có
thêm bản concise 20 epoch riêng.

## 4.6 — Generalization in Classification (4 câu)

### Câu 1

Với classifier cố định, 0–1 error trên mẫu IID, Hoeffding cho
P(|εhat−ε|≥δ)≤2exp(−2nδ²). Đặt δ=0.0001, failure=0.001:
`n≥log(2000)/(2·10⁻⁸)`, lấy trần **380.045.123** là cận đủ (điều kiện
“lớn hơn 99,9%” có thể chọn thêm một mẫu nếu muốn bất đẳng thức nghiêm).
Không phải cỡ mẫu cần thiết tối thiểu trong mọi phân phối.

### Câu 2

Với nhãn nhị phân và oracle trả **exact error**, hỏi một mô hình luôn dự đoán 0,
rồi N mô hình mỗi mô hình chỉ đổi dự đoán ở một xᵢ. Error giảm 1/N nghĩa là
yᵢ=1, tăng 1/N nghĩa là yᵢ=0. Vậy N+1 lần gọi đủ; không khẳng định là tối thiểu.
Đa lớp: baseline lớp 0 và lần lượt đổi sang từng lớp khác, tối đa 1+N(K−1)
lần theo cách đơn giản. Nếu oracle làm tròn/giới hạn truy vấn thì lập luận không còn trực tiếp áp dụng.

### Câu 3

Với classifier 1D `sign(p(x))` và p bậc không quá 5, **VC dimension=6**.
Sáu điểm có thứ tự có tối đa 5 lần đổi dấu, đặt nghiệm giữa những lần đổi dấu
để thực hiện mọi nhãn. Bảy điểm nhãn luân phiên cần 6 nghiệm nên không shatter được.
Nếu “đa thức bậc 5” là đa biến thì cần chỉ rõ số biến, kết quả khác.

### Câu 4

Rectangle song song trục trong R² có **VC dimension=4**. Bốn điểm trái/phải/
trên/dưới hình thoi có thể shatter. Với 5 điểm bất kỳ, chọn tối đa 4 điểm đạt
các cực trị x_min,x_max,y_min,y_max; còn ít nhất một điểm nằm trong bounding box
của các điểm đó. Gán các cực trị positive và điểm còn lại negative thì rectangle
không thực hiện được. Điểm trùng càng không tăng khả năng shatter.

## 4.7 — Environment and Distribution Shift (4 câu)

### Câu 1

Đổi thuật toán xếp hạng làm thay đổi thứ người dùng thấy/click; người dùng đổi
truy vấn, nhà quảng cáo đổi giá thầu/nội dung để thích ứng. Log mới do chính hệ
thống tạo ra khác log cũ, có feedback/selection bias. Cần thử nghiệm và theo dõi
chứ không giả định dữ liệu log luôn IID từ một phân phối cố định.

### Câu 2

Notebook tạo hai phân phối covariate tổng hợp và fit domain classifier source=0,
target=1; đánh giá AUC trên hold-out độc lập. AUC cao hơn 0.5 là bằng chứng phân
biệt được hai miền với classifier đó. AUC gần 0.5 không chứng minh hai phân phối
hoàn toàn giống nhau: có thể detector yếu hoặc dữ liệu ít.

### Câu 3

Nếu p_s(y|x)=p_t(y|x), support có chồng lấp và priors miền bằng nhau,
w(x)=p_t(x)/p_s(x)=D(x)/(1−D(x)). Tổng quát phải nhân thêm prior source/target.
Notebook dùng trọng số trên source hold-out để ước lượng mean ở target, so sánh
mean không trọng số, trọng số và target mean thật. Clip weights giảm variance
nhưng thêm bias; xuất effective sample size. Không dùng cách này cho mọi concept shift.

### Câu 4

Cỡ mẫu hữu hạn, phụ thuộc giữa mẫu, nhãn sai, đo đạc sai, selection bias,
model-selection/multiple testing và leakage đều ảnh hưởng empirical risk.
Ngay khi không có shift, train error vẫn có thể lạc quan do thuật toán tối ưu trực tiếp trên train.
