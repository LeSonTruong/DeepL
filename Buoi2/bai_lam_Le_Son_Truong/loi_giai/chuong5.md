# Chương 5 — Multilayer Perceptrons

**Người làm: Lê Sơn Trường.** Đối chiếu PDF D2L 1.0.3 trang 215–246.
Các nhận xét định lượng nằm ngay sau từng mục trong notebook, lấy từ thí nghiệm
thật. Fashion-MNIST dùng đủ 60.000 train/10.000 validation như sách.

## 5.1 — Multilayer Perceptrons (7 câu)

### Câu 1

Ghép các affine không activation cho W_L…W₁x+b_eff, vẫn là affine. Ví dụ
đầu vào/ra 2D, tầng giữa chỉ 1 chiều thì hạng tích≤1, không biểu diễn được
ánh xạ identity hạng 2. Nhiều tầng tuyến tính không tăng lớp hàm, có thể giảm nó
khi có bottleneck, dù tham số hóa ảnh hưởng cách tối ưu.

### Câu 2

pReLU(x)=x nếu x≥0, αx nếu x<0. Đạo hàm theo x là 1 ở x>0, α ở x<0;
tại 0 không khả vi khi α≠1 (implementation chọn một giá trị quy ước).
Nếu α học được, ∂pReLU/∂α=x khi x<0 và 0 khi x>0.

### Câu 3

Swish=x·sigmoid(βx), đặt s=sigmoid(βx):
`d/dx=s+βx s(1−s)`. Nếu β là tham số học: `∂/∂β=x²s(1−s)`.

### Câu 4

Affine liên tục và tuyến tính từng miền; ReLU/pReLU liên tục và affine từng miền.
Ghép hữu hạn các hàm này cho hàm liên tục piecewise-affine trên các miền đa diện
(sách gọi piecewise-linear; có bias thì chính xác hơn là affine). Số miền có thể
tăng khi thêm tầng, nhưng không tạo gián đoạn tại ranh giới activation.

### Câu 5

Từ định nghĩa: tanh x=(e²ˣ−1)/(e²ˣ+1)=2sigmoid(2x)−1.
Do đó tanh(Wx+b)=2sigmoid(2Wx+2b)−1. Hệ số 2 và −1 được hấp thụ vào
trọng số/bias của tầng affine tiếp theo; chiều ngược dùng sigmoid z=(tanh(z/2)+1)/2.
Với activation ở tầng ẩn và đầu ra affine, hai lớp hàm biểu diễn tương đương
sau tái tham số hóa; điều này không bảo đảm dynamics tối ưu giống nhau.

### Câu 6

Nonlinearity phụ thuộc batch (như batch normalization khi train) làm đầu ra của
một mẫu phụ thuộc các mẫu đi cùng, kích thước batch và thống kê batch.
Batch quá nhỏ hoặc train/eval khác phân phối có thể gây nhiễu/lệch thống kê.
Phải xử lý running statistics và chế độ eval đúng; trong dữ liệu chuỗi cần tránh
thông tin tương lai lọt vào thống kê dùng cho dự đoán quá khứ.

### Câu 7

σ'(x)=σ(x)(1−σ(x))→0 khi |x| lớn, ví dụ x=±30. Nhiều tầng sigmoid nhân
các gradient nhỏ có thể làm gradient về đầu mạng rất nhỏ. Notebook đo gradient
tại −30,−10,0,10,30; FP32 ở phía bão hòa dương còn có thể làm tròn về đúng 0.

## 5.2 — Implementation of MLP (9 câu)

### Câu 1

Baseline sách: 784→256→10, ReLU, SGD LR=0.1, batch=256, 10 epoch.
Notebook khảo sát 1,64,128,256,512 neuron trên cùng seed, vẽ accuracy theo độ rộng.
“Tốt nhất” chỉ là tốt nhất trong lưới này với ngân sách 10 epoch, không phải
siêu tham số tối ưu toàn cục. Kết quả cuối mục được tính tự động từ CSV.

### Câu 2

So sánh baseline một tầng 256 với hai tầng 256→256, giữ LR, batch, epoch.
Thêm tầng tăng khả năng kết hợp phi tuyến và chi phí, nhưng không bảo đảm tốt hơn;
bảng/learning curve trong notebook cho biết kết quả thực tế.

### Câu 3

Một neuron ẩn là bottleneck một chiều: nhiều đầu vào bị nén về một số.
Sau ReLU có thể chết hẳn nếu preactivation âm cho mọi mẫu. Nhiệm vụ có nhiều
lớp/đặc trưng có thể mất thông tin quan trọng, nên đây là lựa chọn quá hạn chế.
Không nói một neuron không thể giải mọi bài toán: một bài rất đơn giản vẫn có thể đủ.

### Câu 4

Giữ width=256, so LR=0.01/0.1/0.5 với 10 epoch. LR nhỏ cần nhiều epoch hơn;
LR lớn có thể hội tụ nhanh hoặc dao động. Chọn theo validation loss/accuracy và
đại lượng mục tiêu; không suy ra LR=0.5 tốt cho mọi kiến trúc chỉ từ một lần đo.

### Câu 5

Thử thêm cấu hình kết hợp: hai tầng 128 với LR=0.1, 15 epoch; hai tầng 256 với
LR=0.05, 15 epoch, rồi so với các cấu hình trước. Báo cáo cấu hình có val loss
thấp nhất và accuracy cao nhất trong tập đã đo, không tuyên bố tối ưu toàn bộ không gian.
Với k lựa chọn ở mỗi d tham số, grid đầy đủ có k^d lần fit; các yếu tố còn tương tác.
Chiến lược hiệu quả: random search trên log LR/weight decay, ngân sách ngắn cho
nhiều cấu hình rồi successive halving/ASHA, tăng ngân sách cho ứng viên tốt;
giữ test riêng nếu cần đánh giá cuối sau tuning.

### Câu 6

Notebook có bản ScratchMLP với W1,b1,W2,b2 và matmul/ReLU rõ ràng, cùng bản
nn.Sequential. Để đo công bằng, copy **cùng trọng số**, cùng batch và loss,
warm-up rồi đo forward+backward 20 lần ở width 64/256/512. Bảng timing này
tách khỏi tổng thời gian học, vì hai bản học gốc có cách khởi tạo khác nhau.
Tăng độ phức tạp có thể làm GEMM/kernel chi phối, giảm tỷ lệ overhead Python;
không khẳng định luôn framework nhanh hơn mà không xem số đo.

### Câu 7

Notebook đo matmul CPU ở 1024/1025/1026/1028/1032, lấy median sau warm-up.
Kích thước chia hết tile/SIMD có thể nhanh hơn nhưng phụ thuộc BLAS/kernel,
cache và hệ thống; không so chỉ thời gian mà quên số phép toán thay đổi theo n³.

Máy có Ryzen 9 7945HX và RTX 4060 Laptop; môi trường hiện tại cài PyTorch CPU,
`torch.cuda.is_available()` là False nên **chưa có số đo GPU**. Không giả lập
GPU timing thành kết quả thật; `benchmark_gpu.py` chạy phép đo cùng kích thước
khi dùng kernel PyTorch hỗ trợ CUDA, có synchronize trước/sau timer.

RAM báo cáo mỗi module DataWidth=64 bit; CPU hỗ trợ hai kênh theo
[AMD](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7945hx.html),
nên bề rộng tổng lý thuyết 2×64=128 bit nếu hai kênh hoạt động. DDR5 có thể
chia mỗi DIMM thành các subchannel; không đồng nhất AddressWidth=64 với memory bus.
RTX 4060 Laptop có giao tiếp VRAM 128 bit theo
[NVIDIA](https://www.nvidia.com/en-ph/geforce/laptops/compare/).
Không dùng WMI AdapterRAM để suy ra dung lượng VRAM vì trường đó có giới hạn.

### Câu 8

Khảo sát ReLU/tanh/sigmoid trên mạng một tầng 256 neuron với cùng LR=0.1,
batch 256, 10 epoch. ReLU tránh bão hòa phía dương nhưng có dead neurons;
sigmoid/tanh có thể bão hòa. Lấy kết luận theo bảng cụ thể; khác LR hoặc
khởi tạo có thể thay thứ tự tốt/xấu.

### Câu 9

So zero, normal std=0.01, Xavier và default của Linear, cùng dữ liệu/seed.
Zero init ở MLP gây đối xứng; với ReLU tại 0, gradient tầng đầu có thể bằng 0,
chỉ bias đầu ra học prior lớp. Xavier/Kaiming giúp kiểm soát phương sai nhưng
cần khớp activation. Trong linear/softmax không tầng ẩn, zero init không gây
cùng dạng đối xứng neuron ẩn; xem mục 5.4.

## 5.3 — Forward/Backward/Computational Graphs (5 câu)

### Câu 1

Loss scalar f và X shape n×m ⇒ gradient ∂f/∂X cũng shape n×m,
mỗi phần tử là độ nhạy của f theo phần tử tương ứng của X.

### Câu 2

Theo quy ước batch hàng: Z=XW1+1b1ᵀ; H=φ(Z); O=HW2;
J=L(O,Y)+(λ/2)(‖W1‖²_F+‖W2‖²_F), không phạt b1.
Đồ thị ở notebook: X,W1,b1→Z→H; H,W2→O; O,Y→L; W1,W2→R; L,R→J.

Đặt G=∂L/∂O; với softmax-CE mean, G=(softmax(O)−onehot(Y))/N.
Backward:

- ∂J/∂W2=HᵀG+λW2.
- D=(GW2ᵀ)⊙φ'(Z).
- ∂J/∂W1=XᵀD+λW1; ∂J/∂b1=Σ_hàng D; ∂J/∂X=DW1ᵀ.

Với vector cột, transpose các ma trận theo quy ước của sách; không trộn hai quy ước.

### Câu 3

Input d, hidden h, output q, batch N: mô hình gốc không bias có P₀=dh+hq;
phiên bản thêm hidden bias ở câu 2 có P=P₀+h tham số.
FP32 tham số chiếm 4P byte, gradient thêm 4P; SGD thuần không có momentum state,
Adam thêm khoảng 8P cho hai moment. Các activation X,Z,H,O có khoảng
N(d+2h+q) phần tử lưu cho backward, thêm bộ nhớ loss/workspace/mask tùy framework.
Ví dụ FP32, d=784,h=256,q=10,N=32: mô hình gốc có 203.264 tham số
(813.056 byte), mô hình thêm bias có 203.520 tham số (814.080 byte).
Prediction không giữ graph/gradient, cần tham số và activation đang dùng; peak
phụ thuộc tái sử dụng buffer, không thể cho một số byte duy nhất nếu chưa chỉ rõ
batch,dtype,optimizer và implementation.

### Câu 4

Phải giữ graph của gradient (`create_graph=True`); graph mới gồm cả phép tính
backward. Hessian đầy đủ có P² phần tử và thường cần nhiều lượt, tốn hơn bậc một.
HVP chỉ cần directional derivative của gradient, tránh materialize Hessian.
Notebook kiểm tra f=Σx³ có H·1=6x.

### Câu 5

Có thể partition theo tầng/pipeline/model parallel qua nhiều GPU, truyền
activation và gradient giữa thiết bị. Ưu điểm: mô hình lớn hơn VRAM một GPU;
nhược: communication, pipeline bubble, đồng bộ và triển khai phức tạp.
Giảm minibatch đơn giản hơn và giảm activation memory, nhưng không giảm kích
thước tham số/optimizer state; model parallel/sharding cần khi phần này đã quá lớn.

## 5.4 — Numerical Stability and Initialization (4 câu)

### Câu 1

Các filter convolution giống nhau trong cùng nhóm hoặc attention heads có toàn bộ
tham số/đường đi giống nhau có thể nhận gradient giống nhau, học trùng chức năng.
Các nhánh residual song song khởi tạo giống nhau cũng có đối xứng tương tự.
Phá bằng khởi tạo ngẫu nhiên độc lập hoặc thiết kế bất đối xứng có chủ đích.

### Câu 2

Linear regression có thể khởi tạo cùng giá trị, kể cả 0. Softmax cũng có thể zero:
nhãn từng lớp tạo gradient khác nhau `Xᵀ(p−y)` nên các cột không nhất thiết giữ
giống nhau. Vấn đề đối xứng nặng là nhiều neuron ẩn đồng vai trò cùng tham số,
không phải mọi mô hình có nhiều tham số đều cấm zero init.

### Câu 3

Không có quy tắc nhân riêng eigenvalue cho hai ma trận bất kỳ. Ví dụ hai ma trận
tam giác A=[[1,M],[0,1]], B=Aᵀ đều có eigenvalue 1, nhưng AB có eigenvalue
lớn cỡ M². Notebook kiểm tra M=100.

Với spectral norm/singular value: σmax(AB)≤σmax(A)σmax(B);
với ma trận vuông khả nghịch σmin(AB)≥σmin(A)σmin(B), nên κ₂(AB)≤κ₂(A)κ₂(B).
Gradient qua nhiều tầng chứa tích Jacobian: σ lớn gây explode, nhỏ gây vanish.
Khởi tạo tốt cần kiểm soát singular values/phương sai và activation, không chỉ eigenvalue.

### Câu 4

LARS dùng local rate gần `η_l=η·τ‖w_l‖/(‖g_l‖+β‖w_l‖+ε)` để scale bước
cập nhật tương đối theo norm tầng; xem
[You, Gitman, Ginsburg 2017](https://arxiv.org/abs/1708.03888).
Có thể dùng giảm LR/warm-up, clipping, normalization và khởi tạo phù hợp để phòng
divergence. Nếu tham số đã thành Inf/NaN, scale sau đó không khôi phục thông tin;
cần phát hiện trước update hoặc quay lại checkpoint hợp lệ. LARS không bảo đảm
sửa mọi dạng bất ổn và không phải một phép “chữa” NaN sau khi mất dữ liệu.

## 5.5 — Generalization in Deep Learning (5 câu)

### Câu 1

Mạng quá tham số có thể fit nhãn ngẫu nhiên nhưng vẫn tổng quát hóa trên dữ liệu
có cấu trúc. Capacity worst-case theo số tham số/VC thường cho cận quá lỏng,
bỏ qua dữ liệu, margin/norm, nghiệm và implicit bias của thuật toán tối ưu.
Không có nghĩa complexity vô nghĩa, chỉ các thước đo thô chưa giải thích đầy đủ.

### Câu 2

Early stopping hạn chế thời gian tối ưu/độ phức tạp hữu hiệu của nghiệm, ngăn
mô hình tiếp tục fit nhiễu train. Với một số bài toán tuyến tính nó có liên hệ
spectral regularization, nhưng không đồng nhất mọi early stopping với cùng một L2.

### Câu 3

Theo dõi validation metric định trước, lưu best checkpoint và dừng sau patience
khi không cải thiện đủ min_delta. Chọn metric/độ kiên nhẫn trước khi nhìn test.
Notebook chọn epoch min validation loss từ một đường học đã đo, không giả định
epoch đó luôn có accuracy cao nhất.

### Câu 4

Lợi ích lớn hơn khi có khoảng cách train/validation, nhãn nhiễu hoặc mô hình
có khả năng ghi nhớ và validation bắt đầu xấu đi trong lúc train còn tốt lên.
Nếu cả hai đang cải thiện hoặc đã plateau giống nhau, early stopping có thể
không cải thiện accuracy; phải xem dữ liệu và đường học cụ thể.

### Câu 5

Tiết kiệm thời gian, điện và chi phí tính toán; dùng ngân sách cho nhiều cấu hình
hơn. Lợi ích tài nguyên vẫn tồn tại ngay khi generalization không tăng.

## 5.6 — Dropout (7 câu)

### Câu 1

Giữ kiến trúc hai tầng 256→256, LR=0.1, batch256; so p=(0.2,0.5),
(0.5,0.2), baseline không dropout và cấu hình sách (0.5,0.5).
Bảng ghi accuracy/loss thật; các cặp hoán đổi chạy cùng 20 epoch.
Drop sớm hơn có thể làm mất tín hiệu đầu vào tầng sau nhiều hơn; không suy ra
hoán đổi p luôn cho kết quả như nhau dù tổng p bằng nhau.

### Câu 2

Hai cấu hình có dropout và baseline không dropout chạy 20 epoch, xuất kết quả
ở epoch10 và epoch20 trên cùng run. Tăng epoch có thể giúp mạng dropout học
bù nhiễu; cũng có thể baseline overfit. Kết luận theo số đo, không hứa dropout luôn thắng.

### Câu 3

Inverted dropout z=m·h/(1−p), m~Bernoulli(1−p). Với h cố định,
E[z|h]=h và Var(z|h)=p·h²/(1−p); không dropout thì mask variance=0.
Nếu H ngẫu nhiên độc lập mask:
Var(Z)=Var(H)+p·E[H²]/(1−p).
Notebook đo empirical variance qua 1.024 ảnh cố định và toàn bộ units ở mỗi
tầng, theo epoch. **Eval và train được chạy riêng**: đường eval tắt toàn bộ dropout;
đường train bật dropout đúng như lúc học. Lưu/khôi phục RNG và mode quanh phép
đo để không làm thay đổi mask của epoch tiếp theo. Đó không phải cùng đại lượng
với variance theo mask khi giữ một activation h cố định; đồ thị ghi rõ cách đo.

### Câu 4

Inference tiêu chuẩn tắt dropout vì inverted dropout đã giữ kỳ vọng khi train;
để mask ngẫu nhiên làm dự đoán biến động không cần thiết. Ngoại lệ MC dropout
cố ý giữ mask và lấy nhiều lần dự đoán để khảo sát bất định; phải thiết kế/đánh giá riêng.

### Câu 5

So baseline, dropout, weight decay=1e−4 và cả hai trên cùng kiến trúc/20 epoch.
Hai cơ chế khác nhau: dropout tạo nhiễu activation, weight decay hạn chế trọng số.
Hiệu quả không nhất thiết cộng; kết hợp có thể lợi ích giảm dần hoặc regularize
quá mạnh. Bảng định lượng trong notebook là căn cứ kết luận cho dữ liệu này.

### Câu 6

Drop trực tiếp trọng số tạo `Wtilde=M⊙W/(1−p)` (DropConnect).
Các output units có mẫu nhiễu liên kết khác activation dropout; phải tính mask
trên ma trận, tắt khi eval, kỳ vọng affine vẫn giữ nhưng kỳ vọng sau nonlinearity
không nhất thiết bằng output mạng mean. Không coi hai kỹ thuật tương đương.

### Câu 7

Thử thêm Gaussian noise mean0,std=0.15 vào mỗi tầng ẩn khi train, tắt lúc eval;
giữ cùng kiến trúc/20 epoch với dropout. Notebook so sánh thật trên Fashion-MNIST.
Nếu không thắng dropout thì báo đúng; câu hỏi mang tính nghiên cứu, không có
bảo đảm một kỹ thuật mới tự nghĩ luôn vượt baseline. Có thể tuning std bằng
validation, nhưng cần ngân sách và test độc lập để khẳng định cải thiện bền vững.

## 5.7 — Predicting House Prices on Kaggle (5 câu)

### Câu 1

Đã xây pipeline tạo **Le_Son_Truong_submission.csv**, hai cột Id/SalePrice,
1.459 dòng, dự đoán giá dương bằng trung bình dự đoán giá của 5 mô hình fold.
Đã kiểm tra schema, ID, số dòng và số hữu hạn. **Chưa nộp lên Kaggle và chưa có
điểm leaderboard**; điểm 5-fold local không được ghi thành điểm Kaggle.
Bạn đăng nhập, chấp nhận quy tắc House Prices và upload CSV ở trang cuộc thi:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques.
Sau khi có điểm, chụp màn hình và bổ sung vào hồ sơ. Đây là bước còn cần tài khoản cá nhân.

### Câu 2

Không luôn nên mean-impute. Ví dụ giá trị thu nhập cao dễ bị người trả lời bỏ trống:
missingness phụ thuộc chính giá trị không quan sát (MNAR), mean từ người trả lời
bị thiên lệch. Với nhà không có garage, missing diện tích garage có thể nghĩa là
không tồn tại, không phải gara “trung bình”. Cần hiểu ngữ nghĩa, có missing indicator,
imputation có điều kiện hoặc mô hình phù hợp; mọi thống kê chỉ fit trên train fold.

### Câu 3

Notebook dùng 5-fold cố định seed42 để so baseline linear SGD LR=.01,10 epoch
của sách với 100 epoch và weight decay. Báo cáo mean±std **log RMSE**;
sách có chỗ in log MSE, phải phân biệt MSE với căn của nó khi so sánh.
Chọn cấu hình theo CV, không dùng test labels. Lưới hữu hạn không bảo đảm tối ưu
toàn cục; repeated tuning có selection bias.

### Câu 4

Thử MLP một tầng64 và hai tầng64→32, ReLU, Adam LR=.003,100 epoch,
Dropout .1 và decay tương ứng 1e−4/1e−3. Đây là mở rộng được câu hỏi yêu cầu,
không phải giả định sách mặc định dùng Adam/100 epoch.
So CV với baseline, chọn mô hình tốt nhất trong tập đo rồi ensemble 5 fold để
dự đoán. Mạng sâu hơn không nhất thiết tốt hơn dữ liệu bảng nhỏ.

### Câu 5

Bỏ scaling làm các feature chênh lệch độ lớn khiến tối ưu kém điều kiện,
gradient bị biến có scale lớn chi phối; cùng LR có thể phân kỳ. Notebook chạy
baseline không scale ở fold1 và ghi trạng thái finite/diverged thực tế.
Scaling còn ảnh hưởng ý nghĩa regularization. Bản này fit mean/std và category
trong mỗi train fold rồi áp cho validation/test, thay vì dùng toàn bộ train+test
như cách đơn giản minh họa trong sách, để tránh leakage thống kê trong CV.
