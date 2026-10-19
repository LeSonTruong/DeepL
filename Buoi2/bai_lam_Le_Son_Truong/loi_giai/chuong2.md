# Chương 2 — Preliminaries

**Người làm: Lê Sơn Trường.** Đối chiếu PDF D2L 1.0.3 trang 77–118.
Các ô code trong notebook cùng chương kiểm chứng những phép tính và thí nghiệm dưới đây.

## 2.1 — Data Manipulation (2 câu)

### Câu 1

`X < Y` và `X > Y` so sánh từng phần tử, trả về tensor `torch.bool` cùng shape
sau broadcasting. Không trả về một boolean chung cho hai tensor. Notebook chạy
các thao tác tạo, reshape, indexing, nối, toán tử và so sánh của mục này.

### Câu 2

Ví dụ `(2,1,4) + (1,3,1)` cho shape `(2,3,4)`. So từ chiều cuối: hai kích thước
phải bằng nhau hoặc một bên bằng 1. `(2,3,4) + (2,2,4)` không tương thích ở
chiều giữa nên báo lỗi. Không tạo bản sao vật lý của mọi phần tử chỉ để broadcast.

## 2.2 — Data Preprocessing (5 câu)

### Câu 1

Tải **Abalone thật từ UCI**, đếm ô thiếu bằng `df.isna().mean()` và xác định kiểu
cột. Bộ này có 4.177 hàng, 9 cột: `Sex` là categorical; 7 số đo liên tục và
`Rings` là số nguyên. Xét toàn bộ bảng: 8/9 cột số, 1/9 categorical, không có
cột văn bản tự do; xét riêng 8 đặc trưng: 7/8 số và 1/8 categorical. Dữ liệu gốc
không có giá trị thiếu theo kiểm tra parser, nhưng không suy ra mọi số đo đều hợp lý.

### Câu 2

`df.loc[:, ['Sex','Length','Rings']]` hoặc `df[['Length','Rings']]` chọn theo tên.
`df.loc[df['Sex']=='F', ['Length','Rings']]` lọc hàng và cột theo điều kiện.
Tên cột giúp code ít phụ thuộc thứ tự hơn `iloc`.

### Câu 3

Đo `memory_usage(deep=True).sum()` và thời gian `read_csv` trên dữ liệu lặp tăng dần
trong notebook; không cố nạp đến khi máy hết RAM. Giới hạn thực tế gồm RAM,
bản sao trung gian, object/string overhead, tốc độ đĩa và parser. CSV thường tốn
RAM nhiều hơn kích thước file. Server nhiều RAM chỉ nâng giới hạn, không bỏ giới hạn.
Với bảng lớn, đọc `chunksize`, chỉ `usecols`, chọn dtype, dùng Parquet/Arrow hoặc xử lý ngoài RAM.
Chưa chạy trên server riêng nên không trình bày số đo server như kết quả thực nghiệm.

### Câu 4

One-hot với quá nhiều category gây bảng rất rộng. Có thể dùng sparse one-hot,
hashing hoặc embedding, gộp nhóm hiếm. ID duy nhất thường nên bỏ nếu không có
ý nghĩa dự đoán; mạng dễ ghi nhớ. Target encoding phải fit trong fold train để tránh leakage.

### Câu 5

NumPy với `save/load` phù hợp mảng đồng nhất; Pillow đọc ảnh; PyArrow/Polars xử lý
bảng; SQLite/DuckDB cho truy vấn đĩa. Notebook thực hiện vòng lưu/đọc `.npy` và
một ảnh PNG bằng Pillow. Chọn công cụ theo cấu trúc dữ liệu, không coi chúng thay thế hoàn toàn nhau.

## 2.3 — Linear Algebra (12 câu)

### Câu 1

`((Aᵀ)ᵀ)ij = (Aᵀ)ji = Aij`, nên `(Aᵀ)ᵀ=A` cho mọi phần tử.

### Câu 2

`(A+B)ᵀij=(A+B)ji=Aji+Bji=(Aᵀ+Bᵀ)ij`; hai ma trận phải cùng shape.

### Câu 3

Với A vuông, `(A+Aᵀ)ᵀ=Aᵀ+(Aᵀ)ᵀ=Aᵀ+A=A+Aᵀ`, nên đối xứng.

### Câu 4

X có shape `(2,3,4)` thì **len(X)=2**, vì len đo chiều đầu. Dự đoán trước rồi kiểm tra bằng code.

### Câu 5

Tensor có ít nhất một chiều: `len(X)==X.shape[0]`. Tensor scalar (0 chiều)
không có chiều đầu, gọi len sẽ lỗi; vì vậy không nói quy tắc đúng cho mọi tensor.

### Câu 6

A trong mục này shape `(5,4)`, `A.sum(axis=1)` shape `(5,)`; phép chia không
broadcast được với chiều cuối 4. Sửa thành `A / A.sum(dim=1,keepdim=True)`.
Nếu A vuông, code cũ có thể chạy nhưng chia theo cột bởi tổng các hàng, không phải
chuẩn hóa từng hàng. Cần xử lý tổng bằng 0 khi dữ liệu có thể gây trường hợp đó.

### Câu 7

Khoảng cách Manhattan là `|x1-x2|+|y1-y2|` nếu chỉ đi dọc các đường vuông góc.
Không đi chéo xuyên ô phố trong mô hình này; khoảng cách Euclid là cận dưới.

### Câu 8

Shape `(2,3,4)`: sum dim 0 → `(3,4)`; dim 1 → `(2,4)`; dim 2 → `(2,3)`.
Nếu `keepdim=True`, giữ chiều cộng với kích thước 1.

### Câu 9

`torch.linalg.norm(X)` không chỉ định dim/ord với tensor nhiều chiều: căn của
tổng bình phương trị tuyệt đối mọi phần tử (norm Euclid của tensor được làm phẳng).
Chỉ định ord/dim có quy tắc khác; không gọi đây là spectral norm của tensor nhiều chiều.

### Câu 10

A: `(2^10,2^16)`, B: `(2^16,2^5)`, C: `(2^5,2^14)`.
Tính `(AB)C` cần khoảng `2^31+2^29` phép nhân vô hướng; A(BC) cần
`2^35+2^40`. Trung gian AB có `2^15` phần tử; BC có `2^30` phần tử
(4 GiB FP32 chỉ cho trung gian), nên thứ tự đầu phù hợp hơn.
Notebook đo bản thu nhỏ cùng tỷ lệ, và tính chính xác chi phí/kích thước cho đề gốc;
không tạo trung gian nhiều GiB làm hết RAM.

### Câu 11

A: `(1024,65536)`, B: `(65536,32)`, C: `(32,65536)` nên AB và ACᵀ có cùng
số phép toán và shape kết quả. Layout contiguous/strides khác có thể khiến kernel
phải pack/copy và thời gian khác; không thể kết luận luôn một bên nhanh hơn.
Nếu `C=B.T` là view không clone thì `C.T` có cùng storage/layout với B,
hai phép toán tương đương hơn. Notebook đo thời gian lặp sau warm-up ở kích thước thu nhỏ.

### Câu 12

`torch.stack([A,B,C],dim=2)` cho `(100,200,3)`; `stacked[:,:,1]` chính là B.
Nếu stack mặc định dim=0 thì shape `(3,100,200)` và B là `stacked[1]`, không
phải lát cắt ở chiều thứ ba. Cần chọn dim=2 theo yêu cầu đề.

## 2.4 — Calculus (10 câu)

### Câu 1

Từ `f'(x)=lim[h→0](f(x+h)-f(x))/h`:

- Hằng c: tử số 0 nên đạo hàm 0.
- xⁿ, n nguyên dương: khai triển nhị thức, chia h còn `nx^(n−1)+O(h)`;
  giới hạn bằng `nx^(n−1)`. Với n thực, x>0, dùng xⁿ=exp(n log x) và chain rule.
- eˣ: thương sai phân `eˣ(eʰ−1)/h`; từ chuỗi eʰ=1+h+h²/2!+…,
  `(eʰ−1)/h→1`, nên đạo hàm eˣ.
- log x, x>0: đặt t=h/x, thương bằng `(1/x) log(1+t)/t`;
  giới hạn log(1+t)/t=1 (từ log là nghịch đảo exp) nên bằng 1/x.

### Câu 2

Quy tắc tổng: tách thương sai phân thành hai thương. Quy tắc tích: thêm/bớt
`f(x+h)g(x)`, được `f(x+h)·Δg/h+g(x)·Δf/h`; lấy giới hạn cho `fg'+gf'`.
Quy tắc thương, g(x)≠0: gộp tử `f(x+h)g(x)-f(x)g(x+h)` rồi chia cho
`h g(x+h)g(x)`; giới hạn là `(f'g−fg')/g²`. Các hàm khả vi nên liên tục tại x.

### Câu 3

Trong quy tắc tích đặt g=c, g'=0: `(cf)'=cf'`.

### Câu 4

Với x>0, log f=x log x; `f'/f=log x+1`, nên `(xˣ)'=xˣ(log x+1)`.

### Câu 5

f'(x)=0 nghĩa là tiếp tuyến ngang/điểm dừng, không tự suy ra cực trị.
f=x² tại 0 là cực tiểu; f=x³ tại 0 có đạo hàm 0 nhưng không cực trị.

### Câu 6

f(x)=x³−1/x, x≠0; f(1)=0 và f'(1)=3+1=4.
Tiếp tuyến tại 1: **y=4(x−1)**. Notebook vẽ miền x>0, không nối qua điểm kỳ dị x=0.

### Câu 7

f=3x₁²+5exp(x₂) ⇒ gradient `[6x₁,5exp(x₂)]`.

### Câu 8

f=‖x‖₂ ⇒ gradient `x/‖x‖₂` khi x≠0. Tại 0 norm không khả vi;
subgradient là mọi vector norm≤1. Nếu là norm bình phương thì gradient=2x,
kể cả ở 0; phân biệt chỉ số ₂ với lũy thừa ².

### Câu 9

`∂u/∂a=u_x x_a+u_y y_a+u_z z_a`; tương tự với b.
Dạng ma trận: Jacobian của ánh xạ ghép bằng tích các Jacobian theo thứ tự chain rule.

### Câu 10

Đạo hàm `f⁻¹(f(x))=x`: `(f⁻¹)'(f(x)) f'(x)=1`.
Vì vậy `(f⁻¹)'(y)=1/f'(f⁻¹(y))`, khi nghịch đảo khả vi và f' tại đó khác 0.

## 2.5 — Automatic Differentiation (8 câu)

### Câu 1

Đạo hàm bậc hai cần giữ/tạo đồ thị cho gradient. Với n tham số, Hessian đầy đủ
có n² phần tử và cần nhiều phép đạo hàm, nên thường tốn hơn gradient bậc một.
Hessian-vector product không cần lưu toàn Hessian và rẻ hơn đáng kể.

### Câu 2

PyTorch giải phóng intermediate sau backward mặc định; backward lại cùng graph
có thể lỗi. Muốn dùng lại graph cần `retain_graph=True` ở lần đầu, hoặc tính forward
lại. `.grad` cộng dồn nên phải zero trước bước cập nhật tiếp theo.
Notebook bắt lỗi chủ động rồi minh họa cách làm đúng.

### Câu 3

Điều kiện `while b.norm()<1000` vẫn là scalar với vector/matrix, nhưng `if b.sum()>0`
phải viết rõ; `if b>0` với nhiều phần tử không có giá trị chân lý duy nhất.
Output không scalar thì `backward()` cần gradient đầu vào cùng shape, hoặc lấy
`output.sum().backward()`; đây là vector-Jacobian product, không tự cho toàn Jacobian.
Ở một nhánh cố định, f(a)=c·a nên gradient của tổng là c ở mỗi tọa độ;
tại ranh giới đổi nhánh phải phân tích riêng.

### Câu 4

Notebook tạo x requires_grad, y=sin(x), rồi `y.sum().backward()`; vẽ y và x.grad.
Không dùng cos để tạo đường đạo hàm; cos chỉ có thể dùng làm đối chiếu sau đó.

### Câu 5

Đồ thị phụ thuộc: `x→a=x²→b=log(a)`; `x→c=sin(x)`;
`(b,c)→d=b*c`; `x→e=1/x`; `(d,e)→f=d+e`. Miền x≠0.

### Câu 6

Các đạo hàm cục bộ: a'=2x; db/da=1/a; c'=cos x;
∂d/∂b=c; ∂d/∂c=b; e'=−1/x²; ∂f/∂d=∂f/∂e=1.
Suy ra **f'=2 sin(x)/x + log(x²)cos(x) −1/x²**.

### Câu 7

Forward mode: truyền tangent dx=1 qua từng nút bằng chain rule để được công thức trên.
Reverse mode: bắt đầu adjoint f̄=1; d̄=ē=1, b̄=c, c̄=b, ā=c/a;
cộng các nhánh về x: x̄=2x·c/a + b cos x −1/x². Hai cách cho cùng kết quả;
notebook so sánh với autograd.

### Câu 8

Hàm Rⁿ→Rᵐ: forward thường phù hợp khi n nhỏ, cần khoảng n lượt cho Jacobian;
reverse phù hợp m nhỏ, cần khoảng m lượt và lưu intermediate của forward.
Loss mạng thường scalar, n lớn nên reverse mode hiệu quả. Có thể dùng JVP/VJP,
checkpointing hoặc vector hóa tùy bộ nhớ và đại lượng cần tính.

## 2.6 — Probability and Statistics (8 câu)

### Câu 1

Với các lần tung đồng xu IID có xác suất p cố định, càng nhiều mẫu càng xác định
chính xác p: Var(p̂)=p(1−p)/n→0. Đây là bất định về tham số, không phải biết
chắc kết quả của lần tung tiếp theo.

### Câu 2

Ngay khi biết p=0.5 chính xác, một lần tung tiếp theo vẫn ngẫu nhiên; bất định
không giảm về 0 chỉ nhờ thêm dữ liệu quá khứ. Phân biệt bất định tham số và nhiễu vốn có.

### Câu 3

Với Bernoulli IID: Var(p̂)=p(1−p)/n, giảm theo 1/n; độ lệch chuẩn theo 1/√n.
Chebyshev: `P(|p̂−p|≥ε)≤p(1−p)/(nε²)`. CLT cho
`√n(p̂−p)→N(0,p(1−p))`, giúp xấp xỉ xác suất chặt hơn khi điều kiện phù hợp.

### Câu 4

Nếu mẫu IID mean 0, variance 1: Var(z_m)=1/m và mỗi m riêng đều có cận
`P(|z_m|≥ε)≤1/(mε²)`. Nhưng zₘ dùng chung các mẫu với zₘ₊₁, nên các sự kiện
không độc lập; không nhân xác suất/cận như thể độc lập. Nếu mẫu không độc lập,
phải thêm covariance, không mặc định Var=1/m.

### Câu 5

Đặt a=P(A), b=P(B): `max(a,b)≤P(A∪B)≤min(1,a+b)`;
`max(0,a+b−1)≤P(A∩B)≤min(a,b)`. Từ công thức cộng xác suất và sự bao hàm tập hợp.

### Câu 6

Chuỗi Markov A→B→C: `P(A,B,C)=P(A)P(B|A)P(C|B)` vì C độc lập có điều kiện với A khi biết B.

### Câu 7

Đây là phép tính xác suất giả định của bài tập, không phải hướng dẫn xét nghiệm thực tế.
Khi H=0: bảng (D1,D2) có `(1,1)=0.02`, `(1,0)=0.08`, `(0,1)=0.08`,
`(0,0)=0.82`. Khi H=1, xác suất cả hai dương là 0.99²=0.9801.
Với prior 0.0015:

- Sau một dương: `0.99·0.0015/(0.99·0.0015+0.1·0.9985)`.
- Sau hai dương: `0.9801·0.0015/(0.9801·0.0015+0.02·0.9985)`.

Notebook tính giá trị số; không dùng 0.1² cho người khỏe vì đề nói hai kết quả có phụ thuộc.

### Câu 8

Đây là bài toán toán học danh mục giả định. Với tổng αᵢ=1:
E[return]=αᵀμ; Var(return)=αᵀΣα. Nếu α≥0 và chỉ tối đa kỳ vọng, đặt toàn bộ vào
tài sản có μ lớn nhất (chia tùy ý nếu đồng hạng). Nếu cho short vô hạn và không
ràng buộc rủi ro, bài toán có thể không bị chặn. Có giới hạn rủi ro v:
`max αᵀμ`, subject to `1ᵀα=1, αᵀΣα≤v` và α≥0 nếu không cho short.
Đề chỉ yêu cầu lập bài toán, không yêu cầu giải quadratic program.

## 2.7 — Documentation

PDF không có Exercises đánh số ở mục này. Notebook minh họa `help`, docstring và
chữ ký hàm để luyện tra API, không tự thêm chúng thành câu hỏi của sách.
