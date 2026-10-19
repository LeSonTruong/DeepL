# Chương 1 — Introduction

**Người làm: Lê Sơn Trường.** Nguồn đối chiếu: D2L 1.0.3, PDF trang 69,
Exercises 1.9. Các câu sau là lời giải/thảo luận, không phải dữ liệu khảo sát cá nhân.

## 1.9 — Exercises

### Câu 1 — Phần nào của chương trình có thể học?

Trong bài phân loại hai nhóm điểm, việc viết thủ công đường chia hoặc ngưỡng
tọa độ là một heuristic. Có thể thay bằng MLP học từ các cặp `(x1, x2, nhãn)`.
Trọng số được tối ưu theo loss thay vì người lập trình chọn từng ngưỡng.
Ví dụ khác: xếp hạng tìm kiếm có thể học từ truy vấn, kết quả hiển thị và phản hồi;
phản hồi phải được xử lý thiên lệch vị trí, không coi mọi click là nhãn hoàn hảo.

### Câu 2 — Nhiều ví dụ nhưng khó viết quy tắc

Nhận dạng chữ viết tay, phân loại ảnh quần áo, chuyển giọng nói thành văn bản.
Ta có thể cung cấp nhiều ví dụ đầu vào/đáp án, nhưng khó viết đầy đủ quy tắc
cho mọi nét chữ, góc chụp hay giọng nói. Deep learning phù hợp khi dữ liệu và
ngân sách tính toán đủ; không phải mọi bài toán đều cần mạng sâu.

### Câu 3 — Thuật toán, dữ liệu và tính toán

Dữ liệu cung cấp thông tin; thuật toán quyết định cách biểu diễn và học;
tính toán giới hạn kích thước mô hình, số thí nghiệm và tốc độ cập nhật.
Dữ liệu ít/dạng bảng thường nên có baseline tuyến tính; ảnh lớn có thể cần CNN.
Nhãn nhiễu hoặc dữ liệu không đại diện không được sửa chỉ bằng GPU mạnh hơn.
Chọn thuật toán phải xét số mẫu, số chiều, kiểu dữ liệu, RAM/VRAM và thời gian.

### Câu 4 — Nơi có thể thử end-to-end

Ví dụ một hệ thống đọc hóa đơn đang ghép OCR, nhận diện trường và luật kiểm tra;
có thể thử học trực tiếp ảnh → các trường cần trích xuất. Một dây chuyền kiểm tra
sản phẩm có thể học ảnh → loại lỗi thay cho đặc trưng thủ công + ngưỡng.
Đây là các tình huống minh họa, không khẳng định end-to-end luôn tốt hơn:
cần dữ liệu có nhãn phù hợp, tiêu chí đánh giá và khả năng kiểm tra lỗi của từng bước.
