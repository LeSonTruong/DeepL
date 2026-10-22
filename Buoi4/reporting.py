"""Bảng từng epoch từ history thật; dùng chung cho script, notebook và báo cáo."""
import csv
import json
import math
import re

BEGIN = "<!-- BEGIN EPOCH_DETAILS -->"
END = "<!-- END EPOCH_DETAILS -->"
VARIANT_BEGIN = "<!-- BEGIN VARIANT_DETAILS -->"
VARIANT_END = "<!-- END VARIANT_DETAILS -->"
TEST_BEGIN = "<!-- BEGIN TEST_COMPARISON -->"
TEST_END = "<!-- END TEST_COMPARISON -->"


def test_comparison_markdown(output):
    """Bảng và hình của mọi checkpoint; không thay đổi lựa chọn theo validation."""
    with (output / "test_all_configs" / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lines = [f"### Ma trận nhầm lẫn trên test của cả {len(rows)} cấu hình", "",
             "Mỗi ma trận dùng **checkpoint validation tốt nhất của cấu hình đó**, "
             "đánh giá trên cùng **10.000 ảnh official test**. Hàng là nhãn thật, cột là dự đoán. "
             "Đây là phần bổ sung để đối chiếu lỗi của các cấu hình; mô hình chính vẫn được "
             "chọn bằng validation trong `selection.json`.", "",
             "| Cấu hình | Epoch | Val accuracy | Test accuracy | Đúng / Sai | Ma trận riêng |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        name = row["model"]
        label = f"**{name} (được chọn)**" if row["selected_by_validation"] == "True" else name
        folder = f"test_all_configs/{name}"
        lines.append(f"| {label} | {row['epoch']} | {float(row['validation_accuracy']):.2%} | "
                     f"{float(row['test_accuracy']):.2%} | {int(row['correct']):,} / {int(row['incorrect']):,} | "
                     f"[Ảnh]({folder}/confusion_matrix.png) · [CSV]({folder}/confusion_matrix.csv) |")
    lines += ["", "Các ô trong hình tổng hợp dùng cùng thang màu 0–1.000; "
              "mỗi hàng có 1.000 ảnh thật của lớp tương ứng. "
              "Tổng đường chéo chia cho 10.000 bằng test accuracy của cấu hình.", "",
              "![Ma trận nhầm lẫn của 6 cấu hình](test_all_configs/confusion_matrices.png)", "",
              "[Bảng test đầy đủ, có test loss](test_all_configs/summary.csv)", ""]
    return "\n".join(lines)


def add_test_comparison(report, output):
    if not (output / "test_all_configs" / "summary.csv").is_file():
        return report
    block = TEST_BEGIN + "\n\n" + test_comparison_markdown(output) + "\n" + TEST_END
    pattern = re.compile(re.escape(TEST_BEGIN) + r".*?" + re.escape(TEST_END), re.S)
    if pattern.search(report):
        return pattern.sub(lambda match: block, report)
    return report.rstrip() + "\n\n" + block + "\n"


def variant_architecture_markdown(architectures):
    lines = ["### Kiến trúc và shape của đủ 6 cấu hình", "",
             "Shape có thứ tự `(N,C,H,W)`; N là số ảnh trong batch. "
             "Các shape dưới đây được ghi bằng hook trên `forward()` thật, "
             "với input giả bằng 0, chế độ eval trên CPU; đây là kiểm tra kiến trúc.", "",
             "| Cấu hình | Sau Conv1 | Sau Pool1 | Sau Conv2 | Sau Pool2 | Sau Flatten | Đầu ra |",
             "|---|---|---|---|---|---|---|"]
    for detail in architectures:
        layers = {layer["step"]: layer for layer in detail["traces"][0]["layers"]}
        shapes = ["`(" + ",".join(["N", *map(str, layers[name]["output_shape"][1:])]) + ")`"
                  for name in ["conv1", "pool1", "conv2", "pool2", "flatten", "fc2"]]
        lines.append(f"| {detail['config']['name']} | " + " | ".join(shapes) + " |")
    lines += ["", "ReLU giữ shape. Kernel3 dùng padding1, kernel5 dùng padding2 nên convolution "
              "giữ H,W; MaxPool2d(2) mới giảm 28→14→7. BatchNorm giữ shape và nằm giữa "
              "Conv–ReLU trong mỗi block. Dropout0.3 nằm sau ReLU của Linear128; "
              "các cấu hình khác có Dropout p=0. Phần phân loại: Flatten → Linear128 → "
              "ReLU → Dropout → Linear10. `argmax` trên 10 logits cho nhãn dự đoán.", "",
              "### Phân bố số tham số", "",
              "| Cấu hình | Block 1 (Conv + BN) | Block 2 (Conv + BN) | Phân loại (2 Linear) | Tổng |",
              "|---|---:|---:|---:|---:|"]
    for detail in architectures:
        lines.append(f"| {detail['config']['name']} | {detail['block1_parameters']:,} | "
                     f"{detail['block2_parameters']:,} | {detail['classifier_parameters']:,} | "
                     f"{detail['parameters']:,} |")
    baseline = architectures[0]
    lines += ["", "Conv có `C_out × (C_in × k × k + 1)` tham số; Linear có "
              "`n_out × (n_in + 1)`, đã tính bias. BatchNorm có 2 tham số học được "
              "cho mỗi kênh (scale và bias); running statistics không tính vào tham số học được. "
              "ReLU, MaxPool, Flatten, Dropout và Identity không có tham số học được.", "",
              f"Ở baseline, phần phân loại chiếm **{baseline['classifier_parameters'] / baseline['parameters']:.2%}** "
              "tham số. Tăng số kênh còn làm tăng đầu vào của Linear đầu tiên. "
              "Dropout không thêm tham số; BatchNorm thêm một lượng nhỏ tham số.", ""]
    return "\n".join(lines)


def add_variant_details(report, output, architectures):
    """Bổ sung thông tin báo cáo bài 2 từ kiến trúc thật và kết quả đã lưu."""
    def read_json(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    def read_csv(name):
        with (output / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    meta, selected, test = [read_json(name) for name in
                             ["metadata.json", "selection.json", "test_metrics.json"]]
    summaries, history = read_csv("summary.csv"), read_csv("history.csv")
    by_name = {row["model"]: row for row in summaries}
    for detail in architectures:
        total = sum(detail[key] for key in ["block1_parameters", "block2_parameters", "classifier_parameters"])
        assert total == detail["parameters"] == int(by_name[detail["config"]["name"]]["parameters"])
    with (output / "confusion_matrix.csv").open(encoding="utf-8-sig", newline="") as stream:
        cm_rows = list(csv.reader(stream))
    labels, matrix = cm_rows[0][1:], [[int(x) for x in row[1:]] for row in cm_rows[1:]]
    correct = sum(matrix[i][i] for i in range(len(labels)))
    total = sum(map(sum, matrix))
    worst = min(range(len(labels)), key=lambda i: matrix[i][i] / sum(matrix[i]))
    mistakes = sorted(((matrix[i][j], labels[i], labels[j]) for i in range(len(labels))
                       for j in range(len(labels)) if i != j), reverse=True)
    baseline = by_name["baseline"]
    lines = ["### Mục tiêu, đầu vào và đầu ra của bài 2", "",
             "Dùng CNN của bài 1 làm baseline, rồi khảo sát riêng từng thay đổi: kernel, "
             "số kênh, BatchNorm và Dropout. Bài 1 so sánh MLP–CNN tại epoch cuối; "
             "bài 2 chọn checkpoint tốt nhất bằng validation rồi mới đánh giá test.", "",
             "| Thành phần | Thông tin |", "|---|---|",
             f"| Dữ liệu | {meta['dataset']}; ảnh xám 28×28, 10 lớp trang phục |",
             f"| Train | {meta['train_size']:,} ảnh từ official train; dùng cập nhật trọng số |",
             f"| Validation | {meta['validation_size']:,} ảnh khác từ official train; chọn epoch/cấu hình |",
             f"| Test | {test['test_size']:,} ảnh official test độc lập; đánh giá cuối và phân tích ma trận từng cấu hình |",
             "| Tiền xử lý | Pixel float32 /255 thuộc [0,1], không augmentation; nhãn long 0–9 |",
             "| Đầu vào | `(N,1,28,28)` |",
             "| Đầu ra | `(N,10)` logits; argmax cho nhãn; CrossEntropyLoss nhận trực tiếp logits |", "",
             "### Thiết lập và quy trình huấn luyện", "",
             "| Thành phần | Giá trị |", "|---|---|",
             f"| Thiết bị / môi trường | {meta['device']}; PyTorch {meta['torch_version']}; {meta['threads']} threads |",
             f"| Seed | {meta['seed']}; cùng split và thứ tự batch cho mọi cấu hình |",
             f"| Optimizer / learning rate | {meta['optimizer']} / {meta['lr']:g} |",
             f"| Epoch | {meta['epochs']} cho mỗi cấu hình; {len(architectures)} cấu hình, {len(history)} dòng log |",
             f"| Batch | Train {meta['batch_size']}; validation/test 256 |",
             f"| Số batch train/epoch | {math.ceil(meta['train_size'] / meta['batch_size'])}; "
             f"batch cuối {meta['train_size'] % meta['batch_size'] or meta['batch_size']} ảnh, vẫn được sử dụng |",
             "| Loss | CrossEntropyLoss; lấy trung bình theo số mẫu |", "",
             "Mỗi epoch: **train() → zero_grad → forward → loss → backward → optimizer.step** "
             "trên các batch train; sau đó **eval() + inference_mode()** trên validation. "
             "Chạy đủ số epoch, không Early Stopping. Thời gian/epoch gồm train + validation, "
             "không gồm kiểm tra shape, ghi checkpoint hay vẽ biểu đồ. "
             "Train loss đo trong lúc cập nhật; validation loss đo ở cuối epoch.", "",
             variant_architecture_markdown(architectures),
             "[Nhật ký shape từng lớp, batch 1 và 64](shapes.txt) · "
             "[Kiến trúc và tham số dạng JSON](architectures.json)", "",
             "### Mức thay đổi so với baseline", "",
             "So sánh accuracy tại checkpoint validation tốt nhất của từng cấu hình; "
             "Δ accuracy tính bằng điểm phần trăm. Tỷ lệ thời gian >1 nghĩa là chậm hơn baseline "
             "trong lần chạy này; thời gian phụ thuộc máy và tải hệ thống.", "",
             "| Cấu hình | Thay đổi duy nhất | Δ tham số | Δ val accuracy (điểm %) | Thời gian / baseline |",
             "|---|---|---:|---:|---:|"]
    changes = {"baseline": "Mốc: k3, kênh16/32", "kernel5": "Kernel3→5, padding1→2",
               "channels8": "Kênh16/32→8/16", "channels32": "Kênh16/32→32/64",
               "batchnorm": "Thêm BatchNorm sau mỗi Conv", "dropout03": "Dropout p=0→0.3 sau Linear128 + ReLU"}
    for row in summaries:
        delta_params = int(row["parameters"]) - int(baseline["parameters"])
        delta_acc = 100 * (float(row["best_val_accuracy"]) - float(baseline["best_val_accuracy"]))
        ratio = float(row["mean_seconds_per_epoch"]) / float(baseline["mean_seconds_per_epoch"])
        lines.append(f"| {row['model']} | {changes[row['model']]} | {delta_params:+,} | {delta_acc:+.2f} | {ratio:.2f}× |")
    lines += ["", "### Chọn checkpoint, đánh giá test và đọc lỗi", "",
              "1. Trong mỗi cấu hình, giữ epoch có validation accuracy lớn nhất; hòa thì chọn "
              "validation loss nhỏ hơn; hòa cả hai giữ epoch gặp trước.",
              "2. So sánh checkpoint tốt nhất của 6 cấu hình theo cùng quy tắc; lưu lựa chọn "
              "vào `selection.json` trước khi đọc test.",
              f"3. Nạp [{selected['model']}_best.pt]({selected['model']}_best.pt) ở epoch "
              f"**{selected['epoch']}**, có validation accuracy **{selected['validation_accuracy']:.2%}**, "
              f"validation loss **{selected['validation_loss']:.4f}**.",
              f"4. Đánh giá đủ **{test['test_size']:,}** ảnh test: accuracy **{test['test_accuracy']:.2%}**, "
              f"loss **{test['test_loss']:.4f}**. Phần ma trận bổ sung đánh giá cả 6 checkpoint "
              "trên test để phân tích lỗi, giữ nguyên lựa chọn đã khóa bằng validation.", "",
              f"Confusion matrix: **hàng là nhãn thật, cột là dự đoán**. Tổng đường chéo "
              f"**{correct:,}/{total:,}** là số ảnh đúng; còn **{total-correct:,}** ảnh sai. "
              f"Lớp có tỷ lệ nhận đúng thấp nhất là **{labels[worst]}**: "
              f"{matrix[worst][worst]}/{sum(matrix[worst])} ảnh "
              f"({matrix[worst][worst] / sum(matrix[worst]):.2%}).", ""]
    lines += [f"- Nhầm {truth} → {predicted}: {count} ảnh." for count, truth, predicted in mistakes[:3]]
    lines += ["", "Ảnh minh họa gồm 5 ảnh đúng và 5 ảnh sai đầu tiên theo thứ tự test, "
              "có nhãn thật/dự đoán; đây là ví dụ để quan sát lỗi. "
              "Độ chính xác và confusion matrix được tính trên toàn bộ test.", "",
              "### Gợi ý trình bày với giảng viên", "",
              "Trình bày theo thứ tự: **mục tiêu → dữ liệu/đầu ra → hai block và shape → "
              "thiết lập/vòng học → so sánh 6 cấu hình → chọn mô hình/test → lỗi và kết luận**.", "",
              f"> Em dùng {meta['dataset']}, chia {meta['train_size']:,} ảnh train và "
              f"{meta['validation_size']:,} ảnh validation từ tập train chính thức. "
              "Đầu vào là ảnh xám 1×28×28; đầu ra gồm 10 logits. CNN baseline có "
              "hai block Conv–ReLU–MaxPool với 16/32 kênh; sau hai lần pooling, "
              "tensor thành 32×7×7, Flatten thành 1568 rồi đưa qua Linear128 và Linear10.", "",
              f"> Em giữ {meta['optimizer']}, learning rate {meta['lr']:g}, batch {meta['batch_size']}, "
              f"seed {meta['seed']} và {meta['epochs']} epoch cho mỗi cấu hình. "
              "Em thay riêng kernel, số kênh, thêm BatchNorm hoặc Dropout để so sánh với baseline. "
              f"Mô hình được chọn là {selected['model']} tại epoch {selected['epoch']}, "
              f"validation accuracy {selected['validation_accuracy']:.2%}. Sau khi khóa lựa chọn, "
              f"em đánh giá {test['test_size']:,} ảnh test độc lập, đạt accuracy {test['test_accuracy']:.2%}.", "",
              f"> Confusion matrix cho thấy lớp {labels[worst]} còn khó phân biệt. "
              "Kết quả mới có một seed và 5 epoch nên em chưa kết luận cấu hình thắng luôn tốt nhất, "
              "hoặc khẳng định overfitting chỉ từ một epoch giảm accuracy. "
              "Hướng cải thiện có thể thử là thêm dữ liệu/augmentation, khảo sát learning rate "
              "và thời gian học dựa trên validation.", ""]
    block = VARIANT_BEGIN + "\n\n" + "\n".join(lines) + "\n" + VARIANT_END
    pattern = re.compile(re.escape(VARIANT_BEGIN) + r".*?" + re.escape(VARIANT_END), re.S)
    if pattern.search(report):
        return pattern.sub(lambda match: block, report)
    title, body = report.split("\n", 1)
    return title + "\n\n" + block + "\n" + body


def cnn_blocks_markdown(heading_level=2):
    """Giải thích đúng CNN2Blocks; dùng chung trong notebook và tài liệu."""
    text = r'''CNN **hai block** nghĩa là lặp hai lần cụm
**Conv2d → ReLU → MaxPool2d(2)**. Sau hai block mới đến phần phân loại.
Trong `common.py`, xem các chú thích `BLOCK 1`, `BLOCK 2`, `PHÂN LOẠI`
ở hàm `CNN2Blocks.forward()`.

Ví dụ dưới đây dùng **baseline**: kernel3, channels16/32, không BatchNorm/Dropout.
`N` là số ảnh trong batch (thường là 64), không phải số kênh.
Shape luôn có thứ tự **(số ảnh, số kênh, chiều cao, chiều rộng)**.

```text
Ảnh xám (N,1,28,28)
  │
  ├─ BLOCK 1
  │   Conv2d(1,16,3,padding=1) → (N,16,28,28)
  │   ReLU                    → (N,16,28,28)
  │   MaxPool2d(2)            → (N,16,14,14)
  │
  ├─ BLOCK 2
  │   Conv2d(16,32,3,padding=1)→ (N,32,14,14)
  │   ReLU                    → (N,32,14,14)
  │   MaxPool2d(2)            → (N,32,7,7)
  │
  └─ PHÂN LOẠI
      Flatten → (N,1568) → Linear → (N,128) → ReLU
      → Linear → (N,10) logits → argmax → nhãn dự đoán
```

**Bên trong từng phép tính:**

| Phép tính | Thực hiện thế nào? | Ý nghĩa |
|---|---|---|
| Conv2d | Trượt bộ lọc trên ảnh/đặc trưng. Tại mỗi vị trí, nhân từng giá trị với trọng số tương ứng, cộng lại trên cả vùng và các kênh đầu vào, rồi cộng bias. | Học đặc trưng cục bộ; dùng cùng trọng số ở các vị trí. |
| ReLU | Tính `max(0,x)`: số âm thành 0, số dương giữ nguyên. Không đổi shape. | Tạo tính phi tuyến, giúp mạng học quan hệ phức tạp. |
| MaxPool2d(2) | Mỗi kênh chia thành vùng 2×2, lấy giá trị lớn nhất; mặc định stride=2. | Giảm chiều cao/rộng còn một nửa, giữ số kênh. |

**Cụ thể với hai block:** Conv1 có 16 bộ lọc, mỗi bộ kích thước
`1×3×3`, tạo 16 bản đồ đặc trưng. Conv2 có 32 bộ lọc, mỗi bộ
`16×3×3`: mỗi giá trị đầu ra tổng hợp thông tin từ **cả 16 kênh**
của block 1, tạo 32 bản đồ đặc trưng mới. Các bộ lọc do mạng học
trong lúc huấn luyện, không phải bộ dò cạnh được đặt sẵn.

`padding=1` thêm viền 0 để convolution kernel3, stride1 giữ nguyên kích thước.
Công thức với dilation=1 là `H_out = floor((H + 2p - k)/s) + 1`;
ở conv1: `(28 + 2×1 - 3)/1 + 1 = 28`. Pooling mới giảm
`28→14` rồi `14→7`. Số ảnh trong batch luôn được giữ nguyên.

**Ví dụ nhỏ cho ReLU và MaxPool** (số minh họa, không phải kết quả huấn luyện):

```text
Một vùng đặc trưng    Sau ReLU        Sau MaxPool2d(2)
[-2,  3]             [0, 3]
[ 1, -4]       →     [1, 0]      →    3
```

**Sau hai block:** `32×7×7 = 1568` giá trị được trải phẳng cho mỗi ảnh.
`Linear(1568,128) → ReLU → Linear(128,10)` tạo 10 điểm số **logits**.
`argmax` chọn lớp có điểm cao nhất. Khi học, đưa logits trực tiếp vào
`CrossEntropyLoss`; không cần thêm Softmax trước loss.

**Các biến thể trong bài 2:**

- Kernel5 dùng padding2 nên vẫn giữ nguyên H,W sau convolution.
- Channels8 nghĩa là hai block có 8/16 kênh; channels32 nghĩa là 32/64 kênh.
  H,W vẫn đi theo `28→14→7`; số chiều sau Flatten thay đổi theo số kênh.
- BatchNorm: mỗi block thành `Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)`.
  Ở baseline, `bn1/bn2` là `Identity`, cho dữ liệu đi qua nguyên vẹn.
- Dropout0.3 nằm **sau ReLU của Linear128**, ngoài hai block convolution.
  Nó ngẫu nhiên đặt một phần giá trị về 0 khi train; tắt khi eval.

**Liên hệ với quá trình học:** luồng trên là **forward**. Tiếp theo,
`loss → backward → optimizer.step()` tính lỗi, truyền gradient ngược
và cập nhật trọng số. Conv2d và Linear có tham số học được; ReLU,
MaxPool và Flatten không có trọng số học được. Với biến thể BatchNorm,
các tham số scale/bias của BatchNorm cũng được học.
'''
    return "#" * heading_level + " CNN hai block: dữ liệu đi qua mạng thế nào?\n\n" + text


def flowchart_steps(task):
    """Một nguồn nội dung cho ảnh sơ đồ và mã Mermaid."""
    flows = {
        "thuc_hanh_1": [
            "Nạp FashionMNIST từ official train",
            "Chia 6.000 train + 1.200 validation",
            "In shape CNN từng lớp với batch 1 và 64",
            "Lần lượt tạo MLP và CNN\nCùng seed, split, Adam, LR, batch, số epoch",
            "Mỗi mô hình: Train → Validation\nLặp đủ 5 epoch; lưu log và checkpoint tốt nhất",
            "So sánh MLP–CNN tại epoch 5\nSố tham số, val accuracy, thời gian/epoch",
            "Xuất bảng kết quả, learning curve và nhận xét"],
        "thuc_hanh_2": [
            "Nạp official train\nChia 6.000 train + 1.200 validation như bài 1",
            "Tạo 6 cấu hình CNN; kiểm tra shape và tham số\nBaseline, kernel5, channels8, channels32, BatchNorm, Dropout0.3",
            "Mỗi cấu hình: Train → Validation, đủ 5 epoch\nGiữ checkpoint validation tốt nhất của từng cấu hình",
            "Chọn cấu hình thắng bằng validation\nAccuracy cao nhất; hòa thì loss thấp hơn",
            "Lưu selection.json rồi nạp checkpoint đã chọn",
            "Đánh giá mô hình đã chọn và các checkpoint còn lại\nCùng 10.000 ảnh official test; eval() + inference_mode()",
            "Xuất ma trận nhầm lẫn của 6 cấu hình\n5 ảnh đúng/sai của mô hình chính, learning curve và nhận xét"],
        "train_validation": [
            "Bắt đầu epoch: TRAIN\nmodel.train()",
            "Với mỗi batch train\nzero_grad → forward → CrossEntropyLoss\n→ backward → optimizer.step",
            "Hết các batch train: VALIDATION\nmodel.eval() + inference_mode()",
            "Duyệt mọi batch validation\nTính val loss trung bình và val accuracy",
            "Ghi train loss, val loss/accuracy, giây/epoch\nLưu checkpoint nếu validation tốt hơn",
            "Đủ 5 epoch: kết thúc mô hình/cấu hình"]}
    if task not in flows:
        raise ValueError(f"Sơ đồ không hợp lệ: {task}")
    return flows[task]


def flowchart_markdown(task):
    steps = flowchart_steps(task)
    labels = {"thuc_hanh_1": "Flow bài 1 — So sánh MLP và CNN",
              "thuc_hanh_2": "Flow bài 2 — Các biến thể CNN",
              "train_validation": "Flow Train → Validation bên trong mỗi epoch"}
    mermaid = ["flowchart TD"]
    for i, step in enumerate(steps):
        label = step.replace("\n", "<br/>")
        mermaid.append(f'    N{i}["{label}"]')
        if i:
            mermaid.append(f"    N{i-1} --> N{i}")
    if task == "train_validation":
        mermaid.append('    N4 -->|"Chưa đủ 5 epoch"| N0')
    return (f"![{labels[task]}](figures/flow_{task}.png)\n\n"
            "<details>\n<summary>Mã Mermaid để chỉnh sửa sơ đồ</summary>\n\n"
            "```mermaid\n" + "\n".join(mermaid) + "\n```\n\n</details>\n")


def training_flow_markdown(heading_level=3):
    return ("#" * heading_level + " Flow Train → Validation bên trong mỗi epoch\n\n"
            + flowchart_markdown("train_validation") + "\n"
            "**Train cập nhật trọng số. Validation chỉ đo chất lượng và chọn checkpoint.** "
            "Bước cập nhật được thực hiện sau mỗi batch train; validation chạy ở cuối epoch. "
            "Cả hai bài đều dùng vòng học này. Test riêng của bài 2 diễn ra sau khi đã chọn mô hình.\n")


def workflow_markdown(task, heading_level=3):
    """Mô tả đúng thứ tự main/fit; dùng chung cho notebook và các tài liệu."""
    first = task == "thuc_hanh_1"
    if task not in {"thuc_hanh_1", "thuc_hanh_2"}:
        raise ValueError(f"Bài thực hành không hợp lệ: {task}")
    title = "Thực hành 1: so sánh MLP–CNN" if first else "Thực hành 2: khảo sát các biến thể CNN"
    lines = ["#" * heading_level + " Quy trình chạy — " + title, "",
             flowchart_markdown(task), "", "**Luồng đọc nhanh:**", ""]
    if first:
        flow = ("Nạp official train → Chia 6000 train / 1200 validation\n"
                "→ In shape → Lần lượt tạo MLP và CNN cùng optimizer\n"
                "→ Với mỗi mô hình: 5 × [Train 1 epoch → Validation → Ghi log/checkpoint]\n"
                "→ So sánh hai mô hình tại epoch 5\n"
                "→ Lưu bảng kết quả, learning curve và nhận xét")
        steps = [
            ("Khởi tạo", "`main()` → `set_seed()`", "Đặt seed42, xác định CPU/GPU; tạo thư mục `results/thuc_hanh_1/`."),
            ("Nạp và chia dữ liệu", "`make_data()` → `read_official(train=True)`", "Kiểm SHA-256; lấy 6.000 train và 1.200 validation từ hoán vị official train; pixel float32 /255, nhãn long. Lưu `split.json`."),
            ("Theo dõi tensor", "`print_shapes()`", "Chạy input giả với batch 1 và 64 qua conv1, pool1, conv2, pool2, Flatten và đầu ra; lưu `shapes.txt`."),
            ("Tạo từng mô hình", "Vòng lặp MLP/CNN trong `main()` → `fit()`", "Reset seed và thứ tự batch; tạo model trên device, Adam LR0.001, CrossEntropyLoss; đếm tham số."),
            ("Huấn luyện một epoch", "`fit()` → `train_one_epoch()`", "Bật train(); với mỗi batch64: zero_grad → forward → loss → backward → step. Cộng loss theo số mẫu."),
            ("Đánh giá và lặp", "`fit()` → `evaluate()`", "Bật eval(), dùng inference_mode trên 1.200 ảnh validation; ghi val loss/accuracy và giây/epoch; lưu checkpoint nếu validation tốt hơn. Lặp đủ 5 epoch rồi chuyển sang mô hình tiếp theo."),
            ("So sánh và xuất kết quả", "Phần cuối `main()`", "So sánh MLP–CNN tại epoch cuối bằng tham số, val accuracy và thời gian trung bình/epoch; xuất CSV, learning curve và báo cáo.")]
        conclusion = ("Checkpoint tốt nhất vẫn được lưu để tham khảo. **Bảng so sánh chính của bài 1 dùng epoch 5**, "
                      "đồng thời bảng chi tiết giữ đủ các epoch 1–5. Bài 1 chỉ dùng train/validation.")
        outputs = ("`split.json`, `shapes.txt`, `history.csv`, `summary.csv`, `metadata.json`, "
                   "`MLP_best.pt`, `CNN_best.pt`, `learning_curves.png`, `report.md`.")
    else:
        flow = ("Nạp official train → Chia 6000 train / 1200 validation\n"
                "→ Với mỗi cấu hình: tạo CNN/optimizer rồi chạy\n"
                "  5 × [Train 1 epoch → Validation → Cập nhật checkpoint tốt nhất]\n"
                "→ Chọn cấu hình thắng và lưu selection.json → Nạp official test\n"
                "→ Đánh giá 10000 ảnh test → Ma trận mô hình chính và cả 6 cấu hình\n"
                "→ 5 ảnh đúng + 5 ảnh sai của mô hình chính\n"
                "→ Xuất bảng so sánh, learning curve và báo cáo")
        steps = [
            ("Khởi tạo và chia dữ liệu", "`main()` → `set_seed()` → `make_data()`", "Cùng split6000/1200 từ official train như bài 1; xác định device. Chưa nạp test ở bước này."),
            ("Khai báo và xem kiến trúc", "`experiments()` → `export_architectures()` → `describe_cnn()`", "Baseline; kernel5; channels8; channels32; BatchNorm; Dropout0.3. Mỗi biến thể chỉ đổi một thuộc tính; lưu cấu hình, shape từng lớp với batch 1/64 và tham số theo block/phân loại. Kiểm tra shape bằng input giả ở eval, trước khi tính thời gian huấn luyện."),
            ("Học từng cấu hình", "Vòng lặp trong `main()` → `fit()`", "Reset seed và batch order, dựng CNN và Adam LR0.001; chạy đủ 5 epoch. Mỗi epoch học trên train rồi đo val loss/accuracy, ghi thời gian."),
            ("Giữ checkpoint từng cấu hình", "Trong `fit()`", "Chọn epoch có val accuracy lớn nhất; hòa thì dùng val loss nhỏ hơn, hòa cả hai giữ epoch gặp trước. Lưu `<tên>_best.pt`."),
            ("Chọn cấu hình thắng", "Sau vòng lặp cấu hình trong `main()`", "So sánh các checkpoint tốt nhất bằng cùng quy tắc validation. Lưu quyết định vào `selection.json` trước khi gọi đánh giá test."),
            ("Đánh giá cuối và đối chiếu lỗi", "Nạp checkpoint → `final_test()` → `evaluate_all_configurations()`", "Nạp đủ 10.000 ảnh official test; eval() và inference_mode(), không cập nhật trọng số. Xuất test loss/accuracy và ma trận của mô hình đã chọn; bổ sung ma trận cả 6 checkpoint tốt nhất theo validation để đối chiếu lỗi."),
            ("Xuất kết quả", "`final_test()` và phần cuối `main()`", "Lưu đủ dự đoán test, ma trận CSV/PNG, 5 ảnh đúng và 5 ảnh sai đầu tiên có nhãn thật/dự đoán; hoàn thiện bảng cấu hình, learning curve, kết quả từng epoch và nhận xét.")]
        conclusion = ("**Chọn epoch và cấu hình hoàn toàn bằng validation.** "
                      "Các ma trận test của 6 cấu hình là phân tích bổ sung; "
                      "giữ nguyên mô hình chính đã chọn trong `selection.json`.")
        outputs = ("`split.json`, `configurations.json`, `architectures.json`, `shapes.txt`, "
                   "`history.csv`, `summary.csv`, `metadata.json`, "
                   "các checkpoint `*_best.pt`, `selection.json`, `test_metrics.json`, `test_predictions.csv`, "
                   "`confusion_matrix.csv/png`, `example_indices.csv`, `correct_examples.png`, "
                   "`incorrect_examples.png`, `learning_curves.png`, `report.md`, "
                   "`test_all_configs/summary.csv`, `test_all_configs/confusion_matrices.png` "
                   "và thư mục ma trận/dự đoán riêng cho mỗi cấu hình.")
    lines += ["```text", flow, "```", "",
              "| Bước | Công việc | Hàm thực hiện | Kết quả / ý nghĩa |",
              "|---:|---|---|---|"]
    lines += [f"| {i} | {name} | {function} | {description} |"
              for i, (name, function, description) in enumerate(steps, 1)]
    lines += ["", conclusion, "", f"**File kết quả trong `results/{task}/`:** {outputs}", ""]
    return "\n".join(lines)


def epoch_details(history, summaries, selection=None):
    lines = ["### Kết quả đầy đủ từng epoch", "",
             "**Train loss**: trung bình theo số mẫu trong lúc cập nhật. "
             "**Val loss/accuracy**: đánh giá ở cuối epoch. "
             "**Δ accuracy**: thay đổi so với epoch trước, tính bằng điểm phần trăm. "
             "**Giây**: thời gian train + validation của epoch đó.", "",
             "Epoch **in đậm** có validation accuracy cao nhất của cấu hình; "
             "nếu hòa, chọn validation loss nhỏ hơn. " +
             ("Bảng tổng kết bài 1 vẫn dùng epoch cuối để so sánh cùng ngân sách."
              if selection is None else
              "Dấu **Mô hình được chọn** chỉ checkpoint thắng giữa các cấu hình theo validation. "
              "Phần bổ sung có ma trận test của từng cấu hình; không dùng test để chọn epoch/mô hình."), "",
             "[Dữ liệu từng epoch: history.csv](history.csv)", ""]
    for summary in summaries:
        name = summary["model"]
        records = sorted((r for r in history if r["model"] == name), key=lambda r: int(r["epoch"]))
        if not records:
            raise ValueError(f"Không có history cho {name}")
        best = max(records, key=lambda r: (float(r["val_accuracy"]), -float(r["val_loss"])))
        min_loss = min(records, key=lambda r: float(r["val_loss"]))
        first, last = records[0], records[-1]
        lines += [f"#### {name}", "",
                  "| Epoch | Train loss | Val loss | Val accuracy | Δ accuracy (điểm %) | Giây | Ghi chú |",
                  "|---:|---:|---:|---:|---:|---:|---|"]
        previous = None
        for row in records:
            epoch, accuracy = int(row["epoch"]), float(row["val_accuracy"])
            delta = "—" if previous is None else f"{100 * (accuracy - previous):+.2f}"
            notes = []
            if epoch == int(best["epoch"]):
                notes.append("Val tốt nhất")
            if epoch == int(last["epoch"]):
                notes.append("Epoch cuối")
            if selection is not None and name == selection["model"] and epoch == int(selection["epoch"]):
                notes.append("**Mô hình được chọn**")
            label = f"**{epoch}**" if epoch == int(best["epoch"]) else str(epoch)
            lines.append(f"| {label} | {float(row['train_loss']):.4f} | {float(row['val_loss']):.4f} | "
                         f"{accuracy:.2%} | {delta} | {float(row['seconds']):.3f} | {'; '.join(notes) or '—'} |")
            previous = accuracy
        lines += ["", f"Từ epoch {int(first['epoch'])} → {int(last['epoch'])}: train loss "
                  f"{float(first['train_loss']):.4f} → {float(last['train_loss']):.4f}; val loss "
                  f"{float(first['val_loss']):.4f} → {float(last['val_loss']):.4f}; val accuracy "
                  f"{float(first['val_accuracy']):.2%} → {float(last['val_accuracy']):.2%}. "
                  f"Accuracy cao nhất là **{float(best['val_accuracy']):.2%} tại epoch {int(best['epoch'])}**.", ""]
        if int(min_loss["epoch"]) != int(best["epoch"]):
            lines += [f"Val loss thấp nhất lại ở **epoch {int(min_loss['epoch'])}** "
                      f"({float(min_loss['val_loss']):.4f}). Loss và accuracy đo hai khía cạnh khác nhau, "
                      "nên epoch có loss thấp nhất không nhất thiết có accuracy cao nhất.", ""]
    return "\n".join(lines).rstrip()


def add_epoch_details(report, history, summaries, selection=None):
    """Chèn hoặc cập nhật đúng một khối; không nhân đôi khi xuất lại báo cáo."""
    block = BEGIN + "\n\n" + epoch_details(history, summaries, selection) + "\n\n" + END
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), flags=re.S)
    if pattern.search(report):
        return pattern.sub(lambda match: block, report)
    position = report.find("![Learning curves]")
    if position < 0:
        return report.rstrip() + "\n\n" + block + "\n"
    return report[:position].rstrip() + "\n\n" + block + "\n\n" + report[position:]
