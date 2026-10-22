"""Tạo hai notebook dùng chung mã .py và báo cáo Markdown tổng hợp.

python Buoi4/build_notebooks.py --execute   # chạy thật cả hai bài, lưu output
python Buoi4/build_notebooks.py --report-only  # cập nhật báo cáo từ kết quả đã có
"""
import argparse
import asyncio
import base64
import csv
import json
from pathlib import Path
import re
import sys

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from reporting import (cnn_blocks_markdown, flowchart_steps, training_flow_markdown,
                       workflow_markdown)

BASE = Path(__file__).resolve().parent
KERNEL = "deepl-pytorch"
TASKS = [("bai03_class_cnn_fashionmnist", "thuc_hanh_1"),
         ("bai03_homework_cnn_variants", "thuc_hanh_2")]

SETUP = '''from pathlib import Path
import importlib
import inspect
import re
import sys

cwd = Path.cwd().resolve()
candidates = [cwd, cwd / "Buoi4", *[p / "Buoi4" for p in cwd.parents]]
BASE = next((p for p in candidates if (p / "De_bai" / "baitap.txt").is_file()), None)
if BASE is None:
    raise FileNotFoundError("Mở notebook từ repository DeepL hoặc thư mục Buoi4.")
sys.path.insert(0, str(BASE))
try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise ModuleNotFoundError(
        "Chọn Select Kernel -> DeepL (PyTorch), rồi Restart Kernel và Run All. "
        f"Python hiện tại chưa có torch: {sys.executable}"
    ) from error

# Nạp đúng mã của Buổi 4, kể cả khi kernel đã dùng common.py của buổi khác.
importlib.invalidate_caches()
for name in ["reporting", "common", "bai03_class_cnn_fashionmnist", "bai03_homework_cnn_variants", "build_notebooks"]:
    sys.modules.pop(name, None)
import common
from IPython.display import Code, Image, Markdown, display
common.set_seed()
print("Người làm: Lê Sơn Trường")
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "| device:", common.DEVICE)
print("Train/validation: 6000/1200 từ official train; Adam; LR=0.001; batch64; 5 epoch.")
'''


def markdown(text):
    return nbf.v4.new_markdown_cell(text)


def flow_markdown_cell(text):
    """Nhúng ảnh vào notebook để sơ đồ hiển thị cả khi chuyển riêng file ipynb."""
    attachments = {}
    for relative in re.findall(r"!\[[^\]]*\]\((figures/flow_[^)]+\.png)\)", text):
        path = BASE / relative
        if not path.is_file():
            write_flowcharts()
        attachments[path.name] = {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
        text = text.replace(f"({relative})", f"(attachment:{path.name})")
    return nbf.v4.new_markdown_cell(text, attachments=attachments)


def code(text):
    return nbf.v4.new_code_cell(text)


def notebook_cells(stem, folder):
    first = folder == "thuc_hanh_1"
    title = "Thực hành 1 — So sánh MLP và CNN" if first else "Thực hành 2 — Các biến thể CNN"
    cells = [markdown(f"# {title}\n\n**Lê Sơn Trường — Buổi 4**. "
              "Theo `De_bai/baitap.txt` và các file mẫu.\n\n"
              "Chọn **Select Kernel → DeepL (PyTorch)**. Môi trường có sẵn: "
              "`Buoi1/Buoi1_PyTorch/.venv/Scripts/python.exe`.\n\n"
              "Notebook đã có output. **Run All huấn luyện lại thực sự** với 5 epoch mỗi cấu hình "
              "và cập nhật kết quả trong `results/`. Các hàm dùng trực tiếp từ `.py`; "
              "cell hiển thị mã lấy đúng nguồn hàm hiện tại, tránh hai bản code khác nhau."),
             flow_markdown_cell(workflow_markdown(folder, heading_level=2)),
             code(SETUP + f'\nimport {stem} as exercise')]
    cells += [markdown("## 1. Đầu vào, đầu ra và cách chia dữ liệu\n\n"
                       "Fashion-MNIST: ảnh xám `(N,1,28,28)`, nhãn long 0–9. "
                       "Chuyển ảnh thành float32 /255, không augmentation. "
                       "Train 6.000 và validation 1.200 ảnh không giao nhau, "
                       "lấy từ cùng hoán vị official train với seed42. "
                       "Đầu ra là 10 logits; `argmax` cho lớp dự đoán.\n\n" +
                       ("Bài này chỉ dùng train/validation; so sánh tại epoch cuối."
                        if first else "Chọn checkpoint và cấu hình bằng validation accuracy; hòa thì xét loss thấp hơn. "
                        "Chỉ sau đó mới đánh giá trên toàn bộ 10.000 ảnh official test.")),
              code('display(Code(inspect.getsource(common.make_data), language="python"))')]
    if first:
        cells += [markdown("## 2. Kiến trúc và hình dạng tensor\n\n"
                           "MLP: 784 → 256 → 10. CNN: hai block Conv → ReLU → MaxPool, "
                           "sau đó Flatten → Linear(1568,128) → ReLU → Linear(128,10). "
                           "Hai mô hình cùng seed, split, Adam, LR, batch và số epoch."),
                  code('display(Code(inspect.getsource(common.MLP), language="python"))\n'
                       'display(Code(inspect.getsource(common.CNN2Blocks), language="python"))\n'
                       'for batch_size in [1, 64]:\n'
                       '    exercise.print_shapes(common.CNN2Blocks(), batch_size)')]
    else:
        cells += [markdown("## 2. Baseline và các cấu hình khảo sát\n\n"
                           "Baseline: channels16/32, kernel3, không BatchNorm/Dropout. "
                           "Các biến thể: kernel5, channels8/16, channels32/64, BatchNorm, Dropout0.3. "
                           "Tổng cộng 6 cấu hình, bao phủ kernel3/5 và channel block1 8/16/32; "
                           "block2 có số kênh gấp đôi block1. "
                           "Mỗi biến thể chỉ thay một thuộc tính so với baseline."),
                  code('display(Code(inspect.getsource(common.CNN2Blocks), language="python"))\n'
                       'display(Code(inspect.getsource(exercise.experiments), language="python"))\n'
                       'display(exercise.experiments())\n'
                       'from reporting import variant_architecture_markdown\n'
                       'architecture_output = BASE / "results" / "thuc_hanh_2"\n'
                       'architecture_output.mkdir(parents=True, exist_ok=True)\n'
                       'architectures = exercise.export_architectures(exercise.experiments(), architecture_output, verbose=True)\n'
                       'display(Markdown(variant_architecture_markdown(architectures)))')]
    # Giải thích cạnh kiến trúc, trước khi chuyển sang vòng huấn luyện.
    architecture_index = next(i for i, c in enumerate(cells)
                              if c.cell_type == "markdown" and c.source.startswith("## 2."))
    cells.insert(architecture_index + 1, markdown(cnn_blocks_markdown(heading_level=3)))
    cells += [flow_markdown_cell(training_flow_markdown(heading_level=2))]
    cells += [markdown("## 3. Huấn luyện và đánh giá\n\n"
                       "Vòng học: zero_grad → forward → CrossEntropyLoss → backward → step. "
                       "Dùng `train()` khi học, `eval()` và `inference_mode()` khi đánh giá. "
                       "Loss lấy trung bình theo số mẫu, kể cả batch cuối thiếu. "
                       "Train loss là loss trong khi cập nhật; validation loss đo ở cuối epoch. "
                       "Thời gian gồm train + validation, không gồm ghi checkpoint hoặc vẽ hình."),
              code('display(Code(inspect.getsource(common.train_one_epoch), language="python"))\n'
                   'display(Code(inspect.getsource(common.evaluate), language="python"))\n'
                   'display(Code(inspect.getsource(common.fit), language="python"))')]
    if not first:
        cells += [markdown("## 4. Đánh giá test và ma trận của cả 6 cấu hình\n\n"
                           "Confusion matrix: hàng thật, cột dự đoán. Mỗi cấu hình dùng checkpoint "
                           "validation tốt nhất, đánh giá cùng 10.000 ảnh test. "
                           "Giữ nguyên mô hình chính được chọn bằng validation. "
                           "5 ảnh đúng và 5 ảnh sai là của mô hình chính."),
                  code('display(Code(inspect.getsource(exercise.predict_test), language="python"))\n'
                       'display(Code(inspect.getsource(exercise.final_test), language="python"))\n'
                       'display(Code(inspect.getsource(exercise.evaluate_all_configurations), language="python"))')]
    cells += [markdown("## Chạy thí nghiệm\n\n"
                       "Cell này gọi trực tiếp `main()` của script cùng tên; học đủ 5 epoch mỗi cấu hình. "
                       "Các file dữ liệu và checkpoint được tạo theo đúng quy trình trong mã nguồn."),
              code('exercise.main()'),
              markdown("## Bảng kết quả từng epoch, learning curve và nhận xét\n\n"
                       "Bảng chi tiết hiển thị đủ epoch 1–5 cho từng mô hình: train loss, val loss, "
                       "val accuracy, thay đổi accuracy và thời gian. Epoch in đậm là checkpoint "
                       "validation tốt nhất; xem ghi chú để phân biệt với epoch cuối và mô hình chính được chọn."),
              code(f'folder = BASE / "results" / "{folder}"\n'
                   'report = (folder / "report.md").read_text(encoding="utf-8")\n'
                   'display(Markdown(re.sub(r"!\\[[^\\]]*\\]\\([^)]+\\)", "", report)))\n'
                   'for filename in ' + repr(["learning_curves.png"] if first else
                       ["learning_curves.png", "confusion_matrix.png", "correct_examples.png", "incorrect_examples.png",
                        "test_all_configs/confusion_matrices.png"]) + ':\n'
                   '    display(Image(filename=str(folder / filename)))'),
              markdown("## Báo cáo tổng hợp\n\n"
                       "`BAO_CAO_BUOI4.md` tổng hợp hai bài, kèm hình và phân tích ma trận nhầm lẫn. "
                       "Khi thay đổi cấu hình, chạy lại cả hai notebook rồi kiểm tra kết quả."),
              code('from build_notebooks import write_combined_report\n'
                   'if all((BASE / "results" / task / "report.md").is_file() for task in ["thuc_hanh_1", "thuc_hanh_2"]):\n'
                   '    print("Đã cập nhật:", write_combined_report())\n'
                   'else:\n'
                   '    print("Chạy đủ hai bài để tạo báo cáo tổng hợp.")')]
    return cells


def build_notebook(stem, folder, execute, refresh=False):
    nb = nbf.v4.new_notebook(cells=notebook_cells(stem, folder), metadata={
        "kernelspec": {"name": KERNEL, "display_name": "DeepL (PyTorch)", "language": "python"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}})
    path = BASE / f"{stem}.ipynb"
    if execute or refresh:
        if refresh:
            previous = nbf.read(path, as_version=4)
            cached = next(c for c in previous.cells if c.cell_type == "code" and c.source == "exercise.main()")
            if cached.execution_count is None or not cached.outputs:
                raise RuntimeError("Chưa có output huấn luyện để dùng lại; chạy --execute trước.")
            training = next(c for c in nb.cells if c.cell_type == "code" and c.source == "exercise.main()")
            training.execution_count, training.outputs = cached.execution_count, cached.outputs
            training.metadata["tags"] = ["reuse-training-output"]
            nb.metadata["training_output_source"] = "Reused from the previous training run; display cells refreshed."
        if KERNEL in KernelSpecManager().find_kernel_specs():
            manager = KernelManager(kernel_name=KERNEL)
        else:
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        action = "Refreshing display cells without retraining" if refresh else "Executing"
        print(f"{action} {path.name}...", flush=True)
        try:
            NotebookClient(nb, km=manager, timeout=600,
                           skip_cells_with_tag="reuse-training-output" if refresh else "skip-execution",
                           resources={"metadata": {"path": str(BASE)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        assert all(c.execution_count is not None for c in nb.cells if c.cell_type == "code")
        assert not any(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    elif path.exists():
        old = nbf.read(path, as_version=4)
        before = [c for c in old.cells if c.cell_type == "code"]
        after = [c for c in nb.cells if c.cell_type == "code"]
        if [c.source for c in before] != [c.source for c in after]:
            raise RuntimeError("Code notebook thay đổi: dùng --execute, hoặc --refresh để cập nhật hiển thị từ kết quả đã lưu.")
        for new, previous in zip(after, before):
            new.execution_count, new.outputs = previous.execution_count, previous.outputs
    nbf.validate(nb)
    nbf.write(nb, path)
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_flowcharts():
    """Vẽ sơ đồ tĩnh từ cùng nội dung với Mermaid; không dùng kết quả huấn luyện."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    folder = BASE / "figures"
    folder.mkdir(exist_ok=True)
    titles = {"thuc_hanh_1": "Bài 1 — So sánh MLP và CNN",
              "thuc_hanh_2": "Bài 2 — Khảo sát các biến thể CNN",
              "train_validation": "Train → Validation trong mỗi epoch"}
    for task, title in titles.items():
        steps = flowchart_steps(task)
        fig, ax = plt.subplots(figsize=(11, 1.17 * len(steps) + 0.9))
        ax.set(xlim=(0, 1), ylim=(-0.1, len(steps) + 0.6))
        ax.axis("off")
        ax.set_title(title, fontsize=17, fontweight="bold", pad=16)
        left, width, height = 0.15, 0.81, 0.76
        for i, label in enumerate(steps):
            y = len(steps) - i - 0.5
            color = "#e2f2e9" if i in {0, len(steps)-1} else "#eaf0fb"
            if "Train → Validation" in label or "batch train" in label:
                color = "#fff0d6"
            ax.add_patch(FancyBboxPatch((left, y-height/2), width, height,
                         boxstyle="round,pad=0.015", facecolor=color,
                         edgecolor="#456078", linewidth=1.3))
            ax.text(left+width/2, y, label, ha="center", va="center", fontsize=11)
            if i < len(steps)-1:
                ax.annotate("", xy=(left+width/2, y-1+height/2+0.02),
                            xytext=(left+width/2, y-height/2-0.02),
                            arrowprops=dict(arrowstyle="->", color="#456078", lw=1.5))
        if task == "train_validation":
            start, end = len(steps)-4-0.5, len(steps)-0.5
            ax.plot([left, 0.065, 0.065], [start, start, end], color="#9b5c16", lw=1.4)
            ax.annotate("", xy=(left, end), xytext=(0.065, end),
                        arrowprops=dict(arrowstyle="->", color="#9b5c16", lw=1.4))
            ax.text(0.035, (start+end)/2, "Chưa đủ 5 epoch: quay lại Train",
                    rotation=90, ha="center", va="center", fontsize=9, color="#9b5c16")
        fig.tight_layout()
        fig.savefig(folder / f"flow_{task}.png", dpi=160, facecolor="white")
        plt.close(fig)


def write_running_guide():
    write_flowcharts()
    explanation = BASE / "GIAI_THICH_CNN_HAI_BLOCK.md"
    explanation.write_text(cnn_blocks_markdown(heading_level=1), encoding="utf-8")
    text = r'''# Quy trình chạy hai bài CNN — Buổi 4

**Sinh viên: Lê Sơn Trường.** Tài liệu mô tả cách thao tác và thứ tự xử lý trong code.

## 1. Chuẩn bị và thứ tự thao tác

1. Mở repository **DeepL** trong VS Code. Giữ các notebook, script, `common.py`,
   `reporting.py` và `build_notebooks.py` cùng thư mục `Buoi4/`.
2. Mở notebook bài 1, bấm **Select Kernel → DeepL (PyTorch)**. Cell thiết lập
   in đường dẫn Python và CPU/GPU để kiểm tra môi trường đang dùng.
3. Chọn **Run All** để chạy từ trên xuống. Cell `exercise.main()` mới bắt đầu
   huấn luyện; các cell `display(Code(...))` chỉ hiển thị mã để đọc.
4. Xem nhật ký shape, kết quả từng epoch, bảng MLP–CNN và learning curve.
5. Mở notebook bài 2, chọn cùng kernel và **Run All**. Đọc bảng cấu hình,
   checkpoint được chọn bằng validation, rồi kết quả test và ảnh dự đoán.
6. Đối chiếu kết quả bằng `verify_results.py`, sau đó mở `BAO_CAO_BUOI4.md`
   bằng Markdown Preview để đọc báo cáo tổng hợp.

Notebook có sẵn output nên có thể đọc ngay. **Run All sẽ học lại thực sự**;
chạy lại chỉ từ một cell giữa notebook có thể thiếu biến hoặc giữ mã cũ trong kernel.
Nếu vừa cập nhật file, mở lại notebook rồi **Restart Kernel → Run All**.

## 2. Chạy bằng terminal

Đứng ở thư mục gốc repository **DeepL**, chạy lần lượt:

```powershell
# Bài 1: MLP và CNN
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/bai03_class_cnn_fashionmnist.py
# Bài 2: các biến thể CNN, chọn bằng validation, đánh giá test
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/bai03_homework_cnn_variants.py
# Chỉ bổ sung/cập nhật ma trận cả 6 cấu hình từ checkpoint đã có
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/bai03_homework_cnn_variants.py --evaluate-only
# Kiểm tra split, checkpoint, dự đoán và confusion matrix
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/verify_results.py
# Cập nhật báo cáo từ các kết quả vừa chạy, không huấn luyện lại
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/build_notebooks.py --report-only
```

Chạy script cập nhật `results/` và báo cáo, nhưng không tự thay output đã lưu
trong notebook. Để chạy lại và đồng bộ cả hai notebook cùng báo cáo:

```powershell
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/build_notebooks.py --execute
```

Sau khi dùng `--evaluate-only`, đồng bộ phần hiển thị notebook từ kết quả đã lưu:

```powershell
& .\Buoi1\Buoi1_PyTorch\.venv\Scripts\python.exe Buoi4/build_notebooks.py --refresh
```

`--refresh` chạy các cell hiển thị, dùng lại output huấn luyện trước đó và bỏ qua
`exercise.main()`. Chạy **Run All** bình thường trong notebook vẫn huấn luyện lại.

Nếu chuyển sang máy khác, dùng Python có các thư viện trong `requirements.txt`
và chọn đúng môi trường đó làm kernel. Dữ liệu được dùng lại từ cache nếu có;
nếu chưa có, code tải Fashion-MNIST chính thức và kiểm SHA-256.

'''
    text += workflow_markdown("thuc_hanh_1", heading_level=2) + "\n"
    text += workflow_markdown("thuc_hanh_2", heading_level=2) + "\n"
    text += training_flow_markdown(heading_level=2) + "\n"
    text += ("## 3. Xem các sản phẩm\n\n"
             "- [Giải thích hai block Conv2d → ReLU → MaxPool, kèm shape từng bước](GIAI_THICH_CNN_HAI_BLOCK.md).\n"
             "- [Notebook bài 1](bai03_class_cnn_fashionmnist.ipynb).\n"
             "- [Notebook bài 2](bai03_homework_cnn_variants.ipynb).\n"
             "- [Báo cáo tổng hợp](BAO_CAO_BUOI4.md).\n\n"
             "Báo cáo và notebook có bảng từng epoch; bài 1 tổng kết ở epoch cuối, "
             "bài 2 tổng kết checkpoint val tốt nhất. Test của bài 2 là tập riêng chính thức.\n")
    path = BASE / "QUY_TRINH_CHAY.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_combined_report():
    from reporting import add_epoch_details, add_variant_details, add_test_comparison

    first, second = [BASE / "results" / task for _, task in TASKS]
    meta = read_json(first / "metadata.json")
    test = read_json(second / "test_metrics.json")
    lines = ["# Báo cáo hai bài thực hành CNN — Buổi 4", "",
             "**Sinh viên: Lê Sơn Trường**", "",
             "Nguồn: [Đề bài](De_bai/baitap.txt) và hai file gợi ý trong `De_bai/`.", "",
             "## 1. Mục tiêu, đầu vào và đầu ra", "",
             "Phân loại ảnh Fashion-MNIST thành 10 loại trang phục. Thực hành 1 so sánh MLP với CNN; "
             "thực hành 2 khảo sát kernel, số channel, BatchNorm và Dropout.", "",
             "- Đầu vào: ảnh xám `(N,1,28,28)`, pixel float32 /255; nhãn long 0–9.",
             "- Đầu ra: `(N,10)` logits; lấy argmax để xác định lớp dự đoán. "
             "CrossEntropyLoss nhận logits thô, không thêm Softmax trước loss.",
             "- Train/validation: 6.000/1.200 mẫu không giao nhau từ official train, "
             "chia bằng hoán vị seed42. Test cuối: toàn bộ 10.000 ảnh official test, chỉ dùng ở bài 2.", "",
             "## 2. Quy trình và cấu hình chung", "",
             f"Adam, LR={meta['lr']:g}, batch={meta['batch_size']}, {meta['epochs']} epoch, seed={meta['seed']}; "
             f"thiết bị **{meta['device']}**, {meta['threads']} threads; PyTorch {meta['torch_version']}.", "",
             "Vòng học: zero_grad → forward → loss → backward → step. Đánh giá dùng eval và inference_mode. "
             "Loss được tính trung bình theo số mẫu; train loss ghi trong lúc cập nhật, validation loss "
             "đo cuối epoch. Thời gian/epoch gồm train + validation, không gồm ghi checkpoint/vẽ hình.", "",
             "Mỗi mô hình reset seed và thứ tự batch. Mô hình khác kiến trúc có số lượng/kích thước "
             "trọng số khác nhau, nên cùng seed không có nghĩa toàn bộ trọng số khởi tạo giống nhau.", ""]
    lines += [training_flow_markdown(heading_level=3), "",
              cnn_blocks_markdown(heading_level=3), ""]
    for index, directory in enumerate([first, second], 3):
        text = (directory / "report.md").read_text(encoding="utf-8")
        with (directory / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
            history = list(csv.DictReader(stream))
        with (directory / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            summaries = list(csv.DictReader(stream))
        selection = read_json(directory / "selection.json") if directory == second else None
        text = add_epoch_details(text, history, summaries, selection)
        if directory == second:
            from bai03_homework_cnn_variants import export_architectures
            architectures = export_architectures(read_json(directory / "configurations.json"), directory)
            text = add_variant_details(text, directory, architectures)
            text = add_test_comparison(text, directory)
        (directory / "report.md").write_text(text, encoding="utf-8")
        title, body = text.split("\n", 1)
        title = title.removeprefix("# ").split(" — ")[0]
        text = f"## {index}. {title}\n" + body
        prefix = directory.relative_to(BASE).as_posix()
        text = re.sub(r"(\[[^\]]*\]\()([^)]+)(\))", lambda m: m[1] + prefix + "/" + m[2] + m[3], text)
        heading, body = text.split("\n", 1)
        text = heading + "\n\n" + workflow_markdown(directory.name) + "\n" + body
        lines += [text, ""]

    with (second / "confusion_matrix.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    labels = [r[0] for r in rows[1:]]
    matrix = [[int(v) for v in row[1:]] for row in rows[1:]]
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(labels)))
    recalls = [matrix[i][i] / sum(matrix[i]) for i in range(len(labels))]
    worst, best = min(range(len(labels)), key=recalls.__getitem__), max(range(len(labels)), key=recalls.__getitem__)
    mistakes = sorted(((matrix[i][j], labels[i], labels[j]) for i in range(len(labels))
                       for j in range(len(labels)) if i != j), reverse=True)
    lines += ["## 5. Phân tích confusion matrix trên test", "",
              "Hàng là nhãn thật, cột là nhãn dự đoán. Đường chéo chính đếm dự đoán đúng; "
              "các ô ngoài đường chéo chỉ ra những lớp bị nhầm lẫn.", "",
              f"Tổng đường chéo là **{correct:,}/{total:,}** ảnh, tương ứng accuracy **{correct / total:.2%}**.", "",
              f"- Lớp khó nhất trong lần chạy: **{labels[worst]}**, nhận đúng "
              f"{matrix[worst][worst]}/{sum(matrix[worst])} ảnh (**{recalls[worst]:.2%}**).",
              f"- Lớp nhận đúng cao nhất: **{labels[best]}**, "
              f"{matrix[best][best]}/{sum(matrix[best])} ảnh (**{recalls[best]:.2%}**)."]
    lines += [f"- Nhầm **{truth} → {predicted}**: {count} ảnh." for count, truth, predicted in mistakes[:3]]
    lines += ["", "Accuracy tổng thể chưa thể hiện hết chênh lệch giữa các lớp; "
              "nên xem thêm các ảnh dự đoán sai để tìm nguyên nhân trước khi đề xuất cải thiện.", "",
              "## 6. Kết luận và giới hạn", "",
              f"Mô hình được chọn là **{test['model']} tại epoch {test['epoch']}**, "
              f"validation accuracy **{test['validation_accuracy']:.2%}**. Đánh giá cuối trên test đạt "
              f"**{test['test_accuracy']:.2%}**, test loss **{test['test_loss']:.4f}**.", "",
              "CNN khai thác thông tin cục bộ và dùng chung trọng số theo vị trí; số tham số gần MLP "
              "không có nghĩa số phép tính hoặc thời gian tương đương. Các biến thể được so sánh "
              "trong cùng điều kiện; chỉ một seed và 5 epoch nên kết luận giới hạn ở thí nghiệm này. "
              "Các ma trận test của cả 6 cấu hình được bổ sung để đối chiếu lỗi; "
              "mô hình chính vẫn được chọn bằng validation trong selection.json.", "",
              "## 7. Mã nguồn và notebook", "",
              "- [Quy trình chạy notebook/terminal và các bước xử lý](QUY_TRINH_CHAY.md).",
              "- [Notebook thực hành 1](bai03_class_cnn_fashionmnist.ipynb) — "
              "[mã Python](bai03_class_cnn_fashionmnist.py).",
              "- [Notebook thực hành 2](bai03_homework_cnn_variants.ipynb) — "
              "[mã Python](bai03_homework_cnn_variants.py).",
              "- [Code dùng chung](common.py); [kiểm tra checkpoint/kết quả](verify_results.py).", ""]
    # Tạo ảnh trước khi kiểm tra đường dẫn trong báo cáo.
    write_running_guide()
    report = "\n".join(lines)
    for relative in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        if not (BASE / relative).is_file():
            raise FileNotFoundError(relative)
    path = BASE / "BAO_CAO_BUOI4.md"
    path.write_text(report, encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--refresh", action="store_true", help="Cập nhật hiển thị từ kết quả đã lưu, bỏ qua huấn luyện.")
    modes.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    print("Guide:", write_running_guide(), flush=True)
    if not args.report_only:
        for stem, folder in TASKS:
            print("Notebook:", build_notebook(stem, folder, args.execute, refresh=args.refresh), flush=True)
    if all((BASE / "results" / task / "report.md").is_file() for _, task in TASKS):
        print("Report:", write_combined_report(), flush=True)
