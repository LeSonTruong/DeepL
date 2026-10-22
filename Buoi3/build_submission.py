"""Tạo notebook từ chính script, chạy thật, xuất hai ảnh gộp và ZIP bài nộp.

python Buoi3/build_submission.py --execute
python Buoi3/build_submission.py --package-only  # xuất lại từ kết quả đã có
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
import textwrap
import zipfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
SCRIPTS = ["bai02_class_training_loop", "bai02_homework_optimizer_compare"]
FOLDERS = ["bai1_learning_rate", "bai2_optimizer", "bai2_early_stopping"]
KERNEL_NAME = "deepl-pytorch"
KERNEL_DISPLAY = "DeepL (PyTorch)"
SETUP = '''from pathlib import Path
import importlib
import sys

cwd = Path.cwd().resolve()
candidates = [cwd, cwd / "Buoi3", *[p / "Buoi3" for p in cwd.parents]]
BASE = next((p for p in candidates if (p / "yeu_cau_thuc_hanh.md").is_file()), None)
if BASE is None:
    raise FileNotFoundError("Mở notebook từ repository DeepL hoặc thư mục Buoi3.")
sys.path.insert(0, str(BASE))
# Nạp lại cả module đúng đường dẫn để kernel không giữ phiên bản trước khi sửa.
importlib.invalidate_caches()
for name in ["common", "data", "bai02_class_training_loop"]:
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve().parent != BASE:
            del sys.modules[name]
        else:
            importlib.reload(sys.modules[name])
from IPython.display import display, Markdown, Image
try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise ModuleNotFoundError(
        f"Kernel hiện tại chưa cài PyTorch: {sys.executable}\\n"
        "Chọn Select Kernel -> DeepL (PyTorch), rồi Restart Kernel và Run All.\\n"
        "Python đã có PyTorch trong repository: "
        "Buoi1/Buoi1_PyTorch/.venv/Scripts/python.exe"
    ) from error
print("Python đang dùng:", sys.executable)
print("Lê Sơn Trường | Thiết bị: CPU | PyTorch:", torch.__version__)
'''


def notebook_from_script(stem, execute):
    source = (BASE / f"{stem}.py").read_text(encoding="utf-8")
    pieces = re.split(r"^# %%([^\n]*)\n", source, flags=re.M)
    cells = []
    for i in range(1, len(pieces), 2):
        marker, body = pieces[i], pieces[i + 1].strip()
        if "markdown" in marker:
            markdown = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(markdown))
        else:
            body = re.sub(r'\nif __name__ == "__main__":\n    main\(\)\s*$', '', body).strip()
            cells.append(nbformat.v4.new_code_cell(body))
    cells[0].source += ("\n\n**Chọn kernel trước khi chạy:** `Select Kernel` → **DeepL (PyTorch)**. "
                        "Python tương ứng: `Buoi1/Buoi1_PyTorch/.venv/Scripts/python.exe`. "
                        "Nếu vừa đổi kernel, chọn Restart rồi Run All.")
    cells.insert(1, nbformat.v4.new_code_cell(SETUP))
    cells.append(nbformat.v4.new_markdown_cell(
        "## Chạy và hiển thị kết quả\n\n**Run All** chạy huấn luyện thực sự, "
        "giữ nguyên 5 epoch. Kết quả đã được lưu sẵn dưới đây; chỉ đọc thì không cần chạy lại."))
    if stem == SCRIPTS[0]:
        call = "summary, history = main()\ndisplay(summary)"
        folders = FOLDERS[:1]
    else:
        call = "baseline, advanced = main()\ndisplay(baseline[0])\ndisplay(advanced[0])"
        folders = FOLDERS[1:]
    cells.append(nbformat.v4.new_code_cell(call))
    for folder in folders:
        cells.append(nbformat.v4.new_code_cell(
            f'folder = BASE / "results" / "{folder}"\n'
            'display(Markdown((folder / "report.md").read_text(encoding="utf-8").split("![")[0]))\n'
            'for name in ["learning_curves", "comparison", "confusion_matrices", "prediction_examples"]:\n'
            '    display(Image(filename=str(folder / f"{name}.png")))'))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": KERNEL_NAME, "display_name": KERNEL_DISPLAY, "language": "python"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}})
    path = BASE / f"{stem}.ipynb"
    if execute:
        if KERNEL_NAME in KernelSpecManager().find_kernel_specs():
            manager = KernelManager(kernel_name=KERNEL_NAME)
        else:
            # Khi mang bài sang máy khác, dùng môi trường Python đang chạy builder.
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        print(f"Executing {path.name} — training, 5 epochs/configuration...", flush=True)
        try:
            NotebookClient(nb, km=manager, timeout=600,
                           resources={"metadata": {"path": str(BASE)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    elif path.exists():
        old = nbformat.read(path, as_version=4)
        previous = [c for c in old.cells if c.cell_type == "code"]
        current = [c for c in nb.cells if c.cell_type == "code"]
        if [c.source for c in previous] != [c.source for c in current]:
            raise RuntimeError("Code notebook thay đổi: cần --execute để output đúng với code.")
        for new, saved in zip(current, previous):
            new.execution_count, new.outputs = saved.execution_count, saved.outputs
    nbformat.validate(nb)
    nbformat.write(nb, path)


def font(size, mono=False):
    candidates = ([Path("C:/Windows/Fonts/consola.ttf")] if mono else []) + [
        Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    return ImageFont.truetype(str(next(p for p in candidates if p.exists())), size)


def text_panel(title, lines, width=2550):
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=145, break_long_words=False) or [""])
    image = Image.new("RGB", (width, 135 + len(wrapped) * 32), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 20), "LÊ SƠN TRƯỜNG — BÀI THỰC HÀNH BUỔI 3", font=font(32), fill="#16324f")
    draw.text((35, 70), title, font=font(28), fill="#16324f")
    for index, line in enumerate(wrapped):
        draw.text((35, 125 + index * 32), line, font=font(23, mono=True), fill="#172b3a")
    return image


def summary_lines(folder):
    frame = pd.read_csv(BASE / "results" / folder / "summary.csv")
    first = frame.iloc[0]
    lines = [f"Train/validation: {first.train_samples}/{first.val_samples}; MLP 784 -> {first.hidden} -> 10; "
             f"ReLU; batch {first.batch_size}; seed42; CPU; tối đa 5 epoch.",
             f"Số tham số: {first.parameters:,}. Loss: CrossEntropyLoss (logits); ảnh float32 /255.", "",
             "Cấu hình              LR      Momentum  Epoch chạy/chọn   Train loss   Val loss   Accuracy   Macro-F1"]
    for r in frame.itertuples():
        lines.append(f"{r.config:<21} {r.lr:<8g} {r.momentum:<9g} {r.epochs_run:>4}/{r.selected_epoch:<9} "
                     f"{r.train_loss:>10.4f} {r.val_loss:>10.4f} {r.val_accuracy:>9.2%} {r.val_macro_f1:>10.4f}")
    if not first.early_stopping:
        history = pd.read_csv(BASE / "results" / folder / "history.csv")
        lines += ["", "KẾT QUẢ SAU TỪNG EPOCH",
                  "Cấu hình              Epoch  Device   Tham số       Train loss    Val loss    Val accuracy"]
        for r in history.itertuples():
            lines.append(f"{r.config:<21} {r.epoch:>3}    {r.device:<6} {r.parameters:>9,} "
                         f"{r.train_loss:>14.4f} {r.val_loss:>11.4f} {r.val_accuracy:>13.2%}")
    report = (BASE / "results" / folder / "report.md").read_text(encoding="utf-8")
    comments = report.split("## Nhận xét từ kết quả chạy\n\n")[1].split("![")[0]
    lines += [""] + [line.replace("**", "") for line in comments.splitlines() if line.strip()]
    return lines


def export_reports():
    reports = ["# Kết quả hai thực hành — Lê Sơn Trường", "",
               "Nguồn: baitap.txt, Slides_Hoc_sau_B2.pdf trang 26–27 và mã mẫu. Giữ đúng 5 epoch.", ""]
    for folder in FOLDERS:
        text = (BASE / "results" / folder / "report.md").read_text(encoding="utf-8")
        reports.append(text)
        reports[-1] = re.sub(r"\]\(([^)]+\.png)\)", rf"](results/{folder}/\1)", reports[-1])
    (BASE / "KET_QUA_VA_NHAN_XET.md").write_text("\n\n".join(reports), encoding="utf-8")

    output = BASE / "anh_nop_bai"
    output.mkdir(exist_ok=True)
    images = []
    for task, folder in [(1, FOLDERS[0]), (2, FOLDERS[1])]:
        pieces = [text_panel(f"THỰC HÀNH {task} — CẤU HÌNH, KẾT QUẢ VÀ NHẬN XÉT", summary_lines(folder))]
        for name in ["learning_curves", "comparison", "confusion_matrices", "prediction_examples"]:
            with Image.open(BASE / "results" / folder / f"{name}.png") as image:
                piece = image.convert("RGB")
                if piece.width > 2550:
                    piece = piece.resize((2550, round(piece.height * 2550 / piece.width)), Image.Resampling.LANCZOS)
                pieces.append(piece)
        if task == 2:
            pieces.append(text_panel("PHẦN NÂNG CAO — EARLY STOPPING", summary_lines(FOLDERS[2])))
            with Image.open(BASE / "results" / FOLDERS[2] / "learning_curves.png") as image:
                pieces.append(image.convert("RGB"))
        width = max(p.width for p in pieces)
        combined = Image.new("RGB", (width + 40, sum(p.height for p in pieces) + 24 * (len(pieces) + 1)), "#e9eef4")
        y = 24
        for piece in pieces:
            combined.paste(piece, ((combined.width - piece.width) // 2, y))
            y += piece.height + 24
        path = output / f"Le_Son_Truong_Buoi3_Bai_{task}.png"
        combined.save(path, compress_level=6)
        images.append(combined)
        print(f"Exported {path.name}: {combined.size}", flush=True)
    images[0].save(BASE / "Le_Son_Truong_Buoi3.pdf", save_all=True, append_images=images[1:], resolution=150)


def package():
    versions = {"python": sys.version, "author": "Lê Sơn Trường", "max_epochs": 5,
                "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob("*.py")},
                "originals_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (BASE / "ma_nguon_goc").glob("*.py")}}
    (BASE / "run_metadata.json").write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")
    for target, images_only in [(BASE / "Le_Son_Truong_Buoi3.zip", False),
                                (BASE / "Le_Son_Truong_Buoi3_Anh_Nop.zip", True)]:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            files = (BASE / "anh_nop_bai").glob("*.png") if images_only else BASE.rglob("*")
            for path in files:
                relative = path.relative_to(BASE)
                if not path.is_file() or path.suffix == ".zip" or any(
                    part in {"__pycache__", "data", ".ipynb_checkpoints"} for part in relative.parts):
                    continue
                archive.write(path, arcname=Path("Le_Son_Truong_Buoi3") / relative)
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
        print(f"ZIP: {target.name} ({target.stat().st_size / 1024**2:.1f} MiB)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    if not args.package_only:
        for stem in SCRIPTS:
            notebook_from_script(stem, args.execute)
    if all((BASE / "results" / f / "summary.csv").is_file() for f in FOLDERS):
        export_reports()
        package()
    else:
        print("Notebook đã tạo; chạy lại với --execute để huấn luyện và xuất kết quả.")
