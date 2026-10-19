"""Gộp các trang kết quả thành 10 ảnh theo chương, giữ nguyên pixel nội dung.

Chạy độc lập sau build_submission.py; không cần huấn luyện lại mô hình.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "anh_nop_bai"
OUTPUT = BASE / "anh_nop_bai_gop"
GAP = 24
LABEL_HEIGHT = 34
HEADER_HEIGHT = 150
BACKGROUND = "#eef2f7"


def font(size):
    candidates = [Path("C:/Windows/Fonts/arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    return ImageFont.truetype(str(next(p for p in candidates if p.exists())), size)


def prepare_page(path):
    with Image.open(path) as original:
        image = original.convert("RGB")
    # Footer cũ nằm ở 97.5% chiều cao; vùng nội dung kết thúc trước 95%.
    # Bỏ footer lặp lại, thay bằng nhãn số trang gốc trên từng ô.
    body = image.crop((0, 0, image.width, int(image.height * .95)))
    ink = ImageChops.difference(body, Image.new("RGB", body.size, "white"))
    bbox = ink.convert("L").point(lambda v: 255 if v > 12 else 0).getbbox()
    if bbox is None:
        raise ValueError(f"Trang trống: {path.name}")
    left, top, right, bottom = bbox
    bbox = (max(0, left - 14), max(0, top - 14),
            min(body.width, right + 14), min(body.height, bottom + 14))
    crop = image.crop(bbox)
    # Biểu đồ cũ có khoảng trắng lớn giữa tiêu đề và hình: chỉ bỏ các dải
    # hoàn toàn trắng, giữ 32 pixel khoảng cách. Không chạm vào chữ/đường vẽ.
    nonwhite_rows = (np.asarray(crop) < 255).any(axis=(1, 2))
    spans = []
    cursor = 0
    row = 0
    while row < crop.height:
        if nonwhite_rows[row]:
            row += 1
            continue
        start = row
        while row < crop.height and not nonwhite_rows[row]:
            row += 1
        if row - start > 80:
            spans.append([cursor, start + 16])
            cursor = row - 16
    spans.append([cursor, crop.height])
    compact = Image.new("RGB", (crop.width, sum(end - start for start, end in spans)), "white")
    offset = 0
    for start, end in spans:
        compact.paste(crop.crop((0, start, crop.width, end)), (0, offset))
        offset += end - start
    return {"path": path, "image": compact, "crop_box": list(bbox), "retained_row_spans": spans,
            "page": int(path.name.split("_")[3])}


def balanced_columns(pages, count):
    """Chia các trang liên tiếp, tối thiểu hóa chiều cao cột dài nhất."""
    heights = [p["image"].height + LABEL_HEIGHT + GAP for p in pages]
    prefix = [0]
    for height in heights:
        prefix.append(prefix[-1] + height)

    @lru_cache(None)
    def solve(end, columns):
        if columns == 1:
            return prefix[end], (end,)
        best = None
        for cut in range(columns - 1, end):
            previous, ends = solve(cut, columns - 1)
            candidate = max(previous, prefix[end] - prefix[cut]), ends + (end,)
            if best is None or candidate < best:
                best = candidate
        return best

    _, ends = solve(len(pages), count)
    start = 0
    columns = []
    for end in ends:
        columns.append(pages[start:end])
        start = end
    return columns


def split_chapter(pages):
    heights = [p["image"].height + LABEL_HEIGHT + GAP for p in pages]
    total = sum(heights)
    # Ưu tiên tách giữa hai mục nếu độ dài hai phần không quá lệch.
    candidates = []
    for cut in range(1, len(pages)):
        imbalance = abs(sum(heights[:cut]) - total / 2) / total
        new_section = bool(re.search(r"_c[1-5]_", pages[cut]["path"].name))
        candidates.append((imbalance + (0 if new_section else .10), cut))
    cut = min(candidates)[1]
    return [pages[:cut], pages[cut:]]


def render(pages, title, target):
    count = 3 if len(pages) >= 9 else 2 if len(pages) >= 4 else 1
    columns = balanced_columns(pages, count)
    cell_width = max(p["image"].width for p in pages)
    width = count * cell_width + (count + 1) * GAP
    height = HEADER_HEIGHT + GAP + max(
        sum(p["image"].height + LABEL_HEIGHT + GAP for p in col)
        for col in columns)
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((GAP, 18), "LÊ SƠN TRƯỜNG — BÀI TẬP BUỔI 2", fill="#16324f", font=font(26))
    draw.text((GAP, 57), title, fill="#16324f", font=font(25))
    draw.text((GAP, 99), f"{len(pages)} trang gốc | Đọc từ trên xuống, rồi sang cột phải.",
              fill="#334155", font=font(18))
    records = []
    for column_index, column in enumerate(columns):
        x = GAP + column_index * (cell_width + GAP)
        y = HEADER_HEIGHT
        for p in column:
            tile = p["image"]
            draw.rectangle((x, y, x + cell_width - 1, y + LABEL_HEIGHT - 1), fill="#dce6f2")
            draw.text((x + 10, y + 5), f"Trang gốc {p['page']:03d}", fill="#16324f", font=font(18))
            y += LABEL_HEIGHT
            draw.rectangle((x, y, x + cell_width - 1, y + tile.height - 1), fill="white")
            canvas.paste(tile, (x, y))
            records.append({"source": p["path"].name, "page": p["page"],
                            "source_sha256": hashlib.sha256(p["path"].read_bytes()).hexdigest(),
                            "crop_box": p["crop_box"],
                            "retained_row_spans": p["retained_row_spans"],
                            "destination_box": [x, y, x + tile.width, y + tile.height]})
            y += tile.height + GAP
    canvas.save(target, compress_level=6)
    # Đọc lại PNG và đối chiếu pixel từng ô, tránh mất chữ khi ghép hoặc nén.
    with Image.open(target) as saved:
        for record in records:
            with Image.open(SOURCE / record["source"]) as original:
                expected = original.convert("RGB").crop(record["crop_box"])
            actual = saved.crop(record["destination_box"])
            offset = 0
            for start, end in record["retained_row_spans"]:
                source_piece = expected.crop((0, start, expected.width, end))
                saved_piece = actual.crop((0, offset, actual.width, offset + end - start))
                if ImageChops.difference(source_piece, saved_piece).getbbox() is not None:
                    raise AssertionError(f"Nội dung bị thay đổi: {record['source']}")
                offset += end - start
    return {"file": target.name, "size": [width, height],
            "columns": count, "source_pages": records,
            "pixel_verification": "passed"}


def main():
    OUTPUT.mkdir(exist_ok=True)
    paths = sorted(SOURCE.glob("Le_Son_Truong_[0-9][0-9][0-9]_*.png"))
    if not paths:
        raise FileNotFoundError(SOURCE)
    chapters = {i: [] for i in range(1, 6)}
    review = []
    chapter = 1
    for path in paths:
        match = re.search(r"_c([1-5])_", path.name)
        if match:
            chapter = int(match[1])
        page = prepare_page(path)
        (review if "_on_tap" in path.name else chapters[chapter]).append(page)

    outputs = []
    for chapter, pages in chapters.items():
        parts = [pages] if chapter == 1 else split_chapter(pages)
        for part, subset in enumerate(parts, 1):
            filename = f"Le_Son_Truong_Chuong_{chapter}_Phan_{part}.png"
            outputs.append(render(subset, f"CHƯƠNG {chapter} — PHẦN {part}/{len(parts)}", OUTPUT / filename))
            print(f"{filename}: {len(subset)} trang -> {outputs[-1]['size']}", flush=True)
    if review:
        filename = "Le_Son_Truong_On_tap_Loss_Backprop_MLP.png"
        outputs.append(render(review, "ÔN TẬP: LOSS, BACKPROPAGATION, TỐI ƯU VÀ MLP", OUTPUT / filename))

    used = [p["source"] for out in outputs for p in out["source_pages"]]
    assert Counter(used) == Counter(p.name for p in paths), "Thiếu hoặc lặp trang nguồn"
    manifest = {"source_count": len(paths), "merged_count": len(outputs),
                "method": "Lossless PNG; crop margins, repeated footer and empty white row bands; no content resizing",
                "verified": "Every source page included exactly once; every retained pixel matches source",
                "images": outputs}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""BỘ ẢNH GỘP NỘP BÀI — LÊ SƠN TRƯỜNG

Đã gộp {len(paths)} trang thành {len(outputs)} ảnh:
- Chương 1: 1 ảnh (kèm giới thiệu).
- Chương 2, 3, 4, 5: mỗi chương 2 ảnh.
- Ôn tập Loss, backpropagation, tối ưu hóa và MLP: 1 ảnh.

Upload các file .png trong thư mục này lên Drive; không cần upload lại 137 ảnh lẻ.
Ảnh dài: mở ảnh rồi phóng to để đọc, từ trên xuống trong cột trái,
sau đó sang cột tiếp theo bên phải. Xem Phần 1 trước Phần 2.
Nội dung giữ nguyên độ phân giải, chỉ cắt khoảng trắng và thay chân trang lặp
bằng nhãn số trang gốc. manifest.json đối chiếu từng trang và kết quả kiểm tra pixel.
Bản PDF đầy đủ vẫn ở ../Le_Son_Truong_Buoi2.pdf nếu cần cách đọc theo trang.

Các phần cần tài khoản cá nhân (điểm Kaggle), số đo GPU và upload Drive
vẫn theo ghi chú trong bài. Bộ ảnh gộp không thêm kết quả chưa thực hiện.
"""
    (OUTPUT / "HUONG_DAN.txt").write_text(readme, encoding="utf-8-sig")
    target = BASE.parent / "Le_Son_Truong_Buoi2_Anh_Gop.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.iterdir()):
            if path.name in {o["file"] for o in outputs} | {"manifest.json", "HUONG_DAN.txt"}:
                archive.write(path, arcname=Path("Le_Son_Truong_Buoi2_Anh_Gop") / path.name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len([p for p in archive.namelist() if p.endswith(".png")]) == len(outputs)
    print(f"Verified {len(paths)} source pages, {len(outputs)} images; ZIP {target.stat().st_size / 1024**2:.1f} MiB.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
