"""Tải các dữ liệu thật mà Exercises chương 1–5 sử dụng."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import urlopen
import zipfile

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def download(name, urls):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    if path.is_file() and path.stat().st_size:
        return path
    errors = []
    for url in urls:
        try:
            with urlopen(url, timeout=90) as response:
                content = response.read()
            path.write_bytes(content)
            return path
        except Exception as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError("\n".join(errors))


def prepare():
    jobs = []
    for name in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        jobs.append((name, [f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{name}"]))
    jobs.append(("abalone.zip", ["https://archive.ics.uci.edu/static/public/1/abalone.zip"]))
    for split in ["train", "test"]:
        name = f"kaggle_house_pred_{split}.csv"
        jobs.append((name, [f"https://d2l-data.s3-accelerate.amazonaws.com/{name}",
                            f"https://d2l-data.s3.amazonaws.com/{name}"]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda job: download(*job), jobs))
    manifest = [{"file": p.name, "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "sources": urls}
                for p, (_, urls) in zip(paths, jobs)]
    (BASE / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Downloaded/verified:", ", ".join(p.name for p in paths), flush=True)


def fashion():
    import numpy as np
    import torch
    arrays = []
    for split in ["train", "t10k"]:
        with gzip.open(DATA / f"{split}-images-idx3-ubyte.gz", "rb") as stream:
            magic, count, height, width = struct.unpack(">IIII", stream.read(16))
            assert magic == 2051 and (height, width) == (28, 28)
            images = np.frombuffer(stream.read(), dtype=np.uint8).copy().reshape(count, 784)
        with gzip.open(DATA / f"{split}-labels-idx1-ubyte.gz", "rb") as stream:
            magic, n_labels = struct.unpack(">II", stream.read(8))
            assert magic == 2049 and n_labels == count
            labels = np.frombuffer(stream.read(), dtype=np.uint8).copy()
        arrays.extend([torch.from_numpy(images).float() / 255, torch.from_numpy(labels).long()])
    return tuple(arrays)


def abalone():
    import pandas as pd
    with zipfile.ZipFile(DATA / "abalone.zip") as archive:
        with archive.open("abalone.data") as stream:
            return pd.read_csv(stream, header=None, names=[
                "Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight",
                "VisceraWeight", "ShellWeight", "Rings",
            ])


if __name__ == "__main__":
    prepare()
