"""Fashion-MNIST gốc; đọc IDX giống ToTensor: float32, [0,1], (1,28,28)."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import TensorDataset

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "fashion"
FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def prepare_data():
    DATA.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in FILES.items():
        target = DATA / name
        reused = False
        if not target.is_file():
            cached = BASE.parent / "Buoi2" / "bai_lam_Le_Son_Truong" / "data" / name
            if cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest() == expected:
                shutil.copyfile(cached, target)
                reused = True
            else:
                with urlopen(URL + name, timeout=90) as response:
                    content = response.read()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Sai SHA-256: {name}")
                target.write_bytes(content)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Dữ liệu bị thay đổi: {target}")
        records.append({"file": name, "sha256": actual, "bytes": target.stat().st_size,
                        "source": URL + name})
        if reused:
            print(f"Dùng lại Fashion-MNIST thật từ Buổi 2: {name}", flush=True)
    (BASE / "data_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_subset(train_size, val_size):
    """Đúng range(N) trên train/test chính thức như hai script mẫu."""
    prepare_data()
    datasets = []
    for split, limit in [("train", train_size), ("t10k", val_size)]:
        with gzip.open(DATA / f"{split}-images-idx3-ubyte.gz", "rb") as stream:
            magic, count, height, width = struct.unpack(">IIII", stream.read(16))
            assert magic == 2051 and (height, width) == (28, 28) and 0 < limit <= count
            raw = stream.read(limit * height * width)
            images = np.frombuffer(raw, dtype=np.uint8).copy().reshape(limit, 1, 28, 28)
        with gzip.open(DATA / f"{split}-labels-idx1-ubyte.gz", "rb") as stream:
            magic, n_labels = struct.unpack(">II", stream.read(8))
            assert magic == 2049 and n_labels == count
            labels = np.frombuffer(stream.read(limit), dtype=np.uint8).copy()
        assert images.size == limit * 784 and len(labels) == limit
        datasets.append(TensorDataset(torch.from_numpy(images).float().div_(255),
                                      torch.from_numpy(labels).long()))
    return tuple(datasets)


if __name__ == "__main__":
    prepare_data()
