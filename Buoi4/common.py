"""Công cụ chung cho hai bài CNN; FashionMNIST thật, không cần torchvision."""
import csv
import gzip
import hashlib
import json
import random
import struct
import time
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent
SEED, EPOCHS, BATCH_SIZE = 42, 5, 64
LR, TRAIN_SIZE, VAL_SIZE = 1e-3, 6000, 1200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
           "Shirt", "Sneaker", "Bag", "Ankle boot"]
FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def read_official(train=True):
    """Đọc IDX đã kiểm SHA-256; /255 giống ToTensor, không augmentation."""
    prefix = "train" if train else "t10k"
    blobs = []
    for kind in ("images-idx3", "labels-idx1"):
        name = f"{prefix}-{kind}-ubyte.gz"
        local = BASE / "data" / name
        candidates = [local, BASE.parent / "Buoi2" / "bai_lam_Le_Son_Truong" / "data" / name,
                      BASE.parent / "Buoi3" / "data" / "fashion" / name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            local.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(URL + name, timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != FILES[name]:
                raise ValueError(f"Sai SHA-256: {name}")
            local.write_bytes(content)
            path = local
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != FILES[name]:
            raise ValueError(f"Sai SHA-256: {path}")
        blobs.append(gzip.decompress(content))
    magic, count, height, width = struct.unpack(">IIII", blobs[0][:16])
    label_magic, label_count = struct.unpack(">II", blobs[1][:8])
    assert magic == 2051 and label_magic == 2049 and count == label_count
    assert count == (60000 if train else 10000) and (height, width) == (28, 28)
    images = torch.from_numpy(np.frombuffer(blobs[0], dtype=np.uint8, offset=16).copy())
    labels = torch.from_numpy(np.frombuffer(blobs[1], dtype=np.uint8, offset=8).copy()).long()
    return images.reshape(count, 1, 28, 28), labels


def make_data(output):
    images, labels = read_official(train=True)
    ids = torch.randperm(len(labels), generator=torch.Generator().manual_seed(SEED))
    train_ids, val_ids = ids[:TRAIN_SIZE], ids[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    assert not set(train_ids.tolist()) & set(val_ids.tolist())
    save_json(output / "split.json", {"seed": SEED, "source": "official train",
                                     "train_indices": train_ids.tolist(),
                                     "validation_indices": val_ids.tolist()})
    return tuple(TensorDataset(images[i].float().div_(255), labels[i])
                 for i in (train_ids, val_ids))


def make_loaders(train_ds, val_ds):
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                       generator=torch.Generator().manual_seed(SEED), num_workers=0),
            DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0))


class CNN2Blocks(nn.Module):
    """Hai block trích đặc trưng rồi phân loại; N là số ảnh trong batch.

    Baseline: channels=16, kernel_size=3, không BatchNorm/Dropout.
    Mỗi block: Conv2d -> ReLU -> MaxPool2d(2).
    Biến thể BatchNorm chèn BN giữa Conv2d và ReLU.
    """

    def __init__(self, channels=16, kernel_size=3, batchnorm=False, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels) if batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, 2 * channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(2 * channels) if batchnorm else nn.Identity()
        self.relu, self.pool = nn.ReLU(), nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * channels * 7 * 7, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # BLOCK 1: (N,1,28,28) -> (N,C,14,14), baseline C=16.
        x = self.conv1(x)  # Học C bộ lọc: (N,C,28,28); padding giữ H,W.
        x = self.bn1(x)    # Identity ở baseline; BatchNorm ở biến thể BN.
        x = self.relu(x)   # max(0,x): bỏ giá trị âm, giữ nguyên shape.
        x = self.pool(x)   # Lấy max mỗi ô 2x2: 28x28 -> 14x14.

        # BLOCK 2: (N,C,14,14) -> (N,2C,7,7), baseline 2C=32.
        x = self.conv2(x)  # Mỗi bộ lọc kết hợp C kênh: (N,2C,14,14).
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)   # 14x14 -> 7x7; không thay số kênh.

        # PHÂN LOẠI: (N,2C,7,7) -> (N,2C*7*7) -> (N,128) -> (N,10).
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)   # p=0 ở baseline; p=0.3 ở biến thể Dropout.
        return self.fc2(x)  # 10 logits; CrossEntropyLoss nhận trực tiếp.


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


def describe_cnn(model, batch_sizes=(1, 64)):
    """Ghi shape từ forward thật với input giả; giữ nguyên trạng thái model."""
    parameters = dict(model.named_parameters())
    count = lambda prefixes: sum(p.numel() for name, p in parameters.items()
                                 if name.split(".")[0] in prefixes and p.requires_grad)
    result = dict(parameters=sum(p.numel() for p in parameters.values() if p.requires_grad),
                  block1_parameters=count({"conv1", "bn1"}),
                  block2_parameters=count({"conv2", "bn2"}),
                  classifier_parameters=count({"fc1", "fc2"}), traces=[])
    modes = {module: module.training for module in model.modules()}
    handles, records, calls = [], [], {}

    def hook(name):
        def record(module, inputs, output):
            calls[name] = calls.get(name, 0) + 1
            step = f"{name}{calls[name]}" if name in {"relu", "pool"} else name
            records.append(dict(step=step, layer=str(module),
                                input_shape=list(inputs[0].shape), output_shape=list(output.shape)))
        return record

    try:
        for name, module in model.named_modules():
            if name and not list(module.children()):
                handles.append(module.register_forward_hook(hook(name)))
        model.eval()
        parameter = next(model.parameters())
        with torch.inference_mode():
            for size in batch_sizes:
                records, calls = [], {}
                model(torch.zeros(size, 1, 28, 28, device=parameter.device, dtype=parameter.dtype))
                result["traces"].append(dict(batch_size=size, layers=records))
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes.items():
            module.training = training
    return result


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        count += y.size(0)
    return total_loss / count


@torch.inference_mode()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        count += y.size(0)
    return total_loss / count, correct / count


def fit(name, model, train_ds, val_ds, output):
    model = model.to(DEVICE)
    train_loader, val_loader = make_loaders(train_ds, val_ds)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{name} | device={DEVICE} | parameters={parameters}", flush=True)
    history, best_key, best_row = [], None, None
    for epoch in range(1, EPOCHS + 1):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = dict(model=name, epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   val_accuracy=val_acc, seconds=elapsed)
        history.append(row)
        print(f"Epoch {epoch}/{EPOCHS}: device={DEVICE}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_acc:.2%}, time={elapsed:.2f}s", flush=True)
        key = (val_acc, -val_loss)
        if best_key is None or key > best_key:
            best_key, best_row = key, row.copy()
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_accuracy": val_acc, "val_loss": val_loss}, output / f"{name}_best.pt")
    summary = dict(model=name, parameters=parameters, final_train_loss=history[-1]["train_loss"],
                   final_val_loss=history[-1]["val_loss"], final_val_accuracy=history[-1]["val_accuracy"],
                   mean_seconds_per_epoch=float(np.mean([r["seconds"] for r in history])),
                   best_epoch=best_row["epoch"], best_val_loss=best_row["val_loss"],
                   best_val_accuracy=best_row["val_accuracy"])
    return summary, history


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata(output):
    save_json(output / "metadata.json", dict(dataset="FashionMNIST", seed=SEED,
              epochs=EPOCHS, batch_size=BATCH_SIZE, optimizer="Adam", lr=LR,
              train_size=TRAIN_SIZE, validation_size=VAL_SIZE, device=str(DEVICE),
              torch_version=str(torch.__version__), threads=torch.get_num_threads(),
              preprocessing="float32 /255; no augmentation", official_data_sha256=FILES,
              train_loss="sample-weighted online training loss",
              timing="training + validation; excludes checkpoint and plot writing"))


def learning_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in dict.fromkeys(r["model"] for r in history):
        rows = [r for r in history if r["model"] == name]
        for ax, field in zip(axes, ("train_loss", "val_loss", "val_accuracy")):
            ax.plot([r["epoch"] for r in rows], [r[field] for r in rows], marker="o", label=name)
    for ax, title in zip(axes, ("Train loss (during updates)", "Validation loss", "Validation accuracy")):
        ax.set(title=title, xlabel="Epoch", xticks=range(1, EPOCHS + 1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def result_table(rows, best=False):
    prefix = "best" if best else "final"
    lines = ["| Mô hình | Tham số | Val loss | Val accuracy | Giây/epoch | Epoch báo cáo |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['parameters']:,} | {r[prefix + '_val_loss']:.4f} | "
                     f"{r[prefix + '_val_accuracy']:.2%} | {r['mean_seconds_per_epoch']:.3f} | "
                     f"{r['best_epoch'] if best else EPOCHS} |")
    return "\n".join(lines)
