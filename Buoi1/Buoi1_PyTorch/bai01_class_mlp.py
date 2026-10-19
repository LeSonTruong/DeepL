"""Bài 1: giữ dữ liệu và 20 epoch của mã trên lớp, bổ sung so sánh độ rộng."""
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.datasets import make_moons
from Buoi1.Buoi1_PyTorch.common import Config, seed_everything, run_experiments


def main():
    seed_everything(42)
    X, y = make_moons(n_samples=800, noise=0.25, random_state=42)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    tr, va = random_split(ds, [640, 160], generator=torch.Generator().manual_seed(42))
    configs = [Config(f"width{hidden}", hidden=(hidden, hidden)) for hidden in [8, 32, 64, 128]]
    return run_experiments(
        configs, tr, va, epochs=20, val_batch_size=128,
        output_dir=Path(__file__).resolve().parent / "results" / "bai1",
        dataset="make_moons, n_samples=800, noise=0.25, random_state=42",
    )


if __name__ == "__main__":
    main()
