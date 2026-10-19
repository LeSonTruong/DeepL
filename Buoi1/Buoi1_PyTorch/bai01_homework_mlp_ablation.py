import itertools
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.datasets import make_moons
from Buoi1.Buoi1_PyTorch.common import Config, seed_everything, run_experiments


def main():
    seed_everything(42)
    X, y = make_moons(n_samples=1200, noise=0.28, random_state=42)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    tr, va = random_split(ds, [960, 240], generator=torch.Generator().manual_seed(42))
    configs = [
        Config(f"width{hidden}_lr{lr:g}", hidden=(hidden, hidden), lr=lr)
        for hidden, lr in itertools.product([8, 32, 64, 128], [1e-2, 1e-3])
    ]
    # Chỉ thay một yếu tố so với baseline: 2 tầng x 32, ReLU, LR=0.001.
    configs += [
        Config("depth1", hidden=(32,)),
        Config("depth3", hidden=(32, 32, 32)),
        Config("tanh", activation="tanh"),
        Config("sigmoid", activation="sigmoid"),
        Config("dropout", dropout=0.2),
        Config("weight_decay", weight_decay=1e-4),
    ]
    return run_experiments(
        configs, tr, va, epochs=15, val_batch_size=256,
        output_dir=Path(__file__).resolve().parent / "results" / "bai2",
        dataset="make_moons, n_samples=1200, noise=0.28, random_state=42",
    )


if __name__ == "__main__":
    main()
