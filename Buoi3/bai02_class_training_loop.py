"""Thực hành 1 — training loop và 3 learning rate, theo slide trang 26.

6000 train, 1000 validation; MLP 784-256-10; Adam; batch64; đúng 5 epoch.
Chạy: python Buoi3/bai02_class_training_loop.py
"""
# %% [markdown]
# # Thực hành 1 — Xây dựng Training Loop
# **Lê Sơn Trường**. Nguồn: `baitap.txt`, slide trang 26 và mã mẫu trong `ma_nguon_goc/`.
# Giữ nguyên 5 epoch, 6000/1000 mẫu, batch64, MLP 784 → 256 → 10, ReLU và Adam.
# Chỉ thay learning rate: **0.01, 0.001, 0.0001** theo yêu cầu của slide.
# Dữ liệu thật, pixel /255 giống ToTensor; validation lấy từ test theo mã mẫu.
# Không có tập test độc lập sau khi dùng validation để so sánh.

# %%
from pathlib import Path
import hashlib
from time import perf_counter

import pandas as pd
import torch
from torch import nn

from common import (BASE, Config, EarlyStopping, build_model, build_optimizer, device,
                    evaluate_details, make_loaders, save_experiment, seed_everything,
                    state_digest, synchronize)
from data import load_subset

CONFIGS = [Config(f"adam_lr_{lr:g}", "adam", lr, 256) for lr in [.01, .001, .0001]]
EPOCHS = 5

# %% [markdown]
# ## Vòng lặp học và đánh giá
# `zero_grad` xóa gradient cũ; forward tạo logits; CE tính loss;
# `backward` tính gradient; `step` mới cập nhật trọng số.
# `model.eval()` và `torch.no_grad()` được dùng khi đánh giá trong common.py.
# `online_train_loss` lấy trung bình có trọng số theo số mẫu trong từng batch.
# `train_loss` được đo lại ở cuối epoch để so sánh cùng cách với validation.
# Đoạn Early Stopping chỉ bật ở phần nâng cao của bài 2; bài 1 học đủ 5 epoch.

# %%
def train_configuration(config, train_set, val_set, batch_size, epochs=5, *, early_stopping=False):
    seed_everything(42)
    dev = device()
    model = build_model(config.hidden).to(dev)
    initial_hash = state_digest(model)
    train_loader, train_eval_loader, val_loader = make_loaders(train_set, val_set, batch_size)
    loss_fn = nn.CrossEntropyLoss()  # Nhận logits, nhãn long; không softmax trước CE.
    optimizer = build_optimizer(model, config)
    stopper = EarlyStopping(patience=2, min_delta=0.0) if early_stopping else None
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{config.name} | device={dev} | parameters={parameters:,} | "
          f"epochs={epochs} | batch={batch_size} | LR={config.lr:g}", flush=True)
    history, training_seconds, batch_hash = [], 0.0, None
    synchronize(dev)
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        synchronize(dev)
        epoch_start = perf_counter()
        for xb, yb in train_loader:
            if batch_hash is None:
                batch_hash = hashlib.sha256(xb.numpy().tobytes() + yb.numpy().tobytes()).hexdigest()
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()           # 1. Xóa gradient cũ
            logits = model(xb)              # 2. Forward
            loss = loss_fn(logits, yb)       # 3. Tính loss
            loss.backward()                 # 4. Backpropagation
            optimizer.step()                # 5. Cập nhật tham số
            loss_sum += loss.item() * len(yb)
            seen += len(yb)
        synchronize(dev)
        training_seconds += perf_counter() - epoch_start

        # Đánh giá không cập nhật trọng số, không tính gradient.
        train_metrics = evaluate_details(model, train_eval_loader, loss_fn)
        val_metrics = evaluate_details(model, val_loader, loss_fn)
        history.append({"config": config.name, "epoch": epoch, "device": str(dev),
                        "parameters": parameters, "lr": config.lr, "online_train_loss": loss_sum / seen,
                        "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"]})
        print(f"Epoch {epoch}/{epochs} | config={config.name} | device={dev} | parameters={parameters:,} | "
              f"train_loss={train_metrics['loss']:.4f} | "
              f"online_train_loss={loss_sum / seen:.4f} | val_loss={val_metrics['loss']:.4f} | "
              f"val_accuracy={val_metrics['accuracy']:.2%}", flush=True)
        if stopper is not None and stopper.step(val_metrics["loss"], model, epoch):
            print(f"Early Stopping ở epoch {epoch}; best epoch={stopper.best_epoch}.", flush=True)
            break
    synchronize(dev)
    total_seconds = perf_counter() - start
    epochs_run = len(history)
    selected_epoch = epochs_run
    if stopper is not None:
        stopper.restore(model)
        selected_epoch = stopper.best_epoch
        train_metrics = evaluate_details(model, train_eval_loader, loss_fn)
        val_metrics = evaluate_details(model, val_loader, loss_fn)
    best_row = min(history, key=lambda row: row["val_loss"])
    summary = {"config": config.name, "optimizer": config.optimizer, "lr": config.lr, "device": str(dev),
               "momentum": config.momentum, "hidden": config.hidden,
               "epochs_requested": epochs, "epochs_run": epochs_run, "selected_epoch": selected_epoch,
               "early_stopping": early_stopping, "stopped_early": epochs_run < epochs,
               "best_epoch": best_row["epoch"], "best_val_loss": best_row["val_loss"],
               "train_samples": len(train_set), "val_samples": len(val_set), "batch_size": batch_size,
               "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
               "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"],
               "val_macro_f1": val_metrics["macro_f1"], "parameters": parameters,
               "training_seconds": training_seconds, "total_seconds": total_seconds}
    predictions = pd.DataFrame({"config": config.name, "sample_index": range(len(val_set)),
                                "target": val_metrics["targets"], "prediction": val_metrics["predictions"],
                                "confidence": val_metrics["confidences"]})
    checkpoint = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "hidden": config.hidden, "selected_epoch": selected_epoch,
                  "config": config.name, "val_loss": val_metrics["loss"]}
    return {"summary": summary, "history": history, "predictions": predictions,
            "checkpoint": checkpoint, "initial_hash": initial_hash, "batch_hash": batch_hash}

# %% [markdown]
# ## Chạy ba learning rate
# Mỗi cấu hình được reset seed42, cùng trọng số ban đầu và cùng thứ tự batch.
# Kết quả cơ bản lấy epoch 5; không lấy epoch đẹp nhất để thay kết quả cuối.

# %%
def main():
    train_set, val_set = load_subset(6000, 1000)
    runs = [train_configuration(c, train_set, val_set, batch_size=64, epochs=EPOCHS) for c in CONFIGS]
    return save_experiment(runs, BASE / "results" / "bai1_learning_rate", val_set,
                           "Thực hành 1 — So sánh learning rate (Adam, 5 epoch)", CONFIGS)


if __name__ == "__main__":
    main()
