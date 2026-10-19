"""Tiện ích PyTorch dùng chung, theo cách huấn luyện trong mã mẫu."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Config:
    name: str
    hidden: tuple = (32, 32)
    activation: str = "relu"
    lr: float = 1e-3
    dropout: float = 0.0
    weight_decay: float = 0.0


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def build_mlp(hidden=(32, 32), activation="relu", dropout=0.0):
    """2 đầu vào -> các tầng ẩn -> 2 logits, tương tự nn.Sequential mã gốc."""
    activation_type = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation]
    layers = []
    n_inputs = 2
    for width in hidden:
        layers.extend([nn.Linear(n_inputs, width), activation_type()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        n_inputs = width
    layers.append(nn.Linear(n_inputs, 2))
    return nn.Sequential(*layers)


def evaluate(model, loader):
    """Đo trên toàn tập, tắt Dropout; loss không cộng hạng phạt trọng số."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets, predictions = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            total_loss += criterion(logits, y).item()
            targets.extend(y.tolist())
            predictions.extend(logits.argmax(dim=1).tolist())
    return {
        "loss": total_loss / len(targets),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def train_classifier(model, train_loader, val_loader, epochs=20, lr=1e-3, weight_decay=0.0):
    """Huấn luyện đủ số epoch và báo cáo mô hình cuối, không chọn checkpoint."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=128, shuffle=False)
    history = []
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        train_metrics = evaluate(model, train_eval_loader)
        val_metrics = evaluate(model, val_loader)
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        })
    elapsed = perf_counter() - start
    return {
        **history[-1],
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "seconds": elapsed,
    }, history


def run_experiments(configs, train_set, val_set, *, epochs, val_batch_size, output_dir, dataset):
    rows, histories, models = [], [], {}
    for config in configs:
        # Cùng seed khởi tạo và seed xáo trộn dữ liệu cho mọi cấu hình.
        seed_everything(42)
        model = build_mlp(config.hidden, config.activation, config.dropout)
        train_loader = DataLoader(
            train_set, batch_size=32, shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False)
        metrics, history = train_classifier(
            model, train_loader, val_loader, epochs=epochs,
            lr=config.lr, weight_decay=config.weight_decay,
        )
        models[config.name] = model
        rows.append({
            **asdict(config), "hidden": "x".join(map(str, config.hidden)), **metrics,
        })
        histories.extend({"config": config.name, **record} for record in history)
        print(
            f"{config.name:20} epochs={epochs} loss={metrics['val_loss']:.4f} "
            f"accuracy={metrics['val_accuracy']:.4f} macro_f1={metrics['val_macro_f1']:.4f} "
            f"params={metrics['parameters']} time={metrics['seconds']:.2f}s", flush=True,
        )
    return save_results(
        configs, train_set, val_set, rows, histories, epochs=epochs,
        val_batch_size=val_batch_size, output_dir=output_dir, dataset=dataset,
        models=models,
    )


def save_results(configs, train_set, val_set, rows, histories, *, epochs, val_batch_size, output_dir, dataset, models=None):
    results, history = pd.DataFrame(rows), pd.DataFrame(histories)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results.csv", index=False, encoding="utf-8-sig")
    history.to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
    save_plots(results, history, output_dir)
    if models is not None and set(results.name) == {f"width{width}" for width in (8, 32, 64, 128)}:
        X, y = train_set.dataset.tensors
        save_decision_boundaries(models, X, y, val_set.indices, results, output_dir)
    write_report(results, history, output_dir, dataset, len(train_set), len(val_set), epochs)
    metadata = {
        "dataset": dataset, "train_samples": len(train_set), "validation_samples": len(val_set),
        "epochs": epochs, "seed": 42, "split": "random_split 80/20, generator seed=42",
        "scaling": "none (same as original code)", "train_batch_size": 32,
        "val_batch_size": val_batch_size, "optimizer": "torch.optim.Adam",
        "criterion": "torch.nn.CrossEntropyLoss", "device": "cpu", "torch_threads": 1,
        "evaluation": "final epoch; validation only, no separate test set",
        "configs": [asdict(config) for config in configs],
        "torch_version": torch.__version__,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, history


def save_plots(results, history, output_dir):
    if set(results.name) == {f"width{width}" for width in (8, 32, 64, 128)}:
        save_width_comparison(results, history, output_dir)
    fields = [
        ("val_loss", "Validation cross-entropy"), ("val_accuracy", "Validation accuracy"),
        ("val_macro_f1", "Validation Macro-F1"), ("parameters", "Trainable parameters"),
        ("seconds", "Training + evaluation (s)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (column, title) in zip(axes.flat, fields):
        ax.bar(results.name, results[column], color="steelblue")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=75, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        if column in ("val_accuracy", "val_macro_f1"):
            ax.set_ylim(0, 1.05)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=130)
    plt.close(fig)
    columns = 2 if len(results) <= 4 else 3
    rows = int(np.ceil(len(results) / columns))
    for metric in ("loss", "accuracy"):
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.5, rows * 3), squeeze=False)
        for ax, name in zip(axes.flat, results.name):
            trace = history[history.config == name]
            ax.plot(trace.epoch, trace[f"train_{metric}"], label="train")
            ax.plot(trace.epoch, trace[f"val_{metric}"], label="validation")
            ax.set(title=name, xlabel="Epoch", ylabel=metric)
            if metric == "accuracy":
                ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.legend()
        for ax in list(axes.flat)[len(results):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(output_dir / f"learning_{metric}.png", dpi=130)
        plt.close(fig)


def save_width_comparison(results, history, output_dir):
    """Bài 1: so sánh trực tiếp bốn độ rộng, cùng điều kiện huấn luyện."""
    widths = [8, 32, 64, 128]
    ordered = results.set_index("name").loc[[f"width{width}" for width in widths]]
    positions = np.arange(len(widths))
    colors = ["#4477AA", "#EE9933", "#228833", "#AA3377"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric, title in zip(
        axes[0], ["loss", "accuracy", "macro_f1"],
        ["Loss (thấp hơn tốt hơn)", "Accuracy (cao hơn tốt hơn)", "Macro-F1 (cao hơn tốt hơn)"],
    ):
        for offset, split, color, label in [
            (-0.19, "train", "#4477AA", "Train"),
            (0.19, "val", "#EE9933", "Validation"),
        ]:
            bars = ax.bar(positions + offset, ordered[f"{split}_{metric}"], width=0.36, color=color, label=label)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_title(title)
        ax.set_xticks(positions, [str(width) for width in widths])
        ax.set_xlabel("Số neuron mỗi tầng ẩn")
        ax.set_ylim(0, 1.12 if metric != "loss" else ordered[["train_loss", "val_loss"]].to_numpy().max() * 1.25)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    for ax, field, title, fmt in [
        (axes[1, 0], "parameters", "Số tham số", "%.0f"),
        (axes[1, 1], "seconds", "Thời gian huấn luyện + đánh giá (giây)", "%.3f"),
    ]:
        bars = ax.bar(positions, ordered[field], color=colors)
        ax.bar_label(bars, fmt=fmt, padding=3)
        ax.set_xticks(positions, [str(width) for width in widths])
        ax.set(title=title, xlabel="Số neuron mỗi tầng ẩn")
        ax.set_ylim(0, ordered[field].max() * 1.2)
        ax.grid(axis="y", alpha=0.2)
    ax = axes[1, 2]
    for width, color in zip(widths, colors):
        trace = history[history.config == f"width{width}"]
        ax.plot(trace.epoch, trace.val_loss, color=color, label=f"{width} neuron", linewidth=2)
    ax.set(title="Validation loss qua các epoch", xlabel="Epoch", ylabel="Cross-entropy")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.suptitle(
        f"Bài 1: so sánh 8, 32, 64, 128 neuron mỗi tầng ẩn\n"
        f"Hai tầng ẩn, ReLU, LR=0.001, {int(ordered.epoch.iloc[0])} epoch; các cột lấy kết quả epoch cuối",
        fontsize=14,
    )
    fig.tight_layout()
    path = Path(output_dir) / "comparison_neurons.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_decision_boundaries(models, X, y, val_indices, results, output_dir):
    """Vẽ cùng các điểm thật và ranh giới học được của bốn mô hình Bài 1."""
    X = torch.as_tensor(X, dtype=torch.float32).cpu()
    y = torch.as_tensor(y).cpu().numpy()
    points = X.numpy()
    x_limits = (points[:, 0].min() - 0.3, points[:, 0].max() + 0.3)
    y_limits = (points[:, 1].min() - 0.3, points[:, 1].max() + 0.3)
    xx, yy = np.meshgrid(np.linspace(*x_limits, 250), np.linspace(*y_limits, 250))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    lookup = results.set_index("name")
    for ax, width in zip(axes.flat, (8, 32, 64, 128)):
        name = f"width{width}"
        model = models[name]
        model.eval()
        with torch.no_grad():
            probability = model(grid).softmax(dim=1)[:, 1].numpy().reshape(xx.shape)
            predictions = model(X).argmax(dim=1).numpy()
        ax.contourf(xx, yy, probability, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.14)
        ax.contour(xx, yy, probability, levels=[0.5], colors="black", linewidths=1.5)
        ax.scatter(points[:, 0], points[:, 1], c=y, cmap="RdBu", vmin=0, vmax=1, s=14, alpha=0.7)
        indices = np.asarray(val_indices)
        wrong = indices[predictions[indices] != y[indices]]
        ax.scatter(points[wrong, 0], points[wrong, 1], marker="x", color="black", s=40, linewidths=1.2)
        ax.set(
            title=f"{width} neuron mỗi tầng | Validation accuracy: {lookup.loc[name, 'val_accuracy']:.2%}",
            xlabel="x1", ylabel="x2", xlim=x_limits, ylim=y_limits,
        )
    fig.suptitle(
        f"make_moons: so sánh ranh giới phân loại của 8, 32, 64, 128 neuron\n"
        f"Hai tầng ẩn, ReLU, {int(results.epoch.iloc[0])} epoch; cùng dữ liệu và seed 42",
        fontsize=13,
    )
    fig.text(
        0.5, 0.015,
        "Chấm đỏ/xanh: nhãn thật của 800 điểm | Vùng nền: nhóm mô hình dự đoán | "
        "Đường đen: ranh giới | Dấu ×: điểm validation dự đoán sai",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = Path(output_dir) / "decision_boundaries.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(results, history, output_dir, dataset, train_size, val_size, epochs):
    lines = [
        "# Kết quả thí nghiệm MLP PyTorch", "", f"Dữ liệu: {dataset}.", "",
        f"Train/validation = {train_size}/{val_size}; seed = 42; {epochs} epoch; "
        "batch train = 32; Adam; CrossEntropyLoss. Bảng lấy mô hình ở epoch cuối.", "",
        "Loss, accuracy và Macro-F1 trong bảng được đo trên validation. Train và validation "
        "dùng cùng cách chia random_split như mã gốc, không chuẩn hóa dữ liệu. "
        "Không có test riêng, vì vậy đây là kết quả khảo sát trên validation, "
        "không phải ước lượng cuối trên tập test độc lập.", "",
        "Thời gian tính trên CPU gồm học và đo train/validation mỗi epoch; "
        "không gồm khởi tạo optimizer, xuất file và vẽ hình. Số tham số gồm trọng số và bias. "
        "Loss đo ở chế độ eval, không cộng phạt weight decay; Dropout được tắt lúc đánh giá.", "",
        "## Bảng so sánh", "",
        "| Cấu hình | Tầng ẩn | Activation | LR | Dropout | Weight decay | Loss | Accuracy | Macro-F1 | Tham số | Giây |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples():
        lines.append(
            f"| {row.name} | {row.hidden} | {row.activation} | {row.lr:g} | {row.dropout:g} | "
            f"{row.weight_decay:g} | {row.val_loss:.4f} | {row.val_accuracy:.4f} | "
            f"{row.val_macro_f1:.4f} | {row.parameters} | {row.seconds:.2f} |"
        )
    best = results.loc[results.val_loss.idxmin()]
    lines += [
        "", "## Nhận xét ngắn", "",
        f"- Cấu hình có validation loss thấp nhất là **{best['name']}** "
        f"({best.val_loss:.4f}), accuracy {best.val_accuracy:.2%}, Macro-F1 {best.val_macro_f1:.4f}.",
    ]
    if len(results) == 4:
        small, large = results.iloc[0], results.iloc[-1]
        lines.append(
            f"- Tăng 8 lên 128 neuron mỗi tầng làm tham số tăng từ {small.parameters} "
            f"lên {large.parameters}; accuracy thay đổi từ {small.val_accuracy:.2%} "
            f"thành {large.val_accuracy:.2%}. Mạng rộng có nhiều khả năng biểu diễn hơn "
            "nhưng cần nhiều phép tính; không bảo đảm accuracy tăng đều."
        )
    else:
        for names, label in [
            (["depth1", "width32_lr0.001", "depth3"], "Số tầng ẩn 1/2/3"),
            ([f"width{w}_lr0.001" for w in (8, 32, 64, 128)], "Số neuron tại LR=0.001"),
            (["width32_lr0.001", "tanh", "sigmoid"], "Activation"),
            (["width32_lr0.001", "dropout", "weight_decay"], "Regularization"),
        ]:
            group = results[results.name.isin(names)]
            winner = group.loc[group.val_loss.idxmin()]
            lines.append(f"- {label}: **{winner['name']}** có validation loss thấp nhất nhóm ({winner.val_loss:.4f}).")
        low_lr = results[results.name.str.endswith("lr0.001")].val_loss.mean()
        high_lr = results[results.name.str.endswith("lr0.01")].val_loss.mean()
        lines.append(
            f"- Khảo sát LR theo mã BTVN gốc: loss trung bình trên bốn độ rộng "
            f"là {high_lr:.4f} với LR=0.01 và {low_lr:.4f} với LR=0.001. "
            "LR quyết định độ lớn bước cập nhật; kết luận phụ thuộc ngân sách 15 epoch."
        )
    overfit = []
    for name, trace in history.groupby("config", sort=False):
        minimum = trace.loc[trace.val_loss.idxmin()]
        last = trace.iloc[-1]
        if last.val_loss > minimum.val_loss + 0.02 and last.train_loss < minimum.train_loss - 0.005:
            overfit.append(name)
    lines.append(
        "- Dấu hiệu overfitting ở **" + ", ".join(overfit) +
        "**: sau epoch có val loss thấp nhất, train loss giảm thêm >0.005 nhưng val loss tăng >0.02."
        if overfit else
        "- Chưa thấy overfitting rõ theo tiêu chí train loss giảm thêm >0.005 nhưng val loss "
        "tăng >0.02 sau epoch có val loss thấp nhất; không nên ép kết luận overfitting khi đường học chưa thể hiện."
    )
    weak = results[(results.train_accuracy < 0.9) & (results.val_accuracy < 0.9)]
    lines.append(
        "- Cả train/validation accuracy cuối đều dưới 90% ở **" + ", ".join(weak.name) +
        "**: có dấu hiệu học chưa đủ tốt. Đây có thể là underfitting hoặc chưa hội tụ; "
        "nếu loss vẫn giảm thì 15/20 epoch có thể chưa đủ để kết luận thiếu năng lực mô hình."
        if not weak.empty else
        "- Không có cấu hình có cả train/validation accuracy cuối dưới 90%; chưa có dấu hiệu "
        "underfitting rõ theo ngưỡng tham khảo này."
    )
    lines += [
        "- Chỉ chạy seed 42 để sát mã mẫu. Các ngưỡng chẩn đoán là tham khảo; "
        "cần đối chiếu learning curve. Không suy rộng kết quả cho mọi dữ liệu hoặc số epoch.", "",
        "## Biểu đồ", "", "![So sánh](comparison.png)", "",
        "![Learning loss](learning_loss.png)", "", "![Learning accuracy](learning_accuracy.png)", "",
    ]
    if set(results.name) == {f"width{width}" for width in (8, 32, 64, 128)}:
        lines += ["### So sánh trực tiếp số neuron", "", "![So sánh 8, 32, 64, 128 neuron](comparison_neurons.png)", ""]
        if (output_dir / "decision_boundaries.png").is_file():
            lines += ["### Các điểm và ranh giới phân loại", "", "![Ranh giới theo số neuron](decision_boundaries.png)", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
