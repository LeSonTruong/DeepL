"""Tiện ích huấn luyện, đánh giá và xuất sản phẩm cho hai bài MLP.

MLPClassifier.partial_fit cập nhật đúng một epoch mỗi lần gọi. Các đường
learning curve là cross-entropy trên toàn bộ train/validation ở chế độ
đánh giá, không cộng hạng phạt L2, nên có thể so sánh trực tiếp.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Config:
    name: str
    hidden: tuple[int, ...]
    activation: str = "relu"
    alpha: float = 0.0


@dataclass
class Data:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def prepare_data(X, y):
    """Chia phân tầng 60/20/20; chỉ fit scaler trên tập train."""
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=42, stratify=y_holdout,
    )
    scaler = StandardScaler().fit(X_train)
    return Data(
        scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test),
        y_train, y_val, y_test, scaler,
    )


def parse_args(description, output_name):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results" / output_name,
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.batch_size < 1:
        parser.error("epochs, lr và batch-size phải lớn hơn 0")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or max(args.seeds) >= 2**32:
        parser.error("seeds phải khác nhau và nằm trong [0, 2**32)")
    return args


def evaluate(model, X, y):
    probs = model.predict_proba(X)
    predictions = model.classes_[np.argmax(probs, axis=1)]
    return {
        "loss": log_loss(y, probs, labels=model.classes_),
        "accuracy": accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
    }


def count_parameters(model):
    return sum(weight.size for weight in model.coefs_) + sum(bias.size for bias in model.intercepts_)


def train_classifier(config, data, *, epochs, seed, lr, batch_size):
    model = MLPClassifier(
        hidden_layer_sizes=config.hidden, activation=config.activation,
        solver="adam", alpha=config.alpha, batch_size=min(batch_size, len(data.y_train)),
        learning_rate_init=lr, random_state=seed, shuffle=True,
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_model = None
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        model.partial_fit(data.X_train, data.y_train, classes=np.unique(data.y_train))
        train_metrics = evaluate(model, data.X_train, data.y_train)
        val_metrics = evaluate(model, data.X_val, data.y_val)
        history.append({
            "config": config.name, "seed": seed, "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        })
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_model = deepcopy(model)
    elapsed = perf_counter() - start
    # Chỉ đánh giá test sau khi đã chọn checkpoint bằng validation loss.
    row = {
        "config": config.name, "seed": seed,
        "hidden_layers": "x".join(map(str, config.hidden)),
        "n_layers": len(config.hidden), "width": config.hidden[0],
        "activation": "sigmoid" if config.activation == "logistic" else config.activation,
        "alpha": config.alpha, "parameters": count_parameters(best_model),
        "seconds": elapsed, "best_epoch": best_epoch,
        **{f"train_{k}": v for k, v in evaluate(best_model, data.X_train, data.y_train).items()},
        **{f"val_{k}": v for k, v in evaluate(best_model, data.X_val, data.y_val).items()},
        **{f"test_{k}": v for k, v in evaluate(best_model, data.X_test, data.y_test).items()},
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_accuracy": history[-1]["val_accuracy"],
    }
    return row, history, best_model


METRICS = [
    "seconds", "best_epoch", "train_loss", "val_loss", "test_loss",
    "train_accuracy", "val_accuracy", "test_accuracy", "test_macro_f1",
    "final_train_loss", "final_val_loss", "final_train_accuracy", "final_val_accuracy",
]


def aggregate(runs):
    keys = ["config", "hidden_layers", "n_layers", "width", "activation", "alpha", "parameters"]
    summary = runs.groupby(keys, sort=False)[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = [f"{a}_{b}" if b else a for a, b in summary.columns]
    # Với một seed, std không xác định; giữ NaN để không ngụ ý độ biến động bằng 0.
    return summary


def save_learning_curves(history, configs, output_dir):
    columns = 2 if len(configs) <= 4 else 3
    rows = int(np.ceil(len(configs) / columns))
    for metric, ylabel in [("loss", "Cross-entropy"), ("accuracy", "Accuracy")]:
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.8, rows * 3.5), squeeze=False)
        for ax, config in zip(axes.flat, configs):
            subset = history[history.config == config.name]
            for split, color in [("train", "tab:blue"), ("val", "tab:orange")]:
                groups = subset.groupby("epoch")[f"{split}_{metric}"]
                mean = groups.mean()
                std = groups.std().fillna(0)
                ax.plot(mean.index, mean.values, label=split, color=color)
                ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
            ax.set(title=config.name, xlabel="Epoch", ylabel=ylabel)
            if metric == "accuracy":
                ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.legend()
        for ax in list(axes.flat)[len(configs):]:
            ax.set_visible(False)
        fig.suptitle(f"Learning curves: mean +/- std ({history.seed.nunique()} seeds)")
        fig.tight_layout()
        fig.savefig(output_dir / f"learning_{metric}.png", dpi=150)
        plt.close(fig)


def save_comparison(summary, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    labels = summary.config.tolist()
    fields = [
        ("test_loss", "Test cross-entropy"), ("test_accuracy", "Test accuracy"),
        ("test_macro_f1", "Test Macro-F1"), ("seconds", "Training + monitoring (s)"),
        ("parameters", "Trainable parameters"), ("best_epoch", "Selected epoch"),
    ]
    for ax, (field, title) in zip(axes.flat, fields):
        values = summary[field] if field == "parameters" else summary[f"{field}_mean"]
        errors = None if field == "parameters" else summary[f"{field}_std"].fillna(0)
        ax.bar(labels, values, yerr=errors, capsize=3, color="steelblue")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60)
        ax.grid(axis="y", alpha=0.2)
        if field in ("test_accuracy", "test_macro_f1"):
            ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(fig)


def save_boundaries(models, configs, data, output_dir):
    """Biên quyết định của checkpoint từ seed đầu; điểm tô là tập test."""
    all_X = data.scaler.inverse_transform(np.vstack([data.X_train, data.X_val, data.X_test]))
    xx, yy = np.meshgrid(
        np.linspace(all_X[:, 0].min() - 0.4, all_X[:, 0].max() + 0.4, 220),
        np.linspace(all_X[:, 1].min() - 0.4, all_X[:, 1].max() + 0.4, 220),
    )
    grid = data.scaler.transform(np.c_[xx.ravel(), yy.ravel()])
    test_X = data.scaler.inverse_transform(data.X_test)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, config in zip(axes.flat, configs):
        probability = models[config.name].predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, probability, levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.65)
        ax.contour(xx, yy, probability, levels=[0.5], colors="black", linewidths=1)
        ax.scatter(test_X[:, 0], test_X[:, 1], c=data.y_test, cmap="RdBu", s=14, edgecolors="black", linewidths=0.3)
        ax.set(title=config.name, xlabel="Feature 1", ylabel="Feature 2")
    fig.suptitle("Decision boundaries (first seed); dots: test set")
    fig.tight_layout()
    fig.savefig(output_dir / "decision_boundaries.png", dpi=150)
    plt.close(fig)


def format_value(row, metric, precision=4):
    mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
    return f"{mean:.{precision}f}" if pd.isna(std) else f"{mean:.{precision}f} ± {std:.{precision}f}"


def write_report(summary, data, args, title, dataset, output_dir, boundaries):
    lines = [
        f"# {title}", "", f"Dữ liệu: **{dataset}**.", "",
        f"Train/validation/test: **{len(data.y_train)}/{len(data.y_val)}/{len(data.y_test)}** mẫu, "
        "chia phân tầng, seed chia dữ liệu = 42. StandardScaler chỉ học từ train.", "",
        f"Adam, learning rate = {args.lr}, batch size = {min(args.batch_size, len(data.y_train))}, "
        f"{args.epochs} epoch, seed mô hình = {args.seeds}. Mỗi cấu hình chạy đủ số epoch; "
        "checkpoint được chọn theo validation cross-entropy thấp nhất rồi mới đánh giá test. "
        "Cùng seed được dùng lại cho từng cấu hình để kiểm soát ngẫu nhiên.", "",
        "Loss là cross-entropy không gồm L2; accuracy và Macro-F1 nằm trong [0, 1]. "
        "Số tham số tính cả trọng số và bias. Thời gian gồm huấn luyện, đo train/validation "
        "và lưu checkpoint trong RAM, không gồm vẽ biểu đồ hoặc đánh giá test. "
        "BLAS dùng một luồng; thời gian còn phụ thuộc máy và tải hệ thống. "
        "Bảng ghi trung bình ± độ lệch chuẩn mẫu giữa các seed trên cùng một lần chia dữ liệu.", "",
        "MLP dùng sigmoid ở đầu ra khi phân loại nhị phân và softmax khi nhiều lớp. "
        "Các activation trong bảng là activation của tầng ẩn. L2 (`alpha`) cộng "
        "hạng phạt trọng số vào mục tiêu huấn luyện, được scikit-learn chuẩn hóa theo "
        "số mẫu của batch; đây không phải decoupled weight decay của AdamW.", "",
        "## Bảng so sánh", "",
        "| Cấu hình | Tầng ẩn | Activation | L2 alpha | Tham số | Val loss | Test loss | Test accuracy | Test Macro-F1 | Giây | Epoch chọn |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.config} | {row.hidden_layers} | {row.activation} | {row.alpha:g} | {row.parameters} | "
            f"{format_value(row, 'val_loss')} | {format_value(row, 'test_loss')} | "
            f"{format_value(row, 'test_accuracy')} | {format_value(row, 'test_macro_f1')} | "
            f"{format_value(row, 'seconds', 2)} | {format_value(row, 'best_epoch', 1)} |"
        )
    best = summary.loc[summary.val_loss_mean.idxmin()]
    lines += [
        "", "## Nhận xét ngắn", "",
        f"- Theo validation loss trung bình, **{best.config}** tốt nhất "
        f"({best.val_loss_mean:.4f}); checkpoint của cấu hình này có test accuracy "
        f"{best.test_accuracy_mean:.2%}, Macro-F1 {best.test_macro_f1_mean:.4f}. "
        "Dùng validation để chọn cấu hình; test chỉ dùng báo cáo đánh giá cuối.",
    ]
    if boundaries:
        small, large = summary.iloc[0], summary.iloc[-1]
        lines.append(
            f"- Từ 8 lên 128 neuron mỗi tầng, số tham số tăng từ {small.parameters} lên "
            f"{large.parameters} ({large.parameters / small.parameters:.1f} lần); "
            f"test accuracy thay đổi từ {small.test_accuracy_mean:.2%} thành {large.test_accuracy_mean:.2%}, "
            f"thời gian từ {small.seconds_mean:.2f}s thành {large.seconds_mean:.2f}s. "
            "Nhiều neuron tăng khả năng biểu diễn và chi phí; chất lượng còn phụ thuộc nhiễu "
            "và tối ưu, không bảo đảm tăng đều theo độ rộng."
        )
    else:
        baseline = summary[summary.config == "baseline"].iloc[0]
        lines.append(
            f"- Baseline 2×32 ReLU: val loss {baseline.val_loss_mean:.4f}, "
            f"test accuracy {baseline.test_accuracy_mean:.2%}. "
            "Các nhóm depth, width, activation và L2 chỉ thay một yếu tố so với baseline; "
            "so sánh validation loss và learning curve để đánh giá từng thay đổi."
        )
        for names, description in [
            (["depth_1", "baseline", "depth_3"], "Số tầng ẩn (1/2/3)"),
            (["width_8", "baseline", "width_64", "width_128"], "Số neuron (8/32/64/128)"),
            (["baseline", "tanh", "sigmoid"], "Activation (ReLU/tanh/sigmoid)"),
            (["baseline", "l2_0.01", "l2_0.1"], "L2 (alpha = 0/0.01/0.1)"),
        ]:
            group = summary[summary.config.isin(names)]
            winner = group.loc[group.val_loss_mean.idxmin()]
            lines.append(
                f"- {description}: **{winner.config}** có val loss thấp nhất nhóm "
                f"({winner.val_loss_mean:.4f}). Kết quả chỉ áp dụng cho dữ liệu, learning rate "
                "và ngân sách epoch của thí nghiệm này."
            )
    # Chẩn đoán dựa trên số đo; không gán mọi khoảng cách train/val là overfitting.
    overfit = summary[
        (summary.final_val_loss_mean > summary.val_loss_mean + 0.02)
        & (summary.final_train_loss_mean < summary.train_loss_mean - 0.005)
    ]
    if not overfit.empty:
        lines.append(
            "- Dấu hiệu overfitting ở **" + ", ".join(overfit.config) +
            "**: sau checkpoint tốt nhất, train loss cuối giảm thêm trên 0.005 nhưng "
            "val loss cuối tăng trên 0.02. Chọn checkpoint sớm giúp hạn chế giảm chất lượng; "
            "đối chiếu đường loss và khoảng cách accuracy train/validation."
        )
    else:
        lines.append(
            "- Chưa thấy dấu hiệu overfitting rõ theo tiêu chí: train loss cuối giảm thêm "
            ">0.005 trong khi val loss tăng >0.02 so với checkpoint tốt nhất. "
            "Khoảng cách train/validation nhỏ không đủ để khẳng định không overfit."
        )
    weak = summary[
        (summary.final_train_accuracy_mean < 0.9)
        & (summary.final_val_accuracy_mean < 0.9)
    ]
    lines.append(
        "- Cấu hình có cả accuracy train/validation cuối dưới 90%: **" +
        ", ".join(weak.config) +
        "**. Đây là dấu hiệu học chưa đủ tốt, có thể do underfitting hoặc chưa hội tụ; "
        "cần xem loss còn giảm không trước khi kết luận thiếu năng lực mô hình."
        if not weak.empty else
        "- Không cấu hình nào có cả accuracy train/validation cuối dưới 90%. "
        "Không có bằng chứng underfitting rõ theo ngưỡng tham khảo này; "
        "hãy đối chiếu learning curve thay vì chỉ nhìn test accuracy."
    )
    lines += [
        "- Các seed chỉ phản ánh biến động khởi tạo và thứ tự batch; "
        "chưa đo độ bất định giữa nhiều lần chia dữ liệu. Các ngưỡng chẩn đoán trên là "
        "quy ước tham khảo, không phải kiểm định thống kê.", "",
        "## Biểu đồ", "", "![So sánh các cấu hình](comparison.png)", "",
        "![Learning curve loss](learning_loss.png)", "",
        "![Learning curve accuracy](learning_accuracy.png)", "",
    ]
    if boundaries:
        lines += ["![Biên quyết định](decision_boundaries.png)", ""]
    lines += [
        "## Dữ liệu kết quả", "",
        "- `summary.csv`: bảng trung bình và độ lệch chuẩn.",
        "- `runs.csv`: kết quả từng cấu hình và seed tại checkpoint đã chọn.",
        "- `history.csv`: train/validation loss, accuracy và Macro-F1 mỗi epoch.",
        "- `metadata.json`: cấu hình chạy, cách chia dữ liệu và phiên bản thư viện.", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiments(configs, data, args, *, title, dataset, boundaries=False):
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, histories, models = [], [], {}
    with threadpool_limits(limits=1):
        # Khởi động thư viện trước khi đo thời gian của cấu hình đầu.
        MLPClassifier(hidden_layer_sizes=(2,), random_state=0).partial_fit(
            data.X_train[:64], data.y_train[:64], classes=np.unique(data.y_train),
        )
        for config in configs:
            for seed in args.seeds:
                row, history, model = train_classifier(
                    config, data, epochs=args.epochs, seed=seed,
                    lr=args.lr, batch_size=args.batch_size,
                )
                rows.append(row)
                histories.extend(history)
                if seed == args.seeds[0]:
                    models[config.name] = model
                print(
                    f"{config.name:12} seed={seed} epoch={row['best_epoch']:3} "
                    f"val_loss={row['val_loss']:.4f} test_acc={row['test_accuracy']:.4f} "
                    f"macro_f1={row['test_macro_f1']:.4f} params={row['parameters']} "
                    f"time={row['seconds']:.2f}s", flush=True,
                )
        runs = pd.DataFrame(rows)
        history = pd.DataFrame(histories)
    summary = save_results(
        configs, data, args, runs, history, models,
        title=title, dataset=dataset, boundaries=boundaries,
    )
    return runs, history, summary, models


def save_results(configs, data, args, runs, history, models, *, title, dataset, boundaries=False):
    """Xuất kết quả chung cho script và notebook; trả về bảng tổng hợp."""
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        summary = aggregate(runs)
        runs.to_csv(output_dir / "runs.csv", index=False, encoding="utf-8-sig")
        history.to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
        save_learning_curves(history, configs, output_dir)
        save_comparison(summary, output_dir)
        if boundaries:
            save_boundaries(models, configs, data, output_dir)
    metadata = {
        "dataset": dataset, "split_seed": 42,
        "split_sizes": {split: len(getattr(data, f"y_{split}")) for split in ("train", "val", "test")},
        "split_policy": "stratified 60/20/20; scaler fit on train only",
        "seeds": args.seeds, "epochs": args.epochs, "learning_rate": args.lr,
        "batch_size": min(args.batch_size, len(data.y_train)), "blas_threads": 1,
        "solver": "adam", "checkpoint_selection": "minimum validation cross-entropy",
        "configs": [config.__dict__ for config in configs],
        "versions": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, data, args, title, dataset, output_dir, boundaries)
    print(f"\nSaved results: {output_dir}", flush=True)
    return summary
