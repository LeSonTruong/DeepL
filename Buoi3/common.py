"""Tiện ích cho hai thực hành. Vòng lặp học đầy đủ nằm trong bài 1."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parent
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
           "Shirt", "Sneaker", "Bag", "Ankle boot"]


@dataclass(frozen=True)
class Config:
    name: str
    optimizer: str
    lr: float
    hidden: int
    momentum: float = 0.0


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def device():
    # CPU mặc định giúp chạy lại cùng môi trường đã đo; có thể truyền CUDA cho fit.
    return torch.device("cpu")


def build_model(hidden):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10))


def build_optimizer(model, config):
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)


def make_loaders(train, val, batch_size):
    return (DataLoader(train, batch_size, shuffle=True, generator=torch.Generator().manual_seed(42)),
            DataLoader(train, 256, shuffle=False), DataLoader(val, 256, shuffle=False))


def synchronize(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def state_digest(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate_details(model, loader, loss_fn):
    """Trung bình loss theo số mẫu, kể cả batch cuối thiếu; không tạo gradient."""
    model.eval()
    dev = next(model.parameters()).device
    total_loss, targets, predictions, confidences = 0.0, [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(yb)
            confidence, predicted = logits.softmax(1).max(1)
            targets.extend(yb.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    if not targets:
        raise ValueError("Không thể đánh giá tập rỗng.")
    y, pred = np.asarray(targets), np.asarray(predictions)
    return {"loss": total_loss / len(y), "accuracy": float((y == pred).mean()),
            "macro_f1": f1_score(y, pred, labels=list(range(10)), average="macro", zero_division=0),
            "targets": y, "predictions": pred, "confidences": np.asarray(confidences)}


def evaluate_classifier(model, loader, loss_fn):
    metrics = evaluate_details(model, loader, loss_fn)
    return metrics["loss"], metrics["accuracy"]


class EarlyStopping:
    """Dừng sau patience epoch không cải thiện; giữ bản sao trọng số tốt nhất."""
    def __init__(self, patience=2, min_delta=0.0):
        if patience < 1 or min_delta < 0:
            raise ValueError("patience >= 1, min_delta >= 0")
        self.patience, self.min_delta = patience, min_delta
        self.best_loss, self.best_epoch = float("inf"), 0
        self.bad_epochs, self.best_state = 0, None

    def step(self, val_loss, model, epoch):
        if not np.isfinite(val_loss):
            raise ValueError("Validation loss không hữu hạn; kiểm tra dữ liệu/LR/gradient.")
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.bad_epochs = val_loss, epoch, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model):
        if self.best_state is None:
            raise RuntimeError("Chưa có epoch được đánh giá.")
        model.load_state_dict(self.best_state)


def plot_results(summary, history, predictions, output, val_set, title):
    colors = ["#4477aa", "#cc6677", "#228833"]
    fig, axes = plt.subplots(2, len(summary), figsize=(17, 7), squeeze=False)
    for col, row in enumerate(summary.itertuples()):
        trace = history[history.config == row.config]
        for line, metric, label in [(0, "loss", "Cross-entropy"), (1, "accuracy", "Accuracy")]:
            ax = axes[line, col]
            ax.plot(trace.epoch, trace[f"train_{metric}"], "o-", label="Train (eval)")
            ax.plot(trace.epoch, trace[f"val_{metric}"], "o-", label="Validation")
            ax.set(xlabel="Epoch", ylabel=label, title=f"{row.config} | LR={row.lr:g}")
            ax.set_xticks(trace.epoch)
            if metric == "accuracy":
                ax.set_ylim(0, 1)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, column, label in zip(axes.flat,
        ["val_loss", "val_accuracy", "val_macro_f1", "total_seconds"],
        ["Validation loss", "Validation accuracy", "Validation Macro-F1", "Training + evaluation (s)"]):
        bars = ax.bar(summary.config, summary[column], color=colors[:len(summary)])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_title(label)
        ax.set_ylim(0, 1.05 if "accuracy" in column or "f1" in column else summary[column].max() * 1.25)
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(summary), figsize=(19, 6), squeeze=False)
    for ax, row in zip(axes.flat, summary.itertuples()):
        pred = predictions[predictions.config == row.config]
        matrix = confusion_matrix(pred.target, pred.prediction, labels=range(10))
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
            output / f"confusion_{row.config}.csv", encoding="utf-8-sig")
        ax.imshow(matrix, cmap="Blues")
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set(title=row.config, xlabel="Predicted", ylabel="True",
               xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES)
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    best = summary.loc[summary.val_loss.idxmin(), "config"]
    pred = predictions[predictions.config == best].reset_index(drop=True)
    wrong = pred.index[pred.target != pred.prediction].tolist()[:6]
    right = pred.index[pred.target == pred.prediction].tolist()[:6]
    indices = right + wrong
    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for ax, index in zip(axes.flat, indices):
        image, target = val_set[index]
        record = pred.iloc[index]
        ax.imshow(image.squeeze().numpy(), cmap="gray")
        ax.set_title(f"True: {CLASSES[target]}\nPred: {CLASSES[int(record.prediction)]} ({record.confidence:.0%})",
                     fontsize=9, color="green" if target == record.prediction else "crimson")
        ax.axis("off")
    for ax in list(axes.flat)[len(indices):]:
        ax.axis("off")
    fig.suptitle(f"{best}: 6 correct + up to 6 incorrect validation examples")
    fig.tight_layout()
    fig.savefig(output / "prediction_examples.png", dpi=150)
    plt.close(fig)


def complete_result_metadata(summary, history):
    """Bổ sung thông tin cấu hình cho bảng từ training loop phiên bản cũ.

    Các loop cũ vẫn dùng device() (CPU trong bài này) và lưu số tham số ở
    summary, nhưng chưa lặp device/parameters/lr vào từng hàng history.
    Chỉ bổ sung metadata; không thay đổi các số đo loss/accuracy.
    """
    summary, history = summary.copy(), history.copy()
    if "config" not in summary or "config" not in history:
        raise ValueError("Summary và history cần có cột config.")
    if summary.config.duplicated().any():
        raise ValueError("Mỗi cấu hình phải có đúng một hàng summary.")
    if "device" not in summary:
        summary["device"] = str(device())
    else:
        summary["device"] = summary.device.fillna(str(device()))
    lookup = summary.set_index("config")
    for column in ["device", "parameters", "lr"]:
        values = history.config.map(lookup[column])
        if values.isna().any():
            raise ValueError("History chứa cấu hình không có thông tin trong summary.")
        if column not in history:
            history[column] = values
        else:
            history[column] = history[column].fillna(values)
    history["parameters"] = history.parameters.astype(int)
    return summary, history


def write_report(summary, history, output, title):
    summary, history = complete_result_metadata(summary, history)
    first = summary.iloc[0]
    lines = [f"# {title}", "", "Người làm: **Lê Sơn Trường**.", "",
             f"Fashion-MNIST; train/validation = {first.train_samples}/{first.val_samples}; "
             f"MLP 784 → {first.hidden} → 10, ReLU; seed42; {first.epochs_requested} epoch tối đa. "
             f"Batch train={first.batch_size}, validation=256; CPU, 2 threads.", "",
             "Train loss/accuracy trên learning curve được đo lại ở cuối epoch bằng eval, "
             "cùng cách với validation. online_train_loss là loss trung bình có trọng số của các batch "
             "trong lúc cập nhật; hai đại lượng có thể khác nhau.", "",
             "| Cấu hình | LR | Momentum | Epoch chạy/chọn | Train loss | Val loss | Val accuracy | Macro-F1 | Giây |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.config} | {r.lr:g} | {r.momentum:g} | {r.epochs_run}/{r.selected_epoch} | "
                     f"{r.train_loss:.4f} | {r.val_loss:.4f} | {r.val_accuracy:.2%} | "
                     f"{r.val_macro_f1:.4f} | {r.total_seconds:.2f} |")
    lines += ["", "## Kết quả sau từng epoch", "",
              "| Cấu hình | Epoch | Device | Số tham số | Train loss | Val loss | Val accuracy |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for r in history.itertuples():
        lines.append(f"| {r.config} | {r.epoch} | {r.device} | {r.parameters:,} | "
                     f"{r.train_loss:.4f} | {r.val_loss:.4f} | {r.val_accuracy:.2%} |")
    best = summary.loc[summary.val_loss.idxmin()]
    highest_accuracy = summary.val_accuracy.max()
    accuracy_winners = ", ".join(summary.loc[np.isclose(summary.val_accuracy, highest_accuracy), "config"])
    lines += ["", "## Nhận xét từ kết quả chạy", "",
              f"- **{best['config']}** có validation loss thấp nhất: **{best.val_loss:.4f}**. "
              f"Accuracy cao nhất thuộc **{accuracy_winners}**: **{highest_accuracy:.2%}**."]
    if first.hidden == 256:
        for r in summary.itertuples():
            trace = history[history.config == r.config]
            lines.append(f"- LR={r.lr:g}: validation loss epoch 1 → 5 là "
                         f"{trace.iloc[0].val_loss:.4f} → {trace.iloc[-1].val_loss:.4f}; "
                         f"accuracy {trace.iloc[0].val_accuracy:.2%} → {trace.iloc[-1].val_accuracy:.2%}.")
        lines.append("- LR quyết định độ lớn bước cập nhật. LR lớn có thể học nhanh hoặc dao động; "
                     "LR nhỏ có thể chưa hội tụ sau 5 epoch. Đối chiếu các đường thực đo, "
                     "không mặc định LR lớn luôn kém hay LR nhỏ luôn tốt.")
    else:
        lines += ["- SGD cập nhật trực tiếp theo gradient, ít trạng thái nhưng nhạy với LR; "
                  "Momentum tích lũy hướng cập nhật, có thể giảm dao động; Adam dùng trung bình "
                  "gradient và bình phương gradient để điều chỉnh bước cho từng tham số.",
                  "- LR của ba optimizer khác nhau đúng theo mã mẫu (.05/.03/.001). "
                  "Đây là so sánh ba cấu hình được giao, chưa tách riêng ảnh hưởng của optimizer "
                  "và chưa tìm siêu tham số tốt nhất cho từng thuật toán."]
        for r in summary.itertuples():
            trace = history[history.config == r.config]
            previous, last = trace.iloc[-2], trace.iloc[-1]
            if last.train_loss > previous.train_loss and last.val_loss > previous.val_loss:
                lines.append(f"- {r.config}, epoch {int(previous.epoch)} → {int(last.epoch)}: "
                             f"train loss {previous.train_loss:.4f} → {last.train_loss:.4f} và "
                             f"val loss {previous.val_loss:.4f} → {last.val_loss:.4f} cùng tăng. "
                             "Đây có thể là dao động cập nhật do LR/mini-batch; chưa đủ để kết luận overfitting. "
                             "Có thể thử giảm LR trong khảo sát riêng, giữ bảng chính đúng cấu hình đề.")
    for r in summary.itertuples():
        trace = history[history.config == r.config]
        minimum = trace.loc[trace.val_loss.idxmin()]
        last = trace.iloc[-1]
        if last.val_loss > minimum.val_loss + .01 and last.train_loss < minimum.train_loss:
            lines.append(f"- {r.config}: train loss còn giảm nhưng validation loss tăng sau epoch "
                         f"{int(minimum.epoch)}; có dấu hiệu overfitting/dao động, cần thêm lần chạy để phân biệt.")
    if first.early_stopping:
        stopped = summary[summary.stopped_early]
        lines += ["- Early Stopping: patience=2, min_delta=0; theo dõi validation loss, "
                  "khôi phục trọng số tốt nhất. Ngân sách vẫn tối đa 5 epoch như đề.",
                  "- " + ("Dừng trước giới hạn: " + ", ".join(stopped.config) + "." if len(stopped)
                           else "Lần chạy này chưa cấu hình nào dừng trước 5 epoch. Cài đặt không bảo đảm sẽ dừng nếu loss còn cải thiện.")]
    else:
        lines.append("- Bảng cơ bản lấy **epoch cuối**, không chọn lại epoch có điểm đẹp hơn.")
    lines += ["- Tất cả cấu hình trong mỗi bài dùng cùng trọng số khởi tạo và cùng thứ tự mini-batch. "
              "Số tham số gồm weight và bias; thời gian gồm huấn luyện và đánh giá mỗi epoch, "
              "không gồm đọc dữ liệu, khởi tạo hay vẽ biểu đồ.",
              "- Giữ split của mã gốc: validation lấy từ test chính thức. Sau khi dùng để so sánh LR/optimizer, "
              "không xem đây là điểm test độc lập. Chỉ một seed và 5 epoch nên kết luận có phạm vi giới hạn.",
              "- Nếu loss không giảm: kiểm tra chuẩn hóa /255, nhãn long, logits thô cho CE, "
              "zero_grad/backward/step và LR. Nếu train tốt nhưng val kém: cân nhắc giảm capacity, "
              "regularization hoặc thêm dữ liệu trong thí nghiệm mở rộng có ghi rõ cấu hình.", ""]
    for name in ["learning_curves", "comparison", "confusion_matrices", "prediction_examples"]:
        lines += [f"![{name}]({name}.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_experiment(runs, output, val_set, title, configs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([r["summary"] for r in runs])
    history = pd.DataFrame([item for r in runs for item in r["history"]])
    summary, history = complete_result_metadata(summary, history)
    predictions = pd.concat([r["predictions"] for r in runs], ignore_index=True)
    for name, frame in [("summary", summary), ("history", history), ("predictions", predictions)]:
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    for run in runs:
        torch.save(run["checkpoint"], output / f"{run['summary']['config']}.pt")
    metadata = {"title": title, "author": "Lê Sơn Trường", "device": "cpu", "torch": torch.__version__,
                "seed": 42, "torch_threads": torch.get_num_threads(),
                "configs": [asdict(c) for c in configs], "preprocessing": "float32 /255, no other normalization",
                "validation": "first N samples of official test split; not independent test after comparison",
                "initial_state_sha256": {r["summary"]["config"]: r["initial_hash"] for r in runs},
                "first_batch_sha256": {r["summary"]["config"]: r["batch_hash"] for r in runs},
                "selection": "best validation loss with early stopping" if summary.early_stopping.iloc[0]
                             else "final epoch, no checkpoint selection"}
    assert len(set(metadata["initial_state_sha256"].values())) == 1
    assert len(set(metadata["first_batch_sha256"].values())) == 1
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(summary, history, predictions, output, val_set, title)
    write_report(summary, history, output, title)
    print("\n", summary[["config", "lr", "epochs_run", "selected_epoch", "val_loss", "val_accuracy", "val_macro_f1"]].to_string(index=False), flush=True)
    return summary, history
