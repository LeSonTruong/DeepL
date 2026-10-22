"""Thực hành 2: chọn cấu hình/checkpoint bằng validation, sau đó đánh giá test."""
import argparse
import csv
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from reporting import add_epoch_details, add_variant_details, add_test_comparison
from common import (BASE, DEVICE, CLASSES, CNN2Blocks, set_seed, make_data,
                    read_official, fit, save_csv, save_json, metadata,
                    learning_curves, result_table, describe_cnn, plt)


def experiments():
    baseline = dict(channels=16, kernel_size=3, batchnorm=False, dropout=0.0)
    return [dict(name="baseline", **baseline),
            dict(name="kernel5", **{**baseline, "kernel_size": 5}),
            dict(name="channels8", **{**baseline, "channels": 8}),
            dict(name="channels32", **{**baseline, "channels": 32}),
            dict(name="batchnorm", **{**baseline, "batchnorm": True}),
            dict(name="dropout03", **{**baseline, "dropout": 0.3})]


def export_architectures(configs, output, verbose=False):
    """Nhật ký kiến trúc từng cấu hình; không tiêu thụ RNG của huấn luyện."""
    architectures, lines = [], ["Shape probe: zero input, CPU, eval mode; not training/test predictions."]
    with torch.random.fork_rng(devices=[]):
        for config in configs:
            model = CNN2Blocks(**{k: v for k, v in config.items() if k != "name"})
            detail = {"config": config, **describe_cnn(model)}
            architectures.append(detail)
            lines += ["", f"=== {config['name']} | parameters={detail['parameters']:,} ==="]
            for trace in detail["traces"]:
                size = trace["batch_size"]
                lines += [f"input: {(size, 1, 28, 28)}"]
                lines += [f"{layer['step']}: {tuple(layer['output_shape'])} | {layer['layer']}"
                          for layer in trace["layers"]]
    save_json(output / "architectures.json", architectures)
    text = "\n".join(lines) + "\n"
    (output / "shapes.txt").write_text(text, encoding="utf-8")
    if verbose:
        print(text, flush=True)
    return architectures


@torch.inference_mode()
def predict_test(model, loader):
    """Forward ở eval; trả dự đoán và loss, không cập nhật trọng số."""
    model.eval()
    predictions, total_loss, count = [], 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        logits = model(x.to(DEVICE))
        total_loss += criterion(logits, y.to(DEVICE)).item() * len(y)
        predictions.append(logits.argmax(1).cpu())
        count += len(y)
    return torch.cat(predictions), total_loss / count


def plot_confusion(ax, cm, title, fontsize=8):
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=int(cm.sum(axis=1).max()))
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted label (columns)", ylabel="True label (rows)", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.sum(axis=1).max() / 2 else "black",
                    fontsize=fontsize)
    return image


def save_test_outputs(output, labels, predicted, title):
    output.mkdir(parents=True, exist_ok=True)
    cm = torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10).numpy()
    assert cm.sum() == len(labels)
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true / predicted", *CLASSES])
        for label, counts in zip(CLASSES, cm):
            writer.writerow([label, *counts.tolist()])
    save_csv(output / "test_predictions.csv", [dict(test_index=i, true=int(y), predicted=int(p))
             for i, (y, p) in enumerate(zip(labels, predicted))])
    fig, ax = plt.subplots(figsize=(10, 9))
    image = plot_confusion(ax, cm, title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return cm


def final_test(model, output, title="Selected model | official test"):
    """Chỉ gọi sau khi đã khóa lựa chọn mô hình bằng validation."""
    images, labels = read_official(train=False)
    dataset = TensorDataset(images.float().div_(255), labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    predicted, loss = predict_test(model, loader)
    cm = save_test_outputs(output, labels, predicted, title)
    example_records = []
    for correct, filename in ((True, "correct_examples.png"), (False, "incorrect_examples.png")):
        ids = torch.where((labels == predicted) == correct)[0][:5]
        if len(ids) < 5:
            raise RuntimeError(f"Chỉ có {len(ids)} ảnh {'đúng' if correct else 'sai'} trên test")
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.4))
        for ax, i in zip(axes, ids.tolist()):
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"#{i}\nTrue: {CLASSES[labels[i]]}\nPred: {CLASSES[predicted[i]]}", fontsize=9)
            ax.axis("off")
            example_records.append(dict(test_index=i, true=int(labels[i]), predicted=int(predicted[i]), correct=correct))
        fig.suptitle("Five correct predictions" if correct else "Five incorrect predictions")
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    save_csv(output / "example_indices.csv", example_records)
    return dict(test_size=len(labels), test_loss=loss,
                test_accuracy=float((labels == predicted).float().mean()),
                confusion_accuracy=float(np.trace(cm) / cm.sum()))


def evaluate_all_configurations(output):
    """Đánh giá các checkpoint đã chọn bằng validation; giữ nguyên selection.json."""
    selected = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    configs = json.loads((output / "configurations.json").read_text(encoding="utf-8"))
    checkpoints = {config["name"]: torch.load(output / f"{config['name']}_best.pt",
                   map_location=DEVICE, weights_only=True) for config in configs}
    assert checkpoints[selected["model"]]["epoch"] == selected["epoch"]
    images, labels = read_official(train=False)
    loader = DataLoader(TensorDataset(images.float().div_(255), labels), batch_size=256, shuffle=False)
    target = output / "test_all_configs"
    rows, matrices = [], []
    for config in configs:
        name = config["name"]
        checkpoint = checkpoints[name]
        model = CNN2Blocks(**{k: v for k, v in config.items() if k != "name"}).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        predicted, loss = predict_test(model, loader)
        folder = target / name
        cm = save_test_outputs(folder, labels, predicted,
                              f"{name} | epoch {checkpoint['epoch']} | official test")
        correct = int(np.trace(cm))
        row = dict(model=name, epoch=checkpoint["epoch"], validation_accuracy=checkpoint["val_accuracy"],
                   validation_loss=checkpoint["val_loss"], test_size=len(labels), test_loss=loss,
                   test_accuracy=correct / len(labels), correct=correct, incorrect=len(labels)-correct,
                   selected_by_validation=name == selected["model"], checkpoint=f"{name}_best.pt")
        save_json(folder / "test_metrics.json", row)
        rows.append(row)
        matrices.append(cm)
        print(f"Test {name}: epoch={row['epoch']} | accuracy={row['test_accuracy']:.2%}", flush=True)
    save_csv(target / "summary.csv", rows)
    fig, axes = plt.subplots(3, 2, figsize=(18, 23), constrained_layout=True)
    for ax, row, cm in zip(axes.flat, rows, matrices):
        marker = " | selected by validation" if row["selected_by_validation"] else ""
        plot_confusion(ax, cm, f"{row['model']} | epoch {row['epoch']} | test {row['test_accuracy']:.2%}{marker}", 8)
    fig.suptitle("Official test: 10,000 images per configuration | rows=true, columns=predicted", fontsize=16)
    fig.savefig(target / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    return rows


def main():
    output = BASE / "results" / "thuc_hanh_2"
    output.mkdir(parents=True, exist_ok=True)
    set_seed()
    train_ds, val_ds = make_data(output)
    print(f"Device: {DEVICE} | FashionMNIST | train: {len(train_ds)} | validation: {len(val_ds)}", flush=True)
    configs = experiments()
    save_json(output / "configurations.json", configs)
    architectures = export_architectures(configs, output, verbose=True)
    rows, history = [], []
    for config in configs:
        set_seed()
        kwargs = {k: v for k, v in config.items() if k != "name"}
        row, records = fit(config["name"], CNN2Blocks(**kwargs), train_ds, val_ds, output)
        rows.append({**row, **kwargs})
        history.extend(records)
    save_csv(output / "summary.csv", rows)
    save_csv(output / "history.csv", history)
    metadata(output)
    learning_curves(history, output / "learning_curves.png")
    # Cả checkpoint theo epoch và cấu hình đều dùng accuracy; hòa thì loss nhỏ hơn.
    winner = max(rows, key=lambda r: (r["best_val_accuracy"], -r["best_val_loss"]))
    config = next(c for c in configs if c["name"] == winner["model"])
    selected = dict(model=winner["model"], epoch=winner["best_epoch"],
                    validation_accuracy=winner["best_val_accuracy"], validation_loss=winner["best_val_loss"],
                    rule="max validation accuracy, tie: min validation loss, exact tie: first")
    save_json(output / "selection.json", selected)
    model = CNN2Blocks(**{k: v for k, v in config.items() if k != "name"}).to(DEVICE)
    checkpoint = torch.load(output / f"{winner['model']}_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = final_test(model, output, f"{winner['model']} | epoch {winner['best_epoch']} | selected model")
    save_json(output / "test_metrics.json", {**selected, **metrics})
    evaluate_all_configurations(output)
    baseline = rows[0]
    comments = [
        "Dùng 6.000 ảnh train và 1.200 ảnh validation không giao nhau từ official train; "
        "10.000 ảnh official test chỉ được đọc sau khi đã lưu lựa chọn mô hình.",
        "Mỗi cấu hình giữ seed42, split, Adam, lr0.001, batch64 và 5 epoch; "
        "mỗi biến thể chỉ thay một thuộc tính so với baseline.",
        f"Baseline đạt validation accuracy tốt nhất {baseline['best_val_accuracy']:.2%} ở epoch "
        f"{baseline['best_epoch']}, với {baseline['parameters']:,} tham số.",
    ]
    mechanisms = {"kernel5": "Kernel 5×5 mở rộng vùng quan sát mỗi convolution và tăng số phép tính",
                  "channels8": "Giảm số kênh từ 16/32 xuống 8/16 làm giảm tham số và khả năng biểu diễn",
                  "channels32": "Tăng số kênh từ 16/32 lên 32/64 làm tăng số tham số và khả năng biểu diễn",
                  "batchnorm": "BatchNorm dùng thống kê batch khi train và running statistics khi eval",
                  "dropout03": "Dropout tắt ngẫu nhiên 30% activation sau fc1 khi train và tắt cơ chế này khi eval"}
    for row in rows[1:]:
        delta = 100 * (row["best_val_accuracy"] - baseline["best_val_accuracy"])
        comments.append(f"{mechanisms[row['model']]}; đạt {row['best_val_accuracy']:.2%} validation accuracy "
                        f"({'tăng' if delta >= 0 else 'giảm'} {abs(delta):.2f} điểm phần trăm so với baseline), "
                        f"{row['parameters']:,} tham số và {row['mean_seconds_per_epoch']:.3f} s/epoch.")
    comments.append(f"Chọn {winner['model']} tại epoch {winner['best_epoch']} theo validation; "
                    f"test accuracy cuối cùng là {metrics['test_accuracy']:.2%}, test loss {metrics['test_loss']:.4f}.")
    comments.append("Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán; ảnh minh họa là 5 mẫu đúng "
                    "và 5 mẫu sai đầu tiên theo thứ tự test, không chọn theo mức độ đẹp của kết quả.")
    comments.append("Đây là kết quả một seed với 5 epoch, chưa chứng minh biến thể nào luôn tốt hơn; "
                    "train loss là trung bình khi cập nhật, validation loss dùng eval ở cuối epoch.")
    report = "# Thực hành 2 — Lê Sơn Trường\n\n"
    report += "| Cấu hình | Channels | Kernel | BatchNorm | Dropout |\n|---|---|---|---|---|\n"
    report += "\n".join(f"| {c['name']} | {c['channels']}/{2*c['channels']} | {c['kernel_size']} | "
                        f"{c['batchnorm']} | {c['dropout']} |" for c in configs)
    report += ("\n\nBaseline đại diện kernel3 và channel16 ở block1. Các biến thể bao phủ "
               "kernel **3/5** và channel block1 **8/16/32**; block2 có số kênh gấp đôi block1. "
               "Mỗi biến thể chỉ đổi một thuộc tính so với baseline để phân tích riêng ảnh hưởng của nó.")
    report += "\n\n" + result_table(rows, best=True)
    report += "\n\nBảng báo cáo checkpoint validation tốt nhất của mỗi cấu hình; history.csv lưu mọi epoch.\n\n"
    report += "![Learning curves](learning_curves.png)\n\n" + "\n\n".join(comments) + "\n\n"
    report += "![Confusion matrix](confusion_matrix.png)\n\n![5 ảnh đúng](correct_examples.png)\n\n![5 ảnh sai](incorrect_examples.png)\n"
    report = add_epoch_details(report, history, rows, selected)
    report = add_variant_details(report, output, architectures)
    report = add_test_comparison(report, output)
    (output / "report.md").write_text(report, encoding="utf-8")
    print("\n" + result_table(rows, best=True), flush=True)
    print(f"Selected: {winner['model']} epoch {winner['best_epoch']} | test accuracy={metrics['test_accuracy']:.2%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Đánh giá cả 6 checkpoint đã lưu và cập nhật báo cáo; không huấn luyện lại.")
    args = parser.parse_args()
    if args.evaluate_only:
        set_seed()
        evaluate_all_configurations(BASE / "results" / "thuc_hanh_2")
        from build_notebooks import write_combined_report
        print("Report:", write_combined_report())
    else:
        main()
