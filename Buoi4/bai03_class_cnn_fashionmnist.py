"""Thực hành 1: MLP 784→256→10 và CNN hai block trên cùng split FashionMNIST."""
import torch
from reporting import add_epoch_details
from common import (BASE, DEVICE, CNN2Blocks, MLP, set_seed, make_data, fit,
                    save_csv, metadata, learning_curves, result_table)


@torch.inference_mode()
def print_shapes(model, batch_size=1):
    model.eval()
    x = torch.zeros(batch_size, 1, 28, 28, device=next(model.parameters()).device)
    lines = [f"input: {tuple(x.shape)}"]
    x = model.conv1(x)
    lines.append(f"conv1: {tuple(x.shape)}")
    x = model.relu(model.bn1(x))
    lines.append(f"relu1: {tuple(x.shape)}")
    x = model.pool(x)
    lines.append(f"pool1: {tuple(x.shape)}")
    x = model.conv2(x)
    lines.append(f"conv2: {tuple(x.shape)}")
    x = model.relu(model.bn2(x))
    lines.append(f"relu2: {tuple(x.shape)}")
    x = model.pool(x)
    lines.append(f"pool2: {tuple(x.shape)}")
    x = model.flatten(x)
    lines.append(f"before Linear: {tuple(x.shape)}")
    x = model.relu(model.fc1(x))
    lines.append(f"Linear + ReLU: {tuple(x.shape)}")
    lines.append(f"output: {tuple(model.fc2(model.drop(x)).shape)}")
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def main():
    output = BASE / "results" / "thuc_hanh_1"
    output.mkdir(parents=True, exist_ok=True)
    set_seed()
    train_ds, val_ds = make_data(output)
    print(f"Device: {DEVICE} | train: {len(train_ds)} | validation: {len(val_ds)}", flush=True)
    shapes = "\n\n".join(print_shapes(CNN2Blocks(), n) for n in (1, 64))
    (output / "shapes.txt").write_text(shapes + "\n", encoding="utf-8")
    rows, history = [], []
    for name, model_cls in (("MLP", MLP), ("CNN", CNN2Blocks)):
        set_seed()
        row, records = fit(name, model_cls(), train_ds, val_ds, output)
        rows.append(row)
        history.extend(records)
    save_csv(output / "summary.csv", rows)
    save_csv(output / "history.csv", history)
    metadata(output)
    learning_curves(history, output / "learning_curves.png")
    mlp, cnn = rows
    delta = 100 * (cnn["final_val_accuracy"] - mlp["final_val_accuracy"])
    comments = [
        "Tensor CNN đi từ (N,1,28,28) qua (N,16,28,28), (N,16,14,14), "
        "(N,32,14,14), (N,32,7,7), rồi flatten thành (N,1568); chiều N là batch và được giữ nguyên.",
        f"MLP có {mlp['parameters']:,} tham số và CNN có {cnn['parameters']:,} tham số; "
        "convolution dùng chung trọng số theo vị trí, nhưng lớp Linear 1568→128 vẫn chiếm phần lớn tham số CNN.",
        f"Sau 5 epoch, MLP đạt {mlp['final_val_accuracy']:.2%} và CNN đạt {cnn['final_val_accuracy']:.2%} "
        f"validation accuracy, tức CNN {'cao' if delta >= 0 else 'thấp'} hơn {abs(delta):.2f} điểm phần trăm.",
        f"Thời gian trung bình gồm train và validation là {mlp['mean_seconds_per_epoch']:.3f} s/epoch cho MLP "
        f"và {cnn['mean_seconds_per_epoch']:.3f} s/epoch cho CNN trên {DEVICE}; số tham số gần nhau không có nghĩa số phép tính giống nhau.",
        "Kết quả chỉ phản ánh một seed và 5 epoch trên split này; train loss đo trong lúc cập nhật, "
        "còn validation loss đo cuối epoch nên hai đường loss không hoàn toàn cùng thời điểm.",
    ]
    report = "# Thực hành 1 — Lê Sơn Trường\n\n" + result_table(rows)
    report += "\n\nBảng dùng epoch cuối (5); cùng seed42, split6000/1200, Adam, lr0.001, batch64.\n\n"
    report += "![Learning curves](learning_curves.png)\n\n```text\n" + shapes + "\n```\n\n"
    report += "\n\n".join(comments) + "\n"
    report = add_epoch_details(report, history, rows)
    (output / "report.md").write_text(report, encoding="utf-8")
    print("\n" + result_table(rows), flush=True)


if __name__ == "__main__":
    main()
