"""Đối chiếu kết quả đã lưu với checkpoint và dữ liệu thật, không học lại."""
import csv
import json
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from common import BASE, DEVICE, CNN2Blocks, MLP, describe_cnn, evaluate, read_official, set_seed, save_json
from bai03_homework_cnn_variants import experiments


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_all_confusions(output, configs, selection, images, labels, selected_predictions):
    target = output / "test_all_configs"
    rows = read_csv(target / "summary.csv")
    assert [row["model"] for row in rows] == list(configs)
    truth = labels.numpy()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(images, labels), batch_size=256, shuffle=False)
    for row in rows:
        name = row["model"]
        folder = target / name
        recorded = read_json(folder / "test_metrics.json")
        assert all(str(value) == row[key] for key, value in recorded.items())
        checkpoint = torch.load(output / f"{name}_best.pt", map_location=DEVICE, weights_only=True)
        assert int(row["epoch"]) == checkpoint["epoch"]
        assert float(row["validation_accuracy"]) == checkpoint["val_accuracy"]
        assert float(row["validation_loss"]) == checkpoint["val_loss"]
        assert row["selected_by_validation"] == str(name == selection["model"])
        model = CNN2Blocks(**configs[name]).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        predictions, total_loss = [], 0.0
        with torch.inference_mode():
            for x, y in loader:
                logits = model(x.to(DEVICE).float() / 255)
                total_loss += criterion(logits, y.to(DEVICE)).item() * len(y)
                predictions.append(logits.argmax(1).cpu())
        actual = torch.cat(predictions).numpy()
        saved = read_csv(folder / "test_predictions.csv")
        assert [int(r["test_index"]) for r in saved] == list(range(len(labels)))
        assert np.array_equal([int(r["true"]) for r in saved], truth)
        assert np.array_equal([int(r["predicted"]) for r in saved], actual)
        cm = np.loadtxt(folder / "confusion_matrix.csv", delimiter=",", skiprows=1,
                        usecols=range(1, 11), dtype=int, encoding="utf-8-sig")
        expected = np.bincount(truth * 10 + actual, minlength=100).reshape(10, 10)
        assert np.array_equal(cm, expected) and cm.sum() == 10000
        assert np.all(cm.sum(axis=1) == 1000)
        correct = int(np.trace(cm))
        assert correct == int(row["correct"])
        assert int(row["incorrect"]) == 10000 - correct and int(row["test_size"]) == 10000
        assert math.isclose(correct / 10000, float(row["test_accuracy"]), abs_tol=1e-9)
        assert math.isclose(total_loss / 10000, float(row["test_loss"]), abs_tol=1e-6)
        assert (folder / "confusion_matrix.png").is_file()
        if name == selection["model"]:
            assert np.array_equal(actual, selected_predictions)
    assert (target / "confusion_matrices.png").is_file()
    return len(rows)


def main():
    set_seed()
    first, second = (BASE / "results" / name for name in ("thuc_hanh_1", "thuc_hanh_2"))
    split = read_json(first / "split.json")
    assert split == read_json(second / "split.json")
    train_ids, val_ids = split["train_indices"], split["validation_indices"]
    assert len(set(train_ids)) == 6000 and len(set(val_ids)) == 1200
    assert not set(train_ids) & set(val_ids)
    expected = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).tolist()
    assert train_ids == expected[:6000] and val_ids == expected[6000:7200]
    images, labels = read_official(train=True)
    val_loader = DataLoader(TensorDataset(images[val_ids].float() / 255, labels[val_ids]), batch_size=256)
    saved_configs = read_json(second / "configurations.json")
    assert saved_configs == experiments(), "Cấu hình đã lưu chưa khớp mã nguồn hiện tại."
    architectures = read_json(second / "architectures.json")
    assert [detail["config"] for detail in architectures] == saved_configs
    architecture_by_name = {detail["config"]["name"]: detail for detail in architectures}
    configs = {c["name"]: {k: v for k, v in c.items() if k != "name"} for c in saved_configs}
    baseline = configs["baseline"]
    for name, config in configs.items():
        assert sum(config[k] != baseline[k] for k in config) == (0 if name == "baseline" else 1)
    checkpoints_checked, training_epochs_checked = 0, 0
    for directory in (first, second):
        history = read_csv(directory / "history.csv")
        rows = read_csv(directory / "summary.csv")
        if directory == second:
            assert [r["model"] for r in rows] == list(configs)
        training_epochs_checked += len(history)
        for row in rows:
            name = row["model"]
            records = [r for r in history if r["model"] == name]
            assert [int(r["epoch"]) for r in records] == [1, 2, 3, 4, 5]
            assert all(math.isfinite(float(r[k])) for r in records
                       for k in ("train_loss", "val_loss", "val_accuracy", "seconds"))
            best = max(records, key=lambda r: (float(r["val_accuracy"]), -float(r["val_loss"])))
            assert int(row["best_epoch"]) == int(best["epoch"])
            assert float(row["final_val_accuracy"]) == float(records[-1]["val_accuracy"])
            assert math.isclose(float(row["mean_seconds_per_epoch"]),
                                sum(float(r["seconds"]) for r in records) / 5)
            model = MLP() if name == "MLP" else CNN2Blocks(**configs.get(name, {}))
            assert sum(p.numel() for p in model.parameters()) == int(row["parameters"])
            model.to(DEVICE)
            checkpoint = torch.load(directory / f"{name}_best.pt", map_location=DEVICE, weights_only=True)
            assert checkpoint["epoch"] == int(best["epoch"])
            model.load_state_dict(checkpoint["state_dict"])
            if directory == second:
                saved = architecture_by_name[name]
                assert {k: v for k, v in saved.items() if k != "config"} == describe_cnn(model)
            loss, accuracy = evaluate(model, val_loader, nn.CrossEntropyLoss())
            assert math.isclose(loss, float(row["best_val_loss"]), abs_tol=1e-6)
            assert math.isclose(accuracy, float(row["best_val_accuracy"]), abs_tol=1e-9)
            checkpoints_checked += 1
    # Baseline bài 2 phải tái lập CNN bài 1 với cùng điều kiện.
    cnn = [r for r in read_csv(first / "history.csv") if r["model"] == "CNN"]
    base = [r for r in read_csv(second / "history.csv") if r["model"] == "baseline"]
    for a, b in zip(cnn, base):
        assert all(a[k] == b[k] for k in ("train_loss", "val_loss", "val_accuracy"))
    winner = max(read_csv(second / "summary.csv"),
                 key=lambda r: (float(r["best_val_accuracy"]), -float(r["best_val_loss"])))
    selection = read_json(second / "selection.json")
    metrics = read_json(second / "test_metrics.json")
    assert selection["model"] == winner["model"] and selection["epoch"] == int(winner["best_epoch"])
    predictions = read_csv(second / "test_predictions.csv")
    assert [int(r["test_index"]) for r in predictions] == list(range(10000))
    images, labels = read_official(train=False)
    truth = np.array([int(r["true"]) for r in predictions])
    predicted = np.array([int(r["predicted"]) for r in predictions])
    assert np.array_equal(truth, labels.numpy()) and np.all((0 <= predicted) & (predicted < 10))
    model = CNN2Blocks(**configs[selection["model"]]).to(DEVICE)
    checkpoint = torch.load(second / f"{selection['model']}_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        actual = torch.cat([model(x.to(DEVICE).float() / 255).argmax(1).cpu()
                            for x in DataLoader(images, batch_size=256)])
    assert np.array_equal(actual.numpy(), predicted)
    cm = np.loadtxt(second / "confusion_matrix.csv", delimiter=",", skiprows=1,
                    usecols=range(1, 11), dtype=int, encoding="utf-8-sig")
    assert np.array_equal(cm, np.bincount(truth * 10 + predicted, minlength=100).reshape(10, 10))
    assert math.isclose(np.trace(cm) / 10000, metrics["test_accuracy"], abs_tol=1e-7)
    examples = read_csv(second / "example_indices.csv")
    assert len(examples) == 10 and len({r["test_index"] for r in examples}) == 10
    for correct in (True, False):
        group = [r for r in examples if r["correct"] == str(correct)]
        assert [int(r["test_index"]) for r in group] == np.flatnonzero((truth == predicted) == correct)[:5].tolist()
        for row in group:
            i = int(row["test_index"])
            assert int(row["true"]) == truth[i] and int(row["predicted"]) == predicted[i]
    all_confusions = verify_all_confusions(second, configs, selection, images, labels, predicted)
    save_json(BASE / "verification.json", dict(status="passed", checkpoints_checked=checkpoints_checked,
              variant_architectures_checked=len(architectures),
              training_epochs_checked=training_epochs_checked, test_predictions_checked=10000,
              all_configuration_confusion_matrices_checked=all_confusions,
              all_configuration_test_predictions_checked=all_confusions * 10000,
              split="6000/1200 disjoint, same official train split in both exercises",
              baseline_reproduces_exercise_1=True, selected_model=selection["model"],
              selected_epoch=selection["epoch"], confusion_matrix_and_10_examples="matched"))
    print(f"PASS: split, {training_epochs_checked} epochs, {checkpoints_checked} checkpoints, "
          f"{len(architectures)} architectures, selection, 10000 predictions, confusion matrix, 10 examples.")
    print(f"PASS: {all_confusions} configuration matrices and {all_confusions * 10000} test predictions.")


if __name__ == "__main__":
    main()
