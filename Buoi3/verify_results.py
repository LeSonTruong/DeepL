"""Kiểm tra độc lập metric, checkpoint, phép tính loss và Early Stopping."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (BASE, Config, EarlyStopping, build_model, complete_result_metadata,
                    evaluate_details, save_experiment, seed_everything, write_report)
from data import DATA, FILES, load_subset


def check_gradient_and_evaluation():
    logits = torch.tensor([[1000., 1001., 999.], [-2., 0., 1.]], requires_grad=True)
    targets = torch.tensor([1, 0])
    loss = nn.CrossEntropyLoss()(logits, targets)
    expected_loss = (torch.logsumexp(logits, 1) - logits[torch.arange(2), targets]).mean()
    torch.testing.assert_close(loss, expected_loss, atol=2e-5, rtol=1e-5)
    loss.backward()
    expected_gradient = (logits.detach().softmax(1) - nn.functional.one_hot(targets, 3)) / 2
    torch.testing.assert_close(logits.grad, expected_gradient)

    # Batch cuối chỉ 1 mẫu: bắt lỗi lấy trung bình các batch không theo số mẫu.
    x = torch.tensor([[2., 0., -1.], [0., 1., 0.], [0., -1., 2.], [0., 1., 0.], [8., -8., 0.]])
    y = torch.tensor([0, 1, 2, 1, 1])
    model = nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(3))
    result = evaluate_details(model, DataLoader(TensorDataset(x, y), batch_size=2), nn.CrossEntropyLoss())
    np.testing.assert_allclose(result["loss"], nn.functional.cross_entropy(x, y).item(), rtol=1e-6)
    assert result["accuracy"] == .8
    assert model.training is False and all(p.grad is None for p in model.parameters())


def check_early_stopping():
    model = nn.Linear(1, 1, bias=False)
    stopper = EarlyStopping(patience=2)
    decisions = []
    for epoch, loss in enumerate([1., .9, .91, .92], 1):
        with torch.no_grad():
            model.weight.fill_(epoch)
        decisions.append(stopper.step(loss, model, epoch))
    assert decisions == [False, False, False, True] and stopper.best_epoch == 2
    stopper.restore(model)
    assert model.weight.item() == 2  # Bản sao epoch tốt nhất, không bị ghi đè về epoch 4.
    stopper = EarlyStopping(patience=2)
    decisions = [stopper.step(v, model, e) for e, v in enumerate([1., .9, .91, .89, .90, .91], 1)]
    assert decisions == [False] * 5 + [True] and stopper.best_epoch == 4
    monotone = EarlyStopping(patience=2)
    assert not any(monotone.step(v, model, e) for e, v in enumerate([1., .9, .8, .7, .6], 1))


def check_data():
    train, val = load_subset(8000, 1500)
    assert train.tensors[0].shape == (8000, 1, 28, 28)
    assert val.tensors[0].shape == (1500, 1, 28, 28)
    assert train.tensors[0].dtype == torch.float32 and train.tensors[1].dtype == torch.long
    for name, expected in FILES.items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected
    # Đối chiếu ảnh/nhãn đầu trực tiếp với byte IDX, không qua loader của bài làm.
    for split, dataset in [("train", train), ("t10k", val)]:
        with gzip.open(DATA / f"{split}-images-idx3-ubyte.gz", "rb") as stream:
            assert struct.unpack(">IIII", stream.read(16))[0] == 2051
            raw = np.frombuffer(stream.read(28 * 28), dtype=np.uint8).copy().reshape(1, 28, 28)
        np.testing.assert_array_equal(dataset.tensors[0][0].numpy(), raw.astype(np.float32) / np.float32(255))
        with gzip.open(DATA / f"{split}-labels-idx1-ubyte.gz", "rb") as stream:
            stream.read(8)
            assert dataset.tensors[1][0].item() == stream.read(1)[0]
    return train, val


def check_saved_results(val):
    checked = []
    for folder, ntrain, nval, hidden, batch in [
        ("bai1_learning_rate", 6000, 1000, 256, 64),
        ("bai2_optimizer", 8000, 1500, 128, 128),
        ("bai2_early_stopping", 8000, 1500, 128, 128)]:
        output = BASE / "results" / folder
        summary, history, predictions = [pd.read_csv(output / f"{name}.csv")
                                          for name in ["summary", "history", "predictions"]]
        assert len(summary) == 3
        if hidden == 256:
            assert set(summary.lr) == {.01, .001, .0001} and set(summary.optimizer) == {"adam"}
        else:
            actual = {r.config: (r.lr, r.momentum) for r in summary.itertuples()}
            assert actual == {"sgd": (.05, 0.), "momentum": (.03, .9), "adam": (.001, 0.)}
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert len(set(metadata["initial_state_sha256"].values())) == 1
        assert len(set(metadata["first_batch_sha256"].values())) == 1
        selected_val = TensorDataset(*(tensor[:nval] for tensor in val.tensors))
        loader = DataLoader(selected_val, 256)
        for row in summary.itertuples():
            assert row.train_samples == ntrain and row.val_samples == nval
            assert row.hidden == hidden and row.batch_size == batch
            assert row.epochs_requested == 5 and 1 <= row.epochs_run <= 5
            assert row.parameters == (784 + 1) * hidden + (hidden + 1) * 10
            trace = history[history.config == row.config]
            assert trace.epoch.tolist() == list(range(1, row.epochs_run + 1))
            assert row.device == "cpu" and set(trace.device) == {"cpu"}
            assert set(trace.parameters) == {row.parameters} and set(trace.lr) == {row.lr}
            selected = trace[trace.epoch == row.selected_epoch].iloc[0]
            for column in ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_macro_f1"]:
                np.testing.assert_allclose(getattr(row, column), selected[column], atol=1e-7)
            assert np.isfinite(trace.select_dtypes(include="number").to_numpy()).all()
            if row.early_stopping:
                assert row.selected_epoch == int(trace.loc[trace.val_loss.idxmin(), "epoch"])
            else:
                assert row.epochs_run == row.selected_epoch == 5
            pred = predictions[predictions.config == row.config]
            assert pred.sample_index.tolist() == list(range(nval))
            np.testing.assert_array_equal(pred.target, selected_val.tensors[1].numpy())
            np.testing.assert_allclose(row.val_accuracy, (pred.target == pred.prediction).mean())
            np.testing.assert_allclose(row.val_macro_f1, f1_score(pred.target, pred.prediction, average="macro"))
            assert pred.confidence.between(0, 1).all()
            cm = pd.read_csv(output / f"confusion_{row.config}.csv", index_col=0)
            np.testing.assert_array_equal(cm.values, confusion_matrix(pred.target, pred.prediction, labels=range(10)))

            # Chạy checkpoint thật lại, đối chiếu toàn bộ nhãn, độ tự tin và loss đã nộp.
            saved = torch.load(output / f"{row.config}.pt", map_location="cpu", weights_only=True)
            model = build_model(hidden)
            model.load_state_dict(saved["state_dict"])
            metrics = evaluate_details(model, loader, nn.CrossEntropyLoss())
            np.testing.assert_array_equal(metrics["predictions"], pred.prediction)
            np.testing.assert_allclose(metrics["confidences"], pred.confidence, atol=1e-7)
            np.testing.assert_allclose(metrics["loss"], row.val_loss, atol=1e-7)
            assert saved["selected_epoch"] == row.selected_epoch
            checked.append(f"{folder}/{row.config}")
    # Bật Early Stopping không được làm thay đổi đường học trước khi dừng.
    baseline = pd.read_csv(BASE / "results/bai2_optimizer/history.csv")
    advanced = pd.read_csv(BASE / "results/bai2_early_stopping/history.csv")
    merged = baseline.merge(advanced, on=["config", "epoch"], suffixes=("_base", "_es"))
    for column in ["online_train_loss", "train_loss", "val_loss", "val_accuracy"]:
        np.testing.assert_allclose(merged[column + "_base"], merged[column + "_es"], atol=1e-7)
    return checked


def check_legacy_history():
    """Tái hiện lỗi người dùng: history cũ không có device/parameters/lr."""
    output = BASE / "results" / "bai1_learning_rate"
    summary = pd.read_csv(output / "summary.csv").drop(columns=["device"], errors="ignore")
    history = pd.read_csv(output / "history.csv").drop(columns=["device", "parameters", "lr"], errors="ignore")
    original_summary, original_history = summary.copy(deep=True), history.copy(deep=True)
    # Đổi thứ tự hàng để bảo đảm metadata được ghép theo config, không theo vị trí.
    shuffled = history.sample(frac=1, random_state=42)
    completed_summary, completed_history = complete_result_metadata(summary, shuffled)
    pd.testing.assert_frame_equal(completed_summary[summary.columns], summary)
    pd.testing.assert_frame_equal(completed_history[history.columns], shuffled)
    lookup = summary.set_index("config")
    for row in completed_history.itertuples():
        assert row.device == "cpu" and row.parameters == lookup.loc[row.config, "parameters"]
        assert row.lr == lookup.loc[row.config, "lr"]
    pd.testing.assert_frame_equal(summary, original_summary)
    pd.testing.assert_frame_equal(history, original_history)

    predictions = pd.read_csv(output / "predictions.csv")
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    runs, configs = [], []
    for row in summary.itertuples(index=False):
        runs.append({"summary": row._asdict(),
                     "history": history[history.config == row.config].to_dict("records"),
                     "predictions": predictions[predictions.config == row.config],
                     "checkpoint": torch.load(output / f"{row.config}.pt", map_location="cpu", weights_only=True),
                     "initial_hash": metadata["initial_state_sha256"][row.config],
                     "batch_hash": metadata["first_batch_sha256"][row.config]})
        configs.append(Config(row.config, row.optimizer, row.lr, row.hidden, row.momentum))
    with TemporaryDirectory(prefix="verify_legacy_", dir=BASE) as directory:
        target = Path(directory).resolve()
        assert target.parent == BASE.resolve() and target.name.startswith("verify_legacy_")
        # Gọi trực tiếp đúng chỗ từng phát sinh AttributeError, rồi thử cả bước lưu.
        write_report(summary, history, target, "Legacy history regression")
        with patch("common.plot_results"):
            saved_summary, saved_history = save_experiment(runs, target, None, "Legacy history regression", configs)
        exported = pd.read_csv(target / "history.csv")
        assert {"device", "parameters", "lr"}.issubset(exported.columns)
        assert len(exported) == len(history) and set(exported.device) == {"cpu"}
        pd.testing.assert_frame_equal(saved_history[history.columns], history)
        pd.testing.assert_frame_equal(saved_summary[summary.columns], summary)
        assert "203,530" in (target / "report.md").read_text(encoding="utf-8")


def check_notebooks():
    counts = {}
    for stem in ["bai02_class_training_loop", "bai02_homework_optimizer_compare"]:
        nb = nbformat.read(BASE / f"{stem}.ipynb", as_version=4)
        nbformat.validate(nb)
        cells = [c for c in nb.cells if c.cell_type == "code"]
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type == "error" for c in cells for o in c.outputs)
        assert any(o.output_type in {"display_data", "execute_result"}
                   and "image/png" in o.get("data", {}) for c in cells for o in c.outputs)
        streams = "\n".join(o.text for c in cells for o in c.outputs if o.output_type == "stream")
        epoch_lines = re.findall(r"^Epoch \d+/5 \|.*$", streams, flags=re.M)
        assert len(epoch_lines) == (15 if stem == "bai02_class_training_loop" else 30)
        for line in epoch_lines:
            assert all(key in line for key in ["device=cpu", "parameters=", "train_loss=", "val_accuracy="])
        counts[stem] = len(cells)
    return counts


def main():
    seed_everything(42)
    check_gradient_and_evaluation()
    check_early_stopping()
    _, val = check_data()
    configurations = check_saved_results(val)
    check_legacy_history()
    notebooks = check_notebooks()
    result = {"status": "passed", "configurations_checked": configurations,
              "executed_notebook_code_cells": notebooks,
              "checks": ["Stable CE and analytic gradient", "Sample-weighted evaluation with uneven last batch",
                         "No gradient in evaluation", "Early stopping trigger, reset, checkpoint copy and restore",
                         "Dataset SHA256, raw IDX pixels/labels and range(N) split",
                         "Exact assignment settings and fair initialization/batch",
                         "History/summary selection, independent accuracy/F1/confusion",
                         "All saved checkpoints reproduce every validation prediction and loss",
                         "Early stopping preserves baseline trajectory", "Executed notebooks contain images and no errors",
                         "Every epoch log includes device, parameters, train loss and validation accuracy",
                         "Legacy history exports without metadata columns; measurements are preserved"]}
    (BASE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
