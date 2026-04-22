from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


EPS = 1e-12


@dataclass
class SeparabilityEpochRaw:
    probe_train_acc_input: float
    probe_train_acc_pre: float
    probe_test_acc_input: float
    probe_test_acc_pre: float
    probe_train_loss_input: float
    probe_train_loss_pre: float
    probe_test_loss_input: float
    probe_test_loss_pre: float
    probe_test_acc_gain: float
    probe_test_loss_reduction: float
    scatter_input: float
    scatter_pre: float
    scatter_gain: float
    scatter_ratio: float
    kta_input: float
    kta_pre: float
    kta_gain: float


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    if labels.shape[0] > 0:
        y[np.arange(labels.shape[0]), labels] = 1.0
    return y


def _extract_input_and_preclassifier_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classifier = getattr(model, "fc", None)
    if classifier is None:
        raise ValueError("Model does not expose classifier layer as 'fc'; cannot compute separability measures.")

    captured: dict[str, torch.Tensor] = {}

    def _hook_input(
        _module: nn.Module,
        hook_input: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        captured["pre_classifier"] = hook_input[0].detach()

    handle = classifier.register_forward_hook(_hook_input)

    raw_parts: list[np.ndarray] = []
    pre_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []

    was_training = model.training
    use_non_blocking = device.type == "cuda"
    model.eval()
    try:
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device, non_blocking=use_non_blocking)
                target = target.to(device, non_blocking=use_non_blocking)

                captured.clear()
                _ = model(data)

                pre = captured.get("pre_classifier")
                if pre is None:
                    raise RuntimeError("Failed to capture pre-classifier features from classifier hook.")

                raw = torch.reshape(data, (data.shape[0], -1)).to(dtype=torch.float64)
                pre = torch.reshape(pre, (pre.shape[0], -1)).to(dtype=torch.float64)

                raw_parts.append(raw.detach().cpu().numpy().astype(np.float64, copy=False))
                pre_parts.append(pre.detach().cpu().numpy().astype(np.float64, copy=False))
                label_parts.append(target.detach().cpu().numpy().astype(np.int64, copy=False))
    finally:
        handle.remove()
        if was_training:
            model.train()

    if not raw_parts:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.concatenate(raw_parts, axis=0),
        np.concatenate(pre_parts, axis=0),
        np.concatenate(label_parts, axis=0),
    )


def _fit_ridge_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    l2_reg: float,
) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((features.shape[1] + 1, num_classes), dtype=np.float64)

    x = np.concatenate(
        [features.astype(np.float64, copy=False), np.ones((features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    y = _one_hot(labels, num_classes)

    gram = x.T @ x
    reg = float(max(0.0, l2_reg)) * np.eye(gram.shape[0], dtype=np.float64)
    # Do not regularize the bias term.
    reg[-1, -1] = 0.0
    rhs = x.T @ y

    try:
        weights = np.linalg.solve(gram + reg, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram + reg) @ rhs
    return weights


def _probe_accuracy_and_loss(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if features.shape[0] == 0:
        return float("nan"), float("nan")

    x = np.concatenate(
        [features.astype(np.float64, copy=False), np.ones((features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    logits = x @ weights

    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == labels))

    stable = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = stable - np.log(np.sum(np.exp(stable), axis=1, keepdims=True) + EPS)
    ce = float(-np.mean(log_probs[np.arange(labels.shape[0]), labels]))
    return acc, ce


def _scatter_ratio(features: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    if features.shape[0] == 0:
        return float("nan")

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts <= 0):
        return float("nan")

    d = features.shape[1]
    means = np.zeros((num_classes, d), dtype=np.float64)
    for class_id in range(num_classes):
        class_feats = features[labels == class_id]
        means[class_id] = np.mean(class_feats, axis=0)

    total = float(np.sum(counts))
    global_mean = np.sum(means * counts[:, None], axis=0) / max(total, 1.0)

    between = float(np.sum(counts * np.sum((means - global_mean[None, :]) ** 2, axis=1)))

    within = 0.0
    for class_id in range(num_classes):
        class_feats = features[labels == class_id]
        diffs = class_feats - means[class_id][None, :]
        within += float(np.sum(diffs * diffs))

    return between / (within + EPS)


def _linear_kta(features: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    if features.shape[0] == 0:
        return float("nan")

    x = features.astype(np.float64, copy=False)
    x_centered = x - np.mean(x, axis=0, keepdims=True)

    y = _one_hot(labels, num_classes)
    y_centered = y - np.mean(y, axis=0, keepdims=True)

    cross = x_centered.T @ y_centered
    numerator = float(np.sum(cross * cross))

    xx = x_centered.T @ x_centered
    yy = y_centered.T @ y_centered
    denom = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)) + EPS)
    return numerator / denom


def collect_separability_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    probe_l2_reg: float,
) -> SeparabilityEpochRaw:
    train_input, train_pre, train_labels = _extract_input_and_preclassifier_features(
        model=model,
        loader=train_loader,
        device=device,
    )
    test_input, test_pre, test_labels = _extract_input_and_preclassifier_features(
        model=model,
        loader=test_loader,
        device=device,
    )

    probe_input = _fit_ridge_linear_probe(
        features=train_input,
        labels=train_labels,
        num_classes=num_classes,
        l2_reg=probe_l2_reg,
    )
    probe_pre = _fit_ridge_linear_probe(
        features=train_pre,
        labels=train_labels,
        num_classes=num_classes,
        l2_reg=probe_l2_reg,
    )

    probe_train_acc_input, probe_train_loss_input = _probe_accuracy_and_loss(train_input, train_labels, probe_input)
    probe_test_acc_input, probe_test_loss_input = _probe_accuracy_and_loss(test_input, test_labels, probe_input)

    probe_train_acc_pre, probe_train_loss_pre = _probe_accuracy_and_loss(train_pre, train_labels, probe_pre)
    probe_test_acc_pre, probe_test_loss_pre = _probe_accuracy_and_loss(test_pre, test_labels, probe_pre)

    scatter_input = _scatter_ratio(train_input, train_labels, num_classes)
    scatter_pre = _scatter_ratio(train_pre, train_labels, num_classes)

    kta_input = _linear_kta(train_input, train_labels, num_classes)
    kta_pre = _linear_kta(train_pre, train_labels, num_classes)

    return SeparabilityEpochRaw(
        probe_train_acc_input=probe_train_acc_input,
        probe_train_acc_pre=probe_train_acc_pre,
        probe_test_acc_input=probe_test_acc_input,
        probe_test_acc_pre=probe_test_acc_pre,
        probe_train_loss_input=probe_train_loss_input,
        probe_train_loss_pre=probe_train_loss_pre,
        probe_test_loss_input=probe_test_loss_input,
        probe_test_loss_pre=probe_test_loss_pre,
        probe_test_acc_gain=probe_test_acc_pre - probe_test_acc_input,
        probe_test_loss_reduction=probe_test_loss_input - probe_test_loss_pre,
        scatter_input=scatter_input,
        scatter_pre=scatter_pre,
        scatter_gain=scatter_pre - scatter_input,
        scatter_ratio=scatter_pre / (scatter_input + EPS),
        kta_input=kta_input,
        kta_pre=kta_pre,
        kta_gain=kta_pre - kta_input,
    )


def initialize_separability_csv(csv_path: Path) -> None:
    header = [
        "epoch",
        "global_step",
        "probe_train_acc_input",
        "probe_train_acc_pre",
        "probe_test_acc_input",
        "probe_test_acc_pre",
        "probe_train_loss_input",
        "probe_train_loss_pre",
        "probe_test_loss_input",
        "probe_test_loss_pre",
        "probe_test_acc_gain",
        "probe_test_loss_reduction",
        "scatter_input",
        "scatter_pre",
        "scatter_gain",
        "scatter_ratio",
        "kta_input",
        "kta_pre",
        "kta_gain",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_separability_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: SeparabilityEpochRaw,
) -> None:
    row = [
        epoch,
        global_step,
        raw.probe_train_acc_input,
        raw.probe_train_acc_pre,
        raw.probe_test_acc_input,
        raw.probe_test_acc_pre,
        raw.probe_train_loss_input,
        raw.probe_train_loss_pre,
        raw.probe_test_loss_input,
        raw.probe_test_loss_pre,
        raw.probe_test_acc_gain,
        raw.probe_test_loss_reduction,
        raw.scatter_input,
        raw.scatter_pre,
        raw.scatter_gain,
        raw.scatter_ratio,
        raw.kta_input,
        raw.kta_pre,
        raw.kta_gain,
    ]
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_rows(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            casted: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    casted[key] = float("nan")
                elif key in {"epoch", "global_step"}:
                    casted[key] = float(int(float(value)))
                else:
                    casted[key] = float(value)
            rows.append(casted)
    return rows


def _col(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)


def finalize_separability_metrics(
    csv_path: Path,
    output_path: Path,
    tpt_step: int = -1,
) -> None:
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"No separability rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)

    fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=True)

    ax = axes[0, 0]
    ax.plot(steps, _col(rows, "probe_test_acc_input"), linewidth=1.8, label="input probe")
    ax.plot(steps, _col(rows, "probe_test_acc_pre"), linewidth=1.8, label="pre-classifier probe")
    ax.set_title("Linear Probe Test Accuracy (higher is better)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, _col(rows, "probe_test_loss_input"), linewidth=1.8, label="input probe")
    ax.plot(steps, _col(rows, "probe_test_loss_pre"), linewidth=1.8, label="pre-classifier probe")
    ax.set_title("Linear Probe Test Cross-Entropy (lower is better)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(steps, _col(rows, "scatter_input"), linewidth=1.8, label="input")
    ax.plot(steps, _col(rows, "scatter_pre"), linewidth=1.8, label="pre-classifier")
    ax.set_title("Fisher-like Separation tr(Sb)/tr(Sw) (higher is better)")
    ax.set_ylabel("Separation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(steps, _col(rows, "kta_input"), linewidth=1.8, label="input")
    ax.plot(steps, _col(rows, "kta_pre"), linewidth=1.8, label="pre-classifier")
    ax.set_title("Linear Kernel-Target Alignment (higher is better)")
    ax.set_ylabel("Alignment")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(steps, _col(rows, "probe_test_acc_gain"), linewidth=1.8, label="acc gain (pre-input, higher better)")
    ax.plot(
        steps,
        _col(rows, "probe_test_loss_reduction"),
        linewidth=1.8,
        label="loss reduction (input-pre, higher better)",
    )
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7)
    ax.set_title("Linear Probe Gain (>0 is better)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Delta")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(steps, _col(rows, "scatter_gain"), linewidth=1.8, label="scatter gain (higher better)")
    ax.plot(steps, _col(rows, "kta_gain"), linewidth=1.8, label="KTA gain (higher better)")
    ax.axhline(0.0, linestyle="--", color="gray", alpha=0.7)
    ax.set_title("Geometry Gain (Pre - Input, >0 is better)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Delta")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Separability Measures", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
