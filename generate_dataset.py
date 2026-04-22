
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets


@dataclass
class DatasetBundle:
    train_dataset: TensorDataset
    test_dataset: TensorDataset
    class_to_indices: Dict[int, np.ndarray]
    num_classes: int
    input_shape: tuple[int, ...]


def _parse_angular_offsets(raw_offsets: object) -> Optional[list[float]]:
    if raw_offsets is None:
        return None
    if isinstance(raw_offsets, (list, tuple)):
        return [float(v) for v in raw_offsets]
    if isinstance(raw_offsets, str):
        text = raw_offsets.strip()
        if text == "":
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [float(v.strip()) for v in text.split(",") if v.strip()]
    raise ValueError("data.angular_offsets must be null, a list/tuple, or a comma-separated string.")


def _validate_spiral_data_config(data_cfg: dict) -> None:
    num_classes = int(data_cfg.get("num_classes", 3))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    revolutions = float(data_cfg.get("revolutions", 4.0))
    noise_std = float(data_cfg.get("noise_std", 0.1))
    test_ratio = float(data_cfg.get("test_ratio", 0.25))
    min_radius = float(data_cfg.get("min_radius", 0.05))

    if num_classes < 2:
        raise ValueError("For spiral data, data.num_classes must be >= 2.")
    if points_per_class <= 0:
        raise ValueError("For spiral data, data.points_per_class must be > 0.")
    if revolutions <= 0:
        raise ValueError("For spiral data, data.revolutions must be > 0.")
    if noise_std < 0:
        raise ValueError("For spiral data, data.noise_std must be >= 0.")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("For spiral data, data.test_ratio must be in (0, 1).")
    if min_radius <= 0 or min_radius >= 1.0:
        raise ValueError("For spiral data, data.min_radius must be in (0, 1).")

    offsets = _parse_angular_offsets(data_cfg.get("angular_offsets"))
    if offsets is not None and len(offsets) != num_classes:
        raise ValueError(
            f"For spiral data, data.angular_offsets must have exactly data.num_classes values ({num_classes})."
        )


def build_class_index_map(labels: torch.Tensor, num_classes: int) -> Dict[int, np.ndarray]:
    class_to_indices: Dict[int, np.ndarray] = {}
    labels_np = labels.detach().cpu().numpy()
    for class_id in range(num_classes):
        class_indices = np.where(labels_np == class_id)[0].astype(np.int64)
        if class_indices.size == 0:
            raise ValueError(f"Class {class_id} has no samples in training data.")
        class_to_indices[class_id] = class_indices
    return class_to_indices


def generate_spiral_data(
    points_per_class: int,
    num_classes: int,
    revolutions: float,
    noise_std: float,
    random_seed: int,
    angular_offsets: Optional[Sequence[float]] = None,
    randomize_offsets: bool = False,
    min_radius: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    n = points_per_class
    c = num_classes

    if angular_offsets is not None:
        if len(angular_offsets) != c:
            raise ValueError(f"angular_offsets must have {c} values")
        offsets = np.deg2rad(np.array(angular_offsets, dtype=np.float32))
    elif randomize_offsets:
        offsets = rng.uniform(0.0, 2.0 * np.pi, size=c).astype(np.float32)
    else:
        offsets = np.array([2.0 * np.pi * j / c for j in range(c)], dtype=np.float32)

    x_all = np.zeros((n * c, 2), dtype=np.float32)
    y_all = np.zeros((n * c,), dtype=np.int64)

    for j in range(c):
        ix = slice(n * j, n * (j + 1))
        r = np.linspace(min_radius, 1.0, n, dtype=np.float32)
        theta = np.linspace(offsets[j], offsets[j] + revolutions * np.pi, n, dtype=np.float32)
        theta += rng.normal(0.0, noise_std, size=n).astype(np.float32)
        x_all[ix, 0] = r * np.cos(theta)
        x_all[ix, 1] = r * np.sin(theta)
        y_all[ix] = j

    return x_all, y_all


def create_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.25,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    classes = np.unique(y)

    train_idx = []
    test_idx = []
    for class_id in classes:
        idx = np.where(y == class_id)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_ratio)
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def save_mnist_visualizations(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    out_dir: Path,
) -> None:
    train_images = x_train[:, 0].detach().cpu().numpy()
    train_labels = y_train.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("MNIST Training Dataset Samples", fontsize=16)
    for class_idx in range(10):
        row, col = divmod(class_idx, 5)
        idxs = np.where(train_labels == class_idx)[0]
        if idxs.size > 0:
            axes[row, col].imshow(train_images[idxs[0]], cmap="gray")
            axes[row, col].set_title(f"Class {class_idx}")
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "training_dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.close()

    train_counts = np.bincount(train_labels, minlength=10)
    test_counts = np.bincount(y_test.detach().cpu().numpy(), minlength=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(np.arange(10), train_counts, alpha=0.8, color="skyblue")
    ax1.set_title("Train Class Distribution")
    ax2.bar(np.arange(10), test_counts, alpha=0.8, color="salmon")
    ax2.set_title("Test Class Distribution")
    for ax in (ax1, ax2):
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "dataset_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_spiral_visualizations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    for class_id in range(num_classes):
        mask = y_train == class_id
        if np.any(mask):
            axes[0].scatter(
                x_train[mask, 0],
                x_train[mask, 1],
                s=12,
                alpha=0.98,
                color=cmap(class_id % 10),
                edgecolors="black",
                linewidths=0.2,
                label=f"Class {class_id}",
            )
    axes[0].set_title("Spiral Training Samples")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    for class_id in range(num_classes):
        mask = y_test == class_id
        if np.any(mask):
            axes[1].scatter(
                x_test[mask, 0],
                x_test[mask, 1],
                s=12,
                alpha=0.98,
                color=cmap(class_id % 10),
                edgecolors="black",
                linewidths=0.2,
                label=f"Class {class_id}",
            )
    axes[1].set_title("Spiral Test Samples")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "spiral_dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.close()


def _load_mnist_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    data_dir = Path(str(data_cfg.get("data_dir", "./data"))).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    train_raw = datasets.MNIST(str(data_dir), train=True, download=True)
    test_raw = datasets.MNIST(str(data_dir), train=False, download=True)

    x_train = train_raw.data.float().unsqueeze(1) / 255.0
    x_test = test_raw.data.float().unsqueeze(1) / 255.0
    x_train = (x_train - 0.1307) / 0.3081
    x_test = (x_test - 0.1307) / 0.3081

    y_train = train_raw.targets.long()
    y_test = test_raw.targets.long()

    save_mnist_visualizations(x_train, y_train, x_test, y_test, output_dir)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    num_classes = int(data_cfg.get("num_classes", 10))
    class_to_indices = build_class_index_map(y_train, num_classes)
    input_shape = tuple(x_train.shape[1:])
    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=input_shape,
    )


def _load_spiral_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    _validate_spiral_data_config(data_cfg)
    num_classes = int(data_cfg.get("num_classes", 3))
    x_full, y_full = generate_spiral_data(
        points_per_class=int(data_cfg.get("points_per_class", 1000)),
        num_classes=num_classes,
        revolutions=float(data_cfg.get("revolutions", 4.0)),
        noise_std=float(data_cfg.get("noise_std", 0.1)),
        random_seed=int(data_cfg.get("random_seed", 0)),
        angular_offsets=_parse_angular_offsets(data_cfg.get("angular_offsets")),
        randomize_offsets=bool(data_cfg.get("randomize_offsets", False)),
        min_radius=float(data_cfg.get("min_radius", 0.05)),
    )

    x_train, y_train, x_test, y_test = create_train_test_split(
        x_full,
        y_full,
        test_ratio=float(data_cfg.get("test_ratio", 0.25)),
        random_seed=int(data_cfg.get("random_seed", 0)),
    )

    save_spiral_visualizations(x_train, y_train, x_test, y_test, num_classes, output_dir)

    x_train_t = torch.from_numpy(x_train).float()
    x_test_t = torch.from_numpy(x_test).float()
    y_train_t = torch.from_numpy(y_train).long()
    y_test_t = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(x_train_t, y_train_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)
    class_to_indices = build_class_index_map(y_train_t, num_classes)

    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=(2,),
    )


def build_dataset_bundle(config: dict, output_dir: Path) -> DatasetBundle:
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset_name", "spiral")).strip().lower()
    if dataset_name == "mnist":
        return _load_mnist_bundle(data_cfg, output_dir)
    if dataset_name == "spiral":
        return _load_spiral_bundle(data_cfg, output_dir)
    raise ValueError(f"Unsupported dataset_name '{dataset_name}'. Expected 'mnist' or 'spiral'.")