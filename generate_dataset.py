from __future__ import annotations

import gzip
import shutil
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}


@dataclass
class DatasetBundle:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
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
    min_radius = float(data_cfg.get("min_radius", 0.05))

    if num_classes < 2:
        raise ValueError("For spiral data, data.num_classes must be >= 2.")
    if points_per_class <= 0:
        raise ValueError("For spiral data, data.points_per_class must be > 0.")
    if revolutions <= 0:
        raise ValueError("For spiral data, data.revolutions must be > 0.")
    if noise_std < 0:
        raise ValueError("For spiral data, data.noise_std must be >= 0.")
    if min_radius <= 0 or min_radius >= 1.0:
        raise ValueError("For spiral data, data.min_radius must be in (0, 1).")

    offsets = _parse_angular_offsets(data_cfg.get("angular_offsets"))
    if offsets is not None and len(offsets) != num_classes:
        raise ValueError(
            f"For spiral data, data.angular_offsets must have exactly data.num_classes values ({num_classes})."
        )


def _validate_gaussian_blobs_config(data_cfg: dict) -> None:
    num_classes = int(data_cfg.get("num_classes", 3))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    noise_std = float(data_cfg.get("noise_std", 0.1))
    center_radius = float(data_cfg.get("center_radius", 2.0))

    if num_classes < 2:
        raise ValueError("For gaussian_blobs data, data.num_classes must be >= 2.")
    if points_per_class <= 0:
        raise ValueError("For gaussian_blobs data, data.points_per_class must be > 0.")
    if noise_std < 0:
        raise ValueError("For gaussian_blobs data, data.noise_std must be >= 0.")
    if center_radius <= 0:
        raise ValueError("For gaussian_blobs data, data.center_radius must be > 0.")


def _validate_rings_config(data_cfg: dict) -> None:
    num_classes = int(data_cfg.get("num_classes", 3))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    noise_std = float(data_cfg.get("noise_std", 0.1))
    center_radius = float(data_cfg.get("center_radius", 1.0))
    class_separation = float(data_cfg.get("class_separation", 0.5))

    if num_classes < 2:
        raise ValueError("For rings data, data.num_classes must be >= 2.")
    if points_per_class <= 0:
        raise ValueError("For rings data, data.points_per_class must be > 0.")
    if noise_std < 0:
        raise ValueError("For rings data, data.noise_std must be >= 0.")
    if center_radius <= 0:
        raise ValueError("For rings data, data.center_radius must be > 0.")
    if class_separation <= 0:
        raise ValueError("For rings data, data.class_separation must be > 0.")
    if noise_std >= class_separation / 2.0:
        raise ValueError(
            "For rings data, noise_std must be < class_separation / 2 to avoid ring overlap."
        )


def _validate_blobs_config(data_cfg: dict) -> None:
    num_classes = int(data_cfg.get("num_classes", 3))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    noise_std = float(data_cfg.get("noise_std", 0.15))
    center_radius = float(data_cfg.get("center_radius", 2.5))
    class_separation = float(data_cfg.get("class_separation", 120.0))

    if num_classes < 2:
        raise ValueError("For blobs data, data.num_classes must be >= 2.")
    if points_per_class <= 0:
        raise ValueError("For blobs data, data.points_per_class must be > 0.")
    if noise_std < 0:
        raise ValueError("For blobs data, data.noise_std must be >= 0.")
    if center_radius <= 0:
        raise ValueError("For blobs data, data.center_radius must be > 0.")
    if not (0.0 < class_separation <= 360.0):
        raise ValueError("For blobs data, data.class_separation must be in (0, 360] degrees.")


def _validate_checkerboard_config(data_cfg: dict) -> None:
    grid_size = int(data_cfg.get("grid_size", 4))
    num_classes = int(data_cfg.get("num_classes", 2))
    noise_std = float(data_cfg.get("noise_std", 0.2))
    points_per_class = int(data_cfg.get("points_per_class", 1000))

    if num_classes < 2:
        raise ValueError("For checkerboard, num_classes must be >= 2.")
    if grid_size < 2:
        raise ValueError("For checkerboard, grid_size must be >= 2.")
    if grid_size < num_classes:
        raise ValueError("For checkerboard, grid_size must be >= num_classes.")
    if points_per_class <= 0:
        raise ValueError("For checkerboard, points_per_class must be > 0.")
    if noise_std <= 0:
        raise ValueError("For checkerboard, noise_std must be > 0.")
    cell_width = 2.0 / grid_size
    if noise_std > cell_width / 2.0:
        raise ValueError(
            f"noise_std={noise_std} > cell_width/2={cell_width/2:.3f}: "
            "point squares would overlap between adjacent cells. "
            "Reduce noise_std or increase grid_size."
        )
    
def _validate_dartboard_config(data_cfg: dict) -> None:
    num_classes      = int(data_cfg.get("num_classes",    2))
    num_rings        = int(data_cfg.get("num_rings",      4))
    num_sectors      = int(data_cfg.get("num_sectors",    8))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    noise_std        = float(data_cfg.get("noise_std",    0.0))
    inner_radius     = float(data_cfg.get("inner_radius", 0.05))
    outer_radius     = float(data_cfg.get("outer_radius", 1.0))
 
    if num_classes < 2:
        raise ValueError("dartboard: num_classes must be >= 2.")
    if num_rings < 1:
        raise ValueError("dartboard: num_rings must be >= 1.")
    if num_sectors < 2:
        raise ValueError("dartboard: num_sectors must be >= 2.")
    if num_rings * num_sectors < num_classes:
        raise ValueError(
            f"dartboard: num_rings * num_sectors ({num_rings * num_sectors}) "
            f"must be >= num_classes ({num_classes})."
        )
    if points_per_class <= 0:
        raise ValueError("dartboard: points_per_class must be > 0.")
    if noise_std < 0:
        raise ValueError("dartboard: noise_std must be >= 0.")
    if inner_radius < 0 or inner_radius >= outer_radius:
        raise ValueError(
            "dartboard: inner_radius must satisfy 0 <= inner_radius < outer_radius."
        )
    if outer_radius <= 0:
        raise ValueError("dartboard: outer_radius must be > 0.")
 
    ring_width    = (outer_radius - inner_radius) / num_rings
    min_arc_width = outer_radius * (2.0 * np.pi / num_sectors)
    if noise_std > min(ring_width, min_arc_width) / 2.0:
        raise ValueError(
            f"dartboard: noise_std={noise_std:.4f} may cause cells to overlap "
            f"(ring_width={ring_width:.4f}, min_arc_width={min_arc_width:.4f}). "
            "Reduce noise_std or increase num_rings/num_sectors."
        )


def build_class_index_map(labels: np.ndarray, num_classes: int) -> Dict[int, np.ndarray]:
    class_to_indices: Dict[int, np.ndarray] = {}
    labels_np = np.asarray(labels, dtype=np.int64)
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


def generate_gaussian_blobs(
    points_per_class: int,
    num_classes: int,
    noise_std: float,
    random_seed: int,
    center_radius: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    angles = np.linspace(0.0, 2.0 * np.pi, num_classes, endpoint=False, dtype=np.float64)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * float(center_radius)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for class_id in range(num_classes):
        pts = rng.normal(centers[class_id], noise_std, size=(points_per_class, 2)).astype(np.float32)
        x_parts.append(pts)
        y_parts.append(np.full(points_per_class, class_id, dtype=np.int64))

    return np.vstack(x_parts), np.concatenate(y_parts)


def generate_rings(
    points_per_class: int,
    num_classes: int,
    noise_std: float,
    random_seed: int,
    center_radius: float = 0.3,
    class_separation: float = 0.15,
    num_rings: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Each class has num_rings concentric rings, interleaved with other classes.
    Ring k of class c has radius: center_radius + (c + k*num_classes) * class_separation
    noise_std is the radial half-width of each ring (keep small relative to class_separation).
    All rings share center (0, 0).
    """
    rng = np.random.default_rng(random_seed)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    pts_per_ring = points_per_class // num_rings
    remainder = points_per_class - pts_per_ring * num_rings

    for class_id in range(num_classes):
        all_pts: list[np.ndarray] = []
        for ring_idx in range(num_rings):
            n = pts_per_ring + (1 if ring_idx < remainder else 0)
            ring_radius = float(center_radius) + (class_id + ring_idx * num_classes) * float(class_separation)
            t = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
            r = ring_radius + rng.uniform(-float(noise_std), float(noise_std), size=n).astype(np.float32)
            pts = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
            all_pts.append(pts.astype(np.float32))
        x_parts.append(np.vstack(all_pts))
        y_parts.append(np.full(points_per_class, class_id, dtype=np.int64))

    return np.vstack(x_parts), np.concatenate(y_parts)


import numpy as np

def generate_blobs_classes(
    points_per_class: int,
    num_classes: int,
    class_separation: float,  # Retained for signature compatibility
    noise_std: float,
    random_seed: int,
    center_radius: float = 0.7,
    num_modes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Each class has exactly num_modes blobs (supports both odd and even numbers).
    Blobs are distributed evenly around a circle and interleaved by class.
    """
    rng = np.random.default_rng(random_seed)

    # num_modes is now the total number of blobs per class
    pts_per_blob = points_per_class // num_modes
    remainder = points_per_class - pts_per_blob * num_modes

    # Total slots around the circle across all classes combined
    angular_step = 360.0 / (num_classes * num_modes)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for class_id in range(num_classes):
        all_pts: list[np.ndarray] = []
        for mode_idx in range(num_modes):
            # Interleave classes perfectly (e.g., Class 0 @ slot 0, Class 1 @ slot 1...)
            angle_deg = (class_id + mode_idx * num_classes) * angular_step
            angle_rad = np.deg2rad(angle_deg)
            
            center = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * float(center_radius)
            
            # Distribute remainder points cleanly across the loops
            n = pts_per_blob + (1 if mode_idx < remainder else 0)
            
            pts = rng.normal(center, float(noise_std), size=(n, 2))
            all_pts.append(pts.astype(np.float32))
            
        x_parts.append(np.vstack(all_pts))
        y_parts.append(np.full(points_per_class, class_id, dtype=np.int64))

    return np.vstack(x_parts), np.concatenate(y_parts)

def generate_checkerboard(
    points_per_class: int,
    num_classes: int,
    noise_std: float,
    random_seed: int,
    grid_size: int = 4,
    random_tile_classes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Places a grid_size x grid_size array of cell centers within [-1, 1]^2.
    Cell centers are at (-1 + (i+0.5)*cell_width, -1 + (j+0.5)*cell_width).
    Class assignment: (i + j) % num_classes by default, or random per tile
    when random_tile_classes is True.
    Points for each cell are sampled uniformly in a square of half-width
    noise_std centered on that cell's center.
    """
    rng = np.random.default_rng(random_seed)
    cell_width = 2.0 / float(grid_size)

    if random_tile_classes:
        tile_classes = rng.integers(0, num_classes, size=(grid_size, grid_size), dtype=np.int64)
        while len(np.unique(tile_classes)) < num_classes:
            tile_classes = rng.integers(0, num_classes, size=(grid_size, grid_size), dtype=np.int64)
    else:
        tile_classes = np.fromfunction(
            lambda i, j: (i + j) % num_classes,
            (grid_size, grid_size),
            dtype=np.int64,
        ).astype(np.int64)

    # collect centers per class
    centers_per_class: list[list[tuple[float, float]]] = [[] for _ in range(num_classes)]
    for i in range(grid_size):
        for j in range(grid_size):
            class_id = int(tile_classes[i, j])
            cx = -1.0 + (i + 0.5) * cell_width
            cy = -1.0 + (j + 0.5) * cell_width
            centers_per_class[class_id].append((float(cx), float(cy)))

    if any(len(centers) == 0 for centers in centers_per_class):
        raise ValueError(
            "checkerboard config produced an empty class; reduce num_classes or increase grid_size."
        )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for class_id in range(num_classes):
        centers = centers_per_class[class_id]
        n_centers = len(centers)
        pts_per_center = points_per_class // n_centers
        remainder = points_per_class - pts_per_center * n_centers

        all_pts: list[np.ndarray] = []
        for k, (cx, cy) in enumerate(centers):
            n = pts_per_center + (1 if k < remainder else 0)
            if n <= 0:
                continue
            pts = rng.uniform(-float(noise_std), float(noise_std), size=(n, 2)).astype(np.float32)
            pts[:, 0] += cx
            pts[:, 1] += cy
            all_pts.append(pts)

        if not all_pts:
            raise ValueError("checkerboard config produced no samples for a class.")

        x_parts.append(np.vstack(all_pts))
        y_parts.append(np.full(points_per_class, class_id, dtype=np.int64))

    return np.vstack(x_parts), np.concatenate(y_parts)


def generate_random_checkerboard(
    points_per_class: int,
    num_classes: int,
    noise_std: float,
    random_seed: int,
    grid_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    return generate_checkerboard(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=noise_std,
        random_seed=random_seed,
        grid_size=grid_size,
        random_tile_classes=True,
    )

def generate_dartboard(
    points_per_class: int,
    num_classes:      int,
    random_seed:      int,
    num_rings:        int   = 4,
    num_sectors:      int   = 8,
    noise_std:        float = 0.0,
    inner_radius:     float = 0.05,
    outer_radius:     float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dartboard dataset.
 
    The plane is divided into (num_rings × num_sectors) cells.  Cell (ri, si)
    is assigned class  (ri + si) % num_classes,  so the label alternates both
    radially and angularly — the polar analogue of the XOR checkerboard.
 
    Points are sampled with uniform area density inside each cell (uniform in
    r², uniform in angle), then optionally perturbed by isotropic Gaussian
    noise in (x, y) space.
 
    Returns
    -------
    x : float32 array, shape (num_classes * points_per_class, 2)
    y : int64   array, shape (num_classes * points_per_class,)
    """
    rng = np.random.default_rng(random_seed)
 
    ring_edges   = np.linspace(inner_radius, outer_radius, num_rings   + 1)
    sector_edges = np.linspace(0.0, 2.0 * np.pi,           num_sectors + 1)
 
    # Group cells by class.
    cells_per_class: list[list[tuple[int, int]]] = [[] for _ in range(num_classes)]
    for ri in range(num_rings):
        for si in range(num_sectors):
            cells_per_class[(ri + si) % num_classes].append((ri, si))
 
    for class_id, cells in enumerate(cells_per_class):
        if not cells:
            raise ValueError(
                f"dartboard: class {class_id} has no cells. "
                "Increase num_rings or num_sectors relative to num_classes."
            )
 
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
 
    for class_id in range(num_classes):
        cells     = cells_per_class[class_id]
        n_cells   = len(cells)
        pts_per_cell = points_per_class // n_cells
        remainder    = points_per_class - pts_per_cell * n_cells
 
        all_pts: list[np.ndarray] = []
        for k, (ri, si) in enumerate(cells):
            n = pts_per_cell + (1 if k < remainder else 0)
            if n <= 0:
                continue
 
            r_lo, r_hi = float(ring_edges[ri]),    float(ring_edges[ri + 1])
            a_lo, a_hi = float(sector_edges[si]),  float(sector_edges[si + 1])
 
            # Uniform area sampling: uniform in r² gives uniform area.
            r2 = rng.uniform(r_lo ** 2, r_hi ** 2, size=n)
            r  = np.sqrt(r2).astype(np.float32)
            a  = rng.uniform(a_lo, a_hi, size=n).astype(np.float32)
 
            pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
 
            if noise_std > 0.0:
                pts += rng.normal(0.0, noise_std, size=pts.shape).astype(np.float32)
 
            all_pts.append(pts)
 
        x_parts.append(np.vstack(all_pts))
        y_parts.append(np.full(points_per_class, class_id, dtype=np.int64))
 
    return np.vstack(x_parts), np.concatenate(y_parts)


def save_mnist_visualizations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
) -> None:
    train_images = x_train[:, 0]
    train_labels = y_train

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
    test_counts = np.bincount(y_test, minlength=10)

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


def _save_2d_dataset_visualizations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    out_dir: Path,
    title_prefix: str,
    output_name: str,
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
    axes[0].set_title(f"{title_prefix} Training Samples")
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
    axes[1].set_title(f"{title_prefix} Test Samples")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / output_name, dpi=150, bbox_inches="tight")
    plt.close()


def save_spiral_visualizations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    out_dir: Path,
) -> None:
    _save_2d_dataset_visualizations(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        out_dir=out_dir,
        title_prefix="Spiral",
        output_name="spiral_dataset_samples.png",
    )


def _ensure_mnist_raw_files(data_dir: Path) -> dict[str, Path]:
    raw_dir = data_dir / "MNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, Path] = {}
    for key, filename in MNIST_FILES.items():
        target = raw_dir / filename
        if not target.exists():
            gz_target = raw_dir / f"{filename}.gz"
            if not gz_target.exists():
                url = f"{MNIST_BASE_URL}/{filename}.gz"
                urllib.request.urlretrieve(url, gz_target)
            with gzip.open(gz_target, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        resolved[key] = target
    return resolved


def _read_mnist_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num_items, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        payload = f.read()
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size != num_items * rows * cols:
        raise ValueError(f"Corrupted MNIST image file: {path}")
    return arr.reshape(num_items, rows, cols)


def _read_mnist_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        payload = f.read()
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size != num_items:
        raise ValueError(f"Corrupted MNIST label file: {path}")
    return arr


def _load_mnist_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    data_dir = Path(str(data_cfg.get("data_dir", "./data"))).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = _ensure_mnist_raw_files(data_dir)
    train_images = _read_mnist_idx_images(paths["train_images"])
    train_labels = _read_mnist_idx_labels(paths["train_labels"])
    test_images = _read_mnist_idx_images(paths["test_images"])
    test_labels = _read_mnist_idx_labels(paths["test_labels"])

    x_train = train_images.astype(np.float32)[:, None, :, :] / 255.0
    x_test = test_images.astype(np.float32)[:, None, :, :] / 255.0
    x_train = (x_train - 0.1307) / 0.3081
    x_test = (x_test - 0.1307) / 0.3081

    # Pad 28×28 → 32×32 with 2-pixel zero border on each side, matching the
    # reference training code: transforms.Pad((padded_im_size - im_size) // 2)
    # i.e. transforms.Pad(2).  Shape goes (N,1,28,28) → (N,1,32,32).
    pad = 2
    x_train = np.pad(x_train, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    x_test  = np.pad(x_test,  ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    y_train = train_labels.astype(np.int64)
    y_test = test_labels.astype(np.int64)

    num_classes = int(data_cfg.get("num_classes", 10))
    if num_classes != 10:
        raise ValueError("MNIST loader expects data.num_classes=10.")

    save_mnist_visualizations(x_train, y_train, x_test, y_test, output_dir)

    class_to_indices = build_class_index_map(y_train, num_classes)
    input_shape = tuple(x_train.shape[1:])
    return DatasetBundle(
        train_inputs=x_train,
        train_targets=y_train,
        test_inputs=x_test,
        test_targets=y_test,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=input_shape,
    )


def _load_spiral_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    _validate_spiral_data_config(data_cfg)
    num_classes = int(data_cfg.get("num_classes", 3))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    random_seed = int(data_cfg.get("random_seed", 0))

    x_train, y_train = generate_spiral_data(
        points_per_class=points_per_class,
        num_classes=num_classes,
        revolutions=float(data_cfg.get("revolutions", 4.0)),
        noise_std=float(data_cfg.get("noise_std", 0.1)),
        random_seed=random_seed,
        angular_offsets=_parse_angular_offsets(data_cfg.get("angular_offsets")),
        randomize_offsets=bool(data_cfg.get("randomize_offsets", False)),
        min_radius=float(data_cfg.get("min_radius", 0.05)),
    )
    x_test, y_test = generate_spiral_data(
        points_per_class=points_per_class,
        num_classes=num_classes,
        revolutions=float(data_cfg.get("revolutions", 4.0)),
        noise_std=float(data_cfg.get("noise_std", 0.1)),
        random_seed=random_seed + 1,
        angular_offsets=_parse_angular_offsets(data_cfg.get("angular_offsets")),
        randomize_offsets=bool(data_cfg.get("randomize_offsets", False)),
        min_radius=float(data_cfg.get("min_radius", 0.05)),
    )

    save_spiral_visualizations(x_train, y_train, x_test, y_test, num_classes, output_dir)

    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    class_to_indices = build_class_index_map(y_train_arr, num_classes)

    return DatasetBundle(
        train_inputs=x_train_arr,
        train_targets=y_train_arr,
        test_inputs=x_test_arr,
        test_targets=y_test_arr,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=(2,),
    )


def _load_gaussian_blobs_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    _validate_gaussian_blobs_config(data_cfg)
    num_classes = int(data_cfg.get("num_classes", 3))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    random_seed = int(data_cfg.get("random_seed", 0))

    x_train, y_train = generate_gaussian_blobs(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=float(data_cfg.get("noise_std", 0.1)),
        random_seed=random_seed,
        center_radius=float(data_cfg.get("center_radius", 2.0)),
    )
    x_test, y_test = generate_gaussian_blobs(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=float(data_cfg.get("noise_std", 0.1)),
        random_seed=random_seed + 1,
        center_radius=float(data_cfg.get("center_radius", 2.0)),
    )

    _save_2d_dataset_visualizations(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        out_dir=output_dir,
        title_prefix="Gaussian Blobs",
        output_name="gaussian_blobs_dataset_samples.png",
    )

    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    class_to_indices = build_class_index_map(y_train_arr, num_classes)

    return DatasetBundle(
        train_inputs=x_train_arr,
        train_targets=y_train_arr,
        test_inputs=x_test_arr,
        test_targets=y_test_arr,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=(2,),
    )


def _load_rings_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    _validate_rings_config(data_cfg)
    num_classes = int(data_cfg.get("num_classes", 4))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    random_seed = int(data_cfg.get("random_seed", 0))

    x_train, y_train = generate_rings(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=float(data_cfg.get("noise_std", 0.1)),
        random_seed=random_seed,
        center_radius=float(data_cfg.get("center_radius", 3.0)),
        class_separation=float(data_cfg.get("class_separation", 1.0)),
        num_rings=int(data_cfg.get("num_rings", 3)),
    )
    x_test, y_test = generate_rings(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=float(data_cfg.get("noise_std", 0.1)),
        random_seed=random_seed + 1,
        center_radius=float(data_cfg.get("center_radius", 3.0)),
        class_separation=float(data_cfg.get("class_separation", 1.0)),
        num_rings=int(data_cfg.get("num_rings", 3)),
    )

    _save_2d_dataset_visualizations(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        out_dir=output_dir,
        title_prefix="Rings",
        output_name="rings_dataset_samples.png",
    )

    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    class_to_indices = build_class_index_map(y_train_arr, num_classes)

    return DatasetBundle(
        train_inputs=x_train_arr,
        train_targets=y_train_arr,
        test_inputs=x_test_arr,
        test_targets=y_test_arr,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=(2,),
    )


def _load_blobs_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    _validate_blobs_config(data_cfg)
    num_classes = int(data_cfg.get("num_classes", 4))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    random_seed = int(data_cfg.get("random_seed", 0))

    x_train, y_train = generate_blobs_classes(
        points_per_class=points_per_class,
        num_classes=num_classes,
        class_separation=float(data_cfg.get("class_separation", 1.5)),
        noise_std=float(data_cfg.get("noise_std", 0.15)),
        random_seed=random_seed,
        center_radius=float(data_cfg.get("center_radius", 2.5)),
        num_modes=int(data_cfg.get("num_modes", 3)),
    )
    x_test, y_test = generate_blobs_classes(
        points_per_class=points_per_class,
        num_classes=num_classes,
        class_separation=float(data_cfg.get("class_separation", 1.5)),
        noise_std=float(data_cfg.get("noise_std", 0.15)),
        random_seed=random_seed + 1,
        center_radius=float(data_cfg.get("center_radius", 2.5)),
        num_modes=int(data_cfg.get("num_modes", 3)),
    )

    _save_2d_dataset_visualizations(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        out_dir=output_dir,
        title_prefix="blobs Classes",
        output_name="blobs_dataset_samples.png",
    )

    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    class_to_indices = build_class_index_map(y_train_arr, num_classes)

    return DatasetBundle(
        train_inputs=x_train_arr,
        train_targets=y_train_arr,
        test_inputs=x_test_arr,
        test_targets=y_test_arr,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=(2,),
    )


def _load_checkerboard_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    _validate_checkerboard_config(data_cfg)
    num_classes = int(data_cfg.get("num_classes", 3))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    random_seed = int(data_cfg.get("random_seed", 0))

    x_train, y_train = generate_checkerboard(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=float(data_cfg.get("noise_std", 0.0)),
        random_seed=random_seed,
        grid_size=int(data_cfg.get("grid_size", 4)),
    )
    x_test, y_test = generate_checkerboard(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=float(data_cfg.get("noise_std", 0.0)),
        random_seed=random_seed + 1,
        grid_size=int(data_cfg.get("grid_size", 4)),
    )

    _save_2d_dataset_visualizations(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        out_dir=output_dir,
        title_prefix="Checkerboard",
        output_name="checkerboard_dataset_samples.png",
    )

    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    class_to_indices = build_class_index_map(y_train_arr, num_classes)

    return DatasetBundle(
        train_inputs=x_train_arr,
        train_targets=y_train_arr,
        test_inputs=x_test_arr,
        test_targets=y_test_arr,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=(2,),
    )


def _load_random_checkerboard_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    _validate_checkerboard_config(data_cfg)
    num_classes = int(data_cfg.get("num_classes", 3))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    random_seed = int(data_cfg.get("random_seed", 0))

    x_train, y_train = generate_random_checkerboard(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=float(data_cfg.get("noise_std", 0.0)),
        random_seed=random_seed,
        grid_size=int(data_cfg.get("grid_size", 4)),
    )
    x_test, y_test = generate_random_checkerboard(
        points_per_class=points_per_class,
        num_classes=num_classes,
        noise_std=float(data_cfg.get("noise_std", 0.0)),
        random_seed=random_seed + 1,
        grid_size=int(data_cfg.get("grid_size", 4)),
    )

    _save_2d_dataset_visualizations(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        num_classes=num_classes,
        out_dir=output_dir,
        title_prefix="Random checkerboard",
        output_name="random_checkerboard_dataset_samples.png",
    )

    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    class_to_indices = build_class_index_map(y_train_arr, num_classes)

    return DatasetBundle(
        train_inputs=x_train_arr,
        train_targets=y_train_arr,
        test_inputs=x_test_arr,
        test_targets=y_test_arr,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=(2,),
    )


def _load_dartboard_bundle(data_cfg: dict, output_dir: Path) -> "DatasetBundle":
    from generate_dataset import (
        DatasetBundle,
        build_class_index_map,
        _save_2d_dataset_visualizations,
    )
 
    _validate_dartboard_config(data_cfg)
 
    num_classes      = int(data_cfg.get("num_classes",    2))
    points_per_class = int(data_cfg.get("points_per_class", 1000))
    random_seed      = int(data_cfg.get("random_seed",    0))
 
    shared_kwargs = dict(
        num_classes      = num_classes,
        num_rings        = int(data_cfg.get("num_rings",      4)),
        num_sectors      = int(data_cfg.get("num_sectors",    8)),
        noise_std        = float(data_cfg.get("noise_std",    0.0)),
        inner_radius     = float(data_cfg.get("inner_radius", 0.05)),
        outer_radius     = float(data_cfg.get("outer_radius", 1.0)),
        points_per_class = points_per_class,
    )
 
    x_train, y_train = generate_dartboard(random_seed=random_seed,     **shared_kwargs)
    x_test,  y_test  = generate_dartboard(random_seed=random_seed + 1, **shared_kwargs)
 
    _save_2d_dataset_visualizations(
        x_train=x_train, y_train=y_train,
        x_test=x_test,   y_test=y_test,
        num_classes=num_classes,
        out_dir=output_dir,
        title_prefix="Dartboard",
        output_name="dartboard_dataset_samples.png",
    )
 
    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr  = np.asarray(x_test,  dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    y_test_arr  = np.asarray(y_test,  dtype=np.int64)
 
    return DatasetBundle(
        train_inputs     = x_train_arr,
        train_targets    = y_train_arr,
        test_inputs      = x_test_arr,
        test_targets     = y_test_arr,
        class_to_indices = build_class_index_map(y_train_arr, num_classes),
        num_classes      = num_classes,
        input_shape      = (2,),
    )


def build_dataset_bundle(config: dict, output_dir: Path) -> DatasetBundle:
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset_name", "spiral")).strip().lower()
    if dataset_name == "mnist":
        return _load_mnist_bundle(data_cfg, output_dir)
    if dataset_name == "spiral":
        return _load_spiral_bundle(data_cfg, output_dir)
    if dataset_name == "gaussian_blobs":
        return _load_gaussian_blobs_bundle(data_cfg, output_dir)
    if dataset_name == "blobs":
        return _load_blobs_bundle(data_cfg, output_dir)
    if dataset_name == "rings":
        return _load_rings_bundle(data_cfg, output_dir)
    if dataset_name == "checkerboard":
        return _load_checkerboard_bundle(data_cfg, output_dir)
    if dataset_name == "random_checkerboard":
        return _load_random_checkerboard_bundle(data_cfg, output_dir)
    if dataset_name == "dartboard":
        return _load_dartboard_bundle(data_cfg, output_dir)
    raise ValueError(
        "Unsupported dataset_name "
        f"'{dataset_name}'. Expected 'mnist', 'spiral', 'gaussian_blobs', 'blobs', 'rings', 'checkerboard', 'random_checkerboard', or 'dartboard'."
    )