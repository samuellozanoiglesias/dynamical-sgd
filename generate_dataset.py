from __future__ import annotations

import gzip
import pickle
import shutil
import struct
import tarfile
import urllib.request
import zipfile
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

# Official CIFAR mirrors (Alex Krizhevsky / U. Toronto).
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

# Full ImageNet (1.28M images, ~150GB, requires a license/account) is not
# practical to auto-download inside a script. Tiny-ImageNet-200 is a real,
# freely-downloadable, ImageNet-derived benchmark (200 classes, 64x64,
# images drawn directly from ImageNet) and is used here as the
# "imagenet"-family entry point.
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2770, 0.2691, 0.2821)


@dataclass
class DatasetBundle:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    class_to_indices: Dict[int, np.ndarray]
    num_classes: int
    input_shape: tuple[int, ...]


def _wants_metaclass_remap(data_cfg: dict) -> bool:
    """True if this data_cfg will remap labels to metaclasses after loading
    (via _maybe_apply_metaclass_mapping) -- in which case data.num_classes
    refers to the FINAL metaclass count, not the dataset's native label
    count, so loaders must not validate it against their native count."""
    return data_cfg.get("metaclass_mapping") is not None or bool(
        data_cfg.get("use_default_metaclass_mapping", False)
    )


def _resolve_native_num_classes(data_cfg: dict, native_count: int, dataset_label: str) -> int:
    """Returns the native (pre-metaclass) class count a loader should use,
    validating data.num_classes against it -- unless a metaclass mapping is
    configured, in which case data.num_classes is the post-remap count and
    this check is skipped (build_dataset_bundle validates the post-remap
    count separately once the mapping has been applied)."""
    if _wants_metaclass_remap(data_cfg):
        return native_count
    num_classes = int(data_cfg.get("num_classes", native_count))
    if num_classes != native_count:
        raise ValueError(
            f"{dataset_label} loader expects data.num_classes={native_count} "
            "(its native label count). To train on fewer coarse labels, use "
            "data.metaclass_mapping (or data.use_default_metaclass_mapping) "
            "instead of setting data.num_classes directly."
        )
    return num_classes


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


def _save_3d_dataset_visualizations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    out_dir: Path,
    title_prefix: str,
    output_name: str,
) -> None:
    """3D counterpart of `_save_2d_dataset_visualizations`. Plots the first
    three columns of x_train/x_test as a 3D scatter (train | test side by
    side), color-coded by class. Requires x_train.shape[1] >= 3 -- callers
    are expected to have validated `plot_dim` against `reduce_dim` already
    (see `_resolve_plot_dim`)."""
    # Local import: mpl_toolkits.mplot3d registers the '3d' projection as a
    # side effect of being imported. Newer matplotlib versions auto-register
    # it, but importing explicitly keeps this working on older versions too.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14, 6))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    cmap = plt.get_cmap("tab10")

    for class_id in range(num_classes):
        mask = y_train == class_id
        if np.any(mask):
            ax0.scatter(
                x_train[mask, 0],
                x_train[mask, 1],
                x_train[mask, 2],
                s=12,
                alpha=0.9,
                color=cmap(class_id % 10),
                edgecolors="black",
                linewidths=0.2,
                label=f"Class {class_id}",
            )
    ax0.set_title(f"{title_prefix} Training Samples")
    ax0.legend(ncol=2, fontsize=8)

    for class_id in range(num_classes):
        mask = y_test == class_id
        if np.any(mask):
            ax1.scatter(
                x_test[mask, 0],
                x_test[mask, 1],
                x_test[mask, 2],
                s=12,
                alpha=0.9,
                color=cmap(class_id % 10),
                edgecolors="black",
                linewidths=0.2,
                label=f"Class {class_id}",
            )
    ax1.set_title(f"{title_prefix} Test Samples")
    ax1.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / output_name, dpi=150, bbox_inches="tight")
    plt.close()


def _resolve_plot_dim(data_cfg: dict, reduce_dim: int) -> int:
    """Resolves data.plot_dim (2 or 3) -- the dimensionality actually drawn
    in the scatter plot -- defaulting to a 2D plot (the original, unchanged
    behavior) unless the user explicitly asks for 3D. plot_dim can never
    exceed reduce_dim (you can't plot a 3rd axis that doesn't exist)."""
    plot_dim = int(data_cfg.get("plot_dim", 2))
    if plot_dim not in (2, 3):
        raise ValueError(f"data.plot_dim must be 2 or 3, got {plot_dim}.")
    if plot_dim > reduce_dim:
        raise ValueError(
            f"data.plot_dim ({plot_dim}) cannot exceed data.reduce_dim ({reduce_dim}). "
            "Use reduce_dim >= 3 to enable a 3D plot."
        )
    return plot_dim


def _reduced_plot_output_name(
    dataset_label: str, reduce_method: str, reduce_dim: int, plot_dim: int
) -> str:
    """Shared filename convention for reduced-dim scatter plots. 2D plots
    keep the original (pre-3D-support) filename unchanged; 3D plots get an
    explicit '_3d' suffix so the two never collide/overwrite each other."""
    suffix = "_3d" if plot_dim == 3 else ""
    return f"{dataset_label.lower()}_reduced_{reduce_method}_dim{reduce_dim}_samples{suffix}.png"


def _save_reduced_dataset_visualizations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    out_dir: Path,
    title_prefix: str,
    output_name: str,
    plot_dim: int = 2,
) -> None:
    """Dispatches to the 2D or 3D scatter-plot saver based on `plot_dim`."""
    if plot_dim == 3:
        _save_3d_dataset_visualizations(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            num_classes=num_classes, out_dir=out_dir,
            title_prefix=title_prefix, output_name=output_name,
        )
    else:
        _save_2d_dataset_visualizations(
            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
            num_classes=num_classes, out_dir=out_dir,
            title_prefix=title_prefix, output_name=output_name,
        )


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

    num_classes = _resolve_native_num_classes(data_cfg, 10, "MNIST")

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

def _ensure_and_extract_archive(url: str, data_dir: Path, marker_dir: Path) -> None:
    """Finds a local archive copy (tar/zip) or downloads it if missing, then extracts it."""
    if marker_dir.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    default_name = Path(url).name
    
    # Build a list of potential local archive file matches
    candidates = [data_dir / default_name]
    if "cifar-100" in url:
        # Gracefully account for your local ".tar.zip" naming variation
        candidates.append(data_dir / "cifar-100-python.tar.zip")
        candidates.append(data_dir / "cifar-100-python.zip")

    archive_path = None
    for candidate in candidates:
        if candidate.exists():
            if tarfile.is_tarfile(candidate) or zipfile.is_zipfile(candidate):
                archive_path = candidate
                break
            else:
                # Discard corrupted/broken incomplete files from a previous run
                candidate.unlink()

    # If no valid local archive file exists, fallback to downloading it
    if archive_path is None:
        archive_path = data_dir / default_name
        tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(response, dst)

            if not tarfile.is_tarfile(tmp_path) and not zipfile.is_zipfile(tmp_path):
                raise IOError(f"Downloaded file from {url} is not a valid archive format.")
            tmp_path.rename(archive_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    # Dynamic extraction based on the detected file type
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(data_dir)
    else:
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(data_dir)
        except tarfile.ReadError:
            # Fallback if tar compression isn't strictly gzip-encoded
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(data_dir)


def _download_and_extract_zip(url: str, data_dir: Path, marker_dir: Path) -> None:
    """Download `url` into data_dir and extract it, unless marker_dir already exists."""
    if marker_dir.exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / Path(url).name

    if archive_path.exists() and not zipfile.is_zipfile(archive_path):
        archive_path.unlink()

    if not archive_path.exists():
        tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")
        try:
            # FIX: Use urllib.request.Request to supply a standard browser User-Agent header
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(response, dst)

            if not zipfile.is_zipfile(tmp_path):
                raise IOError(f"Downloaded file from {url} is not a valid zip archive (download likely failed).")
            tmp_path.rename(archive_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(data_dir)


def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _normalize_image_batch(images_u8: np.ndarray, mean: tuple, std: tuple) -> np.ndarray:
    """images_u8: (N, C, H, W) uint8 -> normalized float32."""
    x = images_u8.astype(np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return (x - mean_arr) / std_arr


# ---------------------------------------------------------------------------
# Optional low-dimensional embedding for image datasets (currently wired up
# for CIFAR-10 only, via data.reduce_dim / data.reduce_method / data.reduce_source).
#
# Motivation: the "stuck without oscillation" failure mode originally seen on
# the 2D spiral/rings + width-50 single-hidden-layer MLP setup has three
# ingredients that raw-pixel CIFAR-10 training doesn't reproduce even with a
# narrow bottleneck: (1) a genuinely narrow *shared* bottleneck every class's
# gradient must fight over, (2) a topologically entangled decision boundary
# that's a property of a *low-dimensional* input geometry (curled-together,
# NOT linearly separable classes), and (3) no landscape-smoothing mechanism
# (BatchNorm, skip connections). Raw CIFAR-10 pixels live in ~3072-D, which
# gives even a narrow model enough near-orthogonal directions at init that
# classes rarely have to destructively interfere the way they do when
# squeezed through a literal 2D bottleneck.
#
# Two independent knobs control the embedding:
#
#   data.reduce_source -- which feature space the reduction algorithm sees:
#     "pixels"        -- raw normalized pixels, flattened (default for
#                         pca/random_projection).
#     "frozen_cnn"     -- pooled features of a small, NEVER-TRAINED,
#                         randomly-initialized conv net (fixed by
#                         data.reduce_seed).
#     "pretrained_cnn" -- pooled features of a small conv net given a SHORT
#                         supervised pretraining run (data.reduce_pretrain_epochs,
#                         default 2) and then frozen (default for umap/tsne).
#                         Enough training for the features to be
#                         discriminative; deliberately not trained to
#                         convergence, since a fully-trained/collapsed
#                         backbone would hand the classes to you already
#                         linearly separable -- defeating the point.
#     "pretrained_checkpoint" -- pooled (pre-classifier) features of a REAL,
#                         already-trained CNN checkpoint (e.g. model_weights.pt
#                         saved by a previous run_training call with
#                         architecture: cnn / resnet18_cnn on the native
#                         10-way labels). The backbone is loaded and used
#                         frozen, purely as a feature extractor -- no
#                         training happens here. Configure it via
#                         data.reduce_cnn (backbone, checkpoint,
#                         pretrained_num_classes, plus that backbone's usual
#                         architecture knobs -- see _pretrained_checkpoint_features
#                         below). Combine with data.metaclass_mapping to get
#                         "features from a CNN trained on the 10 original
#                         classes, then grouped into a handful of metaclasses,
#                         then reduced to 2-3D and plotted/trained on" --
#                         reduction runs BEFORE the metaclass remap (below),
#                         so the embedding reflects the finer 10-class feature
#                         geometry, only the point *labels/colors* change to
#                         the coarser metaclasses.
#
#   data.reduce_method -- the actual dimensionality-reduction algorithm:
#     "pca"              -- linear, preserves global variance. Fast, but on
#                            CIFAR-10 pixels this tends to produce OVERLAPPING
#                            class clouds in 2-3D (classes aren't separable
#                            at all, let alone entangled-but-separable like
#                            spiral/rings) because PCA optimizes for variance,
#                            not class structure.
#     "random_projection" -- linear, same overlap caveat as PCA, useful
#                            mainly as a cheap baseline / sanity check.
#     "umap"              -- nonlinear, preserves local neighborhood/manifold
#                            structure, so same-class points tend to form
#                            distinct (often curled / non-convex) regions
#                            rather than overlapping blobs -- closer to the
#                            "separable but not linearly separable" character
#                            of spiral/rings. Has a real out-of-sample
#                            .transform(), so train/test stay a clean fit/
#                            transform split.
#     "tsne"              -- nonlinear, similar visual character to UMAP.
#                            scikit-learn's TSNE has NO out-of-sample
#                            transform, so this is fit transductively on
#                            train+test concatenated together, then split
#                            back apart -- fine for building a fixed dataset
#                            like this, not meant for genuine held-out
#                            generalization. O(n log n) (barnes_hut, only
#                            for reduce_dim<=3) or O(n^2) (exact, needed for
#                            reduce_dim>3) in the number of points, so use
#                            data.reduce_max_train_samples /
#                            data.reduce_max_test_samples to subsample before
#                            fitting on full CIFAR-10 (60k points).
#
# The resulting DatasetBundle is tabular (input_shape == (reduce_dim,)), so
# it plugs directly into the existing `architecture: mlp` path in model.py
# with no further changes -- e.g. nn_width: 50, num_hidden_layers: 1
# recreates the exact "MLP-50" bottleneck from the synthetic-data
# experiments.
# ---------------------------------------------------------------------------

_VALID_REDUCE_METHODS = {"pca", "random_projection", "umap", "tsne"}
_VALID_REDUCE_SOURCES = {"pixels", "frozen_cnn", "pretrained_cnn", "pretrained_checkpoint"}


def _pca_fit_transform(
    train_flat: np.ndarray, test_flat: np.ndarray, reduce_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fits PCA on `train_flat` (N, D) via economy SVD and projects both
    splits onto the top `reduce_dim` principal components. No sklearn
    dependency -- this is the entire algorithm."""
    mean = train_flat.mean(axis=0, keepdims=True)
    centered = train_flat - mean
    # full_matrices=False -> economy SVD, cheap enough for CIFAR-10 (50k x D).
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:reduce_dim]  # (reduce_dim, D)
    train_embed = centered @ components.T
    test_embed = (test_flat - mean) @ components.T
    return train_embed, test_embed


def _random_projection_fit_transform(
    train_flat: np.ndarray, test_flat: np.ndarray, reduce_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed Gaussian random projection (Johnson-Lindenstrauss style),
    mean-centered on train stats. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    dim_in = train_flat.shape[1]
    proj = rng.normal(size=(dim_in, reduce_dim)) / np.sqrt(float(reduce_dim))
    mean = train_flat.mean(axis=0, keepdims=True)
    train_embed = (train_flat - mean) @ proj
    test_embed = (test_flat - mean) @ proj
    return train_embed, test_embed


def _frozen_cnn_features(
    x_train_img: np.ndarray, x_test_img: np.ndarray, seed: int, batch_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Runs both splits through a small, randomly-initialized, NEVER-TRAINED
    conv net (fixed by `seed`) and returns the flattened pooled features.
    This is the "frozen small conv backbone" embedding option -- meant to be
    followed by `_pca_fit_transform` (or a random projection) down to
    `reduce_dim`, giving a feature space that's already funneled through a
    few conv layers rather than raw pixels. Imports torch lazily so the
    plain 'pca' / 'random_projection' paths never require it."""
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    in_ch = int(x_train_img.shape[1])
    backbone = nn.Sequential(
        nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(4),
    ).eval()

    def _run(x_img: np.ndarray) -> np.ndarray:
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x_img.shape[0], batch_size):
                batch = torch.as_tensor(
                    x_img[start:start + batch_size], dtype=torch.float32
                )
                out = backbone(batch).flatten(1)
                feats.append(out.numpy())
        return np.concatenate(feats, axis=0)

    return _run(x_train_img), _run(x_test_img)


def _pretrained_cnn_features(
    x_train_img: np.ndarray,
    y_train_img: np.ndarray,
    x_test_img: np.ndarray,
    num_classes: int,
    seed: int,
    epochs: int = 2,
    lr: float = 1.0e-3,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Same small conv backbone as `_frozen_cnn_features`, but given a SHORT
    supervised pretraining run (default 2 epochs, real labels, Adam) before
    being frozen and used purely as a feature extractor. Deliberately not
    trained to convergence -- a couple epochs is enough for the pooled
    features to become discriminative (so UMAP/t-SNE have real class
    structure to preserve) without letting the network fully separate the
    classes on its own the way a converged/collapsed backbone would.
    Lazy torch import, like `_frozen_cnn_features`."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    in_ch = int(x_train_img.shape[1])

    backbone = nn.Sequential(
        nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(4),
    )
    classifier_head = nn.Linear(32 * 4 * 4, num_classes)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(classifier_head.parameters()), lr=lr
    )

    x_train_t = torch.as_tensor(x_train_img, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train_img, dtype=torch.long)
    n_train = x_train_t.shape[0]
    rng = np.random.default_rng(seed)

    backbone.train()
    classifier_head.train()
    for _ in range(max(1, int(epochs))):
        order = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            batch_x = x_train_t[idx]
            batch_y = y_train_t[idx]
            optimizer.zero_grad()
            feats = backbone(batch_x).flatten(1)
            logits = classifier_head(feats)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

    backbone.eval()

    def _run(x_img: np.ndarray) -> np.ndarray:
        out_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x_img.shape[0], batch_size):
                batch = torch.as_tensor(x_img[start:start + batch_size], dtype=torch.float32)
                out_parts.append(backbone(batch).flatten(1).numpy())
        return np.concatenate(out_parts, axis=0)

    return _run(x_train_img), _run(x_test_img)


def _pretrained_checkpoint_features(
    x_train_img: np.ndarray,
    x_test_img: np.ndarray,
    cnn_cfg: dict,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs both splits through a REAL CNN backbone (resnet18 / simple_cnn /
    myrtle -- the exact same builders model.py's `architecture: cnn` /
    `resnet18_cnn` path uses) with weights loaded from an already-trained
    checkpoint (a model_weights.pt saved by a previous run_training call),
    and returns the FROZEN pre-classifier pooled features. No training
    happens here -- this is pure feature extraction, unlike
    `_pretrained_cnn_features` above which trains a small ad hoc net for a
    couple epochs from scratch. `cnn_cfg` (data.reduce_cnn in the config)
    uses the same keys as model.cnn in a training config:

      backbone               -- 'resnet18' | 'simple_cnn' | 'myrtle'
      checkpoint              -- path to the model_weights.pt to load (required)
      pretrained_num_classes -- class count the checkpoint's head was
                                 trained with (e.g. 10 for native CIFAR-10);
                                 only used to build a matching-shape
                                 classifier layer so load_state_dict lines up
                                 -- the classifier itself is discarded, only
                                 the pre-classifier pooled features are read.
      plus that backbone's usual architecture knobs (channels/kernel_size/
      use_batchnorm/pool_every_block/dropout/fc_hidden_dim for simple_cnn;
      resnet.blocks_per_stage/num_stages/width_mult for resnet18; myrtle.*
      for myrtle) -- these MUST match the checkpoint's original training
      config, exactly like model.cnn.pretrained_checkpoint's fine-tune path
      in training_runner.py.

    Lazy imports (torch + model_cnn), like `_frozen_cnn_features` /
    `_pretrained_cnn_features` above."""
    import torch
    from model_cnn import build_cnn_model, build_myrtle_cnn_model, build_simple_cnn_model

    checkpoint_path = cnn_cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "data.reduce_source == 'pretrained_checkpoint' requires "
            "data.reduce_cnn.checkpoint (path to a model_weights.pt saved by "
            "a previous training run)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_name = str(cnn_cfg.get("backbone", "simple_cnn")).strip().lower()
    pretrained_num_classes = int(cnn_cfg.get("pretrained_num_classes", 10))
    input_ch = int(x_train_img.shape[1])

    if backbone_name == "resnet18":
        resnet_cfg = cnn_cfg.get("resnet", {}) or {}
        model, _classifier, capture = build_cnn_model(
            num_classes=pretrained_num_classes, input_ch=input_ch, device=device,
            blocks_per_stage=tuple(
                int(b) for b in resnet_cfg.get("blocks_per_stage", [2, 2, 2, 2])
            ),
            num_stages=int(resnet_cfg.get("num_stages", 4)),
            width_mult=float(resnet_cfg.get("width_mult", 1.0)),
        )
    elif backbone_name == "simple_cnn":
        fc_hidden_dim_raw = cnn_cfg.get("fc_hidden_dim")
        model, _classifier, capture = build_simple_cnn_model(
            num_classes=pretrained_num_classes, input_ch=input_ch, device=device,
            channels=[int(c) for c in cnn_cfg.get("channels", [16, 32])],
            kernel_size=int(cnn_cfg.get("kernel_size", 3)),
            use_batchnorm=_as_bool_local(cnn_cfg.get("use_batchnorm", True)),
            pool_every_block=_as_bool_local(cnn_cfg.get("pool_every_block", True)),
            dropout=float(cnn_cfg.get("dropout", 0.0)),
            fc_hidden_dim=int(fc_hidden_dim_raw) if fc_hidden_dim_raw else None,
        )
    elif backbone_name == "myrtle":
        myrtle_cfg = cnn_cfg.get("myrtle", {}) or {}
        blocks_per_stage_raw = myrtle_cfg.get("blocks_per_stage")
        channels_raw = myrtle_cfg.get("channels")
        model, _classifier, capture = build_myrtle_cnn_model(
            num_classes=pretrained_num_classes, input_ch=input_ch, device=device,
            base_width=int(myrtle_cfg.get("base_width", 64)),
            num_stages=int(myrtle_cfg.get("num_stages", 4)),
            blocks_per_stage=(
                [int(b) for b in blocks_per_stage_raw] if blocks_per_stage_raw else None
            ),
            channels=[int(c) for c in channels_raw] if channels_raw else None,
            kernel_size=int(myrtle_cfg.get("kernel_size", 3)),
            use_batchnorm=_as_bool_local(myrtle_cfg.get("use_batchnorm", True)),
            pool_last_stage=_as_bool_local(myrtle_cfg.get("pool_last_stage", True)),
            dropout=float(myrtle_cfg.get("dropout", 0.0)),
        )
    else:
        raise ValueError(
            f"data.reduce_cnn.backbone '{backbone_name}' not recognized. "
            "Expected 'resnet18', 'simple_cnn', or 'myrtle'."
        )

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["torch_model"] if "torch_model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    def _run(x_img: np.ndarray) -> np.ndarray:
        feats: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x_img.shape[0], batch_size):
                batch = torch.as_tensor(
                    x_img[start:start + batch_size], dtype=torch.float32, device=device
                )
                model(batch)  # forward hook populates `capture.value` (pre-classifier features)
                feats.append(capture.value.reshape(batch.shape[0], -1).cpu().numpy())
        return np.concatenate(feats, axis=0)

    return _run(x_train_img), _run(x_test_img)


def _umap_fit_transform(
    train_feats: np.ndarray, test_feats: np.ndarray, reduce_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear, neighborhood-structure-preserving reduction. Real
    out-of-sample .transform(), so train/test stay a clean fit/transform
    split (unlike t-SNE below). Lazy import -- requires umap-learn."""
    import umap

    reducer = umap.UMAP(n_components=reduce_dim, random_state=seed)
    train_embed = reducer.fit_transform(train_feats)
    test_embed = reducer.transform(test_feats)
    return np.asarray(train_embed), np.asarray(test_embed)


def _tsne_fit_transform(
    train_feats: np.ndarray, test_feats: np.ndarray, reduce_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear reduction, fit TRANSDUCTIVELY on train+test concatenated
    together (sklearn's TSNE has no real out-of-sample transform), then
    split back apart by original split sizes. `method='barnes_hut'`
    (sklearn's default, O(n log n)) only supports reduce_dim <= 3; falls
    back to the O(n^2) `method='exact'` above that. Both are expensive at
    full CIFAR-10 scale (60k points combined) -- use
    data.reduce_max_train_samples / data.reduce_max_test_samples to
    subsample first. Lazy import -- requires scikit-learn."""
    from sklearn.manifold import TSNE

    n_train = train_feats.shape[0]
    combined = np.concatenate([train_feats, test_feats], axis=0)
    tsne_method = "barnes_hut" if reduce_dim <= 3 else "exact"
    # perplexity must be < n_samples; scale gently with dataset size but cap
    # it at the usual sane default of 30.
    perplexity = float(min(30.0, max(5.0, combined.shape[0] / 100.0)))

    tsne = TSNE(
        n_components=reduce_dim,
        random_state=seed,
        init="pca",
        method=tsne_method,
        perplexity=perplexity,
    )
    combined_embed = tsne.fit_transform(combined)
    return combined_embed[:n_train], combined_embed[n_train:]


def _stratified_subsample(
    x_img: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Subsamples (x_img, y) down to roughly `max_samples` points, spread as
    evenly as possible across classes. No-op if max_samples is None or
    already >= the current sample count. Used to keep UMAP/t-SNE (and, if
    ever needed, the other methods) tractable on full CIFAR-10."""
    if max_samples is None or x_img.shape[0] <= max_samples:
        return x_img, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, int(max_samples) // len(classes))
    idx_parts: list[np.ndarray] = []
    for class_id in classes:
        class_idx = np.where(y == class_id)[0]
        chosen = rng.choice(class_idx, size=min(per_class, class_idx.size), replace=False)
        idx_parts.append(chosen)
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return x_img[idx], y[idx]


def _fit_transform_dim_reduction(
    x_train_img: np.ndarray,
    x_test_img: np.ndarray,
    method: str,
    reduce_dim: int,
    seed: int,
    standardize: bool,
    source: str = "pixels",
    y_train_img: np.ndarray | None = None,
    num_classes: int | None = None,
    pretrain_epochs: int = 2,
    cnn_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-level entry point for the reduction. x_*_img are the already
    normalized (N, C, H, W) float32 image arrays; returns (N, reduce_dim)
    float32 embeddings for train/test.

    `source` selects the feature space the `method` operates on (see the big
    comment above this section); `pretrain_epochs` only applies when
    source == 'pretrained_cnn'; `cnn_cfg` (data.reduce_cnn) only applies when
    source == 'pretrained_checkpoint'."""
    if method not in _VALID_REDUCE_METHODS:
        raise ValueError(
            f"data.reduce_method '{method}' not recognized. "
            f"Expected one of {sorted(_VALID_REDUCE_METHODS)}."
        )
    if source not in _VALID_REDUCE_SOURCES:
        raise ValueError(
            f"data.reduce_source '{source}' not recognized. "
            f"Expected one of {sorted(_VALID_REDUCE_SOURCES)}."
        )
    if reduce_dim <= 0:
        raise ValueError(f"data.reduce_dim must be > 0, got {reduce_dim}.")

    if source == "pixels":
        n_train = x_train_img.shape[0]
        n_test = x_test_img.shape[0]
        train_feat = x_train_img.reshape(n_train, -1).astype(np.float64)
        test_feat = x_test_img.reshape(n_test, -1).astype(np.float64)
    elif source == "frozen_cnn":
        feats_train, feats_test = _frozen_cnn_features(x_train_img, x_test_img, seed=seed)
        train_feat = feats_train.astype(np.float64)
        test_feat = feats_test.astype(np.float64)
    elif source == "pretrained_checkpoint":
        feats_train, feats_test = _pretrained_checkpoint_features(
            x_train_img, x_test_img, cnn_cfg=cnn_cfg or {},
        )
        train_feat = feats_train.astype(np.float64)
        test_feat = feats_test.astype(np.float64)
    else:  # pretrained_cnn
        if y_train_img is None or num_classes is None:
            raise ValueError(
                "data.reduce_source == 'pretrained_cnn' requires train labels "
                "and num_classes to run the short supervised pretraining pass."
            )
        feats_train, feats_test = _pretrained_cnn_features(
            x_train_img, y_train_img, x_test_img,
            num_classes=num_classes, seed=seed, epochs=pretrain_epochs,
        )
        train_feat = feats_train.astype(np.float64)
        test_feat = feats_test.astype(np.float64)

    if train_feat.shape[1] < reduce_dim:
        raise ValueError(
            f"data.reduce_dim ({reduce_dim}) cannot exceed the "
            f"'{source}' feature dimensionality ({train_feat.shape[1]})."
        )

    if method == "pca":
        train_embed, test_embed = _pca_fit_transform(train_feat, test_feat, reduce_dim)
    elif method == "random_projection":
        train_embed, test_embed = _random_projection_fit_transform(
            train_feat, test_feat, reduce_dim, seed=seed
        )
    elif method == "umap":
        train_embed, test_embed = _umap_fit_transform(train_feat, test_feat, reduce_dim, seed)
    else:  # tsne
        train_embed, test_embed = _tsne_fit_transform(train_feat, test_feat, reduce_dim, seed)

    if standardize:
        # Rescale so the narrow MLP sees the same kind of unit-ish-variance
        # input the synthetic 2D generators (spiral/rings/blobs) already
        # produce -- otherwise a method/source combo that happens to produce
        # a larger-scale embedding could dominate purely by scale.
        std = train_embed.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        mean_e = train_embed.mean(axis=0, keepdims=True)
        train_embed = (train_embed - mean_e) / std
        test_embed = (test_embed - mean_e) / std

    return train_embed.astype(np.float32), test_embed.astype(np.float32)


def save_image_grid_visualizations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    out_dir: Path,
    title_prefix: str,
    output_prefix: str,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """Generic RGB/gray image-grid + class-distribution visualizer (CIFAR/Tiny-ImageNet)."""
    n_show = min(num_classes, 20)
    cols = 5
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    fig.suptitle(f"{title_prefix} Training Dataset Samples", fontsize=16)
    axes = np.atleast_2d(axes)
    for class_idx in range(rows * cols):
        row, col = divmod(class_idx, cols)
        ax = axes[row, col]
        if class_idx < n_show:
            idxs = np.where(y_train == class_idx)[0]
            if idxs.size > 0:
                img = np.transpose(x_train[idxs[0]], (1, 2, 0))
                img = (img - img.min()) / max(img.max() - img.min(), 1e-8)
                ax.imshow(img)
                label = class_names[class_idx] if class_names is not None else str(class_idx)
                ax.set_title(label, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{output_prefix}_dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.close()

    train_counts = np.bincount(y_train, minlength=num_classes)
    test_counts = np.bincount(y_test, minlength=num_classes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(np.arange(num_classes), train_counts, alpha=0.8, color="skyblue")
    ax1.set_title("Train Class Distribution")
    ax2.bar(np.arange(num_classes), test_counts, alpha=0.8, color="salmon")
    ax2.set_title("Test Class Distribution")
    for ax in (ax1, ax2):
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{output_prefix}_dataset_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()


def _load_cifar10_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    data_dir = Path(str(data_cfg.get("data_dir", "./data"))).resolve()
    extracted_dir = data_dir / "cifar-10-batches-py"
    _ensure_and_extract_archive(CIFAR10_URL, data_dir, extracted_dir)

    train_batches = [_unpickle(extracted_dir / f"data_batch_{i}") for i in range(1, 6)]
    train_data = np.concatenate([b[b"data"] for b in train_batches], axis=0)
    train_labels = np.concatenate([np.asarray(b[b"labels"]) for b in train_batches], axis=0)

    test_batch = _unpickle(extracted_dir / "test_batch")
    test_data = test_batch[b"data"]
    test_labels = np.asarray(test_batch[b"labels"])

    # Stored as (N, 3072) row-major R,G,B planes -> (N, 3, 32, 32).
    x_train = train_data.reshape(-1, 3, 32, 32)
    x_test = test_data.reshape(-1, 3, 32, 32)

    x_train = _normalize_image_batch(x_train, CIFAR10_MEAN, CIFAR10_STD)
    x_test = _normalize_image_batch(x_test, CIFAR10_MEAN, CIFAR10_STD)

    y_train = train_labels.astype(np.int64)
    y_test = test_labels.astype(np.int64)

    num_classes = _resolve_native_num_classes(data_cfg, 10, "CIFAR-10")

    meta = _unpickle(extracted_dir / "batches.meta")
    class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

    save_image_grid_visualizations(
        x_train, y_train, x_test, y_test, num_classes, output_dir,
        title_prefix="CIFAR-10", output_prefix="cifar10", class_names=class_names,
    )

    # Optional: collapse CIFAR-10 down to a low-dimensional (typically 2-3D)
    # embedding, turning it into a tabular dataset -- see the big comment
    # above _fit_transform_dim_reduction for why. Off by default (native
    # (3, 32, 32) image bundle, unchanged behavior) unless data.reduce_dim is set.
    reduce_dim_raw = data_cfg.get("reduce_dim")
    if reduce_dim_raw:
        reduce_dim = int(reduce_dim_raw)
        reduce_method = str(data_cfg.get("reduce_method", "pca")).strip().lower()
        # umap/tsne default to features from a lightly-pretrained backbone
        # (real class structure to preserve); pca/random_projection default
        # to raw pixels (their original, cheaper behavior).
        default_source = "pretrained_cnn" if reduce_method in {"umap", "tsne"} else "pixels"
        reduce_source = str(data_cfg.get("reduce_source", default_source)).strip().lower()
        reduce_seed = int(data_cfg.get("reduce_seed", 0))
        reduce_standardize = _as_bool_local(data_cfg.get("reduce_standardize", True))
        reduce_pretrain_epochs = int(data_cfg.get("reduce_pretrain_epochs", 2))
        reduce_cnn_cfg = data_cfg.get("reduce_cnn", {}) or {}

        # umap/tsne (especially tsne, which is transductive over train+test
        # combined) get expensive at full CIFAR-10 scale -- optionally
        # subsample first. No-ops (and defaults to None, i.e. full dataset)
        # unless explicitly set.
        max_train_raw = data_cfg.get("reduce_max_train_samples")
        max_test_raw = data_cfg.get("reduce_max_test_samples")
        max_train_samples = int(max_train_raw) if max_train_raw else None
        max_test_samples = int(max_test_raw) if max_test_raw else None
        x_train, y_train = _stratified_subsample(x_train, y_train, max_train_samples, reduce_seed)
        x_test, y_test = _stratified_subsample(x_test, y_test, max_test_samples, reduce_seed + 1)

        x_train, x_test = _fit_transform_dim_reduction(
            x_train, x_test,
            method=reduce_method, reduce_dim=reduce_dim,
            seed=reduce_seed, standardize=reduce_standardize,
            source=reduce_source, y_train_img=y_train, num_classes=num_classes,
            pretrain_epochs=reduce_pretrain_epochs, cnn_cfg=reduce_cnn_cfg,
        )

        if reduce_dim >= 2:
            plot_dim = _resolve_plot_dim(data_cfg, reduce_dim)
            _save_reduced_dataset_visualizations(
                x_train, y_train, x_test, y_test, num_classes, output_dir,
                title_prefix=f"CIFAR-10 -> {reduce_method} dim={reduce_dim}",
                output_name=_reduced_plot_output_name(
                    "cifar10", reduce_method, reduce_dim, plot_dim
                ),
                plot_dim=plot_dim,
            )

        input_shape: tuple[int, ...] = (reduce_dim,)
    else:
        input_shape = tuple(x_train.shape[1:])

    class_to_indices = build_class_index_map(y_train, num_classes)
    return DatasetBundle(
        train_inputs=x_train, train_targets=y_train,
        test_inputs=x_test, test_targets=y_test,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=input_shape,
    )


def _load_cifar100_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    data_dir = Path(str(data_cfg.get("data_dir", "./data"))).resolve()
    extracted_dir = data_dir / "cifar-100-python"
    _ensure_and_extract_archive(CIFAR100_URL, data_dir, extracted_dir)

    label_mode = str(data_cfg.get("label_mode", "fine")).strip().lower()
    if label_mode not in {"fine", "coarse"}:
        raise ValueError("For CIFAR-100, data.label_mode must be 'fine' (100 classes) or 'coarse' (20 classes).")
    label_key = b"fine_labels" if label_mode == "fine" else b"coarse_labels"

    train_batch = _unpickle(extracted_dir / "train")
    test_batch = _unpickle(extracted_dir / "test")

    x_train = train_batch[b"data"].reshape(-1, 3, 32, 32)
    x_test = test_batch[b"data"].reshape(-1, 3, 32, 32)
    x_train = _normalize_image_batch(x_train, CIFAR100_MEAN, CIFAR100_STD)
    x_test = _normalize_image_batch(x_test, CIFAR100_MEAN, CIFAR100_STD)

    y_train = np.asarray(train_batch[label_key], dtype=np.int64)
    y_test = np.asarray(test_batch[label_key], dtype=np.int64)

    default_num_classes = 100 if label_mode == "fine" else 20
    num_classes = _resolve_native_num_classes(
        data_cfg, default_num_classes, f"CIFAR-100 (label_mode='{label_mode}')"
    )

    meta = _unpickle(extracted_dir / "meta")
    meta_key = b"fine_label_names" if label_mode == "fine" else b"coarse_label_names"
    class_names = [name.decode("utf-8") for name in meta[meta_key]]

    save_image_grid_visualizations(
        x_train, y_train, x_test, y_test, num_classes, output_dir,
        title_prefix=f"CIFAR-100 ({label_mode})", output_prefix=f"cifar100_{label_mode}",
        class_names=class_names,
    )

    class_to_indices = build_class_index_map(y_train, num_classes)
    return DatasetBundle(
        train_inputs=x_train, train_targets=y_train,
        test_inputs=x_test, test_targets=y_test,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=tuple(x_train.shape[1:]),
    )


def _load_tiny_imagenet_bundle(data_cfg: dict, output_dir: Path) -> DatasetBundle:
    from PIL import Image  # local import: only needed for this dataset

    data_dir = Path(str(data_cfg.get("data_dir", "./data"))).resolve()
    extracted_dir = data_dir / "tiny-imagenet-200"
    _download_and_extract_zip(TINY_IMAGENET_URL, data_dir, extracted_dir)

    wnids = sorted((extracted_dir / "train").iterdir())
    wnids = [p.name for p in wnids if p.is_dir()]
    wnid_to_class = {wnid: idx for idx, wnid in enumerate(wnids)}
    num_classes = len(wnids)

    if not _wants_metaclass_remap(data_cfg):
        configured_num_classes = int(data_cfg.get("num_classes", num_classes))
        if configured_num_classes != num_classes:
            raise ValueError(f"Tiny-ImageNet loader found {num_classes} classes on disk; data.num_classes must match.")

    def _load_image(path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    train_images: list[np.ndarray] = []
    train_labels: list[int] = []
    for wnid in wnids:
        img_dir = extracted_dir / "train" / wnid / "images"
        for img_path in sorted(img_dir.glob("*.JPEG")):
            train_images.append(_load_image(img_path))
            train_labels.append(wnid_to_class[wnid])

    val_dir = extracted_dir / "val"
    val_annotations = {}
    with open(val_dir / "val_annotations.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            val_annotations[parts[0]] = parts[1]

    test_images: list[np.ndarray] = []
    test_labels: list[int] = []
    for img_name, wnid in val_annotations.items():
        if wnid not in wnid_to_class:
            continue
        img_path = val_dir / "images" / img_name
        test_images.append(_load_image(img_path))
        test_labels.append(wnid_to_class[wnid])

    x_train = np.transpose(np.stack(train_images, axis=0), (0, 3, 1, 2))
    x_test = np.transpose(np.stack(test_images, axis=0), (0, 3, 1, 2))
    x_train = _normalize_image_batch(x_train, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
    x_test = _normalize_image_batch(x_test, TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)

    y_train = np.asarray(train_labels, dtype=np.int64)
    y_test = np.asarray(test_labels, dtype=np.int64)

    save_image_grid_visualizations(
        x_train, y_train, x_test, y_test, num_classes, output_dir,
        title_prefix="Tiny-ImageNet-200", output_prefix="tiny_imagenet",
    )

    class_to_indices = build_class_index_map(y_train, num_classes)
    return DatasetBundle(
        train_inputs=x_train, train_targets=y_train,
        test_inputs=x_test, test_targets=y_test,
        class_to_indices=class_to_indices,
        num_classes=num_classes,
        input_shape=tuple(x_train.shape[1:]),
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


# ---------------------------------------------------------------------------
# Metaclass / superclass relabeling.
#
# Groups an existing dataset's original labels into a smaller number of
# "metaclasses" (e.g. CIFAR-10's 10 object classes -> 3 coarse groups), while
# reusing every original image. This lets you ask: does a network trained (or
# fine-tuned) on the *coarse* labels rediscover the finer original classes as
# hidden geometric substructure (e.g. cylinders) inside each metaclass, even
# though it never saw the fine labels?
# ---------------------------------------------------------------------------

# CIFAR-10 label order (fixed by the dataset): 0 airplane, 1 automobile,
# 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck.
# Default 3-way grouping used when data.metaclass_mapping is not given for
# CIFAR-10: {vehicles} / {large mammals} / {bird, frog}. This is a reasonable
# starting point, not a "correct" answer -- override via
# data.metaclass_mapping in the config to try a different grouping.
DEFAULT_CIFAR10_METACLASS_MAPPING = [
    [0, 1, 8, 9],   # vehicles: airplane, automobile, ship, truck
    [3, 4, 5, 7],   # mammals: cat, deer, dog, horse
    [2, 6],         # bird, frog
]


def _resolve_metaclass_mapping(raw_mapping: object, num_original_classes: int) -> list[list[int]]:
    """Parses data.metaclass_mapping into a validated list-of-lists of
    original class indices, one sublist per metaclass. Every original class
    index in [0, num_original_classes) must appear in exactly one sublist."""
    if isinstance(raw_mapping, str):
        import yaml as _yaml
        raw_mapping = _yaml.safe_load(raw_mapping)
    if not isinstance(raw_mapping, (list, tuple)) or len(raw_mapping) == 0:
        raise ValueError(
            "data.metaclass_mapping must be a non-empty list of lists of original "
            "class indices, one list per metaclass."
        )

    mapping: list[list[int]] = [[int(c) for c in group] for group in raw_mapping]

    seen: dict[int, int] = {}
    for metaclass_id, group in enumerate(mapping):
        if not group:
            raise ValueError(f"data.metaclass_mapping group {metaclass_id} is empty.")
        for original_class in group:
            if original_class in seen:
                raise ValueError(
                    f"data.metaclass_mapping: original class {original_class} appears in "
                    f"both metaclass {seen[original_class]} and metaclass {metaclass_id}."
                )
            seen[original_class] = metaclass_id

    expected = set(range(num_original_classes))
    if set(seen.keys()) != expected:
        missing = sorted(expected - set(seen.keys()))
        extra = sorted(set(seen.keys()) - expected)
        raise ValueError(
            "data.metaclass_mapping must cover every original class exactly once. "
            f"Missing: {missing}. Unexpected: {extra}."
        )

    return mapping


def _as_bool_local(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def remap_to_metaclasses(
    bundle: "DatasetBundle",
    mapping: list[list[int]],
    output_dir: Path,
    dataset_label: str,
    data_cfg: dict | None = None,
) -> "DatasetBundle":
    """Relabels `bundle` (any DatasetBundle, image or tabular) by grouping its
    original integer labels according to `mapping`. Original images/features,
    ordering, and train/test split are all untouched -- only the label ints
    are remapped from original-class-id -> metaclass-id."""
    num_metaclasses = len(mapping)
    lookup = np.full(bundle.num_classes, fill_value=-1, dtype=np.int64)
    for metaclass_id, group in enumerate(mapping):
        for original_class in group:
            lookup[original_class] = metaclass_id

    y_train_meta = lookup[bundle.train_targets]
    y_test_meta = lookup[bundle.test_targets]

    if len(bundle.input_shape) == 3:
        meta_names = [f"meta_{i}_" + "+".join(str(c) for c in group) for i, group in enumerate(mapping)]
        save_image_grid_visualizations(
            bundle.train_inputs, y_train_meta, bundle.test_inputs, y_test_meta,
            num_metaclasses, output_dir,
            title_prefix=f"{dataset_label} (metaclasses)",
            output_prefix=f"{dataset_label.lower()}_metaclasses",
            class_names=meta_names,
        )
    elif len(bundle.input_shape) == 1 and data_cfg is not None:
        # Tabular bundle -- this is the post-dim-reduction case (PCA/UMAP/
        # t-SNE on CIFAR-10 etc, see _load_cifar10_bundle). The reduced-dim
        # scatter plot was already saved by the loader using the ORIGINAL
        # (pre-metaclass) labels, since reduction happens before this
        # remapping step runs. Regenerate it here with the metaclass labels,
        # overwriting the same file, so the plot the user looks at matches
        # what the model is actually trained on.
        reduce_dim_raw = data_cfg.get("reduce_dim")
        if reduce_dim_raw:
            reduce_dim = int(reduce_dim_raw)
            reduce_method = str(data_cfg.get("reduce_method", "pca")).strip().lower()
            if reduce_dim >= 2:
                plot_dim = _resolve_plot_dim(data_cfg, reduce_dim)
                _save_reduced_dataset_visualizations(
                    bundle.train_inputs, y_train_meta, bundle.test_inputs, y_test_meta,
                    num_metaclasses, output_dir,
                    title_prefix=f"{dataset_label} -> {reduce_method} dim={reduce_dim} (metaclasses)",
                    output_name=_reduced_plot_output_name(
                        dataset_label, reduce_method, reduce_dim, plot_dim
                    ),
                    plot_dim=plot_dim,
                )

    return DatasetBundle(
        train_inputs=bundle.train_inputs,
        train_targets=y_train_meta,
        test_inputs=bundle.test_inputs,
        test_targets=y_test_meta,
        class_to_indices=build_class_index_map(y_train_meta, num_metaclasses),
        num_classes=num_metaclasses,
        input_shape=bundle.input_shape,
    )


def _maybe_apply_metaclass_mapping(
    bundle: "DatasetBundle", data_cfg: dict, output_dir: Path, dataset_label: str,
) -> "DatasetBundle":
    raw_mapping = data_cfg.get("metaclass_mapping")
    use_default = _as_bool_local(data_cfg.get("use_default_metaclass_mapping", False))

    if raw_mapping is None and not use_default:
        return bundle

    if raw_mapping is None:
        if dataset_label.lower() != "cifar10":
            raise ValueError(
                "data.use_default_metaclass_mapping=true only has a built-in default for "
                "cifar10. Provide data.metaclass_mapping explicitly for other datasets."
            )
        mapping = DEFAULT_CIFAR10_METACLASS_MAPPING
    else:
        mapping = _resolve_metaclass_mapping(raw_mapping, bundle.num_classes)

    return remap_to_metaclasses(bundle, mapping, output_dir, dataset_label, data_cfg=data_cfg)


def build_dataset_bundle(config: dict, output_dir: Path) -> DatasetBundle:
    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset_name", "spiral")).strip().lower()
    if dataset_name == "mnist":
        bundle = _load_mnist_bundle(data_cfg, output_dir)
        return _maybe_apply_metaclass_mapping(bundle, data_cfg, output_dir, "mnist")
    if dataset_name == "cifar10":
        bundle = _load_cifar10_bundle(data_cfg, output_dir)
        return _maybe_apply_metaclass_mapping(bundle, data_cfg, output_dir, "cifar10")
    if dataset_name == "cifar100":
        bundle = _load_cifar100_bundle(data_cfg, output_dir)
        return _maybe_apply_metaclass_mapping(bundle, data_cfg, output_dir, "cifar100")
    if dataset_name == "tiny_imagenet":
        bundle = _load_tiny_imagenet_bundle(data_cfg, output_dir)
        return _maybe_apply_metaclass_mapping(bundle, data_cfg, output_dir, "tiny_imagenet")
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
        f"'{dataset_name}'. Expected 'mnist', 'cifar10', 'cifar100', 'tiny_imagenet', 'spiral', "
        "'gaussian_blobs', 'blobs', 'rings', 'checkerboard', 'random_checkerboard', or 'dartboard'."
    )