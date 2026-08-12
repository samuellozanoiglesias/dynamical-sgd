"""PCA_explained_within_class.py
=====================================
Within-class-only variant of PCA_explained_from_checkpoint.py.

Identical pipeline (load run_dir -> rebuild model from config.yaml -> load
model_weights.pt -> extract pre-classifier features on the FINAL checkpoint),
but before running PCA it subtracts each sample's OWN CLASS MEAN from its
feature vector -- the same deflation used in the path-curvature / NC1-NC2
metrics -- so the resulting explained-variance curve reflects pure
within-class shape, with the between-class (inter-cluster) variance removed.

Why this matters for comparing against participation ratio
------------------------------------------------------------
The global (non-deflated) PCA curve mixes two sources of variance:
    (1) how spread out the K class clusters are from each other
        (between-class variance), and
    (2) how "fat" each individual class cluster is
        (within-class variance).
k95 / k99 computed on that mixed curve conflate both effects, so they can't
be compared cleanly against a within-class quantity like the participation
ratio (which by construction is a within-class-shape statistic once you're
computing it off deflated features, as in the path-curvature pipeline).
Subtracting each sample's class mean first removes source (1) entirely,
leaving a curve -- and therefore a k95/k99 -- that is directly comparable to
the participation ratio gap.

USAGE
-----
    python PCA_explained_within_class.py \\
        /mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/\\
cifar10_metaclasses_scratch_intuitive/training_2026_07_23-17_04_15/

Same optional flags as the global-PCA script (--split, --num-samples,
--data-root, --metaclass-mapping[-preset], --dataset-bundle, --device,
--seed). Nothing about model loading / data loading changed -- only the
centering step immediately before the SVD, and the output filename/title.

OUTPUT
------
    <run_dir>/pca_within_class_explained_variance.png

Also prints the participation ratio of the (global, pre-deflation) feature
covariance next to k95/k99 of the deflated curve, since that's the
comparison this script exists to enable. Participation ratio here is defined
the standard way, PR = (sum lambda_i)^2 / sum(lambda_i^2), computed from the
same features BEFORE class-mean subtraction (i.e. on the full pooled
feature matrix) -- if you want the within-class PR instead, that's already
what your deflation-based metrics_for_multiple_classes.py path computes, and
should match closely with the k95/k99 reported here.

CONFIG FORMAT / FINETUNE-CURRICULUM / DATA SOURCE
--------------------------------------------------
Unchanged from PCA_explained_from_checkpoint.py -- see that script's
docstring for full details on config.yaml parsing, the finetune-checkpoint
head-swap subtlety, and the dataset_bundle.npz vs. raw-CIFAR10 fallback.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to parse config.yaml. Install with: "
        "pip install pyyaml --break-system-packages"
    ) from exc

from model_cnn import (
    build_cnn_model,
    build_simple_cnn_model,
    TorchModelAdapter,
)


# ---------------------------------------------------------------------------
# Known metaclass mappings from conversation (used only as a fallback if
# config.txt doesn't specify one explicitly -- see --metaclass-mapping).
# ---------------------------------------------------------------------------

METACLASS_PRESETS: dict[str, list[list[int]]] = {
    "intuitive": [
        [0, 1, 8, 9],       # vehicles: airplane, automobile, ship, truck
        [3, 4, 5, 7],       # large mammals: cat, deer, dog, horse
        [2, 6],             # bird + frog
    ],
    "alternative": [
        [0, 6, 3],          # airplane, frog, cat
        [1, 4, 2],          # automobile, deer, bird
        [8, 9, 5, 7],       # ship, truck, dog, horse
    ],
}


# ---------------------------------------------------------------------------
# config.yaml / config.txt parsing
# ---------------------------------------------------------------------------

def _load_config(run_dir: Path) -> dict:
    """Looks for config.yaml first (the real format), falls back to
    config.txt (flat `key = value` / `key: value` lines, or a JSON /
    Python-literal dump) if that's what's actually present instead."""
    yaml_path = run_dir / "config.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"{yaml_path} did not parse to a dict (got {type(config)})")
        return config

    txt_path = run_dir / "config.txt"
    if txt_path.exists():
        return _parse_flat_config_txt(txt_path)

    raise FileNotFoundError(f"Neither config.yaml nor config.txt found in {run_dir}")


def _parse_flat_config_txt(config_path: Path) -> dict:
    raw = config_path.read_text(encoding="utf-8")

    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    config: dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
        elif ":" in line:
            key, _, value = line.partition(":")
        else:
            continue
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
            continue
        try:
            config[key] = ast.literal_eval(value)
        except Exception:
            config[key] = value
    return config


def _nget(config: dict, dotted_path: str, default: Any, warn_defaults: list[str]) -> Any:
    """Nested get, e.g. _nget(config, 'model.cnn.resnet.width_mult', 1.0, warned)."""
    node: Any = config
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node or node[part] is None:
            warn_defaults.append(f"{dotted_path} -> defaulting to {default!r}")
            return default
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Model reconstruction from config.yaml
# ---------------------------------------------------------------------------

def determine_input_ch_and_dim(
    input_shape: tuple[int, ...], config: dict, warned: list[str]
) -> tuple[int, int | None]:
    """Mirrors training_runner.py exactly:

        is_tabular_input = len(dataset_bundle.input_shape) == 1
        if is_tabular_input:
            input_ch = int(cnn_cfg.get("stem_channels", 1))
            input_dim = int(dataset_bundle.input_shape[0])
        else:
            input_ch = int(dataset_bundle.input_shape[0])
            input_dim = None
    """
    is_tabular_input = len(input_shape) == 1
    if is_tabular_input:
        stem_channels = _nget(config, "model.cnn.stem_channels", 1, warned)
        return int(stem_channels), int(input_shape[0])
    return int(input_shape[0]), None


def build_model_from_config(config: dict, device: torch.device, input_ch: int, input_dim: int | None):
    """Rebuilds the exact architecture described by config.yaml and loads
    model_weights.pt's state dict into it directly. Note: for finetune-
    curriculum runs (model.cnn.pretrained_checkpoint set), this correctly
    builds with data.num_classes (the run's OWN final head, e.g. 3), not
    pretrained_num_classes (e.g. 10) -- pretrained_num_classes only
    describes how this run's training was initialized, and by the final
    checkpoint the head has already been swapped and finetuned.

    `input_ch` / `input_dim` are NOT read from config here -- they come
    from `determine_input_ch_and_dim`, applied to the dataset bundle's own
    `input_shape`, exactly as training_runner.py derives them."""
    warned: list[str] = []

    dataset_name = _nget(config, "data.dataset_name", "cifar10", warned)
    if dataset_name != "cifar10":
        raise ValueError(
            f"data.dataset_name='{dataset_name}' -- this script's data-loading path only "
            "implements CIFAR-10 (with optional metaclass remap). Synthetic 2D datasets "
            "(spiral/blobs/rings/checkerboard/...) need a different loader; let me know "
            "which one and I'll add it."
        )

    backbone = _nget(config, "model.cnn.backbone", "resnet18", warned)
    num_classes = _nget(config, "data.num_classes", None, warned)
    if num_classes is None:
        raise ValueError(
            "config.yaml did not provide 'data.num_classes' and there is no safe default."
        )
    stem_spatial_size = _nget(config, "model.cnn.stem_spatial_size", 8, warned)

    if backbone == "resnet18":
        blocks_per_stage = tuple(_nget(config, "model.cnn.resnet.blocks_per_stage", (2, 2, 2, 2), warned))
        num_stages = _nget(config, "model.cnn.resnet.num_stages", 4, warned)
        width_mult = _nget(config, "model.cnn.resnet.width_mult", 1.0, warned)
        model, classifier, capture = build_cnn_model(
            num_classes=num_classes,
            input_ch=input_ch,
            device=device,
            input_dim=input_dim,
            stem_spatial_size=stem_spatial_size,
            blocks_per_stage=blocks_per_stage,
            num_stages=num_stages,
            width_mult=width_mult,
        )
    elif backbone == "simple_cnn":
        channels = _nget(config, "model.cnn.simple_cnn.channels", [16, 32], warned)
        kernel_size = _nget(config, "model.cnn.simple_cnn.kernel_size", 3, warned)
        use_batchnorm = _nget(config, "model.cnn.simple_cnn.use_batchnorm", True, warned)
        pool_every_block = _nget(config, "model.cnn.simple_cnn.pool_every_block", True, warned)
        dropout = _nget(config, "model.cnn.simple_cnn.dropout", 0.0, warned)
        fc_hidden_dim = _nget(config, "model.cnn.simple_cnn.fc_hidden_dim", None, warned)
        model, classifier, capture = build_simple_cnn_model(
            num_classes=num_classes,
            input_ch=input_ch,
            device=device,
            channels=channels,
            kernel_size=kernel_size,
            use_batchnorm=use_batchnorm,
            pool_every_block=pool_every_block,
            dropout=dropout,
            fc_hidden_dim=fc_hidden_dim,
            input_dim=input_dim,
            stem_spatial_size=stem_spatial_size,
        )
    else:
        raise ValueError(f"Unknown backbone '{backbone}' in config.yaml")

    if warned:
        print("[config] The following architecture fields were missing "
              "and used defaults -- please double-check these match the "
              "actual run:")
        for w in warned:
            print(f"    - {w}")

    return model, classifier, capture, num_classes, input_ch, input_dim


def load_final_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["torch_model"] if isinstance(checkpoint, dict) and "torch_model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()


# ---------------------------------------------------------------------------
# Data loading: dataset_bundle.npz (primary) with raw-CIFAR10 fallback
# ---------------------------------------------------------------------------

_TRAIN_INPUT_KEYS = ["train_inputs", "train_x", "x_train", "inputs_train"]
_TRAIN_TARGET_KEYS = ["train_targets", "train_y", "y_train", "targets_train"]
_TEST_INPUT_KEYS = ["test_inputs", "test_x", "x_test", "inputs_test"]
_TEST_TARGET_KEYS = ["test_targets", "test_y", "y_test", "targets_test"]


def find_dataset_bundle_path(run_dir: Path, data_root: str, config: dict, cli_path: str | None) -> Path | None:
    """Search order: explicit --dataset-bundle > run_dir/dataset_bundle.npz >
    data_root/dataset_bundle.npz > data_root/{dataset_name}_dataset_bundle.npz.
    Returns None if nothing is found (caller falls back to raw CIFAR-10)."""
    if cli_path is not None:
        path = Path(cli_path)
        if not path.exists():
            raise FileNotFoundError(f"--dataset-bundle path does not exist: {path}")
        return path

    warned: list[str] = []
    dataset_name = _nget(config, "data.dataset_name", "cifar10", warned)
    candidates = [
        run_dir / "dataset_bundle.npz",
        Path(data_root) / "dataset_bundle.npz",
        Path(data_root) / f"{dataset_name}_dataset_bundle.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _first_present(npz: np.lib.npyio.NpzFile, keys: list[str]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key in npz.files:
            return npz[key], key
    return None, None


def load_dataset_bundle(bundle_path: Path, split: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Loads train/test inputs+targets straight from dataset_bundle.npz,
    trying a handful of common key-name aliases (since the exact
    dataset_bundle schema wasn't shared). Prints which keys it actually
    matched, so a wrong guess is visible immediately rather than silent."""
    npz = np.load(bundle_path, allow_pickle=True)

    input_keys = _TRAIN_INPUT_KEYS if split == "train" else _TEST_INPUT_KEYS
    target_keys = _TRAIN_TARGET_KEYS if split == "train" else _TEST_TARGET_KEYS

    inputs, input_key = _first_present(npz, input_keys)
    targets, target_key = _first_present(npz, target_keys)
    if inputs is None or targets is None:
        raise KeyError(
            f"Could not find {split} inputs/targets in {bundle_path}. "
            f"Available keys: {list(npz.files)}. Tried input keys {input_keys} "
            f"and target keys {target_keys} -- tell me the real key names and "
            "I'll fix the aliases."
        )

    if "input_shape" in npz.files:
        input_shape = tuple(int(v) for v in np.asarray(npz["input_shape"]).ravel())
    else:
        input_shape = tuple(int(v) for v in inputs.shape[1:])

    print(f"[dataset_bundle] Loaded from {bundle_path}")
    print(f"    inputs key: '{input_key}' -> shape {inputs.shape}")
    print(f"    targets key: '{target_key}' -> shape {targets.shape}, unique labels {sorted(set(np.unique(targets).tolist()))}")
    print(f"    input_shape: {input_shape}")

    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.int64), input_shape


def load_cifar10_arrays_fallback(data_root: str, split: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """FALLBACK ONLY (used if no dataset_bundle.npz can be found): plain
    torchvision CIFAR-10, [0,1] scaling, no per-channel normalization. This
    may not exactly match your training pipeline's preprocessing -- prefer
    the dataset_bundle.npz path whenever possible."""
    print("[WARNING] No dataset_bundle.npz found -- falling back to raw "
          "torchvision CIFAR-10 with [0,1] scaling only. If your training "
          "pipeline applies additional normalization, these features (and "
          "the resulting PCA) will not exactly match what the model saw "
          "during training.")
    transform = T.Compose([T.ToTensor()])
    train = split == "train"
    dataset = torchvision.datasets.CIFAR10(root=data_root, train=train, download=True, transform=transform)
    images = np.stack([np.asarray(img) for img, _ in dataset], axis=0).astype(np.float32)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    return images, labels, tuple(images.shape[1:])


def remap_to_metaclasses(labels: np.ndarray, mapping: list[list[int]]) -> np.ndarray:
    remapped = np.full_like(labels, fill_value=-1)
    for metaclass_idx, original_classes in enumerate(mapping):
        for orig_c in original_classes:
            remapped[labels == orig_c] = metaclass_idx
    if np.any(remapped < 0):
        missing = sorted(set(np.unique(labels[remapped < 0]).tolist()))
        raise ValueError(f"Metaclass mapping does not cover original classes: {missing}")
    return remapped


def resolve_metaclass_mapping(config: dict, cli_mapping: str | None, cli_preset: str | None) -> list[list[int]] | None:
    """Returns None if no remap is needed. Priority: explicit
    --metaclass-mapping > --metaclass-mapping-preset > config.yaml's own
    `data.metaclass_mapping` (the real field, confirmed from an actual
    config.yaml) > the hardcoded intuitive/alternative presets as a last
    resort (only reached for a flat config.txt without the mapping)."""
    if cli_mapping is not None:
        return json.loads(cli_mapping)
    if cli_preset is not None:
        return METACLASS_PRESETS[cli_preset]
    warned: list[str] = []
    mapping = _nget(config, "data.metaclass_mapping", None, warned)
    if mapping is not None:
        return mapping
    return None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_pre_classifier_features(
    adapter: TorchModelAdapter,
    inputs: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    chunks = []
    num_samples = inputs.shape[0]
    for start in range(0, num_samples, batch_size):
        end = min(num_samples, start + batch_size)
        _, intermediates = adapter.apply(None, inputs[start:end], return_intermediates=True)
        chunks.append(np.asarray(intermediates["pre_classifier"]))
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Within-class deflation + PCA explained variance
# ---------------------------------------------------------------------------

def deflate_by_class_mean(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Subtracts each sample's own class mean from its feature vector.
    This is the same operation your path-curvature / NC1-NC2 deflation uses:
    it removes all between-class (inter-cluster-centroid) variance, leaving
    only the within-class spread. Requires every class present in `labels`
    to have at least 1 sample (classes with exactly 1 sample contribute a
    zero vector after deflation, which is fine -- they just won't add rank)."""
    deflated = np.empty_like(features)
    classes, counts = np.unique(labels, return_counts=True)
    single_sample_classes = classes[counts < 2]
    if len(single_sample_classes) > 0:
        print(f"[within-class] WARNING: classes {single_sample_classes.tolist()} have "
              f"< 2 samples in this subset -- their within-class spread is "
              "undefined/degenerate (will deflate to exactly zero). Consider "
              "increasing --num-samples if these are supposed to be well-populated.")
    for c in classes:
        mask = labels == c
        class_mean = features[mask].mean(axis=0, keepdims=True)
        deflated[mask] = features[mask] - class_mean
    return deflated


def compute_explained_variance(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA over the (already-centered-per-class, i.e. globally zero-mean-ish)
    feature matrix. Returns (individual_explained_variance_ratio,
    cumulative_ratio, eigvals), both ratio arrays sorted descending, length
    = min(D, N-1). `eigvals` (unnormalized) is also returned so the caller
    can compute a participation ratio without redoing the SVD."""
    centered = features - features.mean(axis=0, keepdims=True)
    n, d = centered.shape
    # SVD on the centered data matrix is more numerically stable than
    # forming the (D, D) covariance explicitly, and just as cheap here
    # since this runs once (final checkpoint only), not every training step.
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigvals = (singular_values ** 2) / max(n - 1, 1)
    explained_ratio = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained_ratio)
    return explained_ratio, cumulative, eigvals


def participation_ratio(eigvals: np.ndarray) -> float:
    """PR = (sum lambda_i)^2 / sum(lambda_i^2). A soft, scale-free count of
    how many dimensions are 'really' being used."""
    return float((eigvals.sum() ** 2) / np.sum(eigvals ** 2))


def plot_explained_variance(
    explained_ratio: np.ndarray,
    cumulative: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
    participation_ratio_value: float | None = None,
) -> None:
    num_components = len(cumulative)
    x = np.arange(1, num_components + 1)
    cumulative_pct = cumulative * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))

    ax2 = ax.twinx()
    ax2.bar(x, explained_ratio * 100.0, color="lightsalmon", alpha=0.6, width=1.0, label="individual (scree)")
    ax2.set_ylabel("Individual explained variance (%)")
    ax2.set_ylim(0, max(explained_ratio * 100.0) * 1.15)

    ax.plot(x, cumulative_pct, color="firebrick", linewidth=2.0, label="cumulative")
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_ylim(0, 102)
    ax.set_xlim(1, num_components)

    k_at_threshold = {}
    for threshold in (95.0, 99.0):
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0)
        # First component index at which cumulative variance reaches the threshold.
        idx = int(np.searchsorted(cumulative_pct, threshold) + 1)
        idx = min(idx, num_components)
        k_at_threshold[threshold] = idx
        ax.axvline(idx, color="gray", linestyle="--", linewidth=1.0)
        ax.annotate(
            f"{threshold:.0f}% @ k={idx}",
            xy=(idx, threshold),
            xytext=(idx + max(1, num_components * 0.02), threshold - 4),
            fontsize=9,
            color="dimgray",
        )

    if participation_ratio_value is not None:
        ax.axvline(participation_ratio_value, color="steelblue", linestyle=":", linewidth=1.5)
        ax.annotate(
            f"PR (global) = {participation_ratio_value:.1f}",
            xy=(participation_ratio_value, 10),
            xytext=(participation_ratio_value + max(1, num_components * 0.02), 10),
            fontsize=9,
            color="steelblue",
        )

    ax.set_title(f"Within-Class PCA Explained Variance -- Final Checkpoint{title_suffix}")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[within-class] k95={k_at_threshold[95.0]}, k99={k_at_threshold[99.0]}"
          + (f", PR (global, pre-deflation) = {participation_ratio_value:.2f}"
             if participation_ratio_value is not None else ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=str, help="Training run directory containing config.yaml and model_weights.pt")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num-samples", type=int, default=None,
                         help="Subsample size for feature extraction / PCA. Defaults to "
                              "analysis.metric_subset_size from config.yaml if present, else 5000.")
    parser.add_argument("--data-root", type=str, default=None,
                         help="torchvision CIFAR10 root (downloads if missing). Defaults to "
                              "data.data_dir from config.yaml if present, else './data'.")
    parser.add_argument("--metaclass-mapping", type=str, default=None, help="JSON list-of-lists override, e.g. '[[0,1,8,9],[3,4,5,7],[2,6]]'")
    parser.add_argument("--metaclass-mapping-preset", choices=list(METACLASS_PRESETS.keys()), default=None)
    parser.add_argument("--dataset-bundle", type=str, default=None,
                         help="Explicit path to dataset_bundle.npz. If omitted, searched for in "
                              "run_dir and data.data_dir; falls back to raw torchvision CIFAR-10 "
                              "if not found anywhere.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                         help="Defaults to config.yaml's top-level 'device' / 'use_gpu' if present, "
                              "else cuda-if-available.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "model_weights.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"model_weights.pt not found in {run_dir}")

    config = _load_config(run_dir)

    if args.device is not None:
        device_str = args.device
    elif config.get("device") in ("cpu", "cuda"):
        device_str = config["device"]
    elif config.get("use_gpu") is False:
        device_str = "cpu"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    num_samples = args.num_samples
    if num_samples is None:
        warned: list[str] = []
        num_samples = _nget(config, "analysis.metric_subset_size", 5000, warned)

    data_root = args.data_root
    if data_root is None:
        warned = []
        data_root = _nget(config, "data.data_dir", "./data", warned)

    bundle_path = find_dataset_bundle_path(run_dir, data_root, config, args.dataset_bundle)
    if bundle_path is not None:
        images, labels, input_shape = load_dataset_bundle(bundle_path, args.split)
    else:
        images, labels, input_shape = load_cifar10_arrays_fallback(data_root, args.split)

    warned = []
    input_ch, input_dim = determine_input_ch_and_dim(input_shape, config, warned)
    if warned:
        print("[input shape] Derived input_ch/input_dim with some defaults -- verify:")
        for w in warned:
            print(f"    - {w}")

    model, classifier, capture, num_classes, _, _ = build_model_from_config(config, device, input_ch, input_dim)
    load_final_checkpoint(model, checkpoint_path, device)
    adapter = TorchModelAdapter(model, classifier, capture, device)

    # If the bundle's own targets already sit in [0, num_classes), they were
    # generated with the metaclass remap already baked in -- use them as-is.
    # Otherwise (e.g. the raw-CIFAR10 fallback path, or a bundle that still
    # carries original 10-class labels), apply the metaclass remap here.
    labels_already_remapped = labels.max() < num_classes and labels.min() >= 0
    if num_classes != 10 and not labels_already_remapped:
        mapping = resolve_metaclass_mapping(config, args.metaclass_mapping, args.metaclass_mapping_preset)
        if mapping is None:
            raise ValueError(
                f"Model has num_classes={num_classes}, labels in the loaded data don't already "
                "fit that range, and no metaclass mapping was found in config.yaml's "
                "data.metaclass_mapping or passed via --metaclass-mapping / "
                "--metaclass-mapping-preset. Please supply one."
            )
        labels = remap_to_metaclasses(labels, mapping)

    rng = np.random.default_rng(args.seed)
    if num_samples is not None and num_samples < images.shape[0]:
        idx = rng.choice(images.shape[0], size=num_samples, replace=False)
        images, labels = images[idx], labels[idx]

    features = extract_pre_classifier_features(adapter, images)
    print(f"Extracted pre-classifier features: {features.shape[0]} samples x {features.shape[1]} dims")

    # Participation ratio on the GLOBAL (pre-deflation) covariance, for
    # reference alongside the within-class k95/k99 -- this is the quantity
    # the caller wants to compare the deflated k95/k99 gap against.
    _, _, global_eigvals = compute_explained_variance(features)
    pr_global = participation_ratio(global_eigvals)

    within_class_features = deflate_by_class_mean(features, labels)
    explained_ratio, cumulative, _ = compute_explained_variance(within_class_features)

    output_path = run_dir / "pca_within_class_explained_variance.png"
    plot_explained_variance(
        explained_ratio,
        cumulative,
        output_path,
        title_suffix=f" ({args.split} set, N={features.shape[0]}, D={features.shape[1]}, K={len(np.unique(labels))} classes)",
        participation_ratio_value=pr_global,
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()