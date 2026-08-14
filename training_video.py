'''
Opt-in "training video" generation for the JAX MLP path of training_runner.py.

Every `videos.every_n_steps` training steps this records three PNG frames
into `<run_dir>/video_frames/{umap,pca,decision_boundary}/`:

  - umap/step_XXXXXXXX.png              : 2D UMAP projection of a chosen
                                           feature layer (default:
                                           pre_classifier), same primitive
                                           generate_reduce_dim_plot_mlp.py
                                           uses (`_umap_fit_transform` from
                                           generate_dataset.py).
  - pca/step_XXXXXXXX.png                : 2D PCA projection of the same
                                           feature layer (`_pca_fit_transform`).
  - decision_boundary/step_XXXXXXXX.png  : the same decision-boundary plot
                                           `metrics.plot_2d_decision_boundaries`
                                           produces at the end of a run, but
                                           snapshotted throughout training.

At the end of the run, `VideoFrameRecorder.finalize()` stitches each
folder's frames (sorted by step) into a .gif under `<run_dir>/videos/`.

Enable via config.yaml:

    videos:
      enabled: true
      every_n_steps: 5000        # snapshot cadence, in training steps
      fps: 6                     # gif playback speed
      feature_layer: pre_classifier   # input | first_activation_post | pre_classifier
      umap_seed: 0
      pca:
        reduce_dim: 2       # dimensionality PCA reduces the feature layer to
        plot_dim: 2         # 2 or 3 -- how many of those dims to scatter-plot
      umap:
        reduce_dim: 2       # dimensionality UMAP reduces the feature layer to
        plot_dim: 2         # 2 or 3 -- how many of those dims to scatter-plot
      # For both pca and umap, reduce_dim must be >= plot_dim (you can't plot
      # more dimensions than you reduced to).
      decision_boundary_grid_size: 250   # coarser than the final-plot default
                                          # (400) since this runs many times
      align_embeddings: true     # Procrustes-align each PCA/UMAP frame to
                                  # the previous one (rotation/reflection
                                  # only -- no rescaling) so the embedding
                                  # doesn't spin/flip randomly frame to
                                  # frame; genuine shrink/expand of the
                                  # point cloud (e.g. collapse) is preserved.
      standardize_embeddings: false  # if true, rescale each frame to unit
                                      # variance -- off by default because
                                      # this would hide collapse (shrinking
                                      # point clouds would always look the
                                      # same size).

Only supported for the JAX MLP path (model.architecture in
{mlp, mlp_relu, mlp_tanh}) on 2D synthetic datasets; training_runner.py is
responsible for not constructing a VideoFrameRecorder on the PyTorch CNN
path.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from generate_dataset import _pca_fit_transform, _umap_fit_transform
from metrics import plot_2d_decision_boundaries

# Datasets for which the (N,2)-input decision-boundary plot is meaningful.
_DECISION_BOUNDARY_DATASETS = {
    "spiral", "blobs", "gaussian_blobs", "rings",
    "checkerboard", "random_checkerboard", "dartboard",
}
_FEATURE_LAYER_CHOICES = ("input", "first_activation_post", "pre_classifier")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


_PLOT_DIM_CHOICES = (2, 3)


@dataclass(frozen=True)
class VideoConfig:
    enabled: bool
    every_n_steps: int
    fps: int
    feature_layer: str
    umap_seed: int
    pca_reduce_dim: int
    pca_plot_dim: int
    umap_reduce_dim: int
    umap_plot_dim: int
    decision_boundary_grid_size: int
    align_embeddings: bool
    standardize_embeddings: bool


def parse_video_config(config: dict) -> VideoConfig:
    videos_cfg = config.get("videos", {}) or {}

    enabled = _as_bool(videos_cfg.get("enabled", False))
    every_n_steps = int(videos_cfg.get("every_n_steps", 5000))
    fps = int(videos_cfg.get("fps", 6))
    feature_layer = str(videos_cfg.get("feature_layer", "pre_classifier")).strip().lower()
    umap_seed = int(videos_cfg.get("umap_seed", 0))
    grid_size = int(videos_cfg.get("decision_boundary_grid_size", 250))
    align_embeddings = _as_bool(videos_cfg.get("align_embeddings", True))
    standardize_embeddings = _as_bool(videos_cfg.get("standardize_embeddings", False))

    pca_dim_cfg = videos_cfg.get("pca", {}) or {}
    umap_dim_cfg = videos_cfg.get("umap", {}) or {}
    pca_reduce_dim = int(pca_dim_cfg.get("reduce_dim", 2))
    pca_plot_dim = int(pca_dim_cfg.get("plot_dim", 2))
    umap_reduce_dim = int(umap_dim_cfg.get("reduce_dim", 2))
    umap_plot_dim = int(umap_dim_cfg.get("plot_dim", 2))

    if enabled and every_n_steps <= 0:
        raise ValueError("videos.every_n_steps must be > 0")
    if enabled and fps <= 0:
        raise ValueError("videos.fps must be > 0")
    if enabled and feature_layer not in _FEATURE_LAYER_CHOICES:
        raise ValueError(
            f"videos.feature_layer '{feature_layer}' unsupported. "
            f"Expected one of {_FEATURE_LAYER_CHOICES}."
        )
    if enabled and grid_size <= 0:
        raise ValueError("videos.decision_boundary_grid_size must be > 0")
    if enabled and pca_plot_dim not in _PLOT_DIM_CHOICES:
        raise ValueError(
            f"videos.pca.plot_dim must be one of {_PLOT_DIM_CHOICES}, got {pca_plot_dim}."
        )
    if enabled and umap_plot_dim not in _PLOT_DIM_CHOICES:
        raise ValueError(
            f"videos.umap.plot_dim must be one of {_PLOT_DIM_CHOICES}, got {umap_plot_dim}."
        )
    if enabled and pca_reduce_dim < pca_plot_dim:
        raise ValueError(
            f"videos.pca.reduce_dim ({pca_reduce_dim}) must be >= "
            f"videos.pca.plot_dim ({pca_plot_dim})."
        )
    if enabled and umap_reduce_dim < umap_plot_dim:
        raise ValueError(
            f"videos.umap.reduce_dim ({umap_reduce_dim}) must be >= "
            f"videos.umap.plot_dim ({umap_plot_dim})."
        )

    return VideoConfig(
        enabled=enabled,
        every_n_steps=every_n_steps,
        fps=fps,
        feature_layer=feature_layer,
        umap_seed=umap_seed,
        pca_reduce_dim=pca_reduce_dim,
        pca_plot_dim=pca_plot_dim,
        umap_reduce_dim=umap_reduce_dim,
        umap_plot_dim=umap_plot_dim,
        decision_boundary_grid_size=grid_size,
        align_embeddings=align_embeddings,
        standardize_embeddings=standardize_embeddings,
    )


def _align_to_reference(
    train_embed: np.ndarray, test_embed: np.ndarray, reference_train_embed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes alignment (rotation/reflection + translation,
    no scaling) of `train_embed` onto `reference_train_embed`, applying the
    same transform to `test_embed`. Both embeddings must correspond to the
    same (fixed) set of underlying points in the same order, which holds
    here since the metric subset is fixed for the whole run."""
    mean_new = train_embed.mean(axis=0, keepdims=True)
    mean_ref = reference_train_embed.mean(axis=0, keepdims=True)
    centered_new = train_embed - mean_new
    centered_ref = reference_train_embed - mean_ref

    cross_cov = centered_new.T @ centered_ref
    u, _s, vt = np.linalg.svd(cross_cov)
    rotation = u @ vt  # (D, D) orthogonal matrix

    train_aligned = centered_new @ rotation + mean_ref
    test_aligned = (test_embed - mean_new) @ rotation + mean_ref
    return train_aligned, test_aligned


def _standardize_pair(train_embed: np.ndarray, test_embed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = train_embed.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    mean = train_embed.mean(axis=0, keepdims=True)
    return (train_embed - mean) / std, (test_embed - mean) / std


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    v_min, v_max = float(np.min(values)), float(np.max(values))
    margin = max(0.05, 0.1 * (v_max - v_min))
    return v_min - margin, v_max + margin


def _points_cmap(num_classes: int) -> ListedColormap:
    base_cmap = plt.get_cmap("tab10")
    point_colors = [base_cmap(class_id % 10) for class_id in range(num_classes)]
    return ListedColormap(point_colors)


def _save_scatter_frame_2d(
    train_pts: np.ndarray,
    train_y: np.ndarray,
    test_pts: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    output_path: Path,
    title: str,
) -> None:
    points_cmap = _points_cmap(num_classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        train_pts[:, 0], train_pts[:, 1], c=train_y, cmap=points_cmap,
        vmin=0, vmax=max(0, num_classes - 1),
        s=18, alpha=0.95, edgecolors="black", linewidths=0.25, label="Train",
    )
    ax.scatter(
        test_pts[:, 0], test_pts[:, 1], c=test_y, cmap=points_cmap,
        vmin=0, vmax=max(0, num_classes - 1),
        marker="x", s=24, alpha=1.0, linewidths=1.0, label="Test",
    )

    all_pts = np.concatenate([train_pts, test_pts], axis=0)
    ax.set_xlim(*_axis_limits(all_pts[:, 0]))
    ax.set_ylim(*_axis_limits(all_pts[:, 1]))

    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_scatter_frame_3d(
    train_pts: np.ndarray,
    train_y: np.ndarray,
    test_pts: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    output_path: Path,
    title: str,
) -> None:
    points_cmap = _points_cmap(num_classes)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        train_pts[:, 0], train_pts[:, 1], train_pts[:, 2], c=train_y, cmap=points_cmap,
        vmin=0, vmax=max(0, num_classes - 1),
        s=18, alpha=0.95, edgecolors="black", linewidths=0.25, label="Train",
    )
    ax.scatter(
        test_pts[:, 0], test_pts[:, 1], test_pts[:, 2], c=test_y, cmap=points_cmap,
        vmin=0, vmax=max(0, num_classes - 1),
        marker="x", s=24, alpha=1.0, linewidths=1.0, label="Test",
    )

    all_pts = np.concatenate([train_pts, test_pts], axis=0)
    ax.set_xlim(*_axis_limits(all_pts[:, 0]))
    ax.set_ylim(*_axis_limits(all_pts[:, 1]))
    ax.set_zlim(*_axis_limits(all_pts[:, 2]))

    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_scatter_frame(
    train_pts: np.ndarray,
    train_y: np.ndarray,
    test_pts: np.ndarray,
    test_y: np.ndarray,
    num_classes: int,
    output_path: Path,
    title: str,
    plot_dim: int,
) -> None:
    """Dispatches to the 2D or 3D scatter-frame renderer. `train_pts` /
    `test_pts` must already be sliced down to exactly `plot_dim` columns."""
    if plot_dim == 2:
        _save_scatter_frame_2d(train_pts, train_y, test_pts, test_y, num_classes, output_path, title)
    elif plot_dim == 3:
        _save_scatter_frame_3d(train_pts, train_y, test_pts, test_y, num_classes, output_path, title)
    else:
        raise ValueError(f"plot_dim must be 2 or 3, got {plot_dim}")


def _build_gif(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    from PIL import Image

    if not frame_paths:
        return
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    duration_ms = int(round(1000.0 / max(1, fps)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for image in images:
        image.close()


class VideoFrameRecorder:
    """Captures UMAP / PCA / decision-boundary snapshots throughout a JAX
    MLP training run and stitches them into GIFs at the end."""

    def __init__(
        self,
        run_dir: Path,
        cfg: VideoConfig,
        dataset_key: str,
        num_classes: int,
        metric_inputs: np.ndarray,
        metric_targets: np.ndarray,
        test_metric_inputs: np.ndarray,
        test_metric_targets: np.ndarray,
    ) -> None:
        self.enabled = cfg.enabled
        self.cfg = cfg
        if not self.enabled:
            return

        self.run_dir = Path(run_dir)
        self.frames_dir = self.run_dir / "video_frames"
        self.pca_dir = self.frames_dir / "pca"
        self.umap_dir = self.frames_dir / "umap"
        self.boundary_dir = self.frames_dir / "decision_boundary"
        for folder in (self.pca_dir, self.umap_dir, self.boundary_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.videos_dir = self.run_dir / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_key = dataset_key
        self.num_classes = num_classes
        self.metric_inputs = metric_inputs
        self.metric_targets = metric_targets
        self.test_metric_inputs = test_metric_inputs
        self.test_metric_targets = test_metric_targets

        self.supports_decision_boundary = dataset_key in _DECISION_BOUNDARY_DATASETS

        self._pca_ref_train: np.ndarray | None = None
        self._umap_ref_train: np.ndarray | None = None
        self._last_captured_step: int | None = None
        self._captured_steps: list[int] = []
        self._umap_available = True

    def should_capture(self, global_step: int, total_steps: int) -> bool:
        if not self.enabled:
            return False
        if global_step == self._last_captured_step:
            return False
        if global_step <= 0:
            return True
        if global_step >= total_steps:
            return True
        return (global_step % self.cfg.every_n_steps) == 0

    def capture(
        self,
        global_step: int,
        model,
        params,
        run_label: str,
        dataset_name: str,
    ) -> None:
        if not self.enabled or global_step == self._last_captured_step:
            return
        self._last_captured_step = global_step
        step_tag = f"step_{global_step:08d}"

        _, train_intermediates = model.apply(params, self.metric_inputs, return_intermediates=True)
        _, test_intermediates = model.apply(params, self.test_metric_inputs, return_intermediates=True)

        layer = self.cfg.feature_layer
        if layer not in train_intermediates:
            layer = "pre_classifier"
        train_feat = np.asarray(train_intermediates[layer], dtype=np.float64)
        test_feat = np.asarray(test_intermediates[layer], dtype=np.float64)

        # --- PCA frame ---
        try:
            reduce_dim = self.cfg.pca_reduce_dim
            plot_dim = self.cfg.pca_plot_dim
            train_pca, test_pca = _pca_fit_transform(train_feat, test_feat, reduce_dim)
            # Align in the full reduce_dim space so alignment is consistent
            # frame-to-frame even when only a subset of dims gets plotted.
            if self.cfg.align_embeddings and self._pca_ref_train is not None:
                train_pca, test_pca = _align_to_reference(train_pca, test_pca, self._pca_ref_train)
            self._pca_ref_train = train_pca.copy()

            train_pca_plot, test_pca_plot = train_pca[:, :plot_dim], test_pca[:, :plot_dim]
            if self.cfg.standardize_embeddings:
                train_pca_plot, test_pca_plot = _standardize_pair(train_pca_plot, test_pca_plot)

            _save_scatter_frame(
                train_pca_plot, self.metric_targets, test_pca_plot, self.test_metric_targets,
                self.num_classes,
                output_path=self.pca_dir / f"{step_tag}.png",
                title=f"{run_label} | {dataset_name} | PCA({reduce_dim}->{plot_dim}D) of {layer} | step {global_step}",
                plot_dim=plot_dim,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad frame shouldn't kill training
            print(f"[video] PCA frame failed at step {global_step}: {exc}")

        # --- UMAP frame ---
        if self._umap_available:
            try:
                reduce_dim = self.cfg.umap_reduce_dim
                plot_dim = self.cfg.umap_plot_dim
                train_umap, test_umap = _umap_fit_transform(
                    train_feat, test_feat, reduce_dim, self.cfg.umap_seed
                )
                # Align in the full reduce_dim space so alignment is
                # consistent frame-to-frame even when only a subset of dims
                # gets plotted.
                if self.cfg.align_embeddings and self._umap_ref_train is not None:
                    train_umap, test_umap = _align_to_reference(train_umap, test_umap, self._umap_ref_train)
                self._umap_ref_train = train_umap.copy()

                train_umap_plot, test_umap_plot = train_umap[:, :plot_dim], test_umap[:, :plot_dim]
                if self.cfg.standardize_embeddings:
                    train_umap_plot, test_umap_plot = _standardize_pair(train_umap_plot, test_umap_plot)

                _save_scatter_frame(
                    train_umap_plot, self.metric_targets, test_umap_plot, self.test_metric_targets,
                    self.num_classes,
                    output_path=self.umap_dir / f"{step_tag}.png",
                    title=f"{run_label} | {dataset_name} | UMAP({reduce_dim}->{plot_dim}D) of {layer} | step {global_step}",
                    plot_dim=plot_dim,
                )
            except ModuleNotFoundError:
                self._umap_available = False
                print("[video] umap-learn not installed; skipping UMAP frames for the rest of this run.")
            except Exception as exc:  # noqa: BLE001
                print(f"[video] UMAP frame failed at step {global_step}: {exc}")

        # --- Decision boundary frame ---
        if self.supports_decision_boundary:
            try:
                plot_2d_decision_boundaries(
                    model=model,
                    params=params,
                    train_inputs=self.metric_inputs,
                    train_targets=self.metric_targets,
                    test_inputs=self.test_metric_inputs,
                    test_targets=self.test_metric_targets,
                    output_path=self.boundary_dir / f"{step_tag}.png",
                    title=f"{run_label} | {dataset_name} | Decision boundary | step {global_step}",
                    grid_size=self.cfg.decision_boundary_grid_size,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[video] decision boundary frame failed at step {global_step}: {exc}")

        self._captured_steps.append(global_step)

    def finalize(self) -> dict[str, str]:
        outputs: dict[str, str] = {}
        if not self.enabled or not self._captured_steps:
            return outputs

        for name, folder in (
            ("pca", self.pca_dir),
            ("umap", self.umap_dir),
            ("decision_boundary", self.boundary_dir),
        ):
            frame_paths = sorted(folder.glob("step_*.png"))
            if not frame_paths:
                continue
            output_path = self.videos_dir / f"{name}.gif"
            try:
                _build_gif(frame_paths, output_path, self.cfg.fps)
            except Exception as exc:  # noqa: BLE001
                print(f"[video] failed to assemble {name}.gif: {exc}")
                continue
            outputs[name] = str(output_path)
            print(f"[video] wrote {output_path} ({len(frame_paths)} frames)")

        return outputs