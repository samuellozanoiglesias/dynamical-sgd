'''
Reduced-dimensionality feature-space plots for JAX MLP checkpoints
(spiral / rings / blobs / checkerboard / dartboard / gaussian_blobs runs
saved by training_runner.py -- i.e. a run_dir containing config.yaml,
jax_params.pkl, and dataset_bundle.npz).

This is the MLP/synthetic-2D-dataset counterpart of
generate_reduce_dim_plot.py (which does the same thing for pretrained CNN
checkpoints on CIFAR-10). Rather than reducing raw pixels or CNN pooled
features, this script pulls the model's own learned pre-classifier (or any
other requested) hidden-layer representation via model.py's
`return_intermediates=True`, then reduces THAT down to 2 and/or 3 dims with
PCA / random_projection / UMAP / t-SNE, reusing the exact same reduction
primitives generate_dataset.py already uses for the CNN path.

USE:

nohup python generate_reduce_dim_plot_mlp.py \
    --run_dir /data/samuel_lozano/dynamical-sgd/without_bumps/spiral/training_2026_08_10-12_00_00 \
    --reduce_dims 2 3 \
    --reduce_methods pca umap tsne \
    > generate_reduce_dim_plot_mlp.log 2>&1 &

# Single method/dim, feature layer = the first hidden layer's post-activation
# (only meaningful with num_hidden_layers >= 2, where pre_classifier != that):
nohup python generate_reduce_dim_plot_mlp.py \
    --run_dir /data/samuel_lozano/dynamical-sgd/without_bumps/rings/training_2026_08_10-13_10_00 \
    --reduce_dims 3 \
    --reduce_methods umap \
    --feature_layer first_activation_post \
    --plot_dims 2 3 > generate_reduce_dim_plot_mlp.log 2>&1 &

# To batch over several run dirs (e.g. all datasets), just loop in the shell:
#   for d in /data/samuel_lozano/dynamical-sgd/without_bumps/*/training_*; do
#       python generate_reduce_dim_plot_mlp.py --run_dir "$d"
#   done
'''

import sys
import os

# Small MLPs on 2D synthetic data don't need a GPU -- default to CPU so this
# doesn't fight training jobs for a node's L40S. Pass --use_gpu to opt in.
# (Must happen before `import jax`, hence the raw argv scan up here instead
# of after argparse.)
if "--use_gpu" not in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

from model import build_model
from generate_dataset import (
    _pca_fit_transform,
    _random_projection_fit_transform,
    _umap_fit_transform,
    _tsne_fit_transform,
    _save_reduced_dataset_visualizations,
    _reduced_plot_output_name,
)

_FEATURE_LAYER_CHOICES = ("input", "first_activation_post", "pre_classifier")
_REDUCE_METHOD_CHOICES = ("pca", "random_projection", "umap", "tsne")


def _reduce(method: str, train_feat: np.ndarray, test_feat: np.ndarray, reduce_dim: int, seed: int):
    if method == "pca":
        return _pca_fit_transform(train_feat, test_feat, reduce_dim)
    if method == "random_projection":
        return _random_projection_fit_transform(train_feat, test_feat, reduce_dim, seed=seed)
    if method == "umap":
        return _umap_fit_transform(train_feat, test_feat, reduce_dim, seed)
    if method == "tsne":
        return _tsne_fit_transform(train_feat, test_feat, reduce_dim, seed)
    raise ValueError(f"Unsupported reduce_method '{method}'. Expected one of {_REDUCE_METHOD_CHOICES}.")


def _standardize(train_embed: np.ndarray, test_embed: np.ndarray):
    std = train_embed.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    mean = train_embed.mean(axis=0, keepdims=True)
    return (train_embed - mean) / std, (test_embed - mean) / std


def main():
    parser = argparse.ArgumentParser(
        description="Plot reduced-dim feature-space embeddings for a JAX MLP checkpoint."
    )
    parser.add_argument("--run_dir", type=str, required=True,
                         help="Directory containing config.yaml, jax_params.pkl, dataset_bundle.npz "
                              "(as saved by training_runner.py's run_training).")
    parser.add_argument("--config_path", type=str, default=None,
                         help="Override path to the run's config.yaml (default: <run_dir>/config.yaml).")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                         help="Override path to jax_params.pkl (default: <run_dir>/jax_params.pkl).")
    parser.add_argument("--dataset_bundle_path", type=str, default=None,
                         help="Override path to dataset_bundle.npz (default: <run_dir>/dataset_bundle.npz).")

    parser.add_argument("--reduce_dims", type=int, nargs="+", default=[2, 3],
                         help="Target dimensionalities to reduce the feature space to.")
    parser.add_argument("--plot_dims", type=int, nargs="+", default=[2, 3], choices=[2, 3],
                         help="Which scatter-plot dimensionalities to render for each reduce_dim "
                              "(a plot_dim is skipped if it exceeds that reduce_dim).")
    parser.add_argument("--reduce_methods", type=str, nargs="+", default=["pca", "umap", "tsne"],
                         choices=list(_REDUCE_METHOD_CHOICES),
                         help="Dimensionality reduction method(s) to run.")
    parser.add_argument("--feature_layer", type=str, default="pre_classifier",
                         choices=list(_FEATURE_LAYER_CHOICES),
                         help="Which model.py intermediate to reduce: the raw 'input', the first "
                              "hidden layer's 'first_activation_post', or the final hidden "
                              "representation feeding the classifier, 'pre_classifier' (default; "
                              "for a single-hidden-layer MLP this is the same as "
                              "first_activation_post).")
    parser.add_argument("--reduce_seed", type=int, default=0,
                         help="Random seed for reduction methods that need one (random_projection, umap, tsne).")
    parser.add_argument("--standardize", action="store_true", default=True,
                         help="Rescale the embedding to unit-ish variance per axis before plotting (default on).")
    parser.add_argument("--no_standardize", action="store_false", dest="standardize")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="Directory to save plots. Defaults to <run_dir>/plots_reduce_dim_feature_space.")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                         help="Allow JAX to use a GPU if available. Off by default for these small MLPs.")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = Path(args.config_path) if args.config_path else run_dir / "config.yaml"
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else run_dir / "jax_params.pkl"
    bundle_path = Path(args.dataset_bundle_path) if args.dataset_bundle_path else run_dir / "dataset_bundle.npz"

    for path, label in [(config_path, "config"), (checkpoint_path, "checkpoint"), (bundle_path, "dataset bundle")]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find {label} file: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    dataset_name = str(data_cfg.get("dataset_name", "unknown"))
    num_classes = int(data_cfg.get("num_classes"))
    random_seed = int(config.get("training", {}).get("random_seed", data_cfg.get("random_seed", 0)))

    print(f"Loading dataset bundle from: {bundle_path}")
    npz = np.load(bundle_path)
    train_inputs = npz["train_inputs"]
    train_targets = npz["train_targets"]
    test_inputs = npz["test_inputs"]
    test_targets = npz["test_targets"]

    input_shape = train_inputs.shape[1:]
    built_model = build_model(
        model_cfg=model_cfg,
        input_shape=input_shape,
        num_classes=num_classes,
        random_seed=random_seed,
    )
    model = built_model.model

    print(f"Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)

    print(f"Running forward pass to extract '{args.feature_layer}' features...")
    _, train_intermediates = model.apply(params, train_inputs, return_intermediates=True)
    _, test_intermediates = model.apply(params, test_inputs, return_intermediates=True)

    if args.feature_layer not in train_intermediates:
        raise ValueError(
            f"feature_layer '{args.feature_layer}' is not available for this checkpoint's "
            f"architecture. Available intermediates: {sorted(train_intermediates)}."
        )

    train_feat = np.asarray(train_intermediates[args.feature_layer], dtype=np.float64)
    test_feat = np.asarray(test_intermediates[args.feature_layer], dtype=np.float64)
    print(f"Feature space shape: train {train_feat.shape}, test {test_feat.shape}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots_reduce_dim_feature_space"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_plot_dim = max(args.plot_dims)
    for reduce_dim in args.reduce_dims:
        if train_feat.shape[1] < reduce_dim:
            print(f"Skipping reduce_dim={reduce_dim}: feature dim ({train_feat.shape[1]}) is smaller.")
            continue
        applicable_plot_dims = [p for p in args.plot_dims if p <= reduce_dim]
        if not applicable_plot_dims:
            print(f"Skipping reduce_dim={reduce_dim}: no requested plot_dim fits.")
            continue

        for method in args.reduce_methods:
            print(f"Reducing with {method} -> dim {reduce_dim} ...")
            train_embed, test_embed = _reduce(method, train_feat, test_feat, reduce_dim, args.reduce_seed)

            if args.standardize:
                train_embed, test_embed = _standardize(train_embed, test_embed)

            train_embed = train_embed.astype(np.float32)
            test_embed = test_embed.astype(np.float32)

            for plot_dim in applicable_plot_dims:
                output_name = _reduced_plot_output_name(
                    f"{dataset_name}_{args.feature_layer}", method, reduce_dim, plot_dim
                )
                title_prefix = f"{dataset_name.capitalize()} | {args.feature_layer} | {method} (dim={reduce_dim})"
                _save_reduced_dataset_visualizations(
                    x_train=train_embed[:, :plot_dim],
                    y_train=train_targets,
                    x_test=test_embed[:, :plot_dim],
                    y_test=test_targets,
                    num_classes=num_classes,
                    out_dir=out_dir,
                    title_prefix=title_prefix,
                    output_name=output_name,
                    plot_dim=plot_dim,
                )
                print(f"  saved {out_dir / output_name}")

    print(f"Done! Plots saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
