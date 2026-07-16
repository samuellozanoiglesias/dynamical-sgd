import pickle
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import json
from pathlib import Path

# You will need to import your specific model builder here
# to recreate the JAXModel object before applying the loaded parameters.
from model import build_model 

def _desaturate_towards_white(color: tuple[float, float, float, float], mix: float = 0.6) -> tuple[float, float, float, float]:
    r, g, b, _a = color
    return (
        r + (1.0 - r) * mix,
        g + (1.0 - g) * mix,
        b + (1.0 - b) * mix,
        1.0,
    )

def plot_large_decision_boundaries(
    model,
    params,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    output_path: Path,
    title: str,
    grid_size: int = 400,
) -> None:
    """
    An upgraded version of the original plot_2d_decision_boundaries with 
    much larger labels, legends, and ticks.
    """
    train_xy = np.asarray(train_inputs, dtype=np.float32)
    train_y = np.asarray(train_targets, dtype=np.int64)
    test_xy = np.asarray(test_inputs, dtype=np.float32)
    test_y = np.asarray(test_targets, dtype=np.int64)

    all_xy = np.concatenate([train_xy, test_xy], axis=0)
    x_min, x_max = float(np.min(all_xy[:, 0])), float(np.max(all_xy[:, 0]))
    y_min, y_max = float(np.min(all_xy[:, 1])), float(np.max(all_xy[:, 1]))
    x_margin = max(0.05, 0.1 * (x_max - x_min))
    y_margin = max(0.05, 0.1 * (y_max - y_min))

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_margin, x_max + x_margin, grid_size, dtype=np.float32),
        np.linspace(y_min - y_margin, y_max + y_margin, grid_size, dtype=np.float32),
    )
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    input_rank = int(np.asarray(train_inputs).ndim)
    model_input = grid_xy
    if input_rank == 4:
        model_input = np.reshape(model_input, (-1, 2, 1, 1))

    logits = model.apply(params, model_input)
    pred = np.asarray(jnp.argmax(logits, axis=1), dtype=np.int64).reshape(xx.shape)

    max_class = int(max(np.max(train_y), np.max(test_y), np.max(pred)))
    num_classes = max_class + 1
    base_cmap = plt.get_cmap("tab10")
    point_colors = [base_cmap(class_id % 10) for class_id in range(num_classes)]
    background_colors = [_desaturate_towards_white(color, mix=0.62) for color in point_colors]
    background_cmap = ListedColormap(background_colors)
    points_cmap = ListedColormap(point_colors)
    levels = np.arange(-0.5, num_classes + 0.5, 1.0)

    # Increased figure size slightly for better proportions with large text
    fig, ax = plt.subplots(figsize=(10, 9))
    contour = ax.contourf(xx, yy, pred, levels=levels, cmap=background_cmap, alpha=1.0)
    if num_classes > 1:
        ax.contour(
            xx,
            yy,
            pred,
            levels=np.arange(0.5, num_classes, 1.0),
            colors="k",
            linewidths=0.5,
            alpha=0.45,
        )

    # Increased point sizes (s) for training data
    ax.scatter(
        train_xy[:, 0],
        train_xy[:, 1],
        c=train_y,
        cmap=points_cmap,
        vmin=0,
        vmax=max(0, num_classes - 1),
        s=80, 
        alpha=0.98,
        edgecolors="black",
        linewidths=0.5,
        label="Train samples",
    )
    
    # Increased point sizes (s) and linewidths for testing data
    ax.scatter(
        test_xy[:, 0],
        test_xy[:, 1],
        c=test_y,
        cmap=points_cmap,
        vmin=0,
        vmax=max(0, num_classes - 1),
        marker="x",
        s=100, 
        alpha=1.0,
        linewidths=2.0,
        label="Test samples",
    )

    # Adjust colorbar label and tick sizes
    cbar = fig.colorbar(contour, ax=ax, ticks=np.arange(num_classes))
    cbar.set_label("Predicted class", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Upgraded fonts for all textual elements
    ax.set_title(title, fontsize=22, pad=15)
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.2)
    
    # Upgraded legend sizing
    ax.legend(loc="upper right", fontsize=16, markerscale=1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot successfully saved to: {output_path}")

def main():
    # 1. Setup paths
    run_dir = Path("path/to/your/run_directory") # <-- UPDATE THIS
    dataset_path = run_dir / "dataset_bundle.npz"
    params_path = run_dir / "jax_params.pkl"
    output_plot_path = run_dir / "large_spiral_decision_boundaries.png"

    # 2. Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path)
    train_inputs = data['train_inputs']
    train_targets = data['train_targets']
    test_inputs = data['test_inputs']
    test_targets = data['test_targets']

    # 3. Load JAX parameters
    print("Loading parameters...")
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    # 4. Re-instantiate your model (Requires your original config dictionary)
    # You'll need to pass the same basic parameters you used during training.
    # config = {...}
    # built_model = build_model(model_cfg=config.get("model", {}), 
    #                           input_shape=train_inputs.shape[1:], 
    #                           num_classes=int(np.max(train_targets)) + 1, 
    #                           random_seed=42)
    # jax_model = built_model.model
    
    # NOTE: Since I don't have your config, the above is commented out. 
    # Replace `None` below with your initialized `jax_model`.
    jax_model = None 

    # 5. Generate the new plot
    print("Generating updated plot...")
    plot_large_decision_boundaries(
        model=jax_model,
        params=params,
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        output_path=output_plot_path,
        title="Decision Boundaries (Enhanced)"
    )

if __name__ == "__main__":
    main()