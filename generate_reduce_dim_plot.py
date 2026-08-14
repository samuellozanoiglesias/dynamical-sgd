'''
USE:

nohup python generate_reduce_dim_plot.py \
    --checkpoint_type "cifar10_metaclasses_finetune_intuitive" \
    --checkpoint_id "training_2026_07_24-05_24_13" \
    --pretrained_num_classes 3 \
    --reduce_dim 2 \
    --reduce_method tsne > generate_reduce_dim_plot.log 2>&1 &

# For a 3D scatter plot, reduce_dim must be >= 3 and plot_dim must be set to 3:
nohup python generate_reduce_dim_plot.py \
    --checkpoint_type "cifar10_metaclasses_finetune_intuitive" \
    --checkpoint_id "training_2026_07_24-05_24_13" \
    --pretrained_num_classes 3 \
    --reduce_dim 3 \
    --plot_dim 3 \
    --reduce_method tsne > generate_reduce_dim_plot.log 2>&1 &
'''

import argparse
import yaml
from pathlib import Path
from generate_dataset import build_dataset_bundle

mapping_intuitive = [
    [0, 1, 8, 9],   # vehicles: airplane, automobile, ship, truck
    [3, 4, 5, 7],  # mammals: cat, deer, dog, horse
    [2, 6]         # bird, frog
]

mapping_default0 = [
    [0, 6, 3],
    [1, 4, 2],
    [8, 9, 5, 7]
]

mapping_default1 = [
    [0, 1, 2],   
    [3, 4, 5], 
    [6, 7, 8, 9]
]

mapping_default2 = [
    [0, 2, 4],
    [1, 3, 5, 7],
    [6, 8, 9]
]

mapping_default3 = [
    [0, 3, 6, 9],
    [1, 4, 7],
    [2, 5, 8]
]


def load_reduce_cnn_config(checkpoint_file: Path, pretrained_num_classes: int) -> dict:
    """
    Build the 'reduce_cnn' sub-config by reading the run's own config.yaml
    (expected to sit next to model_weights.pt, i.e. in checkpoint_file.parent)
    and copying its 'model.cnn' section as-is.

    model.cnn already uses exactly the layout _pretrained_checkpoint_features
    (generate_dataset.py) and training_runner.py expect for data.reduce_cnn:
      - flat keys for 'simple_cnn' (channels, kernel_size, use_batchnorm,
        pool_every_block, dropout, fc_hidden_dim live directly under cnn.)
      - a nested '<backbone>' sub-dict for 'resnet'/'myrtle' (e.g.
        cnn.resnet = {num_stages, blocks_per_stage, width_mult}).
    So rather than re-deriving that structure, we just copy model.cnn
    wholesale and add the two extra fields reduce_cnn needs on top
    (checkpoint, pretrained_num_classes).
    """
    config_path = checkpoint_file.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config.yaml next to checkpoint at: {config_path}"
        )

    with open(config_path, "r") as f:
        run_config = yaml.safe_load(f)

    try:
        model_cnn = run_config["model"]["cnn"]
        _ = model_cnn["backbone"]  # just validate it's present
    except KeyError as e:
        raise KeyError(
            f"config.yaml at {config_path} is missing expected model.cnn.{e} field"
        )

    reduce_cnn = dict(model_cnn)  # shallow copy: backbone, plus flat or nested params
    reduce_cnn["checkpoint"] = str(checkpoint_file)
    reduce_cnn["pretrained_num_classes"] = pretrained_num_classes
    return reduce_cnn

def main():
    parser = argparse.ArgumentParser(description="Generate reduced CIFAR-10 metaclass dataset plots.")
    
    # New checkpoint parameters
    parser.add_argument("--checkpoint_type", type=str, 
                        default=None, help="Type of the checkpoint to use.")
    parser.add_argument("--checkpoint_id", type=str, 
                        default="training_2026_07_24-02_31_56",
                        help="ID of the checkpoint to use.")
    parser.add_argument("--pretrained_num_classes", type=int, default=3, 
                        help="Number of classes the checkpoint's head was trained with.")
    parser.add_argument("--plotting_num_classes", type=int, default=3,
                        help="Number of classes to plot (should match the checkpoint's head).")

    # Existing reduction parameters
    parser.add_argument("--reduce_dim", type=int, default=2, 
                        help="Target dimensionality for reduction (e.g., 2, 3, 50).")
    parser.add_argument("--plot_dim", type=int, default=2, choices=[2, 3],
                        help="Dimensionality of the scatter plot itself (2 or 3). "
                             "3 requires --reduce_dim >= 3, and produces a 3D plot "
                             "(file name gets a '_3d' suffix) instead of the default 2D one.")
    parser.add_argument("--reduce_method", type=str, default="umap", 
                        choices=["pca", "random_projection", "umap", "tsne"], 
                        help="Method for dimensionality reduction.")
    parser.add_argument("--metaclass_mapping_type", default=None,
                        help="Type of metaclass mapping to use.")
    parser.add_argument("--reduce_seed", type=int, default=0, 
                        help="Random seed for reproducibility in dimensionality reduction.")
    parser.add_argument("--out_dir", type=str, default=None, 
                        help="Directory to save the resulting plot. Defaults to the checkpoint's directory if left blank.")
    
    # GPU usage flag
    parser.add_argument("--use_gpu", action="store_true", default=True, 
                        help="Whether to use GPU. Defaults to True to match the YAML config.")
    parser.add_argument("--no_gpu", action="store_false", dest="use_gpu",
                        help="Disable GPU usage.")

    args = parser.parse_args()

    if args.checkpoint_type.endswith("always_bumps"):
        folder_type="with_bumps"
    else:
        folder_type="without_bumps"

    checkpoint_file = Path(f'/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/{folder_type}/{args.checkpoint_type}/{args.checkpoint_id}/model_weights.pt') 
    
    # Output dir defaults to the checkpoint's directory unless explicitly overridden
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = checkpoint_file.parent
    
    out_dir = out_dir / f"plots_reduce_dim_{args.reduce_method}"

    if args.metaclass_mapping_type is None or args.metaclass_mapping_type == "":
        if args.checkpoint_type.endswith("intuitive"):
            metaclass_mapping = mapping_intuitive
        elif args.checkpoint_type.endswith("default0"):
            metaclass_mapping = mapping_default0
        elif args.checkpoint_type.endswith("default1"):
            metaclass_mapping = mapping_default1
        elif args.checkpoint_type.endswith("default2"):
            metaclass_mapping = mapping_default2
        elif args.checkpoint_type.endswith("default3"):
            metaclass_mapping = mapping_default3
    else:
        if args.metaclass_mapping_type == "intuitive":
            metaclass_mapping = mapping_intuitive
            out_dir = out_dir / "metaclasses_intuitive"
        elif args.metaclass_mapping_type == "default0":
            metaclass_mapping = mapping_default0
            out_dir = out_dir / "metaclasses_default0"
        elif args.metaclass_mapping_type == "default1":
            metaclass_mapping = mapping_default1
            out_dir = out_dir / "metaclasses_default1"
        elif args.metaclass_mapping_type == "default2":
            metaclass_mapping = mapping_default2
            out_dir = out_dir / "metaclasses_default2"
        elif args.metaclass_mapping_type == "default3":
            metaclass_mapping = mapping_default3
            out_dir = out_dir / "metaclasses_default3"
        else:
            raise ValueError(f"Unknown metaclass mapping type: {args.metaclass_mapping_type}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull the reduce_cnn spec (backbone + architecture params) straight from
    # the checkpoint's own config.yaml, rather than hardcoding resnet18.
    reduce_cnn = load_reduce_cnn_config(checkpoint_file, args.pretrained_num_classes)

    # Reconstruct the configuration requested, injecting the sys args
    config = {
        "use_gpu": args.use_gpu,
        "data": {
            "dataset_name": "cifar10",
            "data_dir": "./data",
            "num_classes": args.plotting_num_classes,
            "reduce_dim": args.reduce_dim,
            "reduce_method": args.reduce_method,
            "reduce_source": "pretrained_checkpoint",
            "reduce_seed": args.reduce_seed,
            "plot_dim": args.plot_dim,
            "reduce_standardize": True,
            "reduce_cnn": reduce_cnn,
        },
    }
    
    # Add the key conditionally
    if args.plotting_num_classes == 3:
        config["data"]["metaclass_mapping"] = metaclass_mapping

    print(f"Generating dataset bundle... (Method: {args.reduce_method}, Dim: {args.reduce_dim})")
    print(f"Using checkpoint: {checkpoint_file}")
    print(f"Using backbone from config.yaml: {reduce_cnn['backbone']}")
    print(f"Saving plots to: {out_dir}")
    
    # build_dataset_bundle handles the data loading, the CNN feature extraction, 
    # the dimensionality reduction, the metaclass mapping, and regenerating the 2D plots.
    _ = build_dataset_bundle(config, out_dir)
    
    plot_suffix = "_3d" if args.plot_dim == 3 else ""
    expected_plot_name = f"cifar10_reduced_{args.reduce_method}_dim{args.reduce_dim}_samples{plot_suffix}.png"
    print(f"Done! Plot should be saved as: {out_dir.resolve() / expected_plot_name}")

if __name__ == "__main__":
    main()