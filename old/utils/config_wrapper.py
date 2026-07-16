from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional, List


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RunnerSettings:
    dataset_name: str
    num_classes: int
    im_size: int
    padded_im_size: int
    input_ch: int

    loss_name: str
    lr_decay: float
    epochs: int
    steps_per_epoch: int
    total_training_steps: int
    epochs_lr_decay: list[int]
    batch_size: int
    eval_batch_size: int
    optimizer_type: str
    learning_rate: float
    momentum: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    l2_reg: float
    gradient_clipping: float | None

    bumps_before_tpt: bool
    bumps_at_tpt: bool
    tpt_accuracy_threshold: float
    bump_period_length: int
    bump_w_max: float

    model_architecture: str
    mlp_hidden_dim: int
    mlp_num_hidden_layers: int
    mlp_use_bias: bool

    results_dir: Path
    data_dir: Path

    spiral_points_per_class: int
    spiral_revolutions: float
    spiral_noise_std: float
    spiral_test_ratio: float
    spiral_random_seed: int
    spiral_randomize_offsets: bool
    spiral_min_radius: float
    spiral_angular_offsets: Optional[List[float]]

    debug: bool

    @classmethod
    def from_env(cls) -> "RunnerSettings":
        debug_mode = _env_bool("NC_DEBUG", True)

        dataset_name = os.environ.get("NC_DATASET_NAME", "mnist").strip().lower()
        if dataset_name not in {"mnist", "spiral"}:
            raise ValueError(f"Unsupported NC_DATASET_NAME='{dataset_name}'.")

        model_architecture = os.environ.get("NC_MODEL_ARCHITECTURE", "resnet18").strip().lower()
        if model_architecture not in {"resnet18", "mlp"}:
            raise ValueError(f"Unsupported NC_MODEL_ARCHITECTURE='{model_architecture}'.")

        if dataset_name == "spiral" and model_architecture == "resnet18":
            im_size = 1
            padded_im_size = 1
            input_ch = 2
        else:
            im_size = 28
            padded_im_size = 32
            input_ch = 1

        loss_name = "CrossEntropyLoss"

        if "NC_EPOCHS" not in os.environ:
            raise ValueError("NC_EPOCHS environment variable not set.")
        if "NC_STEPS_PER_EPOCH" not in os.environ:
            raise ValueError("NC_STEPS_PER_EPOCH environment variable not set.")
        if "NC_TOTAL_TRAINING_STEPS" not in os.environ:
            raise ValueError("NC_TOTAL_TRAINING_STEPS environment variable not set.")

        epochs = int(os.environ["NC_EPOCHS"])
        steps_per_epoch = int(os.environ["NC_STEPS_PER_EPOCH"])
        total_training_steps = int(os.environ["NC_TOTAL_TRAINING_STEPS"])

        batch_size = int(os.environ.get("NC_BATCH_SIZE", "128"))
        if batch_size <= 0:
            raise ValueError("NC_BATCH_SIZE must be > 0")

        eval_batch_size_env = os.environ.get("NC_EVAL_BATCH_SIZE")
        if eval_batch_size_env is not None:
            eval_batch_size = int(eval_batch_size_env)
        else:
            # Debug mode should only reduce evaluation workload.
            eval_batch_size = min(batch_size, 32) if debug_mode else batch_size
        if eval_batch_size <= 0:
            raise ValueError("NC_EVAL_BATCH_SIZE must be > 0")

        offsets_raw = os.environ.get("NC_SPIRAL_ANGULAR_OFFSETS", "").strip()
        if offsets_raw:
            spiral_offsets = [float(v.strip()) for v in offsets_raw.split(",") if v.strip()]
        else:
            spiral_offsets = None

        results_dir = Path(os.environ.get("NC_RESULTS_DIR", "./results")).resolve()
        data_dir = Path(os.environ.get("NC_DATA_DIR", Path.cwd() / "data")).resolve()
        results_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Set learning_rate for MLP, lr for ResNet18
        if model_architecture == "mlp":
            learning_rate = float(os.environ.get("NC_LEARNING_RATE", "0.002"))
        else:
            learning_rate = float(os.environ.get("NC_LR", "0.0679"))

        return cls(
            dataset_name=dataset_name,
            num_classes=int(os.environ.get("NC_NUM_CLASSES", "10")),
            im_size=im_size,
            padded_im_size=padded_im_size,
            input_ch=input_ch,
            loss_name=loss_name,
            optimizer_type=os.environ.get("NC_OPTIMIZER_TYPE", "adam"),
            learning_rate=learning_rate,
            momentum=float(os.environ.get("NC_MOMENTUM", "0.9")),
            beta1=float(os.environ.get("NC_BETA1", "0.9")),
            beta2=float(os.environ.get("NC_BETA2", "0.999")),
            eps=float(os.environ.get("NC_EPS", "1e-8")),
            weight_decay=float(os.environ.get("NC_WEIGHT_DECAY", "0")),
            l2_reg=float(os.environ.get("NC_L2_REG", "0")),
            gradient_clipping=float(os.environ.get("NC_GRADIENT_CLIPPING", "nan")) if os.environ.get("NC_GRADIENT_CLIPPING") not in [None, "null", "", "nan"] else None,
            lr_decay=0.1,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            total_training_steps=total_training_steps,
            epochs_lr_decay=[epochs // 3, epochs * 2 // 3],
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            bumps_before_tpt=_env_bool("NC_BUMPS_BEFORE_tpt", False),
            bumps_at_tpt=_env_bool("NC_BUMPS_AT_tpt", False),
            tpt_accuracy_threshold=float(os.environ.get("NC_tpt_ACCURACY_THRESHOLD", "1.0")),
            bump_period_length=int(os.environ.get("NC_PERIOD_LENGTH", "2000")),
            bump_w_max=float(os.environ.get("NC_W_MAX", "50.0")),
            model_architecture=model_architecture,
            mlp_hidden_dim=int(os.environ.get("NC_MLP_HIDDEN_DIM", "512")),
            mlp_num_hidden_layers=int(os.environ.get("NC_MLP_NUM_HIDDEN_LAYERS", "2")),
            mlp_use_bias=_env_bool("NC_MLP_USE_BIAS", True),
            results_dir=results_dir,
            data_dir=data_dir,
            spiral_points_per_class=int(os.environ.get("NC_SPIRAL_POINTS_PER_CLASS", "1000")),
            spiral_revolutions=float(os.environ.get("NC_SPIRAL_REVOLUTIONS", "4.0")),
            spiral_noise_std=float(os.environ.get("NC_SPIRAL_NOISE_STD", "0.1")),
            spiral_test_ratio=float(os.environ.get("NC_SPIRAL_TEST_RATIO", "0.25")),
            spiral_random_seed=int(os.environ.get("NC_SPIRAL_RANDOM_SEED", "0")),
            spiral_randomize_offsets=_env_bool("NC_SPIRAL_RANDOMIZE_OFFSETS", False),
            spiral_min_radius=float(os.environ.get("NC_SPIRAL_MIN_RADIUS", "0.05")),
            spiral_angular_offsets=spiral_offsets,
            debug=debug_mode,
        )
