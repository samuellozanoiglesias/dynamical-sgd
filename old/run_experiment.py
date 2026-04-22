#!/usr/bin/env python3
"""
Unified Neural Collapse runner.

This script always delegates model creation, training, and NC metrics computation
to utils/experimental_runner.py. It supports:
- Single run from one config.
- Multi-run bump-mode orchestration.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER_SCRIPT = PROJECT_ROOT / "utils" / "experimental_runner.py"


def parse_override_value(raw: str) -> Any:
    lower = raw.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        return raw


def apply_override(config: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_config_with_overrides(config_path: Path, overrides: List[str]) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value format.")
        key, raw_value = override.split("=", 1)
        apply_override(config, key, parse_override_value(raw_value))

    return config


def bool_to_env(value: Any, default: bool) -> str:
    val = default if value is None else bool(value)
    return "true" if val else "false"


def to_int(value: Any, fallback: int) -> int:
    return int(fallback if value is None else value)


def to_float(value: Any, fallback: float) -> float:
    return float(fallback if value is None else value)


def build_output_dir(config: Dict[str, Any], config_path: Path) -> Path:
    output_cfg = config.get("output", {})
    output_base = Path(str(output_cfg.get("output_dir", "outputs")))
    experiment_name = str(output_cfg.get("experiment_name", "dynamical_sgd_experiment"))
    config_name = output_cfg.get("config_name") or config_path.stem
    timestamp = output_cfg.get("experiment_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = output_base / experiment_name / str(config_name) / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_notebook_env(config: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    dynamics_cfg = config.get("dynamics", {})

    dataset_name = str(data_cfg.get("dataset_name", "spiral")).strip().lower()
    if dataset_name not in {"mnist", "spiral"}:
        raise ValueError(f"Unsupported dataset_name='{dataset_name}'. Expected 'mnist' or 'spiral'.")

    total_steps = to_int(training_cfg.get("training_steps", training_cfg.get("total_steps", 10000)), 10000)
    steps_per_epoch = to_int(training_cfg.get("steps_per_epoch", min(200, total_steps)), min(200, total_steps))
    if total_steps <= 0 or steps_per_epoch <= 0:
        raise ValueError(
            f"Invalid step configuration: total_steps={total_steps}, steps_per_epoch={steps_per_epoch}."
        )
    epochs = int(math.ceil(total_steps / steps_per_epoch))

    num_classes = to_int(data_cfg.get("num_classes", 10), 10)
    default_architecture = "mlp" if dataset_name == "spiral" else "resnet18"
    architecture = str(model_cfg.get("architecture", default_architecture)).strip().lower()

    env = os.environ.copy()
    env["NC_RESULTS_DIR"] = str(output_dir)
    env["NC_DATASET_NAME"] = dataset_name
    env["NC_NUM_CLASSES"] = str(num_classes)

    env["NC_EPOCHS"] = str(epochs)
    env["NC_STEPS_PER_EPOCH"] = str(steps_per_epoch)
    env["NC_TOTAL_TRAINING_STEPS"] = str(total_steps)
    env["NC_BATCH_SIZE"] = str(to_int(training_cfg.get("batch_size", 128), 128))

    env["NC_BUMPS_BEFORE_TPT"] = bool_to_env(dynamics_cfg.get("bumps_before_TPT"), True)
    env["NC_BUMPS_AT_TPT"] = bool_to_env(dynamics_cfg.get("bumps_at_TPT"), False)
    env["NC_PERIOD_LENGTH"] = str(to_int(dynamics_cfg.get("period_length", 2000), 2000))
    env["NC_W_MAX"] = str(to_float(dynamics_cfg.get("w_max", 50.0), 50.0))
    env["NC_TPT_ACCURACY_THRESHOLD"] = str(to_float(dynamics_cfg.get("tpt_accuracy_threshold", 1.0), 1.0))

    env["NC_MODEL_ARCHITECTURE"] = architecture
    env["NC_MLP_HIDDEN_DIM"] = str(to_int(model_cfg.get("nn_width", 512), 512))
    env["NC_MLP_NUM_HIDDEN_LAYERS"] = str(to_int(model_cfg.get("num_hidden_layers", 2), 2))
    env["NC_MLP_USE_BIAS"] = bool_to_env(model_cfg.get("use_bias"), True)

    if dataset_name == "mnist":
        env["NC_DATA_DIR"] = str(data_cfg.get("data_dir", "./data"))
    else:
        env["NC_SPIRAL_POINTS_PER_CLASS"] = str(to_int(data_cfg.get("points_per_class", 1000), 1000))
        env["NC_SPIRAL_REVOLUTIONS"] = str(to_float(data_cfg.get("revolutions", 4.0), 4.0))
        env["NC_SPIRAL_NOISE_STD"] = str(to_float(data_cfg.get("noise_std", 0.1), 0.1))
        env["NC_SPIRAL_TEST_RATIO"] = str(to_float(data_cfg.get("test_ratio", 0.25), 0.25))
        env["NC_SPIRAL_RANDOM_SEED"] = str(to_int(data_cfg.get("random_seed", 0), 0))
        env["NC_SPIRAL_RANDOMIZE_OFFSETS"] = bool_to_env(data_cfg.get("randomize_offsets"), False)
        offsets = data_cfg.get("angular_offsets")
        env["NC_SPIRAL_ANGULAR_OFFSETS"] = "" if offsets is None else ",".join(str(v) for v in offsets)
        env["NC_SPIRAL_MIN_RADIUS"] = str(to_float(data_cfg.get("min_radius", 0.05), 0.05))

    return env


def run_one(config_path: Path, overrides: List[str]) -> Path:
    config = load_config_with_overrides(config_path, overrides)
    output_dir = build_output_dir(config, config_path)

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    env = build_notebook_env(config, output_dir)

    print("=" * 80)
    print("NOTEBOOK-ONLY EXPERIMENT")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {env['NC_DATASET_NAME']}")
    print(f"Model architecture: {env['NC_MODEL_ARCHITECTURE']}")
    print(f"Total steps: {env['NC_TOTAL_TRAINING_STEPS']}")
    print(f"Steps per epoch: {env['NC_STEPS_PER_EPOCH']}")
    print(f"Epochs: {env['NC_EPOCHS']}")
    print("=" * 80)

    cmd = [sys.executable, "-m", "utils.experimental_runner"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)

    summary = {
        "mode": "notebook_only",
        "runner_script": str(RUNNER_SCRIPT),
        "results_dir": str(output_dir),
        "dataset": env["NC_DATASET_NAME"],
        "architecture": env["NC_MODEL_ARCHITECTURE"],
        "total_steps": int(env["NC_TOTAL_TRAINING_STEPS"]),
        "steps_per_epoch": int(env["NC_STEPS_PER_EPOCH"]),
        "epochs": int(env["NC_EPOCHS"]),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Experiment completed successfully.")
    print(f"Results saved to: {output_dir}")
    return output_dir


def resolve_cluster_base_dir(cluster: str) -> Path:
    if cluster == "brigit":
        return Path("/mnt/lustre/home/samuloza")
    if cluster == "cuenca":
        return Path("/")
    return Path(".")


def mode_plan(mode: str) -> List[tuple[str, bool, bool, str]]:
    all_modes = [
        ("never", False, False, "nc_never"),
        ("always", True, True, "nc_always"),
        ("tpt_only", False, True, "nc_tpt_only"),
        ("pre_tpt", True, False, "nc_pre_tpt"),
        
    ]
    if mode == "all":
        return all_modes
    for item in all_modes:
        if item[0] == mode:
            return [item]
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified neural collapse experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="config/spiral_mlp_3layer.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Compatibility alias for --config (expects file name under config/)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with dot notation (e.g. --override model.nn_width=200)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "always", "tpt_only", "pre_tpt", "never", "all"],
        help="single for one run, or bump-mode sweep like the old run_nc_experiment.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default="local",
        choices=["local", "brigit", "cuenca"],
        help="Cluster preset for output base path in sweep mode.",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="data/samuel_lozano/dynamical-sgd",
        help="Base output directory for sweep mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed override in sweep mode.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Unused; kept for compatibility.",
    )
    args = parser.parse_args()

    if args.config_file is not None:
        raw_config = Path("config") / args.config_file
    else:
        raw_config = Path(args.config)

    config_path = raw_config if raw_config.is_absolute() else PROJECT_ROOT / raw_config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if args.mode == "single":
        run_one(config_path, args.override)
        return

    config_name = config_path.stem
    experiment_timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    base_dir = resolve_cluster_base_dir(args.cluster)
    output_base = (base_dir / args.output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NEURAL COLLAPSE SWEEP")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Cluster: {args.cluster}")
    print(f"Config: {config_path}")
    print(f"Output base: {output_base}")
    print(f"Timestamp: {experiment_timestamp}")
    if args.seed is not None:
        print(f"Seed override: {args.seed}")
    print("=" * 80)

    for idx, (mode_name, bumps_before, bumps_after, experiment_name) in enumerate(mode_plan(args.mode), start=1):
        print()
        print("=" * 80)
        print(f"Sweep Experiment {idx}: {mode_name}")
        print("=" * 80)

        mode_overrides = [
            f"output.output_dir={output_base}",
            f"output.experiment_name={experiment_name}",
            f"output.config_name={config_name}",
            f"output.experiment_timestamp={experiment_timestamp}",
            f"dynamics.bumps_before_TPT={'true' if bumps_before else 'false'}",
            f"dynamics.bumps_at_TPT={'true' if bumps_after else 'false'}",
        ]
        if args.seed is not None:
            mode_overrides.append(f"training.random_seed={args.seed}")
            mode_overrides.append(f"data.random_seed={args.seed}")

        combined_overrides = list(args.override) + mode_overrides
        run_one(config_path, combined_overrides)

    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
