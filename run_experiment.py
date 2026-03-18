#!/usr/bin/env python3
"""
Main experiment script for dynamical SGD analysis.

This script provides a clean interface for running single experiments with
the dynamical SGD approach. It supports configuration files and command-line
overrides for easy experimentation.

Usage:
    python run_experiment.py --config config/default_config.yaml
    python run_experiment.py --config config/default_config.yaml --model.nn_width 200 --training.batch_size 100
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.spiral_classifier import SpiralClassifier
from utils.data_utils import load_dataset, create_train_test_split, DataLoader
from utils.visualization import (
    plot_spiral_dataset, plot_decision_boundary, plot_training_curves,
    plot_training_curves_with_classes, plot_class_focus_dynamics, setup_matplotlib_style
)
from utils.metrics import MetricsTracker
from config.experiment_config import ExperimentConfig, load_config_with_overrides
from analysis.neural_collapse import NeuralCollapseAnalyzer
from analysis.nc_visualization import create_nc_figure1_evolution
from analysis.neural_collapse_integration import (
    plot_nc_metrics_evolution, plot_angle_convergence_evolution
)
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('experiment.log')
        ]
    )


def add_experiment_log_file(output_dir: Path):
    """Add file handler to save training process to experiment directory."""
    # Add handler for training_process.log in the experiment directory
    file_handler = logging.FileHandler(output_dir / 'training_process.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    logging.info(f"Training process log: {output_dir / 'training_process.log'}")


def _gpu_available() -> bool:
    """Return True if a JAX GPU backend is present, without raising."""
    try:
        return jax.device_count('gpu') > 0
    except RuntimeError:
        return False


def setup_device(config: ExperimentConfig):
    """Setup JAX device configuration."""
    if config.device == "gpu":
        if _gpu_available():
            print(f"Using GPU: {jax.devices('gpu')[0]}")
        else:
            print("GPU requested but not available, falling back to CPU")
    elif config.device == "cpu":
        jax.config.update('jax_platform_name', 'cpu')
        print("Using CPU")
    else:  # auto
        if _gpu_available():
            print(f"Auto-detected GPU: {jax.devices('gpu')[0]}")
        else:
            print("Auto-detected CPU")


def create_output_directory(config: ExperimentConfig) -> Path:
    """Create output directory for experiment results."""
    from datetime import datetime
    
    # Use provided timestamp or create new one
    if config.output.experiment_timestamp:
        timestamp = config.output.experiment_timestamp
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Structure: outputs/experiment_name/config_name/experiment_timestamp/
    # If config_name is not provided, skip that level
    base_path = Path(config.output.output_dir) / config.output.experiment_name
    
    if config.output.config_name:
        base_path = base_path / config.output.config_name
    
    output_dir = base_path / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(output_dir / "config.yaml")
    
    logging.info(f"Output directory: {output_dir}")
    return output_dir


def run_notebook_compatible_mnist_pipeline(config: ExperimentConfig, output_dir: Path) -> None:
    """Run MNIST experiment using the exact notebook-compatible torch pipeline."""
    notebook_script = project_root / "neuralcollapse_notebook.py"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NC_RESULTS_DIR"] = str(results_dir)
    # Keep the original notebook default to preserve notebook-equivalent behavior.
    env.setdefault("NC_EPOCHS", "10")

    logging.info("Running notebook-compatible MNIST path via neuralcollapse_notebook.py")
    logging.info("This path uses the notebook metric formulas and plotting structure.")

    if config.dynamics.dynamics_enabled():
        logging.info(
            "Notebook-compatible MNIST path ignores bump dynamics overrides "
            "and follows notebook training behavior exactly."
        )

    cmd = [sys.executable, str(notebook_script)]
    subprocess.run(cmd, cwd=project_root, env=env, check=True)

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        import json

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        summary_data["mnist_execution_mode"] = "notebook_compatible"
        summary_data["notebook_results_dir"] = str(results_dir)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    logging.info(f"Notebook-compatible outputs saved under: {results_dir}")


def generate_and_prepare_data(config: ExperimentConfig):
    """Generate and prepare dataset according to configuration."""
    dataset_name = getattr(config.data, 'dataset_name', 'spiral')
    logging.info(f"Loading {dataset_name} dataset...")
    
    # Use training.random_seed if provided, otherwise fall back to data.random_seed
    # This ensures the seed affects both data generation and training
    seed = config.training.random_seed if config.training.random_seed is not None else config.data.random_seed
    if seed is not None:
        logging.info(f"Using random seed for data generation: {seed}")
    
    # Create dataset configuration for load_dataset function
    if dataset_name.lower() == 'spiral':
        dataset_config = {
            "points_per_class": config.data.points_per_class,
            "num_classes": config.data.num_classes,
            "revolutions": config.data.revolutions,
            "noise_std": config.data.noise_std,
            "test_ratio": getattr(config.data, 'test_ratio', 0.25),
            "angular_offsets": getattr(config.data, 'angular_offsets', None),
            "randomize_offsets": getattr(config.data, 'randomize_offsets', False),
            "min_radius": getattr(config.data, 'min_radius', 0.05)
        }
    elif dataset_name.lower() == 'mnist':
        dataset_config = {
            "data_dir": getattr(config.data, 'data_dir', './data'),
            "flatten": getattr(config.data, 'flatten', True),
            "normalize": getattr(config.data, 'normalize', True),
            "download": getattr(config.data, 'download', True)
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'spiral', 'mnist'")
    
    # Load dataset using unified function
    (X_train, Y_train), (X_test, Y_test) = load_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        random_seed=seed
    )
    
    logging.info(f"Loaded {dataset_name} dataset:")
    logging.info(f"  Training set: {X_train.shape[0]} samples, shape: {X_train.shape}")
    logging.info(f"  Test set: {X_test.shape[0]} samples, shape: {X_test.shape}")
    logging.info(f"  Number of classes: {Y_train.shape[1]}")
    
    return X_train, Y_train, X_test, Y_test


def create_and_train_model(
    config: ExperimentConfig,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    Y_test: jnp.ndarray,
    output_dir: Path
):
    """Create and train the spiral classifier."""
    logging.info("Creating classifier...")
    
    # Get input dimension from data
    # For flattened data (MLP): (N, 784) -> input_dim = 784
    # For images (CNN/DenseNet): (N, 28, 28, 1) -> input_dim = 784 (product of all dims except batch)
    if len(X_train.shape) == 2:
        input_dim = X_train.shape[1]  # Flattened: (N, features)
    else:
        # Multi-dimensional input (images): (N, H, W, C)
        input_dim = int(jnp.prod(jnp.array(X_train.shape[1:])))  # Product of H*W*C
    
    # Handle points_per_class for different datasets
    points_per_class = config.data.points_per_class
    if points_per_class is None:
        # For MNIST or other datasets where points_per_class is not applicable
        # Use a reasonable default (this is mainly for classifier labeling/tracking)
        points_per_class = 100
    
    classifier = SpiralClassifier(
        input_dim=input_dim,           # NEW: Pass input dimension
        points_per_class=points_per_class,
        num_classes=config.data.num_classes,
        nn_width=config.model.nn_width,
        num_hidden_layers=getattr(config.model, 'num_hidden_layers', 1),
        learning_rate=config.optimizer.learning_rate,
        optimizer_type=config.optimizer.optimizer_type,
        period_length=config.dynamics.period_length,
        l2_reg=config.optimizer.l2_reg,
        random_seed=config.training.random_seed if config.training.random_seed is not None else config.random_seed,
        label=config.output.experiment_name,
        track_weight_diff=config.analysis.track_weight_diff,
        weight_diff_step_interval=config.analysis.weight_diff_step_interval,
        real_time_visualization=config.visualization.real_time_visualization,
        vis_step_interval=config.visualization.vis_step_interval,
        use_batchnorm=getattr(config.model, 'use_batchnorm', True),
        use_bias=getattr(config.model, 'use_bias', True),
        # Optimizer parameters
        momentum=getattr(config.optimizer, 'momentum', 0.9),
        beta1=getattr(config.optimizer, 'beta1', 0.9),
        beta2=getattr(config.optimizer, 'beta2', 0.999),
        eps=getattr(config.optimizer, 'eps', 1e-8),
        # Initialization parameters
        weight_init_scale=getattr(config.model, 'weight_init_scale', 1.0),
        # Architecture selection
        architecture=getattr(config.model, 'architecture', 'mlp'),
        growth_rate=getattr(config.model, 'growth_rate', 12),
        compression=getattr(config.model, 'compression', 0.5),
        dropout_rate=getattr(config.model, 'dropout_rate', 0.0)
    )
    
    # Setup metrics tracking
    # Prefer explicit config value so low-memory runs can tune this easily.
    architecture = getattr(config.model, 'architecture', 'mlp').lower()
    default_eval_batch = 256 if architecture == 'densenet40' else 5000
    eval_batch_size = int(getattr(config.training, 'eval_batch_size', default_eval_batch))
    metrics_tracker = MetricsTracker(
        track_calibration=True,
        track_per_class=bool(getattr(config.analysis, 'track_per_class', True)),
        num_classes=config.data.num_classes,
        eval_batch_size=eval_batch_size
    )
    
    # Validate dynamics configuration and warn about potential issues
    if config.dynamics.dynamics_enabled():
        # Compute example class weights to check distribution
        example_weights = classifier.compute_class_weights(
            0, 0, config.dynamics.w_max, config.dynamics.period_length
        )
        example_counts = (example_weights * config.training.batch_size).astype(int)
        
        logging.info("=" * 80)
        logging.info("DYNAMIC CLASS FOCUS CONFIGURATION")
        logging.info("=" * 80)
        logging.info(f"w_max: {config.dynamics.w_max}")
        logging.info(f"period_length: {config.dynamics.period_length}")
        logging.info(f"batch_size: {config.training.batch_size}")
        logging.info(f"Example normalized weights: {example_weights}")
        logging.info(f"Example class counts in batch: {example_counts}")
        
        # Check for extreme imbalance
        min_samples = int(jnp.min(example_counts))
        max_samples = int(jnp.max(example_counts))
        imbalance_ratio = max_samples / max(min_samples, 1)
        
        logging.info(f"Imbalance ratio (max/min): {imbalance_ratio:.1f}:1")
        
        # Warning for extreme imbalance
        if imbalance_ratio > 20:
            logging.warning("⚠️  EXTREME CLASS IMBALANCE DETECTED!")
            logging.warning(f"   Non-focused classes get only {min_samples} samples per batch")
            logging.warning(f"   Focused class gets {max_samples} samples per batch")
            logging.warning(f"   This extreme imbalance (>{imbalance_ratio:.0f}:1) may prevent proper learning")
            logging.warning(f"   Consider reducing w_max to 3-10 for typical bump effects")
        elif imbalance_ratio > 10:
            logging.warning("⚠️  HIGH CLASS IMBALANCE DETECTED")
            logging.warning(f"   Non-focused classes get only {min_samples} samples per batch")
            logging.warning(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
            logging.warning(f"   Typical values: w_max=3-10 (gives ~2-5:1 imbalance)")
        else:
            logging.info(f"✓ Imbalance ratio is reasonable for observing bump dynamics")
        
        # Check if samples per class are sufficient
        if min_samples < 3:
            logging.warning("⚠️  NON-FOCUSED CLASSES GET VERY FEW SAMPLES!")
            logging.warning(f"   Only {min_samples} samples per non-focused class")
            logging.warning(f"   This may cause unstable gradients or prevent learning")
        
        # Check period length vs total training
        num_full_periods = config.training.total_steps / config.dynamics.period_length
        num_periods_per_class = num_full_periods / config.data.num_classes
        logging.info(f"Number of full bump periods: {num_full_periods:.1f}")
        logging.info(f"Periods per class: {num_periods_per_class:.1f}")
        
        # Warning for short period length
        if config.dynamics.period_length < 100:
            logging.warning("⚠️  VERY SHORT PERIOD LENGTH!")
            logging.warning(f"   Period length: {config.dynamics.period_length} steps")
            logging.warning(f"   Each bump lasts only {config.dynamics.period_length} steps")
            logging.warning(f"   Ramp up/down happens in {config.dynamics.period_length//2} steps each")
            logging.warning(f"   This may be too fast for observable bump dynamics")
            logging.warning(f"   Recommended: period_length >= 1000 for clear effects")
        elif config.dynamics.period_length < 500:
            logging.warning("⚠️  SHORT PERIOD LENGTH")
            logging.warning(f"   Period length: {config.dynamics.period_length} steps")
            logging.warning(f"   Consider period_length >= 1000 for clearer bump dynamics")
        
        # Warning for validation interval vs period length mismatch
        if config.training.validation_interval >= config.dynamics.period_length:
            logging.warning("⚠️  VALIDATION INTERVAL TOO LARGE!")
            logging.warning(f"   Validation interval: {config.training.validation_interval}")
            logging.warning(f"   Period length: {config.dynamics.period_length}")
            logging.warning(f"   You may MISS the bump peak entirely!")
            logging.warning(f"   Recommended: validation_interval <= period_length/4")
            logging.warning(f"   (e.g., validation_interval={config.dynamics.period_length//4})")
        elif config.training.validation_interval > config.dynamics.period_length / 2:
            logging.warning("⚠️  VALIDATION INTERVAL MAY MISS BUMP DETAILS")
            logging.warning(f"   Validation interval: {config.training.validation_interval}")
            logging.warning(f"   Period length: {config.dynamics.period_length}")
            logging.warning(f"   Consider validation_interval <= {config.dynamics.period_length//4} for better resolution")
        
        if num_full_periods < 5:
            logging.warning("⚠️  FEW TRAINING PERIODS")
            logging.warning(f"   Only {num_full_periods:.1f} complete periods")
            logging.warning(f"   Consider increasing total_steps or decreasing period_length")
        
        logging.info("=" * 80)
    
    # Setup Neural Collapse analyzer if enabled
    nc_analyzer = None
    if config.analysis.track_neural_collapse:
        logging.info("Neural Collapse tracking enabled")
        
        # Determine feature dimension based on architecture
        architecture = getattr(config.model, 'architecture', 'mlp').lower()
        if architecture == 'densenet40':
            # Calculate DenseNet40 feature dimension
            # Initial: 16, Block1: 16+12*12=160, Trans1: 80, Block2: 80+12*12=224, Trans2: 112, Block3: 112+12*12=256
            num_init_features = getattr(config.model, 'num_init_features', 16)
            growth_rate = getattr(config.model, 'growth_rate', 12)
            compression = getattr(config.model, 'compression', 0.5)
            block_config = getattr(config.model, 'block_config', (12, 12, 12))
            
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                num_features += num_layers * growth_rate
                # Apply transition compression except after last block
                if i < len(block_config) - 1:
                    num_features = int(num_features * compression)
            
            feature_dim = num_features
            logging.info(f"DenseNet-40 feature dimension: {feature_dim}")
        else:
            feature_dim = config.model.nn_width
            logging.info(f"MLP feature dimension: {feature_dim}")
        
        nc_analyzer = NeuralCollapseAnalyzer(
            num_classes=config.data.num_classes,
            feature_dim=feature_dim,
            num_hidden_layers=getattr(config.model, 'num_hidden_layers', 1),
            use_batchnorm=getattr(config.model, 'use_batchnorm', True),
            use_bias=getattr(config.model, 'use_bias', True),
            classifier=classifier,  # Pass classifier object for architecture-specific handling
            feature_batch_size=eval_batch_size,
        )
        # Define snapshot epochs
        snapshot_interval = config.analysis.nc_snapshot_interval
        snapshot_epochs = list(range(0, config.training.total_steps, snapshot_interval))
        snapshot_epochs.append(config.training.total_steps - 1)
        logging.info(f"Will capture NC snapshots at steps: {snapshot_epochs[:5]}...")
    
    # Training loop
    logging.info("Starting training...")
    
    # Use random seed from training config, with fallback to data config
    # This should match the seed used for data generation
    seed = config.training.random_seed if config.training.random_seed is not None else config.data.random_seed
    if seed is None:
        seed = 0
    rng = random.PRNGKey(seed)
    logging.info(f"Using random seed for training: {seed}")
    
    params = classifier.initialize_params()
    opt_state = classifier.optimizer.init(params)
    
    # Prepare class indices for sampling
    from utils.data_utils import compute_class_indices
    class_indices = compute_class_indices(Y_train)
    
    # Create DataLoader once (not inside the training loop!)
    # This prevents recreating it with the same seed on every iteration
    data_loader = DataLoader(X_train, Y_train, config.training.batch_size, random_seed=seed)
    
    # Training metrics storage
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    metric_steps = []  # Track actual step numbers for metrics
    
    # Track when 100% training accuracy is reached
    step_100_acc = None  # Step at which train_acc reaches 100%
    terminal_full_training_reached = False
    progress_bar_position = 0  # Track current progress bar position
    
    with tqdm(total=config.training.total_steps, desc="Training") as pbar:
        for t in range(config.training.total_steps):
            # Determine if we should apply bumping dynamics based on TPT state
            if terminal_full_training_reached:
                # After TPT: use bumps_at_TPT setting
                apply_bumping = config.dynamics.bumps_at_TPT
            else:
                # Before TPT: use bumps_before_TPT setting
                apply_bumping = config.dynamics.bumps_before_TPT
            
            if apply_bumping:
                # Dynamic class focus
                class_focus = int((t // config.dynamics.period_length) % config.data.num_classes)
                current_weights = classifier.compute_class_weights(
                    t % config.dynamics.period_length,
                    class_focus,
                    config.dynamics.w_max,
                    config.dynamics.period_length
                )
                class_counts = (current_weights * config.training.batch_size).astype(int)
                
                # Sample batch with dynamic class weights
                rng, key = random.split(rng)
                X_batch, Y_batch = classifier.sample_by_class(
                    X_train, Y_train, class_counts, key, class_indices
                )
            else:
                # Balanced sampling (using pre-created data_loader)
                X_batch, Y_batch = data_loader.get_balanced_batch()
            
            # Update parameters
            params, opt_state, grads = classifier.update_step(params, opt_state, X_batch, Y_batch)
            
            # Track metrics only at validation intervals
            if t % config.training.validation_interval == 0 or t == config.training.total_steps - 1:
                current_metrics = metrics_tracker.update(
                    classifier.forward_fn, params, X_train, Y_train, X_test, Y_test, grads, l2_reg=config.optimizer.l2_reg
                )
                
                train_losses.append(current_metrics['train_loss'])
                train_accuracies.append(current_metrics['train_accuracy'])
                test_losses.append(current_metrics['test_loss'])
                test_accuracies.append(current_metrics['test_accuracy'])
                metric_steps.append(t)  # Record the actual step number
                
                # Check if TPT accuracy threshold reached for the first time
                tpt_threshold = getattr(config.dynamics, 'tpt_accuracy_threshold', 1.0)
                if not terminal_full_training_reached and current_metrics['train_accuracy'] >= tpt_threshold:
                    step_100_acc = t
                    terminal_full_training_reached = True
                    logging.info(f"🎯 Terminal Phase Training ({tpt_threshold*100:.1f}% accuracy) reached at step {t}")
                    if config.dynamics.dynamics_enabled():
                        if config.dynamics.bumps_at_TPT:
                            logging.info(f"   Continuing bumps after {tpt_threshold*100:.1f}% accuracy (bumps_at_TPT=true)")
                        else:
                            logging.info(f"   Stopping bumps after {tpt_threshold*100:.1f}% accuracy (bumps_at_TPT=false)")
                
                # Update progress bar with latest metrics and refresh display
                pbar.set_postfix({
                    'Train Loss': f"{current_metrics['train_loss']:.8f}",
                    'Train Acc': f"{current_metrics['train_accuracy']:.3f}",
                    'Test Acc': f"{current_metrics['test_accuracy']:.3f}"
                })
                # Update progress bar to current step position
                steps_to_update = (t + 1) - progress_bar_position
                pbar.update(steps_to_update)
                progress_bar_position = t + 1
                
                # Log simple training progress
                logging.info(f"Step {t}: Train Loss={current_metrics['train_loss']:.6f}, "
                           f"Train Acc={current_metrics['train_accuracy']:.6f}, "
                           f"Test Loss={current_metrics['test_loss']:.6f}, "
                           f"Test Acc={current_metrics['test_accuracy']:.6f}")
            
            # Capture Neural Collapse snapshot if enabled (but don't log detailed metrics)
            if nc_analyzer is not None and t in snapshot_epochs:
                # Use the analyzer's method to extract features, compute metrics, and create snapshot
                # CRITICAL: Must use FULL training dataset (X_train, Y_train), NOT batch (X_batch, Y_batch)!
                # NC metrics require all samples from all classes to compute proper class means and scatter matrices
                snapshot = nc_analyzer.extract_features_and_classifiers(
                    model_fn=None,  # Not used
                    params=params,
                    X=X_train,  # ✅ FULL training dataset (NOT X_batch!)
                    Y=Y_train,  # ✅ FULL training labels (NOT Y_batch!)
                    epoch=t,
                    X_test=X_test,  # ✅ FULL test dataset for NC4 metric
                    Y_test=Y_test   # ✅ FULL test labels for NC4 metric
                )
                
                # Clear GPU memory after NC snapshot (memory-intensive operation)
                import gc
                gc.collect()
                
                # Store NC metrics for later saving (don't log them)
    
    # Store final parameters
    classifier.last_params = params
    
    # Clear GPU cache before final evaluation to avoid OOM
    import gc
    gc.collect()
    
    # Final evaluation with batching (handled inside accuracy method)
    final_train_acc = float(classifier.accuracy(params, X_train, Y_train))
    gc.collect()  # Clear memory between train and test
    final_test_acc = float(classifier.accuracy(params, X_test, Y_test))
    
    logging.info(f"Final training accuracy: {final_train_acc:.8f}")
    logging.info(f"Final test accuracy: {final_test_acc:.8f}")
    
    if step_100_acc is not None:
        logging.info(f"Terminal Phase Training (100% accuracy) was reached at step: {step_100_acc}")
    else:
        logging.info("Terminal Full Training (100% accuracy) was NOT reached during training")
    
    return classifier, train_losses, train_accuracies, test_losses, test_accuracies, metric_steps, nc_analyzer, step_100_acc, metrics_tracker


def create_visualizations(
    config: ExperimentConfig,
    classifier: SpiralClassifier,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    Y_test: jnp.ndarray,
    train_losses: list,
    train_accuracies: list,
    test_losses: list,
    test_accuracies: list,
    metric_steps: list,
    output_dir: Path,
    step_100_acc: Optional[int] = None,
    all_per_class_metrics: Optional[List[Dict]] = None
):
    """Create and save visualizations."""
    logging.info("Creating visualizations...")
    
    # Setup matplotlib style
    setup_matplotlib_style(config.visualization.plot_style)
    
    # Note: Dataset plots are now saved at the beginning of the experiment
    # (see main function after data generation)
    
    # Plot decision boundary
    if config.visualization.save_decision_boundaries and classifier.last_params is not None:
        plot_decision_boundary(
            classifier.forward_fn,
            classifier.last_params,
            X_train,
            Y_train,
            X_test,
            Y_test,
            title="Final Decision Boundary"
        )
        plt.savefig(output_dir / "decision_boundary.png", dpi=config.visualization.figure_dpi)
        plt.close()
    
    # Plot training curves
    if config.visualization.save_training_curves:
        tpt_threshold = getattr(config.dynamics, 'tpt_accuracy_threshold', 1.0)
        
        # Check if we have per-class data
        if all_per_class_metrics is not None and len(all_per_class_metrics) > 0:
            # Extract per-class arrays
            num_classes = config.data.num_classes
            train_losses_per_class = [[] for _ in range(num_classes)]
            train_accuracies_per_class = [[] for _ in range(num_classes)]
            test_losses_per_class = [[] for _ in range(num_classes)]
            test_accuracies_per_class = [[] for _ in range(num_classes)]
            
            for per_class in all_per_class_metrics:
                for c in range(num_classes):
                    train_losses_per_class[c].append(per_class[f'Train_Loss_Class_{c}'])
                    train_accuracies_per_class[c].append(per_class[f'Train_Acc_Class_{c}'])
                    test_losses_per_class[c].append(per_class[f'Test_Loss_Class_{c}'])
                    test_accuracies_per_class[c].append(per_class[f'Test_Acc_Class_{c}'])
            
            # Use the new 4x2 layout function
            plot_training_curves_with_classes(
                train_losses,
                train_accuracies,
                test_losses,
                test_accuracies,
                train_losses_per_class=train_losses_per_class,
                train_accuracies_per_class=train_accuracies_per_class,
                test_losses_per_class=test_losses_per_class,
                test_accuracies_per_class=test_accuracies_per_class,
                metric_steps=metric_steps,
                period_length=config.dynamics.period_length if config.dynamics.dynamics_enabled() else None,
                title="Training Progress with Per-Class Breakdown",
                step_100_acc=step_100_acc,
                tpt_threshold=tpt_threshold,
                num_classes=num_classes
            )
        else:
            # Fall back to original function
            plot_training_curves(
                train_losses,
                train_accuracies,
                test_losses,
                test_accuracies,
                metric_steps=metric_steps,
                period_length=config.dynamics.period_length if config.dynamics.dynamics_enabled() else None,
                title="Training Progress",
                step_100_acc=step_100_acc,
                tpt_threshold=tpt_threshold
            )
        plt.savefig(output_dir / "training_curves.png", dpi=config.visualization.figure_dpi)
        plt.close()
    
    # Plot class focus dynamics
    if config.dynamics.dynamics_enabled():
        # Adaptive step size for visualization based on total training steps
        viz_step_size = max(1, config.training.total_steps // 50)  # At least 1, aim for ~50 points
        steps = jnp.arange(0, config.training.total_steps, viz_step_size)
        plot_class_focus_dynamics(
            steps,
            classifier.compute_class_weights,
            [config.dynamics.w_max],
            config.dynamics.period_length,
            config.data.num_classes,
            title="Dynamic Class Focus"
        )
        plt.savefig(output_dir / "class_focus_dynamics.png", dpi=config.visualization.figure_dpi)
        plt.close()


def save_nc_results(nc_metrics_history: list, output_dir: Path):
    """Save Neural Collapse metrics to CSV file."""
    if not nc_metrics_history:
        return
    
    with open(output_dir / 'nc_results.csv', 'w') as f:
        # Write CSV header with all NC metrics
        header = ("Step,NC1,NC2_CV_means,NC2_CV_cls,NC2_Std_means,NC2_Std_cls,"
                 "NC2_Mean_means,NC2_Mean_cls,NC2_Target,NC3,NC4")
        f.write(header + "\n")
        
        # Write metrics data
        for epoch, metrics in nc_metrics_history:
            nc4_value = metrics.get('nc7_ncc_mismatch', 0.0)  # NC4 was called nc7_ncc_mismatch internally
            line = (
                f"{epoch},{metrics['nc1_variability']:.6f},"
                f"{metrics['nc2_equinorm_cv_means']:.6f},{metrics['nc2_equinorm_cv_classifiers']:.6f},"
                f"{metrics['nc2_equiangular_std_means']:.6f},{metrics['nc2_equiangular_std_classifiers']:.6f},"
                f"{metrics['nc2_equiangular_mean_means']:.6f},{metrics['nc2_equiangular_mean_classifiers']:.6f},"
                f"{metrics['nc2_equiangular_target']:.6f},{metrics['nc3_self_duality']:.6f},{nc4_value:.6f}"
            )
            f.write(line + "\n")
    
    logging.info(f"NC results saved to: {output_dir / 'nc_results.csv'}")

def compute_per_class_metrics(classifier, params, X_train, Y_train, X_test, Y_test, num_classes):
    """Compute per-class accuracy and loss metrics."""
    from jax import nn
    
    # Convert one-hot to class indices
    train_labels = jnp.argmax(Y_train, axis=1)
    test_labels = jnp.argmax(Y_test, axis=1)
    
    per_class_metrics = {}
    
    for c in range(num_classes):
        # Training metrics for class c
        train_mask = train_labels == c
        if jnp.sum(train_mask) > 0:
            X_train_c = X_train[train_mask]
            Y_train_c = Y_train[train_mask]
            
            train_logits_c = classifier.forward_fn(params, X_train_c)
            train_loss_c = float(-jnp.mean(jnp.sum(Y_train_c * nn.log_softmax(train_logits_c), axis=1)))
            train_acc_c = float(jnp.mean(jnp.argmax(train_logits_c, axis=1) == jnp.argmax(Y_train_c, axis=1)))
        else:
            train_loss_c = 0.0
            train_acc_c = 0.0
        
        # Test metrics for class c
        test_mask = test_labels == c
        if jnp.sum(test_mask) > 0:
            X_test_c = X_test[test_mask]
            Y_test_c = Y_test[test_mask]
            
            test_logits_c = classifier.forward_fn(params, X_test_c)
            test_loss_c = float(-jnp.mean(jnp.sum(Y_test_c * nn.log_softmax(test_logits_c), axis=1)))
            test_acc_c = float(jnp.mean(jnp.argmax(test_logits_c, axis=1) == jnp.argmax(Y_test_c, axis=1)))
        else:
            test_loss_c = 0.0
            test_acc_c = 0.0
        
        per_class_metrics[f'Train_Loss_Class_{c}'] = train_loss_c
        per_class_metrics[f'Train_Acc_Class_{c}'] = train_acc_c
        per_class_metrics[f'Test_Loss_Class_{c}'] = test_loss_c
        per_class_metrics[f'Test_Acc_Class_{c}'] = test_acc_c
    
    return per_class_metrics

def save_results(
    config: ExperimentConfig,
    classifier: SpiralClassifier,
    train_losses: list,
    train_accuracies: list,
    test_losses: list,
    test_accuracies: list,
    metric_steps: list,
    output_dir: Path,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    Y_test: jnp.ndarray,
    step_100_acc: Optional[int] = None,
    all_per_class_metrics: Optional[List[Dict]] = None
):
    """Save experiment results to files."""
    logging.info("Saving results...")
    
    if config.output.save_metrics:
        # Use pre-computed per-class metrics if available, otherwise compute them
        if all_per_class_metrics is None:
            all_per_class_metrics = []
            for i in range(len(metric_steps)):
                # Get params at this step (use final params as approximation)
                step_per_class = compute_per_class_metrics(
                    classifier, classifier.last_params, X_train, Y_train, X_test, Y_test, config.data.num_classes
                )
                all_per_class_metrics.append(step_per_class)
        
        # Save training_results.csv file with per-class metrics
        with open(output_dir / 'training_results.csv', 'w') as f:
            # Write CSV header
            header = "Step,Train_Acc,Test_Acc,Train_Loss,Test_Loss"
            for c in range(config.data.num_classes):
                header += f",Train_Loss_Class_{c},Train_Acc_Class_{c},Test_Loss_Class_{c},Test_Acc_Class_{c}"
            f.write(header + "\n")
            
            # Write metrics at evaluation points
            for i in range(len(train_accuracies)):
                step = metric_steps[i] if i < len(metric_steps) else i * config.training.validation_interval
                train_loss = train_losses[i] if i < len(train_losses) else 0.0
                test_loss = test_losses[i] if i < len(test_losses) else 0.0
                train_acc = train_accuracies[i]
                test_acc = test_accuracies[i]
                
                line = f"{step},{train_acc:.6f},{test_acc:.6f},{train_loss:.6f},{test_loss:.6f}"
                
                # Add per-class metrics
                if i < len(all_per_class_metrics):
                    per_class = all_per_class_metrics[i]
                    for c in range(config.data.num_classes):
                        line += f",{per_class[f'Train_Loss_Class_{c}']:.6f}"
                        line += f",{per_class[f'Train_Acc_Class_{c}']:.6f}"
                        line += f",{per_class[f'Test_Loss_Class_{c}']:.6f}"
                        line += f",{per_class[f'Test_Acc_Class_{c}']:.6f}"
                else:
                    # Fill with zeros if no per-class data
                    for c in range(config.data.num_classes):
                        line += ",0.0,0.0,0.0,0.0"
                
                f.write(line + "\n")
        
        logging.info(f"Training results saved to: {output_dir / 'training_results.csv'}")
    
    if config.output.save_final_model and classifier.last_params is not None:
        with open(output_dir / 'final_model_params.pkl', 'wb') as f:
            pickle.dump(classifier.last_params, f)
    
    # Save experiment summary
    summary = {
        'config': config.to_dict(),
        'final_train_accuracy': train_accuracies[-1] if train_accuracies else 0,
        'final_test_accuracy': test_accuracies[-1] if test_accuracies else 0,
        'total_parameters': sum(
            p.size for p in jax.tree_util.tree_leaves(classifier.last_params)
        ) if classifier.last_params else 0,
        'step_100_acc': step_100_acc
    }
    
    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    """Main experiment function."""
    parser = argparse.ArgumentParser(description="Run dynamical SGD experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Add support for configuration overrides
    parser.add_argument(
        "--override",
        action="append",
        help="Override configuration parameters (e.g., --override model.nn_width=200)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Parse overrides
        overrides = {}
        if args.override:
            for override in args.override:
                key, value = override.split('=')
                # Try to convert to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                        # Otherwise keep as string
                
                overrides[key] = value
        
        # Load configuration
        config = load_config_with_overrides(args.config, overrides)
        logging.info(f"Loaded configuration from {args.config}")
        if overrides:
            logging.info(f"Applied overrides: {overrides}")
        
        # Setup device
        setup_device(config)
        
        # Create output directory
        output_dir = create_output_directory(config)
        
        # Add training process log to experiment directory
        add_experiment_log_file(output_dir)

        # MNIST path: execute the notebook-compatible implementation directly.
        if getattr(config.data, 'dataset_name', 'spiral').lower() == 'mnist':
            run_notebook_compatible_mnist_pipeline(config, output_dir)
            logging.info("Experiment completed successfully!")
            logging.info(f"Results saved to: {output_dir}")
            return
        
        # Generate data
        X_train, Y_train, X_test, Y_test = generate_and_prepare_data(config)
        
        # Save dataset plots immediately after generation
        logging.info("Saving initial dataset visualizations...")
        setup_matplotlib_style(config.visualization.plot_style)
        
        # Only create scatter plots for 2D data (spiral dataset)
        if X_train.shape[1] == 2:
            # Spiral dataset - create scatter plots
            plot_spiral_dataset(X_train, Y_train, title="Training Dataset")
            plt.savefig(output_dir / "training_dataset.png", dpi=config.visualization.figure_dpi)
            plt.close()
            logging.info(f"Saved training dataset plot: {output_dir / 'training_dataset.png'}")
            
            plot_spiral_dataset(X_test, Y_test, title="Test Dataset")
            plt.savefig(output_dir / "test_dataset.png", dpi=config.visualization.figure_dpi)
            plt.close()
            logging.info(f"Saved test dataset plot: {output_dir / 'test_dataset.png'}")
        else:
            # High-dimensional data (MNIST) - create sample visualization
            logging.info(f"High-dimensional data ({X_train.shape[1]}D) detected - creating sample visualizations...")
            
            # Create a grid of sample images for MNIST
            fig, axes = plt.subplots(2, 5, figsize=(12, 6))
            fig.suptitle("Training Dataset Samples", fontsize=16)
            
            # Show one sample from each class (for MNIST: 0-9)
            labels_train = jnp.argmax(Y_train, axis=1)
            for class_idx in range(min(10, Y_train.shape[1])):  # Up to 10 classes
                # Find first sample of this class
                mask = labels_train == class_idx
                if jnp.sum(mask) > 0:
                    sample_idx = jnp.where(mask)[0][0]
                    sample_data = X_train[sample_idx]
                    
                    # Reshape to image if it's flattened (MNIST: 784 -> 28x28)
                    if len(sample_data.shape) == 1 and sample_data.shape[0] == 784:
                        sample_image = sample_data.reshape(28, 28)
                    else:
                        sample_image = sample_data.squeeze()
                    
                    row, col = divmod(class_idx, 5)
                    axes[row, col].imshow(sample_image, cmap='gray')
                    axes[row, col].set_title(f'Class {class_idx}')
                    axes[row, col].axis('off')
                else:
                    # No samples for this class
                    row, col = divmod(class_idx, 5)
                    axes[row, col].text(0.5, 0.5, f'Class {class_idx}\n(No samples)', 
                                       ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / "training_dataset_samples.png", dpi=config.visualization.figure_dpi)
            plt.close()
            logging.info(f"Saved training dataset samples: {output_dir / 'training_dataset_samples.png'}")
            
            # Create dataset statistics plot
            labels_train = jnp.argmax(Y_train, axis=1)
            labels_test = jnp.argmax(Y_test, axis=1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Class distribution for training set
            unique_train, counts_train = jnp.unique(labels_train, return_counts=True)
            ax1.bar(unique_train, counts_train, alpha=0.7, color='skyblue')
            ax1.set_title('Training Set Class Distribution')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Number of Samples')
            ax1.grid(True, alpha=0.3)
            
            # Class distribution for test set  
            unique_test, counts_test = jnp.unique(labels_test, return_counts=True)
            ax2.bar(unique_test, counts_test, alpha=0.7, color='lightcoral')
            ax2.set_title('Test Set Class Distribution')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Number of Samples')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "dataset_statistics.png", dpi=config.visualization.figure_dpi)
            plt.close()
            logging.info(f"Saved dataset statistics: {output_dir / 'dataset_statistics.png'}")
        
        # Train model
        classifier, train_losses, train_accuracies, test_losses, test_accuracies, metric_steps, nc_analyzer, step_100_acc, metrics_tracker = create_and_train_model(
            config, X_train, Y_train, X_test, Y_test, output_dir
        )
        
        # Extract per-class metrics from metrics tracker history
        all_per_class_metrics = []
        if config.output.save_metrics and metrics_tracker.track_per_class:
            for i in range(len(metric_steps)):
                step_per_class = {}
                for c in range(config.data.num_classes):
                    # Extract metrics for class c at step i
                    step_per_class[f'Train_Loss_Class_{c}'] = metrics_tracker.metrics_history[f'train_loss_class_{c}'][i]
                    step_per_class[f'Train_Acc_Class_{c}'] = metrics_tracker.metrics_history[f'train_accuracy_class_{c}'][i]
                    step_per_class[f'Test_Loss_Class_{c}'] = metrics_tracker.metrics_history[f'test_loss_class_{c}'][i]
                    step_per_class[f'Test_Acc_Class_{c}'] = metrics_tracker.metrics_history[f'test_accuracy_class_{c}'][i]
                all_per_class_metrics.append(step_per_class)
        
        # Create visualizations
        create_visualizations(
            config, classifier, X_train, Y_train, X_test, Y_test,
            train_losses, train_accuracies, test_losses, test_accuracies, metric_steps, output_dir,
            step_100_acc=step_100_acc, all_per_class_metrics=all_per_class_metrics
        )
        
        # Save results
        save_results(
            config, classifier, train_losses, train_accuracies,
            test_losses, test_accuracies, metric_steps, output_dir,
            X_train, Y_train, X_test, Y_test, step_100_acc, all_per_class_metrics
        )
        
        # Save Neural Collapse analysis if enabled
        if nc_analyzer is not None and len(nc_analyzer.snapshots) > 0:
            logging.info("Saving Neural Collapse analysis...")
            # nc_analyzer.save_snapshots(output_dir / 'nc_snapshots.pkl')  # Disabled - too large
            
            # Compute metrics history
            nc_metrics_history = [
                (snap.epoch, nc_analyzer.compute_nc_metrics(snap))
                for snap in nc_analyzer.snapshots
            ]
            
            # Save NC metrics to CSV
            save_nc_results(nc_metrics_history, output_dir)
            
            # Plot NC metrics evolution with TPT threshold
            tpt_threshold = getattr(config.dynamics, 'tpt_accuracy_threshold', 1.0)
            plot_nc_metrics_evolution(nc_metrics_history, output_dir, step_100_acc, tpt_threshold)
            
            # Plot angle convergence evolution (true geometric angles in R^p)
            plot_angle_convergence_evolution(nc_analyzer, output_dir)
            
            # NOTE: 3D/2D visualizations disabled in simplified NC version
            # (visualize_neural_collapse methods are stubs)
            
            # Create NC Figure 1 visualization (Papyan et al. style)
            # Skip for large datasets to avoid OOM (PCA on 60k samples requires too much memory)
            num_samples = X_train.shape[0]
            if config.visualization.save_nc_visualizations and num_samples <= 5000:
                logging.info("Creating NC Figure 1 visualizations (Papyan et al., 2020 style)...")
                vis_interval = getattr(config.visualization, 'vis_step_interval', 1000)
                selected_epochs = [snap.epoch for snap in nc_analyzer.snapshots 
                                 if snap.epoch % vis_interval == 0 or snap.epoch == nc_analyzer.snapshots[-1].epoch]
                
                create_nc_figure1_evolution(
                    snapshots=nc_analyzer.snapshots,
                    output_dir=output_dir,
                    selected_epochs=selected_epochs,
                    normalize_vectors=True,  # Normalize for angular structure
                    show_etf=True,
                    show_features=True,
                    fps=2.0
                )
            elif config.visualization.save_nc_visualizations:
                logging.info(f"Skipping NC Figure 1 visualizations for large dataset ({num_samples} samples > 5000 threshold)")
            
            logging.info(f"Neural Collapse analysis saved ({len(nc_analyzer.snapshots)} snapshots)")
        
        logging.info("Experiment completed successfully!")
        logging.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()