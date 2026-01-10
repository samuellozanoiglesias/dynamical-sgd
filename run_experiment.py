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
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.spiral_classifier import SpiralClassifier
from utils.data_utils import generate_spiral_data, create_train_test_split, DataLoader
from utils.visualization import (
    plot_spiral_dataset, plot_decision_boundary, plot_training_curves,
    plot_class_focus_dynamics, setup_matplotlib_style
)
from utils.metrics import MetricsTracker
from config.experiment_config import ExperimentConfig, load_config_with_overrides
from analysis.neural_collapse import (
    NeuralCollapseAnalyzer, NeuralCollapseSnapshot, create_video_from_images
)
from analysis.neural_collapse_integration import (
    extract_penultimate_features, plot_nc_metrics_evolution, plot_angle_convergence_evolution
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


def setup_device(config: ExperimentConfig):
    """Setup JAX device configuration."""
    if config.device == "gpu":
        if jax.device_count('gpu') > 0:
            print(f"Using GPU: {jax.devices('gpu')[0]}")
        else:
            print("GPU requested but not available, falling back to CPU")
    elif config.device == "cpu":
        jax.config.update('jax_platform_name', 'cpu')
        print("Using CPU")
    else:  # auto
        if jax.device_count('gpu') > 0:
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


def generate_and_prepare_data(config: ExperimentConfig):
    """Generate and prepare dataset according to configuration."""
    logging.info("Generating spiral dataset...")
    
    # Use training.random_seed if provided, otherwise fall back to data.random_seed
    # This ensures the seed affects both data generation and training
    seed = config.training.random_seed if config.training.random_seed is not None else config.data.random_seed
    if seed is not None:
        logging.info(f"Using random seed for data generation: {seed}")
    
    # Generate training data
    X_train, Y_train = generate_spiral_data(
        points_per_class=config.data.points_per_class,
        num_classes=config.data.num_classes,
        revolutions=config.data.revolutions,
        noise_std=config.data.noise_std,
        random_seed=seed,
        angular_offsets=getattr(config.data, 'angular_offsets', None),
        randomize_offsets=getattr(config.data, 'randomize_offsets', False)
    )
    
    # Generate test data with different seed
    test_seed = seed + 1 if seed is not None else 1
    X_test, Y_test = generate_spiral_data(
        points_per_class=config.data.points_per_class,
        num_classes=config.data.num_classes,
        revolutions=config.data.revolutions,
        noise_std=config.data.noise_std,
        random_seed=test_seed,
        angular_offsets=getattr(config.data, 'angular_offsets', None),
        randomize_offsets=getattr(config.data, 'randomize_offsets', False)
    )
    
    logging.info(f"Generated training set: {X_train.shape[0]} samples")
    logging.info(f"Generated test set: {X_test.shape[0]} samples")
    
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
    
    # Create classifier
    classifier = SpiralClassifier(
        points_per_class=config.data.points_per_class,
        num_classes=config.data.num_classes,
        nn_width=config.model.nn_width,
        learning_rate=config.optimizer.learning_rate,
        optimizer_type=config.optimizer.optimizer_type,
        period_length=config.dynamics.period_length,
        l2_reg=config.optimizer.l2_reg,
        random_seed=config.random_seed,
        label=config.output.experiment_name,
        track_weight_diff=config.analysis.track_weight_diff,
        weight_diff_step_interval=config.analysis.weight_diff_step_interval,
        real_time_visualization=config.visualization.real_time_visualization,
        vis_step_interval=config.visualization.vis_step_interval
    )
    
    # Setup metrics tracking
    metrics_tracker = MetricsTracker(track_calibration=True)
    
    # Validate dynamics configuration and warn about potential issues
    if config.dynamics.enable_dynamics:
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
        if config.dynamics.period_length < 500:
            logging.warning("⚠️  VERY SHORT PERIOD LENGTH!")
            logging.warning(f"   Period length: {config.dynamics.period_length} steps")
            logging.warning(f"   Each bump lasts only {config.dynamics.period_length} steps")
            logging.warning(f"   Ramp up/down happens in {config.dynamics.period_length//2} steps each")
            logging.warning(f"   This may be too fast for observable bump dynamics")
            logging.warning(f"   Recommended: period_length >= 1000 for clear effects")
        elif config.dynamics.period_length < 1000:
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
        nc_analyzer = NeuralCollapseAnalyzer(
            num_classes=config.data.num_classes,
            feature_dim=config.model.nn_width
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
    
    # Training metrics storage
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    metric_steps = []  # Track actual step numbers for metrics
    
    with tqdm(total=config.training.total_steps, desc="Training") as pbar:
        for t in range(config.training.total_steps):
            if config.dynamics.enable_dynamics:
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
                # Balanced sampling
                data_loader = DataLoader(X_train, Y_train, config.training.batch_size)
                X_batch, Y_batch = data_loader.get_balanced_batch()
            
            # Update parameters
            params, opt_state, grads = classifier.update_step(params, opt_state, X_batch, Y_batch)
            
            # Track metrics only at validation intervals
            if t % config.training.validation_interval == 0 or t == config.training.total_steps - 1:
                current_metrics = metrics_tracker.update(
                    classifier.model[1], params, X_train, Y_train, X_test, Y_test, grads, l2_reg=config.optimizer.l2_reg
                )
                
                train_losses.append(current_metrics['train_loss'])
                train_accuracies.append(current_metrics['train_accuracy'])
                test_losses.append(current_metrics['test_loss'])
                test_accuracies.append(current_metrics['test_accuracy'])
                metric_steps.append(t)  # Record the actual step number
                
                # Update progress bar with latest metrics
                pbar.set_postfix({
                    'Train Loss': f"{current_metrics['train_loss']:.4f}",
                    'Train Acc': f"{current_metrics['train_accuracy']:.3f}",
                    'Test Acc': f"{current_metrics['test_accuracy']:.3f}"
                })
            
            # Capture Neural Collapse snapshot if enabled
            if nc_analyzer is not None and t in snapshot_epochs:
                # Extract features from penultimate layer
                features = extract_penultimate_features(params, X_train)
                labels = jnp.argmax(Y_train, axis=1)
                
                # Compute class means
                class_means = []
                for c in range(config.data.num_classes):
                    mask = labels == c
                    if jnp.sum(mask) > 0:
                        class_mean = jnp.mean(features[mask], axis=0)
                    else:
                        class_mean = jnp.zeros(config.model.nn_width)
                    class_means.append(class_mean)
                class_means = jnp.stack(class_means)
                
                # Extract classifiers (W) and biases (b) from last layer
                # params[-1] is (W, b) where W is (hidden_dim, num_classes)
                classifiers = params[-1][0].T  # (num_classes, hidden_dim)
                biases = params[-1][1]  # (num_classes,)
                
                # Create snapshot
                snapshot = NeuralCollapseSnapshot(
                    epoch=t,
                    features=features,
                    labels=labels,
                    class_means=class_means,
                    classifiers=classifiers,
                    biases=biases,
                    num_classes=config.data.num_classes,
                    feature_dim=config.model.nn_width
                )
                nc_analyzer.snapshots.append(snapshot)
                
                # Compute and log metrics
                nc_metrics = nc_analyzer.compute_nc_metrics(snapshot)
                logging.info(
                        f"Step {t} NC Metrics: "
                        f"NC1={nc_metrics['nc1_within_class_variance']:.6f}, "
                        f"NC2={nc_metrics['nc2_etf_alignment']:.4f}, "
                        f"NC3={nc_metrics['nc3_self_duality']:.4f}"
                    )
            
            pbar.update(1)
    
    # Store final parameters
    classifier.last_params = params
    
    # Final evaluation
    final_train_acc = float(classifier.accuracy(params, X_train, Y_train))
    final_test_acc = float(classifier.accuracy(params, X_test, Y_test))
    
    logging.info(f"Final training accuracy: {final_train_acc:.4f}")
    logging.info(f"Final test accuracy: {final_test_acc:.4f}")
    
    return classifier, train_losses, train_accuracies, test_losses, test_accuracies, metric_steps, nc_analyzer


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
    output_dir: Path
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
            classifier.model[1],
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
        plot_training_curves(
            train_losses,
            train_accuracies,
            test_losses,
            test_accuracies,
            metric_steps=metric_steps,
            period_length=config.dynamics.period_length if config.dynamics.enable_dynamics else None,
            title="Training Progress"
        )
        plt.savefig(output_dir / "training_curves.png", dpi=config.visualization.figure_dpi)
        plt.close()
    
    # Plot class focus dynamics
    if config.dynamics.enable_dynamics:
        steps = jnp.arange(0, config.training.total_steps, 100)
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


def save_results(
    config: ExperimentConfig,
    classifier: SpiralClassifier,
    train_losses: list,
    train_accuracies: list,
    test_losses: list,
    test_accuracies: list,
    metric_steps: list,
    output_dir: Path
):
    """Save experiment results to files."""
    logging.info("Saving results...")
    
    if config.output.save_metrics:
        # Save results as CSV file
        with open(output_dir / 'results.csv', 'w') as f:
            # Write CSV header
            f.write("Step,Train_Acc,Test_Acc,Train_Loss,Test_Loss\n")
            
            # Write metrics at evaluation points
            for i in range(len(train_accuracies)):
                step = metric_steps[i] if i < len(metric_steps) else i * config.training.validation_interval
                train_loss = train_losses[i] if i < len(train_losses) else 0.0
                test_loss = test_losses[i] if i < len(test_losses) else 0.0
                train_acc = train_accuracies[i]
                test_acc = test_accuracies[i]
                f.write(f"{step},{train_acc:.6f},{test_acc:.6f},{train_loss:.6f},{test_loss:.6f}\n")
        
        logging.info(f"Results saved to: {output_dir / 'results.csv'}")
    
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
        ) if classifier.last_params else 0
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
        
        # Generate data
        X_train, Y_train, X_test, Y_test = generate_and_prepare_data(config)
        
        # Save dataset plots immediately after generation
        logging.info("Saving initial dataset visualizations...")
        setup_matplotlib_style(config.visualization.plot_style)
        
        plot_spiral_dataset(X_train, Y_train, title="Training Dataset")
        plt.savefig(output_dir / "training_dataset.png", dpi=config.visualization.figure_dpi)
        plt.close()
        logging.info(f"Saved training dataset plot: {output_dir / 'training_dataset.png'}")
        
        plot_spiral_dataset(X_test, Y_test, title="Test Dataset")
        plt.savefig(output_dir / "test_dataset.png", dpi=config.visualization.figure_dpi)
        plt.close()
        logging.info(f"Saved test dataset plot: {output_dir / 'test_dataset.png'}")
        
        # Train model
        classifier, train_losses, train_accuracies, test_losses, test_accuracies, metric_steps, nc_analyzer = create_and_train_model(
            config, X_train, Y_train, X_test, Y_test, output_dir
        )
        
        # Create visualizations
        create_visualizations(
            config, classifier, X_train, Y_train, X_test, Y_test,
            train_losses, train_accuracies, test_losses, test_accuracies, metric_steps, output_dir
        )
        
        # Save results
        save_results(
            config, classifier, train_losses, train_accuracies,
            test_losses, test_accuracies, metric_steps, output_dir
        )
        
        # Save Neural Collapse analysis if enabled
        if nc_analyzer is not None and len(nc_analyzer.snapshots) > 0:
            logging.info("Saving Neural Collapse analysis...")
            nc_analyzer.save_snapshots(output_dir / 'nc_snapshots.pkl')
            
            # Compute metrics history
            nc_metrics_history = [
                (snap.epoch, nc_analyzer.compute_nc_metrics(snap))
                for snap in nc_analyzer.snapshots
            ]
            
            # Plot NC metrics evolution
            plot_nc_metrics_evolution(nc_metrics_history, output_dir)
            
            # Plot angle convergence evolution (true geometric angles in R^p)
            plot_angle_convergence_evolution(nc_analyzer, output_dir)
            
            # Visualize selected snapshots with FIXED axes for proper comparison
            if config.visualization.save_nc_visualizations:
                # Compute consistent axis limit based on ETF scale
                # ETF represents the theoretical target configuration
                etf = nc_analyzer.compute_simplex_etf(num_classes=config.data.num_classes)
                etf_scale = float(jnp.max(jnp.abs(etf))) * 1.5  # 50% margin for clarity
                
                logging.info(f"Using fixed axis limit: ±{etf_scale:.3f} (based on ETF scale)")
                
                # Create folders for 3D and 2D visualizations
                viz_3d_dir = output_dir / '3d-snapshots'
                viz_2d_dir = output_dir / '2d-snapshots'
                viz_3d_dir.mkdir(exist_ok=True)
                viz_2d_dir.mkdir(exist_ok=True)
                
                # Visualize ALL snapshots with fixed axes (every nc_snapshot_interval steps)
                logging.info(f"Creating {len(nc_analyzer.snapshots)} 3D and 2D visualizations...")
                logging.info("  3D visualizations in: 3d-snapshots/")
                logging.info("  2D visualizations in: 2d-snapshots/ (better for angle interpretation)")
                
                for i, snapshot in enumerate(nc_analyzer.snapshots):
                    if (i + 1) % 50 == 0:  # Progress indicator every 50 snapshots
                        logging.info(f"  Progress: {i + 1}/{len(nc_analyzer.snapshots)} visualizations")
                    
                    # Create 3D visualization
                    nc_analyzer.visualize_neural_collapse(
                        snapshot=snapshot,
                        selected_classes=list(range(min(3, config.data.num_classes))),
                        samples_per_class=20,
                        save_path=viz_3d_dir / f'nc_viz_step_{snapshot.epoch:08d}.png',
                        axis_limit=etf_scale  # Fixed axes across all snapshots
                    )
                    
                    # Create 2D visualization (easier to interpret angles)
                    nc_analyzer.visualize_neural_collapse_2d(
                        snapshot=snapshot,
                        selected_classes=list(range(min(3, config.data.num_classes))),
                        samples_per_class=20,
                        save_path=viz_2d_dir / f'nc_viz_step_{snapshot.epoch:08d}.png',
                        axis_limit=etf_scale  # Fixed axes across all snapshots
                    )
                
                logging.info(f"Completed all {len(nc_analyzer.snapshots)} visualizations")
                
                # Create videos from snapshots
                logging.info("Creating videos from snapshots...")
                
                # Create 3D video
                video_3d_path = output_dir / 'nc_evolution_3d.mp4'
                create_video_from_images(
                    image_dir=viz_3d_dir,
                    output_path=video_3d_path,
                    pattern='nc_viz_step_*.png',
                    fps=2.0  # 0.5 seconds per frame
                )
                
                # Create 2D video
                video_2d_path = output_dir / 'nc_evolution_2d.mp4'
                create_video_from_images(
                    image_dir=viz_2d_dir,
                    output_path=video_2d_path,
                    pattern='nc_viz_step_*.png',
                    fps=2.0  # 0.5 seconds per frame
                )
                
                logging.info("Videos created successfully!")
            
            logging.info(f"Neural Collapse analysis saved ({len(nc_analyzer.snapshots)} snapshots)")
        
        logging.info("Experiment completed successfully!")
        logging.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()