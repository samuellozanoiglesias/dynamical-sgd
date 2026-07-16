#!/usr/bin/env python3
"""
Generate Averaged Training Figures

This script creates averaged plots from multiple experiment runs (seeds) for a given
neural collapse configuration. It processes:
- Training curves (loss, accuracy) -> training_curves_config_{config}.png  
- Neural collapse metrics -> nc_metrics_paper_style_config_{config}.png

The script reads data from:
- training_results.csv: Training metrics over time including per-class metrics
- nc_results.csv: Neural collapse metrics for NC analysis

Usage:
    python generate_averaged_figures.py --cluster brigit --mode nc_without_dynamics --config super_big_nn
    python generate_averaged_figures.py --cluster brigit --mode nc_with_bumps_tpt --config baseline
    python generate_averaged_figures.py --cluster brigit --mode nc_without_bumps_tpt  # Process all configs
    
Examples:
    # Process specific config
    python generate_averaged_figures.py --cluster brigit --mode nc_without_dynamics --config super_big_nn --save_csv
    
    # Process all configs in a mode
    python generate_averaged_figures.py --cluster brigit --mode nc_with_bumps_tpt --save_csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# We no longer need to import the neural collapse analyzer since we parse logs directly


@dataclass
class TrainingCurveData:
    """Container for training curve data from one experiment."""
    steps: np.ndarray
    train_acc: np.ndarray
    test_acc: np.ndarray
    train_loss: np.ndarray
    test_loss: np.ndarray
    # Per-class metrics (variable number of classes)
    train_acc_per_class: np.ndarray  # Shape: (n_steps, n_classes)
    test_acc_per_class: np.ndarray   # Shape: (n_steps, n_classes)
    train_loss_per_class: np.ndarray # Shape: (n_steps, n_classes)
    test_loss_per_class: np.ndarray  # Shape: (n_steps, n_classes)
    experiment_name: str


@dataclass
class NCMetricsData:
    """Container for neural collapse metrics data from one experiment."""
    steps: np.ndarray
    nc1: np.ndarray  # Within-class variance (variability collapse)
    nc2_cv_means: np.ndarray  # CV of class means norms (equinorm)
    nc2_cv_cls: np.ndarray  # CV of classifier norms (equinorm)  
    nc2_std_means: np.ndarray  # Std of class means cosines (equiangularity)
    nc2_std_cls: np.ndarray  # Std of classifier cosines (equiangularity)
    nc2_mean_means: np.ndarray  # Mean deviation from target for class means (equiangularity)
    nc2_mean_cls: np.ndarray  # Mean deviation from target for classifiers (equiangularity)
    nc2_target: np.ndarray  # Target value for equiangularity (-1/(C-1))
    nc3: np.ndarray  # Self-duality
    nc4: np.ndarray  # NCC mismatch
    experiment_name: str


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def find_experiment_directories(base_path: Path) -> List[Path]:
    """Find all experiment directories in the base path."""
    experiment_dirs = []
    for path in base_path.iterdir():
        if path.is_dir() and path.name.startswith('experiment_'):
            experiment_dirs.append(path)
    
    experiment_dirs.sort()  # Sort for consistent ordering
    logging.info(f"Found {len(experiment_dirs)} experiment directories")
    for exp_dir in experiment_dirs:
        logging.info(f"  - {exp_dir.name}")
    
    return experiment_dirs


def load_training_curves(experiment_dir: Path) -> TrainingCurveData:
    """Load training curves from training_results.csv."""
    csv_path = experiment_dir / "training_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        steps = df['Step'].values
        train_acc = df['Train_Acc'].values
        test_acc = df['Test_Acc'].values
        train_loss = df['Train_Loss'].values
        test_loss = df['Test_Loss'].values
        # Per-class metrics
        class_columns = [col for col in df.columns if col.startswith('Train_Loss_Class_')]
        num_classes = len(class_columns)
        if num_classes > 0:
            logging.info(f"Found per-class metrics for {num_classes} classes in {csv_path}")
            train_acc_per_class = np.zeros((len(steps), num_classes))
            test_acc_per_class = np.zeros((len(steps), num_classes))
            train_loss_per_class = np.zeros((len(steps), num_classes))
            test_loss_per_class = np.zeros((len(steps), num_classes))
            for c in range(num_classes):
                train_loss_per_class[:, c] = df[f'Train_Loss_Class_{c}'].values
                train_acc_per_class[:, c] = df[f'Train_Acc_Class_{c}'].values
                test_loss_per_class[:, c] = df[f'Test_Loss_Class_{c}'].values
                test_acc_per_class[:, c] = df[f'Test_Acc_Class_{c}'].values
        else:
            logging.warning(f"No per-class metrics found in {csv_path}, using total metrics only")
            num_classes = 1
            train_acc_per_class = np.zeros((len(steps), 1))
            test_acc_per_class = np.zeros((len(steps), 1))
            train_loss_per_class = np.zeros((len(steps), 1))
            test_loss_per_class = np.zeros((len(steps), 1))
        return TrainingCurveData(
            steps=steps,
            train_acc=train_acc,
            test_acc=test_acc,
            train_loss=train_loss,
            test_loss=test_loss,
            train_acc_per_class=train_acc_per_class,
            test_acc_per_class=test_acc_per_class,
            train_loss_per_class=train_loss_per_class,
            test_loss_per_class=test_loss_per_class,
            experiment_name=experiment_dir.name
        )
    else:
        # Try to load from nc_results.csv
        nc_csv_path = experiment_dir / "nc_results.csv"
        if not nc_csv_path.exists():
            raise FileNotFoundError(f"Neither training_results.csv nor nc_results.csv found in {experiment_dir}")
        df = pd.read_csv(nc_csv_path)
        # Use TrainingStep for steps
        steps = df['TrainingStep'].values
        # Use TrainLossEpoch and TrainAccEpoch for training metrics
        train_loss = df['TrainLossEpoch'].values if 'TrainLossEpoch' in df.columns else np.zeros_like(steps)
        train_acc = df['TrainAccEpoch'].values if 'TrainAccEpoch' in df.columns else np.zeros_like(steps)
        # Use TestLoss and TestAcc for test metrics
        test_loss = df['TestLoss'].values if 'TestLoss' in df.columns else np.zeros_like(steps)
        test_acc = df['TestAcc'].values if 'TestAcc' in df.columns else np.zeros_like(steps)
        # No per-class metrics in nc_results.csv, so fill with zeros
        num_classes = 1
        train_acc_per_class = np.zeros((len(steps), num_classes))
        test_acc_per_class = np.zeros((len(steps), num_classes))
        train_loss_per_class = np.zeros((len(steps), num_classes))
        test_loss_per_class = np.zeros((len(steps), num_classes))
        return TrainingCurveData(
            steps=steps,
            train_acc=train_acc,
            test_acc=test_acc,
            train_loss=train_loss,
            test_loss=test_loss,
            train_acc_per_class=train_acc_per_class,
            test_acc_per_class=test_acc_per_class,
            train_loss_per_class=train_loss_per_class,
            test_loss_per_class=test_loss_per_class,
            experiment_name=experiment_dir.name
        )


def load_nc_metrics(experiment_dir: Path) -> Optional[NCMetricsData]:
    """Load neural collapse metrics from nc_results.csv."""
    csv_path = experiment_dir / "nc_results.csv"
    
    if not csv_path.exists():
        logging.warning(f"nc_results.csv not found in {experiment_dir}")
        return None
    
    try:
        df = pd.read_csv(csv_path)

        # Map your CSV columns to expected names
        steps = df['TrainingStep'].values
        nc1_values = df['NC1'].values
        nc2_cv_means_values = df['NC2_Means_Equinorm'].values
        nc2_cv_cls_values = df['NC2_Classifiers_Equinorm'].values
        nc2_std_means_values = df['NC2_Means_Equiangularity'].values
        nc2_std_cls_values = df['NC2_Classifiers_Equiangularity'].values
        # The following columns are not present in your CSV, so fill with zeros
        nc2_mean_means_values = np.zeros_like(steps)
        nc2_mean_cls_values = np.zeros_like(steps)
        nc2_target_values = np.zeros_like(steps)
        nc3_values = df['NC3'].values
        nc4_values = df['NC4'].values

        logging.info(f"Loaded {len(steps)} NC metric entries from {csv_path}")

        return NCMetricsData(
            steps=np.array(steps),
            nc1=np.array(nc1_values),
            nc2_cv_means=np.array(nc2_cv_means_values),
            nc2_cv_cls=np.array(nc2_cv_cls_values),
            nc2_std_means=np.array(nc2_std_means_values),
            nc2_std_cls=np.array(nc2_std_cls_values),
            nc2_mean_means=np.array(nc2_mean_means_values),
            nc2_mean_cls=np.array(nc2_mean_cls_values),
            nc2_target=np.array(nc2_target_values),
            nc3=np.array(nc3_values),
            nc4=np.array(nc4_values),
            experiment_name=experiment_dir.name
        )

    except Exception as e:
        logging.error(f"Failed to load NC metrics from {experiment_dir}: {str(e)}")
        return None


def interpolate_to_common_steps(data_list: List[TrainingCurveData]) -> Tuple[np.ndarray, List[TrainingCurveData]]:
    """Interpolate all training curves to a common set of steps."""
    # Find the common step range
    min_steps = max(data.steps[0] for data in data_list)
    max_steps = min(data.steps[-1] for data in data_list)
    
    # Create common step array (use the densest sampling)
    all_steps = []
    for data in data_list:
        mask = (data.steps >= min_steps) & (data.steps <= max_steps)
        all_steps.extend(data.steps[mask])
    
    common_steps = np.unique(np.array(all_steps))
    common_steps = common_steps[(common_steps >= min_steps) & (common_steps <= max_steps)]
    
    # Interpolate each experiment to common steps
    interpolated_data = []
    for data in data_list:
        train_acc_interp = np.interp(common_steps, data.steps, data.train_acc)
        test_acc_interp = np.interp(common_steps, data.steps, data.test_acc)
        train_loss_interp = np.interp(common_steps, data.steps, data.train_loss)
        test_loss_interp = np.interp(common_steps, data.steps, data.test_loss)
        
        # Interpolate per-class data
        num_classes = data.train_acc_per_class.shape[1]
        train_acc_per_class_interp = np.zeros((len(common_steps), num_classes))
        test_acc_per_class_interp = np.zeros((len(common_steps), num_classes))
        train_loss_per_class_interp = np.zeros((len(common_steps), num_classes))
        test_loss_per_class_interp = np.zeros((len(common_steps), num_classes))
        
        for c in range(num_classes):
            train_acc_per_class_interp[:, c] = np.interp(common_steps, data.steps, data.train_acc_per_class[:, c])
            test_acc_per_class_interp[:, c] = np.interp(common_steps, data.steps, data.test_acc_per_class[:, c])
            train_loss_per_class_interp[:, c] = np.interp(common_steps, data.steps, data.train_loss_per_class[:, c])
            test_loss_per_class_interp[:, c] = np.interp(common_steps, data.steps, data.test_loss_per_class[:, c])
        
        interpolated_data.append(TrainingCurveData(
            steps=common_steps,
            train_acc=train_acc_interp,
            test_acc=test_acc_interp,
            train_loss=train_loss_interp,
            test_loss=test_loss_interp,
            train_acc_per_class=train_acc_per_class_interp,
            test_acc_per_class=test_acc_per_class_interp,
            train_loss_per_class=train_loss_per_class_interp,
            test_loss_per_class=test_loss_per_class_interp,
            experiment_name=data.experiment_name
        ))
    
    return common_steps, interpolated_data


def plot_averaged_training_curves(training_data: List[TrainingCurveData], output_dir: Path, config: str):
    """Create averaged training curves plot with total and per-class metrics in dynamic layout."""
    if not training_data:
        logging.warning("No training data to plot")
        return
    
    # Interpolate to common steps
    common_steps, interpolated_data = interpolate_to_common_steps(training_data)
    
    # Compute statistics for total metrics
    train_accs = np.array([data.train_acc for data in interpolated_data])
    test_accs = np.array([data.test_acc for data in interpolated_data])
    train_losses = np.array([data.train_loss for data in interpolated_data])
    test_losses = np.array([data.test_loss for data in interpolated_data])
    
    train_acc_mean = np.mean(train_accs, axis=0)
    train_acc_std = np.std(train_accs, axis=0)
    test_acc_mean = np.mean(test_accs, axis=0)
    test_acc_std = np.std(test_accs, axis=0)
    
    train_loss_mean = np.mean(train_losses, axis=0)
    train_loss_std = np.std(train_losses, axis=0)
    test_loss_mean = np.mean(test_losses, axis=0)
    test_loss_std = np.std(test_losses, axis=0)
    
    # Check if we have meaningful per-class data
    num_classes = interpolated_data[0].train_acc_per_class.shape[1]
    has_per_class_data = num_classes > 1
    
    if has_per_class_data:
        # Compute statistics for per-class metrics
        train_accs_per_class = np.array([data.train_acc_per_class for data in interpolated_data])  # (n_exp, n_steps, n_classes)
        test_accs_per_class = np.array([data.test_acc_per_class for data in interpolated_data])
        train_losses_per_class = np.array([data.train_loss_per_class for data in interpolated_data])
        test_losses_per_class = np.array([data.test_loss_per_class for data in interpolated_data])
        
        train_acc_per_class_mean = np.mean(train_accs_per_class, axis=0)  # (n_steps, n_classes)
        train_acc_per_class_std = np.std(train_accs_per_class, axis=0)
        test_acc_per_class_mean = np.mean(test_accs_per_class, axis=0)
        test_acc_per_class_std = np.std(test_accs_per_class, axis=0)
        
        train_loss_per_class_mean = np.mean(train_losses_per_class, axis=0)
        train_loss_per_class_std = np.std(train_losses_per_class, axis=0)
        test_loss_per_class_mean = np.mean(test_losses_per_class, axis=0)
        test_loss_per_class_std = np.std(test_losses_per_class, axis=0)
        
        # Create plot with dynamic layout: (num_classes+1) rows x 2 columns 
        # (Row 0: total metrics, Rows 1+: per-class metrics)
        fig, axes = plt.subplots(num_classes + 1, 2, figsize=(15, 5 * (num_classes + 1)))
        plot_title = f'Training Progress with Per-Class Breakdown (Averaged over {len(training_data)} seeds)'
    else:
        # Only total metrics available - use simple 1x2 layout
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes = axes.reshape(1, 2)  # Ensure consistent indexing
        plot_title = f'Training Progress (Averaged over {len(training_data)} seeds)'
    
    alpha_individual = 0.2
    
    # Row 0: Total metrics (always present)
    ax_loss, ax_acc = axes[0, 0], axes[0, 1]
    
    # Plot individual curves (light, thin lines) for total metrics
    for data in interpolated_data:
        ax_loss.plot(data.steps, data.train_loss, alpha=alpha_individual, color='blue', linewidth=0.5)
        ax_loss.plot(data.steps, data.test_loss, alpha=alpha_individual, color='red', linewidth=0.5)
        ax_acc.plot(data.steps, data.train_acc, alpha=alpha_individual, color='blue', linewidth=0.5)
        ax_acc.plot(data.steps, data.test_acc, alpha=alpha_individual, color='red', linewidth=0.5)
    
    # Plot averaged curves (bold) for total metrics
    ax_loss.plot(common_steps, train_loss_mean, 'b-', label='Training Loss', alpha=0.8, linewidth=1.5)
    ax_loss.fill_between(common_steps, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, 
                        alpha=0.3, color='blue')
    ax_loss.plot(common_steps, test_loss_mean, 'r--', label='Test Loss', alpha=0.8, linewidth=1.5)
    ax_loss.fill_between(common_steps, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, 
                        alpha=0.3, color='red')
    
    ax_acc.plot(common_steps, train_acc_mean, 'b-', label='Training Accuracy', alpha=0.8, linewidth=1.5)
    ax_acc.fill_between(common_steps, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, 
                       alpha=0.3, color='blue')
    ax_acc.plot(common_steps, test_acc_mean, 'r--', label='Test Accuracy', alpha=0.8, linewidth=1.5)
    ax_acc.fill_between(common_steps, test_acc_mean - test_acc_std, test_acc_mean + test_acc_std, 
                       alpha=0.3, color='red')
    
    # Format total metrics plots
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Total Loss Curves')
    ax_loss.set_yscale('log')
    
    # Add horizontal dashed lines for total loss
    all_losses = list(train_loss_mean) + list(test_loss_mean)
    loss_min = max(min(all_losses), 1e-6)
    loss_max = max(all_losses) * 1.1
    # Set bottom limit a bit lower than minimum to make horizontal lines visible
    loss_bottom = loss_min * 0.5  # 50% lower than minimum
    ax_loss.set_ylim(bottom=loss_bottom, top=loss_max)
    
    loss_horizontal_lines = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for loss_val in loss_horizontal_lines:
        if loss_min <= loss_val <= loss_max:
            ax_loss.axhline(y=loss_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
    
    ax_loss.legend()
    
    ax_acc.set_xlabel('Step')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Total Accuracy Curves')
    ax_acc.set_ylim(-0.01, 1.01)  # Set y-axis range from -0.01 to 1.01
    
    # Add horizontal dashed lines for total accuracy
    accuracy_horizontal_lines = [0.2, 0.4, 0.6, 0.8, 1.0]
    for acc_val in accuracy_horizontal_lines:
        ax_acc.axhline(y=acc_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
    
    ax_acc.legend()
    
    # Add per-class metrics only if available
    if has_per_class_data:
        # Rows 1+: Per-class metrics
        for class_idx in range(num_classes):
            row_idx = class_idx + 1
            ax_loss_class = axes[row_idx, 0]
            ax_acc_class = axes[row_idx, 1]
            
            # Plot individual curves for this class
            for data in interpolated_data:
                ax_loss_class.plot(data.steps, data.train_loss_per_class[:, class_idx], 
                                  alpha=alpha_individual, color='blue', linewidth=0.5)
                ax_loss_class.plot(data.steps, data.test_loss_per_class[:, class_idx], 
                                  alpha=alpha_individual, color='red', linewidth=0.5)
                ax_acc_class.plot(data.steps, data.train_acc_per_class[:, class_idx], 
                                 alpha=alpha_individual, color='blue', linewidth=0.5)
                ax_acc_class.plot(data.steps, data.test_acc_per_class[:, class_idx], 
                                 alpha=alpha_individual, color='red', linewidth=0.5)
            
            # Plot averaged curves for this class
            ax_loss_class.plot(common_steps, train_loss_per_class_mean[:, class_idx], 
                              'b-', label=f'Training Loss Class {class_idx}', alpha=0.8, linewidth=1.5)
            ax_loss_class.fill_between(common_steps, 
                                      train_loss_per_class_mean[:, class_idx] - train_loss_per_class_std[:, class_idx],
                                      train_loss_per_class_mean[:, class_idx] + train_loss_per_class_std[:, class_idx],
                                      alpha=0.3, color='blue')
            
            ax_loss_class.plot(common_steps, test_loss_per_class_mean[:, class_idx], 
                              'r--', label=f'Test Loss Class {class_idx}', alpha=0.8, linewidth=1.5)
            ax_loss_class.fill_between(common_steps, 
                                      test_loss_per_class_mean[:, class_idx] - test_loss_per_class_std[:, class_idx],
                                      test_loss_per_class_mean[:, class_idx] + test_loss_per_class_std[:, class_idx],
                                      alpha=0.3, color='red')
            
            ax_acc_class.plot(common_steps, train_acc_per_class_mean[:, class_idx], 
                             'b-', label=f'Training Acc Class {class_idx}', alpha=0.8, linewidth=1.5)
            ax_acc_class.fill_between(common_steps, 
                                     train_acc_per_class_mean[:, class_idx] - train_acc_per_class_std[:, class_idx],
                                     train_acc_per_class_mean[:, class_idx] + train_acc_per_class_std[:, class_idx],
                                     alpha=0.3, color='blue')
            
            ax_acc_class.plot(common_steps, test_acc_per_class_mean[:, class_idx], 
                             'r--', label=f'Test Acc Class {class_idx}', alpha=0.8, linewidth=1.5)
            ax_acc_class.fill_between(common_steps, 
                                     test_acc_per_class_mean[:, class_idx] - test_acc_per_class_std[:, class_idx],
                                     test_acc_per_class_mean[:, class_idx] + test_acc_per_class_std[:, class_idx],
                                     alpha=0.3, color='red')
            
            # Format per-class plots
            ax_loss_class.set_xlabel('Step')
            ax_loss_class.set_ylabel('Loss')
            ax_loss_class.set_title(f'Class {class_idx} Loss Curves')
            ax_loss_class.set_yscale('log')
            ax_loss_class.legend()
            
            # Add grid for readability
            ax_loss_class.grid(True, alpha=0.3)
            
            ax_acc_class.set_xlabel('Step')
            ax_acc_class.set_ylabel('Accuracy')
            ax_acc_class.set_title(f'Class {class_idx} Accuracy Curves')
            ax_acc_class.set_ylim(-0.01, 1.01)  # Set y-axis range from -0.01 to 1.01
            ax_acc_class.legend()
            
            # Add horizontal lines for accuracy
            for acc_val in accuracy_horizontal_lines:
                ax_acc_class.axhline(y=acc_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
            
            ax_acc_class.grid(True, alpha=0.3)
    
    # Main title
    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f'training_curves_config_{config}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved averaged training curves to {output_path}")


def plot_averaged_nc_metrics(nc_data: List[NCMetricsData], output_dir: Path, config: str):
    """Create averaged neural collapse metrics plot matching the original paper style from neural_collapse.py."""
    if not nc_data:
        logging.warning("No NC metrics data to plot")
        return
    
    # Find common step range
    min_steps = max(data.steps[0] for data in nc_data)
    max_steps = min(data.steps[-1] for data in nc_data)
    
    # Create common step array
    all_steps = []
    for data in nc_data:
        mask = (data.steps >= min_steps) & (data.steps <= max_steps)
        all_steps.extend(data.steps[mask])
    
    common_steps = np.unique(np.array(all_steps))
    common_steps = common_steps[(common_steps >= min_steps) & (common_steps <= max_steps)]
    
    # Interpolate each experiment to common steps
    nc1_values = []
    nc2_cv_means_values = []
    nc2_cv_cls_values = []
    nc2_std_means_values = []
    nc2_std_cls_values = []
    nc2_mean_means_values = []
    nc2_mean_cls_values = []
    nc2_target_values = []
    nc3_values = []
    nc4_values = []
    
    for data in nc_data:
        nc1_interp = np.interp(common_steps, data.steps, data.nc1)
        nc2_cv_means_interp = np.interp(common_steps, data.steps, data.nc2_cv_means)
        nc2_cv_cls_interp = np.interp(common_steps, data.steps, data.nc2_cv_cls)
        nc2_std_means_interp = np.interp(common_steps, data.steps, data.nc2_std_means)
        nc2_std_cls_interp = np.interp(common_steps, data.steps, data.nc2_std_cls)
        nc2_mean_means_interp = np.interp(common_steps, data.steps, data.nc2_mean_means)
        nc2_mean_cls_interp = np.interp(common_steps, data.steps, data.nc2_mean_cls)
        nc2_target_interp = np.interp(common_steps, data.steps, data.nc2_target)
        nc3_interp = np.interp(common_steps, data.steps, data.nc3)
        nc4_interp = np.interp(common_steps, data.steps, data.nc4)
        
        nc1_values.append(nc1_interp)
        nc2_cv_means_values.append(nc2_cv_means_interp)
        nc2_cv_cls_values.append(nc2_cv_cls_interp)
        nc2_std_means_values.append(nc2_std_means_interp)
        nc2_std_cls_values.append(nc2_std_cls_interp)
        nc2_mean_means_values.append(nc2_mean_means_interp)
        nc2_mean_cls_values.append(nc2_mean_cls_interp)
        nc2_target_values.append(nc2_target_interp)
        nc3_values.append(nc3_interp)
        nc4_values.append(nc4_interp)
    
    nc1_values = np.array(nc1_values)
    nc2_cv_means_values = np.array(nc2_cv_means_values)
    nc2_cv_cls_values = np.array(nc2_cv_cls_values)
    nc2_std_means_values = np.array(nc2_std_means_values)
    nc2_std_cls_values = np.array(nc2_std_cls_values)
    nc2_mean_means_values = np.array(nc2_mean_means_values)
    nc2_mean_cls_values = np.array(nc2_mean_cls_values)
    nc2_target_values = np.array(nc2_target_values)
    nc3_values = np.array(nc3_values)
    nc4_values = np.array(nc4_values)
    
    # Compute statistics
    nc1_mean = np.mean(nc1_values, axis=0)
    nc1_std = np.std(nc1_values, axis=0)
    nc2_cv_means_mean = np.mean(nc2_cv_means_values, axis=0)
    nc2_cv_means_std = np.std(nc2_cv_means_values, axis=0)
    nc2_cv_cls_mean = np.mean(nc2_cv_cls_values, axis=0)
    nc2_cv_cls_std = np.std(nc2_cv_cls_values, axis=0)
    nc2_std_means_mean = np.mean(nc2_std_means_values, axis=0)
    nc2_std_means_std = np.std(nc2_std_means_values, axis=0)
    nc2_std_cls_mean = np.mean(nc2_std_cls_values, axis=0)
    nc2_std_cls_std = np.std(nc2_std_cls_values, axis=0)
    nc2_mean_means_mean = np.mean(nc2_mean_means_values, axis=0)
    nc2_mean_means_std = np.std(nc2_mean_means_values, axis=0)
    nc2_mean_cls_mean = np.mean(nc2_mean_cls_values, axis=0)
    nc2_mean_cls_std = np.std(nc2_mean_cls_values, axis=0)
    nc2_target_mean = np.mean(nc2_target_values, axis=0)
    nc3_mean = np.mean(nc3_values, axis=0)
    nc3_std = np.std(nc3_values, axis=0)
    nc4_mean = np.mean(nc4_values, axis=0)
    nc4_std = np.std(nc4_values, axis=0)
    
    # Check if we have meaningful NC4 data
    has_nc4 = np.any(nc4_mean > 0.001)  # Only show if there's meaningful data
    
    # Create figure with 2x3 subplots (matching original paper style)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Figure 2: NC2 - Equinorm (CV) - BOTH means and classifiers
    ax = axes[0, 0]
    
    # Plot individual curves (light)
    alpha_individual = 0.2
    for i in range(len(nc_data)):
        ax.plot(common_steps, nc2_cv_means_values[i], alpha=alpha_individual, color='#2E86AB', linewidth=0.5)
        ax.plot(common_steps, nc2_cv_cls_values[i], alpha=alpha_individual, color='#F18F01', linewidth=0.5)
    
    # Plot averaged curves (bold)
    ax.plot(common_steps, nc2_cv_means_mean, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Class-means')
    ax.fill_between(common_steps, nc2_cv_means_mean - nc2_cv_means_std, nc2_cv_means_mean + nc2_cv_means_std, 
                     alpha=0.3, color='#2E86AB')
    
    ax.plot(common_steps, nc2_cv_cls_mean, 's-', linewidth=2, markersize=6, color='#F18F01', label='Classifiers')
    ax.fill_between(common_steps, nc2_cv_cls_mean - nc2_cv_cls_std, nc2_cv_cls_mean + nc2_cv_cls_std, 
                     alpha=0.3, color='#F18F01')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: NC2 - Equinorm\nStd(||μ_c - μ_G||) / Avg(||μ_c - μ_G||)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
    # Set y-axis limits: fixed range -0.02 to 0.2
    ax.set_ylim(bottom=-0.02, top=0.4)
    ax.legend(fontsize=10)
    
    # Figure 3: NC2 - Equiangularity (Std) - BOTH means and classifiers
    ax = axes[0, 1]
    
    # Plot individual curves (light)
    for i in range(len(nc_data)):
        ax.plot(common_steps, nc2_std_means_values[i], alpha=alpha_individual, color='#2E86AB', linewidth=0.5)
        ax.plot(common_steps, nc2_std_cls_values[i], alpha=alpha_individual, color='#F18F01', linewidth=0.5)
    
    # Plot averaged curves (bold)
    ax.plot(common_steps, nc2_std_means_mean, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Class-means')
    ax.fill_between(common_steps, nc2_std_means_mean - nc2_std_means_std, nc2_std_means_mean + nc2_std_means_std, 
                     alpha=0.3, color='#2E86AB')
    
    ax.plot(common_steps, nc2_std_cls_mean, 's-', linewidth=2, markersize=6, color='#F18F01', label='Classifiers')
    ax.fill_between(common_steps, nc2_std_cls_mean - nc2_std_cls_std, nc2_std_cls_mean + nc2_std_cls_std, 
                     alpha=0.3, color='#F18F01')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Std of Cosines', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: NC2 - Equiangularity (Std)\nStd(cos(c,c\'))', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
    # Set y-axis limits: fixed range -0.02 to 0.6
    ax.set_ylim(bottom=-0.02, top=0.6)
    ax.legend(fontsize=10)
    
    # Figure 4: NC2 - Equiangularity (Mean) - BOTH means and classifiers
    ax = axes[0, 2]
    # Check if data is all zeros (missing)
    if np.all(nc2_mean_means_mean == 0) and np.all(nc2_mean_cls_mean == 0):
        ax.axis('off')
        ax.text(0.5, 0.5, 'NC2 Mean Equiangularity\n(no data available)', 
                ha='center', va='center', fontsize=11, color='gray')
    else:
        # Plot individual curves (light)
        for i in range(len(nc_data)):
            ax.plot(common_steps, nc2_mean_means_values[i], alpha=alpha_individual, color='#2E86AB', linewidth=0.5)
            ax.plot(common_steps, nc2_mean_cls_values[i], alpha=alpha_individual, color='#F18F01', linewidth=0.5)
        # Plot averaged curves (bold)
        ax.plot(common_steps, nc2_mean_means_mean, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Class-means')
        ax.fill_between(common_steps, nc2_mean_means_mean - nc2_mean_means_std, nc2_mean_means_mean + nc2_mean_means_std, 
                        alpha=0.3, color='#2E86AB')
        ax.plot(common_steps, nc2_mean_cls_mean, 's-', linewidth=2, markersize=6, color='#F18F01', label='Classifiers')
        ax.fill_between(common_steps, nc2_mean_cls_mean - nc2_mean_cls_std, nc2_mean_cls_mean + nc2_mean_cls_std, 
                        alpha=0.3, color='#F18F01')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg |cos + 1/(C-1)|', fontsize=12, fontweight='bold')
        ax.set_title(f'Figure 4: NC2 - Equiangularity (Mean)\nAvg|cos(c,c\') + 1/(C-1)|', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
                label=f'Target: 0')
        # Set y-axis limits: fixed range -0.02 to 0.6
        ax.set_ylim(bottom=-0.02, top=0.6)
        ax.legend(fontsize=10)
    
    # Figure 5: NC3 - Self-Duality
    ax = axes[1, 0]
    
    # Plot individual curves (light)
    for i in range(len(nc_data)):
        ax.plot(common_steps, nc3_values[i], alpha=alpha_individual, color='#D62246', linewidth=0.5)
    
    # Plot averaged curve (bold)
    ax.plot(common_steps, nc3_mean, 'o-', linewidth=2, markersize=6, color='#D62246')
    ax.fill_between(common_steps, nc3_mean - nc3_std, nc3_mean + nc3_std, 
                     alpha=0.3, color='#D62246')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('NC3 Metric', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5: NC3 - Self-Duality\n||Ŵ^T - M̂||_F', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
    # Set y-axis limits: fixed range -0.02 to 1.5
    ax.set_ylim(bottom=-0.02, top=1.5)
    ax.legend()
    
    # Figure 6: NC1 - Variability Collapse
    ax = axes[1, 1]
    
    # Plot individual curves (light)
    for i in range(len(nc_data)):
        ax.plot(common_steps, nc1_values[i], alpha=alpha_individual, color='#2E86AB', linewidth=0.5)
    
    # Plot averaged curve (bold)
    ax.plot(common_steps, nc1_mean, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Within-Class Variation')
    ax.fill_between(common_steps, nc1_mean - nc1_std, nc1_mean + nc1_std, 
                     alpha=0.3, color='#2E86AB')
    
    # Always use log scale for NC1 to see collapse clearly
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tr{W @ B†} / C (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 6: NC1 - Variability Collapse\nTr{W @ B†} / C', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    # Set y-axis limits: fixed log range 20 to 45
    ax.set_ylim(bottom=0.01, top=50)
    ax.legend()
    
    # Figure 7: NC4 - Nearest Class-Center Mismatch (if available)
    ax = axes[1, 2]
    if has_nc4:
        # Plot individual curves (light)
        for i in range(len(nc_data)):
            ax.plot(common_steps, nc4_values[i], alpha=alpha_individual, color='#A23B72', linewidth=0.5)
        
        # Plot averaged curve (bold)
        ax.plot(common_steps, nc4_mean, 'o-', linewidth=2, markersize=6, color='#A23B72')
        ax.fill_between(common_steps, nc4_mean - nc4_std, nc4_mean + nc4_std, 
                         alpha=0.3, color='#A23B72')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion of Disagreements', fontsize=12, fontweight='bold')
        ax.set_title('Figure 7: Classifier → NCC\nProportion where Classifier ≠ arg min||h-μ_c||', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
        # Set y-axis limits: fixed range -0.02 to 1
        ax.set_ylim(bottom=-0.02, top=0.4)
        ax.legend()
    else:
        # Hide subplot if NC4 not available
        ax.axis('off')
        ax.text(0.5, 0.5, 'NC4: Nearest Class-Center\n(no meaningful test data)', 
                ha='center', va='center', fontsize=11, color='gray')
    
    plt.suptitle(f'Neural Collapse Metrics Evolution (Averaged over {len(nc_data)} seeds)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot as paper-style filename
    output_path_paper = output_dir / f'nc_metrics_paper_style_config_{config}.png'
    plt.savefig(output_path_paper, dpi=300, bbox_inches='tight')
    logging.info(f"Saved averaged NC metrics to {output_path_paper}")

    # Also save as nc_metrics_evolution.png for spiral configs
    if 'spiral' in config.lower():
        output_path_evolution = output_dir / 'nc_metrics_evolution.png'
        plt.savefig(output_path_evolution, dpi=300, bbox_inches='tight')
        logging.info(f"Saved spiral NC metrics to {output_path_evolution}")
    plt.close()


def save_averaged_data(training_data: List[TrainingCurveData], nc_data: List[NCMetricsData], 
                      output_dir: Path, config: str):
    """Save averaged data as CSV files for further analysis."""
    
    # Save averaged training curves
    if training_data:
        common_steps, interpolated_data = interpolate_to_common_steps(training_data)
        
        train_accs = np.array([data.train_acc for data in interpolated_data])
        test_accs = np.array([data.test_acc for data in interpolated_data])
        train_losses = np.array([data.train_loss for data in interpolated_data])
        test_losses = np.array([data.test_loss for data in interpolated_data])
        
        # Per-class data
        train_accs_per_class = np.array([data.train_acc_per_class for data in interpolated_data])
        test_accs_per_class = np.array([data.test_acc_per_class for data in interpolated_data])
        train_losses_per_class = np.array([data.train_loss_per_class for data in interpolated_data])
        test_losses_per_class = np.array([data.test_loss_per_class for data in interpolated_data])
        
        # Create DataFrame with total and per-class metrics
        df_columns = {
            'Step': common_steps,
            'Train_Acc_Mean': np.mean(train_accs, axis=0),
            'Train_Acc_Std': np.std(train_accs, axis=0),
            'Test_Acc_Mean': np.mean(test_accs, axis=0),
            'Test_Acc_Std': np.std(test_accs, axis=0),
            'Train_Loss_Mean': np.mean(train_losses, axis=0),
            'Train_Loss_Std': np.std(train_losses, axis=0),
            'Test_Loss_Mean': np.mean(test_losses, axis=0),
            'Test_Loss_Std': np.std(test_losses, axis=0)
        }
        
        # Add per-class columns
        num_classes = train_accs_per_class.shape[2]
        for c in range(num_classes):
            df_columns[f'Train_Acc_Class_{c}_Mean'] = np.mean(train_accs_per_class[:, :, c], axis=0)
            df_columns[f'Train_Acc_Class_{c}_Std'] = np.std(train_accs_per_class[:, :, c], axis=0)
            df_columns[f'Test_Acc_Class_{c}_Mean'] = np.mean(test_accs_per_class[:, :, c], axis=0)
            df_columns[f'Test_Acc_Class_{c}_Std'] = np.std(test_accs_per_class[:, :, c], axis=0)
            df_columns[f'Train_Loss_Class_{c}_Mean'] = np.mean(train_losses_per_class[:, :, c], axis=0)
            df_columns[f'Train_Loss_Class_{c}_Std'] = np.std(train_losses_per_class[:, :, c], axis=0)
            df_columns[f'Test_Loss_Class_{c}_Mean'] = np.mean(test_losses_per_class[:, :, c], axis=0)
            df_columns[f'Test_Loss_Class_{c}_Std'] = np.std(test_losses_per_class[:, :, c], axis=0)
        
        df_training = pd.DataFrame(df_columns)
        
        training_csv_path = output_dir / f'averaged_training_curves_config_{config}.csv'
        df_training.to_csv(training_csv_path, index=False)
        logging.info(f"Saved averaged training data to {training_csv_path}")
    
    # Save averaged NC metrics
    if nc_data:
        # Find common steps for NC data
        min_steps = max(data.steps[0] for data in nc_data)
        max_steps = min(data.steps[-1] for data in nc_data)
        
        all_steps = []
        for data in nc_data:
            mask = (data.steps >= min_steps) & (data.steps <= max_steps)
            all_steps.extend(data.steps[mask])
        
        common_steps = np.unique(np.array(all_steps))
        common_steps = common_steps[(common_steps >= min_steps) & (common_steps <= max_steps)]
        
        nc1_values = []
        nc2_cv_means_values = []
        nc2_cv_cls_values = []
        nc2_std_means_values = []
        nc2_std_cls_values = []
        nc2_mean_means_values = []
        nc2_mean_cls_values = []
        nc2_target_values = []
        nc3_values = []
        nc4_values = []
        
        for data in nc_data:
            nc1_interp = np.interp(common_steps, data.steps, data.nc1)
            nc2_cv_means_interp = np.interp(common_steps, data.steps, data.nc2_cv_means)
            nc2_cv_cls_interp = np.interp(common_steps, data.steps, data.nc2_cv_cls)
            nc2_std_means_interp = np.interp(common_steps, data.steps, data.nc2_std_means)
            nc2_std_cls_interp = np.interp(common_steps, data.steps, data.nc2_std_cls)
            nc2_mean_means_interp = np.interp(common_steps, data.steps, data.nc2_mean_means)
            nc2_mean_cls_interp = np.interp(common_steps, data.steps, data.nc2_mean_cls)
            nc2_target_interp = np.interp(common_steps, data.steps, data.nc2_target)
            nc3_interp = np.interp(common_steps, data.steps, data.nc3)
            nc4_interp = np.interp(common_steps, data.steps, data.nc4)
            
            nc1_values.append(nc1_interp)
            nc2_cv_means_values.append(nc2_cv_means_interp)
            nc2_cv_cls_values.append(nc2_cv_cls_interp)
            nc2_std_means_values.append(nc2_std_means_interp)
            nc2_std_cls_values.append(nc2_std_cls_interp)
            nc2_mean_means_values.append(nc2_mean_means_interp)
            nc2_mean_cls_values.append(nc2_mean_cls_interp)
            nc2_target_values.append(nc2_target_interp)
            nc3_values.append(nc3_interp)
            nc4_values.append(nc4_interp)
        
        nc1_values = np.array(nc1_values)
        nc2_cv_means_values = np.array(nc2_cv_means_values)
        nc2_cv_cls_values = np.array(nc2_cv_cls_values)
        nc2_std_means_values = np.array(nc2_std_means_values)
        nc2_std_cls_values = np.array(nc2_std_cls_values)
        nc2_mean_means_values = np.array(nc2_mean_means_values)
        nc2_mean_cls_values = np.array(nc2_mean_cls_values)
        nc2_target_values = np.array(nc2_target_values)
        nc3_values = np.array(nc3_values)
        nc4_values = np.array(nc4_values)
        
        df_nc = pd.DataFrame({
            'Step': common_steps,
            'NC1_Mean': np.mean(nc1_values, axis=0),
            'NC1_Std': np.std(nc1_values, axis=0),
            'NC2_CV_Means_Mean': np.mean(nc2_cv_means_values, axis=0),
            'NC2_CV_Means_Std': np.std(nc2_cv_means_values, axis=0),
            'NC2_CV_Cls_Mean': np.mean(nc2_cv_cls_values, axis=0),
            'NC2_CV_Cls_Std': np.std(nc2_cv_cls_values, axis=0),
            'NC2_Std_Means_Mean': np.mean(nc2_std_means_values, axis=0),
            'NC2_Std_Means_Std': np.std(nc2_std_means_values, axis=0),
            'NC2_Std_Cls_Mean': np.mean(nc2_std_cls_values, axis=0),
            'NC2_Std_Cls_Std': np.std(nc2_std_cls_values, axis=0),
            'NC2_Mean_Means_Mean': np.mean(nc2_mean_means_values, axis=0),
            'NC2_Mean_Means_Std': np.std(nc2_mean_means_values, axis=0),
            'NC2_Mean_Cls_Mean': np.mean(nc2_mean_cls_values, axis=0),
            'NC2_Mean_Cls_Std': np.std(nc2_mean_cls_values, axis=0),
            'NC2_Target': np.mean(nc2_target_values, axis=0),
            'NC3_Mean': np.mean(nc3_values, axis=0),
            'NC3_Std': np.std(nc3_values, axis=0),
            'NC4_Mean': np.mean(nc4_values, axis=0),
            'NC4_Std': np.std(nc4_values, axis=0)
        })
        
        nc_csv_path = output_dir / f'averaged_nc_metrics_config_{config}.csv'
        df_nc.to_csv(nc_csv_path, index=False)
        logging.info(f"Saved averaged NC data to {nc_csv_path}")


def find_configs_in_mode(base_path: Path, mode: str) -> List[str]:
    """Find all config directories in the specified mode directory."""
    mode_path = base_path / mode
    if not mode_path.exists():
        return []
    
    configs = []
    for path in mode_path.iterdir():
        if path.is_dir() and path.name.startswith('nc_config_'):
            # Extract config name (remove nc_config_ prefix)
            config_name = path.name.replace('nc_config_', '')
            configs.append(config_name)
    
    configs.sort()  # Sort for consistent ordering
    return configs


def process_single_config(base_path: Path, mode: str, config: str, cluster: str, 
                         save_csv: bool, skip_nc: bool):
    """Process a single config and generate averaged figures."""
    # Construct experiment path
    config_name = f"{config}"
    experiment_path = base_path / mode / config_name
    
    if not experiment_path.exists():
        logging.error(f"Experiment path does not exist: {experiment_path}")
        return
    
    # Create output directory inside training_figures
    output_dir = base_path / mode / "training_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 80)
    logging.info(f"PROCESSING CONFIG: {config}")
    logging.info("=" * 80)
    logging.info(f"Mode: {mode}")
    logging.info(f"Config: {config}")
    logging.info(f"Experiment path: {experiment_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 80)
    
    # Find experiment directories
    experiment_dirs = find_experiment_directories(experiment_path)
    
    if not experiment_dirs:
        logging.error(f"No experiment directories found in {experiment_path}")
        return
    
    # Load training curves
    logging.info("Loading training curves...")
    training_data = []
    failed_loads = []
    
    for exp_dir in experiment_dirs:
        try:
            data = load_training_curves(exp_dir)
            training_data.append(data)
            logging.info(f"  ✓ Loaded {exp_dir.name}")
        except Exception as e:
            logging.error(f"  ✗ Failed to load {exp_dir.name}: {str(e)}")
            failed_loads.append(exp_dir.name)
    
    if failed_loads:
        logging.warning(f"Failed to load training curves from: {failed_loads}")
    
    # Load neural collapse metrics (if not skipped)
    nc_data = []
    if not skip_nc:
        logging.info("Loading neural collapse metrics...")
        for exp_dir in experiment_dirs:
            try:
                data = load_nc_metrics(exp_dir)
                if data is not None:
                    nc_data.append(data)
                    logging.info(f"  ✓ Loaded NC metrics from {exp_dir.name}")
                else:
                    logging.info(f"  - No NC metrics in {exp_dir.name}")
            except Exception as e:
                logging.error(f"  ✗ Failed to load NC metrics from {exp_dir.name}: {str(e)}")
    
    # Generate plots with config-specific names
    if training_data:
        logging.info(f"Generating averaged training curves from {len(training_data)} experiments...")
        plot_averaged_training_curves(training_data, output_dir, config)
    else:
        logging.error("No training data loaded!")
    
    if nc_data and not skip_nc:
        logging.info(f"Generating averaged NC metrics from {len(nc_data)} experiments...")
        plot_averaged_nc_metrics(nc_data, output_dir, config)
    elif not skip_nc:
        logging.warning("No NC metrics data loaded!")
    
    # Save CSV data if requested
    if save_csv:
        logging.info("Saving averaged data as CSV files...")
        save_averaged_data(training_data, nc_data, output_dir, config)
    
    logging.info(f"✓ Completed processing config: {config}")
    logging.info(f"Output saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate averaged training figures from experiment data')
    
    parser.add_argument(
        '--cluster',
        type=str,
        required=True,
        choices=['brigit', 'cuenca'],
        help='Cluster name (brigit or cuenca) - determines base data path'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        help='Experiment mode: nc_with_bumps_tpt, nc_without_bumps_tpt, or nc_without_dynamics'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Configuration name (e.g., baseline, big_nn, super_big_nn). If not provided, process all configs found.'
    )
    
    parser.add_argument(
        '--save_csv',
        action='store_true',
        help='Save averaged data as CSV files'
    )
    
    parser.add_argument(
        '--skip_nc',
        action='store_true',
        help='Skip neural collapse metrics (only generate training curves)'
    )
    
    args = parser.parse_args()
    
    # Setup base directory based on cluster (following run_nc_experiment.py pattern)
    if args.cluster == 'brigit':
        base_dir = Path('/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd')
    elif args.cluster == 'cuenca':
        base_dir = Path('/data/samuel_lozano/dynamical-sgd')
    else:
        base_dir = Path('/data/samuel_lozano/dynamical-sgd')
    
    # Setup
    setup_logging()
    
    if not base_dir.exists():
        logging.error(f"Base data directory does not exist: {base_dir}")
        return
    
    logging.info("=" * 80)
    logging.info("GENERATE AVERAGED TRAINING FIGURES")
    logging.info("=" * 80)
    logging.info(f"Cluster: {args.cluster}")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Base directory: {base_dir}")
    if args.config:
        logging.info(f"Config: {args.config}")
    else:
        logging.info("Config: All configs found")
    logging.info("=" * 80)
    
    # Determine which configs to process
    if args.config:
        configs_to_process = [args.config]
    else:
        # Find all configs in the mode directory
        configs_to_process = find_configs_in_mode(base_dir, args.mode)
        if not configs_to_process:
            logging.error(f"No configs found in {base_dir / args.mode}")
            return
        logging.info(f"Found configs: {configs_to_process}")
    
    # Process each config
    for config in configs_to_process:
        try:
            process_single_config(
                base_path=base_dir,
                mode=args.mode,
                config=config,
                cluster=args.cluster,
                save_csv=args.save_csv,
                skip_nc=args.skip_nc
            )
        except Exception as e:
            logging.error(f"Failed to process config {config}: {str(e)}")
            continue
    
    logging.info("=" * 80)
    logging.info("COMPLETED SUCCESSFULLY")
    logging.info(f"Processed {len(configs_to_process)} config(s)")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()