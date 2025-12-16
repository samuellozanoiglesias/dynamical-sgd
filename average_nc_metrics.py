#!/usr/bin/env python3
"""
Average Neural Collapse Metrics Across Seeds

This script averages metrics across multiple experiment runs (seeds) for a given
neural collapse configuration. It processes:
- Training curves (loss, accuracy)
- Neural collapse metrics (NC1, NC2, NC3)
- Angles between SFT (Simplex ETF) vertices and class means
- Angles between classifiers and class means

Usage:
    nohup python average_nc_metrics.py --config nc_config_narrow_bump --experiment_name nc_with_dynamics > average_nc_metrics.log 2>&1 &
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import yaml
from tqdm import tqdm
import pandas as pd

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from analysis.neural_collapse import NeuralCollapseAnalyzer, NeuralCollapseSnapshot


@dataclass
class AveragedMetrics:
    """Container for averaged metrics across seeds."""
    
    # Training curves
    train_steps: np.ndarray
    train_loss_mean: np.ndarray
    train_loss_std: np.ndarray
    train_acc_mean: np.ndarray
    train_acc_std: np.ndarray
    test_loss_mean: np.ndarray
    test_loss_std: np.ndarray
    test_acc_mean: np.ndarray
    test_acc_std: np.ndarray
    
    # Neural Collapse metrics over time
    nc_steps: np.ndarray
    nc1_mean: np.ndarray
    nc1_std: np.ndarray
    nc2_mean: np.ndarray
    nc2_std: np.ndarray
    nc3_mean: np.ndarray
    nc3_std: np.ndarray
    
    # Angles (averaged at each snapshot)
    angle_steps: np.ndarray
    mean_angles_mean: np.ndarray  # Angles between class means
    mean_angles_std: np.ndarray
    classifier_angles_mean: np.ndarray  # Angles between classifiers
    classifier_angles_std: np.ndarray
    etf_angles_mean: np.ndarray  # Angles in theoretical Simplex ETF
    etf_angles_std: np.ndarray
    
    # Summary statistics
    num_seeds: int
    seeds_used: List[int]
    config_name: str
    experiment_name: str


def find_experiment_directories(
    base_dir: Path,
    experiment_name: str,
    config_name: str,
    timestamp_filter: Optional[str] = None
) -> List[Path]:
    """
    Find all experiment directories for a given config.
    
    Args:
        base_dir: Base directory containing experiments
        experiment_name: Name of experiment (e.g., 'nc_with_dynamics')
        config_name: Configuration name (e.g., 'nc_config_narrow_bump')
        timestamp_filter: Optional timestamp to filter experiments (e.g., '2025_12_14')
    
    Returns:
        List of experiment directory paths
    """
    search_path = base_dir / experiment_name / config_name
    
    if not search_path.exists():
        print(f"Warning: Path does not exist: {search_path}")
        return []
    
    # Find all experiment_* directories
    experiment_dirs = sorted(search_path.glob("experiment_*"))
    
    if timestamp_filter:
        experiment_dirs = [d for d in experiment_dirs if timestamp_filter in d.name]
    
    return experiment_dirs


def load_training_curves(experiment_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training curves from results.csv.
    
    Returns:
        steps, train_acc, test_acc, train_loss, test_loss
    """
    csv_path = experiment_dir / "results.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found in {experiment_dir}")
    
    df = pd.read_csv(csv_path)
    
    steps = df['Step'].values
    train_acc = df['Train_Acc'].values
    test_acc = df['Test_Acc'].values
    train_loss = df['Train_Loss'].values
    test_loss = df['Test_Loss'].values
    
    return steps, train_acc, test_acc, train_loss, test_loss


def load_nc_snapshots(experiment_dir: Path) -> List[NeuralCollapseSnapshot]:
    """Load Neural Collapse snapshots from pickle file."""
    pkl_path = experiment_dir / "nc_snapshots.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"nc_snapshots.pkl not found in {experiment_dir}")
    
    with open(pkl_path, 'rb') as f:
        snapshots = pickle.load(f)
    
    return snapshots


def compute_angles_from_snapshot(
    snapshot: NeuralCollapseSnapshot,
    analyzer: NeuralCollapseAnalyzer
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute angles between class means, classifiers, and ETF.
    
    Returns:
        mean_angles, classifier_angles, etf_angles
    """
    # Angles between class means
    mean_angles = analyzer.compute_pairwise_angles(snapshot.class_means)
    
    # Angles between classifiers
    classifier_angles = analyzer.compute_pairwise_angles(snapshot.classifiers)
    
    # Angles in theoretical Simplex ETF
    etf = analyzer.compute_simplex_etf(snapshot.num_classes)
    etf_angles = analyzer.compute_pairwise_angles(etf)
    
    return mean_angles, classifier_angles, etf_angles


def interpolate_to_common_steps(
    steps_list: List[np.ndarray],
    values_list: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Interpolate all metrics to a common set of steps.
    
    Args:
        steps_list: List of step arrays (one per seed)
        values_list: List of value arrays (one per seed)
    
    Returns:
        common_steps, interpolated_values_list
    """
    # Find common step range
    min_step = max(steps[0] for steps in steps_list)
    max_step = min(steps[-1] for steps in steps_list)
    
    # Create common step array (use finest resolution available)
    all_steps = np.concatenate(steps_list)
    unique_steps = np.unique(all_steps)
    common_steps = unique_steps[(unique_steps >= min_step) & (unique_steps <= max_step)]
    
    # Interpolate each seed's values to common steps
    interpolated_values = []
    for steps, values in zip(steps_list, values_list):
        interp_values = np.interp(common_steps, steps, values)
        interpolated_values.append(interp_values)
    
    return common_steps, interpolated_values


def average_experiment_metrics(
    experiment_dirs: List[Path],
    config_name: str,
    experiment_name: str
) -> AveragedMetrics:
    """
    Average metrics across all experiments (seeds).
    
    Args:
        experiment_dirs: List of experiment directory paths
        config_name: Configuration name
        experiment_name: Experiment name
    
    Returns:
        AveragedMetrics object with mean and std for all metrics
    """
    num_seeds = len(experiment_dirs)
    print(f"\nProcessing {num_seeds} experiments...")
    
    # Storage for all seeds
    all_train_steps = []
    all_train_acc = []
    all_test_acc = []
    all_train_loss = []
    all_test_loss = []
    
    all_nc_steps = []
    all_nc1 = []
    all_nc2 = []
    all_nc3 = []
    
    all_angle_steps = []
    all_mean_angles = []
    all_classifier_angles = []
    all_etf_angles = []
    
    seeds_used = []
    
    # Create analyzer for angle computations
    analyzer = None
    
    for exp_dir in tqdm(experiment_dirs, desc="Loading experiments"):
        try:
            # Extract seed from config if available
            config_path = exp_dir / "config.yaml"
            if config_path.exists():
                try:
                    # Try to load with safe_load first
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        seed = config.get('training', {}).get('random_seed', None)
                        if seed is not None:
                            seeds_used.append(seed)
                except yaml.YAMLError:
                    # If safe_load fails (e.g., due to Python tuples), use unsafe_load
                    # This is necessary for configs that include Python objects
                    with open(config_path, 'r') as f:
                        config = yaml.unsafe_load(f)
                        seed = config.get('training', {}).get('random_seed', None)
                        if seed is not None:
                            seeds_used.append(seed)
            
            # Load training curves
            steps, train_acc, test_acc, train_loss, test_loss = load_training_curves(exp_dir)
            all_train_steps.append(steps)
            all_train_acc.append(train_acc)
            all_test_acc.append(test_acc)
            all_train_loss.append(train_loss)
            all_test_loss.append(test_loss)
            
            # Load NC snapshots
            snapshots = load_nc_snapshots(exp_dir)
            
            if analyzer is None:
                # Initialize analyzer based on first snapshot
                analyzer = NeuralCollapseAnalyzer(
                    num_classes=snapshots[0].num_classes,
                    feature_dim=snapshots[0].feature_dim
                )
            
            # Extract NC metrics and angles
            nc_steps = []
            nc1_vals = []
            nc2_vals = []
            nc3_vals = []
            
            angle_steps = []
            mean_angles = []
            classifier_angles = []
            etf_angles = []
            
            for snapshot in snapshots:
                # Compute NC metrics
                metrics = analyzer.compute_nc_metrics(snapshot)
                nc_steps.append(snapshot.epoch)
                nc1_vals.append(metrics['nc1_within_class_variance'])
                nc2_vals.append(metrics['nc2_etf_alignment'])
                nc3_vals.append(metrics['nc3_self_duality'])
                
                # Compute angles
                m_angles, c_angles, e_angles = compute_angles_from_snapshot(snapshot, analyzer)
                angle_steps.append(snapshot.epoch)
                mean_angles.append(np.mean(m_angles))  # Average over all pairs
                classifier_angles.append(np.mean(c_angles))
                etf_angles.append(np.mean(e_angles))
            
            all_nc_steps.append(np.array(nc_steps))
            all_nc1.append(np.array(nc1_vals))
            all_nc2.append(np.array(nc2_vals))
            all_nc3.append(np.array(nc3_vals))
            
            all_angle_steps.append(np.array(angle_steps))
            all_mean_angles.append(np.array(mean_angles))
            all_classifier_angles.append(np.array(classifier_angles))
            all_etf_angles.append(np.array(etf_angles))
            
        except Exception as e:
            print(f"\nWarning: Failed to process {exp_dir.name}: {str(e)}")
            continue
    
    if len(all_train_steps) == 0:
        raise ValueError("No valid experiments found!")
    
    print(f"\nSuccessfully loaded {len(all_train_steps)} experiments")
    
    # Interpolate all metrics to common steps
    print("Interpolating training curves...")
    common_train_steps, interp_train_acc = interpolate_to_common_steps(all_train_steps, all_train_acc)
    _, interp_test_acc = interpolate_to_common_steps(all_train_steps, all_test_acc)
    _, interp_train_loss = interpolate_to_common_steps(all_train_steps, all_train_loss)
    _, interp_test_loss = interpolate_to_common_steps(all_train_steps, all_test_loss)
    
    print("Interpolating NC metrics...")
    common_nc_steps, interp_nc1 = interpolate_to_common_steps(all_nc_steps, all_nc1)
    _, interp_nc2 = interpolate_to_common_steps(all_nc_steps, all_nc2)
    _, interp_nc3 = interpolate_to_common_steps(all_nc_steps, all_nc3)
    
    print("Interpolating angles...")
    common_angle_steps, interp_mean_angles = interpolate_to_common_steps(all_angle_steps, all_mean_angles)
    _, interp_classifier_angles = interpolate_to_common_steps(all_angle_steps, all_classifier_angles)
    _, interp_etf_angles = interpolate_to_common_steps(all_angle_steps, all_etf_angles)
    
    # Compute mean and std
    print("Computing statistics...")
    averaged_metrics = AveragedMetrics(
        # Training curves
        train_steps=common_train_steps,
        train_loss_mean=np.mean(interp_train_loss, axis=0),
        train_loss_std=np.std(interp_train_loss, axis=0),
        train_acc_mean=np.mean(interp_train_acc, axis=0),
        train_acc_std=np.std(interp_train_acc, axis=0),
        test_loss_mean=np.mean(interp_test_loss, axis=0),
        test_loss_std=np.std(interp_test_loss, axis=0),
        test_acc_mean=np.mean(interp_test_acc, axis=0),
        test_acc_std=np.std(interp_test_acc, axis=0),
        
        # NC metrics
        nc_steps=common_nc_steps,
        nc1_mean=np.mean(interp_nc1, axis=0),
        nc1_std=np.std(interp_nc1, axis=0),
        nc2_mean=np.mean(interp_nc2, axis=0),
        nc2_std=np.std(interp_nc2, axis=0),
        nc3_mean=np.mean(interp_nc3, axis=0),
        nc3_std=np.std(interp_nc3, axis=0),
        
        # Angles
        angle_steps=common_angle_steps,
        mean_angles_mean=np.mean(interp_mean_angles, axis=0),
        mean_angles_std=np.std(interp_mean_angles, axis=0),
        classifier_angles_mean=np.mean(interp_classifier_angles, axis=0),
        classifier_angles_std=np.std(interp_classifier_angles, axis=0),
        etf_angles_mean=np.mean(interp_etf_angles, axis=0),
        etf_angles_std=np.std(interp_etf_angles, axis=0),
        
        # Metadata
        num_seeds=len(all_train_steps),
        seeds_used=sorted(seeds_used) if seeds_used else list(range(len(all_train_steps))),
        config_name=config_name,
        experiment_name=experiment_name
    )
    
    return averaged_metrics


def plot_averaged_training_curves(metrics: AveragedMetrics, output_dir: Path):
    """Plot averaged training curves with confidence bands."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot accuracy
    ax = axes[0]
    ax.plot(metrics.train_steps, metrics.train_acc_mean, 'b-', label='Train Accuracy', linewidth=2)
    ax.fill_between(
        metrics.train_steps,
        metrics.train_acc_mean - metrics.train_acc_std,
        metrics.train_acc_mean + metrics.train_acc_std,
        alpha=0.3, color='b'
    )
    
    ax.plot(metrics.train_steps, metrics.test_acc_mean, 'r-', label='Test Accuracy', linewidth=2)
    ax.fill_between(
        metrics.train_steps,
        metrics.test_acc_mean - metrics.test_acc_std,
        metrics.test_acc_mean + metrics.test_acc_std,
        alpha=0.3, color='r'
    )
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Averaged Training Accuracy (n={metrics.num_seeds} seeds)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot loss
    ax = axes[1]
    ax.plot(metrics.train_steps, metrics.train_loss_mean, 'b-', label='Train Loss', linewidth=2)
    ax.fill_between(
        metrics.train_steps,
        metrics.train_loss_mean - metrics.train_loss_std,
        metrics.train_loss_mean + metrics.train_loss_std,
        alpha=0.3, color='b'
    )
    
    ax.plot(metrics.train_steps, metrics.test_loss_mean, 'r-', label='Test Loss', linewidth=2)
    ax.fill_between(
        metrics.train_steps,
        metrics.test_loss_mean - metrics.test_loss_std,
        metrics.test_loss_mean + metrics.test_loss_std,
        alpha=0.3, color='r'
    )
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Averaged Training Loss (n={metrics.num_seeds} seeds)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'averaged_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'averaged_training_curves.png'}")


def plot_averaged_nc_metrics(metrics: AveragedMetrics, output_dir: Path):
    """Plot averaged Neural Collapse metrics."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    # NC1: Within-class variance
    ax = axes[0]
    ax.plot(metrics.nc_steps, metrics.nc1_mean, 'b-', linewidth=2)
    ax.fill_between(
        metrics.nc_steps,
        metrics.nc1_mean - metrics.nc1_std,
        metrics.nc1_mean + metrics.nc1_std,
        alpha=0.3, color='b'
    )
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Within-Class Variance', fontsize=12)
    ax.set_title(f'NC1: Variability Collapse (n={metrics.num_seeds} seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # NC2: ETF alignment
    ax = axes[1]
    ax.plot(metrics.nc_steps, metrics.nc2_mean, 'g-', linewidth=2)
    ax.fill_between(
        metrics.nc_steps,
        metrics.nc2_mean - metrics.nc2_std,
        metrics.nc2_mean + metrics.nc2_std,
        alpha=0.3, color='g'
    )
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('ETF Alignment', fontsize=12)
    ax.set_title(f'NC2: Convergence to Simplex ETF (n={metrics.num_seeds} seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax.legend(fontsize=11)
    
    # NC3: Self-duality
    ax = axes[2]
    ax.plot(metrics.nc_steps, metrics.nc3_mean, 'r-', linewidth=2)
    ax.fill_between(
        metrics.nc_steps,
        metrics.nc3_mean - metrics.nc3_std,
        metrics.nc3_mean + metrics.nc3_std,
        alpha=0.3, color='r'
    )
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Classifier-Mean Alignment', fontsize=12)
    ax.set_title(f'NC3: Self-Duality (n={metrics.num_seeds} seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect alignment')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'averaged_nc_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'averaged_nc_metrics.png'}")


def plot_averaged_angles(metrics: AveragedMetrics, output_dir: Path):
    """Plot averaged angles between vectors."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot mean angles
    ax.plot(metrics.angle_steps, metrics.mean_angles_mean, 'b-', 
            label='Class Means', linewidth=2, marker='o', markersize=4)
    ax.fill_between(
        metrics.angle_steps,
        metrics.mean_angles_mean - metrics.mean_angles_std,
        metrics.mean_angles_mean + metrics.mean_angles_std,
        alpha=0.3, color='b'
    )
    
    # Plot classifier angles
    ax.plot(metrics.angle_steps, metrics.classifier_angles_mean, 'r-', 
            label='Classifiers', linewidth=2, marker='s', markersize=4)
    ax.fill_between(
        metrics.angle_steps,
        metrics.classifier_angles_mean - metrics.classifier_angles_std,
        metrics.classifier_angles_mean + metrics.classifier_angles_std,
        alpha=0.3, color='r'
    )
    
    # Plot ETF angles (theoretical)
    ax.plot(metrics.angle_steps, metrics.etf_angles_mean, 'g--', 
            label='Simplex ETF (Theoretical)', linewidth=2, marker='^', markersize=4)
    ax.fill_between(
        metrics.angle_steps,
        metrics.etf_angles_mean - metrics.etf_angles_std,
        metrics.etf_angles_mean + metrics.etf_angles_std,
        alpha=0.3, color='g'
    )
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Pairwise Angle (degrees)', fontsize=12)
    ax.set_title(f'Averaged Pairwise Angles (n={metrics.num_seeds} seeds)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'averaged_angles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'averaged_angles.png'}")


def save_averaged_metrics(metrics: AveragedMetrics, output_dir: Path):
    """Save averaged metrics to files."""
    
    # Save as pickle for later analysis
    with open(output_dir / 'averaged_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Saved: {output_dir / 'averaged_metrics.pkl'}")
    
    # Save training curves as CSV
    train_df = pd.DataFrame({
        'Step': metrics.train_steps,
        'Train_Acc_Mean': metrics.train_acc_mean,
        'Train_Acc_Std': metrics.train_acc_std,
        'Test_Acc_Mean': metrics.test_acc_mean,
        'Test_Acc_Std': metrics.test_acc_std,
        'Train_Loss_Mean': metrics.train_loss_mean,
        'Train_Loss_Std': metrics.train_loss_std,
        'Test_Loss_Mean': metrics.test_loss_mean,
        'Test_Loss_Std': metrics.test_loss_std
    })
    train_df.to_csv(output_dir / 'averaged_training_curves.csv', index=False)
    print(f"Saved: {output_dir / 'averaged_training_curves.csv'}")
    
    # Save NC metrics as CSV
    nc_df = pd.DataFrame({
        'Step': metrics.nc_steps,
        'NC1_Mean': metrics.nc1_mean,
        'NC1_Std': metrics.nc1_std,
        'NC2_Mean': metrics.nc2_mean,
        'NC2_Std': metrics.nc2_std,
        'NC3_Mean': metrics.nc3_mean,
        'NC3_Std': metrics.nc3_std
    })
    nc_df.to_csv(output_dir / 'averaged_nc_metrics.csv', index=False)
    print(f"Saved: {output_dir / 'averaged_nc_metrics.csv'}")
    
    # Save angles as CSV
    angles_df = pd.DataFrame({
        'Step': metrics.angle_steps,
        'Mean_Angles_Mean': metrics.mean_angles_mean,
        'Mean_Angles_Std': metrics.mean_angles_std,
        'Classifier_Angles_Mean': metrics.classifier_angles_mean,
        'Classifier_Angles_Std': metrics.classifier_angles_std,
        'ETF_Angles_Mean': metrics.etf_angles_mean,
        'ETF_Angles_Std': metrics.etf_angles_std
    })
    angles_df.to_csv(output_dir / 'averaged_angles.csv', index=False)
    print(f"Saved: {output_dir / 'averaged_angles.csv'}")
    
    # Save summary statistics
    summary = {
        'config_name': metrics.config_name,
        'experiment_name': metrics.experiment_name,
        'num_seeds': metrics.num_seeds,
        'seeds_used': metrics.seeds_used,
        'final_train_acc_mean': float(metrics.train_acc_mean[-1]),
        'final_train_acc_std': float(metrics.train_acc_std[-1]),
        'final_test_acc_mean': float(metrics.test_acc_mean[-1]),
        'final_test_acc_std': float(metrics.test_acc_std[-1]),
        'final_nc1_mean': float(metrics.nc1_mean[-1]),
        'final_nc1_std': float(metrics.nc1_std[-1]),
        'final_nc2_mean': float(metrics.nc2_mean[-1]),
        'final_nc2_std': float(metrics.nc2_std[-1]),
        'final_nc3_mean': float(metrics.nc3_mean[-1]),
        'final_nc3_std': float(metrics.nc3_std[-1]),
    }
    
    with open(output_dir / 'averaged_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_dir / 'averaged_summary.json'}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("AVERAGED METRICS SUMMARY")
    print("=" * 80)
    print(f"Configuration: {metrics.config_name}")
    print(f"Experiment: {metrics.experiment_name}")
    print(f"Number of seeds: {metrics.num_seeds}")
    print(f"Seeds used: {metrics.seeds_used}")
    print("-" * 80)
    print(f"Final Train Accuracy: {summary['final_train_acc_mean']:.4f} ± {summary['final_train_acc_std']:.4f}")
    print(f"Final Test Accuracy:  {summary['final_test_acc_mean']:.4f} ± {summary['final_test_acc_std']:.4f}")
    print("-" * 80)
    print(f"Final NC1 (Within-class variance): {summary['final_nc1_mean']:.6f} ± {summary['final_nc1_std']:.6f}")
    print(f"Final NC2 (ETF alignment):          {summary['final_nc2_mean']:.6f} ± {summary['final_nc2_std']:.6f}")
    print(f"Final NC3 (Self-duality):           {summary['final_nc3_mean']:.6f} ± {summary['final_nc3_std']:.6f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Average Neural Collapse metrics across multiple seeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Average all experiments for narrow_bump with dynamics
  python average_nc_metrics.py --config nc_config_narrow_bump --experiment_name nc_with_dynamics
  
  # Average without dynamics (baseline)
  python average_nc_metrics.py --config nc_config_baseline --experiment_name nc_without_dynamics
  
  # Specify custom output directory
  python average_nc_metrics.py --config nc_config_wide_bump --experiment_name nc_with_dynamics \\
      --output_dir /path/to/results
  
  # Filter by timestamp
  python average_nc_metrics.py --config nc_config_narrow_bump --experiment_name nc_with_dynamics \\
      --timestamp 2025_12_14
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration name (e.g., nc_config_narrow_bump)'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        choices=['nc_with_dynamics', 'nc_without_dynamics'],
        help='Experiment name'
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd',
        help='Base directory containing experiment results'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for averaged results (default: base_dir/<experiment_name>/<config>/averaged)'
    )
    
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help='Filter experiments by timestamp (e.g., 2025_12_14)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Save in the same directory structure as experiments
        output_dir = base_dir / args.experiment_name / args.config / 'averaged'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AVERAGING NEURAL COLLAPSE METRICS ACROSS SEEDS")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    if args.timestamp:
        print(f"Timestamp filter: {args.timestamp}")
    print("=" * 80)
    
    # Find experiment directories
    print("\nSearching for experiments...")
    experiment_dirs = find_experiment_directories(
        base_dir=base_dir,
        experiment_name=args.experiment_name,
        config_name=args.config,
        timestamp_filter=args.timestamp
    )
    
    if not experiment_dirs:
        print(f"\nError: No experiments found!")
        print(f"Search path: {base_dir / args.experiment_name / args.config}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir.name}")
    
    # Average metrics
    averaged_metrics = average_experiment_metrics(
        experiment_dirs=experiment_dirs,
        config_name=args.config,
        experiment_name=args.experiment_name
    )
    
    # Create plots
    print("\nCreating visualizations...")
    plot_averaged_training_curves(averaged_metrics, output_dir)
    plot_averaged_nc_metrics(averaged_metrics, output_dir)
    plot_averaged_angles(averaged_metrics, output_dir)
    
    # Save results
    print("\nSaving results...")
    save_averaged_metrics(averaged_metrics, output_dir)
    
    print("\n" + "=" * 80)
    print("AVERAGING COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
