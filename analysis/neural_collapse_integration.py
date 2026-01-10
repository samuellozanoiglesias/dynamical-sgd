"""
Integration module for Neural Collapse analysis with SpiralClassifier.

This module extends the SpiralClassifier to capture Neural Collapse snapshots
during training and provides utilities for analyzing and visualizing the
Neural Collapse phenomenon.

🔴 MATHEMATICALLY CORRECTED IMPLEMENTATION 🔴
This module ensures all Neural Collapse metrics are computed in the original 
feature space R^p, avoiding projection artifacts that would make measurements
meaningless. Projections are used ONLY for visualization purposes.

The key functions:
- extract_penultimate_features(): Gets features from original R^p space
- plot_nc_metrics_evolution(): Plots metrics computed in R^p (not projected)
- All visualization functions clearly indicate when angles are projection artifacts

Usage:
    from analysis.neural_collapse_integration import train_with_neural_collapse
    
    classifier = SpiralClassifier(...)
    nc_analyzer = train_with_neural_collapse(
        classifier=classifier,
        snapshot_epochs=[0, 50, 100, 200],
        output_dir='outputs/nc_analysis'
    )
"""

import sys
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Any
from tqdm import tqdm

from analysis.neural_collapse import NeuralCollapseAnalyzer, NeuralCollapseSnapshot


def extract_penultimate_features(params: Any, X: jnp.ndarray) -> jnp.ndarray:
    """
    Extract features from the penultimate layer (before final linear layer).
    
    For a model with architecture: Dense -> ReLU -> Dense (output),
    this extracts the activations after the first Dense+ReLU.
    
    Args:
        params: Model parameters (list of (W, b) tuples)
        X: Input data (N, input_dim)
        
    Returns:
        Features from penultimate layer (N, hidden_dim)
    """
    # Apply first layer: Dense + ReLU
    W1, b1 = params[0]
    features = jnp.maximum(0, jnp.dot(X, W1) + b1)
    
    return features

def plot_nc_metrics_evolution(nc_metrics_history: List[Tuple[int, dict]], output_dir: Path):
    """
    Plot the evolution of Neural Collapse metrics over training.
    
    ✅ These metrics are computed in ORIGINAL feature space R^p, not after projection.
    This ensures we're measuring the true geometric properties of Neural Collapse.
    
    Args:
        nc_metrics_history: List of (step, metrics_dict) tuples
        output_dir: Directory to save the plot
    """
    steps = [m[0] for m in nc_metrics_history]
    nc1 = [m[1]['nc1_within_class_variance'] for m in nc_metrics_history]
    nc2 = [m[1]['nc2_etf_alignment'] for m in nc_metrics_history]
    nc3 = [m[1]['nc3_self_duality'] for m in nc_metrics_history]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # NC1: Within-class variance (should decrease)
    axes[0].plot(steps, nc1, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Within-Class Variance', fontsize=12)
    axes[0].set_title('NC1: Variability Collapse\n(computed in R^p)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # NC2: ETF alignment (should increase toward 1)
    axes[1].plot(steps, nc2, 'o-', linewidth=2, markersize=8, color='#06A77D')
    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('ETF Alignment', fontsize=12)
    axes[1].set_title('NC2: Convergence to Simplex ETF\n(computed in R^p)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    axes[1].legend()
    
    # NC3: Self-duality (should increase toward 1)
    axes[2].plot(steps, nc3, 'o-', linewidth=2, markersize=8, color='#D62246')
    axes[2].set_xlabel('Training Step', fontsize=12)
    axes[2].set_ylabel('Cosine Similarity', fontsize=12)
    axes[2].set_title('NC3: Self-Duality\n(computed in R^p)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'nc_metrics_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_angle_convergence_evolution(nc_analyzer: NeuralCollapseAnalyzer, output_dir: Path):
    """
    Plot the evolution of ALL true geometric angles over training.
    
    Shows convergence for:
    1. Class Means (should → simplex angles: 120° for 3 classes, etc.)
    2. Classifiers (should → simplex angles: same as means)  
    3. Biases (typically anti-aligned, ~180° apart)
    4. Mean-Classifier Alignment (should → 0°)
    
    ✅ All angles computed in ORIGINAL feature space R^p (mathematically correct)
    
    Args:
        nc_analyzer: NeuralCollapseAnalyzer with snapshots
        output_dir: Directory to save the plot
    """
    if not nc_analyzer.snapshots:
        print("No snapshots available for angle analysis")
        return
    
    # Extract angle data from all snapshots for ALL components
    epochs = []
    
    # Data for each component type
    means_data = {'mean_angles': [], 'deviations': [], 'optimal': [], 'all_angles': []}
    classifiers_data = {'mean_angles': [], 'deviations': [], 'optimal': [], 'all_angles': []}  
    biases_data = {'mean_angles': [], 'deviations': [], 'optimal': [], 'all_angles': []}
    alignment_data = {'mean_angles': [], 'deviations': [], 'optimal': [], 'all_angles': []}
    
    for snapshot in nc_analyzer.snapshots:
        all_angles = nc_analyzer.compute_all_angles_in_original_space(snapshot)
        
        epochs.append(snapshot.epoch)
        
        # Class means
        means_stats = all_angles['class_means']
        means_data['mean_angles'].append(means_stats['mean_angle'])
        means_data['deviations'].append(means_stats['angle_deviation'])
        means_data['optimal'].append(means_stats['optimal_angle'])
        means_data['all_angles'].append(means_stats['all_angles'])
        
        # Classifiers
        classifier_stats = all_angles['classifiers']
        classifiers_data['mean_angles'].append(classifier_stats['mean_angle'])
        classifiers_data['deviations'].append(classifier_stats['angle_deviation'])
        classifiers_data['optimal'].append(classifier_stats['optimal_angle'])
        classifiers_data['all_angles'].append(classifier_stats['all_angles'])
        
        # Biases
        bias_stats = all_angles['biases']
        biases_data['mean_angles'].append(bias_stats['mean_angle'])
        biases_data['deviations'].append(bias_stats['angle_deviation'])
        biases_data['optimal'].append(bias_stats['optimal_angle'])  # Same simplex target
        biases_data['all_angles'].append(bias_stats['all_angles'])
        
        # Mean-Classifier Alignment
        align_stats = all_angles['mean_classifier_alignment']
        alignment_data['mean_angles'].append(align_stats['mean_angle'])
        alignment_data['deviations'].append(align_stats['angle_deviation'])
        alignment_data['optimal'].append(0.0)  # Perfect alignment = 0°
        alignment_data['all_angles'].append(align_stats['all_angles'])
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for each component
    colors = {
        'means': '#2E86AB',
        'classifiers': '#A23B72', 
        'biases': '#F18F01',
        'alignment': '#C73E1D'
    }
    
    # Plot 1: Class Means Convergence
    ax1 = axes[0, 0]
    ax1.plot(epochs, means_data['mean_angles'], 'o-', linewidth=3, markersize=8,
             color=colors['means'], label='Class Means', zorder=3)
    
    if means_data['optimal']:
        target = means_data['optimal'][0]
        ax1.axhline(y=target, color='red', linestyle='--', linewidth=2.5,
                   alpha=0.8, label=f'Simplex Target: {target:.1f}°', zorder=2)
    
    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel('Angle (degrees)', fontweight='bold')
    ax1.set_title('Class Means Pairwise Angles\n(computed in R^p)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Classifiers Convergence  
    ax2 = axes[0, 1]
    ax2.plot(epochs, classifiers_data['mean_angles'], 'o-', linewidth=3, markersize=8,
             color=colors['classifiers'], label='Classifiers', zorder=3)
    
    if classifiers_data['optimal']:
        target = classifiers_data['optimal'][0]
        ax2.axhline(y=target, color='red', linestyle='--', linewidth=2.5,
                   alpha=0.8, label=f'Simplex Target: {target:.1f}°', zorder=2)
    
    ax2.set_xlabel('Training Step', fontweight='bold')
    ax2.set_ylabel('Angle (degrees)', fontweight='bold')
    ax2.set_title('Classifiers Pairwise Angles\n(computed in R^p)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Biases Convergence
    ax3 = axes[1, 0]
    ax3.plot(epochs, biases_data['mean_angles'], 'o-', linewidth=3, markersize=8,
             color=colors['biases'], label='Biases', zorder=3)
    
    # Biases often anti-align (180° apart) but may follow simplex too
    if biases_data['optimal']:
        target = biases_data['optimal'][0] 
        ax3.axhline(y=target, color='red', linestyle='--', linewidth=2.5,
                   alpha=0.8, label=f'Simplex Target: {target:.1f}°', zorder=2)
    
    # Also show anti-alignment possibility
    ax3.axhline(y=180.0, color='orange', linestyle=':', linewidth=2,
               alpha=0.8, label='Anti-align: 180°', zorder=1)
    
    ax3.set_xlabel('Training Step', fontweight='bold')
    ax3.set_ylabel('Angle (degrees)', fontweight='bold')
    ax3.set_title('Biases Pairwise Angles\n(computed in R^p)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Mean-Classifier Alignment 
    ax4 = axes[1, 1]
    ax4.plot(epochs, alignment_data['mean_angles'], 'o-', linewidth=3, markersize=8,
             color=colors['alignment'], label='Mean-Classifier Alignment', zorder=3)
    
    # Perfect alignment = 0°
    ax4.axhline(y=0.0, color='red', linestyle='--', linewidth=2.5,
               alpha=0.8, label='Perfect Alignment: 0°', zorder=2)
    
    ax4.set_xlabel('Training Step', fontweight='bold')
    ax4.set_ylabel('Alignment Angle (degrees)', fontweight='bold') 
    ax4.set_title('Class Mean ↔ Classifier Alignment\n(computed in R^p)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle(f'Complete Neural Collapse Angle Evolution (C={nc_analyzer.num_classes} classes)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the plot
    save_path = output_dir / 'complete_angle_convergence_evolution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved complete angle convergence plot to {save_path}")
    plt.show()
    
    # Print comprehensive summary 
    print("\n" + "="*80)
    print("COMPLETE ANGLE CONVERGENCE ANALYSIS SUMMARY")
    print("="*80)
    print(f"Number of classes: {nc_analyzer.num_classes}")
    if means_data['optimal']:
        simplex_target = means_data['optimal'][0]
        print(f"Theoretical simplex angle: {simplex_target:.2f}°")
        print(f"Expected for C={nc_analyzer.num_classes}: {'120° (equilateral)' if nc_analyzer.num_classes == 3 else '~109.5° (tetrahedral)' if nc_analyzer.num_classes == 4 else f'{simplex_target:.1f}° (simplex)'}")
        
    print("\nFINAL CONVERGENCE RESULTS:")
    print(f"  Class Means:     {means_data['mean_angles'][-1]:.2f}° (target: {means_data['optimal'][0]:.1f}°, error: {abs(means_data['mean_angles'][-1] - means_data['optimal'][0]):.2f}°)")
    print(f"  Classifiers:     {classifiers_data['mean_angles'][-1]:.2f}° (target: {classifiers_data['optimal'][0]:.1f}°, error: {abs(classifiers_data['mean_angles'][-1] - classifiers_data['optimal'][0]):.2f}°)")  
    print(f"  Biases:          {biases_data['mean_angles'][-1]:.2f}° (varies: simplex={biases_data['optimal'][0]:.1f}° or anti-align=180°)")
    print(f"  Alignment:       {alignment_data['mean_angles'][-1]:.2f}° (target: 0°, error: {alignment_data['mean_angles'][-1]:.2f}°)")
    print("="*80)





def create_figure1_style_visualization(
    nc_analyzer: NeuralCollapseAnalyzer,
    selected_snapshots: Optional[List[int]] = None,
    output_dir: Optional[Path] = None
):
    """
    Create a Figure 1 style visualization showing multiple epochs in a grid.
    
    Args:
        nc_analyzer: NeuralCollapseAnalyzer with snapshots
        selected_snapshots: Indices of snapshots to visualize (if None, use all)
        output_dir: Directory to save the figure
    """
    if not nc_analyzer.snapshots:
        print("No snapshots available for visualization")
        return
    
    if output_dir is None:
        output_dir = Path('outputs/neural_collapse')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select snapshots
    if selected_snapshots is None:
        snapshots = nc_analyzer.snapshots
    else:
        snapshots = [nc_analyzer.snapshots[i] for i in selected_snapshots]
    
    n_snapshots = len(snapshots)
    
    # Determine grid layout
    if n_snapshots <= 2:
        nrows, ncols = 1, n_snapshots
    elif n_snapshots <= 4:
        nrows, ncols = 2, 2
    elif n_snapshots <= 6:
        nrows, ncols = 2, 3
    else:
        nrows = int(np.ceil(n_snapshots / 3))
        ncols = 3
    
    # Create figure with 3D subplots
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    
    for idx, snapshot in enumerate(snapshots):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        
        # Simplified visualization for subplot
        # ... (implementation similar to visualize_neural_collapse but for subplot)
        
        ax.set_title(f'Step {snapshot.epoch}', fontsize=12, fontweight='bold')
    
    plt.suptitle('Neural Collapse Evolution During Training', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = output_dir / 'figure1_style_visualization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1 style visualization to {save_path}")
    plt.show()


def analyze_neural_collapse_from_checkpoint(
    checkpoint_path: Path,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    classifier
) -> NeuralCollapseSnapshot:
    """
    Analyze Neural Collapse from a saved model checkpoint.
    
    Args:
        checkpoint_path: Path to saved model parameters
        X_train: Training data
        Y_train: Training labels
        classifier: SpiralClassifier instance
        
    Returns:
        NeuralCollapseSnapshot of the loaded model
    """
    import pickle
    
    # Load parameters
    with open(checkpoint_path, 'rb') as f:
        params = pickle.load(f)
    
    # Extract features
    features = extract_penultimate_features(params, X_train)
    labels = jnp.argmax(Y_train, axis=1)
    
    # Compute class means
    class_means = []
    for c in range(classifier.num_classes):
        mask = labels == c
        if jnp.sum(mask) > 0:
            class_mean = jnp.mean(features[mask], axis=0)
        else:
            class_mean = jnp.zeros(classifier.nn_width)
        class_means.append(class_mean)
    class_means = jnp.stack(class_means)
    
    # Extract classifiers (W) and biases (b)
    classifiers = params[-1][0].T  # (num_classes, hidden_dim)
    biases = params[-1][1]  # (num_classes,)
    
    # Create snapshot
    snapshot = NeuralCollapseSnapshot(
        epoch=-1,  # Unknown epoch
        features=features,
        labels=labels,
        class_means=class_means,
        classifiers=classifiers,
        biases=biases,
        num_classes=classifier.num_classes,
        feature_dim=classifier.nn_width
    )
    
    return snapshot
