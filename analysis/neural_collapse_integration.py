"""
Integration module for Neural Collapse analysis with SpiralClassifier.

This module extends the SpiralClassifier to capture Neural Collapse snapshots
during training and provides utilities for analyzing and visualizing the
Neural Collapse phenomenon.

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
    axes[0].set_title('NC1: Variability Collapse', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # NC2: ETF alignment (should increase toward 1)
    axes[1].plot(steps, nc2, 'o-', linewidth=2, markersize=8, color='#06A77D')
    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('ETF Alignment', fontsize=12)
    axes[1].set_title('NC2: Convergence to Simplex ETF', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    axes[1].legend()
    
    # NC3: Self-duality (should increase toward 1)
    axes[2].plot(steps, nc3, 'o-', linewidth=2, markersize=8, color='#D62246')
    axes[2].set_xlabel('Training Step', fontsize=12)
    axes[2].set_ylabel('Cosine Similarity', fontsize=12)
    axes[2].set_title('NC3: Self-Duality', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect alignment')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'nc_metrics_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()


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
