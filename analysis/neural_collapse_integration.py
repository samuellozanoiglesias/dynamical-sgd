"""
Integration module for Neural Collapse analysis (Simplified).

This module provides simple integration between the training loop and Neural Collapse
metrics computation, using the paper's mathematical definitions.

Usage:
    from analysis.neural_collapse_integration import (
        plot_nc_metrics_evolution
    )
"""

from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

# Import from simplified neural_collapse module
from analysis.neural_collapse import (
    NeuralCollapseAnalyzer,
    NeuralCollapseSnapshot,
    compute_nc_metrics,
    get_features_and_weights,
    plot_nc_metrics
)


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================
# Note: Use classifier.get_features(params, X) to extract h(x) features


def plot_nc_metrics_evolution(nc_metrics_history: List[Tuple[int, dict]], output_dir: Path, step_100_acc: Optional[int] = None, tpt_threshold: float = 1.0):
    """
    Plot the evolution of Neural Collapse metrics over training (wrapper for compatibility).
    
    ✅ These metrics are computed in ORIGINAL feature space R^p, not after projection.
    
    Args:
        nc_metrics_history: List of (step, metrics_dict) tuples
        output_dir: Directory to save the plot
        step_100_acc: If provided, add vertical line at step where TPT accuracy was reached
        tpt_threshold: TPT accuracy threshold used (e.g., 1.0 = 100%, 0.99 = 99%)
    """
    # Use the new simplified plotting function
    plot_nc_metrics(nc_metrics_history, output_dir, log_scale=True, step_100_acc=step_100_acc, tpt_threshold=tpt_threshold)


def plot_angle_convergence_evolution(nc_analyzer: NeuralCollapseAnalyzer, output_dir: Path):
    """
    Plot angle convergence evolution (simplified version - placeholder).
    
    In the simplified version, we focus on the paper's metrics.
    This function is kept for backward compatibility but doesn't generate plots.
    
    Args:
        nc_analyzer: NeuralCollapseAnalyzer with snapshots
        output_dir: Directory to save the plot
    """
    import logging
    logging.info("Angle convergence plotting skipped in simplified NC version")
    logging.info("All NC metrics are captured in nc_metrics_paper_style.png")


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def train_with_neural_collapse(
    classifier,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_test=None,
    Y_test=None,
    snapshot_epochs: List[int] = None,
    output_dir: Path = Path('outputs/nc_analysis'),
    **training_kwargs
) -> NeuralCollapseAnalyzer:
    """
    Train classifier with Neural Collapse analysis integration (placeholder).
    
    Note: For full training integration, use run_experiment.py with track_neural_collapse=true
    
    Args:
        classifier: SpiralClassifier instance
        X_train: Training data
        Y_train: Training labels  
        X_test: Test data (optional)
        Y_test: Test labels (optional)
        snapshot_epochs: List of epochs to capture snapshots
        output_dir: Output directory for results
        **training_kwargs: Additional training arguments
        
    Returns:
        NeuralCollapseAnalyzer with captured snapshots
    """
    import logging
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize NC analyzer with num_hidden_layers
    nc_analyzer = NeuralCollapseAnalyzer(
        num_classes=classifier.num_classes,
        feature_dim=classifier.nn_width,
        num_hidden_layers=getattr(classifier, 'num_hidden_layers', 1),
        use_batchnorm=getattr(classifier, 'use_batchnorm', True),
        use_bias=getattr(classifier, 'use_bias', True)
    )
    
    if snapshot_epochs is None:
        snapshot_epochs = [0, 1000, 5000, 10000, 25000, 50000, 75000]
    
    logging.info(f"Neural Collapse training wrapper initialized")
    logging.info(f"Snapshot epochs: {snapshot_epochs}")
    logging.info(f"For full training integration, use run_experiment.py with track_neural_collapse=true")
    
    return nc_analyzer


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
    
    # Extract features using classifier's get_features method
    features = classifier.get_features(params, X_train)
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
    
    # Extract classifiers (W) and biases (b) from final layer
    num_hidden_layers = getattr(classifier, 'num_hidden_layers', 1)
    use_batchnorm = getattr(classifier, 'use_batchnorm', True)
    
    # Calculate classifier index based on architecture
    # In JAX stax, each LAYER creates ONE entry (including empty tuples for ReLU):
    #   - Dense = 1 entry: (W, b)
    #   - BatchNorm = 1 entry: (gamma, beta, mean, var)
    #   - ReLU = 1 entry: () empty tuple
    if use_batchnorm:
        classifier_idx = num_hidden_layers * 3  # Dense + BatchNorm + ReLU
    else:
        classifier_idx = num_hidden_layers * 2  # Dense + ReLU
    
    classifiers = params[classifier_idx][0].T  # (num_classes, hidden_dim)
    biases = params[classifier_idx][1]  # (num_classes,)
    
    # Compute metrics
    metrics = compute_nc_metrics(features, labels, classifiers)
    
    # Create snapshot
    snapshot = NeuralCollapseSnapshot(
        epoch=-1,  # Unknown epoch
        features=features,
        labels=labels,
        class_means=class_means,
        classifiers=classifiers,
        biases=biases,
        num_classes=classifier.num_classes,
        feature_dim=classifier.nn_width,
        metrics=metrics
    )
    
    return snapshot
