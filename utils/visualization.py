"""
Visualization utilities for neural network dynamics analysis.

This module provides functions for visualizing decision boundaries, weight evolution,
gradient distributions, and other aspects of neural network training dynamics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import jax.numpy as jnp
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict
import jax

# Configure matplotlib and seaborn
sns.set(style="white")


def plot_spiral_dataset(
    X: jnp.ndarray, 
    Y: jnp.ndarray, 
    title: str = "Spiral Dataset",
    figsize: Tuple[int, int] = (8, 8),
    colors: List[str] = None
) -> None:
    """
    Plot the spiral dataset with different colors for each class.
    
    Args:
        X: Input features of shape (N, 2)
        Y: One-hot encoded labels of shape (N, num_classes)
        title: Plot title
        figsize: Figure size
        colors: List of colors for each class
    """
    if colors is None:
        colors = ['#8B0000', '#00008B', '#FFD700']  # Dark red, dark blue, gold
    
    plt.figure(figsize=figsize)
    
    # Convert one-hot to class indices
    labels = np.argmax(Y, axis=1)
    
    # Plot each class
    for class_idx in range(Y.shape[1]):
        mask = labels == class_idx
        plt.scatter(
            X[mask, 0], X[mask, 1],
            c=colors[class_idx], 
            s=40, 
            edgecolor='k', 
            marker='o',
            label=f'Class {class_idx + 1}',
            alpha=0.7
        )
    
    plt.title(title, fontsize=16)
    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Y coordinate', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(
    model_fn: Any,
    params: Any, 
    X_train: jnp.ndarray, 
    Y_train: jnp.ndarray,
    X_test: Optional[jnp.ndarray] = None,
    Y_test: Optional[jnp.ndarray] = None,
    title: str = "Decision Boundary",
    resolution: float = 0.01,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Visualize the decision boundary of a trained neural network.
    
    Args:
        model_fn: Model function that takes (params, X) and returns predictions
        params: Trained model parameters
        X_train: Training data features
        Y_train: Training data labels
        X_test: Optional test data features
        Y_test: Optional test data labels
        title: Plot title
        resolution: Grid resolution for decision boundary
        figsize: Figure size
    """
    # Define plot boundaries
    x_min, x_max = float(X_train[:, 0].min() - 0.3), float(X_train[:, 0].max() + 0.3)
    y_min, y_max = float(X_train[:, 1].min() - 0.3), float(X_train[:, 1].max() + 0.3)
    
    # Create mesh grid
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    
    # Get predictions on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_fn(params, jax.device_put(grid_points))
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Plot decision boundary
    colors = ['#FFB6C1', '#ADD8E6', '#FFFFE0']  # Light colors for regions
    plt.contourf(xx, yy, Z, alpha=0.6, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors)
    
    # Plot training data
    train_labels = np.argmax(Y_train, axis=1)
    train_colors = ['#8B0000', '#00008B', '#FFD700']  # Darker colors for points
    
    for class_idx in range(Y_train.shape[1]):
        mask = train_labels == class_idx
        plt.scatter(
            X_train[mask, 0], X_train[mask, 1],
            c=train_colors[class_idx],
            s=50,
            edgecolor='black',
            marker='o',
            label=f'Train Class {class_idx + 1}',
            alpha=0.8
        )
    
    # Plot test data if provided
    if X_test is not None and Y_test is not None:
        test_labels = np.argmax(Y_test, axis=1)
        test_colors = ['#FF6347', '#1E90FF', '#FFD700']  # Tomato, dodger blue, gold
        
        for class_idx in range(Y_test.shape[1]):
            mask = test_labels == class_idx
            plt.scatter(
                X_test[mask, 0], X_test[mask, 1],
                c=test_colors[class_idx],
                s=70,
                edgecolor='none',
                marker='s',
                label=f'Test Class {class_idx + 1}',
                alpha=0.7
            )
    
    plt.title(title, fontsize=16)
    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Y coordinate', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


def plot_training_curves(
    losses: List[float],
    accuracies: List[float],
    test_losses: Optional[List[float]] = None,
    test_accuracies: Optional[List[float]] = None,
    metric_steps: Optional[List[int]] = None,
    period_length: Optional[int] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        losses: Training losses
        accuracies: Training accuracies
        test_losses: Optional test losses
        test_accuracies: Optional test accuracies
        metric_steps: Actual step numbers corresponding to metrics (if None, uses indices)
        period_length: If provided, add vertical lines at period boundaries
        title: Plot title
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Expand metrics to fill all intermediate steps (step function)
    if metric_steps is not None and len(metric_steps) > 0:
        # Create expanded arrays where metrics are held constant between validation intervals
        max_step = max(metric_steps)
        expanded_steps = list(range(max_step + 1))
        expanded_losses = []
        expanded_accuracies = []
        expanded_test_losses = []
        expanded_test_accuracies = []
        
        for step in expanded_steps:
            # Find the most recent metric_step <= current step
            idx = 0
            for i, metric_step in enumerate(metric_steps):
                if metric_step <= step:
                    idx = i
                else:
                    break
            
            expanded_losses.append(losses[idx])
            expanded_accuracies.append(accuracies[idx])
            if test_losses is not None and idx < len(test_losses):
                expanded_test_losses.append(test_losses[idx])
            if test_accuracies is not None and idx < len(test_accuracies):
                expanded_test_accuracies.append(test_accuracies[idx])
        
        steps = expanded_steps
        losses = expanded_losses
        accuracies = expanded_accuracies
        test_losses = expanded_test_losses if expanded_test_losses else None
        test_accuracies = expanded_test_accuracies if expanded_test_accuracies else None
    else:
        steps = range(len(losses))
    
    # Plot losses
    ax1.plot(steps, losses, 'b-', label='Training Loss', alpha=0.8, linewidth=1.5)
    
    if test_losses is not None:
        ax1.plot(steps[:len(test_losses)], test_losses, 'r--', 
                label='Test Loss', alpha=0.8, linewidth=1.5)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(steps, accuracies, 'b-', label='Training Accuracy', alpha=0.8, linewidth=1.5)
    
    if test_accuracies is not None:
        ax2.plot(steps[:len(test_accuracies)], test_accuracies, 'r--', 
                label='Test Accuracy', alpha=0.8, linewidth=1.5)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add period markers if specified
    if period_length is not None and metric_steps is not None:
        max_step = max(metric_steps)
        for ax in [ax1, ax2]:
            for i in range(1, max_step // period_length + 1):
                ax.axvline(x=i * period_length, color='gray', linestyle=':', alpha=0.5)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_class_focus_dynamics(
    steps: jnp.ndarray,
    class_weights_fn: Any,
    w_max_values: List[float],
    T: int,
    num_classes: int = 3,
    colors: List[str] = None,
    title: str = "Dynamic Class Focus",
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Plot the dynamic class focus over training steps.
    
    Args:
        steps: Array of training steps
        class_weights_fn: Function to compute class weights
        w_max_values: List of maximum weight values to plot
        T: Period length
        num_classes: Number of classes
        colors: Colors for each class
        title: Plot title
        figsize: Figure size
    """
    if colors is None:
        colors = ['red', 'blue', 'gold']
    
    plt.figure(figsize=figsize)
    
    for w_max in w_max_values:
        style = 'solid' if w_max == max(w_max_values) else 'dashed'
        
        for class_idx in range(num_classes):
            weights = []
            for t in steps:
                class_focus = int((t // T) % num_classes)
                current_weights = class_weights_fn(t % T, class_focus, w_max, T)
                weights.append(current_weights[class_idx])
            
            label = f'Class {class_idx + 1}, w_max={w_max}'
            plt.plot(steps, weights, color=colors[class_idx], 
                    linestyle=style, linewidth=2, label=label, alpha=0.8)
    
    plt.title(title, fontsize=18)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Class Weight Proportion', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_weight_distances(
    distances: List[float],
    labels: List[str],
    title: str = "Weight Distances Over Time",
    colors: List[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot weight distances over training time.
    
    Args:
        distances: List of distance arrays
        labels: Labels for each distance type
        title: Plot title
        colors: Colors for each line
        figsize: Figure size
    """
    if colors is None:
        colors = plt.cm.Set1(np.linspace(0, 1, len(distances)))
    
    plt.figure(figsize=figsize)
    
    for i, (dist, label, color) in enumerate(zip(distances, labels, colors)):
        steps = range(len(dist))
        plt.plot(steps, dist, color=color, linewidth=2, label=label, alpha=0.8)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('L2 Distance', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_animation_from_frames(
    frame_files: List[Path],
    output_path: Path,
    fps: int = 2,
    cleanup: bool = True
) -> None:
    """
    Create an MP4 animation from a list of frame files.
    
    Args:
        frame_files: List of image file paths
        output_path: Output video file path
        fps: Frames per second
        cleanup: Whether to delete frame files after creating video
    """
    import imageio
    
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame_file in frame_files:
            if frame_file.exists():
                image = imageio.imread(frame_file)
                writer.append_data(image)
    
    if cleanup:
        for frame_file in frame_files:
            if frame_file.exists():
                frame_file.unlink()
    
    print(f"Animation saved to {output_path}")


def setup_matplotlib_style(style: str = "seaborn-v0_8") -> None:
    """
    Set up matplotlib style for consistent plotting.
    
    Args:
        style: Matplotlib style to use
    """
    try:
        plt.style.use(style)
    except OSError:
        # Fallback if style is not available
        plt.style.use('default')
        sns.set_style("whitegrid")
    
    # Set default parameters
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10