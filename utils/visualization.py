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


def plot_training_curves_with_classes(
    losses: List[float],
    accuracies: List[float],
    test_losses: Optional[List[float]] = None,
    test_accuracies: Optional[List[float]] = None,
    train_losses_per_class: Optional[List[List[float]]] = None,
    train_accuracies_per_class: Optional[List[List[float]]] = None,
    test_losses_per_class: Optional[List[List[float]]] = None,
    test_accuracies_per_class: Optional[List[List[float]]] = None,
    metric_steps: Optional[List[int]] = None,
    period_length: Optional[int] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 20),
    step_100_acc: Optional[int] = None,
    tpt_threshold: float = 1.0,
    num_classes: Optional[int] = None
) -> None:
    """
    Plot training and validation curves with per-class breakdown in 4x2 layout.
    
    Args:
        losses: Training losses
        accuracies: Training accuracies
        test_losses: Optional test losses
        test_accuracies: Optional test accuracies
        train_losses_per_class: Training losses per class [num_steps x num_classes]
        train_accuracies_per_class: Training accuracies per class [num_steps x num_classes]
        test_losses_per_class: Test losses per class [num_steps x num_classes]
        test_accuracies_per_class: Test accuracies per class [num_steps x num_classes]
        metric_steps: Actual step numbers corresponding to metrics (if None, uses indices)
        period_length: If provided, add vertical lines at period boundaries
        title: Plot title
        figsize: Figure size
        step_100_acc: If provided, add vertical line at step where TPT accuracy threshold was reached
        tpt_threshold: TPT accuracy threshold value (default 1.0 for 100%)
        num_classes: Number of classes
    """
    # Determine number of classes if not provided
    if num_classes is None:
        if train_losses_per_class is not None:
            num_classes = len(train_losses_per_class)
        elif train_accuracies_per_class is not None:
            num_classes = len(train_accuracies_per_class)
        else:
            # Default fallback
            num_classes = 3
    
    # Create subplot layout: (num_classes + 1) rows x 2 columns
    # Row 0: Total metrics, Rows 1+: Per-class metrics
    fig, axes = plt.subplots(num_classes + 1, 2, figsize=(15, 5 * (num_classes + 1)))
    
    # Expand metrics to fill all intermediate steps (step function)
    if metric_steps is not None and len(metric_steps) > 0:
        # Create expanded arrays where metrics are held constant between validation intervals
        max_step = max(metric_steps)
        expanded_steps = list(range(max_step + 1))
        expanded_losses = []
        expanded_accuracies = []
        expanded_test_losses = []
        expanded_test_accuracies = []
        if num_classes is not None:
            expanded_train_losses_per_class = [[] for _ in range(num_classes)]
            expanded_train_accuracies_per_class = [[] for _ in range(num_classes)]
            expanded_test_losses_per_class = [[] for _ in range(num_classes)]
            expanded_test_accuracies_per_class = [[] for _ in range(num_classes)]
        else:
            expanded_train_losses_per_class = []
            expanded_train_accuracies_per_class = []
            expanded_test_losses_per_class = []
            expanded_test_accuracies_per_class = []
        
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
            
            # Expand per-class data
            # train_losses_per_class is [num_classes][num_metric_steps]
            if train_losses_per_class is not None and num_classes is not None:
                for c in range(num_classes):
                    if idx < len(train_losses_per_class[c]):
                        expanded_train_losses_per_class[c].append(train_losses_per_class[c][idx])
            if train_accuracies_per_class is not None and num_classes is not None:
                for c in range(num_classes):
                    if idx < len(train_accuracies_per_class[c]):
                        expanded_train_accuracies_per_class[c].append(train_accuracies_per_class[c][idx])
            if test_losses_per_class is not None and num_classes is not None:
                for c in range(num_classes):
                    if idx < len(test_losses_per_class[c]):
                        expanded_test_losses_per_class[c].append(test_losses_per_class[c][idx])
            if test_accuracies_per_class is not None and num_classes is not None:
                for c in range(num_classes):
                    if idx < len(test_accuracies_per_class[c]):
                        expanded_test_accuracies_per_class[c].append(test_accuracies_per_class[c][idx])
        
        steps = expanded_steps
        losses = expanded_losses
        accuracies = expanded_accuracies
        test_losses = expanded_test_losses if expanded_test_losses else None
        test_accuracies = expanded_test_accuracies if expanded_test_accuracies else None
        train_losses_per_class = expanded_train_losses_per_class if train_losses_per_class is not None else None
        train_accuracies_per_class = expanded_train_accuracies_per_class if train_accuracies_per_class is not None else None
        test_losses_per_class = expanded_test_losses_per_class if test_losses_per_class is not None else None
        test_accuracies_per_class = expanded_test_accuracies_per_class if test_accuracies_per_class is not None else None
    else:
        # When metric_steps is None, use indices as steps
        steps = list(range(len(losses)))
        # Per-class metrics need to match the number of steps
        # They should already be in the correct format [num_classes][num_steps]
    
    # Row 1: Total metrics
    ax_loss = axes[0, 0]
    ax_acc = axes[0, 1]
    
    # Plot total losses
    ax_loss.plot(steps, losses, 'b-', label='Training Loss', alpha=0.8, linewidth=1.5)
    if test_losses is not None:
        ax_loss.plot(steps[:len(test_losses)], test_losses, 'r--', 
                    label='Test Loss', alpha=0.8, linewidth=1.5)
    
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Total Loss')
    ax_loss.set_yscale('log')
    
    # Set y-axis limits for loss
    all_losses = list(losses)
    if test_losses is not None:
        all_losses.extend(test_losses)
    loss_min = max(min(all_losses), 1e-6)  # Avoid log(0)
    loss_max = max(all_losses) * 1.1  # Add 10% padding
    ax_loss.set_ylim(bottom=loss_min, top=loss_max)
    
    # Add horizontal dashed lines
    loss_horizontal_lines = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for loss_val in loss_horizontal_lines:
        if loss_min <= loss_val <= loss_max:
            ax_loss.axhline(y=loss_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
    
    ax_loss.legend()
    
    # Plot total accuracies
    ax_acc.plot(steps, accuracies, 'b-', label='Training Accuracy', alpha=0.8, linewidth=1.5)
    if test_accuracies is not None:
        ax_acc.plot(steps[:len(test_accuracies)], test_accuracies, 'r--', 
                   label='Test Accuracy', alpha=0.8, linewidth=1.5)
    
    ax_acc.set_xlabel('Step')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Total Accuracy')
    ax_acc.set_ylim(-0.01, 1.01)  # Set y-axis range from -0.01 to 1.01
    
    # Add horizontal dashed lines for total accuracy
    accuracy_horizontal_lines = [0.2, 0.4, 0.6, 0.8, 1.0]
    for acc_val in accuracy_horizontal_lines:
        ax_acc.axhline(y=acc_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
    
    ax_acc.legend()
    
    # Rows 1+: Per-class metrics
    for class_idx in range(num_classes):
        row_idx = class_idx + 1
        ax_loss_class = axes[row_idx, 0]
        ax_acc_class = axes[row_idx, 1]
        
        # Plot per-class losses
        if train_losses_per_class is not None:
            ax_loss_class.plot(steps, train_losses_per_class[class_idx], 'b-', 
                              label=f'Training Loss Class {class_idx}', alpha=0.8, linewidth=1.5)
        if test_losses_per_class is not None:
            ax_loss_class.plot(steps, test_losses_per_class[class_idx], 'r--', 
                              label=f'Test Loss Class {class_idx}', alpha=0.8, linewidth=1.5)
        
        ax_loss_class.set_xlabel('Step')
        ax_loss_class.set_ylabel('Loss')
        ax_loss_class.set_title(f'Class {class_idx} Loss')
        ax_loss_class.set_yscale('log')
        
        # Plot per-class accuracies
        if train_accuracies_per_class is not None:
            ax_acc_class.plot(steps, train_accuracies_per_class[class_idx], 'b-', 
                             label=f'Training Accuracy Class {class_idx}', alpha=0.8, linewidth=1.5)
        if test_accuracies_per_class is not None:
            ax_acc_class.plot(steps, test_accuracies_per_class[class_idx], 'r--', 
                             label=f'Test Accuracy Class {class_idx}', alpha=0.8, linewidth=1.5)
        
        ax_acc_class.set_xlabel('Step')
        ax_acc_class.set_ylabel('Accuracy')
        ax_acc_class.set_title(f'Class {class_idx} Accuracy')
        ax_acc_class.set_ylim(-0.01, 1.01)  # Set y-axis range from -0.01 to 1.01
        
        # Add horizontal dashed lines for class accuracy
        for acc_val in accuracy_horizontal_lines:
            ax_acc_class.axhline(y=acc_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
        
        ax_loss_class.legend()
        ax_acc_class.legend()
    
    # Add WIDE vertical BLACK line at TPT accuracy threshold step (Terminal Phase Training)
    if step_100_acc is not None:
        threshold_label = f'TPT Threshold ({tpt_threshold*100:.0f}% Train Acc)'
        for i in range(num_classes + 1):
            for j in range(2):
                ax = axes[i, j]
                # WIDE BLACK LINE - linewidth=4 to make it very visible
                ax.axvline(x=step_100_acc, color='black', linestyle='-', alpha=0.9, linewidth=4, 
                          label=threshold_label, zorder=10)
                # Update legend to include the TPT marker
                ax.legend(loc='best')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    # DO NOT CALL plt.show() - it prevents saving!


def plot_training_curves(
    losses: List[float],
    accuracies: List[float],
    test_losses: Optional[List[float]] = None,
    test_accuracies: Optional[List[float]] = None,
    metric_steps: Optional[List[int]] = None,
    period_length: Optional[int] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 5),
    step_100_acc: Optional[int] = None,
    tpt_threshold: float = 1.0
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
        step_100_acc: If provided, add vertical line at step where TPT accuracy threshold was reached
        tpt_threshold: TPT accuracy threshold value (default 1.0 for 100%)
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
    
    # Set y-axis limits to include ALL values (min to max of actual data)
    all_losses = list(losses)
    if test_losses is not None:
        all_losses.extend(test_losses)
    loss_min = max(min(all_losses), 1e-6)  # Avoid log(0)
    loss_max = max(all_losses) * 1.1  # Add 10% padding
    # Set bottom limit a bit lower than minimum to make horizontal lines visible
    loss_bottom = loss_min * 0.5  # 50% lower than minimum
    ax1.set_ylim(bottom=loss_bottom, top=loss_max)
    
    # Add horizontal dashed lines at specific loss values (soft/subtle)
    loss_horizontal_lines = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for loss_val in loss_horizontal_lines:
        if loss_min <= loss_val <= loss_max:
            ax1.axhline(y=loss_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
    
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(steps, accuracies, 'b-', label='Training Accuracy', alpha=0.8, linewidth=1.5)
    
    if test_accuracies is not None:
        ax2.plot(steps[:len(test_accuracies)], test_accuracies, 'r--', 
                label='Test Accuracy', alpha=0.8, linewidth=1.5)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_ylim(-0.01, 1.01)  # Set y-axis range from -0.01 to 1.01
    
    # Add horizontal dashed lines at specific accuracy values (soft/subtle)
    accuracy_horizontal_lines = [0.2, 0.4, 0.6, 0.8, 1.0]
    for acc_val in accuracy_horizontal_lines:
        ax2.axhline(y=acc_val, color='gray', linestyle='--', alpha=0.2, linewidth=0.8)
    
    ax2.legend()
    
    # Add WIDE vertical BLACK line at TPT accuracy threshold step (Terminal Phase Training)
    if step_100_acc is not None:
        threshold_label = f'TPT Threshold ({tpt_threshold*100:.0f}% Train Acc)'
        for ax in [ax1, ax2]:
            # WIDE BLACK LINE - linewidth=4 to make it very visible
            ax.axvline(x=step_100_acc, color='black', linestyle='-', alpha=0.9, linewidth=4, 
                      label=threshold_label, zorder=10)
        # Update legend to include the TPT marker
        ax1.legend(loc='best')
        ax2.legend(loc='best')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    # DO NOT CALL plt.show() - it prevents saving!


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