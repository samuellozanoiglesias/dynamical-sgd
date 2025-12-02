#!/usr/bin/env python3
"""
Weight correlation and trajectory analysis for dynamical SGD experiments.

This module provides advanced analysis tools for tracking weight evolution,
computing correlation matrices, and visualizing parameter trajectories during
dynamical SGD training with periodic class focus.

Usage:
    python analysis/correlation_analysis.py --weights_file weights_history.pkl
    
    # Or import specific functions
    from analysis.correlation_analysis import compute_weight_correlation_matrix, visualize_weight_evolution
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import argparse
from dataclasses import dataclass

# Set style
sns.set_style("whitegrid")


@dataclass
class WeightAnalysisConfig:
    """Configuration for weight correlation analysis."""
    correlation_window: int = 1000  # Number of steps for correlation computation
    trajectory_subsample: int = 10  # Subsample rate for trajectory plotting
    plot_top_correlations: int = 20  # Number of top correlations to highlight
    save_plots: bool = True
    plot_format: str = 'png'
    dpi: int = 150


def flatten_params(params: Any, return_shapes: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, List]]:
    """
    Flatten nested parameter structure into 1D array.
    
    Args:
        params: JAX parameter pytree
        return_shapes: Whether to return shape information for reconstruction
        
    Returns:
        Flattened parameters, optionally with shape information
    """
    flat_params = []
    shapes = []
    
    def process_layer(layer_params, layer_name=""):
        if isinstance(layer_params, (tuple, list)):
            for i, param in enumerate(layer_params):
                param_flat = param.flatten()
                flat_params.append(param_flat)
                if return_shapes:
                    shapes.append({
                        'shape': param.shape,
                        'size': param.size,
                        'layer': layer_name,
                        'sublayer': i,
                        'type': 'weights' if i == 0 else 'biases'
                    })
        else:
            param_flat = layer_params.flatten()
            flat_params.append(param_flat)
            if return_shapes:
                shapes.append({
                    'shape': layer_params.shape,
                    'size': layer_params.size,
                    'layer': layer_name,
                    'sublayer': 0,
                    'type': 'weights'
                })
    
    # Process each layer
    for i, layer in enumerate(params):
        process_layer(layer, f"layer_{i}")
    
    # Concatenate all flattened parameters
    all_flat = jnp.concatenate(flat_params)
    
    if return_shapes:
        return all_flat, shapes
    else:
        return all_flat


def unflatten_params(flat_params: jnp.ndarray, shapes: List[Dict]) -> Any:
    """
    Reconstruct parameter structure from flattened array.
    
    Args:
        flat_params: Flattened parameter array
        shapes: Shape information from flatten_params
        
    Returns:
        Reconstructed parameter pytree
    """
    params = []
    start_idx = 0
    
    current_layer = -1
    layer_params = []
    
    for shape_info in shapes:
        layer_idx = int(shape_info['layer'].split('_')[1])
        
        # If we've moved to a new layer, save the previous one
        if layer_idx != current_layer and current_layer >= 0:
            if len(layer_params) == 1:
                params.append(layer_params[0])
            else:
                params.append(tuple(layer_params))
            layer_params = []
        
        current_layer = layer_idx
        
        # Extract parameter chunk and reshape
        size = shape_info['size']
        param_chunk = flat_params[start_idx:start_idx + size]
        param_reshaped = param_chunk.reshape(shape_info['shape'])
        layer_params.append(param_reshaped)
        start_idx += size
    
    # Don't forget the last layer
    if len(layer_params) == 1:
        params.append(layer_params[0])
    else:
        params.append(tuple(layer_params))
    
    return tuple(params)


def compute_weight_correlation_matrix(weight_history: List[Any], 
                                    window_size: Optional[int] = None,
                                    subsample_rate: int = 1) -> Tuple[jnp.ndarray, List[Dict]]:
    """
    Compute correlation matrix for weight evolution over time.
    
    Args:
        weight_history: List of parameter states over training
        window_size: Size of sliding window (None for full history)
        subsample_rate: Rate at which to subsample history
        
    Returns:
        Tuple of (correlation_matrix, parameter_shapes)
    """
    # Subsample the history if requested
    if subsample_rate > 1:
        weight_history = weight_history[::subsample_rate]
    
    # Use sliding window if specified
    if window_size and len(weight_history) > window_size:
        weight_history = weight_history[-window_size:]
    
    # Flatten all parameter states
    flat_weights = []
    shapes = None
    
    for params in weight_history:
        if shapes is None:
            flat_param, shapes = flatten_params(params, return_shapes=True)
        else:
            flat_param = flatten_params(params, return_shapes=False)
        flat_weights.append(flat_param)
    
    # Convert to matrix: rows are time steps, columns are parameters
    weight_matrix = jnp.stack(flat_weights)
    
    # Compute correlation matrix
    correlation_matrix = jnp.corrcoef(weight_matrix.T)
    
    return correlation_matrix, shapes


def visualize_correlation_matrix(correlation_matrix: jnp.ndarray,
                               shapes: List[Dict],
                               step: int,
                               config: WeightAnalysisConfig,
                               output_dir: Optional[Path] = None) -> None:
    """
    Visualize parameter correlation matrix with layer annotations.
    
    Args:
        correlation_matrix: Parameter correlation matrix
        shapes: Parameter shape information
        step: Current training step
        config: Analysis configuration
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(15, 12))
    
    # Create correlation plot
    mask = jnp.triu(jnp.ones_like(correlation_matrix, dtype=bool), k=1)
    correlation_masked = jnp.where(mask, jnp.nan, correlation_matrix)
    
    im = plt.imshow(correlation_masked, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, label='Correlation Coefficient', shrink=0.8)
    
    # Add layer boundaries and labels
    tick_positions = []
    tick_labels = []
    nested_labels = []
    position = 0
    
    for shape_info in shapes:
        layer_name = shape_info['layer']
        param_type = shape_info['type']
        size = shape_info['size']
        
        center_pos = position + size // 2
        
        # Create labels
        label = f"{layer_name}\\n{param_type}"
        nested_labels.append((label, center_pos, size))
        
        # Add tick positions (every 10th parameter for readability)
        layer_ticks = list(range(position, position + size, max(1, size // 10)))
        tick_positions.extend(layer_ticks)
        tick_labels.extend([f'{i-position}' for i in layer_ticks])
        
        position += size
    
    # Set ticks and labels
    plt.xticks(tick_positions[::5], [tick_labels[i] for i in range(0, len(tick_labels), 5)], 
               rotation=90, fontsize=8)
    plt.yticks(tick_positions[::5], [tick_labels[i] for i in range(0, len(tick_labels), 5)], 
               fontsize=8)
    
    # Add layer boundary lines
    position = 0
    for shape_info in shapes[:-1]:  # Don't draw line after last layer
        position += shape_info['size']
        plt.axhline(y=position-0.5, color='white', linewidth=2, alpha=0.7)
        plt.axvline(x=position-0.5, color='white', linewidth=2, alpha=0.7)
    
    # Add layer labels as text
    position = 0
    for label, center, size in nested_labels:
        plt.text(center, -len(shapes)*0.02, label, 
                ha='center', va='top', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        plt.text(-len(shapes)*0.02, center, label, 
                ha='right', va='center', fontsize=10, rotation=90,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.title(f'Parameter Correlation Matrix - Step {step}', fontsize=16, pad=20)
    plt.xlabel('Parameter Index', fontsize=12)
    plt.ylabel('Parameter Index', fontsize=12)
    
    # Highlight top correlations
    if config.plot_top_correlations > 0:
        # Find top correlations (excluding diagonal)
        correlation_no_diag = correlation_matrix.copy()
        correlation_no_diag = correlation_no_diag.at[jnp.diag_indices_from(correlation_no_diag)].set(0)
        
        # Get top absolute correlations
        abs_corr = jnp.abs(correlation_no_diag)
        top_indices = jnp.unravel_index(jnp.argsort(abs_corr.ravel())[-config.plot_top_correlations:], 
                                       abs_corr.shape)
        
        # Plot top correlations as circles
        for i, j in zip(top_indices[0], top_indices[1]):
            if i != j:  # Skip diagonal
                corr_val = correlation_matrix[i, j]
                color = 'red' if corr_val > 0 else 'blue'
                alpha = min(abs(corr_val), 0.8)
                circle = plt.Circle((j, i), radius=2, color=color, alpha=alpha, fill=False, linewidth=2)
                plt.gca().add_patch(circle)
    
    plt.tight_layout()
    
    if config.save_plots and output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"correlation_matrix_step_{step}.{config.plot_format}"
        plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
        print(f"Saved correlation matrix: {filename}")
    
    plt.show()


def analyze_parameter_trajectories(weight_history: List[Any],
                                 shapes: List[Dict],
                                 config: WeightAnalysisConfig,
                                 output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Analyze parameter trajectories and compute trajectory statistics.
    
    Args:
        weight_history: List of parameter states over training
        shapes: Parameter shape information  
        config: Analysis configuration
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with trajectory analysis results
    """
    # Flatten all parameter states
    flat_trajectories = []
    for params in weight_history:
        flat_param = flatten_params(params, return_shapes=False)
        flat_trajectories.append(flat_param)
    
    trajectories = jnp.stack(flat_trajectories)  # Shape: (time_steps, num_params)
    
    # Compute trajectory statistics
    trajectory_stats = {
        'mean_displacement': jnp.mean(jnp.abs(jnp.diff(trajectories, axis=0)), axis=0),
        'total_displacement': jnp.sum(jnp.abs(jnp.diff(trajectories, axis=0)), axis=0),
        'variance': jnp.var(trajectories, axis=0),
        'final_values': trajectories[-1],
        'initial_values': trajectories[0],
        'net_change': trajectories[-1] - trajectories[0]
    }
    
    # Plot trajectory statistics by layer
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    position = 0
    layer_stats = {}
    
    for shape_info in shapes:
        layer_name = shape_info['layer']
        param_type = shape_info['type']
        size = shape_info['size']
        
        # Extract layer statistics
        layer_slice = slice(position, position + size)
        layer_key = f"{layer_name}_{param_type}"
        
        layer_stats[layer_key] = {
            'mean_displacement': trajectory_stats['mean_displacement'][layer_slice],
            'variance': trajectory_stats['variance'][layer_slice],
            'net_change': trajectory_stats['net_change'][layer_slice],
            'positions': list(range(position, position + size))
        }
        
        position += size
    
    # Plot 1: Mean displacement per layer
    axes[0].set_title('Mean Parameter Displacement by Layer')
    for layer_key, stats in layer_stats.items():
        axes[0].plot(stats['positions'], stats['mean_displacement'], 
                    label=layer_key, marker='o', alpha=0.7)
    axes[0].set_xlabel('Parameter Index')
    axes[0].set_ylabel('Mean Displacement')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Parameter variance by layer  
    axes[1].set_title('Parameter Variance by Layer')
    for layer_key, stats in layer_stats.items():
        axes[1].plot(stats['positions'], stats['variance'], 
                    label=layer_key, marker='s', alpha=0.7)
    axes[1].set_xlabel('Parameter Index')
    axes[1].set_ylabel('Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Net parameter change by layer
    axes[2].set_title('Net Parameter Change by Layer')
    for layer_key, stats in layer_stats.items():
        axes[2].plot(stats['positions'], stats['net_change'], 
                    label=layer_key, marker='^', alpha=0.7)
    axes[2].set_xlabel('Parameter Index')
    axes[2].set_ylabel('Net Change')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Trajectory evolution for sample parameters
    axes[3].set_title('Sample Parameter Trajectories')
    num_sample_params = min(5, trajectories.shape[1])
    sample_indices = jnp.linspace(0, trajectories.shape[1]-1, num_sample_params, dtype=int)
    
    for idx in sample_indices:
        axes[3].plot(trajectories[:, idx], label=f'Param {idx}', alpha=0.8)
    axes[3].set_xlabel('Training Step')
    axes[3].set_ylabel('Parameter Value')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if config.save_plots and output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"parameter_trajectories.{config.plot_format}"
        plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
        print(f"Saved trajectory analysis: {filename}")
    
    plt.show()
    
    return {
        'trajectory_stats': trajectory_stats,
        'layer_stats': layer_stats,
        'trajectories': trajectories
    }


def compute_layer_correlation_evolution(weight_history: List[Any],
                                      window_size: int = 100,
                                      step_size: int = 10) -> Dict[str, List[float]]:
    """
    Compute evolution of correlations between different layers over training.
    
    Args:
        weight_history: List of parameter states over training
        window_size: Size of sliding window for correlation computation
        step_size: Step size for sliding window
        
    Returns:
        Dictionary with correlation evolution for each layer pair
    """
    correlations = {}
    
    # Process sliding windows
    for start_idx in range(0, len(weight_history) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_weights = weight_history[start_idx:end_idx]
        
        # Flatten each layer separately
        layer_trajectories = {}
        
        for params in window_weights:
            for layer_idx, layer in enumerate(params):
                layer_key = f"layer_{layer_idx}"
                
                if isinstance(layer, (tuple, list)):
                    # Concatenate weights and biases
                    layer_flat = jnp.concatenate([param.flatten() for param in layer])
                else:
                    layer_flat = layer.flatten()
                
                if layer_key not in layer_trajectories:
                    layer_trajectories[layer_key] = []
                layer_trajectories[layer_key].append(layer_flat)
        
        # Compute correlations between layers
        layer_keys = list(layer_trajectories.keys())
        for i, layer1 in enumerate(layer_keys):
            for j, layer2 in enumerate(layer_keys):
                if i < j:  # Avoid duplicates
                    corr_key = f"{layer1}_vs_{layer2}"
                    
                    # Compute average correlation between layer parameters
                    layer1_matrix = jnp.stack(layer_trajectories[layer1])
                    layer2_matrix = jnp.stack(layer_trajectories[layer2])
                    
                    # Compute cross-correlation and take mean
                    cross_corr = jnp.corrcoef(layer1_matrix.mean(axis=1), layer2_matrix.mean(axis=1))[0, 1]
                    
                    if corr_key not in correlations:
                        correlations[corr_key] = []
                    correlations[corr_key].append(float(cross_corr))
    
    return correlations


def plot_layer_correlation_evolution(correlations: Dict[str, List[float]],
                                   config: WeightAnalysisConfig,
                                   output_dir: Optional[Path] = None) -> None:
    """
    Plot evolution of inter-layer correlations over training.
    
    Args:
        correlations: Dictionary with correlation evolution for each layer pair
        config: Analysis configuration
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for idx, (pair_name, corr_values) in enumerate(correlations.items()):
        color = colors[idx % len(colors)]
        plt.plot(corr_values, label=pair_name, color=color, linewidth=2.5, alpha=0.8)
    
    plt.title('Evolution of Inter-Layer Correlations', fontsize=16)
    plt.xlabel('Training Window', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if config.save_plots and output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"layer_correlation_evolution.{config.plot_format}"
        plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
        print(f"Saved layer correlation evolution: {filename}")
    
    plt.show()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Weight correlation and trajectory analysis')
    parser.add_argument('--weights_file', type=str, required=True,
                       help='Path to pickled weights history file')
    parser.add_argument('--output_dir', type=str, default='correlation_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--correlation_window', type=int, default=1000,
                       help='Window size for correlation computation')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save plots to files')
    
    args = parser.parse_args()
    
    weights_file = Path(args.weights_file)
    output_dir = Path(args.output_dir)
    
    if not weights_file.exists():
        print(f"Error: Weights file {weights_file} does not exist")
        return
    
    # Load weights history
    print(f"Loading weights from {weights_file}...")
    with open(weights_file, 'rb') as f:
        weight_history = pickle.load(f)
    
    print(f"Loaded {len(weight_history)} weight states")
    
    # Create configuration
    config = WeightAnalysisConfig(
        correlation_window=args.correlation_window,
        save_plots=not args.no_save
    )
    
    # Perform analysis
    print("Computing correlation matrix...")
    correlation_matrix, shapes = compute_weight_correlation_matrix(
        weight_history, window_size=config.correlation_window
    )
    
    print("Visualizing correlation matrix...")
    visualize_correlation_matrix(correlation_matrix, shapes, 
                               len(weight_history), config, output_dir)
    
    print("Analyzing parameter trajectories...")
    trajectory_results = analyze_parameter_trajectories(
        weight_history, shapes, config, output_dir
    )
    
    print("Computing layer correlation evolution...")
    layer_correlations = compute_layer_correlation_evolution(weight_history)
    
    print("Plotting layer correlation evolution...")
    plot_layer_correlation_evolution(layer_correlations, config, output_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()