#!/usr/bin/env python3
"""
Advanced gradient and distribution analysis for dynamical SGD experiments.

This module extracts the sophisticated gradient distribution analysis methods
from the original research code, providing tools for KL divergence computation,
disjoint distribution analysis, and advanced visualization of neural network
dynamics during training.

Usage:
    python analysis/advanced_gradient_analysis.py --experiment_dir outputs/experiment_001
    
    # Or import specific analysis functions
    from analysis.advanced_gradient_analysis import compute_kl_divergences, plot_disjoint_distributions
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import entropy, gaussian_kde
from scipy.interpolate import interp1d
import pickle
import argparse

# Set style
sns.set_style("darkgrid")


def compute_kl_divergence(current_distribution: jnp.ndarray, 
                         previous_distribution: jnp.ndarray,
                         epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        current_distribution: Current probability distribution
        previous_distribution: Previous probability distribution  
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence value
    """
    # Normalize to ensure they are proper probability distributions
    current_norm = current_distribution / jnp.sum(current_distribution)
    previous_norm = previous_distribution / jnp.sum(previous_distribution)
    
    # Add epsilon to avoid log(0)
    current_norm = current_norm + epsilon
    previous_norm = previous_norm + epsilon
    
    # Compute KL divergence
    kl_div = jnp.sum(current_norm * jnp.log(current_norm / previous_norm))
    return float(kl_div)


def compute_kl_divergence_modified(current_distribution: jnp.ndarray,
                                  previous_distribution: jnp.ndarray,
                                  epsilon: float = 1e-10) -> float:
    """
    Modified KL divergence computation with additional normalization.
    
    Args:
        current_distribution: Current probability distribution
        previous_distribution: Previous probability distribution
        epsilon: Small value to avoid numerical issues
        
    Returns:
        Modified KL divergence value
    """
    # Flatten and normalize
    current_flat = current_distribution.flatten()
    previous_flat = previous_distribution.flatten()
    
    # Apply softmax normalization
    current_soft = jax.nn.softmax(current_flat)
    previous_soft = jax.nn.softmax(previous_flat)
    
    # Add epsilon for numerical stability
    current_soft = current_soft + epsilon
    previous_soft = previous_soft + epsilon
    
    # Compute symmetric KL divergence
    kl_forward = jnp.sum(current_soft * jnp.log(current_soft / previous_soft))
    kl_backward = jnp.sum(previous_soft * jnp.log(previous_soft / current_soft))
    
    return float((kl_forward + kl_backward) / 2)


def compute_shannon_entropy(distribution: jnp.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    Args:
        distribution: Probability distribution
        
    Returns:
        Shannon entropy value
    """
    # Normalize to proper probability distribution
    dist_norm = distribution / jnp.sum(distribution)
    # Remove zeros to avoid log(0)
    dist_nonzero = dist_norm[dist_norm > 0]
    return float(-jnp.sum(dist_nonzero * jnp.log(dist_nonzero)))


def compute_mean_gradients(initial_grads: List[jnp.ndarray], 
                          final_grads: List[jnp.ndarray],
                          current_period: int,
                          period_length: int,
                          use_kl: bool = False) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Compute mean gradients for different phases of training.
    
    Args:
        initial_grads: Gradients from initial phase
        final_grads: Gradients from final phase  
        current_period: Current training period
        period_length: Length of each period
        use_kl: Whether to apply KL-based normalization
        
    Returns:
        Tuple of (mean_initial_grads, mean_final_grads)
    """
    def normalize_gradients(grads_list):
        """Normalize gradients using L2 norm."""
        normalized = []
        for grads in grads_list:
            if isinstance(grads, (list, tuple)):
                # Handle nested structure (multiple layers)
                layer_norms = []
                for layer_grad in grads:
                    norm = jnp.linalg.norm(layer_grad.flatten())
                    if norm > 0:
                        layer_norms.append(layer_grad / norm)
                    else:
                        layer_norms.append(layer_grad)
                normalized.append(layer_norms)
            else:
                # Handle single array
                norm = jnp.linalg.norm(grads.flatten())
                if norm > 0:
                    normalized.append(grads / norm)
                else:
                    normalized.append(grads)
        return normalized
    
    # Normalize all gradients
    initial_normalized = normalize_gradients(initial_grads)
    final_normalized = normalize_gradients(final_grads)
    
    # Compute means
    def compute_mean(grad_list):
        if not grad_list:
            return []
        
        if isinstance(grad_list[0], (list, tuple)):
            # Handle multiple layers
            num_layers = len(grad_list[0])
            mean_layers = []
            for layer_idx in range(num_layers):
                layer_grads = [grads[layer_idx] for grads in grad_list]
                mean_layers.append(jnp.mean(jnp.stack(layer_grads), axis=0))
            return mean_layers
        else:
            # Handle single layer
            return jnp.mean(jnp.stack(grad_list), axis=0)
    
    mean_initial = compute_mean(initial_normalized)
    mean_final = compute_mean(final_normalized)
    
    return mean_initial, mean_final


def plot_distributions(distributions: List[List[jnp.ndarray]], 
                      title: str, 
                      current_period: int,
                      output_dir: Optional[Path] = None,
                      save_plot: bool = False) -> None:
    """
    Plot gradient distributions across layers and periods.
    
    Args:
        distributions: List of distributions per period and layer
        title: Plot title
        current_period: Current training period
        output_dir: Directory to save plots
        save_plot: Whether to save the plot
    """
    sns.set_context("talk")
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    
    fig, axs = plt.subplots(4, 1, figsize=(25, 20))
    axs = axs.flatten()
    
    colors = ['red', 'blue', 'yellow']
    markers = ['s', '^', 'o']
    
    for layer_idx in range(4):
        indices = np.arange(len(distributions[0][layer_idx].flatten()))
        
        for period_idx, distribution in enumerate(distributions):
            dist_flat = distribution[layer_idx].flatten()
            
            # Interpolate for smooth plotting
            if len(dist_flat) >= 4:
                f_interp = interp1d(indices, dist_flat, kind='cubic')
            else:
                f_interp = interp1d(indices, dist_flat, kind='linear')
            
            indices_new = np.linspace(indices.min(), indices.max(), num=500)
            dist_interp = f_interp(indices_new)
            
            # Plot with styling
            color = colors[period_idx % len(colors)]
            marker = markers[period_idx % len(markers)]
            
            axs[layer_idx].plot(indices_new, dist_interp, 
                               color=color, marker=marker, markersize=8,
                               linewidth=2.5, alpha=0.8, 
                               label=f'Period {period_idx + 1}')
            
            axs[layer_idx].fill_between(indices_new, dist_interp, 
                                       alpha=0.3, color=color)
        
        axs[layer_idx].set_title(f'{title} - Layer {layer_idx + 1}', fontsize=18)
        axs[layer_idx].set_xlabel('Parameter Index', fontsize=14)
        axs[layer_idx].set_ylabel('Normalized Gradient Magnitude', fontsize=14)
        axs[layer_idx].legend(fontsize=12)
        axs[layer_idx].grid(True, alpha=0.7)
    
    plt.suptitle(f'{title} - Period {current_period}', fontsize=20)
    plt.tight_layout()
    
    if save_plot and output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"{title.lower().replace(' ', '_')}_period_{current_period}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    
    plt.show()


def plot_disjoint_distributions(part1_distributions: List[jnp.ndarray],
                               part2_distributions: List[jnp.ndarray], 
                               current_period: int,
                               class_color: str,
                               output_dir: Optional[Path] = None,
                               save_plot: bool = False) -> None:
    """
    Plot disjoint gradient distributions for different training phases.
    
    Args:
        part1_distributions: Distributions from first part of period
        part2_distributions: Distributions from second part of period
        current_period: Current training period
        class_color: Color identifier for the class focus
        output_dir: Directory to save plots
        save_plot: Whether to save the plot
    """
    fig, axs = plt.subplots(4, 1, figsize=(25, 20))
    axs = axs.flatten()
    
    cmap = plt.get_cmap('Reds') if class_color == 'red' else plt.get_cmap('Blues')
    num_colors = 10
    color_range = np.linspace(0.3, 1.0, num_colors)
    
    def plot_single_distribution(distribution, layer_idx, linestyle, color, label):
        """Plot a single distribution with interpolation."""
        indices = np.arange(len(distribution.flatten()))
        
        if len(distribution.flatten()) >= 4:
            f_interp = interp1d(indices, distribution.flatten(), kind='cubic')
        else:
            f_interp = interp1d(indices, distribution.flatten(), kind='linear')
        
        indices_new = np.linspace(indices.min(), indices.max(), num=500)
        dist_interp = f_interp(indices_new)
        
        axs[layer_idx].plot(indices_new, dist_interp, 
                           linestyle=linestyle, color=color, 
                           linewidth=2.5, label=label, alpha=0.8)
        axs[layer_idx].fill_between(indices_new, dist_interp, 
                                   alpha=0.3, color=color)
    
    for layer_idx in range(4):
        part1_color = cmap(color_range[current_period % num_colors])
        part2_color = cmap(color_range[current_period % num_colors])
        
        # Plot both parts
        plot_single_distribution(part1_distributions[layer_idx], layer_idx, 
                               '-', part1_color, f'Part 1 ({class_color})')
        plot_single_distribution(part2_distributions[layer_idx], layer_idx, 
                               '--', part2_color, f'Part 2 ({class_color})')
        
        axs[layer_idx].set_title(f'Disjoint Distributions - Layer {layer_idx + 1}', fontsize=18)
        axs[layer_idx].set_xlabel('Parameter Index', fontsize=14)
        axs[layer_idx].set_ylabel('Mean Normalized Gradients', fontsize=14)
        axs[layer_idx].legend(fontsize=12)
        axs[layer_idx].grid(True, alpha=0.7)
    
    plt.suptitle(f'Disjoint Gradient Analysis - Period {current_period} ({class_color.capitalize()} Focus)', 
                 fontsize=20)
    plt.tight_layout()
    
    if save_plot and output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"disjoint_distributions_{class_color}_period_{current_period}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    
    plt.show()


def plot_shannon_entropies(entropy_data: Dict[int, Dict[str, List[float]]],
                          part_name: str = "Part 1",
                          output_dir: Optional[Path] = None,
                          save_plot: bool = False) -> None:
    """
    Plot Shannon entropies across layers and periods.
    
    Args:
        entropy_data: Dictionary with layer -> period -> entropy values
        part_name: Name of the training part being analyzed
        output_dir: Directory to save plots
        save_plot: Whether to save the plot
    """
    sns.set_context("talk")
    sns.set_style("white")
    palette = ['#FF4500', '#1E90FF', '#FFD700']  # Orange, Blue, Gold
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs = axs.flatten()
    
    for layer_idx in range(4):
        for period_key, values in entropy_data[layer_idx].items():
            if values:  # Check if there are values to plot
                color_idx = int(period_key) - 1 if period_key.isdigit() else 0
                color = palette[color_idx % len(palette)]
                
                axs[layer_idx].plot(values, label=f'{part_name} - Period {period_key}', 
                                   marker='o', markersize=8, alpha=0.8, 
                                   color=color, linewidth=2.5)
        
        axs[layer_idx].set_title(f'Shannon Entropy - Layer {layer_idx + 1} ({part_name})', 
                                fontsize=16)
        axs[layer_idx].set_xlabel('Training Step', fontsize=12)
        axs[layer_idx].set_ylabel('Shannon Entropy', fontsize=12)
        axs[layer_idx].legend(fontsize=10)
        axs[layer_idx].grid(True, alpha=0.6)
    
    plt.suptitle(f'Shannon Entropy Evolution - {part_name}', fontsize=18)
    plt.tight_layout()
    
    if save_plot and output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"shannon_entropies_{part_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    
    plt.show()


def plot_kl_divergences(kl_data: Dict[int, Dict[str, List[float]]],
                       part_name: str = "Part 1",
                       output_dir: Optional[Path] = None,
                       save_plot: bool = False) -> None:
    """
    Plot KL divergences between different class-focused training phases.
    
    Args:
        kl_data: Dictionary with layer -> comparison -> KL values
        part_name: Name of the training part being analyzed
        output_dir: Directory to save plots  
        save_plot: Whether to save the plot
    """
    sns.set_context("talk")
    sns.set_style("white")
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    axs = axs.flatten()
    
    comparison_colors = {
        '12': '#FF6B6B',  # Red-Blue comparison
        '13': '#4ECDC4',  # Red-Yellow comparison  
        '23': '#45B7D1'   # Blue-Yellow comparison
    }
    
    comparison_labels = {
        '12': 'Red vs Blue Focus',
        '13': 'Red vs Yellow Focus', 
        '23': 'Blue vs Yellow Focus'
    }
    
    for layer_idx in range(4):
        for comparison, values in kl_data[layer_idx].items():
            if values:  # Check if there are values to plot
                color = comparison_colors.get(comparison, '#333333')
                label = comparison_labels.get(comparison, f'Comparison {comparison}')
                
                axs[layer_idx].plot(values, label=label,
                                   marker='s', markersize=6, alpha=0.8,
                                   color=color, linewidth=2)
        
        axs[layer_idx].set_title(f'KL Divergences - Layer {layer_idx + 1} ({part_name})',
                                fontsize=16)
        axs[layer_idx].set_xlabel('Training Period', fontsize=12)
        axs[layer_idx].set_ylabel('KL Divergence', fontsize=12)
        axs[layer_idx].legend(fontsize=10)
        axs[layer_idx].grid(True, alpha=0.6)
        axs[layer_idx].set_yscale('log')
    
    plt.suptitle(f'KL Divergence Evolution - {part_name}', fontsize=18)
    plt.tight_layout()
    
    if save_plot and output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"kl_divergences_{part_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    
    plt.show()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Advanced gradient distribution analysis')
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Directory to save analysis plots')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory {experiment_dir} does not exist")
        return
    
    # Look for gradient data files
    gradient_files = list(experiment_dir.glob('*gradients*.pkl'))
    
    if not gradient_files:
        print(f"No gradient files found in {experiment_dir}")
        return
    
    print(f"Found {len(gradient_files)} gradient files")
    print("Advanced gradient analysis functionality available.")
    print("Import specific functions for detailed analysis:")
    print("  - compute_kl_divergence")
    print("  - plot_disjoint_distributions") 
    print("  - plot_shannon_entropies")
    print("  - plot_kl_divergences")


if __name__ == "__main__":
    main()