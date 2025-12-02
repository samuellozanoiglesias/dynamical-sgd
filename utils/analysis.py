"""
Analysis utilities for studying neural network dynamics.

This module provides functions for analyzing gradient distributions, weight evolution,
KL divergences, and other dynamical properties of neural networks during training.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from scipy.stats import entropy
from scipy.interpolate import interp1d


def compute_gradient_statistics(
    gradients: List[Any],
    layer_idx: int = 0
) -> Dict[str, float]:
    """
    Compute statistics for gradients of a specific layer.
    
    Args:
        gradients: List of gradient dictionaries/tuples
        layer_idx: Index of the layer to analyze
        
    Returns:
        Dictionary with gradient statistics
    """
    # Extract gradients for the specified layer
    layer_grads = []
    for grad_step in gradients:
        if isinstance(grad_step, (list, tuple)):
            if layer_idx < len(grad_step):
                layer_grads.append(grad_step[layer_idx])
        else:
            layer_grads.append(grad_step)
    
    if not layer_grads:
        return {}
    
    # Flatten gradients
    flat_grads = [jnp.array(g).flatten() for g in layer_grads]
    all_grads = jnp.concatenate(flat_grads)
    
    return {
        'mean': float(jnp.mean(all_grads)),
        'std': float(jnp.std(all_grads)),
        'min': float(jnp.min(all_grads)),
        'max': float(jnp.max(all_grads)),
        'l2_norm': float(jnp.linalg.norm(all_grads))
    }


def compute_weight_trajectory_analysis(
    weights_history: List[jnp.ndarray]
) -> Dict[str, Any]:
    """
    Analyze the trajectory of weights during training.
    
    Args:
        weights_history: List of flattened weight vectors
        
    Returns:
        Dictionary with trajectory analysis results
    """
    if len(weights_history) < 2:
        return {}
    
    weights_matrix = jnp.stack(weights_history)
    
    # Compute distances from origin
    distances_from_origin = [float(jnp.linalg.norm(w)) for w in weights_history]
    
    # Compute consecutive differences
    consecutive_diffs = []
    for i in range(1, len(weights_history)):
        diff = jnp.linalg.norm(weights_history[i] - weights_history[i-1])
        consecutive_diffs.append(float(diff))
    
    # Compute correlation matrix
    correlation_matrix = jnp.corrcoef(weights_matrix.T)
    
    # Principal component analysis
    centered_weights = weights_matrix - jnp.mean(weights_matrix, axis=0)
    cov_matrix = jnp.cov(centered_weights.T)
    eigenvals, eigenvecs = jnp.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order
    sorted_indices = jnp.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sorted_indices]
    eigenvecs = eigenvecs[:, sorted_indices]
    
    return {
        'distances_from_origin': distances_from_origin,
        'consecutive_differences': consecutive_diffs,
        'correlation_matrix': correlation_matrix,
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs,
        'explained_variance_ratio': eigenvals / jnp.sum(eigenvals)
    }


def compute_kl_divergence(dist1: jnp.ndarray, dist2: jnp.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence between two distributions.
    
    Args:
        dist1: First distribution
        dist2: Second distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence D(dist1 || dist2)
    """
    # Normalize distributions
    p = dist1 / jnp.sum(jnp.abs(dist1))
    q = dist2 / jnp.sum(jnp.abs(dist2))
    
    # Add epsilon to avoid log(0)
    p = jnp.where(p == 0, epsilon, jnp.abs(p))
    q = jnp.where(q == 0, epsilon, jnp.abs(q))
    
    # Compute KL divergence
    kl_div = jnp.sum(p * jnp.log(p / q))
    return float(kl_div)


def compute_jensen_shannon_divergence(dist1: jnp.ndarray, dist2: jnp.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Args:
        dist1: First distribution
        dist2: Second distribution
        
    Returns:
        Jensen-Shannon divergence
    """
    # Normalize distributions
    p = jnp.abs(dist1) / jnp.sum(jnp.abs(dist1))
    q = jnp.abs(dist2) / jnp.sum(jnp.abs(dist2))
    
    # Compute average distribution
    m = (p + q) / 2
    
    # Compute JS divergence
    js_div = 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)
    return js_div


def compute_cosine_similarity(vec1: jnp.ndarray, vec2: jnp.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity
    """
    vec1_flat = jnp.array(vec1).flatten()
    vec2_flat = jnp.array(vec2).flatten()
    
    dot_product = jnp.dot(vec1_flat, vec2_flat)
    norm1 = jnp.linalg.norm(vec1_flat)
    norm2 = jnp.linalg.norm(vec2_flat)
    
    return float(dot_product / (norm1 * norm2 + 1e-10))


def analyze_gradient_distributions(
    gradients_history: List[List[jnp.ndarray]],
    period_length: int,
    initial_range: Tuple[int, int],
    final_range: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Analyze gradient distributions across different periods and phases.
    
    Args:
        gradients_history: List of gradient snapshots over time
        period_length: Length of each period
        initial_range: (start, end) steps for initial phase analysis
        final_range: (start, end) steps for final phase analysis
        
    Returns:
        Dictionary with distribution analysis results
    """
    results = {
        'period_distributions_initial': [],
        'period_distributions_final': [],
        'kl_divergences': {'initial': [], 'final': []},
        'js_divergences': {'initial': [], 'final': []}
    }
    
    num_periods = len(gradients_history) // period_length
    
    for period_idx in range(num_periods):
        period_start = period_idx * period_length
        
        # Extract initial and final phase gradients for this period
        initial_start = period_start + initial_range[0]
        initial_end = period_start + initial_range[1]
        final_start = period_start + final_range[0]
        final_end = period_start + final_range[1]
        
        if (initial_end < len(gradients_history) and 
            final_end < len(gradients_history)):
            
            # Compute mean gradients for initial and final phases
            initial_grads = gradients_history[initial_start:initial_end]
            final_grads = gradients_history[final_start:final_end]
            
            if initial_grads and final_grads:
                # Average across time steps in each phase
                mean_initial = compute_mean_gradient_distribution(initial_grads)
                mean_final = compute_mean_gradient_distribution(final_grads)
                
                results['period_distributions_initial'].append(mean_initial)
                results['period_distributions_final'].append(mean_final)
    
    # Compute divergences between consecutive periods
    for i in range(1, len(results['period_distributions_initial'])):
        # Initial phase comparisons
        kl_initial = compute_kl_divergence(
            results['period_distributions_initial'][i],
            results['period_distributions_initial'][i-1]
        )
        js_initial = compute_jensen_shannon_divergence(
            results['period_distributions_initial'][i],
            results['period_distributions_initial'][i-1]
        )
        
        results['kl_divergences']['initial'].append(kl_initial)
        results['js_divergences']['initial'].append(js_initial)
        
        # Final phase comparisons
        kl_final = compute_kl_divergence(
            results['period_distributions_final'][i],
            results['period_distributions_final'][i-1]
        )
        js_final = compute_jensen_shannon_divergence(
            results['period_distributions_final'][i],
            results['period_distributions_final'][i-1]
        )
        
        results['kl_divergences']['final'].append(kl_final)
        results['js_divergences']['final'].append(js_final)
    
    return results


def compute_mean_gradient_distribution(gradient_snapshots: List[List[jnp.ndarray]]) -> jnp.ndarray:
    """
    Compute mean gradient distribution across multiple snapshots.
    
    Args:
        gradient_snapshots: List of gradient snapshots
        
    Returns:
        Mean gradient distribution
    """
    if not gradient_snapshots:
        return jnp.array([])
    
    # Flatten and concatenate all gradients
    all_grads = []
    for snapshot in gradient_snapshots:
        for layer_grad in snapshot:
            if isinstance(layer_grad, (tuple, list)):
                for g in layer_grad:
                    all_grads.append(jnp.array(g).flatten())
            else:
                all_grads.append(jnp.array(layer_grad).flatten())
    
    if all_grads:
        concatenated = jnp.concatenate(all_grads)
        return jnp.abs(concatenated) / jnp.sum(jnp.abs(concatenated))
    else:
        return jnp.array([])


def compute_shannon_entropy(distribution: jnp.ndarray) -> float:
    """
    Compute Shannon entropy of a distribution.
    
    Args:
        distribution: Probability distribution
        
    Returns:
        Shannon entropy
    """
    # Normalize to probability distribution
    p = jnp.abs(distribution)
    p = p / jnp.sum(p)
    
    # Avoid log(0)
    p = jnp.where(p == 0, 1e-10, p)
    
    entropy_val = -jnp.sum(p * jnp.log(p))
    return float(entropy_val)


def analyze_weight_layer_distances(
    params_history: List[Any],
    initial_params: Any,
    layer_names: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """
    Analyze L2 distances of each layer from initial parameters.
    
    Args:
        params_history: History of parameter snapshots
        initial_params: Initial parameters for reference
        layer_names: Optional names for layers
        
    Returns:
        Dictionary with distance evolution for each layer
    """
    if not params_history:
        return {}
    
    num_layers = len(initial_params)
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(num_layers)]
    
    distances = {name: [] for name in layer_names}
    
    for params in params_history:
        for i, (current_layer, initial_layer, layer_name) in enumerate(
            zip(params, initial_params, layer_names)
        ):
            # Compute L2 distance for this layer
            if isinstance(current_layer, (tuple, list)):
                # Handle layers with multiple components (weights, biases)
                layer_dist = 0.0
                for curr_comp, init_comp in zip(current_layer, initial_layer):
                    diff = curr_comp - init_comp
                    layer_dist += float(jnp.sum(diff ** 2))
                distances[layer_name].append(np.sqrt(layer_dist))
            else:
                # Single component layer
                diff = current_layer - initial_layer
                layer_dist = float(jnp.sqrt(jnp.sum(diff ** 2)))
                distances[layer_name].append(layer_dist)
    
    return distances


def compute_effective_rank(matrix: jnp.ndarray, threshold: float = 0.01) -> int:
    """
    Compute the effective rank of a matrix.
    
    The effective rank counts the number of singular values that are
    above a threshold (as a fraction of the largest singular value).
    
    Args:
        matrix: Input matrix
        threshold: Threshold as fraction of largest singular value
        
    Returns:
        Effective rank
    """
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    max_sv = jnp.max(singular_values)
    significant_svs = singular_values > (threshold * max_sv)
    return int(jnp.sum(significant_svs))


def analyze_activation_patterns(
    model_fn: Any,
    params_history: List[Any],
    X_sample: jnp.ndarray,
    layer_idx: int = 0
) -> Dict[str, Any]:
    """
    Analyze how activation patterns change over training.
    
    Args:
        model_fn: Model function
        params_history: History of parameters
        X_sample: Sample input data
        layer_idx: Layer to analyze
        
    Returns:
        Dictionary with activation pattern analysis
    """
    activations_history = []
    
    for params in params_history:
        # Get activations for the specified layer
        # This is a simplified version - would need model-specific implementation
        try:
            if hasattr(model_fn, '__getitem__'):
                # For stax models, extract intermediate activations
                activations = model_fn[1](params, X_sample)
            else:
                activations = model_fn(params, X_sample)
            
            activations_history.append(activations)
        except Exception:
            # Skip if activation extraction fails
            continue
    
    if not activations_history:
        return {}
    
    # Compute statistics
    activation_norms = [float(jnp.linalg.norm(act)) for act in activations_history]
    activation_means = [float(jnp.mean(act)) for act in activations_history]
    activation_stds = [float(jnp.std(act)) for act in activations_history]
    
    # Compute effective rank evolution
    effective_ranks = []
    for activations in activations_history:
        if activations.ndim == 2:  # (batch_size, features)
            eff_rank = compute_effective_rank(activations)
            effective_ranks.append(eff_rank)
    
    return {
        'norms': activation_norms,
        'means': activation_means,
        'stds': activation_stds,
        'effective_ranks': effective_ranks
    }