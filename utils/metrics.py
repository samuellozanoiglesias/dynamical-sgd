"""
Metrics and evaluation utilities for neural network analysis.

This module provides functions for computing various metrics during training,
including loss functions, accuracy measures, and custom dynamical metrics.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


@partial(jit, static_argnums=(0, 4))  # Mark model_fn and l2_reg as static
def cross_entropy_loss(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    l2_reg: float = 0.0
) -> float:
    """
    Compute cross-entropy loss with optional L2 regularization.
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        l2_reg: L2 regularization coefficient
        
    Returns:
        Loss value
    """
    # Forward pass
    predictions = model_fn(params, X)
    
    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(predictions)
    ce_loss = -jnp.mean(jnp.sum(log_probs * Y, axis=-1))
    
    # Add L2 regularization (always compute, harmless when l2_reg=0)
    l2_penalty = sum(
        jnp.sum(param ** 2) 
        for param in jax.tree_util.tree_leaves(params)
    )
    ce_loss = ce_loss + l2_reg * l2_penalty
    
    return ce_loss


@partial(jit, static_argnums=(0,))
def classification_accuracy(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        
    Returns:
        Accuracy (fraction of correct predictions)
    """
    predictions = model_fn(params, X)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(Y, axis=1)
    return jnp.mean(predicted_classes == true_classes)


def cross_entropy_loss_batched(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    l2_reg: float = 0.0,
    batch_size: int = 1000
) -> float:
    """
    Compute cross-entropy loss in batches to avoid OOM on large datasets.
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        l2_reg: L2 regularization coefficient
        batch_size: Batch size for computing loss
        
    Returns:
        Average loss value
    """
    num_samples = X.shape[0]
    total_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        
        # Compute loss for this batch (without l2_reg to avoid counting it multiple times)
        batch_loss = cross_entropy_loss(model_fn, params, X_batch, Y_batch, l2_reg=0.0)
        total_loss += float(batch_loss) * (end_idx - start_idx)
    
    # Average loss across all samples
    avg_loss = total_loss / num_samples
    
    # Add L2 regularization once (not per batch)
    if l2_reg > 0.0:
        l2_penalty = sum(
            jnp.sum(param ** 2) 
            for param in jax.tree_util.tree_leaves(params)
        )
        avg_loss = avg_loss + l2_reg * float(l2_penalty)
    
    return avg_loss


def classification_accuracy_batched(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    batch_size: int = 1000
) -> float:
    """
    Compute classification accuracy in batches to avoid OOM on large datasets.
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        batch_size: Batch size for computing accuracy
        
    Returns:
        Overall accuracy (fraction of correct predictions)
    """
    num_samples = X.shape[0]
    correct_predictions = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        
        # Compute accuracy for this batch
        batch_acc = classification_accuracy(model_fn, params, X_batch, Y_batch)
        correct_predictions += float(batch_acc) * (end_idx - start_idx)
    
    return correct_predictions / num_samples


def compute_per_class_accuracy(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray
) -> Dict[int, float]:
    """
    Compute accuracy for each class separately.
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        
    Returns:
        Dictionary mapping class index to accuracy
    """
    predictions = model_fn(params, X)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(Y, axis=1)
    
    num_classes = Y.shape[1]
    per_class_acc = {}
    
    for class_idx in range(num_classes):
        class_mask = true_classes == class_idx
        if jnp.sum(class_mask) > 0:  # Avoid division by zero
            class_correct = jnp.sum(
                (predicted_classes == true_classes) & class_mask
            )
            class_total = jnp.sum(class_mask)
            per_class_acc[class_idx] = float(class_correct / class_total)
        else:
            per_class_acc[class_idx] = 0.0
    
    return per_class_acc


def compute_confusion_matrix(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    predictions = model_fn(params, X)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(Y, axis=1)
    
    num_classes = Y.shape[1]
    confusion_matrix = jnp.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix = confusion_matrix.at[i, j].set(
                jnp.sum((true_classes == i) & (predicted_classes == j))
            )
    
    return confusion_matrix


def compute_top_k_accuracy(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    k: int = 2
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy
    """
    predictions = model_fn(params, X)
    true_classes = jnp.argmax(Y, axis=1)
    
    # Get top-k predictions
    top_k_preds = jnp.argsort(predictions, axis=1)[:, -k:]
    
    # Check if true class is in top-k predictions
    correct = jnp.any(top_k_preds == true_classes[:, None], axis=1)
    
    return jnp.mean(correct)


def compute_calibration_metrics(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics (Expected Calibration Error, etc.).
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        num_bins: Number of confidence bins
        
    Returns:
        Dictionary with calibration metrics
    """
    predictions = model_fn(params, X)
    probabilities = jax.nn.softmax(predictions)
    
    predicted_classes = jnp.argmax(probabilities, axis=1)
    true_classes = jnp.argmax(Y, axis=1)
    confidences = jnp.max(probabilities, axis=1)
    correctness = (predicted_classes == true_classes).astype(float)
    
    # Create bins
    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_indices = jnp.digitize(confidences, bin_boundaries) - 1
    bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)
    
    # Compute ECE
    ece = 0.0
    total_samples = len(confidences)
    
    for bin_idx in range(num_bins):
        bin_mask = bin_indices == bin_idx
        bin_size = jnp.sum(bin_mask)
        
        if bin_size > 0:
            bin_accuracy = jnp.mean(correctness[bin_mask])
            bin_confidence = jnp.mean(confidences[bin_mask])
            ece += (bin_size / total_samples) * jnp.abs(bin_accuracy - bin_confidence)
    
    # Compute MCE (Maximum Calibration Error)
    mce = 0.0
    for bin_idx in range(num_bins):
        bin_mask = bin_indices == bin_idx
        bin_size = jnp.sum(bin_mask)
        
        if bin_size > 0:
            bin_accuracy = jnp.mean(correctness[bin_mask])
            bin_confidence = jnp.mean(confidences[bin_mask])
            mce = jnp.maximum(mce, jnp.abs(bin_accuracy - bin_confidence))
    
    return {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'average_confidence': float(jnp.mean(confidences)),
        'average_accuracy': float(jnp.mean(correctness))
    }


def compute_gradient_norm(gradients: Any) -> float:
    """
    Compute the L2 norm of gradients.
    
    Args:
        gradients: Gradient pytree
        
    Returns:
        L2 norm of gradients
    """
    flat_grads = jax.tree_util.tree_leaves(gradients)
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in flat_grads))
    return float(grad_norm)


def compute_parameter_norm(params: Any) -> float:
    """
    Compute the L2 norm of parameters.
    
    Args:
        params: Parameter pytree
        
    Returns:
        L2 norm of parameters
    """
    flat_params = jax.tree_util.tree_leaves(params)
    param_norm = jnp.sqrt(sum(jnp.sum(p ** 2) for p in flat_params))
    return float(param_norm)


def compute_spectral_norm(weight_matrix: jnp.ndarray) -> float:
    """
    Compute spectral norm (largest singular value) of a weight matrix.
    
    Args:
        weight_matrix: Weight matrix
        
    Returns:
        Spectral norm
    """
    if weight_matrix.ndim != 2:
        # Reshape or flatten if not 2D
        weight_matrix = weight_matrix.reshape(weight_matrix.shape[0], -1)
    
    singular_values = jnp.linalg.svd(weight_matrix, compute_uv=False)
    return float(jnp.max(singular_values))


def compute_weight_distribution_stats(params: Any) -> Dict[str, float]:
    """
    Compute statistics of weight distributions.
    
    Args:
        params: Model parameters
        
    Returns:
        Dictionary with distribution statistics
    """
    flat_params = jax.tree_util.tree_leaves(params)
    all_weights = jnp.concatenate([p.flatten() for p in flat_params])
    
    return {
        'mean': float(jnp.mean(all_weights)),
        'std': float(jnp.std(all_weights)),
        'min': float(jnp.min(all_weights)),
        'max': float(jnp.max(all_weights)),
        'l2_norm': float(jnp.linalg.norm(all_weights)),
        'sparsity': float(jnp.mean(jnp.abs(all_weights) < 1e-6))
    }


class MetricsTracker:
    """
    Class for tracking multiple metrics during training.
    """
    
    def __init__(self, track_calibration: bool = False, track_per_class: bool = False, num_classes: Optional[int] = None, eval_batch_size: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            track_calibration: Whether to track calibration metrics
            track_per_class: Whether to track per-class accuracy and loss metrics
            num_classes: Number of classes (required if track_per_class=True)
            eval_batch_size: Batch size for computing metrics on large datasets (to avoid OOM)
        """
        self.track_calibration = track_calibration
        self.track_per_class = track_per_class
        self.num_classes = num_classes
        self.eval_batch_size = eval_batch_size
        self.metrics_history = {}
        self.step = 0
        
        if self.track_per_class and self.num_classes is None:
            raise ValueError("num_classes must be specified when track_per_class=True")
    
    def update(
        self,
        model_fn: Any,
        params: Any,
        X_train: jnp.ndarray,
        Y_train: jnp.ndarray,
        X_test: Optional[jnp.ndarray] = None,
        Y_test: Optional[jnp.ndarray] = None,
        gradients: Optional[Any] = None,
        l2_reg: float = 0.0
    ) -> Dict[str, float]:
        """
        Update metrics for current step.
        
        Args:
            model_fn: Model function
            params: Current parameters
            X_train: Training features
            Y_train: Training labels
            X_test: Optional test features
            Y_test: Optional test labels
            gradients: Optional gradients
            l2_reg: L2 regularization coefficient
            
        Returns:
            Dictionary with current metrics
        """
        current_metrics = {}
        
        # Ensure l2_reg is a float, not a string
        l2_reg_float = float(l2_reg) if l2_reg is not None else 0.0
        
        # Training metrics - use batched versions for large datasets
        current_metrics['train_loss'] = float(
            cross_entropy_loss_batched(model_fn, params, X_train, Y_train, l2_reg_float, batch_size=self.eval_batch_size)
        )
        current_metrics['train_accuracy'] = float(
            classification_accuracy_batched(model_fn, params, X_train, Y_train, batch_size=self.eval_batch_size)
        )
        
        # Test metrics - use batched versions for large datasets
        if X_test is not None and Y_test is not None:
            current_metrics['test_loss'] = float(
                cross_entropy_loss_batched(model_fn, params, X_test, Y_test, l2_reg_float, batch_size=self.eval_batch_size)
            )
            current_metrics['test_accuracy'] = float(
                classification_accuracy_batched(model_fn, params, X_test, Y_test, batch_size=self.eval_batch_size)
            )
        
        # Per-class metrics
        if self.track_per_class and self.num_classes is not None:
            # Compute per-class metrics using the same approach as compute_per_class_metrics
            from jax import nn
            
            # Convert one-hot to class indices
            train_labels = jnp.argmax(Y_train, axis=1)
            test_labels = jnp.argmax(Y_test, axis=1) if Y_test is not None else None
            
            for c in range(self.num_classes):
                # Training metrics for class c
                train_mask = train_labels == c
                if jnp.sum(train_mask) > 0:
                    X_train_c = X_train[train_mask]
                    Y_train_c = Y_train[train_mask]
                    
                    train_logits_c = model_fn(params, X_train_c)
                    train_loss_c = float(-jnp.mean(jnp.sum(Y_train_c * nn.log_softmax(train_logits_c), axis=1)))
                    train_acc_c = float(jnp.mean(jnp.argmax(train_logits_c, axis=1) == jnp.argmax(Y_train_c, axis=1)))
                else:
                    train_loss_c = 0.0
                    train_acc_c = 0.0
                
                current_metrics[f'train_loss_class_{c}'] = train_loss_c
                current_metrics[f'train_accuracy_class_{c}'] = train_acc_c
                
                # Test metrics for class c
                if Y_test is not None and test_labels is not None:
                    test_mask = test_labels == c
                    if jnp.sum(test_mask) > 0:
                        X_test_c = X_test[test_mask]
                        Y_test_c = Y_test[test_mask]
                        
                        test_logits_c = model_fn(params, X_test_c)
                        test_loss_c = float(-jnp.mean(jnp.sum(Y_test_c * nn.log_softmax(test_logits_c), axis=1)))
                        test_acc_c = float(jnp.mean(jnp.argmax(test_logits_c, axis=1) == jnp.argmax(Y_test_c, axis=1)))
                    else:
                        test_loss_c = 0.0
                        test_acc_c = 0.0
                    
                    current_metrics[f'test_loss_class_{c}'] = test_loss_c
                    current_metrics[f'test_accuracy_class_{c}'] = test_acc_c
        
        # Parameter metrics
        current_metrics['parameter_norm'] = compute_parameter_norm(params)
        weight_stats = compute_weight_distribution_stats(params)
        current_metrics.update({f'weight_{k}': v for k, v in weight_stats.items()})
        
        # Gradient metrics
        if gradients is not None:
            current_metrics['gradient_norm'] = compute_gradient_norm(gradients)
        
        # Calibration metrics
        if self.track_calibration and X_test is not None and Y_test is not None:
            calibration_metrics = compute_calibration_metrics(
                model_fn, params, X_test, Y_test
            )
            current_metrics.update(
                {f'calibration_{k}': v for k, v in calibration_metrics.items()}
            )
        
        # Store in history
        for key, value in current_metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        self.step += 1
        return current_metrics
    
    def get_history(self, metric_name: str) -> List[float]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values over time
        """
        return self.metrics_history.get(metric_name, [])
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        """
        Get all tracked metrics.
        
        Returns:
            Dictionary with all metric histories
        """
        return self.metrics_history.copy()
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.metrics_history, f)
    
    def load_metrics(self, filepath: str) -> None:
        """
        Load metrics from file.
        
        Args:
            filepath: Path to load metrics from
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.metrics_history = pickle.load(f)


def compute_class_separation_score(
    model_fn: Any,
    params: Any,
    X: jnp.ndarray,
    Y: jnp.ndarray
) -> float:
    """
    Compute a score measuring how well classes are separated in the feature space.
    
    Args:
        model_fn: Model function
        params: Model parameters
        X: Input features
        Y: One-hot encoded labels
        
    Returns:
        Class separation score (higher = better separation)
    """
    # Get model predictions/features (assuming last layer before softmax)
    predictions = model_fn(params, X)
    
    true_classes = jnp.argmax(Y, axis=1)
    num_classes = Y.shape[1]
    
    # Compute within-class and between-class distances
    within_class_distances = []
    between_class_distances = []
    
    for class_i in range(num_classes):
        mask_i = true_classes == class_i
        class_i_features = predictions[mask_i]
        
        if len(class_i_features) > 1:
            # Within-class distances
            class_i_center = jnp.mean(class_i_features, axis=0)
            within_distances = jnp.linalg.norm(
                class_i_features - class_i_center, axis=1
            )
            within_class_distances.extend(within_distances)
        
        for class_j in range(class_i + 1, num_classes):
            mask_j = true_classes == class_j
            class_j_features = predictions[mask_j]
            
            if len(class_i_features) > 0 and len(class_j_features) > 0:
                # Between-class distances
                class_i_center = jnp.mean(class_i_features, axis=0)
                class_j_center = jnp.mean(class_j_features, axis=0)
                between_distance = jnp.linalg.norm(class_i_center - class_j_center)
                between_class_distances.append(between_distance)
    
    if within_class_distances and between_class_distances:
        avg_within = jnp.mean(jnp.array(within_class_distances))
        avg_between = jnp.mean(jnp.array(between_class_distances))
        separation_score = avg_between / (avg_within + 1e-10)
    else:
        separation_score = 0.0
    
    return float(separation_score)