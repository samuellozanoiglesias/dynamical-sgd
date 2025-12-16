"""
Data processing utilities for the dynamical SGD project.

This module provides functions for data generation, preprocessing, and batch sampling
with dynamic class composition.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


def generate_spiral_data(
    points_per_class: int = 100,
    num_classes: int = 3,
    revolutions: float = 4.0,
    noise_std: float = 0.2,
    random_seed: Optional[int] = None,
    angular_offsets: Optional[List[float]] = None,
    randomize_offsets: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate spiral dataset with multiple classes.
    
    Args:
        points_per_class: Number of points per class
        num_classes: Number of spiral classes
        revolutions: Number of spiral revolutions
        noise_std: Standard deviation of angular noise
        random_seed: Random seed for reproducibility
        angular_offsets: Custom angular offsets for each class in degrees (must have num_classes values)
                        If None and randomize_offsets=False: uses uniform spacing (120°, 240° for 3 classes)
                        If None and randomize_offsets=True: generates random offsets
        randomize_offsets: If True and angular_offsets is None, generates random angular offsets
                          This breaks the perfect 120° symmetry for neural collapse experiments
        
    Returns:
        Tuple of (features, labels) where features are (N, 2) and labels are one-hot (N, num_classes)
    """
    key = random.PRNGKey(random_seed if random_seed is not None else 0)
    N, C, pi = points_per_class, num_classes, jnp.pi
    
    # Determine angular offsets for each spiral
    if angular_offsets is not None:
        if len(angular_offsets) != C:
            raise ValueError(f"angular_offsets must have {C} values, got {len(angular_offsets)}")
        # Convert from degrees to radians
        offsets = jnp.array(angular_offsets) * pi / 180.0
    elif randomize_offsets:
        # Generate random offsets in [0, 2π) to break symmetry
        key, subkey = random.split(key)
        offsets = random.uniform(subkey, (C,), minval=0.0, maxval=2.0 * pi)
    else:
        # Default: uniform spacing (creates 120° for 3 classes, etc.)
        offsets = jnp.array([2 * pi * j / C for j in range(C)])
    
    # Initialize arrays
    X = jnp.zeros((N * C, 2))
    Y = jnp.zeros((N * C, C))
    
    for j in range(C):
        # Index slice for this class
        ix = slice(N * j, N * (j + 1))
        
        # Spiral parameters
        r = jnp.linspace(0., 1, N)  # radius
        omega = offsets[j]  # angular offset for this class
        theta_max = revolutions * pi
        
        # Generate angles with noise
        key, subkey = random.split(key)
        noise = random.normal(subkey, (N,)) * noise_std
        angles = jnp.linspace(omega, omega + theta_max, N) + noise
        
        # Convert to Cartesian coordinates
        x_coords = r * jnp.cos(angles)
        y_coords = r * jnp.sin(angles)
        
        # Store data
        X = X.at[ix].set(jnp.c_[x_coords, y_coords])
        Y = Y.at[ix, j].set(1)
    
    return jax.device_put(X), jax.device_put(Y)


def compute_class_indices(Y: jnp.ndarray) -> List[jnp.ndarray]:
    """
    Compute indices for each class from one-hot encoded labels.
    
    Args:
        Y: One-hot encoded labels of shape (N, num_classes)
        
    Returns:
        List of arrays, each containing indices for one class
    """
    num_classes = Y.shape[1]
    return [jnp.where(Y[:, i] == 1)[0] for i in range(num_classes)]


def sample_balanced_batch(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    batch_size: int,
    class_indices: List[jnp.ndarray],
    rng_key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a balanced batch with equal representation from each class.
    
    Args:
        X: Input features
        Y: One-hot encoded labels
        batch_size: Size of batch to sample
        class_indices: Pre-computed indices for each class
        rng_key: Random key for sampling
        
    Returns:
        Tuple of (batch_X, batch_Y)
    """
    num_classes = len(class_indices)
    samples_per_class = batch_size // num_classes
    
    indices = []
    for i in range(num_classes):
        rng_key, sub_key = random.split(rng_key)
        class_sample = random.choice(
            sub_key,
            class_indices[i],
            shape=(samples_per_class,),
            replace=True
        )
        indices.append(class_sample)
    
    # Handle remainder if batch_size is not divisible by num_classes
    remainder = batch_size % num_classes
    if remainder > 0:
        rng_key, sub_key = random.split(rng_key)
        extra_class = random.randint(sub_key, (), 0, num_classes)
        extra_samples = random.choice(
            sub_key,
            class_indices[extra_class],
            shape=(remainder,),
            replace=True
        )
        indices.append(extra_samples)
    
    final_indices = jnp.concatenate(indices)
    return X[final_indices], Y[final_indices]


def sample_weighted_batch(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    class_weights: jnp.ndarray,
    batch_size: int,
    class_indices: List[jnp.ndarray],
    rng_key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a batch with class weights determining the composition.
    
    Args:
        X: Input features
        Y: One-hot encoded labels
        class_weights: Weights for each class (should sum to 1)
        batch_size: Size of batch to sample
        class_indices: Pre-computed indices for each class
        rng_key: Random key for sampling
        
    Returns:
        Tuple of (batch_X, batch_Y)
    """
    # Convert weights to counts
    class_counts = (class_weights * batch_size).astype(int)
    
    # Ensure we have exactly batch_size samples
    total_samples = jnp.sum(class_counts)
    if total_samples < batch_size:
        # Add remaining samples to the class with highest weight
        max_weight_class = jnp.argmax(class_weights)
        class_counts = class_counts.at[max_weight_class].add(batch_size - total_samples)
    
    indices = []
    for i, count in enumerate(class_counts):
        if count > 0:
            rng_key, sub_key = random.split(rng_key)
            class_sample = random.choice(
                sub_key,
                class_indices[i],
                shape=(count,),
                replace=True
            )
            indices.append(class_sample)
    
    if indices:  # Only concatenate if we have samples
        final_indices = jnp.concatenate(indices)
        return X[final_indices], Y[final_indices]
    else:
        # Fallback to balanced sampling
        return sample_balanced_batch(X, Y, batch_size, class_indices, rng_key)


def normalize_features(X: jnp.ndarray, method: str = "standardize") -> Tuple[jnp.ndarray, Dict[str, float]]:
    """
    Normalize input features.
    
    Args:
        X: Input features of shape (N, D)
        method: Normalization method ("standardize", "minmax", or "none")
        
    Returns:
        Tuple of (normalized_X, normalization_params)
    """
    if method == "standardize":
        mean = jnp.mean(X, axis=0)
        std = jnp.std(X, axis=0)
        X_norm = (X - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
        params = {"mean": mean, "std": std, "method": method}
    
    elif method == "minmax":
        min_val = jnp.min(X, axis=0)
        max_val = jnp.max(X, axis=0)
        X_norm = (X - min_val) / (max_val - min_val + 1e-8)
        params = {"min": min_val, "max": max_val, "method": method}
    
    elif method == "none":
        X_norm = X
        params = {"method": method}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_norm, params


def apply_normalization(X: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
    """
    Apply normalization using previously computed parameters.
    
    Args:
        X: Input features to normalize
        params: Normalization parameters from normalize_features
        
    Returns:
        Normalized features
    """
    method = params["method"]
    
    if method == "standardize":
        return (X - params["mean"]) / (params["std"] + 1e-8)
    elif method == "minmax":
        return (X - params["min"]) / (params["max"] - params["min"] + 1e-8)
    elif method == "none":
        return X
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_train_test_split(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    test_ratio: float = 0.2,
    random_seed: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create train-test split while maintaining class balance.
    
    Args:
        X: Input features
        Y: One-hot encoded labels
        test_ratio: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, Y_train, X_test, Y_test)
    """
    key = random.PRNGKey(random_seed if random_seed is not None else 42)
    
    num_classes = Y.shape[1]
    class_indices = compute_class_indices(Y)
    
    train_indices = []
    test_indices = []
    
    for class_idx in range(num_classes):
        class_size = len(class_indices[class_idx])
        test_size = int(class_size * test_ratio)
        
        # Randomly shuffle class indices
        key, subkey = random.split(key)
        shuffled_indices = random.permutation(subkey, class_indices[class_idx])
        
        # Split into train and test
        test_indices.append(shuffled_indices[:test_size])
        train_indices.append(shuffled_indices[test_size:])
    
    # Concatenate all indices
    train_idx = jnp.concatenate(train_indices)
    test_idx = jnp.concatenate(test_indices)
    
    # Shuffle the final indices
    key, subkey = random.split(key)
    train_idx = random.permutation(subkey, train_idx)
    
    key, subkey = random.split(key)
    test_idx = random.permutation(subkey, test_idx)
    
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def augment_data(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    noise_std: float = 0.05,
    rotation_range: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply data augmentation to increase dataset size.
    
    Args:
        X: Input features of shape (N, 2)
        Y: One-hot encoded labels
        noise_std: Standard deviation of Gaussian noise to add
        rotation_range: Maximum rotation angle in radians
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (augmented_X, augmented_Y) with doubled size
    """
    key = random.PRNGKey(random_seed if random_seed is not None else 123)
    N = X.shape[0]
    
    # Add Gaussian noise
    key, subkey = random.split(key)
    noise = random.normal(subkey, X.shape) * noise_std
    X_noisy = X + noise
    
    # Add random rotations
    key, subkey = random.split(key)
    angles = random.uniform(subkey, (N,), minval=-rotation_range, maxval=rotation_range)
    
    # Create rotation matrices
    cos_angles = jnp.cos(angles)
    sin_angles = jnp.sin(angles)
    
    # Apply rotations
    x_rot = X[:, 0] * cos_angles - X[:, 1] * sin_angles
    y_rot = X[:, 0] * sin_angles + X[:, 1] * cos_angles
    X_rotated = jnp.column_stack([x_rot, y_rot])
    
    # Combine original and augmented data
    X_augmented = jnp.concatenate([X, X_noisy, X_rotated], axis=0)
    Y_augmented = jnp.concatenate([Y, Y, Y], axis=0)
    
    return X_augmented, Y_augmented


class DataLoader:
    """
    Data loader for batch sampling with dynamic class weights.
    """
    
    def __init__(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        batch_size: int,
        random_seed: int = 0
    ):
        """
        Initialize the data loader.
        
        Args:
            X: Input features
            Y: One-hot encoded labels
            batch_size: Batch size
            random_seed: Random seed
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.key = random.PRNGKey(random_seed)
        self.class_indices = compute_class_indices(Y)
        self.num_classes = Y.shape[1]
    
    def get_batch(self, class_weights: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get a batch with specified class weights.
        
        Args:
            class_weights: Optional class weights (uniform if None)
            
        Returns:
            Tuple of (batch_X, batch_Y)
        """
        self.key, subkey = random.split(self.key)
        
        if class_weights is None:
            # Uniform sampling
            class_weights = jnp.ones(self.num_classes) / self.num_classes
        
        return sample_weighted_batch(
            self.X, self.Y, class_weights, self.batch_size, self.class_indices, subkey
        )
    
    def get_balanced_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get a balanced batch with equal representation from each class.
        
        Returns:
            Tuple of (batch_X, batch_Y)
        """
        self.key, subkey = random.split(self.key)
        return sample_balanced_batch(
            self.X, self.Y, self.batch_size, self.class_indices, subkey
        )