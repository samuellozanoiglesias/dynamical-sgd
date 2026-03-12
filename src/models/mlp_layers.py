"""
MLP Layer Utilities for Neural Collapse Analysis

Custom layer implementations for building MLP architectures compatible with
Neural Collapse analysis, particularly layers without bias terms.

Author: Nicolas Ratier Werbin
Email: nicolasratierwerbin@gmail.com
"""

import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax


def DenseNoBias(out_dim, W_init=None):
    """
    Dense layer without bias parameter for Neural Collapse analysis.
    
    This layer is used for the final classifier when studying Neural Collapse
    with bias-free architectures (following some NC paper configurations).
    
    Args:
        out_dim: Output dimension
        W_init: Weight initialization function (defaults to glorot_normal)
        
    Returns:
        init_fun, apply_fun: JAX stax-compatible layer functions
        
    Example:
        >>> layers = [stax.Dense(128), stax.Relu, DenseNoBias(10)]
        >>> model = stax.serial(*layers)
    """
    if W_init is None:
        W_init = stax.glorot_normal()
    
    def init_fun(rng, input_shape):
        """Initialize weights without bias."""
        k = random.split(rng)[0]
        W = W_init(k, (input_shape[-1], out_dim))
        return input_shape[:-1] + (out_dim,), W

    def apply_fun(W, inputs, **kwargs):
        """Apply linear transformation: output = inputs @ W"""
        return jnp.dot(inputs, W)

    return init_fun, apply_fun


def create_mlp_architecture(
    input_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    num_classes: int,
    use_batchnorm: bool = True,
    use_bias_classifier: bool = True
):
    """
    Create a standard MLP architecture for Neural Collapse analysis.
    
    Architecture:
    - Feature Extractor: [Dense → BatchNorm → ReLU] × num_hidden_layers
    - Classifier: Dense (with or without bias)
    
    Args:
        input_dim: Input dimension (e.g., 784 for MNIST, 2 for spiral)
        hidden_dim: Width of hidden layers (feature dimension)
        num_hidden_layers: Number of hidden layer blocks
        num_classes: Number of output classes
        use_batchnorm: Whether to include BatchNorm layers
        use_bias_classifier: Whether the final classifier has bias
        
    Returns:
        JAX stax model (init_fun, apply_fun)
        
    Example:
        >>> model = create_mlp_architecture(
        ...     input_dim=784, hidden_dim=128, num_hidden_layers=2,
        ...     num_classes=10, use_batchnorm=True, use_bias_classifier=True
        ... )
    """
    layers = []
    
    # First hidden layer: input_dim → hidden_dim
    layers.append(stax.Dense(hidden_dim))
    if use_batchnorm:
        layers.append(stax.BatchNorm(axis=(0,)))
    layers.append(stax.Relu)
    
    # Additional hidden layers: hidden_dim → hidden_dim
    for _ in range(num_hidden_layers - 1):
        layers.append(stax.Dense(hidden_dim))
        if use_batchnorm:
            layers.append(stax.BatchNorm(axis=(0,)))
        layers.append(stax.Relu)
    
    # Final classifier layer
    if use_bias_classifier:
        layers.append(stax.Dense(num_classes))
    else:
        layers.append(DenseNoBias(num_classes))
    
    return stax.serial(*layers)
