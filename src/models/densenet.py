"""
DenseNet-40 Architecture for MNIST (Neural Collapse Paper)

Implementation of DenseNet-40 following the configuration used in Neural Collapse papers:
- 3 dense blocks with 12 layers each (36 convolution layers total)
- Growth rate k=12
- Compression factor 0.5 in transition layers
- Dropout rate 0.0 (configurable)
- Initial convolution: 16 filters

Reference: Papyan et al. (2020) "Prevalence of Neural Collapse during the terminal 
phase of deep learning training"

Author: Samuel Lozano Iglesias
Email: samuel.lozano@ucm.es
"""

import jax.numpy as jnp
from flax import linen as nn


class DenseLayer(nn.Module):
    """
    Single dense layer: BN -> ReLU -> Conv3x3
    
    Implements one layer of a dense block following the DenseNet architecture.
    Each layer produces `growth_rate` feature maps that are concatenated with
    the input (dense connection).
    """
    growth_rate: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        out = nn.BatchNorm(use_running_average=not train)(x)
        out = nn.relu(out)
        out = nn.Conv(features=self.growth_rate, kernel_size=(3, 3), padding='SAME')(out)
        if self.dropout_rate > 0.0:
            out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(out)
        return jnp.concatenate([x, out], axis=-1)  # Dense connection


class DenseBlock(nn.Module):
    """
    Dense block with multiple dense layers
    
    A dense block consists of multiple dense layers where each layer receives
    feature maps from all preceding layers (dense connections).
    """
    num_layers: int
    growth_rate: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        for _ in range(self.num_layers):
            x = DenseLayer(growth_rate=self.growth_rate, dropout_rate=self.dropout_rate)(x, train)
        return x


class TransitionLayer(nn.Module):
    """
    Transition layer: BN -> ReLU -> Conv1x1 -> AvgPool2x2
    
    Transition layers are placed between dense blocks to reduce spatial dimensions
    and control the number of feature maps (compression).
    """
    num_output_features: int
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        out = nn.BatchNorm(use_running_average=not train)(x)
        out = nn.relu(out)
        out = nn.Conv(features=self.num_output_features, kernel_size=(1, 1))(out)
        out = nn.avg_pool(out, window_shape=(2, 2), strides=(2, 2))
        return out


class DenseNet40(nn.Module):
    """
    DenseNet-40 for MNIST (Neural Collapse Paper Configuration)
    
    Architecture:
    - Initial Conv: 16 filters, 3x3, padding='SAME'
    - Dense Block 1: 12 layers (growth_rate=12) -> 16 + 12*12 = 160 features
    - Transition 1: compression=0.5 -> 80 features
    - Dense Block 2: 12 layers (growth_rate=12) -> 80 + 12*12 = 224 features
    - Transition 2: compression=0.5 -> 112 features
    - Dense Block 3: 12 layers (growth_rate=12) -> 112 + 12*12 = 256 features
    - Global Average Pooling -> 256-dimensional feature vector
    - Fully Connected Classifier -> num_classes outputs
    
    Total: ~40 layers (36 conv + 2 transition conv + 1 initial conv + 1 classifier)
    Dropout: 0.0 (as specified in Neural Collapse papers)
    
    Input shape: (batch_size, 28, 28, 1) for MNIST
    Output shape: (batch_size, num_classes) for logits
                 or (batch_size, 256) for features when return_features=True
    """
    num_classes: int = 10
    growth_rate: int = 12
    block_config: tuple = (12, 12, 12)  # 3 blocks with 12 layers each
    num_init_features: int = 16
    compression: float = 0.5
    dropout_rate: float = 0.0
    use_bias: bool = True  # For classifier layer (Neural Collapse analysis)
    
    @nn.compact
    def __call__(self, x, train: bool = True, return_features: bool = False):
        """
        Forward pass through DenseNet-40.
        
        Args:
            x: Input tensor of shape (batch_size, 28, 28, 1)
            train: Whether in training mode (affects BatchNorm and Dropout)
            return_features: If True, return 256-d feature vector instead of logits
            
        Returns:
            If return_features=True: Feature vector of shape (batch_size, 256)
            If return_features=False: Logits of shape (batch_size, num_classes)
        """
        # Initial convolution: (28, 28, 1) -> (28, 28, 16)
        features = nn.Conv(features=self.num_init_features, kernel_size=(3, 3), padding='SAME')(x)
        
        num_features = self.num_init_features
        
        # Dense Block 1
        features = DenseBlock(
            num_layers=self.block_config[0],
            growth_rate=self.growth_rate,
            dropout_rate=self.dropout_rate
        )(features, train)
        num_features += self.block_config[0] * self.growth_rate
        
        # Transition 1
        num_features = int(num_features * self.compression)
        features = TransitionLayer(num_output_features=num_features)(features, train)
        
        # Dense Block 2
        features = DenseBlock(
            num_layers=self.block_config[1],
            growth_rate=self.growth_rate,
            dropout_rate=self.dropout_rate
        )(features, train)
        num_features += self.block_config[1] * self.growth_rate
        
        # Transition 2
        num_features = int(num_features * self.compression)
        features = TransitionLayer(num_output_features=num_features)(features, train)
        
        # Dense Block 3 (final block, no transition after)
        features = DenseBlock(
            num_layers=self.block_config[2],
            growth_rate=self.growth_rate,
            dropout_rate=self.dropout_rate
        )(features, train)
        
        # Final batch norm and ReLU
        features = nn.BatchNorm(use_running_average=not train)(features)
        features = nn.relu(features)
        
        # Global average pooling: (H, W, C) -> (C,)
        features = jnp.mean(features, axis=(1, 2))  # Feature vector h(x)
        
        # Return features for Neural Collapse analysis if requested
        if return_features:
            return features
        
        # Classifier: W·h(x) + b
        if self.use_bias:
            logits = nn.Dense(features=self.num_classes, use_bias=True)(features)
        else:
            logits = nn.Dense(features=self.num_classes, use_bias=False)(features)
        
        return logits


def calculate_densenet40_feature_dim(
    num_init_features: int = 16,
    growth_rate: int = 12,
    block_config: tuple = (12, 12, 12),
    compression: float = 0.5
) -> int:
    """
    Calculate the feature dimension (output of global pooling) for DenseNet-40.
    
    Args:
        num_init_features: Number of filters in initial convolution
        growth_rate: Growth rate k (filters added per layer)
        block_config: Number of layers in each dense block
        compression: Compression factor in transition layers
        
    Returns:
        Feature dimension (256 for standard DenseNet-40)
    """
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        num_features += num_layers * growth_rate
        # Apply transition compression except after last block
        if i < len(block_config) - 1:
            num_features = int(num_features * compression)
    
    return num_features
