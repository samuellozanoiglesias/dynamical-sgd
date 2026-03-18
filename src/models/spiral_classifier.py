"""
Spiral Classifier for Dynamical SGD Analysis

This module implements a neural network classifier with dynamic batch training that 
focuses on different classes periodically. Supports both MLP and DenseNet-40 architectures.

Key Features:
- Supports MLP and DenseNet-40 architectures
- Implements periodic class-focused training dynamics
- Tracks weight evolution and gradient distributions over time
- Provides visualization tools for decision boundaries and internal dynamics
- Analyzes layer-wise parameter changes and correlations
- Compatible with Neural Collapse analysis

Author: Nicolas Ratier Werbin
Email: nicolasratierwerbin@gmail.com
"""

import jax
import jax.numpy as jnp
from jax import random, jit, grad
from jax.example_libraries import stax
from jax.tree_util import tree_map
import optax
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import imageio

# Import model architectures
from .densenet import DenseNet40, calculate_densenet40_feature_dim
from .mlp_layers import DenseNoBias, create_mlp_architecture

# Configure JAX to use GPU if available
def _gpu_available() -> bool:
    try:
        return jax.device_count('gpu') > 0
    except RuntimeError:
        return False

if _gpu_available():
    jax.config.update('jax_platform_name', 'gpu')
    print("GPU configured for JAX")
else:
    print("Using CPU for JAX (GPU not available)")


# =============================================================================
# SpiralClassifier (Unified Classifier for MLP and DenseNet40)
# =============================================================================

class SpiralClassifier:
    """
    A neural network classifier for spiral datasets following the Neural Collapse paper.
    
    Architecture strictly follows Papyan et al. (2020) "Prevalence of Neural Collapse":
    - Feature Extractor h(x): Multiple Dense → BatchNorm → ReLU blocks
    - Linear Classifier: W·h(x) + b
    
    This separation allows studying Neural Collapse in the Terminal Phase of Training (TPT).
    
    Attributes:
        points_per_class (int): Number of data points per class in the spiral dataset
        num_classes (int): Number of spiral classes (typically 3)
        nn_width (int): Width of the hidden layers (feature dimension p)
        num_hidden_layers (int): Number of hidden layers in feature extractor
        learning_rate (float): Learning rate for the optimizer
        period_length (int): Period length T for the dynamic class focus
        encoder: Feature extractor h: R^d → R^p
        classifier: Linear classifier W·h(x) + b
        optimizer: Optax optimizer instance
        last_params: Final trained parameters
        weights_history (List): History of weight vectors during training
        all_gradients (List): History of gradient information
        kl_divergences_part1 (Dict): KL divergences for first part of each period
        kl_divergences_part2 (Dict): KL divergences for second part of each period
    """
    
    def __init__(
        self,
        input_dim: int = 2,        # NEW: Input dimension (2 for spiral, 784 for MNIST)
        points_per_class: int = 100,
        num_classes: int = 3,
        nn_width: int = 100,
        num_hidden_layers: int = 1,
        learning_rate: float = 0.01,
        optimizer_type: str = "adam",
        period_length: int = 5000,
        l2_reg: float = 1e-4,
        random_seed: int = 0,
        label: str = "experiment",
        track_weight_diff: bool = False,
        weight_diff_step_interval: int = 100,
        real_time_visualization: bool = False,
        vis_step_interval: int = 100,
        use_batchnorm: bool = True,
        use_bias: bool = True,
        # Optimizer parameters
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        # Initialization parameters
        weight_init_scale: float = 1.0,
        # Architecture selection
        architecture: str = "mlp",  # "mlp" or "densenet40"
        growth_rate: int = 12,  # For DenseNet
        compression: float = 0.5,  # For DenseNet
        dropout_rate: float = 0.0,  # For DenseNet
    ):
        """
        Initialize the SpiralClassifier (Neural Collapse compliant).
        
        Args:
            points_per_class: Number of data points per class
            num_classes: Number of spiral classes (default: 3)
            nn_width: Width of the hidden layers (feature dimension p)
            num_hidden_layers: Number of hidden layers in feature extractor (default: 1)
            learning_rate: Learning rate for optimization
            optimizer_type: Type of optimizer ("adam" or "sgd")
            period_length: Period T for dynamic class focus
            l2_reg: L2 regularization (weight decay, paper uses 5e-4)
            random_seed: Random seed for reproducibility
            label: Label for experiment identification
            track_weight_diff: Whether to track weight differences over time
            weight_diff_step_interval: Interval for weight difference tracking
            real_time_visualization: Whether to show real-time visualizations
            vis_step_interval: Interval for visualizations
            use_batchnorm: Whether to include BatchNorm layers (default: True)
            use_bias: Whether to include bias terms in Dense layers (default: True)
        """
        # Basic configuration
        self.input_dim = input_dim              # NEW: Store input dimension
        self.points_per_class = points_per_class
        self.num_classes = num_classes
        self.nn_width = nn_width
        self.num_hidden_layers = num_hidden_layers
        self.learning_rate = float(learning_rate)
        self.period_length = period_length
        self.l2_reg = float(l2_reg)  # Ensure it's a float, not a string
        self.label = label
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        
        # Architecture configuration
        self.architecture = architecture.lower()
        self.growth_rate = growth_rate
        self.compression = compression
        self.dropout_rate = dropout_rate
        
        # Optimizer parameters
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialization parameters
        self.weight_init_scale = weight_init_scale
        
        # Tracking configuration
        self.track_weight_diff = track_weight_diff
        self.weight_diff_step_interval = weight_diff_step_interval
        self.real_time_visualization = real_time_visualization
        self.vis_step_interval = vis_step_interval
        
        # Initialize model and optimizer
        self.model = self._create_model()
        
        if optimizer_type.lower() == "adam":
            self.optimizer = optax.adam(
                learning_rate=learning_rate,
                b1=self.beta1,
                b2=self.beta2,
                eps=self.eps
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optax.sgd(
                learning_rate=learning_rate,
                momentum=self.momentum
            )
        elif optimizer_type.lower() == "rmsprop":
            self.optimizer = optax.rmsprop(
                learning_rate=learning_rate,
                decay=self.beta2,  # Use beta2 as decay rate for RMSprop
                eps=self.eps
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Supported: adam, sgd, rmsprop")
        
        # Initialize random key
        self.key = random.PRNGKey(random_seed)
        
        # Initialize tracking variables
        self.last_params = None
        self.initial_weights = None
        self.weights_after_period = None
        self.weights_history = []
        self.param_shapes = None
        
        # Initialize gradient and distribution tracking
        self.all_gradients = []
        self.previous_distributions = []
        self.previous_distributions_part1 = []
        self.previous_distributions_part2 = []
        
        # Initialize KL divergence tracking dictionaries
        self.kl_divergences_part1 = {
            i: {'12': [], '13': [], '23': []} for i in range(4)
        }
        self.kl_divergences_part2 = {
            i: {'12': [], '13': [], '23': []} for i in range(4)
        }
        
        # Initialize entropy tracking
        self.entropies_part1 = {
            layer: {'1': [], '2': [], '3': []} for layer in range(4)
        }
        self.entropies_part2 = {
            layer: {'1': [], '2': [], '3': []} for layer in range(4)
        }
        
        # Training data references (set during training)
        self.X_train = None
        self.Y_train = None
    
    def _create_model(self) -> Tuple:
        """
        Create Neural Collapse compliant architecture.
        
        Architecture follows Papyan et al. (2020):
        - Feature Extractor h(x): Dense → [BatchNorm] → ReLU (repeated) OR DenseNet40
        - Linear Classifier: W·h(x) + b
        
        Returns:
            A tuple containing the initialization and application functions (for MLP)
            or the Flax module (for DenseNet40)
        """
        if self.architecture == "densenet40":
            # Return DenseNet40 Flax module
            return DenseNet40(
                num_classes=self.num_classes,
                growth_rate=self.growth_rate,
                compression=self.compression,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias
            )
        
        # Default MLP architecture using helper function
        return create_mlp_architecture(
            input_dim=self.input_dim,
            hidden_dim=self.nn_width,
            num_hidden_layers=self.num_hidden_layers,
            num_classes=self.num_classes,
            use_batchnorm=self.use_batchnorm,
            use_bias_classifier=self.use_bias
        )
    
    def initialize_params(self, random_seed: Optional[int] = None) -> Any:
        """
        Initialize model parameters.
        
        Args:
            random_seed: Optional random seed for parameter initialization
            
        Returns:
            Initialized model parameters
        """
        seed = random_seed if random_seed is not None else 0
        key = random.PRNGKey(seed)
        
        if self.architecture == "densenet40":
            # DenseNet40 expects images of shape (B, H, W, C)
            # For MNIST: (B, 28, 28, 1)
            dummy_input = jnp.ones((1, 28, 28, 1))
            params = self.model.init(key, dummy_input, train=False)
            return params
        else:
            # MLP architecture
            _, params = self.model[0](key, (-1, self.input_dim))
            return params
    
    def get_features(self, params: Any, X: jnp.ndarray, is_training: bool = False) -> jnp.ndarray:
        """
        Extract last-layer features h(x) from the feature extractor.
        
        This method extracts features from the penultimate layer (before the final
        linear classifier), following the Neural Collapse paper's definition of h(x).
        
        Architecture with BatchNorm (MLP):
            x → [Dense → BatchNorm → ReLU] × num_hidden_layers → h(x) → [Dense] → logits
                └────────── Feature Extractor ──────────┘              └─ Classifier ─┘
        
        Architecture (DenseNet40):
            x → DenseNet Blocks → Global Pool → h(x) → [Dense] → logits
        
        Args:
            params: Model parameters (list of layer parameters or Flax params dict)
            X: Input data of shape (N, input_dim) for MLP or (N, H, W, C) for DenseNet
            is_training: Whether in training mode (affects BatchNorm)
            
        Returns:
            Features h(x) of shape (N, feature_dim)
        """
        if self.architecture == "densenet40":
            # Avoid full-dataset feature extraction for MNIST-sized tensors.
            if X.shape[0] > 2048:
                return self.get_features_batched(params, X, batch_size=256)
            # DenseNet40 using Flax - always use inference mode for feature extraction
            # (we don't need to update batch_stats when just extracting features)
            return self.model.apply(params, X, train=False, return_features=True)
        
        # MLP architecture
        h = X
        num_blocks = self.num_hidden_layers
        
        if self.use_batchnorm:
            # Process each Dense → BatchNorm → ReLU block
            # NOTE: In JAX stax params list, each LAYER creates ONE entry (even ReLU makes empty tuple):
            #   - Dense layer = 1 entry: (W, b) tuple  
            #   - BatchNorm layer = 1 entry: (gamma, beta, mean, var) tuple
            #   - ReLU layer = 1 entry: () empty tuple
            # So each block has 3 entries total (Dense + BatchNorm + ReLU)
            for block_idx in range(num_blocks):
                param_idx = block_idx * 3  # 3 entries per block
                
                # Dense layer - params[param_idx] is a tuple (W, b)
                W, b = params[param_idx]
                h = jnp.dot(h, W) + b
                
                # BatchNorm layer - params[param_idx + 1] is a tuple (gamma, beta, mean, var)
                bn_params = params[param_idx + 1]
                if len(bn_params) == 4:
                    gamma, beta, running_mean, running_var = bn_params
                    # Normalize using running statistics
                    h = gamma * (h - running_mean) / jnp.sqrt(running_var + 1e-5) + beta
                
                # ReLU (creates empty tuple at param_idx + 2)
                h = jnp.maximum(0, h)
        else:
            # Process each Dense → ReLU block (no BatchNorm)
            # Each block has 2 entries: Dense + ReLU (empty tuple)
            for block_idx in range(num_blocks):
                param_idx = block_idx * 2  # 2 entries per block
                
                # Dense layer - params[param_idx] is a tuple (W, b)
                W, b = params[param_idx]
                h = jnp.dot(h, W) + b
                
                # ReLU (creates empty tuple at param_idx + 1)
                h = jnp.maximum(0, h)
        
        return h

    def get_features_batched(self, params: Any, X: jnp.ndarray, batch_size: int = 256) -> jnp.ndarray:
        """
        Extract features in batches to reduce peak memory usage.

        Args:
            params: Model parameters
            X: Input data
            batch_size: Batch size used for feature extraction

        Returns:
            Features h(x) concatenated across all batches
        """
        if X.shape[0] <= batch_size:
            return self.get_features(params, X, is_training=False)

        num_samples = X.shape[0]
        features = []
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X[start_idx:end_idx]
            features.append(self.get_features(params, X_batch, is_training=False))

        return jnp.concatenate(features, axis=0)
    
    def predict(self, params: Any, X: jnp.ndarray, is_training: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        """
        Make predictions using the model.
        
        Args:
            params: Model parameters
            X: Input data
            is_training: Whether in training mode (affects BatchNorm/Dropout)
            
        Returns:
            Model predictions (logits), or (logits, updated_params) for DenseNet during training
        """
        if self.architecture == "densenet40":
            if is_training:
                # During training, batch_stats need to be mutable
                logits, new_model_state = self.model.apply(
                    params, X, train=True, return_features=False,
                    mutable=['batch_stats']
                )
                return logits, new_model_state
            else:
                # During inference, use frozen batch_stats
                return self.model.apply(params, X, train=False, return_features=False)
        else:
            # MLP: use stax model
            return self.model[1](params, X)
    
    @property
    def forward_fn(self):
        """
        Return a unified callable (params, X) -> logits for inference.
        
        Works for both MLP (stax) and DenseNet40 (Flax) architectures.
        Use this instead of classifier.model[1] to remain architecture-agnostic.
        """
        if self.architecture == "densenet40":
            return lambda params, X: self.model.apply(params, X, train=False, return_features=False)
        else:
            return self.model[1]

    def get_classifier_weights(self, params: Any) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Extract classifier weights W and biases b from the final layer.
        
        Args:
            params: Model parameters
            
        Returns:
            Tuple of (W, b):
                - W: Classifier weights of shape (num_classes, feature_dim)
                - b: Classifier biases of shape (num_classes,) or None if use_bias=False
        """
        if self.architecture == "densenet40":
            # Flax DenseNet40: params is a nested dict
            # Structure: params['params']['Dense_N']['kernel'] and optionally ['bias']
            # Find the last Dense layer (classifier) - it should be the only one at top level
            dense_params = params['params']
            
            # Find all Dense layers
            dense_keys = [key for key in dense_params.keys() if 'Dense' in key]
            if not dense_keys:
                raise ValueError("Could not find Dense classifier in DenseNet40 parameters")
            
            # Get the last Dense layer (highest number or last in list)
            # Sort to ensure we get the final classifier
            dense_keys_sorted = sorted(dense_keys)
            classifier_key = dense_keys_sorted[-1]
            
            W = dense_params[classifier_key]['kernel'].T  # (num_classes, feature_dim)
            b = dense_params[classifier_key].get('bias', None)  # (num_classes,) or None
            
            return W, b
        else:
            # MLP: stax format
            # Calculate classifier index
            if self.use_batchnorm:
                classifier_idx = self.num_hidden_layers * 3  # Dense + BatchNorm + ReLU
            else:
                classifier_idx = self.num_hidden_layers * 2  # Dense + ReLU
            
            classifier_params = params[classifier_idx]
            if self.use_bias:
                W_last, b_last = classifier_params
                W = W_last.T  # (num_classes, feature_dim)
                b = b_last    # (num_classes,)
            else:
                W_last = classifier_params[0] if isinstance(classifier_params, tuple) else classifier_params
                W = W_last.T  # (num_classes, feature_dim)
                b = None
            
            return W, b
    
    @partial(jit, static_argnums=(0,))
    def make_dataset(
        self, 
        revolutions: float = 4.0, 
        seed: Optional[int] = None,
        min_radius: float = 0.05
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate a spiral dataset with multiple classes.
        
        Each class forms a spiral in 2D space with different starting angles.
        The spirals have radius increasing from min_radius to 1 and multiple revolutions.
        
        Args:
            revolutions: Number of spiral revolutions (default: 4.0)
            seed: Random seed for dataset generation
            min_radius: Minimum radius to avoid points at origin (0,0). Default 0.05.
            
        Returns:
            Tuple of (X, Y) where:
            - X: Input features of shape (num_classes * points_per_class, 2)
            - Y: One-hot encoded labels of shape (num_classes * points_per_class, num_classes)
        """
        key = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
        N, C, pi = self.points_per_class, self.num_classes, jnp.pi
        
        # Initialize arrays
        X = jnp.zeros((N * C, 2))
        Y = jnp.zeros((N * C, C))
        
        for j in range(C):
            # Define index slice for this class
            ix = slice(N * j, N * (j + 1))
            
            # Create spiral parameters
            # Start from min_radius instead of 0 to avoid points at origin (0,0)
            r = jnp.linspace(min_radius, 1, N)  # radius from min_radius to 1
            omega = 2 * pi / C  # angular offset between classes
            theta_max = revolutions * pi  # maximum angle
            
            # Generate angles with noise
            key, subkey = random.split(key)
            noise = random.normal(subkey, (N,)) * 0.2
            t = jnp.linspace(omega * j, omega * j + theta_max, N) + noise
            
            # Convert to Cartesian coordinates
            x_coords = r * jnp.cos(t)
            y_coords = r * jnp.sin(t)
            
            # Store coordinates and labels
            X = X.at[ix].set(jnp.c_[x_coords, y_coords])
            Y = Y.at[ix, j].set(1)
        
        return jax.device_put(X), jax.device_put(Y)
    
    def compute_class_weights(
        self, 
        t: int, 
        focus_class: int, 
        w_max: float, 
        T: int
    ) -> jnp.ndarray:
        """
        Compute dynamic class weights for time step t following a periodic pattern.
        
        The weight for the focus class follows a triangular wave:
        - Increases linearly from 1 to w_max in the first half of the period
        - Decreases linearly from w_max to 1 in the second half
        - Other classes maintain weight 1
        
        Args:
            t: Current time step within the period
            focus_class: Index of the class currently in focus
            w_max: Maximum weight for the focus class
            T: Period length
            
        Returns:
            Normalized weights for all classes
        """
        slope = 2 * (w_max - 1) / T
        
        # Triangular wave for focus class weight
        w_main_class = jnp.where(
            t < T / 2., 
            1 + t * slope,  # Increasing phase
            2 * w_max - t * slope - 1  # Decreasing phase
        )
        
        # Apply weight to focus class, others remain at 1
        weights = jnp.ones(self.num_classes) + (w_main_class - 1) * jnp.eye(self.num_classes)[focus_class]
        
        # Normalize weights to sum to 1
        return weights / jnp.sum(weights)
    
    def loss_fn(self, params: Any, X: jnp.ndarray, Y: jnp.ndarray) -> Union[float, Tuple[float, Any]]:
        """
        Compute cross-entropy loss for a batch of data.
        
        Args:
            params: Model parameters
            X: Input features of shape (batch_size, input_dim)
            Y: One-hot encoded labels of shape (batch_size, num_classes)
            
        Returns:
            Cross-entropy loss value, or (loss, new_model_state) for DenseNet
        """
        # Forward pass
        pred_result = self.predict(params, X, is_training=True)
        
        # Handle DenseNet which returns (predictions, new_state) during training
        if self.architecture == "densenet40":
            predictions, new_model_state = pred_result
        else:
            predictions = pred_result
            new_model_state = None
        
        # Cross-entropy loss
        log_probs = jax.nn.log_softmax(predictions)
        loss = -jnp.mean(jnp.sum(log_probs * Y, axis=-1))
        
        # Add L2 regularization if specified
        if self.l2_reg > 0:
            l2_penalty = sum(
                jnp.sum(param ** 2) 
                for param in jax.tree_util.tree_leaves(params)
            )
            loss += self.l2_reg * l2_penalty
        
        if new_model_state is not None:
            return loss, new_model_state
        else:
            return loss
    
    def accuracy(self, params: Any, X: jnp.ndarray, Y: jnp.ndarray, batch_size: int = 5000) -> float:
        """
        Compute classification accuracy with batching to avoid OOM on large datasets.
        
        Args:
            params: Model parameters
            X: Input features
            Y: One-hot encoded labels
            batch_size: Batch size for computation (to avoid OOM on large datasets)
            
        Returns:
            Classification accuracy (fraction of correct predictions)
        """
        # For large datasets, use batching
        if X.shape[0] > 10000:
            num_samples = X.shape[0]
            num_batches = (num_samples + batch_size - 1) // batch_size
            correct_predictions = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                X_batch = X[start_idx:end_idx]
                Y_batch = Y[start_idx:end_idx]
                
                predictions = self.predict(params, X_batch, is_training=False)
                # predictions is just logits during inference
                predicted_classes = jnp.argmax(predictions, axis=1)
                true_classes = jnp.argmax(Y_batch, axis=1)
                batch_acc = jnp.mean(predicted_classes == true_classes)
                correct_predictions += float(batch_acc) * (end_idx - start_idx)
            
            return correct_predictions / num_samples
        else:
            # For small datasets, compute directly
            predictions = self.predict(params, X, is_training=False)
            # predictions is just logits during inference
            predicted_classes = jnp.argmax(predictions, axis=1)
            true_classes = jnp.argmax(Y, axis=1)
            return jnp.mean(predicted_classes == true_classes)
    
    def update_step(
        self, 
        params: Any, 
        opt_state: Any, 
        X_batch: jnp.ndarray, 
        Y_batch: jnp.ndarray
    ) -> Tuple[Any, Any, Any]:
        """
        Perform a single optimization step.
        
        Args:
            params: Current model parameters
            opt_state: Current optimizer state
            X_batch: Batch of input features
            Y_batch: Batch of one-hot encoded labels
            
        Returns:
            Tuple of (new_params, new_opt_state, gradients)
        """
        if self.architecture == "densenet40":
            # For DenseNet, handle batch_stats separately
            def loss_fn_for_grad(params_dict):
                loss, new_model_state = self.loss_fn(params_dict, X_batch, Y_batch)
                return loss, new_model_state
            
            # Compute gradients and get new batch_stats
            (loss_val, new_model_state), grads = jax.value_and_grad(loss_fn_for_grad, has_aux=True)(params)
            
            # Update parameters with gradients
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            
            # Merge updated batch_stats into params
            new_params = {**new_params, **new_model_state}
        else:
            # For MLP, standard gradient computation
            grads = grad(self.loss_fn)(params, X_batch, Y_batch)
            
            # Update parameters
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
        
        # Track weight history if enabled
        if self.track_weight_diff:
            flat_weights = self._flatten_params(new_params)
            self.weights_history.append(flat_weights)
        
        return new_params, new_opt_state, grads
    
    def sample_by_class(
        self, 
        X_train: jnp.ndarray, 
        Y_train: jnp.ndarray, 
        class_counts: jnp.ndarray, 
        rng_key: jnp.ndarray, 
        class_indices: List[jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample data points by class according to specified counts.
        
        Args:
            X_train: Training input features
            Y_train: Training labels
            class_counts: Number of samples to draw from each class
            rng_key: Random key for sampling
            class_indices: List of indices for each class
            
        Returns:
            Tuple of (sampled_X, sampled_Y)
        """
        indices = []
        
        for i in range(self.num_classes):
            rng_key, sub_key = random.split(rng_key)
            sampled_indices = random.choice(
                sub_key, 
                class_indices[i], 
                shape=(class_counts[i],), 
                replace=True
            )
            indices.append(sampled_indices)
        
        final_indices = jnp.concatenate(indices)
        return X_train[final_indices], Y_train[final_indices]
    
    def _flatten_params(self, params: Any, return_shapes: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, List]]:
        """
        Flatten model parameters into a single vector.
        
        Args:
            params: Model parameters
            return_shapes: Whether to also return parameter shapes
            
        Returns:
            Flattened parameter vector, optionally with shapes
        """
        flat_params = []
        param_shapes = []
        
        for layer_params in params:
            if isinstance(layer_params, (tuple, list)):
                for p in layer_params:
                    if p.size > 0:  # Skip empty parameters
                        flat_params.append(p.flatten())
                        param_shapes.append(p.shape)
            elif layer_params.size > 0:  # Skip empty parameters
                flat_params.append(layer_params.flatten())
                param_shapes.append(layer_params.shape)
        
        flattened = jnp.concatenate(flat_params)
        
        if return_shapes:
            return flattened, param_shapes
        else:
            return flattened
    
    def calculate_l2_distance(self, weights_a: Any, weights_b: Any) -> float:
        """
        Calculate L2 distance between two sets of weights.
        
        Args:
            weights_a: First set of weights
            weights_b: Second set of weights
            
        Returns:
            L2 distance between the weights
        """
        squared_diff = tree_map(
            lambda a, b: jnp.sum((a - b) ** 2), 
            weights_a, 
            weights_b
        )
        total_squared_diff = sum(jax.tree_util.tree_leaves(squared_diff))
        return float(total_squared_diff)
    
    def save_experiment_data(
        self, 
        losses: List[float], 
        accuracies: List[float], 
        output_dir: Path
    ) -> None:
        """
        Save experiment data to files.
        
        Args:
            losses: Training losses
            accuracies: Training accuracies
            output_dir: Directory to save files
        """
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create description for filenames
        description = f"width_{self.nn_width}-lr_{self.learning_rate}-period_{self.period_length}-label_{self.label}"
        
        # Save losses and accuracies
        with open(output_dir / f'losses_{description}.pkl', 'wb') as f:
            pickle.dump(losses, f)
        
        with open(output_dir / f'accuracies_{description}.pkl', 'wb') as f:
            pickle.dump(accuracies, f)
        
        # Save final parameters
        if self.last_params is not None:
            with open(output_dir / f'final_params_{description}.pkl', 'wb') as f:
                pickle.dump(self.last_params, f)
        
        print(f"Experiment data saved to {output_dir}")