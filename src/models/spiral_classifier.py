"""
Spiral Classifier for Dynamical SGD Analysis

This module implements a neural network classifier for spiral datasets with dynamic batch 
training that focuses on different classes periodically. The classifier is designed to 
study the internal reconfiguration of neural networks when trained with dynamically 
composed batches.

Key Features:
- Generates spiral datasets with configurable parameters
- Implements periodic class-focused training dynamics
- Tracks weight evolution and gradient distributions over time
- Provides visualization tools for decision boundaries and internal dynamics
- Analyzes layer-wise parameter changes and correlations

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

# Configure JAX to use GPU if available
try:
    jax.config.update('jax_platform_name', 'gpu')
    print("GPU configured for JAX")
except:
    print("Using CPU for JAX (GPU not available)")


class SpiralClassifier:
    """
    A neural network classifier for spiral datasets with dynamic batch composition.
    
    This class implements a multi-layer perceptron that is trained on spiral data
    using periodically varying class focus. The training process alternates between
    emphasizing different classes over time, allowing the study of how neural networks
    adapt their internal representations.
    
    Attributes:
        points_per_class (int): Number of data points per class in the spiral dataset
        num_classes (int): Number of spiral classes (typically 3)
        nn_width (int): Width of the hidden layer
        learning_rate (float): Learning rate for the optimizer
        period_length (int): Period length T for the dynamic class focus
        model: JAX/Stax neural network model
        optimizer: Optax optimizer instance
        last_params: Final trained parameters
        weights_history (List): History of weight vectors during training
        all_gradients (List): History of gradient information
        kl_divergences_part1 (Dict): KL divergences for first part of each period
        kl_divergences_part2 (Dict): KL divergences for second part of each period
    """
    
    def __init__(
        self,
        points_per_class: int = 100,
        num_classes: int = 3,
        nn_width: int = 100,
        learning_rate: float = 0.01,
        optimizer_type: str = "adam",
        period_length: int = 5000,
        l2_reg: float = 1e-4,
        random_seed: int = 0,
        label: str = "experiment",
        track_weight_diff: bool = False,
        weight_diff_step_interval: int = 100,
        real_time_visualization: bool = False,
        vis_step_interval: int = 100
    ):
        """
        Initialize the SpiralClassifier.
        
        Args:
            points_per_class: Number of data points per class
            num_classes: Number of spiral classes (default: 3)
            nn_width: Width of the hidden layer
            learning_rate: Learning rate for optimization
            optimizer_type: Type of optimizer ("adam" or "sgd")
            period_length: Period T for dynamic class focus
            l2_reg: L2 regularization coefficient
            random_seed: Random seed for reproducibility
            label: Label for experiment identification
            track_weight_diff: Whether to track weight differences over time
            weight_diff_step_interval: Interval for weight difference tracking
            real_time_visualization: Whether to show real-time visualizations
            vis_step_interval: Interval for visualizations
        """
        # Basic configuration
        self.points_per_class = points_per_class
        self.num_classes = num_classes
        self.nn_width = nn_width
        self.learning_rate = learning_rate
        self.period_length = period_length
        self.l2_reg = l2_reg
        self.label = label
        
        # Tracking configuration
        self.track_weight_diff = track_weight_diff
        self.weight_diff_step_interval = weight_diff_step_interval
        self.real_time_visualization = real_time_visualization
        self.vis_step_interval = vis_step_interval
        
        # Initialize model and optimizer
        self.model = self._create_model()
        
        if optimizer_type.lower() == "adam":
            self.optimizer = optax.adam(learning_rate)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optax.sgd(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
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
        Create the neural network model architecture.
        
        Returns:
            A tuple containing the initialization and application functions
        """
        return stax.serial(
            stax.Dense(self.nn_width), 
            stax.Relu,
            stax.Dense(self.num_classes)
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
        _, params = self.model[0](random.PRNGKey(seed), (-1, 2))
        return params
    
    @partial(jit, static_argnums=(0,))
    def make_dataset(
        self, 
        revolutions: float = 4.0, 
        seed: Optional[int] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate a spiral dataset with multiple classes.
        
        Each class forms a spiral in 2D space with different starting angles.
        The spirals have radius increasing from 0 to 1 and multiple revolutions.
        
        Args:
            revolutions: Number of spiral revolutions (default: 4.0)
            seed: Random seed for dataset generation
            
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
            r = jnp.linspace(0., 1, N)  # radius from 0 to 1
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
    
    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params: Any, X: jnp.ndarray, Y: jnp.ndarray) -> float:
        """
        Compute cross-entropy loss for a batch of data.
        
        Args:
            params: Model parameters
            X: Input features of shape (batch_size, 2)
            Y: One-hot encoded labels of shape (batch_size, num_classes)
            
        Returns:
            Cross-entropy loss value
        """
        # Forward pass
        predictions = self.model[1](params, X)
        
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
        
        return loss
    
    @partial(jit, static_argnums=(0,))
    def accuracy(self, params: Any, X: jnp.ndarray, Y: jnp.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Args:
            params: Model parameters
            X: Input features
            Y: One-hot encoded labels
            
        Returns:
            Classification accuracy (fraction of correct predictions)
        """
        predictions = self.model[1](params, X)
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
        # Compute gradients
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