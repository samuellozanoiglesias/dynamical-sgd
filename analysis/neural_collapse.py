"""
Neural Collapse Analysis Module (Simplified)

This module implements simplified Neural Collapse metrics following the paper's
mathematical definitions from "Prevalence of Neural Collapse during the terminal 
phase of deep learning training" (Papyan et al., 2020).

Neural Collapse metrics (paper-exact formulas):
1. NC1 (Variability Collapse): Trace(Σ_W @ Σ_B^†) / C
   - Σ_W = Σ_c π_c E[(h - μ_c)(h - μ_c)^T | y=c]
   - Σ_B = Σ_c π_c (μ_c - μ_G)(μ_c - μ_G)^T
   - Uses Moore-Penrose pseudoinverse (no regularization)
2. NC2 (Equinorm): Coefficient of variation of ||μ_c - μ_G||
3. NC2 (Equiangularity): Std and Mean of off-diagonal Gram matrix
4. NC3 (Self-Duality): ||Ŵ^T - M̂||_F (Frobenius norm)
5. NC7 (Nearest Class-Center): Proportion of test samples where classifier ≠ NCC

Bias-Agnostic Design:
- Supports classifiers with or without bias terms (configurable via use_bias flag)
- use_bias applies ONLY to the final classifier layer (hidden layers always have bias)
- With bias: Uses feature space augmentation h̃=[h;1], W̃=[W;b] for theory-aligned metrics
- Without bias: Standard computation in original feature space
- All metrics remain mathematically valid in both configurations

All metrics are computed in the appropriate feature space (R^p or R^{p+1}).

Author: Samuel Lozano Iglesias  
Email: samuel.lozano@ucm.es
"""

import logging
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass


# =============================================================================
# CORE FUNCTIONS: Compute NC Metrics (Paper Definitions)
# =============================================================================

def compute_nc_metrics(H: jnp.ndarray, labels: jnp.ndarray, W: jnp.ndarray, b: Optional[jnp.ndarray] = None, H_test: Optional[jnp.ndarray] = None, labels_test: Optional[jnp.ndarray] = None) -> dict:
    """
    Compute Neural Collapse metrics following the paper's exact mathematical definitions.
    
    Supports classifiers with or without bias by augmenting the feature space:
    - With bias: h̃ = [h; 1], W̃ = [W; b] → bias-free classifier in augmented space
    - Without bias: Standard computation in original space
    
    Args:
        H: Last-layer features (N, p) where N=num_samples, p=feature_dim
        labels: Class labels (N,) as integers 0, 1, ..., C-1
        W: Classifier weights (C, p) where C=num_classes
        b: Classifier biases (C,) - optional, if None assumes no bias
        H_test: Last-layer test features (N_test, p) for NC7 computation (optional)
        labels_test: Test class labels (N_test,) for NC7 computation (optional)
        
    Returns:
        Dictionary with metrics:
            - nc1_variability: Trace(Σ_W @ Σ_B^†) [paper-exact, no 1/C factor]
            - nc2_equinorm_cv: Coefficient of variation of centered mean norms
            - nc2_equiangular_std: Std of off-diagonal Gram matrix elements
            - nc2_equiangular_mean: Mean of off-diagonal Gram matrix elements
            - nc3_self_duality: ||Ŵ^T - M̂||_F
            - nc7_ncc_mismatch: Proportion of test samples where classifier disagrees with NCC
    """
    # =============================================================================
    # Feature Space Augmentation (if bias exists)
    # =============================================================================
    # Augment feature space to absorb bias: [h; 1] and [W; b]
    # This makes all NC metrics bias-agnostic and theory-aligned
    if b is not None:
        # Augment training features: H → [H, ones]
        ones = jnp.ones((H.shape[0], 1))
        H = jnp.concatenate([H, ones], axis=1)
        
        # Augment test features if provided
        if H_test is not None:
            ones_test = jnp.ones((H_test.shape[0], 1))
            H_test = jnp.concatenate([H_test, ones_test], axis=1)
        
        # Augment classifier weights: W → [W, b]
        W = jnp.concatenate([W, b[:, None]], axis=1)
    
    N, p = H.shape
    classes = jnp.unique(labels)
    C = len(classes)
    
    # 1. Compute Class Means (μ_c)
    class_means = []
    pi = []
    for c in range(C):
        mask = labels == c
        N_c = jnp.sum(mask)
        if N_c > 0:
            mu_c = jnp.sum(H[mask], axis=0) / N_c
        else:
            mu_c = jnp.zeros(p)
        pi.append(N_c / N)
        class_means.append(mu_c)
    class_means = jnp.stack(class_means)  # (C, p)
    pi = jnp.array(pi)
    
    # 2. Compute Global Mean (μ_G) weighted by class proportions π_c (paper definition)
    mu_G = jnp.sum(class_means * pi[:, None], axis=0)
    
    # 3. Centered Class Means Matrix M (p, C)
    M = (class_means - mu_G).T  # (p, C)
    
    # =============================================================================
    # NC2: Equinorm (Coefficient of Variation) - Fig 2
    # =============================================================================
    # Class-means norms (blue line in paper)
    norms_means = jnp.linalg.norm(class_means - mu_G, axis=1)  # (C,)
    nc2_equinorm_cv_means = jnp.std(norms_means) / (jnp.mean(norms_means) + 1e-8)
    
    # Classifiers norms (orange line in paper)
    norms_classifiers = jnp.linalg.norm(W, axis=1)  # (C,)
    nc2_equinorm_cv_classifiers = jnp.std(norms_classifiers) / (jnp.mean(norms_classifiers) + 1e-8)
    
    # =============================================================================
    # NC2: Equiangularity - Fig 3 & Fig 4
    # =============================================================================
    # Normalize class-means
    M_normalized = M / (jnp.linalg.norm(M, axis=0, keepdims=True) + 1e-8)  # (p, C)
    
    # Normalize classifiers
    W_normalized = W / (jnp.linalg.norm(W, axis=1, keepdims=True) + 1e-8)  # (C, p)
    
    # Gram matrices
    G_means = jnp.dot(M_normalized.T, M_normalized)  # (C, C)
    G_classifiers = jnp.dot(W_normalized, W_normalized.T)  # (C, C)
    
    # Extract off-diagonal elements
    mask_off_diag = ~jnp.eye(C, dtype=bool)
    off_diag_means = G_means[mask_off_diag]
    off_diag_classifiers = G_classifiers[mask_off_diag]
    
    # Fig 3: Std of cosines (blue = means, orange = classifiers)
    nc2_equiangular_std_means = jnp.std(off_diag_means)
    nc2_equiangular_std_classifiers = jnp.std(off_diag_classifiers)
    
    # Fig 4: Mean deviation from target -1/(C-1)
    target_angle = -1.0 / (C - 1) if C > 1 else 0.0
    nc2_equiangular_mean_means = jnp.mean(jnp.abs(off_diag_means - target_angle))
    nc2_equiangular_mean_classifiers = jnp.mean(jnp.abs(off_diag_classifiers - target_angle))
    
    # =============================================================================
    # NC3: Self-Duality - Fig 5 (Paper's definition)
    # =============================================================================
    # Double centering M for numerical stability with class imbalance
    # Center across classes (already done) and across features
    M_centered = M - jnp.mean(M, axis=1, keepdims=True)
    M_hat = M_centered / (jnp.linalg.norm(M_centered, 'fro') + 1e-8)
    
    # Center W: subtract mean across classes (Papyan's implicit assumption)
    # W̃ = W - (1/C) * 1 * 1^T * W = W - mean(W, axis=0)
    W_centered = W - jnp.mean(W, axis=0, keepdims=True)
    
    # Normalize W^T by Frobenius norm (W is (C, p), so W.T is (p, C))
    W_T = W_centered.T
    W_hat = W_T / (jnp.linalg.norm(W_T, 'fro') + 1e-8)
    
    # NC3 Metric: ||W^T - M||_F^2 (paper's definition)
    nc3_self_duality = jnp.linalg.norm(W_hat - M_hat, 'fro')
    
    # =============================================================================
    # NC1: Variability Collapse (Paper-Exact Definition) - Fig 6
    # =============================================================================
    # Paper formulas:
    # W = Ave_{i,c} {(h_{i,c} - μ_c)(h_{i,c} - μ_c)^T}
    # B = Ave_c {(μ_c - μ_G)(μ_c - μ_G)^T}
    # NC1 = Tr{W @ B^†} / C
    
    # Within-Class Covariance W: average over ALL samples
    Sigma_W = jnp.zeros((p, p))
    for c in range(C):
        mask = labels == c
        if jnp.sum(mask) > 0:
            H_c = H[mask]  # Features for class c (N_c, p)
            centered = H_c - class_means[c]  # (N_c, p)
            Sigma_W += jnp.dot(centered.T, centered)  # (p, p)
    Sigma_W = Sigma_W / N  # Ave_{i,c}
    
    # Between-Class Covariance B: weighted average over classes (weighted by π_c)
    # Paper definition: Σ_B = Σ_c π_c (μ_c - μ_G)(μ_c - μ_G)^T
    Sigma_B = jnp.zeros((p, p))
    for c in range(C):
        centered_mean = class_means[c] - mu_G  # (p,)
        Sigma_B += pi[c] * jnp.outer(centered_mean, centered_mean)  # (p, p)
    # Already weighted by π_c in the loop, no need to divide by C
    
    # NC1 Metric: Tr{W @ B^†} / C using Moore-Penrose pseudoinverse
    B_pinv = jnp.linalg.pinv(Sigma_B)
    nc1_variability = jnp.trace(Sigma_W @ B_pinv) / C
    
    # =============================================================================
    # NC4: Nearest Class-Center (NCC) Mismatch - Fig 7
    # =============================================================================
    # Proportion of test examples where classifier disagrees with NCC decision
    # Following Papyan's definition: use centered features and inner products
    if H_test is not None and labels_test is not None:
        # Classifier predictions: arg max_c (W @ h_test)
        # Note: If bias was present, it's already absorbed in augmented W and H_test
        logits_test = jnp.dot(H_test, W.T)  # (N_test, C)
        classifier_predictions = jnp.argmax(logits_test, axis=1)  # (N_test,)
        
        # NCC predictions: arg max_c ⟨h - μ_G, μ_c - μ_G⟩ (centered inner product)
        # This is theoretically equivalent to NCC and numerically more stable
        # than distance-based NCC, especially when NC2 is not fully collapsed.
        # In Neural Collapse terminal phase: classifier decision ≈ NCC decision
        h_centered = H_test - mu_G  # (N_test, p) - center test features
        means_centered = class_means - mu_G  # (C, p) - center class means
        
        # Compute inner products: (N_test, p) @ (p, C) = (N_test, C)
        scores = jnp.dot(h_centered, means_centered.T)  # (N_test, C)
        ncc_predictions = jnp.argmax(scores, axis=1)  # (N_test,)
        
        # Proportion of disagreements
        disagreements = jnp.sum(classifier_predictions != ncc_predictions)
        nc4_ncc_mismatch = float(disagreements / H_test.shape[0])
    else:
        nc4_ncc_mismatch = None
    
    result = {
        # Fig 2: NC2 - Equinorm (blue = means, orange = classifiers)
        'nc2_equinorm_cv_means': float(nc2_equinorm_cv_means),
        'nc2_equinorm_cv_classifiers': float(nc2_equinorm_cv_classifiers),
        # Fig 3: NC2 - Equiangularity Std (blue = means, orange = classifiers)
        'nc2_equiangular_std_means': float(nc2_equiangular_std_means),
        'nc2_equiangular_std_classifiers': float(nc2_equiangular_std_classifiers),
        # Fig 4: NC2 - Equiangularity Mean (blue = means, orange = classifiers)
        'nc2_equiangular_mean_means': float(nc2_equiangular_mean_means),
        'nc2_equiangular_mean_classifiers': float(nc2_equiangular_mean_classifiers),
        'nc2_equiangular_target': float(-1.0 / (C - 1)) if C > 1 else 0.0,
        # Fig 5: NC3 - Self-Duality
        'nc3_self_duality': float(nc3_self_duality),
        # Fig 6: NC1 - Variability Collapse
        'nc1_variability': float(nc1_variability),
    }
    
    # Fig 7: NC4 - Nearest Class-Center Mismatch (only if test data provided)
    if nc4_ncc_mismatch is not None:
        result['nc7_ncc_mismatch'] = nc4_ncc_mismatch  # Keep old key for backward compatibility
    
    return result


def get_features_and_weights(params: Any, X: jnp.ndarray, num_hidden_layers: int = 1, use_batchnorm: bool = True, use_bias: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Extract last-layer features H, classifier weights W, and biases b from model.
    
    Works with Neural Collapse compliant architecture:
    - With BatchNorm: x → [Dense → BatchNorm → ReLU] × num_hidden_layers → h(x) → Dense → logits
    - Without BatchNorm: x → [Dense → ReLU] × num_hidden_layers → h(x) → Dense → logits
    
    Args:
        params: Model parameters (list of layer params)
        X: Input data (N, input_dim)
        num_hidden_layers: Number of hidden layer blocks (default: 1)
        use_batchnorm: Whether the model uses BatchNorm (default: True)
        use_bias: Whether the CLASSIFIER uses bias (default: True)
                  Note: Hidden layers always use bias (standard architecture)
        
    Returns:
        Tuple of (H, W, b):
            - H: Last-layer features (N, p)
            - W: Classifier weights (C, p)
            - b: Classifier biases (C,) if use_bias=True, else None
    """
    # Extract features from all hidden layer blocks
    h = X
    
    if use_batchnorm:
        # Process each Dense → BatchNorm → ReLU block
        # NOTE: In JAX stax params list, each LAYER is ONE entry:
        #   - Dense layer = 1 entry: (W, b) tuple
        #   - BatchNorm layer = 1 entry: (gamma, beta, mean, var) tuple
        # So each block has 2 entries total (Dense + BatchNorm)
        for block_idx in range(num_hidden_layers):
            param_idx = block_idx * 2  # 2 entries per block
            
            # Dense layer - params[param_idx] is a tuple (W, b)
            # Hidden layers ALWAYS have bias (standard architecture)
            W_dense, b_dense = params[param_idx]
            h = jnp.dot(h, W_dense) + b_dense
            
            # BatchNorm layer - params[param_idx + 1] is a tuple (gamma, beta, mean, var)
            bn_params = params[param_idx + 1]
            if len(bn_params) == 4:
                gamma, beta, running_mean, running_var = bn_params
                h = gamma * (h - running_mean) / jnp.sqrt(running_var + 1e-5) + beta
            
            # ReLU activation
            h = jnp.maximum(0, h)
        
        # Final classifier layer index
        # JAX stax creates empty tuples for ReLU, so: Dense + BatchNorm + ReLU = 3 entries
        classifier_idx = num_hidden_layers * 3
    else:
        # Process each Dense → ReLU block (no BatchNorm)
        for block_idx in range(num_hidden_layers):
            # Dense layer - hidden layers ALWAYS have bias
            W_dense, b_dense = params[block_idx * 2]  # Skip ReLU empty tuples
            h = jnp.dot(h, W_dense) + b_dense
            
            # ReLU activation
            h = jnp.maximum(0, h)
        
        # Final classifier layer index
        # JAX stax creates empty tuples for ReLU, so: Dense + ReLU = 2 entries
        classifier_idx = num_hidden_layers * 2
    
    H = h  # Last-layer features
    
    # Extract classifier weights (final Dense layer)
    # use_bias flag applies ONLY to the classifier, not hidden layers
    try:
        classifier_params = params[classifier_idx]
        if use_bias:
            W_last, b_last = classifier_params
            W = W_last.T  # (C, p)
            b = b_last    # (C,)
        else:
            # No bias in classifier
            W_last = classifier_params[0] if isinstance(classifier_params, tuple) else classifier_params
            W = W_last.T  # (C, p)
            b = None
    except (ValueError, IndexError) as e:
        # Debug info for parameter indexing issues
        import logging
        logging.error(f"Error extracting classifier weights:")
        logging.error(f"  use_batchnorm={use_batchnorm}")
        logging.error(f"  use_bias={use_bias}")
        logging.error(f"  num_hidden_layers={num_hidden_layers}")
        logging.error(f"  classifier_idx={classifier_idx}")
        logging.error(f"  len(params)={len(params)}")
        logging.error(f"  params structure: {[type(p).__name__ if not isinstance(p, tuple) else f'tuple(len={len(p)})' for p in params]}")
        for i, p in enumerate(params):
            if isinstance(p, tuple):
                logging.error(f"    params[{i}] = tuple with {len(p)} elements")
                if len(p) == 2:
                    logging.error(f"      shapes: ({p[0].shape}, {p[1].shape})")
            else:
                logging.error(f"    params[{i}] = {type(p).__name__}")
        raise
    
    return H, W, b


# =============================================================================
# SNAPSHOT DATA CLASS
# =============================================================================

@dataclass
class NeuralCollapseSnapshot:
    """
    Snapshot of network state for Neural Collapse analysis.
    
    Attributes:
        epoch: Training epoch number
        features: Last-layer features (N, p) where N is number of samples, p is feature dim
        labels: Class labels (N,) as integers
        class_means: Class mean vectors (C, p) where C is number of classes
        classifiers: Classifier weight vectors (C, p) from last layer
        biases: Bias vectors (C,) from last layer (b in W*h+b)
        num_classes: Number of classes
        feature_dim: Feature dimension p
        metrics: Dictionary of NC metrics computed in original R^p space
    """
    epoch: int
    features: jnp.ndarray
    labels: jnp.ndarray
    class_means: jnp.ndarray
    classifiers: jnp.ndarray
    biases: jnp.ndarray
    num_classes: int
    feature_dim: int
    metrics: Optional[Dict[str, float]] = None


# =============================================================================
# NEURAL COLLAPSE ANALYZER (Simplified)
# =============================================================================

class NeuralCollapseAnalyzer:
    """
    Simplified Neural Collapse analyzer using paper's metric definitions.
    """
    
    def __init__(self, num_classes: int, feature_dim: int, num_hidden_layers: int = 1, use_batchnorm: bool = True, use_bias: bool = True):
        """
        Initialize Neural Collapse analyzer.
        
        Args:
            num_classes: Number of classes in the classification task
            feature_dim: Dimension of last-layer features
            num_hidden_layers: Number of Dense → [BatchNorm] → ReLU blocks (default: 1)
            use_batchnorm: Whether the model uses BatchNorm layers (default: True)
            use_bias: Whether the CLASSIFIER uses bias (default: True)
                      Note: Hidden layers always use bias (standard architecture)
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_hidden_layers = num_hidden_layers
        self.use_batchnorm = use_batchnorm
        self.use_bias = use_bias
        self.snapshots: List[NeuralCollapseSnapshot] = []
        
    def extract_features_and_classifiers(
        self,
        model_fn: Any,
        params: Any,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        epoch: int,
        X_test: Optional[jnp.ndarray] = None,
        Y_test: Optional[jnp.ndarray] = None
    ) -> NeuralCollapseSnapshot:
        """
        Extract features, classifiers and compute NC metrics.
        
        Args:
            model_fn: Model function (not used, kept for compatibility)
            params: Current model parameters
            X: Input data (N, input_dim)
            Y: One-hot labels (N, num_classes)
            epoch: Current training epoch
            X_test: Test input data (N_test, input_dim) for NC7 computation (optional)
            Y_test: Test one-hot labels (N_test, num_classes) for NC7 computation (optional)
            
        Returns:
            NeuralCollapseSnapshot containing extracted information and metrics
        """
        # Convert one-hot labels to class indices
        labels = jnp.argmax(Y, axis=1)
        
        # Extract features and weights using the new architecture
        features, classifiers, biases = get_features_and_weights(params, X, self.num_hidden_layers, self.use_batchnorm, self.use_bias)
        
        # Extract test features if provided
        H_test = None
        labels_test = None
        if X_test is not None and Y_test is not None:
            labels_test = jnp.argmax(Y_test, axis=1)
            H_test, _, _ = get_features_and_weights(params, X_test, self.num_hidden_layers, self.use_batchnorm, self.use_bias)
        
        # Compute class means
        class_means = []
        for c in range(self.num_classes):
            mask = labels == c
            if jnp.sum(mask) > 0:
                class_mean = jnp.mean(features[mask], axis=0)
            else:
                class_mean = jnp.zeros(self.feature_dim)
            class_means.append(class_mean)
        class_means = jnp.stack(class_means)
        
        # Extract biases from final classifier layer
        # Calculate classifier index based on architecture
        # In JAX stax, each LAYER creates ONE entry (including empty tuples for ReLU):
        #   - Dense = 1 entry: (W, b)
        #   - BatchNorm = 1 entry: (gamma, beta, mean, var)
        #   - ReLU = 1 entry: () empty tuple
        if self.use_batchnorm:
            classifier_idx = self.num_hidden_layers * 3  # Dense + BatchNorm + ReLU
        else:
            classifier_idx = self.num_hidden_layers * 2  # Dense + ReLU
        # biases already extracted in get_features_and_weights
        
        # Compute NC metrics in original R^p space (including NC7 if test data provided)
        # Pass biases to enable bias-agnostic computation via feature augmentation
        metrics = compute_nc_metrics(features, labels, classifiers, biases, H_test, labels_test)
        
        snapshot = NeuralCollapseSnapshot(
            epoch=epoch,
            features=features,
            labels=labels,
            class_means=class_means,
            classifiers=classifiers,
            biases=biases,
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            metrics=metrics
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def compute_nc_metrics(self, snapshot: NeuralCollapseSnapshot) -> Dict[str, float]:
        """
        Compute NC metrics for a snapshot (uses metrics already computed).
        
        Args:
            snapshot: Network state snapshot
            
        Returns:
            Dictionary of NC metrics
        """
        if snapshot.metrics is not None:
            return snapshot.metrics
        
        # Recompute if not available (pass biases for bias-agnostic computation)
        return compute_nc_metrics(snapshot.features, snapshot.labels, snapshot.classifiers, snapshot.biases)
    
    def save_snapshots(self, filepath: Path):
        """Save snapshots to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.snapshots, f)
        logging.info(f"Saved {len(self.snapshots)} snapshots to {filepath}")
    
    def load_snapshots(self, filepath: Path):
        """Load snapshots from file."""
        import pickle
        with open(filepath, 'rb') as f:
            self.snapshots = pickle.load(f)
        logging.info(f"Loaded {len(self.snapshots)} snapshots from {filepath}")
    
    def compute_simplex_etf(self, num_classes=None):
        """
        Compute simplex ETF (stub for compatibility).
        
        In simplified version, this is not needed for metrics.
        Returns identity matrix for compatibility.
        """
        if num_classes is None:
            num_classes = self.num_classes
        
        C = num_classes
        # Simple identity-based placeholder
        identity = jnp.eye(C)
        centered = identity - jnp.ones((C, C)) / C
        U, S, Vt = jnp.linalg.svd(centered, full_matrices=False)
        etf = U[:, :C-1] * jnp.sqrt(C / (C - 1))
        return etf
    
    def visualize_neural_collapse(self, snapshot, selected_classes=None, samples_per_class=50, 
                                  save_path=None, title=None, elevation=20, azimuth=45, 
                                  axis_limit=None, **kwargs):
        """
        Visualize Neural Collapse (stub for simplified version).
        
        In the simplified version, we focus on metrics rather than visualizations.
        """
        logging.info(f"Skipping 3D visualization for epoch {snapshot.epoch} (simplified NC version)")
    
    def visualize_neural_collapse_2d(self, snapshot, selected_classes=None, samples_per_class=50,
                                     save_path=None, title=None, axis_limit=None, **kwargs):
        """
        Visualize Neural Collapse in 2D (stub for simplified version).
        
        In the simplified version, we focus on metrics rather than visualizations.
        """
        logging.info(f"Skipping 2D visualization for epoch {snapshot.epoch} (simplified NC version)")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_classifier_mean_norms(snapshots: List[NeuralCollapseSnapshot], output_dir: Path):
    """
    Sanity check plot: ||W_c - μ_c|| across classes and epochs.
    
    In Neural Collapse terminal phase, classifier rows W_c should align with
    class means μ_c, so this norm should collapse toward 0.
    
    Args:
        snapshots: List of NeuralCollapseSnapshot objects from training
        output_dir: Directory to save plot
    """
    if not snapshots:
        print("No snapshots to plot")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect ||W_c - μ_c|| for each epoch and class
    epochs = []
    norms_by_class = {}
    
    num_classes = snapshots[0].num_classes
    for c in range(num_classes):
        norms_by_class[c] = []
    
    for snapshot in snapshots:
        epochs.append(snapshot.epoch)
        classifiers = snapshot.classifiers  # (C, p)
        class_means = snapshot.class_means  # (C, p)
        
        for c in range(num_classes):
            diff = classifiers[c] - class_means[c]  # (p,)
            norm = float(jnp.linalg.norm(diff))
            norms_by_class[c].append(norm)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use different colors for each class
    colors = plt.cm.tab10(jnp.linspace(0, 1, num_classes))
    
    for c in range(num_classes):
        ax.plot(epochs, norms_by_class[c], 'o-', linewidth=2, markersize=5,
                color=colors[c], label=f'Class {c}')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('||W_c - μ_c||', fontsize=12, fontweight='bold')
    ax.set_title('Sanity Check: Classifier-Mean Alignment\nNC3 requires these norms to collapse → 0',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
    ax.legend(fontsize=10, ncol=2, loc='upper right')
    ax.set_yscale('log')  # Log scale to see collapse clearly
    
    save_path = output_dir / 'nc_sanity_check_classifier_mean_norms.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved classifier-mean alignment sanity check to {save_path}")
    plt.close()

def plot_nc_metrics(metrics_history: List[Tuple[int, dict]], output_dir: Path, log_scale: bool = True, step_100_acc: Optional[int] = None, tpt_threshold: float = 1.0):
    """
    Plot Neural Collapse metrics evolution over training.
    
    Args:
        metrics_history: List of (epoch, metrics_dict) tuples
        output_dir: Directory to save plots
        log_scale: Use log scale for y-axis where appropriate
        step_100_acc: If provided, add vertical line at step where TPT accuracy was reached
        tpt_threshold: TPT accuracy threshold used (for label, e.g., 1.0 = 100%, 0.99 = 99%)
    """
    if not metrics_history:
        print("No metrics to plot")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = [m[0] for m in metrics_history]
    # Fig 2: Equinorm (blue = means, orange = classifiers)
    nc2_cv_means = [m[1]['nc2_equinorm_cv_means'] for m in metrics_history]
    nc2_cv_classifiers = [m[1]['nc2_equinorm_cv_classifiers'] for m in metrics_history]
    # Fig 3: Equiangularity Std (blue = means, orange = classifiers)
    nc2_std_means = [m[1]['nc2_equiangular_std_means'] for m in metrics_history]
    nc2_std_classifiers = [m[1]['nc2_equiangular_std_classifiers'] for m in metrics_history]
    # Fig 4: Equiangularity Mean (blue = means, orange = classifiers)
    nc2_mean_means = [m[1]['nc2_equiangular_mean_means'] for m in metrics_history]
    nc2_mean_classifiers = [m[1]['nc2_equiangular_mean_classifiers'] for m in metrics_history]
    nc2_target = metrics_history[0][1]['nc2_equiangular_target']
    # Fig 5: Self-Duality
    nc3 = [m[1]['nc3_self_duality'] for m in metrics_history]
    # Fig 6: Variability Collapse
    nc1 = [m[1]['nc1_variability'] for m in metrics_history]
    # Fig 7: NCC Mismatch (if available)
    nc4 = [m[1].get('nc7_ncc_mismatch', None) for m in metrics_history]  # Use old key for compatibility
    has_nc4 = all(v is not None for v in nc4)
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Figure 2: NC2 - Equinorm (CV) - BOTH means and classifiers
    ax = axes[0, 0]
    ax.plot(epochs, nc2_cv_means, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Class-means')
    ax.plot(epochs, nc2_cv_classifiers, 's-', linewidth=2, markersize=6, color='#F18F01', label='Classifiers')
    # Linear scale for better visibility near 0
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: NC2 - Equinorm\nStd(||μ_c - μ_G||) / Avg(||μ_c - μ_G||)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
    # Set y-axis limits: fixed range -0.02 to 0.2
    ax.set_ylim(bottom=-0.02, top=0.2)
    ax.legend(fontsize=10)
    
    # Figure 3: NC2 - Equiangularity (Std) - BOTH means and classifiers
    ax = axes[0, 1]
    ax.plot(epochs, nc2_std_means, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Class-means')
    ax.plot(epochs, nc2_std_classifiers, 's-', linewidth=2, markersize=6, color='#F18F01', label='Classifiers')
    # Linear scale for better visibility near 0
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Std of Cosines', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: NC2 - Equiangularity (Std)\nStd(cos(c,c\'))', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
    # Set y-axis limits: fixed range -0.02 to 0.6
    ax.set_ylim(bottom=-0.02, top=0.6)
    ax.legend(fontsize=10)
    
    # Figure 4: NC2 - Equiangularity (Mean) - BOTH means and classifiers
    ax = axes[0, 2]
    ax.plot(epochs, nc2_mean_means, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Class-means')
    ax.plot(epochs, nc2_mean_classifiers, 's-', linewidth=2, markersize=6, color='#F18F01', label='Classifiers')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg |cos + 1/(C-1)|', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure 4: NC2 - Equiangularity (Mean)\nAvg|cos(c,c\') + 1/(C-1)|', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
               label=f'Target: 0')
    # Set y-axis limits: fixed range -0.02 to 0.6
    ax.set_ylim(bottom=-0.02, top=0.6)
    ax.legend(fontsize=10)
    
    # Figure 5: NC3 - Self-Duality
    ax = axes[1, 0]
    ax.plot(epochs, nc3, 'o-', linewidth=2, markersize=6, color='#D62246')
    # Linear scale for better visibility near 0
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('NC3 Metric', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5: NC3 - Self-Duality\n||Ŵ^T - M̂||_F', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
    # Set y-axis limits: fixed range -0.02 to 1.5
    ax.set_ylim(bottom=-0.02, top=1.5)
    ax.legend()
    
    # Figure 6: NC1 - Variability Collapse
    ax = axes[1, 1]
    ax.plot(epochs, nc1, 'o-', linewidth=2, markersize=6, color='#2E86AB', label='Within-Class Variation')
    # Always use log scale for NC1 to see collapse clearly
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tr{W @ B^†} / C (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 6: NC1 - Variability Collapse\nTr{W @ B^†} / C', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    # Set y-axis limits: fixed log range 20 to 45
    ax.set_ylim(bottom=20, top=45)
    ax.legend()
    
    # Figure 7: NC4 - Nearest Class-Center Mismatch (if available)
    ax = axes[1, 2]
    if has_nc4:
        ax.plot(epochs, nc4, 'o-', linewidth=2, markersize=6, color='#A23B72')
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion of Disagreements', fontsize=12, fontweight='bold')
        ax.set_title('Figure 7: Classifier → NCC\nProportion where Classifier ≠ arg min||h-μ_c||', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Target: 0')
        # Set y-axis limits: fixed range -0.02 to 1
        ax.set_ylim(bottom=-0.02, top=1)
        ax.legend()
    else:
        # Hide subplot if NC4 not available
        ax.axis('off')
        ax.text(0.5, 0.5, 'NC4: Nearest Class-Center\n(requires test data)', 
                ha='center', va='center', fontsize=11, color='gray')
    
    # Add vertical line at TPT accuracy threshold step (Terminal Phase Training)
    if step_100_acc is not None:
        tpt_label = f'{tpt_threshold*100:.0f}% Train Acc' if tpt_threshold < 1.0 else '100% Train Acc'
        for ax in axes.flat:
            if ax.axison:  # Only add to active subplots
                ax.axvline(x=step_100_acc, color='black', linestyle='-', alpha=0.8, linewidth=2)
        # Add text annotation to the first subplot
        axes[0, 0].text(step_100_acc, axes[0, 0].get_ylim()[1] * 0.9, 
                       tpt_label, rotation=90, verticalalignment='top',
                       fontsize=10, color='black', fontweight='bold')
    
    plt.suptitle('Neural Collapse Metrics Evolution', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = output_dir / 'nc_metrics_paper_style.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Neural Collapse metrics plot to {save_path}")
    plt.close()