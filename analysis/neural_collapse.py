"""
Neural Collapse Analysis Module

This module implements the analysis and visualization of Neural Collapse phenomena
as described in "Prevalence of Neural Collapse during the terminal phase of deep 
learning training" (Papyan et al., 2020).

Neural Collapse refers to the phenomenon where, during the terminal phase of training:
1. NC1 (Variability Collapse): Within-class features collapse to their class means
2. NC2 (Convergence to Simplex ETF): Class means converge to vertices of a Simplex 
   Equiangular Tight Frame
3. NC3 (Self-Duality): Classifiers and class means become dual to each other
4. NC4 (Simplification to NCC): Decision boundaries simplify to nearest class center

🔴 CRITICAL MATHEMATICAL PRINCIPLES 🔴
This implementation follows strict geometric principles to avoid measurement artifacts:

1️⃣ Rule 1 — Never measure geometry after adaptive projection
   ✅ All NC metrics computed in ORIGINAL feature space R^p
   ✅ Projection used ONLY for visualization
   ❌ Never compute angles/distances on projected data

2️⃣ Rule 2 — ETF is a target, not a coordinate system  
   ✅ Fixed theoretical Simplex ETF for comparison
   ✅ Rotation-invariant alignment via Procrustes analysis
   ❌ Never use ETF to define projection axes

3️⃣ Rule 3 — Normalization only after centering
   ✅ Center class means first: μ_c ← μ_c - mean(μ)
   ✅ Then normalize: μ_c ← μ_c / ||μ_c||
   ❌ Never: project → normalize → measure

This module provides tools to:
- Extract last-layer features, class means, and classifiers during training
- Compute NC metrics in original R^p space (mathematically correct)
- Project to low-dimensional subspace for visualization only
- Recreate Figure 1 from the Neural Collapse paper with correct geometry

Author: Samuel Lozano Iglesias  
Email: samuel.lozano@ucm.es
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.plugins.ffmpeg
from matplotlib import cm
import pickle
from pathlib import Path
from dataclasses import dataclass
import imageio
import logging
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
from pathlib import Path
from dataclasses import dataclass


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
    """
    epoch: int
    features: jnp.ndarray
    labels: jnp.ndarray
    class_means: jnp.ndarray
    classifiers: jnp.ndarray
    biases: jnp.ndarray
    num_classes: int
    feature_dim: int


class NeuralCollapseAnalyzer:
    """
    Analyzer for Neural Collapse phenomena in neural networks.
    
    This class provides methods to:
    1. Extract and store network states during training
    2. Compute Neural Collapse metrics
    3. Generate visualizations similar to Figure 1 in the Neural Collapse paper
    """
    
    def __init__(self, num_classes: int, feature_dim: int):
        """
        Initialize Neural Collapse analyzer.
        
        Args:
            num_classes: Number of classes in the classification task
            feature_dim: Dimension of last-layer features
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.snapshots: List[NeuralCollapseSnapshot] = []
        
    def extract_features_and_classifiers(
        self,
        model_fn: Any,
        params: Any,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        epoch: int
    ) -> NeuralCollapseSnapshot:
        """
        Extract last-layer features, class means, and classifiers from the network.
        
        Args:
            model_fn: Model function that returns both predictions and last-layer features
            params: Current model parameters
            X: Input data (N, input_dim)
            Y: One-hot labels (N, num_classes)
            epoch: Current training epoch
            
        Returns:
            NeuralCollapseSnapshot containing extracted information
        """
        # Convert one-hot labels to class indices
        labels = jnp.argmax(Y, axis=1)
        
        # Extract last-layer features
        # Assuming model_fn can return features when requested
        # We'll need to modify the model to expose features
        features = self._extract_features(model_fn, params, X)

        # Compute global means of features (average of all feature vectors)
        global_mean = jnp.mean(features, axis=0)
        
        # Compute class means (average feature vector per class)
        class_means = self._compute_class_means(features, labels)

        # Compute centered means
        centered_means = class_means - global_mean
        
        # Extract classifier weights (last layer weights) and biases
        classifiers, biases = self._extract_classifiers(params)
        
        snapshot = NeuralCollapseSnapshot(
            epoch=epoch,
            features=features,
            labels=labels,
            class_means=class_means,
            classifiers=classifiers,
            biases=biases,
            num_classes=self.num_classes,
            feature_dim=self.feature_dim
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _extract_features(
        self,
        model_fn: Any,
        params: Any,
        X: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Extract last-layer features (activations before final linear layer).
        
        For SpiralClassifier architecture: Dense -> ReLU -> Dense
        This extracts features after the first Dense+ReLU.
        """
        # Apply first layer: Dense + ReLU (same as extract_penultimate_features)
        W1, b1 = params[0]
        features = jnp.maximum(0, jnp.dot(X, W1) + b1)
        return features
    
    def _compute_class_means(
        self,
        features: jnp.ndarray,
        labels: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute class mean vectors (centroids) for each class.
        
        Args:
            features: Feature vectors (N, p)
            labels: Class labels (N,)
            
        Returns:
            Class means (C, p)
        """
        class_means = []
        for c in range(self.num_classes):
            mask = labels == c
            if jnp.sum(mask) > 0:
                class_mean = jnp.mean(features[mask], axis=0)
            else:
                class_mean = jnp.zeros(self.feature_dim)
            class_means.append(class_mean)
        
        return jnp.stack(class_means)
    
    def _extract_classifiers(self, params: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extract classifier weight vectors and biases from the last layer.
        
        Args:
            params: Model parameters
            
        Returns:
            Tuple of (classifiers, biases):
                - classifiers: Weight matrix (C, p) where each row is a class weight vector
                - biases: Bias vector (C,) where each element is the bias for that class
        """
        # For JAX stax models, params is a list of (W, b) tuples
        # Last layer weights are params[-1][0], shape (p, C)
        # Last layer biases are params[-1][1], shape (C,)
        last_layer_weights = params[-1][0]
        last_layer_biases = params[-1][1]
        
        # Transpose weights to (C, p) for easier handling
        classifiers = last_layer_weights.T
        
        return classifiers, last_layer_biases
    
    def compute_simplex_etf(self, num_classes: Optional[int] = None) -> jnp.ndarray:
        """
        Compute the Simplex Equiangular Tight Frame (ETF).
        
        The Simplex ETF is the theoretical optimal configuration for class means
        in Neural Collapse. It consists of C vectors in R^(C-1) that are:
        - Unit norm
        - Equidistant from each other
        - Centered at origin
        
        The inner product between any two different vectors is -1/(C-1).
        
        Args:
            num_classes: Number of classes (uses self.num_classes if None)
            
        Returns:
            Simplex ETF vectors (C, C-1)
        """
        if num_classes is None:
            num_classes = self.num_classes
        
        C = num_classes
        
        # Create the Simplex ETF in R^(C-1)
        # We construct it using the standard simplex and centering
        
        # Start with standard basis vectors in R^C
        identity = jnp.eye(C)
        
        # Center them (subtract mean)
        centered = identity - jnp.ones((C, C)) / C
        
        # Remove the zero eigenvalue direction (project to R^(C-1))
        # Use SVD to get the top (C-1) components
        U, S, Vt = jnp.linalg.svd(centered, full_matrices=False)
        
        # Take the top (C-1) singular vectors
        etf = U[:, :C-1] * jnp.sqrt(C / (C - 1))
        
        return etf
    
    def project_to_subspace(
        self,
        features: jnp.ndarray,
        class_means: jnp.ndarray,
        classifiers: jnp.ndarray,
        target_dim: int = 3,
        normalize: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Project features, class means, and classifiers to low-dimensional subspace.
        
        ⚠️  VISUALIZATION ONLY - NEVER COMPUTE METRICS ON PROJECTED DATA ⚠️
        
        This function is PURELY for visualization purposes. All Neural Collapse
        metrics MUST be computed in the original feature space R^p before calling
        this function. Projection creates visual artifacts that would make metrics
        meaningless.
        
        Uses SVD of class means to find the principal subspace for visualization.
        The projection basis is fixed per snapshot to avoid adaptive artifacts.
        
        Args:
            features: Feature vectors (N, p)
            class_means: Class mean vectors (C, p)
            classifiers: Classifier weight vectors (C, p)
            target_dim: Target dimension for visualization (default: 3 for 3D plots)
            normalize: If True, normalize projected vectors for better visualization
            
        Returns:
            Tuple of (projected_features, projected_means, projected_classifiers, etf, basis)
            where all outputs are for VISUALIZATION ONLY:
            - projected_features: (N, target_dim)
            - projected_means: (C, target_dim)
            - projected_classifiers: (C, target_dim)
            - etf: (C, target_dim) - Theoretical ETF for visual reference
            - basis: (p, target_dim) - projection matrix
        """
        # Center the class means by subtracting their global centroid
        # This is crucial for PCA to work correctly and avoid all points
        # appearing on one side of the principal components
        global_mean = jnp.mean(class_means, axis=0, keepdims=True)
        centered_class_means = class_means - global_mean
        
        # Compute SVD of centered class means
        # This finds the subspace spanned by the class centroids
        U, S, Vt = jnp.linalg.svd(centered_class_means, full_matrices=False)
        
        # The top (C-1) or target_dim components span the relevant subspace
        k = min(target_dim, self.num_classes - 1, class_means.shape[1])
        basis = Vt[:k, :].T  # (p, k) - projection matrix
        
        # Center all data by the same global mean before projection
        centered_features = features - global_mean
        centered_classifiers = classifiers - global_mean
        
        # Project everything onto this subspace
        projected_features = centered_features @ basis  # (N, k)
        projected_means = centered_class_means @ basis  # (C, k)
        projected_classifiers = centered_classifiers @ basis  # (C, k)
        
        # Compute Simplex ETF directly in the target dimension
        # This is the FIXED theoretical optimal configuration
        C = self.num_classes
        identity = jnp.eye(C)
        centered = identity - jnp.ones((C, C)) / C
        U_etf, S_etf, Vt_etf = jnp.linalg.svd(centered, full_matrices=False)
        etf_full = U_etf[:, :C-1] * jnp.sqrt(C / (C - 1))
        
        # Take first k dimensions of ETF
        if k < C - 1:
            etf = etf_full[:, :k]
        else:
            etf = etf_full
        
        # Pad to target_dim if needed
        if etf.shape[1] < k:
            padding = jnp.zeros((C, k - etf.shape[1]))
            etf = jnp.concatenate([etf, padding], axis=1)
        
        # Normalize vectors to unit length for angle visualization
        if normalize:
            # Scale features to fit within unit disk (not normalize to unit circle)
            # Find the maximum norm among all features
            feature_norms = jnp.linalg.norm(projected_features, axis=1, keepdims=True)
            max_feature_norm = jnp.max(feature_norms)
            # Scale features so the largest one touches the unit circle, others inside
            if max_feature_norm > 1e-8:
                projected_features = projected_features / (max_feature_norm + 1e-8)
            
            # Normalize class means to unit length
            mean_norms = jnp.linalg.norm(projected_means, axis=1, keepdims=True)
            projected_means = projected_means / (mean_norms + 1e-8)
            
            # Normalize classifiers to unit length
            classifier_norms = jnp.linalg.norm(projected_classifiers, axis=1, keepdims=True)
            projected_classifiers = projected_classifiers / (classifier_norms + 1e-8)
            
            # ETF is already unit normalized
        
        return projected_features, projected_means, projected_classifiers, etf, basis
    
    def compute_angle_degrees(self, v1: jnp.ndarray, v2: jnp.ndarray) -> float:
        """
        Compute angle in degrees between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Angle in degrees
        """
        norm1 = jnp.linalg.norm(v1)
        norm2 = jnp.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        cos_angle = jnp.dot(v1, v2) / (norm1 * norm2)
        cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
        return float(jnp.arccos(cos_angle) * 180.0 / jnp.pi)
    
    def compute_pairwise_angles(self, vectors: jnp.ndarray, vector_name: str):
        """
        Compute all pairwise angles between a set of vectors.
        
        Args:
            vectors: Array of shape (N, D) containing N vectors
            
        Returns:
            Array of angles in degrees (excluding diagonal), shape (N*(N-1)/2,)
        """
        N = vectors.shape[0]
        angles = []
        pair_names = []
        for i in range(N):
            for j in range(i+1, N):
                angle = self.compute_angle_degrees(vectors[i], vectors[j])
                angles.append(angle)
                pair_names.append(f"{vector_name}_{i}-{vector_name}_{j}")
        
        angles = jnp.array(angles)

        # Theoretical optimal angle for C classes
        if N > 1:
            optimal_cos = -1.0 / (N - 1)
            optimal_angle = float(jnp.arccos(optimal_cos) * 180.0 / jnp.pi)
        else:
            optimal_angle = 0.0
        
        return {
            'mean_angle': float(jnp.mean(angles)) if len(angles) > 0 else 0.0,
            'std_angle': float(jnp.std(angles)) if len(angles) > 0 else 0.0,
            'min_angle': float(jnp.min(angles)) if len(angles) > 0 else 0.0,
            'max_angle': float(jnp.max(angles)) if len(angles) > 0 else 0.0,
            'optimal_angle': optimal_angle,
            'angle_deviation': float(jnp.mean(jnp.abs(angles - optimal_angle))) if len(angles) > 0 else 0.0,
            'all_angles': angles.tolist(),
            'pair_names': pair_names
        }
    
    def compute_all_angles_in_original_space(self, snapshot: NeuralCollapseSnapshot) -> Dict[str, Dict[str, float]]:
        """
        Compute true geometric angles for ALL components in original feature space R^p.
        
        This computes angles for:
        1. Class means (should converge to simplex angles)
        2. Classifiers (should converge to simplex angles) 
        3. Biases (typically anti-aligned, ~180° apart)
        4. Alignment between class means and classifiers (should → 0°)
        
        For perfect Neural Collapse simplex configuration:
        - 3 classes: 120° between all pairs
        - 4 classes: ~109.47° (tetrahedral angles)
        - General C classes: arccos(-1/(C-1))
        
        Args:
            snapshot: Network state snapshot
            
        Returns:
            Dictionary with angle statistics for each component type
        """        
        # 1. Class means angles
        class_means_angles = self.compute_pairwise_angles(snapshot.class_means, "mean")
        
        # 2. Classifiers angles  
        classifiers_angles = self.compute_pairwise_angles(snapshot.classifiers, "classifier")
        
        # 3. Bias angles (note: biases are 1D, so we need to embed them in feature space)
        # For neural collapse, biases tend to anti-align (opposite directions)
        # We'll treat each bias as a vector pointing in the direction of its classifier
        bias_vectors = []
        for i in range(snapshot.num_classes):
            # Bias direction aligned with classifier direction but scaled by bias magnitude
            classifier_dir = snapshot.classifiers[i] / (jnp.linalg.norm(snapshot.classifiers[i]) + 1e-8)
            bias_vector = classifier_dir * snapshot.biases[i]
            bias_vectors.append(bias_vector)
        bias_vectors = jnp.stack(bias_vectors)
        
        bias_angles = self.compute_pairwise_angles(bias_vectors, "bias")
        
        # 4. Alignment between class means and classifiers (should be 0° for perfect NC)
        centered_means = snapshot.class_means - jnp.mean(snapshot.class_means, axis=0, keepdims=True)
        centered_classifiers = snapshot.classifiers - jnp.mean(snapshot.classifiers, axis=0, keepdims=True)
        
        # Normalize both
        mean_norms = jnp.linalg.norm(centered_means, axis=1, keepdims=True)
        classifier_norms = jnp.linalg.norm(centered_classifiers, axis=1, keepdims=True)
        
        normalized_means = centered_means / (mean_norms + 1e-8)
        normalized_classifiers = centered_classifiers / (classifier_norms + 1e-8)
        
        # Compute alignment angles (same class mean vs same class classifier)
        alignment_angles = []
        alignment_names = []
        for i in range(snapshot.num_classes):
            cos_angle = jnp.dot(normalized_means[i], normalized_classifiers[i])
            cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
            angle_deg = float(jnp.arccos(cos_angle) * 180.0 / jnp.pi)
            alignment_angles.append(angle_deg)
            alignment_names.append(f"mean_{i}-classifier_{i}")
        
        alignment_angles = jnp.array(alignment_angles)
        
        alignment_stats = {
            'mean_angle': float(jnp.mean(alignment_angles)),
            'std_angle': float(jnp.std(alignment_angles)),
            'min_angle': float(jnp.min(alignment_angles)),
            'max_angle': float(jnp.max(alignment_angles)),
            'optimal_angle': 0.0,  # Perfect alignment should be 0°
            'angle_deviation': float(jnp.mean(jnp.abs(alignment_angles - 0.0))),
            'all_angles': alignment_angles.tolist(),
            'pair_names': alignment_names
        }
        
        return {
            'class_means': class_means_angles,
            'classifiers': classifiers_angles, 
            'biases': bias_angles,
            'mean_classifier_alignment': alignment_stats
        }
    
    def compute_angles_in_original_space(self, snapshot: NeuralCollapseSnapshot) -> Dict[str, float]:
        """
        Backward compatibility wrapper - returns just class means angles.
        
        For full analysis, use compute_all_angles_in_original_space().
        """
        all_angles = self.compute_all_angles_in_original_space(snapshot)
        return all_angles['class_means']
    
    def compute_nc_metrics(self, snapshot: NeuralCollapseSnapshot) -> Dict[str, float]:
        """
        Compute Neural Collapse metrics for a given snapshot.
        
        Metrics include:
        - NC1: Within-class variance (variability collapse)
        - NC2: Distance to Simplex ETF (convergence to ETF)
        - NC3: Alignment between classifiers and class means (self-duality)
        - NC4: Classification accuracy using nearest class center
        
        Args:
            snapshot: Network state snapshot
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # NC1: Within-class variance
        total_variance = 0.0
        for c in range(snapshot.num_classes):
            mask = snapshot.labels == c
            if jnp.sum(mask) > 0:
                class_features = snapshot.features[mask]
                class_mean = snapshot.class_means[c]
                variance = jnp.mean(jnp.sum((class_features - class_mean) ** 2, axis=1))
                total_variance += variance
        metrics['nc1_within_class_variance'] = float(total_variance / snapshot.num_classes)
        
        # NC2: Distance to Simplex ETF with improved alignment
        # Project class means to (C-1)-dimensional subspace and compute ETF in same space
        C = snapshot.num_classes
        k = C - 1
        _, projected_means, _, etf, _ = self.project_to_subspace(
            snapshot.features, snapshot.class_means, snapshot.classifiers, 
            target_dim=k, normalize=True
        )
        
        # Compute correlation matrix between normalized means and ETF
        corr_matrix = projected_means @ etf.T
        
        # Better alignment metric: Use Frobenius norm after optimal permutation
        # For now, use mean of maximum correlations (greedy but consistent)
        # Note: Proper Procrustes would require Hungarian algorithm for optimal matching
        row_max = jnp.max(corr_matrix, axis=1)
        etf_alignment = float(jnp.mean(row_max))
        
        # Alternative: Use Frobenius norm of best rotation
        # U, _, Vt = jnp.linalg.svd(corr_matrix)
        # R = U @ Vt  # Optimal rotation
        # aligned_means = projected_means @ R.T
        # etf_alignment = float(jnp.mean(jnp.sum(aligned_means * etf, axis=1)))
        
        metrics['nc2_etf_alignment'] = etf_alignment
        
        # NC3: Alignment between classifiers and class means
        # Compute cosine similarity
        classifier_norms = jnp.linalg.norm(snapshot.classifiers, axis=1, keepdims=True)
        mean_norms = jnp.linalg.norm(snapshot.class_means, axis=1, keepdims=True)
        
        normalized_classifiers = snapshot.classifiers / (classifier_norms + 1e-8)
        normalized_class_means = snapshot.class_means / (mean_norms + 1e-8)
        
        # Diagonal elements = alignment for each class
        alignment = jnp.diag(normalized_classifiers @ normalized_class_means.T)
        metrics['nc3_self_duality'] = float(jnp.mean(alignment))
        
        return metrics
    
    def visualize_neural_collapse(
        self,
        snapshot: NeuralCollapseSnapshot,
        selected_classes: Optional[List[int]] = None,
        samples_per_class: int = 50,
        figsize: Tuple[int, int] = (14, 12),
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        elevation: int = 20,
        azimuth: int = 45,
        axis_limit: Optional[float] = None
    ):
        """
        Recreate Figure 1 from the Neural Collapse paper EXACTLY.
        
        Visualizes (matching paper description):
        - Green spheres: Simplex ETF vertices (theoretical optimal configuration)
        - Red ball-and-sticks: Linear classifiers W (one per class)
        - Blue ball-and-sticks: Class-means (centroids of features)
        - Small blue spheres: Last-layer features (individual samples)
        - Orange ball-and-sticks: Bias vectors b (from W*h+b, showing antialignment)
        - Different shades distinguish classes
        
        As training proceeds: features collapse onto class-means (NC1),
        class-means converge to Simplex ETF (NC2), classifiers approach
        class-means (NC3), and biases antialign (separate by 120° for 3 classes).
        
        Args:
            snapshot: Network state to visualize
            selected_classes: Which classes to visualize (None = all, max 3 for clarity)
            samples_per_class: Number of feature samples to show per class
            figsize: Figure size (width, height)
            save_path: Path to save figure (if provided)
            title: Custom title (default: based on epoch)
            elevation: 3D view elevation angle
            azimuth: 3D view azimuth angle
            axis_limit: Fixed axis limit for all plots (if None, auto-scale from ETF)
                       Use same value across all snapshots for consistent comparison
        """
        # Select classes to visualize
        if selected_classes is None:
            selected_classes = list(range(min(3, snapshot.num_classes)))
        
        n_vis_classes = len(selected_classes)
        
        # Convert selected_classes to array for JAX indexing
        selected_classes_array = jnp.array(selected_classes)
        
        # Filter data for selected classes
        mask = jnp.isin(snapshot.labels, selected_classes_array)
        filtered_features = snapshot.features[mask]
        filtered_labels = snapshot.labels[mask]
        filtered_means = snapshot.class_means[selected_classes_array]
        filtered_classifiers = snapshot.classifiers[selected_classes_array]
        filtered_biases = snapshot.biases[selected_classes_array]
        
        # Sample features for visualization
        sampled_features = []
        sampled_labels = []
        for i, c in enumerate(selected_classes):
            class_mask = filtered_labels == c
            class_features = filtered_features[class_mask]
            
            n_samples = min(samples_per_class, len(class_features))
            if n_samples > 0:
                indices = np.random.choice(len(class_features), n_samples, replace=False)
                sampled_features.append(class_features[indices])
                sampled_labels.append(np.full(n_samples, i))
        
        if sampled_features:
            sampled_features = jnp.concatenate(sampled_features, axis=0)
            sampled_labels = np.concatenate(sampled_labels, axis=0)
        else:
            sampled_features = jnp.array([])
            sampled_labels = np.array([])
        
        # Project to 3D subspace using SVD with normalization
        if len(sampled_features) > 0:
            proj_features, proj_means, proj_classifiers, etf_3d, basis = self.project_to_subspace(
                sampled_features, filtered_means, filtered_classifiers, target_dim=3, normalize=True
            )
        else:
            _, proj_means, proj_classifiers, etf_3d, basis = self.project_to_subspace(
                filtered_features[:1], filtered_means, filtered_classifiers, target_dim=3, normalize=True
            )
            proj_features = jnp.array([]).reshape(0, 3)
        
        # Project biases to same 3D space
        # Biases are in the output space, project them as vectors
        # For visualization, we treat bias as a direction vector in the output space
        # and project it to the same subspace
        bias_vectors = []
        for i in range(n_vis_classes):
            # Use negative classifier direction scaled by bias magnitude
            # This shows the anti-alignment property of biases in NC
            bias_vector = -filtered_classifiers[i] * jnp.abs(filtered_biases[i])
            bias_vectors.append(bias_vector)
        bias_vectors = jnp.stack(bias_vectors)
        
        proj_biases = bias_vectors @ basis
        # Normalize biases for angle visualization
        bias_norms = jnp.linalg.norm(proj_biases, axis=1, keepdims=True)
        proj_biases = proj_biases / (bias_norms + 1e-8)
        
        # ETF is already computed and normalized in project_to_subspace
        # It's now FIXED and doesn't scale with data
        
        # Create 3D plot with better styling
        fig = plt.figure(figsize=figsize, facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        
        # Define colors for each class (using shades of the main colors)
        # Blue shades for features and class means
        blue_shades = [
            (0.2, 0.4 + 0.2*i/n_vis_classes, 0.8),  # Darker to lighter blues
            (0.3, 0.5 + 0.2*i/n_vis_classes, 0.9),
            (0.4, 0.6 + 0.2*i/n_vis_classes, 1.0)
        ][:n_vis_classes]
        
        # Red shades for classifiers
        red_shades = [
            (0.7 + 0.15*i/n_vis_classes, 0.1, 0.1),  # Different red intensities
            (0.85 + 0.15*i/n_vis_classes, 0.15, 0.15),
            (1.0, 0.2, 0.2)
        ][:n_vis_classes]
        
        # Orange shades for biases
        orange_shades = [
            (1.0, 0.5 + 0.2*i/n_vis_classes, 0.1),  # Different orange shades
            (1.0, 0.6 + 0.2*i/n_vis_classes, 0.15),
            (1.0, 0.7 + 0.2*i/n_vis_classes, 0.2)
        ][:n_vis_classes]
        
        # Green shades for ETF
        green_base = (0.2, 0.8, 0.2)
        
        # 1. Plot Simplex ETF (green spheres) - LARGEST
        for i in range(n_vis_classes):
            green_shade = (
                green_base[0] + 0.15*i/n_vis_classes,
                green_base[1],
                green_base[2] - 0.15*i/n_vis_classes
            )
            ax.scatter(
                [etf_3d[i, 0]], [etf_3d[i, 1]], [etf_3d[i, 2]],
                c=[green_shade], s=600, alpha=0.7, marker='o',
                edgecolors='darkgreen', linewidths=3,
                label='Simplex ETF' if i == 0 else None,
                depthshade=True
            )
        
        # 2. Plot individual features (small blue spheres, by class)
        if len(proj_features) > 0:
            for i in range(n_vis_classes):
                class_mask = sampled_labels == i
                if jnp.sum(class_mask) > 0:
                    ax.scatter(
                        proj_features[class_mask, 0],
                        proj_features[class_mask, 1],
                        proj_features[class_mask, 2],
                        c=[blue_shades[i]], s=25, alpha=0.3,
                        edgecolors='none',
                        label=f'Features Class {selected_classes[i]}' if i == 0 else None,
                        depthshade=True
                    )
        
        # 3. Plot class means (blue ball-and-sticks)
        for i in range(n_vis_classes):
            # Ball at the mean (larger than features)
            ax.scatter(
                [proj_means[i, 0]], [proj_means[i, 1]], [proj_means[i, 2]],
                c=[blue_shades[i]], s=400, alpha=0.9, marker='o',
                edgecolors='darkblue', linewidths=2.5,
                label='Class Means' if i == 0 else None,
                depthshade=True
            )
            
            # Stick from origin to mean (thicker)
            ax.plot(
                [0, proj_means[i, 0]],
                [0, proj_means[i, 1]],
                [0, proj_means[i, 2]],
                c=blue_shades[i], linewidth=4, alpha=0.8
            )
        
        # 4. Plot classifiers (red ball-and-sticks)
        for i in range(n_vis_classes):
            # Ball at the classifier
            ax.scatter(
                [proj_classifiers[i, 0]], [proj_classifiers[i, 1]], [proj_classifiers[i, 2]],
                c=[red_shades[i]], s=400, alpha=0.9, marker='o',
                edgecolors='darkred', linewidths=2.5,
                label='Classifiers W' if i == 0 else None,
                depthshade=True
            )
            
            # Stick from origin to classifier (thicker)
            ax.plot(
                [0, proj_classifiers[i, 0]],
                [0, proj_classifiers[i, 1]],
                [0, proj_classifiers[i, 2]],
                c=red_shades[i], linewidth=4, alpha=0.8
            )
        
        # 5. Plot biases (orange ball-and-sticks) - showing antialignment
        for i in range(n_vis_classes):
            # Ball at the bias position (normalized direction)
            ax.scatter(
                [proj_biases[i, 0]], [proj_biases[i, 1]], [proj_biases[i, 2]],
                c=[orange_shades[i]], s=300, alpha=0.85, marker='s',  # Square markers
                edgecolors='darkorange', linewidths=2.5,
                label='Biases (anti-aligned)' if i == 0 else None,
                depthshade=True
            )
            
            # Stick from origin to bias (dashed line)
            ax.plot(
                [0, proj_biases[i, 0]],
                [0, proj_biases[i, 1]],
                [0, proj_biases[i, 2]],
                c=orange_shades[i], linewidth=3.5, alpha=0.8, linestyle='--'
            )
        
        # Compute TRUE angles in original feature space (all components)
        all_angles = self.compute_all_angles_in_original_space(snapshot)
        
        if n_vis_classes >= 2:
            # Extract angles for all components  
            means_angles = all_angles['class_means']
            classifiers_angles = all_angles['classifiers']
            biases_angles = all_angles['biases']
            alignment_angles = all_angles['mean_classifier_alignment']
        
        # Origin marker
        ax.scatter([0], [0], [0], c='black', s=100, marker='x', linewidths=3,
                  label='Origin', alpha=0.7)
        
        # Set labels and title with better fonts and detailed angle information
        ax.set_xlabel('Principal Component 1', fontsize=13, fontweight='bold')
        ax.set_ylabel('Principal Component 2', fontsize=13, fontweight='bold')
        ax.set_zlabel('Principal Component 3', fontsize=13, fontweight='bold')
        
        if title is None:
            if n_vis_classes >= 2:
                title = f'Neural Collapse at Step {snapshot.epoch}\n'
                title += f'TRUE Angles in R^{self.feature_dim}: '
                title += f'Means: {means_angles["mean_angle"]:.1f}° | '
                title += f'Classifiers: {classifiers_angles["mean_angle"]:.1f}° | '
                title += f'Biases: {biases_angles["mean_angle"]:.1f}°\n'
                title += f'Alignment: {alignment_angles["mean_angle"]:.1f}° | '
                title += f'Targets: {means_angles["optimal_angle"]:.1f}° (simplex), 0° (alignment)'
            else:
                title = f'Neural Collapse at Step {snapshot.epoch}'
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Add legend with better positioning
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9, 
                 bbox_to_anchor=(0.02, 0.98))
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Fixed axis limits based on unit sphere (since everything is normalized)
        # ETF vertices are unit-normalized, so use 1.2 as consistent scale
        if axis_limit is not None:
            scale = axis_limit
        else:
            scale = 1.5  # Fixed scale for unit-normalized vectors
        
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_zlim(-scale, scale)
        
        # Grid styling
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Close figure to save memory
        else:
            plt.show()
    
    def visualize_neural_collapse_2d(
        self,
        snapshot: NeuralCollapseSnapshot,
        selected_classes: Optional[List[int]] = None,
        samples_per_class: int = 50,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        axis_limit: Optional[float] = None
    ):
        """
        Create 2D visualization of Neural Collapse (easier to interpret angles).
        
        For 3-class problems, the optimal configuration has classes separated by 
        exactly 120° angles, which is much easier to see and measure in 2D.
        
        This uses the SAME SVD-based projection as the 3D version (first 2 components),
        ensuring consistency between visualizations. All vectors are normalized to
        unit length to clearly show angular relationships.
        
        Args:
            snapshot: Network state to visualize
            selected_classes: Which classes to visualize (None = all, max 3 for clarity)
            samples_per_class: Number of feature samples to show per class
            figsize: Figure size (width, height)
            save_path: Path to save figure (if provided)
            title: Custom title (default: based on epoch)
            axis_limit: Fixed axis limit for consistent comparison (default: 1.5 for unit vectors)
        """
        # Select classes to visualize
        if selected_classes is None:
            selected_classes = list(range(min(3, snapshot.num_classes)))
        
        n_vis_classes = len(selected_classes)
        
        # Convert selected_classes to array for JAX indexing
        selected_classes_array = jnp.array(selected_classes)
        
        # Filter data for selected classes
        mask = jnp.isin(snapshot.labels, selected_classes_array)
        filtered_features = snapshot.features[mask]
        filtered_labels = snapshot.labels[mask]
        filtered_means = snapshot.class_means[selected_classes_array]
        filtered_classifiers = snapshot.classifiers[selected_classes_array]
        filtered_biases = snapshot.biases[selected_classes_array]
        
        # Sample features for visualization
        sampled_features = []
        sampled_labels = []
        for i, c in enumerate(selected_classes):
            class_mask = filtered_labels == c
            class_features = filtered_features[class_mask]
            
            n_samples = min(samples_per_class, len(class_features))
            if n_samples > 0:
                indices = np.random.choice(len(class_features), n_samples, replace=False)
                sampled_features.append(class_features[indices])
                sampled_labels.append(np.full(n_samples, i))
        
        if sampled_features:
            sampled_features = jnp.concatenate(sampled_features, axis=0)
            sampled_labels = jnp.concatenate(sampled_labels, axis=0)
        else:
            sampled_features = jnp.array([]).reshape(0, snapshot.features.shape[1])
            sampled_labels = jnp.array([])
        
        # Project to 2D using SAME SVD method as 3D (target_dim=2, with normalization)
        if len(sampled_features) > 0:
            proj_features, proj_means, proj_classifiers, etf_2d, basis = self.project_to_subspace(
                sampled_features, filtered_means, filtered_classifiers, target_dim=2, normalize=True
            )
        else:
            _, proj_means, proj_classifiers, etf_2d, basis = self.project_to_subspace(
                filtered_features[:1] if len(filtered_features) > 0 else jnp.zeros((1, snapshot.feature_dim)),
                filtered_means, filtered_classifiers, target_dim=2, normalize=True
            )
            proj_features = jnp.array([]).reshape(0, 2)
        
        # Project biases (same method as 3D)
        bias_vectors = []
        for i in range(n_vis_classes):
            # Bias as negative classifier direction scaled by magnitude
            bias_vector = -filtered_classifiers[i] * jnp.abs(filtered_biases[i])
            bias_vectors.append(bias_vector)
        bias_vectors = jnp.stack(bias_vectors)
        
        proj_biases = bias_vectors @ basis
        # Normalize biases
        bias_norms = jnp.linalg.norm(proj_biases, axis=1, keepdims=True)
        proj_biases = proj_biases / (bias_norms + 1e-8)
        
        # ETF is now FIXED and properly computed in 2D space
        
        # Create 2D plot
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Define colors (same as 3D version)
        blue_shades = [(0.2, 0.4 + 0.2*i/n_vis_classes, 0.8) for i in range(n_vis_classes)]
        red_shades = [(0.8, 0.2 + 0.3*i/n_vis_classes, 0.2) for i in range(n_vis_classes)]
        green_shades = [(0.2 + 0.3*i/n_vis_classes, 0.7, 0.2) for i in range(n_vis_classes)]
        orange_shades = [(1.0, 0.5 + 0.3*i/n_vis_classes, 0.1) for i in range(n_vis_classes)]
        
        # 1. Plot Simplex ETF (green circles) - FIXED theoretical target
        for i in range(n_vis_classes):
            ax.scatter(
                [etf_2d[i, 0]], [etf_2d[i, 1]],
                c=[green_shades[i]], s=300, alpha=0.7, marker='o',
                edgecolors='darkgreen', linewidths=2,
                label='Simplex ETF' if i == 0 else None,
                zorder=10
            )
        
        # 2. Plot individual features (small blue dots by class)
        if len(proj_features) > 0:
            for i in range(n_vis_classes):
                class_mask = sampled_labels == i
                if jnp.sum(class_mask) > 0:
                    ax.scatter(
                        proj_features[class_mask, 0], proj_features[class_mask, 1],
                        c=[blue_shades[i]], s=20, alpha=0.6, marker='.',
                        label='Features' if i == 0 else None,
                        zorder=3
                    )
        
        # 3. Plot class means (blue circles with sticks from origin)
        for i in range(n_vis_classes):
            # Circle for mean
            ax.scatter(
                [proj_means[i, 0]], [proj_means[i, 1]],
                c=[blue_shades[i]], s=200, alpha=0.9, marker='o',
                edgecolors='darkblue', linewidths=2,
                label='Class Means' if i == 0 else None,
                zorder=8
            )
            
            # Stick from origin to mean
            ax.plot(
                [0, proj_means[i, 0]], [0, proj_means[i, 1]],
                color=blue_shades[i], linewidth=3, alpha=0.8, zorder=7
            )
        
        # 4. Plot classifiers (red triangles with sticks from origin)
        for i in range(n_vis_classes):
            # Triangle for classifier
            ax.scatter(
                [proj_classifiers[i, 0]], [proj_classifiers[i, 1]],
                c=[red_shades[i]], s=200, alpha=0.9, marker='^',
                edgecolors='darkred', linewidths=2,
                label='Classifiers' if i == 0 else None,
                zorder=8
            )
            
            # Stick from origin to classifier
            ax.plot(
                [0, proj_classifiers[i, 0]], [0, proj_classifiers[i, 1]],
                color=red_shades[i], linewidth=3, alpha=0.8, zorder=7
            )
        
        # 5. Plot bias vectors (orange squares with normalized directions)
        for i in range(n_vis_classes):
            ax.scatter(
                [proj_biases[i, 0]], [proj_biases[i, 1]],
                c=[orange_shades[i]], s=150, alpha=0.8, marker='s',
                edgecolors='darkorange', linewidths=2,
                label='Biases (anti-aligned)' if i == 0 else None,
                zorder=6
            )
            
            # Stick from origin to bias (dashed line)
            ax.plot(
                [0, proj_biases[i, 0]], [0, proj_biases[i, 1]],
                color=orange_shades[i], linewidth=2, alpha=0.7, linestyle='--', zorder=5
            )
        
        # Add origin marker
        ax.scatter([0], [0], c='black', s=100, marker='x', linewidths=3, zorder=9)
        
        # Compute TRUE angles in original feature space (all components)
        all_angles = self.compute_all_angles_in_original_space(snapshot)
        
        if n_vis_classes >= 2:
            # Extract angles for all components
            means_angles = all_angles['class_means']
            classifiers_angles = all_angles['classifiers']
            biases_angles = all_angles['biases']
            alignment_angles = all_angles['mean_classifier_alignment']
        
        # Set title with comprehensive TRUE angle information
        if title is None:
            if n_vis_classes >= 2:
                title = f'Neural Collapse (2D View) - Step {snapshot.epoch}\n'
                title += f'TRUE Angles in R^{self.feature_dim}: '
                title += f'Means: {means_angles["mean_angle"]:.1f}° | '
                title += f'Classifiers: {classifiers_angles["mean_angle"]:.1f}° | '
                title += f'Biases: {biases_angles["mean_angle"]:.1f}°\n'
                title += f'Mean-Classifier Alignment: {alignment_angles["mean_angle"]:.1f}° | '
                title += f'Target: {means_angles["optimal_angle"]:.1f}° (simplex), 0° (alignment)\n'
                title += '⚠️ 2D view angles are projection artifacts - TRUE angles shown above'
            else:
                title = f'Neural Collapse (2D View) - Step {snapshot.epoch}'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        
        # Set axis properties with FIXED scale (since vectors are normalized)
        if axis_limit is not None:
            scale = axis_limit
        else:
            scale = 1.5  # Fixed scale for unit-normalized vectors (same as 3D)
        
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        
        # Ensure equal aspect ratio for accurate angle visualization (CRITICAL!)
        ax.set_aspect('equal')
        
        # Grid and styling
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        # Legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Labels
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Close figure to save memory
        else:
            plt.show()
    
    def create_temporal_visualization(
        self,
        epochs_to_visualize: Optional[List[int]] = None,
        selected_classes: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (20, 15),
        save_path: Optional[Path] = None
    ):
        """
        Create a temporal visualization showing Neural Collapse evolution.
        
        Similar to Figure 1 in the paper, showing multiple epochs in a grid.
        
        Args:
            epochs_to_visualize: List of epoch numbers to visualize
            selected_classes: Which classes to show
            figsize: Figure size
            save_path: Path to save figure
        """
        if not self.snapshots:
            raise ValueError("No snapshots available. Run training with snapshot capture first.")
        
        # Select snapshots
        if epochs_to_visualize is None:
            # Select evenly spaced snapshots
            n_snapshots = min(4, len(self.snapshots))
            indices = np.linspace(0, len(self.snapshots) - 1, n_snapshots, dtype=int)
            selected_snapshots = [self.snapshots[i] for i in indices]
        else:
            selected_snapshots = [
                s for s in self.snapshots if s.epoch in epochs_to_visualize
            ]
        
        n_snapshots = len(selected_snapshots)
        
        # Create grid of subplots
        fig = plt.figure(figsize=figsize)
        
        for idx, snapshot in enumerate(selected_snapshots):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            
            # This is a simplified version - you would call the visualization
            # logic here for each snapshot
            # For now, just show the structure
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved temporal visualization to {save_path}")
        
        plt.show()
    
    def save_snapshots(self, filepath: Path):
        """Save all snapshots to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.snapshots, f)
        print(f"Saved {len(self.snapshots)} snapshots to {filepath}")
    
    def load_snapshots(self, filepath: Path):
        """Load snapshots from disk."""
        with open(filepath, 'rb') as f:
            self.snapshots = pickle.load(f)
        print(f"Loaded {len(self.snapshots)} snapshots from {filepath}")


def create_feature_extractor_model(base_model, params):
    """
    Create a feature extractor from a trained model.
    
    This function wraps a model to expose intermediate features.
    
    Args:
        base_model: Original model (init_fn, apply_fn) tuple
        params: Model parameters
        
    Returns:
        Function that returns both predictions and features
    """
    init_fn, apply_fn = base_model
    
    def feature_apply_fn(params, x):
        """Apply function that returns features and predictions."""
        # This is model-specific and needs to be implemented
        # based on the architecture
        raise NotImplementedError(
            "Feature extraction needs model-specific implementation"
        )
    
    return feature_apply_fn


def create_video_from_images(
    image_dir: Path,
    output_path: Path,
    pattern: str = "nc_viz_step_*.png",
    fps: float = 2.0,
    sort_key=None
) -> None:
    """
    Create a video from a sequence of images.
    
    Args:
        image_dir: Directory containing the images
        output_path: Path for the output video file
        pattern: Glob pattern to match image files (default: "nc_viz_step_*.png")
        fps: Frames per second (default: 2.0 for 0.5 seconds per frame)
        sort_key: Optional function to sort image files (default: sorts by filename)
    
    Example:
        create_video_from_images(
            Path("outputs/experiment/2d-snapshots"),
            Path("outputs/experiment/nc_evolution_2d.mp4"),
            fps=2.0  # 0.5 seconds per frame
        )
    """
    # Find all matching images
    image_files = sorted(image_dir.glob(pattern), key=sort_key if sort_key else lambda x: x.name)
    
    if not image_files:
        logging.warning(f"No images found matching pattern '{pattern}' in {image_dir}")
        return
    
    logging.info(f"Creating video from {len(image_files)} images...")
    logging.info(f"  FPS: {fps} ({1.0/fps:.2f} seconds per frame)")
    logging.info(f"  Output: {output_path}")
    
    # Try to create MP4 video with ffmpeg backend
    try:
        # Try using ffmpeg plugin
        with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
            for i, img_file in enumerate(image_files):
                if (i + 1) % 50 == 0:
                    logging.info(f"  Processing frame {i + 1}/{len(image_files)}")
                
                image = imageio.imread(img_file)
                writer.append_data(image)
        
        logging.info(f"Video created successfully: {output_path}")
        logging.info(f"  Duration: {len(image_files) / fps:.2f} seconds")
        
    except (ImportError, ValueError, RuntimeError) as e:
        # Fallback: Create GIF instead
        logging.warning(f"Could not create MP4 video: {e}")
        logging.info("Falling back to GIF format...")
        
        gif_path = output_path.with_suffix('.gif')
        try:
            with imageio.get_writer(gif_path, mode='I', fps=fps, loop=0) as writer:
                for i, img_file in enumerate(image_files):
                    if (i + 1) % 50 == 0:
                        logging.info(f"  Processing frame {i + 1}/{len(image_files)}")
                    
                    image = imageio.imread(img_file)
                    writer.append_data(image)
            
            logging.info(f"GIF created successfully: {gif_path}")
            logging.info(f"  Duration: {len(image_files) / fps:.2f} seconds")
            logging.warning(f"NOTE: To create MP4 videos, install ffmpeg: pip install imageio[ffmpeg]")
            
        except Exception as gif_error:
            logging.error(f"Failed to create both MP4 and GIF: {gif_error}")
            logging.error(f"Video creation skipped. Individual frames are still available in {image_dir}")
