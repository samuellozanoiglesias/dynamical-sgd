"""
Neural Collapse Figure 1 Visualization (Papyan et al., 2020)

This module creates 2D projections showing the geometry of:
- Last-layer features H
- Class means μ_c  
- Classifier weights W
- Theoretical Simplex ETF

Following the exact methodology from the paper's Figure 1.

Author: Samuel Lozano Iglesias
Email: samuel.lozano@ucm.es
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from typing import Tuple, Optional, List
import logging


def compute_simplex_etf(C: int, p: int) -> jnp.ndarray:
    """
    Compute theoretical Simplex Equiangular Tight Frame (ETF).
    
    Properties:
    - C vectors in ℝ^p
    - Unit norm: ||v_c|| = 1
    - Pairwise dot product: <v_i, v_j> = -1/(C-1) for i≠j
    - Zero global mean: Σ_c v_c = 0
    
    Args:
        C: Number of classes
        p: Feature dimension
        
    Returns:
        ETF matrix (p, C) where each column is a vertex
    """
    # Start with centered identity in C dimensions
    I = np.eye(C)
    ones = np.ones((C, C)) / C
    centered_I = I - ones  # (C, C), rank C-1
    
    # SVD to get orthonormal basis in rank C-1 subspace
    U, S, Vt = np.linalg.svd(centered_I, full_matrices=False)
    
    # Take first C-1 columns (the non-zero singular values)
    U_reduced = U[:, :C-1]  # (C, C-1)
    
    # Scale to get equiangular tight frame
    # ETF vertices have norm sqrt(C/(C-1))
    scale = np.sqrt(C / (C - 1))
    etf_reduced = U_reduced * scale  # (C, C-1)
    
    # Embed in p dimensions (pad with zeros if p > C-1)
    if p >= C - 1:
        etf = np.zeros((C, p))
        etf[:, :C-1] = etf_reduced
    else:
        # If p < C-1, just take first p dimensions (shouldn't happen in practice)
        etf = etf_reduced[:, :p]
    
    # Return as (p, C) to match M format
    return jnp.array(etf.T)


def compute_joint_pca_projection(
    H: jnp.ndarray,
    class_means: jnp.ndarray,
    W: jnp.ndarray,
    ETF: jnp.ndarray,
    mu_G: jnp.ndarray,
    normalize_vectors: bool = True
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute joint PCA projection to 2D for all objects.
    
    All objects are centered and projected using the SAME transformation.
    
    Args:
        H: Last-layer features (N, p)
        class_means: Class mean vectors (C, p)
        W: Classifier weights (C, p)
        ETF: Simplex ETF (p, C)
        mu_G: Global mean (p,)
        normalize_vectors: If True, normalize class means, classifiers, ETF before projection
        
    Returns:
        Tuple of (projection_matrix, centered_data):
            - projection_matrix: (2, p) - first 2 principal components
            - centered_data: dict with all centered objects
    """
    N, p = H.shape
    C = class_means.shape[0]
    
    # 1. Center all objects relative to global mean
    H_centered = H - mu_G  # (N, p)
    M_centered = class_means - mu_G  # (C, p)
    W_centered = W - jnp.mean(W, axis=0)  # Center classifiers by their own mean
    ETF_centered = ETF - jnp.mean(ETF, axis=1, keepdims=True)  # (p, C) → center columns
    
    # 2. Optional: Normalize vectors (NOT features) for angular structure
    if normalize_vectors:
        # Normalize class means
        M_norms = jnp.linalg.norm(M_centered, axis=1, keepdims=True) + 1e-8
        M_normalized = M_centered / M_norms
        
        # Normalize classifiers
        W_norms = jnp.linalg.norm(W_centered, axis=1, keepdims=True) + 1e-8
        W_normalized = W_centered / W_norms
        
        # Normalize ETF columns
        ETF_norms = jnp.linalg.norm(ETF_centered, axis=0, keepdims=True) + 1e-8
        ETF_normalized = (ETF_centered / ETF_norms).T  # (C, p)
    else:
        M_normalized = M_centered
        W_normalized = W_centered
        ETF_normalized = ETF_centered.T  # (C, p)
    
    # 3. Build joint data matrix for PCA
    # Stack: [features (N, p), class_means (C, p), classifiers (C, p), ETF (C, p)]
    joint_data = jnp.vstack([
        H_centered,
        M_normalized,
        W_normalized,
        ETF_normalized
    ])  # (N + 3C, p)
    
    # 4. Compute PCA via SVD
    U, S, Vt = jnp.linalg.svd(joint_data, full_matrices=False)
    
    # First 2 principal components
    projection_matrix = Vt[:2, :]  # (2, p)
    
    # Store centered/normalized objects
    centered_data = {
        'H_centered': H_centered,
        'M_normalized': M_normalized,
        'W_normalized': W_normalized,
        'ETF_normalized': ETF_normalized,
        'M_centered': M_centered,  # Also store unnormalized for optional use
        'W_centered': W_centered,
    }
    
    return projection_matrix, centered_data


def project_to_2d(data: jnp.ndarray, projection_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Project data to 2D using projection matrix.
    
    Args:
        data: (N, p) data matrix
        projection_matrix: (2, p) projection matrix
        
    Returns:
        (N, 2) projected data
    """
    return jnp.dot(data, projection_matrix.T)


def visualize_nc_figure1(
    H: jnp.ndarray,
    labels: jnp.ndarray,
    class_means: jnp.ndarray,
    W: jnp.ndarray,
    mu_G: jnp.ndarray,
    epoch: int,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    normalize_vectors: bool = False,
    show_etf: bool = True,
    show_features: bool = True,
    feature_samples: Optional[int] = 50,
    axis_limit: float = 1.5
) -> plt.Figure:
    """
    Create Figure 1 style visualization of Neural Collapse geometry.
    
    Args:
        H: Last-layer features (N, p)
        labels: Class labels (N,)
        class_means: Class mean vectors (C, p)
        W: Classifier weights (C, p)
        mu_G: Global mean (p,)
        epoch: Training epoch/step number
        save_path: Path to save figure (optional)
        title: Custom title (optional)
        colors: List of colors for each class (optional)
        normalize_vectors: Normalize means/classifiers/ETF before projection
        show_etf: Whether to show theoretical ETF
        show_features: Whether to show individual features
        feature_samples: Max samples per class to plot (None = all)
        axis_limit: Fixed axis limits for all frames
        
    Returns:
        Matplotlib figure
    """
    N, p = H.shape
    C = class_means.shape[0]
    
    # Color scheme per object type, with intensity variations per class
    # Green for ETF (light -> dark)
    green_colors = ['#90EE90', '#32CD32', '#006400']  # Light green, lime green, dark green
    # Red for classifiers (light -> dark)
    red_colors = ['#FF6B6B', '#DC143C', '#8B0000']  # Light red, crimson, dark red
    # Blue for class means and features (light -> dark)
    blue_colors = ['#87CEEB', '#1E90FF', '#00008B']  # Sky blue, dodger blue, dark blue
    
    # 1. Compute Simplex ETF
    ETF = compute_simplex_etf(C, p)  # (p, C)
    
    # 2. Compute joint PCA projection
    projection_matrix, centered_data = compute_joint_pca_projection(
        H, class_means, W, ETF, mu_G, normalize_vectors
    )
    
    # 3. Project all objects to 2D
    H_2d = project_to_2d(centered_data['H_centered'], projection_matrix)
    M_2d = project_to_2d(centered_data['M_normalized'], projection_matrix)
    W_2d = project_to_2d(centered_data['W_normalized'], projection_matrix)
    ETF_2d_initial = project_to_2d(centered_data['ETF_normalized'], projection_matrix)
    origin_2d = np.array([0, 0])
    
    # 3.5. Fix ETF at canonical angles (90°, 210°, 330°) and rotate everything to align
    # This makes the theoretical target stationary and shows convergence clearly
    
    # Define target ETF positions (evenly spaced, starting at 90°)
    target_angles = np.array([np.pi/2, 7*np.pi/6, 11*np.pi/6])  # 90°, 210°, 330°
    ETF_2d_fixed = np.array([[np.cos(a), np.sin(a)] for a in target_angles])  # (C, 2)
    
    # Normalize projected ETF to unit circle
    ETF_norms = jnp.linalg.norm(ETF_2d_initial, axis=1, keepdims=True) + 1e-8
    ETF_2d_projected = ETF_2d_initial / ETF_norms
    
    # Find rotation matrix that aligns projected ETF with fixed positions
    # Using Procrustes: R = V @ U^T where ETF_fixed^T = U S V^T @ ETF_projected^T
    # Simplified: find rotation minimizing ||ETF_fixed - ETF_projected @ R^T||
    H_procrustes = ETF_2d_projected.T @ ETF_2d_fixed  # (2, 2)
    U, S, Vt = np.linalg.svd(H_procrustes)
    R = U @ Vt  # Optimal rotation matrix (2, 2)
    
    # Apply rotation to all objects
    H_2d = np.dot(H_2d, R.T)
    M_2d = np.dot(M_2d, R.T)
    W_2d = np.dot(W_2d, R.T)
    ETF_2d = ETF_2d_fixed  # Use fixed positions
    
    # Normalize vectors to unit circle in 2D
    M_norms_2d = jnp.linalg.norm(M_2d, axis=1, keepdims=True) + 1e-8
    M_2d = M_2d / M_norms_2d
    
    W_norms_2d = jnp.linalg.norm(W_2d, axis=1, keepdims=True) + 1e-8
    W_2d = W_2d / W_norms_2d
    
    # Normalize features to unit disk (scale by max so they fit inside circle)
    if show_features:
        max_feature_norm = jnp.max(jnp.linalg.norm(H_2d, axis=1)) + 1e-8
        H_2d = H_2d / max_feature_norm  # Features now within unit circle
    
    # 4. Create figure with FIXED SIZE for consistent video frames
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    
    # 5. Plot features (small BLUE spheres, more visible)
    if show_features:
        for c in range(C):
            mask = labels == c
            H_c = H_2d[mask]
            
            # Subsample if too many points
            if feature_samples is not None and len(H_c) > feature_samples:
                indices = np.random.choice(len(H_c), feature_samples, replace=False)
                H_c = H_c[indices]
            
            ax.scatter(H_c[:, 0], H_c[:, 1], 
                      c=blue_colors[c], s=30, alpha=0.7, 
                      edgecolors='none', label=f'Class {c+1} Features')
    
    # 6. Plot class means (BLUE ball-and-stick)
    for c in range(C):
        # Draw stick from origin to mean
        ax.plot([origin_2d[0], M_2d[c, 0]], 
               [origin_2d[1], M_2d[c, 1]], 
               color=blue_colors[c], linewidth=4, alpha=0.9, zorder=3)
        
        # Draw ball at mean
        ax.scatter(M_2d[c, 0], M_2d[c, 1], 
                  c=blue_colors[c], s=350, alpha=1.0, edgecolors='black', 
                  linewidth=2.5, zorder=5, marker='o',
                  label=f'Class {c+1} Mean')
    
    # 7. Plot classifiers (RED ball-and-stick)
    for c in range(C):
        # Draw stick from origin to classifier
        ax.plot([origin_2d[0], W_2d[c, 0]], 
               [origin_2d[1], W_2d[c, 1]], 
               color=red_colors[c], linewidth=4, alpha=0.9, zorder=3)
        
        # Draw ball at classifier
        ax.scatter(W_2d[c, 0], W_2d[c, 1], 
                  c=red_colors[c], s=350, alpha=1.0, edgecolors='black', 
                  linewidth=2.5, marker='o', zorder=5,
                  label=f'Class {c+1} Classifier')
    
    # 8. Plot ETF vertices (GREEN spheres)
    if show_etf:
        for c in range(C):
            # Draw ETF vertex (sphere only, no stick)
            ax.scatter(ETF_2d[c, 0], ETF_2d[c, 1], 
                      c=green_colors[c], s=350, alpha=1.0, edgecolors='black', 
                      linewidth=2.5, marker='o', zorder=5,
                      label=f'Class {c+1} ETF')
    
    # 9. Mark origin
    ax.scatter(0, 0, c='black', s=150, marker='x', linewidth=4, zorder=5)
    
    # 10. Formatting with FIXED AXIS LIMITS
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    
    # Grid and axes (NO unit circle)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    
    # Labels and title
    if title is None:
        title = f'Neural Collapse Geometry (Step {epoch})'
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('PC 1 (normalized)', fontsize=16, fontweight='bold')
    ax.set_ylabel('PC 2 (normalized)', fontsize=16, fontweight='bold')
    
    # Legend - organize by class
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=1)
    
    plt.tight_layout(pad=1.5)
    
    # Save with FIXED SIZE to avoid video encoding issues
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        logging.info(f"Saved NC Figure 1 visualization: {save_path}")
    
    return fig


def create_nc_figure1_evolution(
    snapshots: List,
    output_dir: Path,
    selected_epochs: Optional[List[int]] = None,
    normalize_vectors: bool = False,
    show_etf: bool = True,
    show_features: bool = True,
    fps: float = 2.0,
    axis_limit: float = 1.2
) -> None:
    """
    Create evolution visualization across training epochs.
    
    Args:
        snapshots: List of NeuralCollapseSnapshot objects
        output_dir: Directory to save visualizations
        selected_epochs: Specific epochs to visualize (None = all)
        normalize_vectors: Normalize vectors before projection
        show_etf: Show theoretical ETF
        show_features: Show individual features
        fps: Frames per second for video
        axis_limit: Fixed axis limits for all frames
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / 'nc_figure1_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Select snapshots
    if selected_epochs is not None:
        selected_snapshots = [s for s in snapshots if s.epoch in selected_epochs]
    else:
        selected_snapshots = snapshots
    
    logging.info(f"Creating {len(selected_snapshots)} NC Figure 1 visualizations...")
    
    frame_files = []
    for snap in selected_snapshots:
        # Create visualization
        frame_path = frames_dir / f'nc_fig1_epoch_{snap.epoch:06d}.png'
        
        fig = visualize_nc_figure1(
            H=snap.features,
            labels=snap.labels,
            class_means=snap.class_means,
            W=snap.classifiers,
            mu_G=jnp.mean(snap.class_means, axis=0),  # Compute global mean
            epoch=snap.epoch,
            save_path=frame_path,
            normalize_vectors=normalize_vectors,
            show_etf=show_etf,
            show_features=show_features,
            axis_limit=axis_limit
        )
        plt.close(fig)
        
        frame_files.append(frame_path)
    
    # Create video from frames
    video_path = output_dir / 'nc_figure1_evolution.mp4'
    create_video_from_frames(frame_files, video_path, fps=fps)
    
    logging.info(f"✅ Created NC Figure 1 evolution video: {video_path}")


def create_video_from_frames(
    frame_files: List[Path],
    output_path: Path,
    fps: float = 2.0,
    cleanup: bool = False
) -> None:
    """
    Create MP4 video from image frames with consistent sizing.
    
    Args:
        frame_files: List of image file paths
        output_path: Output video path
        fps: Frames per second
        cleanup: Whether to delete frames after creating video
    """
    try:
        import imageio
        from PIL import Image as PILImage
        
        # Read first image to get target size
        first_image = imageio.imread(frame_files[0])
        target_height, target_width = first_image.shape[:2]
        
        # Ensure dimensions are divisible by 16 (required for most video codecs)
        target_width = (target_width // 16) * 16
        target_height = (target_height // 16) * 16
        
        logging.info(f"Creating video with resolution: {target_width}x{target_height}")
        
        with imageio.get_writer(str(output_path), fps=fps, codec='libx264', 
                                macro_block_size=16) as writer:
            for frame_file in frame_files:
                if frame_file.exists():
                    image = imageio.imread(frame_file)
                    
                    # Resize to ensure all frames have same size
                    if image.shape[:2] != (target_height, target_width):
                        pil_image = PILImage.fromarray(image)
                        pil_image = pil_image.resize((target_width, target_height), 
                                                     PILImage.Resampling.LANCZOS)
                        image = np.array(pil_image)
                    
                    writer.append_data(image)
        
        logging.info(f"Video saved to {output_path}")
        
        if cleanup:
            for frame_file in frame_files:
                if frame_file.exists():
                    frame_file.unlink()
            logging.info("Cleaned up frame files")
            
    except ImportError:
        logging.warning("imageio or PIL not installed. Cannot create video. Install with: pip install imageio imageio-ffmpeg pillow")
    except Exception as e:
        logging.error(f"Error creating video: {e}")
