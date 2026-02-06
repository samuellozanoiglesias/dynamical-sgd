# Neural Collapse Figure 1 Visualization

## Overview

This module creates 2D visualizations inspired by **Figure 1 from Papyan et al., 2020** ("Prevalence of Neural Collapse during the terminal phase of deep learning training").

The visualization shows the geometric relationship between:
- 🔵 **Last-layer features** (small colored spheres)
- 🔷 **Class means** (blue ball-and-stick)
- 🔴 **Classifier weights** (red ball-and-stick)
- 🟢 **Simplex ETF** (green triangular vertices)

## Files

- **`analysis/nc_figure1_visualization.py`**: Core visualization module
- **Usage**: Automatically integrated into `run_experiment.py`

## Key Features

### 1. Joint PCA Projection
All objects are projected using the **same** 2D transformation:
```python
# Build joint matrix
joint_data = [H_centered, M_normalized, W_normalized, ETF_normalized]

# Compute PCA
U, S, Vt = SVD(joint_data)
projection = first_2_principal_components
```

### 2. Proper Centering
- Features centered relative to global mean μ_G
- Class means centered
- Classifiers centered by their own mean
- ETF columns centered

### 3. Vector Normalization (Optional)
- **Class means**: Normalized to unit norm
- **Classifiers**: Normalized to unit norm
- **ETF**: Normalized to unit norm
- **Features**: NOT normalized (preserves within-class variation)

This highlights **angular structure** (NC2) over magnitude.

### 4. Simplex ETF Computation

Theoretical target configuration with:
- C vectors in ℝ^p
- Unit norm: ||v_c|| = 1
- Equiangular: ⟨v_i, v_j⟩ = -1/(C-1) for i≠j
- Zero mean: Σ_c v_c = 0

```python
ETF = compute_simplex_etf(num_classes=C, feature_dim=p)
```

## Visualization Elements

### Features (Small Spheres)
- Color: Class-specific
- Size: 20
- Alpha: 0.3
- Subsampled: 50 per class (configurable)

### Class Means (Blue Ball-and-Stick)
- Color: `#2E86AB` (blue)
- Line: From origin to mean
- Marker: Circle, size 200

### Classifiers (Red Ball-and-Stick)
- Color: `#D62246` (red)
- Line: Dashed from origin
- Marker: Square, size 200

### Simplex ETF (Green Triangles)
- Color: `#27AE60` (green)
- Line: Dotted from origin
- Marker: Triangle, size 150

### Formatting
- Equal aspect ratio
- Origin marked with black X
- Grid enabled
- Axes through origin

## Usage

### Automatic (via run_experiment.py)

The visualization runs automatically when NC tracking is enabled:

```yaml
# config/nc_config_*.yaml
analysis:
  track_neural_collapse: true
  nc_snapshot_interval: 100

visualization:
  vis_step_interval: 1000  # Create Figure 1 every 1000 steps
```

Creates:
- `nc_figure1_frames/`: Individual PNG frames
- `nc_figure1_evolution.mp4`: Video animation

### Manual Usage

```python
from analysis.nc_figure1_visualization import visualize_nc_figure1

fig = visualize_nc_figure1(
    H=features,                    # (N, p)
    labels=labels,                 # (N,)
    class_means=class_means,       # (C, p)
    W=classifiers,                 # (C, p)
    mu_G=global_mean,              # (p,)
    epoch=current_epoch,
    save_path=Path('nc_fig1.png'),
    normalize_vectors=True,        # Highlight angular structure
    show_etf=True,
    show_features=True
)
```

### Create Evolution Video

```python
from analysis.nc_figure1_visualization import create_nc_figure1_evolution

create_nc_figure1_evolution(
    snapshots=nc_analyzer.snapshots,
    output_dir=output_dir,
    selected_epochs=[0, 1000, 2000, 3000, 4000],  # or None for all
    normalize_vectors=True,
    show_etf=True,
    show_features=True,
    fps=2.0  # 2 frames per second
)
```

## Output

### Frame Files
```
output_dir/
  nc_figure1_frames/
    nc_fig1_epoch_000000.png
    nc_fig1_epoch_000100.png
    nc_fig1_epoch_000200.png
    ...
```

### Video File
```
output_dir/
  nc_figure1_evolution.mp4  # Animation across training
```

## Interpretation

### What to Look For

1. **NC1 (Variability Collapse)**
   - Features collapse toward class means
   - Blue spheres cluster tightly

2. **NC2 (Equinorm & Equiangularity)**
   - Class means equalize in norm
   - Equal spacing between means
   - Blue sticks approach equal length
   - Equal angles between blue sticks

3. **NC3 (Self-Duality)**
   - Classifier weights align with class means
   - Red sticks overlap with blue sticks

4. **Convergence to ETF**
   - All objects approach green ETF vertices
   - Perfect Neural Collapse: all overlap at ETF

## Dependencies

```bash
pip install jax jaxlib numpy matplotlib imageio imageio-ffmpeg
```

## Configuration

### Recommended Settings

```yaml
analysis:
  track_neural_collapse: true
  nc_snapshot_interval: 100  # Balance detail vs. disk space

visualization:
  vis_step_interval: 1000    # Figure 1 created every 1000 steps
  figure_dpi: 300
```

### Performance Notes

- Each frame: ~0.5 MB (300 dpi)
- Video creation requires `imageio-ffmpeg`
- 5000 steps @ 100 interval = 50 frames = ~25 MB

## References

Papyan, V., Han, X. Y., & Donoho, D. L. (2020). **Prevalence of neural collapse during the terminal phase of deep learning training.** *Proceedings of the National Academy of Sciences*, 117(40), 24652-24663.

## Author

Samuel Lozano Iglesias  
Email: samuel.lozano@ucm.es
