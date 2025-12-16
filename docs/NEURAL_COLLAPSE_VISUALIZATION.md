# Neural Collapse Visualization Guide

## Overview

This document describes the Neural Collapse visualizations implemented in this codebase, which exactly match the visualizations from the paper **"Prevalence of Neural Collapse during the terminal phase of deep learning training"** by Papyan et al. (2020).

## What is Being Visualized

The 3D visualizations show the geometric structure of the network's learned representations as training progresses. All high-dimensional features (in R^p) are projected onto the principal 3D subspace using SVD for visualization.

### Key Components (with Exact Colors from Paper)

#### 1. **Green Spheres: Simplex ETF (Equiangular Tight Frame)**
- **What:** The theoretical optimal configuration for class representations
- **Color:** Green with different shades for each class
- **Size:** Largest spheres (s=600)
- **Meaning:** These represent where class means *should* converge in the terminal phase of training
- **For 3 classes:** Forms an equilateral triangle with vertices 120° apart

#### 2. **Blue Ball-and-Sticks: Class Means (μ_c)**
- **What:** The centroid (average) of all features belonging to each class
- **Color:** Blue with different shades for each class
- **Size:** Large balls (s=400) with thick sticks (linewidth=4)
- **Meaning:** As training proceeds (NC2), these should converge toward the green ETF vertices
- **Connection:** Sticks from origin show the direction and magnitude of each class centroid

#### 3. **Small Blue Spheres: Last-Layer Features (h_i)**
- **What:** Individual feature vectors from the penultimate layer (before final classification)
- **Color:** Blue, same shade as their class mean, semi-transparent
- **Size:** Small spheres (s=25, alpha=0.3)
- **Meaning:** As training proceeds (NC1), these should collapse onto their class mean (reduced within-class variance)

#### 4. **Red Ball-and-Sticks: Linear Classifiers (W)**
- **What:** The weight vectors from the final linear layer (W in W*h+b)
- **Color:** Red with different shades for each class
- **Size:** Large balls (s=400) with thick sticks (linewidth=4)
- **Meaning:** As training proceeds (NC3), these should align with their corresponding class means (self-duality)
- **Note:** Each class has its own classifier weight vector

#### 5. **Orange Ball-and-Sticks: Bias Vectors (b)**
- **What:** The bias terms from the final linear layer (b in W*h+b)
- **Color:** Orange with different shades for each class
- **Size:** Medium balls (s=300, square markers) with dashed sticks (linewidth=3.5)
- **Meaning:** Show how biases antialign as training proceeds
- **For 3 classes:** Biases should separate toward 120° apart (equiangular)
- **NEW:** This is our addition to show bias dynamics not shown in original paper

#### 6. **Black X: Origin**
- **What:** The zero point in feature space
- **Color:** Black
- **Meaning:** Reference point for all vectors

## Neural Collapse Phenomena Visualized

### NC1: Variability Collapse (Within-Class Compression)
- **Visual:** Small blue spheres clustering tightly around blue balls
- **Metric:** Within-class variance → 0
- **Interpretation:** Features from the same class become nearly identical

### NC2: Convergence to Simplex ETF (Between-Class Structure)
- **Visual:** Blue balls (class means) moving toward green spheres (ETF)
- **Metric:** ETF alignment → 1.0
- **Interpretation:** Class means form optimal geometric configuration (equidistant, centered)

### NC3: Self-Duality (Classifier-Mean Alignment)
- **Visual:** Red balls (classifiers) overlapping with blue balls (class means)
- **Metric:** Cosine similarity between W and μ → 1.0
- **Interpretation:** Classifiers become proportional to class means

### Bias Antialignment (Our Addition)
- **Visual:** Orange balls (biases) separating to equiangular positions
- **Metric:** For 3 classes, angles between biases → 120°
- **Interpretation:** Biases position the decision boundaries optimally

## Mathematical Details

### Simplex ETF Construction
For C classes, the Simplex ETF consists of C vectors in R^(C-1) with properties:
- Unit norm: ||v_c|| = 1
- Centered: Σ v_c = 0
- Equiangular: ⟨v_i, v_j⟩ = -1/(C-1) for i ≠ j

For 3 classes:
- Vectors in R^2 
- Form equilateral triangle
- 120° separation

### Projection to 3D
High-dimensional features (p >> 3) are projected using SVD:
1. Compute SVD of class means: M = U Σ V^T
2. Take top 3 right singular vectors: P = V[:, :3]
3. Project all objects: x_3d = x · P

This captures the subspace where Neural Collapse occurs.

### Bias Visualization
Biases are scalar offsets in the output space. For visualization:
- Position: b_c scaled in direction of class mean
- Magnitude: |b_c| determines distance from origin
- Angles: Computed between bias position vectors

## Color Coding Philosophy

**From the paper's description:**
> "Green spheres represent the vertices of the standard Simplex ETF, red ball-and-sticks represent linear classifiers, blue ball-and-sticks represent class-means, and small blue spheres represent last-layer features. For all objects, we distinguish different classes via the shade of the color."

We follow this EXACTLY with the addition of:
- **Orange for biases:** Distinct from W (red) and μ (blue) but harmonious
- **Shading:** Progressively lighter/darker shades distinguish classes within each object type

## Usage

### During Training
```python
from analysis.neural_collapse import NeuralCollapseAnalyzer

# Analyzer is created with model
nc_analyzer = NeuralCollapseAnalyzer(num_classes=3, feature_dim=hidden_dim)

# Snapshots are captured at intervals
# In run_experiment.py, configured via:
config.analysis.nc_snapshot_interval = 2500  # Every 2500 steps
```

### Generating Visualizations
```python
# Automatic during training (if save_nc_visualizations=True)
# Or manually:
nc_analyzer.visualize_neural_collapse(
    snapshot=snapshot,
    selected_classes=[0, 1, 2],  # Which classes to show
    samples_per_class=50,         # Features per class
    elevation=20,                 # 3D view angle
    azimuth=45,                   # 3D view angle
    save_path='nc_viz.png'
)
```

### Interpreting Results

**Early Training:**
- Blue spheres widely scattered (high NC1)
- Blue balls far from green spheres (low NC2)
- Red balls not aligned with blue balls (low NC3)
- Orange balls randomly positioned

**Terminal Phase (Neural Collapse):**
- Blue spheres collapsed onto blue balls (NC1 ≈ 0)
- Blue balls overlapping green spheres (NC2 ≈ 1)
- Red balls overlapping blue balls (NC3 ≈ 1)
- Orange balls separated 120° apart (optimal bias configuration)

**With Dynamic Class Focus (Bumping):**
- Observe if NC accelerates or changes character
- Check if bumping affects final geometry
- Compare bias evolution with/without bumping

## Comparison With/Without Bumping

Run both experiments:
```bash
python run_nc_experiment.py --cluster brigit --mode both
```

Compare:
1. **NC metrics evolution:** `nc_metrics_evolution.png`
2. **3D snapshots:** `nc_viz_step_*.png` at same steps
3. **Final geometry:** Last snapshot shows terminal configuration
4. **Bias angles:** Title shows average bias separation angle

## Output Files

Generated in `output_dir/` (e.g., `experiments/nc_with_dynamics/`):

- `nc_viz_step_00000000.png`: Initial state
- `nc_viz_step_00002500.png`: After 2500 steps
- `nc_viz_step_00005000.png`: After 5000 steps
- ... (every nc_snapshot_interval steps)
- `nc_metrics_evolution.png`: NC1/NC2/NC3 over time
- `nc_snapshots.pkl`: Raw data for further analysis

## Technical Notes

### Why Different Sizes?
- **ETF largest:** Reference configuration, most important
- **Class means & classifiers:** Key objects being tracked
- **Features smallest:** Too many to show all, just sampling
- **Biases medium:** Important but secondary to W

### Why Shades?
Following the paper: "we distinguish different classes via the shade of the color"
- Makes it clear which class each object belongs to
- Maintains the semantic grouping (all blue = data-driven, all red = classifiers, etc.)

### Viewing Angles
Default: elevation=20°, azimuth=45°
- Chosen to show 3D structure clearly
- Can be adjusted for different perspectives

### Fixed Axes for Temporal Comparison
**CRITICAL:** All visualizations use **fixed axis limits** based on the Simplex ETF scale.

- **Why:** Allows direct visual comparison across training
- **Scale:** ETF max value × 1.5 (provides 50% margin)
- **Effect:** Green ETF spheres stay in same position across all snapshots
- **Benefit:** Can clearly see blue/red/orange vectors moving toward/aligning with ETF

Without fixed axes, each plot would auto-scale, making movement impossible to see.

**Example:** If axis_limit = 2.0:
- All plots show range [-2.0, 2.0] on all axes
- Early training: Blue/red balls far from green ETF
- Late training: Blue/red balls overlap green ETF
- Movement is clearly visible!

To override with custom limit:
```python
nc_analyzer.visualize_neural_collapse(
    snapshot=snapshot,
    axis_limit=2.5  # Custom fixed limit
)
```

To use auto-scaling (not recommended for comparison):
```python
nc_analyzer.visualize_neural_collapse(
    snapshot=snapshot,
    axis_limit=None  # Auto-scale each plot independently
)
```

### Performance
- Only visualizes subset of features (default: 50 per class)
- Full dataset captured in snapshots for metrics
- High-res PNG (dpi=300) for publication quality

## References

**Main Paper:**
Papyan, V., Han, X. Y., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. *Proceedings of the National Academy of Sciences*, 117(40), 24652-24663.

**Figure 1:** The exact visualization we reproduce (plus bias dynamics)

## Future Enhancements

Possible additions:
- Animation showing temporal evolution
- Interactive 3D plots (Plotly)
- Multiple viewing angles in single figure
- Quantitative comparison overlays
- Decision boundary visualization
