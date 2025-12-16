# Neural Collapse Visualization and Metrics - Interpretation Guide

## Overview

This guide explains how to interpret Neural Collapse (NC) metrics and visualizations after the recent fixes that ensure correctness and consistency.

---

## What Was Fixed

### 1. **Simplex ETF Now Fixed in Position** ✅
- **Problem**: ETF vertices were being scaled and moved based on data, making them inconsistent across timesteps
- **Fix**: ETF is now computed directly in the projection subspace and stays FIXED at unit-normalized positions
- **Result**: Green spheres/circles stay in the same position across all visualizations, serving as the true reference frame

### 2. **All Vectors Normalized to Unit Length** ✅
- **Problem**: Class means, classifiers, and biases had different magnitudes, making some invisible or off-plot
- **Fix**: All vectors are normalized to unit length for visualization
- **Result**: Clear angular relationships visible on a unit sphere, everything stays within the plot bounds

### 3. **Consistent 2D and 3D Projection** ✅
- **Problem**: 2D used PCA on mixed data (different from 3D's SVD approach)
- **Fix**: Both 2D and 3D now use the same SVD-based projection onto the (C-1)-dimensional Neural Collapse subspace
- **Result**: 2D and 3D show the same geometry, just from different viewing angles

### 4. **Comprehensive Angle Measurements** ✅
- **Problem**: Titles only showed partial angle information ("Mean Angles")
- **Fix**: Now displays 4 key angle measurements in the title
- **Result**: Complete information about convergence to Neural Collapse at a glance

### 5. **Proper Bias Visualization** ✅
- **Problem**: Biases (scalars) were visualized arbitrarily
- **Fix**: Biases shown as unit-normalized vectors in the direction opposite to classifiers (showing anti-alignment)
- **Result**: Clear visualization of the bias anti-alignment property of Neural Collapse

---

## Understanding the Plots

### 3D Plots (`3d-snapshots/nc_viz_step_*.png`)

**Elements:**
- 🟢 **Green Spheres**: Simplex ETF vertices (FIXED theoretical optimal positions)
- 🔵 **Small Blue Dots**: Individual feature samples from the last layer
- 🔵 **Blue Spheres + Sticks**: Class means (centroids of features) - NORMALIZED
- 🔴 **Red Spheres + Sticks**: Classifier weight vectors - NORMALIZED
- 🟠 **Orange Squares + Dashed Sticks**: Bias vectors (anti-aligned) - NORMALIZED

**Title Information** (4 lines):
```
Neural Collapse at Step 5000
Means: 119.8° | Classifiers: 120.2° | Biases: 121.1° (Target: 120°)
Mean-Classifier Alignment: 2.3° (Target: 0°)
```

**Interpretation:**
1. **Means Angle (119.8°)**: Average angle between class mean pairs
   - Target: 120° for 3 classes (perfect simplex)
   - Closer to 120° = better NC2 (class means converging to ETF)

2. **Classifiers Angle (120.2°)**: Average angle between classifier pairs
   - Target: 120° for 3 classes
   - Shows geometric organization of decision boundaries

3. **Biases Angle (121.1°)**: Average angle between bias vectors
   - Target: 120° for 3 classes
   - Shows anti-alignment property (biases point opposite to classifiers)

4. **Mean-Classifier Alignment (2.3°)**: Average angle between each class mean and its corresponding classifier
   - Target: 0° (perfect alignment)
   - This is NC3 - shows self-duality

**What to Look For:**
- **NC1**: Blue dots clustering tightly around blue spheres
- **NC2**: Blue spheres (means) moving toward green spheres (ETF)
- **NC3**: Red spheres (classifiers) aligning with blue spheres (means)
- **All angles approaching 120°** and **alignment approaching 0°**

---

### 2D Plots (`2d-snapshots/nc_viz_step_*.png`)

**Same elements as 3D, projected to 2D:**
- 🟢 **Green Circles**: ETF vertices (FIXED positions)
- 🔵 **Blue Dots**: Features
- 🔵 **Blue Circles**: Class means (NORMALIZED)
- 🔴 **Red Triangles**: Classifiers (NORMALIZED)
- 🟠 **Orange Squares**: Biases (NORMALIZED)

**Title has same 4-line format as 3D**

**Advantages of 2D:**
- Easier to measure angles visually
- Clear view of 120° separations (forms perfect triangle)
- Better for understanding angular relationships
- Equal aspect ratio ensures angles are accurate

**What Perfect Neural Collapse Looks Like in 2D:**
```
         ETF (green)
              *
            /   \
          /       \
        /           \
      *-------*-------* 
     
    All at 120° angles
    Means, classifiers, and ETF perfectly aligned
    Biases pointing opposite directions
```

---

## Understanding NC Metrics Evolution Plot

**File**: `nc_metrics_evolution.png`

### NC1: Within-Class Variance (Left, Log Scale)
**What it measures**: Average squared distance of features from their class mean

**Calculation**:
```python
For each class c:
    variance_c = mean(||feature - class_mean_c||²)
NC1 = mean(variance_c across all classes)
```

**Interpretation**:
- **Should DECREASE** (log scale shows this clearly)
- Lower = features collapsing onto class means
- Target: Near 0 (perfect collapse)
- **Typical behavior**: Exponential decrease during terminal phase

### NC2: ETF Alignment (Middle)
**What it measures**: How well class means align with Simplex ETF vertices

**Calculation**:
```python
Project class means to (C-1)-D subspace, normalize
Compute correlation with ETF vertices
NC2 = mean(max_correlation per mean)
```

**Interpretation**:
- **Should INCREASE toward 1.0** (red dashed line)
- 1.0 = perfect simplex configuration (120° angles for 3 classes)
- Measures NC2: "Convergence to Simplex ETF"
- **Typical behavior**: Increases during terminal phase, saturates at ~0.95-1.0

**Note**: Uses greedy matching (not optimal Procrustes). Values 0.9-1.0 indicate strong convergence.

### NC3: Self-Duality (Right)
**What it measures**: Cosine similarity between classifiers and class means

**Calculation**:
```python
For each class c:
    similarity_c = cosine(classifier_c, class_mean_c)
NC3 = mean(similarity_c)
```

**Interpretation**:
- **Should INCREASE toward 1.0** (red dashed line)
- 1.0 = classifiers perfectly aligned with class means
- Measures NC3: "Self-Duality"
- **Typical behavior**: Increases during terminal phase, saturates at ~0.95-1.0

---

## Typical Neural Collapse Evolution

### Early Training (Steps 0-2000)
- **NC1**: High (~0.1-1.0) - Features scattered
- **NC2**: Low (~0.3-0.6) - Means not aligned with ETF
- **NC3**: Medium (~0.5-0.7) - Partial classifier-mean alignment
- **Visually**: Blue dots spread out, vectors pointing in various directions

### Mid Training (Steps 2000-5000)
- **NC1**: Decreasing (~0.01-0.1) - Features clustering
- **NC2**: Increasing (~0.6-0.9) - Means approaching ETF
- **NC3**: Increasing (~0.7-0.9) - Stronger alignment
- **Visually**: Blue dots clustering, vectors moving toward ETF positions

### Terminal Phase (Steps 5000+)
- **NC1**: Very low (<0.001) - Near-perfect clustering
- **NC2**: High (~0.9-1.0) - Simplex structure formed
- **NC3**: High (~0.95-1.0) - Strong self-duality
- **Visually**: Blue dots tight around means, all vectors aligned with ETF at 120° angles

---

## Comparing With vs. Without Dynamics

When comparing experiments with and without dynamic class focus:

### Look for:
1. **NC1 plots**: Does dynamics lead to faster collapse?
2. **NC2 plots**: Does dynamics help converge to simplex faster?
3. **NC3 plots**: Does dynamics improve self-duality?
4. **Angle measurements**: Do angles reach 120° more consistently with dynamics?
5. **Alignment**: Does mean-classifier alignment reach 0° better?

### In 2D/3D Snapshots:
- Compare same timestep across experiments
- Check if ETF vertices are in SAME positions (they should be!)
- See if vectors cluster faster around ETF with dynamics
- Observe tightness of feature clusters (NC1)

---

## Technical Details

### Projection Method (Both 2D and 3D)

1. **Compute SVD** of class means matrix $M \in \mathbb{R}^{C \times p}$:
   $$M = U \Sigma V^T$$

2. **Extract basis**: Top $k$ right singular vectors from $V$ (where $k=2$ for 2D, $k=3$ for 3D)

3. **Project**:
   - Features: $h_{\text{proj}} = h \cdot V_{:k}$
   - Class means: $\mu_{\text{proj}} = \mu \cdot V_{:k}$
   - Classifiers: $W_{\text{proj}} = W \cdot V_{:k}$

4. **Normalize** all vectors to unit length: $v_{\text{norm}} = \frac{v}{||v||}$

5. **Compute ETF** directly in $k$-dimensional space (stays FIXED)

### Simplex ETF for 3 Classes

The optimal configuration in 2D:
$$\text{ETF} = \begin{bmatrix}
\cos(0°) & \sin(0°) \\
\cos(120°) & \sin(120°) \\
\cos(240°) & \sin(240°)
\end{bmatrix} \times \text{scale}$$

Forms an equilateral triangle with vertices on the unit circle separated by 120°.

---

## Troubleshooting

### "ETF vertices are moving between plots"
- **Fixed**: They should now stay in the same position across all timesteps
- If still moving, check that `normalize=True` in `project_to_subspace` calls

### "Vectors going off the plot"
- **Fixed**: All vectors now normalized to unit length
- Fixed axis limits set to [-1.5, 1.5] for both 2D and 3D
- If still issues, check `normalize=True` is used

### "2D and 3D look completely different"
- **Fixed**: Both now use same SVD-based projection
- They should show the same geometry from different viewing angles
- ETF positions should be consistent (though rotated between 2D/3D views)

### "Angle measurements seem wrong"
- Ensure `ax.set_aspect('equal')` is set (2D only)
- Check that vectors are unit-normalized
- Verify angle computation uses proper inverse cosine

---

## Summary Checklist

When analyzing your NC results, verify:

- [ ] ETF vertices stay in same positions across timesteps ✅
- [ ] All vectors (means, classifiers, biases) visible within plot ✅
- [ ] 2D and 3D show consistent geometry ✅
- [ ] Title shows 4 angle measurements ✅
- [ ] Angles approach 120° for 3-class problems ✅
- [ ] Mean-classifier alignment approaches 0° ✅
- [ ] NC1 decreases over time ✅
- [ ] NC2 increases toward 1.0 ✅
- [ ] NC3 increases toward 1.0 ✅

---

## References

- Paper: "Prevalence of Neural Collapse during the terminal phase of deep learning training" (Papyan et al., 2020)
- See also: `docs/NEURAL_COLLAPSE.md` for theory
- Code: `analysis/neural_collapse.py`

---

**Author**: Samuel Lozano Iglesias  
**Email**: samuel.lozano@ucm.es  
**Date**: December 2025
