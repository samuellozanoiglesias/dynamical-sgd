# Neural Collapse Visualization Fixes - Summary

## Changes Made (December 2025)

### Core Issues Fixed

1. **ETF Positioning** - Fixed to stay constant across all visualizations
2. **Vector Normalization** - All vectors normalized to unit length for clear angle comparison
3. **Projection Consistency** - Both 2D and 3D now use same SVD-based method
4. **Angle Measurements** - Comprehensive 4-line angle display in titles
5. **Bias Visualization** - Proper normalized representation of bias vectors

---

## Modified Functions in `analysis/neural_collapse.py`

### New Helper Functions Added

```python
def compute_angle_degrees(v1, v2) -> float
    """Compute angle in degrees between two vectors"""
    
def compute_pairwise_angles(vectors) -> array
    """Compute all pairwise angles between a set of vectors"""
```

### Modified Functions

#### `project_to_subspace()` - Major rewrite
**Changes:**
- Added `normalize` parameter (default: True)
- Now returns ETF as 5th output: `(features, means, classifiers, etf, basis)`
- ETF computed directly in target dimension (FIXED reference frame)
- All vectors normalized to unit length when `normalize=True`

**Old signature:**
```python
def project_to_subspace(features, class_means, classifiers, target_dim=3)
    -> (proj_features, proj_means, proj_classifiers, basis)
```

**New signature:**
```python
def project_to_subspace(features, class_means, classifiers, target_dim=3, normalize=True)
    -> (proj_features, proj_means, proj_classifiers, etf, basis)
```

#### `visualize_neural_collapse()` - 3D visualization
**Changes:**
- Uses updated `project_to_subspace()` with normalization
- ETF now stays fixed (not scaled to data)
- Bias visualization uses normalized directions
- Title shows 4 angle measurements:
  - Mean angles between class means
  - Classifier angles
  - Bias angles
  - Mean-classifier alignment
- Fixed axis limits: [-1.5, 1.5] for all axes (unit sphere + margin)

#### `visualize_neural_collapse_2d()` - 2D visualization  
**Changes:**
- Complete rewrite - now uses SAME projection as 3D
- Removed PCA-based projection (was inconsistent)
- Now calls `project_to_subspace(target_dim=2, normalize=True)`
- Bias visualization matches 3D approach
- Title shows same 4 angle measurements as 3D
- Fixed axis limits: [-1.5, 1.5] (consistent with 3D)
- Ensures `ax.set_aspect('equal')` for accurate angles

#### `compute_nc_metrics()` - Metric calculation
**Changes:**
- Updated to use new `project_to_subspace()` signature
- NC2 calculation now uses ETF from projection (more consistent)
- Added comments explaining greedy vs optimal Procrustes matching
- More robust normalization handling

---

## Files Modified

1. **`analysis/neural_collapse.py`** - Core visualization and metrics code
2. **`docs/NEURAL_COLLAPSE_INTERPRETATION_GUIDE.md`** - NEW: Comprehensive interpretation guide

---

## Key Improvements

### Before:
❌ ETF moves/scales between timesteps  
❌ Vectors have different magnitudes, some off-plot  
❌ 2D uses PCA, 3D uses SVD (inconsistent)  
❌ Title shows limited angle info  
❌ Bias visualization arbitrary  

### After:
✅ ETF fixed in position (green reference frame)  
✅ All vectors normalized to unit sphere  
✅ 2D and 3D use same SVD projection  
✅ Title shows 4 comprehensive angle measurements  
✅ Bias properly normalized and anti-aligned  

---

## Backward Compatibility

**Breaking changes:**
- `project_to_subspace()` now returns 5 values instead of 4 (added `etf`)
- Visualization scale changed (now fixed at [-1.5, 1.5])

**Impact:**
- Any code calling `project_to_subspace()` directly needs updating
- Old snapshots may need regeneration for proper comparison
- Visualization scale is now consistent but different from before

**Migration:**
```python
# Old:
proj_feat, proj_mean, proj_class, basis = analyzer.project_to_subspace(...)

# New:
proj_feat, proj_mean, proj_class, etf, basis = analyzer.project_to_subspace(...)
```

---

## Testing Recommendations

1. **Run new visualizations** and verify:
   - ETF stays in same position across timesteps
   - All vectors visible within plot bounds
   - Angles displayed correctly in titles
   - 2D and 3D show consistent geometry

2. **Compare metrics** with previous runs:
   - NC1 trend should be similar
   - NC2 values may differ slightly (improved calculation)
   - NC3 should be similar

3. **Check edge cases**:
   - Early training (scattered features)
   - Terminal phase (converged)
   - Different numbers of classes

---

## Usage

After these changes, visualizations will automatically:
- Show ETF in fixed positions
- Normalize all vectors for clarity
- Display comprehensive angle information
- Use consistent projection between 2D and 3D

No code changes needed in calling code (except handling the extra return value from `project_to_subspace`).

---

## Documentation

New documentation created:
- **`docs/NEURAL_COLLAPSE_INTERPRETATION_GUIDE.md`** - Complete guide on interpreting the fixed visualizations

Existing documentation still valid:
- `docs/NEURAL_COLLAPSE.md` - Theory and background
- `docs/NEURAL_COLLAPSE_VISUALIZATION.md` - Usage instructions

---

## Contact

For questions about these changes:
- **Author**: Samuel Lozano Iglesias
- **Email**: samuel.lozano@ucm.es
