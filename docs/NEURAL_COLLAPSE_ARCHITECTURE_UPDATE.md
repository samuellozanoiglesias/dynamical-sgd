# Neural Collapse Architecture Update

## Summary

The SpiralClassifier has been updated to strictly follow the Neural Collapse paper architecture (Papyan et al., 2020).

## Key Changes

### 1. **Architecture Split** (h(x) vs Classifier)

The model now explicitly separates:
- **Feature Extractor h(x)**: `x → [Dense → BatchNorm → ReLU] × num_hidden_layers → h(x)`
- **Linear Classifier**: `h(x) → Dense → logits` (implements `W·h(x) + b`)

This matches the paper's mathematical definition where:
- `h: ℝ^d → ℝ^p` is the feature extractor
- `W ∈ ℝ^(C×p)` and `b ∈ ℝ^C` form the linear classifier

### 2. **VGG-Style Blocks** (Paper Recipe)

Replaced simple layers with **Dense → BatchNorm → ReLU** blocks, following the paper's exact recipe:
- **Batch Normalization**: Essential for Neural Collapse (replaces Dropout)
- **No Dropout**: Removed to allow rigid feature collapse
- **Bias Term**: Final layer includes bias (necessary for simplex centering)

### 3. **Feature Extraction Method**

Added `get_features(params, X)` method to extract last-layer features `h(x)` for:
- NC1: Variability collapse visualization
- NC2: Simplex geometry analysis
- NC3: Self-duality measurements

### 4. **Configuration Parameter**

Added `num_hidden_layers` parameter (default: 2):
- Controls depth of feature extractor
- More layers → better spiral separation
- Paper uses 2-3 blocks for simple datasets

## Architecture Comparison

### Before (Simple MLP)
```
x → Dense(nn_width) → ReLU → Dense(num_classes) → logits
```

### After (NC-Compliant)
```
x → Dense(nn_width) → BatchNorm → ReLU 
  → Dense(nn_width) → BatchNorm → ReLU  (repeat num_hidden_layers times)
  → Dense(num_classes) → logits
  └────────── h(x) ──────────┘   └─ Classifier ─┘
```

## Paper's Training Recipe for Neural Collapse

### Critical Requirements:

1. **Overparameterization**: Use `nn_width` much larger than needed (e.g., 100-500 for spirals)

2. **Terminal Phase Training (TPT)**:
   - **DO NOT** stop at 100% accuracy
   - Continue training for hundreds/thousands of epochs
   - Neural Collapse emerges during this "terminal phase"
   - Set `bumps_TPT: false` to stop dynamics after perfect accuracy

3. **Weight Decay**: Use moderate L2 regularization (e.g., `l2_reg: 5e-4`)

4. **Batch Normalization**: Essential (now included in architecture)

## Code Changes

### Files Modified:

1. **`src/models/spiral_classifier.py`**:
   - Added `num_hidden_layers` parameter
   - Updated `_create_model()` with VGG-style blocks
   - Added `get_features()` method for h(x) extraction

2. **`analysis/neural_collapse.py`**:
   - Updated `get_features_and_weights()` to handle BatchNorm layers
   - Added `num_hidden_layers` to `NeuralCollapseAnalyzer`
   - Updated feature extraction to process all blocks

3. **`analysis/neural_collapse_integration.py`**:
   - Updated `extract_penultimate_features()` for new architecture
   - Added `num_hidden_layers` parameter to all functions

4. **`run_experiment.py`**:
   - Pass `num_hidden_layers` to SpiralClassifier
   - Pass `num_hidden_layers` to NeuralCollapseAnalyzer

5. **`config/experiment_config.py`**:
   - Added `num_hidden_layers: int = 2` to ModelConfig

6. **`average_nc_metrics.py`**:
   - Updated to use default `num_hidden_layers=2` for backward compatibility

## Usage

### Configuration Example:
```yaml
model:
  nn_width: 500          # Overparameterize!
  num_hidden_layers: 2   # Number of Dense → BatchNorm → ReLU blocks
  
optimizer:
  learning_rate: 0.01
  l2_reg: 5e-4          # Weight decay (paper value)
  
training:
  total_steps: 100000   # Train past 100% accuracy!
  
dynamics:
  bumps_TPT: false      # Stop bumps at 100% acc (Terminal Phase Training)
```

### Extracting Features h(x):
```python
# Get last-layer features
features = classifier.get_features(params, X_train)

# Or use integration function
from analysis.neural_collapse_integration import extract_penultimate_features
features = extract_penultimate_features(params, X_train, num_hidden_layers=2)
```

## Expected Neural Collapse Behavior

With this architecture and proper training:

1. **NC1 (Variability Collapse)**: Features within each class collapse to their mean
   - Metric → 0 during TPT

2. **NC2 (Equinorm)**: All class means have equal distance from global mean
   - CV of norms → 0

3. **NC2 (Equiangularity)**: Class means form a simplex with equal angles
   - Std of angles → 0
   - Mean angle → -1/(C-1)

4. **NC3 (Self-Duality)**: Classifier weights align with class means
   - ||M̂ - Ŵ||²_F → 0

## Backward Compatibility

- **Default**: `num_hidden_layers=2` maintains similar behavior to old single-layer
- **Old experiments**: Will work with default value
- **New experiments**: Can increase depth for better spiral classification

## References

Papyan, V., Han, X. Y., & Donoho, D. L. (2020). "Prevalence of Neural Collapse during the terminal phase of deep learning training." PNAS, 117(40), 24652-24663.
