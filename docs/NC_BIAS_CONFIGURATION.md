# Neural Collapse with Configurable Bias Support

## Overview

The Neural Collapse implementation now fully supports **both biased and unbiased classifiers** through a configuration flag. This allows you to train models with or without bias terms in the **final classifier layer** while maintaining mathematically correct NC metrics.

**Important**: The `use_bias` flag applies **only to the classifier** (final layer). Hidden layers always use bias, which is standard practice in deep learning architectures.

## Configuration

### Setting the Bias Flag

When initializing the `NeuralCollapseAnalyzer`, pass the `use_bias` parameter:

```python
from analysis.neural_collapse import NeuralCollapseAnalyzer

# With bias (default)
analyzer = NeuralCollapseAnalyzer(
    num_classes=5,
    feature_dim=128,
    num_hidden_layers=1,
    use_batchnorm=True,
    use_bias=True  # ← Enables bias support
)

# Without bias
analyzer = NeuralCollapseAnalyzer(
    num_classes=5,
    feature_dim=128,
    num_hidden_layers=1,
    use_batchnorm=True,
    use_bias=False  # ← Disables bias
)
```

### YAML Configuration

In your experiment config file:

```yaml
model:
  use_bias: true  # or false

neural_collapse:
  use_bias: ${model.use_bias}  # Inherit from model config
  feature_dim: 128
  num_classes: 5
```

## How It Works

### 1. With Bias (`use_bias=True`)

When bias is enabled in the classifier:

#### Feature Extraction
- Hidden layers: `h = W·x + b` (always have bias)
- Classifier: `logits = W_c·h + b_c` (bias enabled)

#### Feature Space Augmentation
The code automatically augments the feature space to absorb bias:

```python
# Augment features: h̃ = [h; 1]
H_augmented = jnp.concatenate([H, jnp.ones((N, 1))], axis=1)

# Augment weights: W̃ = [W; b]
W_augmented = jnp.concatenate([W, b[:, None]], axis=1)
```

**Result**: Bias-free classifier in $(p+1)$-dimensional space
- $f_c(\tilde{h}) = \tilde{W}_c \tilde{h} = W_c h + b_c$ (mathematically equivalent!)

#### NC Metrics
All metrics are computed in the augmented space:
- **NC1**: Within/between-class covariance in $\mathbb{R}^{p+1}$
- **NC2**: Class mean norms and angles in $\mathbb{R}^{p+1}$
- **NC3**: Classifier-mean alignment in $\mathbb{R}^{p+1}$
- **NC4**: Classifier vs NCC predictions (bias absorbed)

### 2. Without Bias (`use_bias=False`)

When bias is disabled in the classifier:

#### Feature Extraction
- Hidden layers: `h = W·x + b` (still have bias)
- Classifier: `logits = W_c·h` (no bias in classifier)

#### No Augmentation
```python
b = None  # No bias extracted
# compute_nc_metrics(H, labels, W, b=None, ...)
# → No augmentation occurs
```

**Result**: Standard computation in original $p$-dimensional space

#### NC Metrics
All metrics are computed in the original space:
- **NC1**: Within/between-class covariance in $\mathbb{R}^p$
- **NC2**: Class mean norms and angles in $\mathbb{R}^p$
- **NC3**: Classifier-mean alignment in $\mathbb{R}^p$
- **NC4**: Classifier vs NCC predictions (no bias)

## Implementation Details

### Modified Functions

#### `get_features_and_weights()`

**Signature**:
```python
def get_features_and_weights(
    params: Any,
    X: jnp.ndarray,
    num_hidden_layers: int = 1,
    use_batchnorm: bool = True,
    use_bias: bool = True  # ← NEW parameter
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
```

**Behavior**:
- `use_bias=True`: Returns `(H, W, b)` where `b` is the classifier bias vector
- `use_bias=False`: Returns `(H, W, None)` where classifier bias is `None`
- **Note**: Hidden layers always extract with bias regardless of `use_bias` flag

**Parameter Extraction**:
```python
# Hidden layers - ALWAYS use bias
W_dense, b_dense = dense_params
h = jnp.dot(h, W_dense) + b_dense

# Classifier - depends on use_bias flag
if use_bias:
    W_last, b_last = classifier_params
    W = W_last.T  # (C, p)
    b = b_last    # (C,)
else:
    W_last = classifier_params[0]
    W = W_last.T  # (C, p)
    b = None
```

#### `NeuralCollapseAnalyzer.__init__()`

**Signature**:
```python
def __init__(
    self,
    num_classes: int,
    feature_dim: int,
    num_hidden_layers: int = 1,
    use_batchnorm: bool = True,
    use_bias: bool = True  # ← NEW parameter
):
```

**Stores**:
```python
self.use_bias = use_bias  # Saved for feature extraction
```

#### `compute_nc_metrics()`

**Already supports optional bias** (no changes needed):
```python
def compute_nc_metrics(
    H: jnp.ndarray,
    labels: jnp.ndarray,
    W: jnp.ndarray,
    b: Optional[jnp.ndarray] = None,  # ← Optional
    H_test: Optional[jnp.ndarray] = None,
    labels_test: Optional[jnp.ndarray] = None
) -> dict:
```

**Automatic behavior**:
- If `b is not None`: Augments feature space
- If `b is None`: Uses original feature space

## Usage Examples

### Example 1: Standard Training with Bias

```python
# Config
config = {
    'model': {'use_bias': True},
    'nc': {'num_classes': 5, 'feature_dim': 128}
}

# Initialize
analyzer = NeuralCollapseAnalyzer(
    num_classes=config['nc']['num_classes'],
    feature_dim=config['nc']['feature_dim'],
    use_bias=config['model']['use_bias']
)

# Extract features during training
snapshot = analyzer.extract_features_and_classifiers(
    model_fn, params, X_train, Y_train, epoch,
    X_test=X_test, Y_test=Y_test
)

# Metrics automatically use augmented space
print(snapshot.metrics['nc3_self_duality'])  # Uses R^{p+1}
```

### Example 2: Training without Bias

```python
# Config
config = {
    'model': {'use_bias': False},  # ← No bias
    'nc': {'num_classes': 5, 'feature_dim': 128}
}

# Initialize
analyzer = NeuralCollapseAnalyzer(
    num_classes=config['nc']['num_classes'],
    feature_dim=config['nc']['feature_dim'],
    use_bias=config['model']['use_bias']  # ← False
)

# Extract features during training
snapshot = analyzer.extract_features_and_classifiers(
    model_fn, params, X_train, Y_train, epoch,
    X_test=X_test, Y_test=Y_test
)

# Metrics use original space
print(snapshot.metrics['nc3_self_duality'])  # Uses R^p
```

### Example 3: Comparing Both Configurations

Run two experiments side-by-side:

```bash
# With bias
python run_nc_experiment.py --config nc_config_with_bias.yaml

# Without bias
python run_nc_experiment.py --config nc_config_no_bias.yaml
```

Compare:
- NC metric evolution plots
- Classifier-mean alignment (`||W_c - μ_c||`)
- Terminal phase convergence speed

## Expected Behavior

### During Neural Collapse

**Both configurations should show**:
1. **NC1 → 0** (variability collapse)
2. **NC2 → 0** (equinorm and equiangularity)
3. **NC3 → 0** (classifier-mean alignment)
4. **NC4 → 0** (classifier ≈ NCC)

### Sanity Check Plot

The `plot_classifier_mean_norms()` function shows $||W_c - \mu_c||$ over training:

**With bias**: Shows norms in augmented $\mathbb{R}^{p+1}$ space
**Without bias**: Shows norms in original $\mathbb{R}^p$ space

Both should collapse to 0 during terminal phase! ✅

## Theoretical Justification

### Why Feature Augmentation Works

The augmentation $\tilde{h} = [h; 1]$ and $\tilde{W} = [W; b]$ is not just a computational trick—it's **geometrically meaningful**:

1. **Preserves Decision Boundaries**: 
   - $\tilde{W}_c \tilde{h} = W_c h + b_c$ (exact equivalence)

2. **Maintains Collapse Geometry**:
   - Class means in $\mathbb{R}^{p+1}$ encode both mean features AND bias
   - Simplex ETF structure preserved in augmented space

3. **Theoretical Alignment**:
   - Papyan's NC theory holds in **any** feature space
   - Augmentation is just a change of coordinates

### When to Use Each Configuration

**Use bias (`use_bias=True`)** when:
- Training standard neural networks
- Matching typical deep learning practice
- Comparing against existing NC literature

**Use no bias (`use_bias=False`)** when:
- Studying pure geometric collapse (no affine offset)
- Analyzing simplex ETF in minimal dimensionality
- Investigating theoretical NC properties

## Error Handling

### Common Issues

**Issue**: Model trained with bias, but analyzer initialized with `use_bias=False`

**Error**: 
```
ValueError: too many values to unpack (expected 1)
```

**Fix**: Ensure `use_bias` matches your model architecture!

**Issue**: Model trained without bias, but analyzer initialized with `use_bias=True`

**Error**:
```
ValueError: not enough values to unpack (expected 2, got 1)
```

**Fix**: Set `use_bias=False` in analyzer configuration.

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The code will print:
- `use_bias` flag value
- Parameter structure inspection
- Classifier extraction details

## Testing Checklist

- [ ] Config file has `model.use_bias` set correctly
- [ ] `NeuralCollapseAnalyzer` initialized with matching `use_bias`
- [ ] Model architecture uses `use_bias` in Dense layers
- [ ] NC metrics computed without errors
- [ ] Sanity check plot shows convergence
- [ ] Both with/without bias experiments complete successfully

## References

- Papyan, V., Han, X., & Donoho, D. L. (2020). *Prevalence of neural collapse during the terminal phase of deep learning training*. arXiv:2008.08186.
- Zhou, J., Li, Q., & Zhu, Z. (2022). *Neural collapse with cross-entropy loss*. arXiv:2210.08863.

---

**Note**: This implementation provides **mathematically rigorous** handling of bias through feature space augmentation, ensuring NC metrics remain valid regardless of whether bias is used! 🎯
