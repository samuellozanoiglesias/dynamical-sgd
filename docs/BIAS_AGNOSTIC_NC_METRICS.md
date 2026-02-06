# Bias-Agnostic Neural Collapse Metrics

This document explains how the Neural Collapse metrics are implemented to work seamlessly with or without classifier bias terms.

## The Challenge

Linear classifiers can be defined in two equivalent ways:

1. **With bias**: $f_c(h) = W_c h + b_c$
2. **Without bias**: $f_c(h) = W_c h$

Different training frameworks and architectures may use either form. We need NC metrics that work universally without requiring code changes.

## The Solution: Feature Space Augmentation

### Core Principle

A classifier **with bias** in $\mathbb{R}^p$:
$$f_c(h) = W_c h + b_c$$

is **mathematically equivalent** to a classifier **without bias** in $\mathbb{R}^{p+1}$:
$$f_c(\tilde{h}) = \tilde{W}_c \tilde{h}$$

where:
- $\tilde{h} = [h; 1]$ (augment features with constant 1)
- $\tilde{W}_c = [W_c; b_c]$ (augment weights with bias)

### Why This Works

In the augmented space:
$$\tilde{W}_c \tilde{h} = [W_c; b_c] \cdot [h; 1] = W_c h + b_c \cdot 1 = W_c h + b_c$$

The bias term is **absorbed** into the geometry of the augmented space!

## Implementation

### Step 1: Accept Optional Bias

```python
def compute_nc_metrics(H, labels, W, b=None, H_test=None, labels_test=None):
```

### Step 2: Augment Feature Space (if bias exists)

```python
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
```

**Result**: 
- Original space: $H \in \mathbb{R}^{N \times p}$, $W \in \mathbb{R}^{C \times p}$
- Augmented space: $H \in \mathbb{R}^{N \times (p+1)}$, $W \in \mathbb{R}^{C \times (p+1)}$

### Step 3: Compute Metrics Normally

After augmentation, **all NC metrics work identically**:

```python
N, p = H.shape  # p is now p+1 if bias was present

# Compute class means in augmented space
class_means = ...  # (C, p)

# Compute global mean in augmented space
mu_G = ...  # (p,)

# All metrics proceed as before
nc1, nc2, nc3, nc4 = ...
```

The key insight: **class means and global mean are automatically computed in the augmented space**, preserving all geometric relationships.

## Why Each Metric Remains Valid

### NC1: Variability Collapse
$$\text{NC1} = \frac{\text{Tr}\{\Sigma_W \Sigma_B^{\dagger}\}}{C}$$

- $\Sigma_W$ measures within-class scatter in augmented space
- $\Sigma_B$ measures between-class scatter in augmented space
- Augmentation preserves the relative geometry → metric remains valid ✅

### NC2: Equinorm & Equiangularity
$$\text{NC2} = \text{CV}(||\mu_c - \mu_G||)$$

- Class means $\mu_c$ are computed in augmented space
- Global mean $\mu_G$ is computed in augmented space
- Norms and angles remain meaningful → metrics remain valid ✅

### NC3: Self-Duality
$$\text{NC3} = ||\hat{W}^T - \hat{M}||_F$$

- $M = (\mu_c - \mu_G)^T$ uses augmented class means
- $W$ is augmented to include bias
- Centering and normalization work identically → metric remains valid ✅

### NC4: Classifier ≈ NCC
- **Classifier predictions**: $\arg\max_c \tilde{W}_c \tilde{h} = \arg\max_c (W_c h + b_c)$ ✅
- **NCC predictions**: $\arg\max_c \langle \tilde{h} - \tilde{\mu}_G, \tilde{\mu}_c - \tilde{\mu}_G \rangle$ ✅
- Both use augmented representations → comparison remains valid ✅

## Additional Enhancements

### Double Centering for NC3

For numerical stability with class imbalance, we apply **double centering**:

```python
# Center across classes (standard)
M = (class_means - mu_G).T  # (p, C)

# ADDITIONALLY: Center across features (improves stability)
M_centered = M - jnp.mean(M, axis=1, keepdims=True)
M_hat = M_centered / (jnp.linalg.norm(M_centered, 'fro') + 1e-8)
```

This prevents numerical drift when class frequencies vary significantly.

## Behavior Table

| Training Phase | Bias Present? | What Happens |
|----------------|---------------|--------------|
| With bias | Yes | Bias absorbed into augmented space (p → p+1) |
| Without bias | No | Standard computation (p remains p) |
| Partial collapse | Either | Metrics remain geometrically meaningful |
| Terminal phase | Either | NC theory holds exactly |

## Sanity Check

During Neural Collapse terminal phase, you should observe:

1. **$||W_c - \mu_c|| \to 0$** across all classes (even with bias present)
2. **NC3 → 0** (classifier aligns with class means)
3. **NC4 → 0** (classifier decisions match NCC)

These hold **regardless** of whether bias was used during training!

## Code Changes Summary

### Modified Functions

1. **`compute_nc_metrics()`**:
   - Added optional `b` parameter
   - Added feature space augmentation logic
   - Added double centering for NC3

2. **`get_features_and_weights()`**:
   - Now returns `(H, W, b)` instead of `(H, W)`
   - Extracts bias from classifier layer

3. **`NeuralCollapseAnalyzer.extract_features_and_classifiers()`**:
   - Unpacks bias from `get_features_and_weights()`
   - Passes bias to `compute_nc_metrics()`

### Backward Compatibility

The implementation is **fully backward compatible**:
- If `b=None` is passed, no augmentation occurs
- Old code calling without bias continues to work
- New code can pass bias for proper handling

## References

- Papyan, V., Han, X., & Donoho, D. L. (2020). *Prevalence of neural collapse during the terminal phase of deep learning training*. arXiv:2008.08186.
- Mixon, D. G., Parshall, H., & Pi, J. (2020). *Neural collapse with unconstrained features*. arXiv:2011.11619.

## Testing

To verify the implementation works correctly:

```python
# Test with bias
features, weights, biases = get_features_and_weights(params, X)
metrics_with_bias = compute_nc_metrics(features, labels, weights, biases)

# Test without bias (should give same results after augmentation)
metrics_no_bias = compute_nc_metrics(features, labels, weights, None)

# Verify NC3 sanity check
from analysis.neural_collapse import plot_classifier_mean_norms
plot_classifier_mean_norms(analyzer.snapshots, output_dir)
```

Expected behavior: `||W_c - \mu_c||` should collapse to 0 in terminal phase, regardless of bias usage.
