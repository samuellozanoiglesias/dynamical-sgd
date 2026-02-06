# Simple Neural Collapse Metrics - Usage Guide

## Overview

The Neural Collapse implementation has been **simplified** to focus on the core metrics from the paper:
- **NC1**: Variability Collapse - `Tr(Σ_W @ Σ_B⁻¹) / C`
- **NC2**: Equinorm - Coefficient of Variation of `||μ_c - μ_G||`
- **NC2**: Equiangularity - Std and Mean of off-diagonal Gram matrix
- **NC3**: Self-Duality - `||M̂ - Ŵ||²_F`

All complex visualizations and angle tracking have been removed. The system now focuses purely on the mathematical metrics.

## Quick Start

### Run Neural Collapse Experiments

The simplified version works seamlessly with your existing workflow:

```bash
# Run with bumping (dynamics enabled)
python run_nc_experiment.py --cluster brigit --mode with_dynamics --config_file nc_config_baseline.yaml

# Run without bumping (standard training)  
python run_nc_experiment.py --cluster brigit --mode without_dynamics --config_file nc_config_baseline.yaml

# Run both for comparison
python run_nc_experiment.py --cluster brigit --mode both --config_file nc_config_baseline.yaml
```

### What You Get

After training completes, you'll find in your output directory:

1. **`nc_metrics_paper_style.png`** - Main metrics plot with 5 subplots:
   - Figure 6: NC1 - Variability Collapse
   - Figure 2: NC2 - Equinorm (CV)
   - Figure 3: NC2 - Equiangularity (Std)
   - Figure 4: NC2 - Equiangularity (Mean)
   - Figure 5: NC3 - Self-Duality

2. **`nc_snapshots.pkl`** - Raw snapshots for further analysis
3. **`config.yaml`** - Full experiment configuration
4. **`results.txt`** - Training metrics and final accuracies

### Output Example

```
Results saved to:
  - /data/samuel_lozano/dynamical-sgd/nc_with_dynamics/nc_config_baseline/experiment_2026_02_03-14_30_00/
  
Files generated:
  1. config.yaml: Full experiment configuration
  2. results.txt: Training metrics and final accuracies
  3. nc_metrics_paper_style.png: All NC metrics in one plot
  4. nc_snapshots.pkl: Raw snapshots for analysis
  5. training_curves.png: Loss and accuracy progression
```

```python
from analysis.neural_collapse_integration import (
    compute_nc_metrics,
    get_features_and_weights,
    plot_nc_metrics
)

# During training, at each epoch (or snapshot interval):
H, W = get_features_and_weights(params, X_train)
labels = jnp.argmax(Y_train, axis=1)  # Convert one-hot to integers
metrics = compute_nc_metrics(H, labels, W)

# Store metrics
metrics_history.append((epoch, metrics))

# After training, plot results
plot_nc_metrics(metrics_history, output_dir)
```

## Core Functions

### 1. `compute_nc_metrics(H, labels, W)`

Computes all Neural Collapse metrics from the paper.

**Arguments:**
- `H`: Last-layer features, shape `(N, p)` where N=num_samples, p=feature_dim
- `labels`: Class labels, shape `(N,)` as integers `[0, 1, ..., C-1]`
- `W`: Classifier weights, shape `(C, p)` where C=num_classes

**Returns:** Dictionary with:
- `nc1_variability`: NC1 metric (Variability Collapse)
- `nc2_equinorm_cv`: NC2 Equinorm (Coefficient of Variation)
- `nc2_equiangular_std`: NC2 Equiangularity standard deviation
- `nc2_equiangular_mean`: NC2 Equiangularity mean
- `nc2_equiangular_target`: Theoretical target for equiangularity
- `nc3_alignment`: NC3 Self-Duality metric

### 2. `get_features_and_weights(params, X)`

Extracts features H and weights W from a trained model.

**Arguments:**
- `params`: Model parameters (list of (W, b) tuples from JAX/Stax)
- `X`: Input data, shape `(N, input_dim)`

**Returns:**
- `H`: Last-layer features `(N, p)`
- `W`: Classifier weights `(C, p)`

### 3. `plot_nc_metrics(metrics_history, output_dir, log_scale=True)`

Creates plots matching the paper's figures.

**Arguments:**
- `metrics_history`: List of `(epoch, metrics_dict)` tuples
- `output_dir`: Directory to save plots
- `log_scale`: Use log scale for y-axis (default: True)

**Output:** Saves `nc_metrics_paper_style.png` with 5 subplots:
- Figure 6: NC1 - Variability Collapse
- Figure 2: NC2 - Equinorm (CV)
- Figure 3: NC2 - Equiangularity (Std)
- Figure 4: NC2 - Equiangularity (Mean)
- Figure 5: NC3 - Self-Duality

## Mathematical Definitions

### Global Mean and Class Means

$$\mu_c = \frac{1}{N_c} \sum_{i: y_i=c} h_i$$

$$\mu_G = \frac{1}{C} \sum_{c=1}^C \mu_c$$

### NC1: Variability Collapse (Figure 6)

Within-Class Scatter:
$$\Sigma_W = \frac{1}{N} \sum_{c=1}^C \sum_{i: y_i=c} (h_i - \mu_c)(h_i - \mu_c)^T$$

Between-Class Scatter:
$$\Sigma_B = \frac{1}{C} \sum_{c=1}^C (\mu_c - \mu_G)(\mu_c - \mu_G)^T$$

**Metric:**
$$\text{NC1} = \frac{\text{Tr}(\Sigma_W \Sigma_B^{-1})}{C}$$

**Target:** → 0 (features collapse to class means)

### NC2: Equinorm (Figure 2)

Calculate norms of centered class means:
$$n_c = ||\mu_c - \mu_G||_2$$

**Metric:**
$$\text{CV} = \frac{\text{Std}(n_c)}{\text{Mean}(n_c)}$$

**Target:** → 0 (all class means have equal norm)

### NC2: Equiangularity (Figures 3 & 4)

Normalize centered class means:
$$\tilde{m}_c = \frac{\mu_c - \mu_G}{||\mu_c - \mu_G||_2}$$

Gram matrix:
$$G = \tilde{M}^T \tilde{M} \text{ where } M = [\tilde{m}_1, \ldots, \tilde{m}_C]$$

**Metric (Figure 3):**
$$\text{Std}(G_{\text{off-diagonal}})$$

**Target:** → 0 (all angles equal)

**Metric (Figure 4):**
$$\text{Mean}(G_{\text{off-diagonal}})$$

**Target:** → $-\frac{1}{C-1}$ (simplex configuration)

### NC3: Self-Duality (Figure 5)

Normalize matrices:
$$\hat{M} = \frac{M}{||M||_F}, \quad \hat{W} = \frac{W^T}{||W||_F}$$

**Metric:**
$$\text{NC3} = ||\hat{M} - \hat{W}||_F^2$$

**Target:** → 0 (class means and classifiers align)

## Integration Example

Here's a complete training loop example:

```python
import jax.numpy as jnp
from pathlib import Path
from analysis.neural_collapse_integration import (
    compute_nc_metrics, 
    get_features_and_weights, 
    plot_nc_metrics
)

# Initialize
metrics_history = []
snapshot_interval = 50  # Compute metrics every 50 epochs

# Training loop
for epoch in range(num_epochs + 1):
    # Your training step
    params = train_step(params, X_train, Y_train)
    
    # Compute NC metrics periodically
    if epoch % snapshot_interval == 0:
        # Extract features and weights
        H, W = get_features_and_weights(params, X_train)
        
        # Convert one-hot labels to integers
        labels = jnp.argmax(Y_train, axis=1)
        
        # Compute metrics
        metrics = compute_nc_metrics(H, labels, W)
        
        # Store
        metrics_history.append((epoch, metrics))
        
        # Print
        print(f"Epoch {epoch}:")
        print(f"  NC1: {metrics['nc1_variability']:.6f}")
        print(f"  NC2 CV: {metrics['nc2_equinorm_cv']:.6f}")
        print(f"  NC3: {metrics['nc3_alignment']:.6f}")

# After training, generate plots
output_dir = Path('outputs/my_experiment')
plot_nc_metrics(metrics_history, output_dir)
```

## Expected Behavior

During successful Neural Collapse, you should observe:

1. **NC1 decreases**: Features collapse to class means
2. **NC2 CV decreases**: Class means become equinorm (equal magnitude)
3. **NC2 Std decreases**: Angles between class means become equal
4. **NC2 Mean approaches -1/(C-1)**: Class means form simplex configuration
5. **NC3 decreases**: Classifiers align with class means

## File Locations

- **Core functions**: `analysis/neural_collapse_integration.py`
- **Example script**: `simple_nc_example.py`
- **This guide**: `docs/SIMPLE_NC_USAGE.md`

## Tips

1. **Snapshot interval**: Start with every 50-100 epochs to avoid overhead
2. **Log scale**: Use `log_scale=True` for NC1 and NC3 plots (they decay exponentially)
3. **Class balance**: Metrics assume roughly balanced classes
4. **Feature dimension**: Ensure hidden layer is large enough (p ≥ C-1)

## Troubleshooting

**Q: Metrics are NaN or Inf**
- Check that your model is training properly (loss decreasing)
- Ensure feature dimension p > 0
- Verify labels are integers in range [0, C-1]

**Q: NC metrics not converging**
- Train for more epochs (Neural Collapse happens in terminal phase)
- Increase learning rate or decrease regularization
- Ensure model can overfit the training data

**Q: Shapes don't match**
- `H` should be (N, p) - features from penultimate layer
- `W` should be (C, p) - last layer weights transposed if needed
- `labels` should be (N,) - integer class labels, not one-hot

## References

This implementation follows the mathematical definitions from:

**Papyan, V., Han, X. Y., & Donoho, D. L. (2020).** *Prevalence of neural collapse during the terminal phase of deep learning training.* Proceedings of the National Academy of Sciences, 117(40), 24652-24663.
