# Neural Collapse Visualization

This module implements the visualization and analysis of **Neural Collapse** phenomena as described in the paper:

> **"Prevalence of Neural Collapse during the terminal phase of deep learning training"**  
> Papyan, V., Han, X. Y., & Donoho, D. L. (2020)

## What is Neural Collapse?

Neural Collapse refers to a remarkable geometric phenomenon that occurs during the terminal phase of training (TPT) deep neural networks. As training progresses toward zero training loss, the network's last-layer geometry converges to a highly symmetric and organized structure characterized by four properties:

### The Four Properties of Neural Collapse

1. **NC1 (Variability Collapse)**: Within-class features collapse to their class means
   - Individual sample features cluster tightly around their class centroids
   - Within-class variance approaches zero

2. **NC2 (Convergence to Simplex ETF)**: Class means converge to vertices of a Simplex Equiangular Tight Frame
   - Class centroids form a perfect simplex configuration
   - All class pairs are equidistant and maximally separated
   - Inner product between different class means: $-1/(C-1)$ where $C$ is the number of classes

3. **NC3 (Self-Duality)**: Classifiers and class means become dual to each other
   - Last-layer classifier weights align perfectly with class means
   - The learned weight matrix becomes proportional to class mean matrix

4. **NC4 (Simplification to NCC)**: Decision boundaries simplify to Nearest Class Center
   - Classification reduces to finding the nearest class mean
   - Maximum margin decision boundaries emerge naturally

## Understanding Figure 1

Figure 1 from the paper visualizes Neural Collapse in 3D. Here's what each element represents:

- **🟢 Green Spheres**: Vertices of the Simplex ETF (theoretical optimal configuration)
- **🔴 Red Ball-and-Sticks**: Learned classifier weight vectors from the last layer
- **🔵 Blue Ball-and-Sticks**: Class means (centroids) of training data features
- **🔵 Small Blue Spheres**: Individual training sample features (sampled)

As training progresses:
- Blue spheres cluster tightly around blue sticks (NC1)
- Blue sticks move toward green spheres (NC2)
- Red sticks align with blue sticks (NC3)

## Key Technical Detail: High-Dimensional to 3D Projection

**Question**: How can features in $\mathbb{R}^p$ (where $p$ might be 512 or larger) be visualized in $\mathbb{R}^3$?

**Answer**: The Simplex ETF for $C$ classes lives in a $(C-1)$-dimensional subspace within $\mathbb{R}^p$.

### The Projection Method

1. **Compute SVD** of the class means matrix $M \in \mathbb{R}^{C \times p}$:
   ```
   M = U Σ V^T
   ```

2. **Extract top principal components**: The top $(C-1)$ right singular vectors in $V$ define the subspace where Neural Collapse occurs

3. **Project to subspace**: Use $V_{:k}$ (first $k$ columns, typically $k=3$ for visualization) to project:
   - Features: $h_{\text{proj}} = h \cdot V_{:k}$
   - Class means: $\mu_{\text{proj}} = \mu \cdot V_{:k}$
   - Classifiers: $W_{\text{proj}} = W \cdot V_{:k}$

This preserves the relative geometry (distances and angles) relevant to Neural Collapse while enabling 3D visualization.

## Installation

Ensure you have the required dependencies:

```bash
pip install jax jaxlib optax numpy matplotlib scipy
```

For GPU support (optional):
```bash
pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

### Integration with SpiralClassifier

```python
from src.models.spiral_classifier import SpiralClassifier
from analysis.neural_collapse_integration import train_with_neural_collapse
from utils.data_utils import generate_spiral_data

# Generate data
X_train, Y_train = generate_spiral_data(
    points_per_class=100,
    num_classes=3,
    noise_std=0.1
)

# Create classifier
classifier = SpiralClassifier(
    num_classes=3,
    nn_width=128,
    learning_rate=0.01
)

# Train with Neural Collapse capture
nc_analyzer = train_with_neural_collapse(
    classifier=classifier,
    X_train=X_train,
    Y_train=Y_train,
    num_steps=10000,
    snapshot_epochs=[0, 100, 500, 1000, 5000, 10000],
    output_dir='outputs/nc_analysis',
    visualize_snapshots=True
)
```

### Analyzing Saved Models

```python
from analysis.neural_collapse_integration import analyze_neural_collapse_from_checkpoint

# Load and analyze a checkpoint
snapshot = analyze_neural_collapse_from_checkpoint(
    checkpoint_path='path/to/checkpoint.pkl',
    X_train=X_train,
    Y_train=Y_train,
    classifier=classifier
)

# Compute metrics
metrics = nc_analyzer.compute_nc_metrics(snapshot)
print(f"NC1 (variance): {metrics['nc1_within_class_variance']:.6f}")
print(f"NC2 (ETF alignment): {metrics['nc2_etf_alignment']:.4f}")
print(f"NC3 (self-duality): {metrics['nc3_self_duality']:.4f}")

# Visualize
nc_analyzer.visualize_neural_collapse(
    snapshot=snapshot,
    save_path='outputs/nc_visualization.png'
)
```

### Advanced: Custom Analysis

```python
from analysis.neural_collapse import NeuralCollapseAnalyzer

# Create analyzer
nc_analyzer = NeuralCollapseAnalyzer(
    num_classes=3,
    feature_dim=128
)

# Manually create snapshots during training
for epoch in [0, 50, 100, 200]:
    features = extract_features(model, params, X_train)
    labels = jnp.argmax(Y_train, axis=1)
    class_means = compute_class_means(features, labels)
    classifiers = params[-1][0].T
    
    snapshot = NeuralCollapseSnapshot(
        epoch=epoch,
        features=features,
        labels=labels,
        class_means=class_means,
        classifiers=classifiers,
        num_classes=3,
        feature_dim=128
    )
    
    nc_analyzer.snapshots.append(snapshot)

# Compute metrics for all snapshots
for snapshot in nc_analyzer.snapshots:
    metrics = nc_analyzer.compute_nc_metrics(snapshot)
    print(f"Epoch {snapshot.epoch}: NC1={metrics['nc1_within_class_variance']:.6f}")

# Create visualizations
for snapshot in nc_analyzer.snapshots:
    nc_analyzer.visualize_neural_collapse(
        snapshot=snapshot,
        save_path=f'outputs/nc_epoch_{snapshot.epoch}.png'
    )
```

## Output Structure

When you run the analysis, you'll get:

```
outputs/neural_collapse/
├── training_curves.png              # Loss and accuracy over time
├── nc_metrics_evolution.png         # NC1, NC2, NC3 metrics over time
├── nc_viz_step_00000000.png          # Initial state visualization
├── nc_viz_step_00001000.png          # Early training
├── nc_viz_step_00005000.png          # Mid training
├── nc_viz_step_00010000.png          # Final state
├── nc_snapshots.pkl                 # Saved snapshots for later analysis
└── figure1_style_visualization.png  # Multi-panel comparison
```

## Interpreting the Results

### NC Metrics Over Time

- **NC1 (Within-Class Variance)**: Should **decrease** exponentially
  - Early training: High variance as features are still separating
  - Terminal phase: Near-zero variance as features collapse to means

- **NC2 (ETF Alignment)**: Should **increase** toward 1.0
  - Measures how well class means approximate the Simplex ETF
  - Value of 1.0 = perfect alignment

- **NC3 (Self-Duality)**: Should **increase** toward 1.0
  - Measures cosine similarity between classifiers and class means
  - Value of 1.0 = perfect alignment

### Visual Indicators of Neural Collapse

In the 3D visualizations:

**Early Training** (e.g., epoch 0-50):
- 🔵 Blue spheres are spread out (high within-class variance)
- 🔵 Blue sticks point in arbitrary directions
- 🔴 Red sticks are misaligned with blue sticks

**Terminal Phase** (e.g., epoch 500+):
- 🔵 Blue spheres cluster tightly at blue stick tips
- 🔵 Blue sticks align with 🟢 green spheres (ETF)
- 🔴 Red sticks overlap with blue sticks
- Perfect simplex geometry emerges

## Mathematical Background

### Simplex ETF Construction

For $C$ classes, the Simplex ETF in $\mathbb{R}^{C-1}$ is constructed as:

1. Start with identity matrix $I \in \mathbb{R}^{C \times C}$
2. Center: $M = I - \frac{1}{C} \mathbf{1}\mathbf{1}^T$
3. SVD: $M = U \Sigma V^T$
4. Take top $(C-1)$ components: $\text{ETF} = U_{:,1:C-1} \sqrt{\frac{C}{C-1}}$

Properties:
- Each row has unit norm
- Inner product between any two rows: $-\frac{1}{C-1}$
- Rows sum to zero (centered)

### Neural Collapse Metrics

**NC1**: Within-class variance
```python
NC1 = (1/C) Σ_c E_{x∈class_c}[||h(x) - μ_c||²]
```

**NC2**: Alignment with ETF
```python
NC2 = (1/C) Σ_c max_j <normalize(μ_c), ETF_j>
```

**NC3**: Self-duality
```python
NC3 = (1/C) Σ_c <normalize(w_c), normalize(μ_c)>
```

## Adapting to Your Dataset

### For Image Datasets (CIFAR-10, ImageNet)

```python
# Use with ResNet or VGG
from analysis.neural_collapse import NeuralCollapseAnalyzer

# Extract features from penultimate layer
features = model.extract_features(images)  # (N, feature_dim)
labels = labels  # (N,)

# Create analyzer for 10 classes, feature_dim from model
nc_analyzer = NeuralCollapseAnalyzer(
    num_classes=10,
    feature_dim=features.shape[1]
)

# ... rest of analysis
```

### For Different Architectures

The key is extracting features from the **penultimate layer** (last layer before classification):

```python
def extract_penultimate_features(model, params, X):
    """Extract features before final linear layer."""
    # Architecture-specific implementation
    # For ResNet: features after global average pooling
    # For VGG: features after last ReLU
    # For Transformer: CLS token or pooled output
    pass
```

## Troubleshooting

### Issue: Poor NC2 alignment

**Solution**: Train longer or reduce learning rate in terminal phase. NC2 convergence is slowest.

### Issue: Snapshots too large

**Solution**: Subsample features in snapshots:
```python
# Sample 1000 features instead of all
indices = np.random.choice(len(features), 1000, replace=False)
snapshot.features = features[indices]
snapshot.labels = labels[indices]
```

### Issue: Visualization looks cluttered

**Solution**: Reduce samples_per_class or select fewer classes:
```python
nc_analyzer.visualize_neural_collapse(
    snapshot=snapshot,
    selected_classes=[0, 1, 2],  # Only first 3 classes
    samples_per_class=10  # Fewer samples
)
```

## References

1. Papyan, V., Han, X. Y., & Donoho, D. L. (2020). "Prevalence of neural collapse during the terminal phase of deep learning training." *Proceedings of the National Academy of Sciences*, 117(40), 24652-24663.

2. Mixon, D. G., Parshall, H., & Pi, J. (2020). "Neural collapse with unconstrained features." *arXiv preprint arXiv:2011.11619*.

3. Fang, C., He, H., Long, Q., & Su, W. J. (2021). "Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training." *Proceedings of the National Academy of Sciences*, 118(43).

## Citation

If you use this code in your research, please cite:

```bibtex
@software{neural_collapse_viz,
  author = {Nicolas Ratier Werbin},
  title = {Neural Collapse Visualization for Dynamical SGD},
  year = {2024},
  url = {https://github.com/yourusername/dynamical-sgd}
}

@article{papyan2020prevalence,
  title={Prevalence of neural collapse during the terminal phase of deep learning training},
  author={Papyan, Vardan and Han, X Yu and Donoho, David L},
  journal={Proceedings of the National Academy of Sciences},
  volume={117},
  number={40},
  pages={24652--24663},
  year={2020}
}
```

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Support for more architectures (ResNet, Transformers)
- [ ] Interactive 3D visualizations (Plotly)
- [ ] Additional metrics (NC4, collapse rate)
- [ ] Comparison with imbalanced datasets
- [ ] Animation of temporal evolution

## License

MIT License - see LICENSE file for details.
