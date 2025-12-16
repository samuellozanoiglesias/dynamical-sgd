# Neural Collapse with Dynamic Class Focus

This integration adds Neural Collapse analysis to your existing dynamical SGD training. Now you can study how **periodic class emphasis (bumping)** affects the terminal phase geometry!

## What's Been Added

✅ Neural Collapse snapshot capture integrated into [run_experiment.py](run_experiment.py)  
✅ Works with your **existing bumping mechanism** (dynamic class focus)  
✅ Two config files for comparison studies

## Your Bumping Implementation

The bumping is in [run_experiment.py](run_experiment.py#L187-L197):

```python
if config.dynamics.enable_dynamics:
    # Compute which class to focus on
    class_focus = int((t // config.dynamics.period_length) % config.data.num_classes)
    
    # Compute class weights (the "bumping")
    current_weights = classifier.compute_class_weights(
        t % config.dynamics.period_length,
        class_focus,
        config.dynamics.w_max,
        config.dynamics.period_length
    )
    
    # Sample batch according to weights
    X_batch, Y_batch = classifier.sample_by_class(...)
```

Parameters:
- **`period_length`**: How long to focus on each class (T)
- **`w_max`**: Amplitude of the bumping (how much to emphasize the focused class)
- **`num_classes`**: Cycles through all classes

## Usage

### Run WITH Bumping (Dynamic Focus)
```bash
python run_experiment.py --config config/nc_dynamics_config.yaml
```

This will:
- Train with periodic class emphasis
- Capture NC snapshots every 2500 steps
- Save 3D visualizations showing NC evolution
- Output: `outputs/experiments/nc_with_dynamics/`

### Run WITHOUT Bumping (Standard Training)
```bash
python run_experiment.py --config config/nc_standard_config.yaml
```

This will:
- Train with balanced sampling (control)
- Capture NC snapshots at same intervals
- Same visualizations for comparison
- Output: `outputs/experiments/nc_without_dynamics/`

### Compare Results
```bash
# Compare NC metrics
ls outputs/experiments/nc_with_dynamics/nc_metrics_evolution.png
ls outputs/experiments/nc_without_dynamics/nc_metrics_evolution.png

# Compare final snapshots
ls outputs/experiments/nc_with_dynamics/nc_viz_step_*.png
ls outputs/experiments/nc_without_dynamics/nc_viz_step_*.png
```

## Configuration Parameters

### Enable Neural Collapse Tracking
```yaml
analysis:
  track_neural_collapse: true  # Enable NC analysis
  nc_snapshot_interval: 2500   # Capture every N steps
```

### Enable/Disable Bumping
```yaml
dynamics:
  enable_dynamics: true   # true = with bumping, false = without
  period_length: 5000     # Period for class cycling
  w_max: 70.0            # Amplitude of emphasis
```

### Visualization Options
```yaml
visualization:
  save_nc_visualizations: true  # Save 3D plots
  figure_dpi: 300              # High quality
```

## Output Files

After running, you'll get:

```
outputs/experiments/nc_with_dynamics/
├── training_curves.png          # Loss & accuracy (with period markers)
├── nc_metrics_evolution.png     # NC1, NC2, NC3 over time
├── nc_viz_step_00000000.png      # Initial state
├── nc_viz_step_00002500.png      # Early training
├── nc_viz_step_00025000.png      # Mid training
├── nc_viz_step_00049999.png      # Terminal phase
├── nc_snapshots.pkl             # All snapshots (for later analysis)
├── class_focus_dynamics.png     # Bumping visualization
└── decision_boundary.png        # Final decision boundary
```

## Research Questions You Can Answer

1. **Does bumping affect NC convergence rate?**
   - Compare NC2 curves between dynamics vs. standard

2. **Does bumping change terminal phase geometry?**
   - Compare final NC1, NC2, NC3 values

3. **How does bumping amplitude affect NC?**
   - Run with different `w_max` values

4. **Does bumping period matter?**
   - Run with different `period_length` values

5. **Does NC happen during bumping transitions?**
   - Look at snapshots at period boundaries

## Advanced Usage

### Custom Snapshot Times
```bash
python run_experiment.py \
  --config config/nc_dynamics_config.yaml \
  --override analysis.nc_snapshot_interval=1000
```

### Vary Bumping Parameters
```bash
# Strong bumping
python run_experiment.py \
  --config config/nc_dynamics_config.yaml \
  --override dynamics.w_max=100.0

# Fast cycling
python run_experiment.py \
  --config config/nc_dynamics_config.yaml \
  --override dynamics.period_length=2000
```

### Longer Training (More Complete NC)
```bash
python run_experiment.py \
  --config config/nc_dynamics_config.yaml \
  --override training.total_steps=100000
```

## Analyzing Saved Snapshots

Load and re-analyze later:

```python
from analysis.neural_collapse import NeuralCollapseAnalyzer
import pickle

# Load snapshots
nc_analyzer = NeuralCollapseAnalyzer(num_classes=3, feature_dim=128)
nc_analyzer.load_snapshots('outputs/.../nc_snapshots.pkl')

# Recompute metrics
for snapshot in nc_analyzer.snapshots:
    metrics = nc_analyzer.compute_nc_metrics(snapshot)
    print(f"Step {snapshot.epoch}: NC1={metrics['nc1_within_class_variance']:.6f}")

# Create new visualizations
nc_analyzer.visualize_neural_collapse(
    snapshot=nc_analyzer.snapshots[-1],
    save_path='custom_viz.png'
)
```

## Expected Results

### With Bumping
- NC might converge at different rates
- Periodic fluctuations in NC1 (variance) during bumping
- Possibly slower/faster convergence depending on period

### Without Bumping
- Smooth NC convergence
- Monotonic decrease in NC1
- Standard terminal phase behavior

## Tips

1. **Start small**: Test with `--override training.total_steps=5000` first
2. **Watch GPU memory**: Snapshots store all features, might need to reduce interval
3. **Compare fairly**: Use same seeds and total steps for both conditions
4. **Look at periods**: Interesting things happen at period boundaries

## What's Happening Under the Hood

1. **During training**: At specified steps, the code:
   - Extracts features from penultimate layer
   - Computes class means (centroids)
   - Extracts classifier weights
   - Creates NeuralCollapseSnapshot
   - Computes NC1, NC2, NC3 metrics

2. **After training**: 
   - Saves all snapshots to disk
   - Plots metric evolution
   - Creates 3D visualizations
   - Saves everything to output directory

3. **The bumping continues**: NC analysis doesn't interfere with training dynamics

## Troubleshooting

**Issue**: Out of memory
```bash
# Reduce snapshot frequency
--override analysis.nc_snapshot_interval=5000
```

**Issue**: Too slow
```bash
# Disable visualizations during training
--override visualization.save_nc_visualizations=false
# Visualize later from snapshots
```

**Issue**: Want more snapshots at end
```python
# Manually specify: capture every 1000 steps in last 10k
snapshot_epochs = list(range(0, 40000, 5000)) + list(range(40000, 50000, 1000))
```

## Next Steps

Try these experiments:

1. **Baseline**: Run without bumping
2. **Standard bumping**: Run with default parameters  
3. **Strong bumping**: Increase `w_max` to 100
4. **Fast cycling**: Decrease `period_length` to 2000
5. **Compare**: Look at NC metrics evolution plots

Then write a paper! 📝

---

**This is your complete Neural Collapse + Bumping analysis system!** 🎉
