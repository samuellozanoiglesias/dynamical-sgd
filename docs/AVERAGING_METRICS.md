# Averaging Neural Collapse Metrics Across Seeds

## Overview

The `average_nc_metrics.py` script averages metrics across multiple experiment runs (seeds) for a given neural collapse configuration. This is essential for obtaining robust statistics and reducing the impact of random initialization.

## What Gets Averaged

The script computes mean and standard deviation across all seeds for:

### 1. Training Curves
- **Training accuracy** (mean ± std)
- **Test accuracy** (mean ± std)
- **Training loss** (mean ± std)
- **Test loss** (mean ± std)

### 2. Neural Collapse Metrics
- **NC1 (Variability Collapse)**: Within-class variance of features
- **NC2 (Convergence to ETF)**: Alignment of class means with Simplex ETF
- **NC3 (Self-Duality)**: Alignment between classifiers and class means

### 3. Angles
- **Mean angles**: Average pairwise angles between class mean vectors
- **Classifier angles**: Average pairwise angles between classifier vectors
- **ETF angles**: Theoretical Simplex ETF angles (for reference)

## Usage

### Basic Usage

```bash
# Average experiments for a specific configuration
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics
```

### Common Examples

```bash
# Average experiments WITH dynamic class focus (bumping)
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics

# Average experiments WITHOUT dynamic class focus (baseline)
python average_nc_metrics.py \
    --config nc_config_baseline \
    --experiment_name nc_without_dynamics

# Average wide bump configuration
python average_nc_metrics.py \
    --config nc_config_wide_bump \
    --experiment_name nc_with_dynamics

# Filter by timestamp (e.g., only experiments from Dec 14, 2025)
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics \
    --timestamp 2025_12_14
```

### Custom Directories

```bash
# Specify custom base directory
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics \
    --base_dir /path/to/your/results

# Specify custom output directory
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics \
    --output_dir /path/to/output
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | Yes | - | Configuration name (e.g., `nc_config_narrow_bump`) |
| `--experiment_name` | Yes | - | Experiment name: `nc_with_dynamics` or `nc_without_dynamics` |
| `--base_dir` | No | `/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd` | Base directory containing experiments |
| `--output_dir` | No | `{base_dir}/{experiment_name}/{config}/averaged` | Output directory for results |
| `--timestamp` | No | None | Filter experiments by timestamp (e.g., `2025_12_14`) |

## Output Files

The script generates the following files in the output directory:

### Visualization Files (PNG)
1. **`averaged_training_curves.png`**: Training and test accuracy/loss with confidence bands
2. **`averaged_nc_metrics.png`**: NC1, NC2, NC3 evolution with confidence bands
3. **`averaged_angles.png`**: Pairwise angles evolution with confidence bands

### Data Files
4. **`averaged_training_curves.csv`**: Training curves data (mean ± std)
5. **`averaged_nc_metrics.csv`**: NC metrics data (mean ± std)
6. **`averaged_angles.csv`**: Angles data (mean ± std)
7. **`averaged_metrics.pkl`**: Complete averaged metrics object (for further analysis)
8. **`averaged_summary.json`**: Summary statistics (final values, metadata)

### CSV Format Examples

**averaged_training_curves.csv**:
```csv
Step,Train_Acc_Mean,Train_Acc_Std,Test_Acc_Mean,Test_Acc_Std,Train_Loss_Mean,Train_Loss_Std,Test_Loss_Mean,Test_Loss_Std
0,0.3333,0.0012,0.3345,0.0015,1.0986,0.0023,1.0954,0.0019
100,0.4521,0.0034,0.4489,0.0041,0.9876,0.0045,0.9923,0.0038
...
```

**averaged_nc_metrics.csv**:
```csv
Step,NC1_Mean,NC1_Std,NC2_Mean,NC2_Std,NC3_Mean,NC3_Std
0,45.6789,2.3456,0.2345,0.0123,0.3456,0.0234
2500,12.3456,0.8765,0.5678,0.0234,0.6789,0.0345
...
```

**averaged_angles.csv**:
```csv
Step,Mean_Angles_Mean,Mean_Angles_Std,Classifier_Angles_Mean,Classifier_Angles_Std,ETF_Angles_Mean,ETF_Angles_Std
0,89.5,3.2,91.2,2.8,120.0,0.0
2500,125.3,1.5,122.8,1.8,120.0,0.0
...
```

## Expected Directory Structure

The script expects experiments to be organized as:

```
base_dir/
├── nc_with_dynamics/
│   ├── nc_config_narrow_bump/
│   │   ├── experiment_2025_12_14-10_30_00/  (seed 0)
│   │   │   ├── results.csv
│   │   │   ├── nc_snapshots.pkl
│   │   │   ├── config.yaml
│   │   │   └── ...
│   │   ├── experiment_2025_12_14-11_15_00/  (seed 10)
│   │   │   └── ...
│   │   └── averaged/  (output directory)
│   │       ├── averaged_training_curves.png
│   │       ├── averaged_nc_metrics.png
│   │       └── ...
│   ├── nc_config_wide_bump/
│   │   └── ...
│   └── ...
└── nc_without_dynamics/
    └── ...
```

## How It Works

### 1. Data Collection
- Scans for all experiment directories matching the config and experiment name
- Loads `results.csv` for training curves
- Loads `nc_snapshots.pkl` for Neural Collapse analysis
- Extracts seed information from `config.yaml`

### 2. Interpolation
- Interpolates all metrics to common time steps
- Ensures alignment across experiments with different sampling rates
- Handles varying experiment lengths gracefully

### 3. Statistics
- Computes mean and standard deviation for all metrics
- Handles missing data robustly
- Provides detailed error messages for failed experiments

### 4. Visualization
- Creates publication-quality plots with confidence bands
- Uses consistent styling for easy comparison
- Includes metadata (number of seeds, configuration info)

## Integration with SBATCH Scripts

The averaging script works seamlessly with the SBATCH experiment runner:

```bash
# 1. Run experiments with multiple seeds (in brigit/run_neural_collapse.sbatch)
SEEDS=(0 10 20 30 40 50 60 70 80 90 100)

# 2. After experiments complete, average the results
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics
```

## Tips and Best Practices

### 1. Consistent Seeds
Use consistent seeds across experiments for reproducibility:
```bash
SEEDS=(0 10 20 30 40 50 60 70 80 90 100)
```

### 2. Timestamp Filtering
When re-running experiments, use timestamp filtering to separate old and new runs:
```bash
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics \
    --timestamp 2025_12_14
```

### 3. Compare WITH vs WITHOUT Dynamics
Average both experiment types and compare the results:
```bash
# Average WITH dynamics
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics

# Average WITHOUT dynamics
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_without_dynamics

# Compare results in:
# nc_with_dynamics/nc_config_narrow_bump/averaged/
# nc_without_dynamics/nc_config_narrow_bump/averaged/
```

### 4. Multiple Configurations
Average multiple configurations to compare different hyperparameters:
```bash
for config in nc_config_baseline nc_config_narrow_bump nc_config_wide_bump; do
    python average_nc_metrics.py \
        --config $config \
        --experiment_name nc_with_dynamics
done
```

## Troubleshooting

### Issue: "No experiments found"
**Solution**: Check that the base_dir path is correct and experiments exist:
```bash
ls -la /mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/nc_with_dynamics/nc_config_narrow_bump/
```

### Issue: "Failed to process experiment"
**Solution**: Check that all required files exist in experiment directories:
- `results.csv` (training curves)
- `nc_snapshots.pkl` (NC analysis)
- `config.yaml` (configuration)

### Issue: Inconsistent number of snapshots
**Solution**: Ensure all experiments use the same `nc_snapshot_interval` in config:
```yaml
analysis:
  nc_snapshot_interval: 2500  # Must be consistent across experiments
```

## Example Workflow

Complete workflow for analyzing Neural Collapse with bumping:

```bash
# 1. Run experiments with multiple seeds (via SBATCH)
sbatch brigit/run_neural_collapse.sbatch

# 2. Wait for completion, then average results
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_with_dynamics

# 3. Check output
ls averaged/nc_config_narrow_bump/nc_with_dynamics/
ls /mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/nc_with_dynamics/nc_config_narrow_bump/averaged/
# averaged_training_curves.png
# averaged_nc_metrics.png
# averaged_angles.png
# averaged_summary.json
# ...

# 4. View summary
cat /mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/nc_with_dynamics/nc_config_narrow_bump/averaged/averaged_summary.json

# 5. Compare with baseline (no dynamics)
python average_nc_metrics.py \
    --config nc_config_narrow_bump \
    --experiment_name nc_without_dynamics
```

## Further Analysis

The `averaged_metrics.pkl` file can be loaded for custom analysis:

```python
import pickle
from pathlib import Path

# Load averaged metrics
path = '/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/nc_with_dynamics/nc_config_narrow_bump/averaged/averaged_metrics.pkl'
with open(path, 'rb') as f:
    metrics = pickle.load(f)

# Access data
print(f"Number of seeds: {metrics.num_seeds}")
print(f"Final test accuracy: {metrics.test_acc_mean[-1]:.4f} ± {metrics.test_acc_std[-1]:.4f}")

# Create custom plots
import matplotlib.pyplot as plt
plt.plot(metrics.train_steps, metrics.train_acc_mean)
plt.show()
```

## See Also

- [NEURAL_COLLAPSE.md](NEURAL_COLLAPSE.md) - Neural Collapse theory and implementation
- [NEURAL_COLLAPSE_VISUALIZATION.md](NEURAL_COLLAPSE_VISUALIZATION.md) - Visualization details
- [NEURAL_COLLAPSE_WITH_BUMPING.md](NEURAL_COLLAPSE_WITH_BUMPING.md) - Dynamic class focus experiments
- [run_neural_collapse.sbatch](../brigit/run_neural_collapse.sbatch) - SBATCH script for running experiments
