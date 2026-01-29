# Neural Collapse Experiments - SLURM Batch Scripts

This folder contains SLURM batch scripts for running neural collapse experiments on the Brigit cluster.

## New Folder Structure

Experiments are now organized with the following structure:

```
.../dynamical-sgd/
├── nc_with_dynamics/
│   ├── nc_config_baseline/
│   │   ├── experiment_2024_12_13-14_30_00/  (seed=42)
│   │   ├── experiment_2024_12_13-15_45_00/  (seed=123)
│   │   └── experiment_2024_12_13-17_00_00/  (seed=456)
│   ├── nc_config_high_bump/
│   │   └── experiment_2024_12_13-14_30_00/
│   └── nc_config_narrow_bump/
│       └── experiment_2024_12_13-14_30_00/
└── nc_without_dynamics/
    ├── nc_config_baseline/
    │   └── experiment_2024_12_13-14_30_00/
    └── ...
```

Benefits:
- Easy to compare different configurations (config folders)
- Easy to collect statistics (multiple seeds in same config folder)
- Clear separation between dynamics modes

## Quick Start

### Single Experiment, Single Seed

Edit the configuration section in your sbatch file:

```bash
CONFIG_FILE="nc_config_high_bump.yaml"
MODE="with_dynamics"
CLUSTER="brigit"
SEEDS=(42)  # Single seed
```

Submit:
```bash
sbatch run_neural_collapse.sbatch
```

### Multiple Seeds for Statistics

To run the same experiment with multiple seeds (for statistical analysis):

```bash
CONFIG_FILE="nc_config_baseline.yaml"
MODE="both"  # Runs both with and without dynamics
CLUSTER="brigit"
SEEDS=(42 123 456 789 1011)  # 5 different seeds
```

This will run 5 experiments sequentially, each with a different random seed. Results will be saved in separate `experiment_` folders within the same config folder.

### Run Both Modes (with and without dynamics)

```bash
MODE="both"  # Compares dynamics vs standard training
```

This creates experiments in both `nc_with_dynamics/` and `nc_without_dynamics/` folders, making comparison easy.

## Available Scripts

### `run_neural_collapse.sbatch`
Current active script for `nc_config_high_bump`. Edit this for your experiments.

### `run_neural_collapse_template.sbatch`
Template with helpful comments. Copy and customize for new experiments:

```bash
cp run_neural_collapse_template.sbatch run_neural_collapse_myconfig.sbatch
# Edit the new file with your settings
sbatch run_neural_collapse_myconfig.sbatch
```

## Configuration Options

### CONFIG_FILE
Name of the YAML configuration file in the `config/` directory:
- `nc_config_baseline.yaml` - Standard baseline
- `nc_config_high_bump.yaml` - High amplitude bumping
- `nc_config_narrow_bump.yaml` - Narrow bump window
- `nc_config_wide_bump.yaml` - Wide bump window
- `nc_config_0.1.yaml` - Custom configuration

### MODE
Which experiment to run:
- `with_dynamics` - Only with dynamic class focus (bumping)
- `without_dynamics` - Only standard training
- `both` - Run both for comparison (RECOMMENDED)

### SEEDS
Array of random seeds for reproducibility:
- Single seed: `SEEDS=(42)`
- Multiple seeds: `SEEDS=(42 123 456 789 1011)`
- The script will run experiments sequentially for each seed

## Output Files

Each experiment folder contains:
1. `config.yaml` - Full experiment configuration
2. `results.txt` - Training metrics and final accuracies
3. `nc_metrics_evolution.png` - NC1/NC2/NC3 over time
4. `nc_evolution_3d.mp4` - 3D animation video (0.5s per frame)
5. `nc_evolution_2d.mp4` - 2D animation video (0.5s per frame, best for angles)
6. `3d-snapshots/` - Individual 3D visualizations (~200 PNG files)
7. `2d-snapshots/` - Individual 2D visualizations (~200 PNG files)
8. `training_curves.png` - Loss and accuracy progression
9. `nc_snapshots.pkl` - Raw snapshots for analysis

## Monitoring Your Jobs

Check job status:
```bash
squeue -u $USER
```

Check output (while running):
```bash
tail -f results-GPU-neural_collapse_high_bump-JOBID.out
```

Check errors:
```bash
tail -f error-GPU-neural_collapse_high_bump-JOBID.err
```

## Collecting Statistics

After running multiple seeds, you can analyze statistics across runs:

```bash
# All experiments with same config are in one folder
ls /mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/nc_with_dynamics/nc_config_baseline/

# Each experiment_* folder represents one seed
# You can write scripts to aggregate metrics across seeds
```

## Tips

1. **Use descriptive job names**: Edit `#SBATCH --job-name=` to identify your experiment
2. **Test with single seed first**: Use `SEEDS=(42)` to verify everything works
3. **Run statistics with 5+ seeds**: Use `SEEDS=(42 123 456 789 1011)` or more
4. **Compare modes**: Use `MODE="both"` to see impact of dynamics vs standard training
5. **Monitor disk space**: Videos and snapshots take significant space (~500MB per experiment)

## Example Workflows

### Baseline Comparison (with vs without dynamics)
```bash
CONFIG_FILE="nc_config_baseline.yaml"
MODE="both"
SEEDS=(42 123 456)
```
Results: 6 experiments (2 modes × 3 seeds)

### Testing New Configuration
```bash
CONFIG_FILE="nc_config_my_new_test.yaml"
MODE="with_dynamics"
SEEDS=(42)
```
Results: 1 experiment (quick test)

### Full Statistical Study
```bash
CONFIG_FILE="nc_config_high_bump.yaml"
MODE="both"
SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)
```
Results: 20 experiments (2 modes × 10 seeds for robust statistics)
