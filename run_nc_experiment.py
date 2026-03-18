#!/usr/bin/env python3
"""
Run Neural Collapse experiments with three different bumping configurations.

This script can run experiments with:
  - always: Bumping continues throughout training (including after 100% accuracy)
  - tpt_only: Bumping only during Terminal Phase Training (after reaching 100% accuracy)
  - pre_tpt: Bumping only before Terminal Phase Training (stops at 100% accuracy)  
  - never: No bumping at all (standard training)
  - all: Run all four experiments for comparison

Usage:
    python run_nc_experiment.py --cluster brigit --mode all
    python run_nc_experiment.py --cluster cuenca --mode always
    python run_nc_experiment.py --cluster brigit --mode tpt_only
    python run_nc_experiment.py --cluster brigit --mode pre_tpt
    python run_nc_experiment.py --cluster brigit --mode never
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import yaml


def _is_mnist_config(config_path: Path) -> bool:
    """Return True when configuration explicitly targets MNIST dataset."""
    if not config_path.exists():
        return False
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f) or {}
    dataset_name = str(config_data.get('data', {}).get('dataset_name', 'spiral')).lower()
    return dataset_name == 'mnist'


def _load_config_dict(config_path: Path):
    if not config_path.exists():
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def main():
    parser = argparse.ArgumentParser(
        description='Run Neural Collapse experiments with configurable cluster settings'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        required=True,
        choices=['brigit', 'cuenca'],
        help='Cluster name (brigit or cuenca)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['always', 'tpt_only', 'pre_tpt', 'never', 'all'],
        help='Which experiment to run: always (bumps throughout), tpt_only (bumps only after 100%%), pre_tpt (bumps stop at 100%%), never (no bumps), or all (run all four)'
    )

    parser.add_argument(
        '--config_file',
        type=str,
        default='nc_config.yaml',
        help='Path to the base configuration file'
    )
    
    parser.add_argument(
        '--output_base',
        type=str,
        default='data/samuel_lozano/dynamical-sgd',
        help='Base output directory'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup base directory based on cluster
    if args.cluster == 'brigit':
        base_dir = Path('/mnt/lustre/home/samuloza')
    elif args.cluster == 'cuenca':
        base_dir = Path('/')
    else:
        base_dir = Path('/')
    
    output_base = base_dir / args.output_base
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Extract config name (without .yaml extension)
    config_name = Path(args.config_file).stem
    args.config_file = f'config/{args.config_file}'
    config_path = Path(args.config_file)

    # Generate single timestamp for paired experiments
    experiment_timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    print("=" * 80)
    print("NEURAL COLLAPSE EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Cluster: {args.cluster}")
    print(f"Mode: {args.mode}")
    print(f"Config file: {args.config_file}")
    print(f"Config name: {config_name}")
    print(f"Output base: {output_base}")
    print(f"Paired timestamp: {experiment_timestamp}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print("=" * 80)
    print()

    if _is_mnist_config(config_path):
        print("MNIST config detected. Switching to notebook-equivalent pipeline.")
        print("Running MNIST notebook-equivalent pipeline with bump mode support.")
        print()

        config_dict = _load_config_dict(config_path)
        dynamics_cfg = config_dict.get('dynamics', {})
        training_cfg = config_dict.get('training', {})
        model_cfg = config_dict.get('model', {})

        period_length = int(dynamics_cfg.get('period_length', 2000))
        w_max = float(dynamics_cfg.get('w_max', 50.0))
        tpt_threshold = float(dynamics_cfg.get('tpt_accuracy_threshold', 1.0))

        model_architecture = str(model_cfg.get('architecture', 'resnet18')).lower()
        if model_architecture not in {'resnet18', 'mlp'}:
            raise ValueError(
                f"Unsupported model.architecture='{model_architecture}' in {config_path}. "
                "Expected one of: resnet18, mlp"
            )

        mlp_hidden_dim = int(model_cfg.get('nn_width', 512))
        mlp_num_hidden_layers = int(model_cfg.get('num_hidden_layers', 2))
        mlp_use_bias = bool(model_cfg.get('use_bias', True))
        
        # Extract training configuration for step-capped epochs.
        if 'training_steps' not in training_cfg or 'steps_per_epoch' not in training_cfg:
            raise ValueError(
                f"Config {config_path} missing required training parameters for MNIST pipeline: "
                f"training_steps={training_cfg.get('training_steps')}, "
                f"steps_per_epoch={training_cfg.get('steps_per_epoch')}"
            )

        total_training_steps = int(training_cfg['training_steps'])
        steps_per_epoch = int(training_cfg['steps_per_epoch'])
        if total_training_steps <= 0 or steps_per_epoch <= 0:
            raise ValueError(
                f"Invalid MNIST training parameters: training_steps={total_training_steps}, steps_per_epoch={steps_per_epoch}. "
                "Both must be > 0."
            )

        epochs = total_training_steps // steps_per_epoch
        if total_training_steps % steps_per_epoch != 0:
            epochs += 1
        
        print(f"Training configuration from {config_path}:")
        print(f"  training_steps: {total_training_steps}")
        print(f"  epochs: {epochs}")
        print(f"  steps_per_epoch: {steps_per_epoch}")
        print(f"  model.architecture: {model_architecture}")
        if model_architecture == 'mlp':
            print(f"  model.num_hidden_layers: {mlp_num_hidden_layers}")
            print(f"  model.nn_width: {mlp_hidden_dim}")
            print(f"  model.use_bias: {mlp_use_bias}")
        print()

        if args.mode == 'all':
            mnist_modes = [
                ('never', False, False, 'nc_never'),
                ('pre_tpt', True, False, 'nc_pre_tpt'),
                ('tpt_only', False, True, 'nc_tpt_only'),
                ('always', True, True, 'nc_always'),
            ]
        elif args.mode == 'never':
            mnist_modes = [('never', False, False, 'nc_never')]
        elif args.mode == 'pre_tpt':
            mnist_modes = [('pre_tpt', True, False, 'nc_pre_tpt')]
        elif args.mode == 'tpt_only':
            mnist_modes = [('tpt_only', False, True, 'nc_tpt_only')]
        else:
            mnist_modes = [('always', True, True, 'nc_always')]

        for idx, (mode_name, bumps_before, bumps_after, exp_name) in enumerate(mnist_modes, 1):
            output_dir = (
                output_base
                / exp_name
                / config_name
                / f'experiment_{experiment_timestamp}'
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env['NC_RESULTS_DIR'] = str(output_dir)
            env['NC_EPOCHS'] = str(epochs)
            env['NC_STEPS_PER_EPOCH'] = str(steps_per_epoch)
            env['NC_TOTAL_TRAINING_STEPS'] = str(total_training_steps)
            env['NC_BUMPS_BEFORE_TPT'] = 'true' if bumps_before else 'false'
            env['NC_BUMPS_AT_TPT'] = 'true' if bumps_after else 'false'
            env['NC_PERIOD_LENGTH'] = str(period_length)
            env['NC_W_MAX'] = str(w_max)
            env['NC_TPT_ACCURACY_THRESHOLD'] = str(tpt_threshold)
            env['NC_MODEL_ARCHITECTURE'] = model_architecture
            env['NC_MLP_HIDDEN_DIM'] = str(mlp_hidden_dim)
            env['NC_MLP_NUM_HIDDEN_LAYERS'] = str(mlp_num_hidden_layers)
            env['NC_MLP_USE_BIAS'] = 'true' if mlp_use_bias else 'false'

            # Save the actual config being used to config.txt
            config_txt_path = output_dir / "config.txt"
            with open(config_txt_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Neural Collapse MNIST Experiment Configuration\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Experiment Name: {mode_name.upper()}\n")
                f.write(f"Config File: {config_path}\n")
                f.write(f"Timestamp: {experiment_timestamp}\n")
                f.write(f"Bumping Mode: {mode_name}\n\n")
                f.write("=" * 80 + "\n")
                f.write("Training Configuration\n")
                f.write("=" * 80 + "\n")
                f.write(f"  Training Steps: {total_training_steps}\n")
                f.write(f"  Epochs: {epochs}\n")
                f.write(f"  Steps Per Epoch: {steps_per_epoch}\n")
                f.write(f"  Batch Size: 128\n\n")
                f.write("=" * 80 + "\n")
                f.write("Model Configuration\n")
                f.write("=" * 80 + "\n")
                f.write(f"  Architecture: {model_architecture}\n")
                if model_architecture == 'mlp':
                    f.write(f"  Hidden Layers: {mlp_num_hidden_layers}\n")
                    f.write(f"  Hidden Width: {mlp_hidden_dim}\n")
                    f.write(f"  Use Bias: {mlp_use_bias}\n")
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write("Bumping Configuration\n")
                f.write("=" * 80 + "\n")
                f.write(f"  Mode: {mode_name}\n")
                f.write(f"  Bumps Before TPT: {bumps_before}\n")
                f.write(f"  Bumps At TPT: {bumps_after}\n")
                f.write(f"  Period Length (steps): {period_length}\n")
                f.write(f"  W Max (peak weight): {w_max}\n")
                f.write(f"  TPT Accuracy Threshold: {tpt_threshold}\n\n")
                f.write("=" * 80 + "\n")
                f.write("Full Configuration (YAML)\n")
                f.write("=" * 80 + "\n\n")
                # Append the full YAML config
                config_yaml_content = config_path.read_text(encoding="utf-8")
                f.write(config_yaml_content)

            cmd = ['python', 'neuralcollapse_notebook.py']
            print(f"[{idx}/{len(mnist_modes)}] Mode={mode_name} -> {' '.join(cmd)}")
            print(f"    NC_RESULTS_DIR={output_dir}")
            print(f"    Config saved to: {config_txt_path}")

            try:
                subprocess.run(cmd, check=True, env=env)
                print(f"✓ MNIST notebook-equivalent experiment completed for mode={mode_name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ MNIST notebook-equivalent experiment failed for mode={mode_name} (error {e.returncode})")
                sys.exit(e.returncode)

        print("All MNIST notebook-equivalent experiments completed.")
        return
    
    experiments = []

    # Use single config file, just override the two bump parameters
    if args.mode in ['never', 'all']:
        # NEVER: No bumps at all
        overrides = [
            f'output.output_dir={output_base}',
            'dynamics.bumps_before_TPT=false',
            'dynamics.bumps_at_TPT=false',
            'output.experiment_name=nc_never',
            f'output.config_name={config_name}',
            f'output.experiment_timestamp={experiment_timestamp}'
        ]
        if args.seed is not None:
            overrides.append(f'training.random_seed={args.seed}')
            overrides.append(f'data.random_seed={args.seed}')
        
        experiments.append({
            'name': 'NEVER: No Bumps (Standard Training)',
            'config': args.config_file,
            'overrides': overrides
        })
    
    if args.mode in ['pre_tpt', 'all']:
        # PRE_TPT: Bumps only before TPT (stops when reaching 100% accuracy)
        overrides = [
            f'output.output_dir={output_base}',
            'dynamics.bumps_before_TPT=true',
            'dynamics.bumps_at_TPT=false',
            'output.experiment_name=nc_pre_tpt',
            f'output.config_name={config_name}',
            f'output.experiment_timestamp={experiment_timestamp}'
        ]
        if args.seed is not None:
            overrides.append(f'training.random_seed={args.seed}')
            overrides.append(f'data.random_seed={args.seed}')
        
        experiments.append({
            'name': 'PRE_TPT: Bumps Only Before Terminal Phase Training (stops at 100% accuracy)',
            'config': args.config_file,
            'overrides': overrides
        })
    
    if args.mode in ['tpt_only', 'all']:
        # TPT_ONLY: Bumps only during Terminal Phase Training (after reaching 100% accuracy)
        overrides = [
            f'output.output_dir={output_base}',
            'dynamics.bumps_before_TPT=false',
            'dynamics.bumps_at_TPT=true',
            'output.experiment_name=nc_tpt_only',
            f'output.config_name={config_name}',
            f'output.experiment_timestamp={experiment_timestamp}'
        ]
        if args.seed is not None:
            overrides.append(f'training.random_seed={args.seed}')
            overrides.append(f'data.random_seed={args.seed}')
        
        experiments.append({
            'name': 'TPT_ONLY: Bumps Only During Terminal Phase Training (after 100% accuracy)',
            'config': args.config_file,
            'overrides': overrides
        })
    
    if args.mode in ['always', 'all']:
        # ALWAYS: Bumps throughout (before AND during/after TPT)
        overrides = [
            f'output.output_dir={output_base}',
            'dynamics.bumps_before_TPT=true',
            'dynamics.bumps_at_TPT=true',
            'output.experiment_name=nc_always',
            f'output.config_name={config_name}',
            f'output.experiment_timestamp={experiment_timestamp}'
        ]
        if args.seed is not None:
            overrides.append(f'training.random_seed={args.seed}')
            overrides.append(f'data.random_seed={args.seed}')
        
        experiments.append({
            'name': 'ALWAYS: Bumps Throughout Training (before AND during/after TPT)',
            'config': args.config_file,
            'overrides': overrides
        })
    
    # Run experiments
    for i, exp in enumerate(experiments, 1):
        print()
        print("=" * 80)
        print(f"Experiment {i}/{len(experiments)}: {exp['name']}")
        print("=" * 80)
        print()
        
        # Build command
        cmd = ['python', 'run_experiment.py', '--config', exp['config']]
        
        for override in exp['overrides']:
            cmd.extend(['--override', override])
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run experiment
        try:
            result = subprocess.run(cmd, check=True)
            print()
            print(f"✓ Experiment {i} completed successfully")
        except subprocess.CalledProcessError as e:
            print()
            print(f"✗ Experiment {i} failed with error code {e.returncode}")
            sys.exit(e.returncode)
    
    # Print summary
    print()
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print()
    print("Results saved to:")
    
    if args.mode in ['always', 'all']:
        print(f"  - {output_base}/nc_always/{config_name}/experiment_{experiment_timestamp}/")
    if args.mode in ['tpt_only', 'all']:
        print(f"  - {output_base}/nc_tpt_only/{config_name}/experiment_{experiment_timestamp}/")
    if args.mode in ['pre_tpt', 'all']:
        print(f"  - {output_base}/nc_pre_tpt/{config_name}/experiment_{experiment_timestamp}/")
    if args.mode in ['never', 'all']:
        print(f"  - {output_base}/nc_never/{config_name}/experiment_{experiment_timestamp}/")
    
    print()
    print("Compare the following:")
    print("  1. config.yaml: Full experiment configuration")
    print("  2. results.txt: Training metrics and final accuracies")
    print("  3. nc_metrics_evolution.png: NC1/NC2/NC3 over time")
    print("  4. complete_angle_convergence_evolution.png: ALL component angles convergence (means, classifiers, biases, alignment in R^p)")
    print("  5. nc_evolution_3d.mp4: 3D animation video of neural collapse evolution (0.5s per frame)")
    print("  6. nc_evolution_2d.mp4: 2D animation video of neural collapse evolution (0.5s per frame, best for angles)")
    print("  7. 3d-snapshots/nc_viz_step_*.png: Individual 3D visualizations (~200 files)")
    print("  8. 2d-snapshots/nc_viz_step_*.png: Individual 2D visualizations (~200 files)")
    print("  9. training_curves.png: Loss and accuracy progression")
    print(" 10. nc_snapshots.pkl: Raw snapshots for analysis")
    print()
    print("See docs/NEURAL_COLLAPSE_VISUALIZATION.md for details!")
    print("=" * 80)

if __name__ == '__main__':
    main()
