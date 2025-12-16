#!/usr/bin/env python3
"""
Run Neural Collapse experiments with or without dynamic class focus (bumping).

This script can run both experiments for comparison or just one.

Usage:
    python run_nc_experiment.py --cluster brigit
    python run_nc_experiment.py --cluster cuenca --mode with_dynamics
    python run_nc_experiment.py --cluster brigit --mode without_dynamics
    python run_nc_experiment.py --cluster brigit --mode both
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

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
        default='both',
        choices=['with_dynamics', 'without_dynamics', 'both'],
        help='Which experiment to run: with_dynamics (bumping), without_dynamics (standard), or both'
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
    
    experiments = []
    
    # Use single config file, just override enable_dynamics
    if args.mode in ['with_dynamics', 'both']:
        overrides = [
            f'output.output_dir={output_base}',
            'dynamics.enable_dynamics=true',
            'output.experiment_name=nc_with_dynamics',
            f'output.config_name={config_name}',
            f'output.experiment_timestamp={experiment_timestamp}'
        ]
        if args.seed is not None:
            overrides.append(f'training.random_seed={args.seed}')
        
        experiments.append({
            'name': 'WITH Dynamic Class Focus (Bumping)',
            'config': args.config_file,
            'overrides': overrides
        })
    
    if args.mode in ['without_dynamics', 'both']:
        overrides = [
            f'output.output_dir={output_base}',
            'dynamics.enable_dynamics=false',
            'output.experiment_name=nc_without_dynamics',
            f'output.config_name={config_name}',
            f'output.experiment_timestamp={experiment_timestamp}'
        ]
        if args.seed is not None:
            overrides.append(f'training.random_seed={args.seed}')
        
        experiments.append({
            'name': 'WITHOUT Dynamic Class Focus (Standard Training)',
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
    
    if args.mode in ['with_dynamics', 'both']:
        print(f"  - {output_base}/nc_with_dynamics/{config_name}/experiment_{experiment_timestamp}/")
    if args.mode in ['without_dynamics', 'both']:
        print(f"  - {output_base}/nc_without_dynamics/{config_name}/experiment_{experiment_timestamp}/")
    
    print()
    print("Compare the following:")
    print("  1. config.yaml: Full experiment configuration")
    print("  2. results.txt: Training metrics and final accuracies")
    print("  3. nc_metrics_evolution.png: NC1/NC2/NC3 over time")
    print("  4. nc_evolution_3d.mp4: 3D animation video of neural collapse evolution (0.5s per frame)")
    print("  5. nc_evolution_2d.mp4: 2D animation video of neural collapse evolution (0.5s per frame, best for angles)")
    print("  6. 3d-snapshots/nc_viz_step_*.png: Individual 3D visualizations (~200 files)")
    print("  7. 2d-snapshots/nc_viz_step_*.png: Individual 2D visualizations (~200 files)")
    print("  8. training_curves.png: Loss and accuracy progression")
    print("  9. nc_snapshots.pkl: Raw snapshots for analysis")
    print()
    print("See docs/NEURAL_COLLAPSE_VISUALIZATION.md for details!")
    print("=" * 80)

if __name__ == '__main__':
    main()
