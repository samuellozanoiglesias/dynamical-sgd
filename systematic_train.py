#!/usr/bin/env python3
"""
Systematic parameter studies for dynamical SGD experiments.

This script provides a clean, configurable interface for running systematic
parameter studies, replacing the old monolithic systematic_train.py with a
modern, organized approach using the new codebase structure.

Usage:
    python systematic_train.py                           # Run default systematic study
    python systematic_train.py --config my_config.yaml  # Use custom configuration
    python systematic_train.py --quick                   # Quick test run
    python systematic_train.py --extensive              # Extensive parameter sweep
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import pickle
from dataclasses import dataclass, field
import itertools

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.experiment_config import ExperimentConfig, ModelConfig, TrainingConfig, DynamicsConfig
from src.models.spiral_classifier import SpiralClassifier
from utils.visualization import plot_training_curves, plot_decision_boundary
from analysis.dynamics_experiments import DynamicsExperimentConfig

# GPU check
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("🚀 GPU detected - experiments will run faster!")
        print(f"   {result.stdout.split(chr(10))[8].strip()}")
    else:
        print("💻 No NVIDIA GPU detected, using CPU")
except FileNotFoundError:
    print("❓ GPU status unknown (nvidia-smi not found)")


@dataclass
class SystematicStudyConfig:
    """Configuration for systematic parameter studies."""
    
    # Parameter ranges to sweep
    w_max_values: List[float] = field(default_factory=lambda: [1, 50, 100, 150])
    period_values: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    learning_rates: List[float] = field(default_factory=lambda: [0.002, 1.0])
    batch_sizes: List[int] = field(default_factory=lambda: [50, 200])
    network_widths: List[int] = field(default_factory=lambda: [50])
    
    # Training configuration
    total_periods: int = 15
    base_period: int = 5000
    points_per_class: int = 100
    num_classes: int = 3
    
    # Output configuration
    output_dir: str = "systematic_study_results"
    save_individual_results: bool = True
    save_summary: bool = True
    create_plots: bool = False  # Set to False to avoid overwhelming output
    
    # Execution configuration
    skip_w_max_1_variants: bool = True  # Only run w_max=1 with period=5000
    
    @classmethod
    def quick_config(cls) -> 'SystematicStudyConfig':
        """Create configuration for quick testing."""
        return cls(
            w_max_values=[1, 70],
            period_values=[1000, 5000],
            learning_rates=[0.002],
            batch_sizes=[50],
            total_periods=3,
            create_plots=True
        )
    
    @classmethod
    def extensive_config(cls) -> 'SystematicStudyConfig':
        """Create configuration for extensive parameter sweep."""
        return cls(
            w_max_values=[1, 10, 30, 50, 70, 100, 150, 200],
            period_values=[100, 300, 500, 1000, 3000, 5000],
            learning_rates=[0.001, 0.002, 0.005, 0.01, 0.1, 1.0],
            batch_sizes=[25, 50, 100, 200],
            network_widths=[30, 50, 100, 200],
            total_periods=25
        )


def create_experiment_config(base_config: SystematicStudyConfig, 
                           w_max: float, 
                           period: int,
                           lr: float,
                           batch_size: int,
                           width: int = 50) -> ExperimentConfig:
    """Create experiment configuration for specific parameter combination."""
    
    return ExperimentConfig(
        data_config={
            'points_per_class': base_config.points_per_class,
            'num_classes': base_config.num_classes,
            'noise_level': 0.2,
            'revolutions': 4
        },
        model_config=ModelConfig(
            nn_width=width,
            num_classes=base_config.num_classes,
            activation='relu',
            use_bias=True
        ),
        training_config=TrainingConfig(
            learning_rate=lr,
            optimizer='adam',
            batch_size=batch_size,
            total_steps=base_config.total_periods * base_config.base_period + 1,
            l2_regularization=0.0
        ),
        dynamics_config=DynamicsConfig(
            enable_dynamics=True,
            period_length=period,
            w_max=w_max,
            class_focus_schedule='cyclic'
        ),
        output_config={
            'save_model': False,  # Don't save models for systematic studies
            'save_plots': base_config.create_plots,
            'save_history': False,  # Don't save full history to save space
            'plot_interval': 1000
        }
    )


def run_single_experiment(config: ExperimentConfig, 
                         experiment_id: str,
                         output_dir: Path) -> Dict[str, Any]:
    """Run a single experiment with the given configuration."""
    
    print(f"🔬 Running experiment: {experiment_id}")
    print(f"   w_max={config.dynamics_config.w_max}, "
          f"T={config.dynamics_config.period_length}, "
          f"lr={config.training_config.learning_rate}, "
          f"batch_size={config.training_config.batch_size}")
    
    # Create classifier
    classifier = SpiralClassifier(config)
    
    # Generate datasets with fixed seeds for reproducibility
    X_train, Y_train = classifier.generate_spiral_data(seed=0)
    X_test, Y_test = classifier.generate_spiral_data(seed=1)
    
    # Run training
    training_results = classifier.train(
        X_train, Y_train, 
        steps=config.training_config.total_steps
    )
    
    # Evaluate final performance
    final_train_accuracy = classifier.compute_accuracy(X_train, Y_train)
    final_test_accuracy = classifier.compute_accuracy(X_test, Y_test)
    final_train_loss = classifier.compute_loss(X_train, Y_train)
    final_test_loss = classifier.compute_loss(X_test, Y_test)
    
    print(f"   ✅ Train Acc: {final_train_accuracy:.4f}, "
          f"Test Acc: {final_test_accuracy:.4f}")
    
    # Package results
    results = {
        'experiment_id': experiment_id,
        'config': config,
        'final_metrics': {
            'train_accuracy': float(final_train_accuracy),
            'test_accuracy': float(final_test_accuracy),
            'train_loss': float(final_train_loss),
            'test_loss': float(final_test_loss)
        },
        'training_history': training_results,
        'parameters': {
            'w_max': config.dynamics_config.w_max,
            'period_length': config.dynamics_config.period_length,
            'learning_rate': config.training_config.learning_rate,
            'batch_size': config.training_config.batch_size,
            'network_width': config.model_config.nn_width
        }
    }
    
    return results


def run_systematic_study(study_config: SystematicStudyConfig) -> Dict[str, Any]:
    """Run systematic parameter study with the given configuration."""
    
    print("🧪 Starting Systematic Parameter Study")
    print("=" * 60)
    print(f"Parameter ranges:")
    print(f"  w_max: {study_config.w_max_values}")
    print(f"  periods: {study_config.period_values}")
    print(f"  learning rates: {study_config.learning_rates}")
    print(f"  batch sizes: {study_config.batch_sizes}")
    print(f"  network widths: {study_config.network_widths}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(study_config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        study_config.w_max_values,
        study_config.period_values,
        study_config.learning_rates,
        study_config.batch_sizes,
        study_config.network_widths
    ))
    
    # Filter combinations if requested
    if study_config.skip_w_max_1_variants:
        # For w_max=1, only use period=5000 (no oscillation baseline)
        filtered_combinations = []
        for w_max, period, lr, batch_size, width in param_combinations:
            if w_max == 1 and period != 5000:
                continue  # Skip this combination
            filtered_combinations.append((w_max, period, lr, batch_size, width))
        param_combinations = filtered_combinations
    
    print(f"Running {len(param_combinations)} parameter combinations...")
    
    all_results = {}
    summary_data = []
    
    for i, (w_max, period, lr, batch_size, width) in enumerate(param_combinations):
        
        # Create experiment configuration
        experiment_config = create_experiment_config(
            study_config, w_max, period, lr, batch_size, width
        )
        
        # Generate experiment ID
        experiment_id = f"w{w_max}_T{period}_lr{lr}_bs{batch_size}_width{width}"
        
        print(f"\n[{i+1}/{len(param_combinations)}] {experiment_id}")
        
        try:
            # Run experiment
            results = run_single_experiment(
                experiment_config, experiment_id, output_dir
            )
            
            all_results[experiment_id] = results
            
            # Add to summary data
            summary_data.append({
                'experiment_id': experiment_id,
                'w_max': w_max,
                'period_length': period,
                'learning_rate': lr,
                'batch_size': batch_size,
                'network_width': width,
                'train_accuracy': results['final_metrics']['train_accuracy'],
                'test_accuracy': results['final_metrics']['test_accuracy'],
                'train_loss': results['final_metrics']['train_loss'],
                'test_loss': results['final_metrics']['test_loss']
            })
            
            # Save individual results if requested
            if study_config.save_individual_results:
                result_file = output_dir / f"{experiment_id}.pkl"
                with open(result_file, 'wb') as f:
                    pickle.dump(results, f)
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    # Save summary results
    if study_config.save_summary:
        summary_file = output_dir / "systematic_study_summary.pkl"
        with open(summary_file, 'wb') as f:
            pickle.dump({
                'study_config': study_config,
                'all_results': all_results,
                'summary_data': summary_data
            }, f)
        print(f"\n📊 Saved summary to: {summary_file}")
        
        # Also save as CSV for easy analysis
        try:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            csv_file = output_dir / "systematic_study_summary.csv"
            df.to_csv(csv_file, index=False)
            print(f"📈 Saved CSV summary to: {csv_file}")
        except ImportError:
            print("📝 Install pandas to get CSV summary: pip install pandas")
    
    print("\n🎉 Systematic study completed!")
    print(f"📁 Results saved in: {output_dir}")
    
    return {
        'study_config': study_config,
        'results': all_results,
        'summary': summary_data
    }


def analyze_results(results_file: str):
    """Analyze and display results from a systematic study."""
    
    print("📈 Analyzing systematic study results...")
    
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    # Load results
    with open(results_path, 'rb') as f:
        data = pickle.load(f)
    
    summary_data = data['summary_data']
    
    if not summary_data:
        print("❌ No summary data found")
        return
    
    print(f"📊 Loaded {len(summary_data)} experiments")
    
    # Basic statistics
    print("\n🏆 Top 10 Test Accuracies:")
    sorted_by_test_acc = sorted(summary_data, 
                               key=lambda x: x['test_accuracy'], 
                               reverse=True)
    
    for i, result in enumerate(sorted_by_test_acc[:10]):
        print(f"  {i+1:2d}. {result['experiment_id']:30s} "
              f"Test Acc: {result['test_accuracy']:.4f}")
    
    # Performance by parameter
    print("\n📊 Performance by w_max:")
    w_max_groups = {}
    for result in summary_data:
        w_max = result['w_max']
        if w_max not in w_max_groups:
            w_max_groups[w_max] = []
        w_max_groups[w_max].append(result['test_accuracy'])
    
    for w_max in sorted(w_max_groups.keys()):
        accuracies = w_max_groups[w_max]
        mean_acc = sum(accuracies) / len(accuracies)
        print(f"  w_max={w_max:3.0f}: {mean_acc:.4f} "
              f"(±{(max(accuracies) - min(accuracies))/2:.4f})")
    
    print("\n📊 Performance by period length:")
    period_groups = {}
    for result in summary_data:
        period = result['period_length']
        if period not in period_groups:
            period_groups[period] = []
        period_groups[period].append(result['test_accuracy'])
    
    for period in sorted(period_groups.keys()):
        accuracies = period_groups[period]
        mean_acc = sum(accuracies) / len(accuracies)
        print(f"  T={period:4d}: {mean_acc:.4f} "
              f"(±{(max(accuracies) - min(accuracies))/2:.4f})")


def main():
    """Main function for systematic parameter studies."""
    
    parser = argparse.ArgumentParser(
        description="Run systematic parameter studies for dynamical SGD experiments"
    )
    parser.add_argument('--config', type=str, 
                       help='YAML configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test configuration')
    parser.add_argument('--extensive', action='store_true',
                       help='Run extensive parameter sweep')
    parser.add_argument('--analyze', type=str,
                       help='Analyze results from previous run (provide .pkl file)')
    parser.add_argument('--output_dir', type=str, default='systematic_study_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results(args.analyze)
        return
    
    # Create study configuration
    if args.quick:
        print("🏃 Running quick systematic study...")
        study_config = SystematicStudyConfig.quick_config()
    elif args.extensive:
        print("🌍 Running extensive systematic study...")
        study_config = SystematicStudyConfig.extensive_config()
    elif args.config:
        print(f"📋 Loading configuration from {args.config}...")
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        study_config = SystematicStudyConfig(**config_dict)
    else:
        print("📊 Running default systematic study...")
        study_config = SystematicStudyConfig()
    
    # Override output directory if specified
    if args.output_dir != 'systematic_study_results':
        study_config.output_dir = args.output_dir
    
    # Run the study
    results = run_systematic_study(study_config)
    
    # Quick analysis
    if results['summary']:
        print("\n" + "="*60)
        analyze_results(Path(study_config.output_dir) / "systematic_study_summary.pkl")


if __name__ == "__main__":
    main()