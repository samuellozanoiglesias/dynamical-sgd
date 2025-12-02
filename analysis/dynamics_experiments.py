#!/usr/bin/env python3
"""
Specialized analysis experiments extracted from original research notebooks.

This script recreates the specific experimental setups and analysis workflows
from the original dynamics_checks.py file, converted into a proper Python script
with configuration options and result saving.

Usage:
    python analysis/dynamics_experiments.py --experiment_type gradient_phase_analysis
    python analysis/dynamics_experiments.py --experiment_type batch_size_comparison  
    python analysis/dynamics_experiments.py --experiment_type zoomed_analysis
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import argparse
from dataclasses import dataclass, field
import sys
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.spiral_classifier import SpiralClassifier
from utils.visualization import plot_decision_boundary, plot_training_curves, plot_class_focus_dynamics
from config.experiment_config import ExperimentConfig, ModelConfig, TrainingConfig, DynamicsConfig

# Set style
sns.set(style="white")


@dataclass
class DynamicsExperimentConfig:
    """Configuration for dynamics analysis experiments."""
    # Gradient analysis periods
    gradient_initial1: int = 0
    gradient_final1: int = 1500
    gradient_initial2: int = 2300
    gradient_final2: int = 3000
    
    # Alternative gradient periods
    alt_initial1: int = 2300
    alt_final1: int = 3000
    alt_initial2: int = 3500
    alt_final2: int = 5000
    
    # Training parameters
    period: int = 5000
    total_periods: int = 25
    
    # Experiment variants
    network_widths: List[int] = field(default_factory=lambda: [50, 500])
    learning_rates: List[float] = field(default_factory=lambda: [0.002, 0.005, 1.0])
    batch_sizes: List[int] = field(default_factory=lambda: [50, 200])
    w_max_values: List[int] = field(default_factory=lambda: [1, 50, 70, 100, 150])
    period_values: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    
    # Output configuration
    save_results: bool = True
    output_base_dir: str = "dynamics_experiments"
    create_plots: bool = True


def create_base_config() -> ExperimentConfig:
    """Create base experiment configuration."""
    return ExperimentConfig(
        data_config={
            'points_per_class': 100,
            'num_classes': 3,
            'noise_level': 0.2,
            'revolutions': 4
        },
        model_config=ModelConfig(
            nn_width=50,
            num_classes=3,
            activation='relu'
        ),
        training_config=TrainingConfig(
            learning_rate=0.002,
            optimizer='adam',
            batch_size=50,
            total_steps=125001,
            l2_regularization=0.0
        ),
        dynamics_config=DynamicsConfig(
            enable_dynamics=True,
            period_length=5000,
            w_max=70,
            class_focus_schedule='cyclic'
        ),
        output_config={
            'save_model': True,
            'save_plots': True,
            'save_history': True,
            'plot_interval': 100
        }
    )


def run_gradient_phase_analysis(config: DynamicsExperimentConfig) -> Dict[str, Any]:
    """
    Run gradient phase analysis experiments.
    
    Recreates the gradient analysis experiments from the original code:
    - inicio y escalon (initial and step)
    - escalon y final (step and final)  
    - inicio y final (initial and final)
    """
    results = {}
    output_dir = Path(config.output_base_dir) / "gradient_phase_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define phase combinations
    phase_combinations = [
        ("inicio_escalon", config.gradient_initial1, config.gradient_final1, 
         config.gradient_initial2, config.gradient_final2),
        ("escalon_final", config.alt_initial1, config.alt_final1, 
         config.alt_initial2, config.alt_final2),
        ("inicio_final", config.gradient_initial1, config.gradient_final1,
         config.alt_initial2, config.alt_final2)
    ]
    
    for analysis_type in ["gradients", "weights"]:
        results[analysis_type] = {}
        
        for phase_name, initial1, final1, initial2, final2 in phase_combinations:
            print(f"\nRunning {analysis_type} analysis: {phase_name}")
            print(f"Phase 1: {initial1}-{final1}, Phase 2: {initial2}-{final2}")
            
            # Create experiment configuration
            exp_config = create_base_config()
            exp_config.model_config.nn_width = 50
            exp_config.training_config.learning_rate = 0.002
            exp_config.training_config.batch_size = 50
            exp_config.dynamics_config.w_max = 70
            exp_config.dynamics_config.period_length = config.period
            
            # Create classifier and run experiment
            classifier = SpiralClassifier(exp_config)
            
            # Generate datasets
            X_train, Y_train = classifier.generate_spiral_data(seed=0)
            X_test, Y_test = classifier.generate_spiral_data(seed=1)
            
            # Run training with specific analysis parameters
            training_results = classifier.train(
                X_train, Y_train,
                steps=config.total_periods * config.period + 1,
                # These would be passed to analysis functions if they existed
                analysis_config={
                    'gradient_analysis': analysis_type == "gradients",
                    'weight_analysis': analysis_type == "weights", 
                    'phase1_start': initial1,
                    'phase1_end': final1,
                    'phase2_start': initial2,
                    'phase2_end': final2,
                    'period_length': config.period
                }
            )
            
            # Evaluate final performance
            test_accuracy = classifier.compute_accuracy(X_test, Y_test)
            print(f"Test Accuracy: {test_accuracy * 100:.4f}%")
            
            # Save results
            phase_results = {
                'config': exp_config,
                'training_results': training_results,
                'test_accuracy': test_accuracy,
                'phase_config': {
                    'initial1': initial1, 'final1': final1,
                    'initial2': initial2, 'final2': final2
                }
            }
            
            if config.save_results:
                result_file = output_dir / f"{analysis_type}_{phase_name}_results.pkl"
                with open(result_file, 'wb') as f:
                    pickle.dump(phase_results, f)
                print(f"Saved results: {result_file}")
            
            results[analysis_type][phase_name] = phase_results
    
    return results


def run_no_oscillation_comparison(config: DynamicsExperimentConfig) -> Dict[str, Any]:
    """
    Run comparison experiments with oscillations disabled (w_max=1).
    
    Recreates the "No osc" experiments from the original code.
    """
    results = {}
    output_dir = Path(config.output_base_dir) / "no_oscillation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning no-oscillation comparison experiments...")
    
    for analysis_type in ["gradients", "weights"]:
        print(f"\nAnalysis type: {analysis_type}")
        
        # Create experiment configuration with no oscillations
        exp_config = create_base_config()
        exp_config.model_config.nn_width = 50
        exp_config.training_config.learning_rate = 0.005  # Higher LR as in original
        exp_config.training_config.batch_size = 50
        exp_config.dynamics_config.w_max = 1  # No oscillations
        exp_config.dynamics_config.period_length = config.period
        
        # Create classifier and run experiment
        classifier = SpiralClassifier(exp_config)
        
        # Generate datasets
        X_train, Y_train = classifier.generate_spiral_data(seed=0)
        X_test, Y_test = classifier.generate_spiral_data(seed=1)
        
        # Run training
        training_results = classifier.train(
            X_train, Y_train,
            steps=config.total_periods * config.period + 1,
            analysis_config={
                'gradient_analysis': analysis_type == "gradients",
                'weight_analysis': analysis_type == "weights",
                'phase1_start': config.alt_initial1,
                'phase1_end': config.alt_final1,
                'phase2_start': config.alt_initial2,
                'phase2_end': config.alt_final2,
                'period_length': config.period
            }
        )
        
        # Evaluate final performance
        test_accuracy = classifier.compute_accuracy(X_test, Y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.4f}%")
        
        # Save results
        no_osc_results = {
            'config': exp_config,
            'training_results': training_results,
            'test_accuracy': test_accuracy,
            'analysis_type': analysis_type
        }
        
        if config.save_results:
            result_file = output_dir / f"no_oscillation_{analysis_type}_results.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump(no_osc_results, f)
            print(f"Saved results: {result_file}")
        
        results[analysis_type] = no_osc_results
    
    return results


def run_network_width_comparison(config: DynamicsExperimentConfig) -> Dict[str, Any]:
    """
    Run comparison of different network widths.
    
    Recreates the "angulo 50" and "angulo 500" experiments from original code.
    """
    results = {}
    output_dir = Path(config.output_base_dir) / "network_width_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning network width comparison experiments...")
    
    for width in config.network_widths:
        print(f"\nTesting network width: {width}")
        
        # Create experiment configuration
        exp_config = create_base_config()
        exp_config.model_config.nn_width = width
        exp_config.training_config.learning_rate = 0.002
        exp_config.training_config.batch_size = 50
        exp_config.dynamics_config.w_max = 70
        exp_config.dynamics_config.period_length = config.period
        
        # Create classifier and run experiment
        classifier = SpiralClassifier(exp_config)
        
        # Generate datasets
        X_train, Y_train = classifier.generate_spiral_data(seed=0)
        X_test, Y_test = classifier.generate_spiral_data(seed=1)
        
        # Run training with gradient analysis
        training_results = classifier.train(
            X_train, Y_train,
            steps=config.total_periods * config.period + 1,
            analysis_config={
                'gradient_analysis': True,
                'phase1_start': config.alt_initial1,
                'phase1_end': config.alt_final1,
                'phase2_start': config.alt_initial2,
                'phase2_end': config.alt_final2,
                'period_length': config.period
            }
        )
        
        # Evaluate final performance
        test_accuracy = classifier.compute_accuracy(X_test, Y_test)
        print(f"Network width {width} - Test Accuracy: {test_accuracy * 100:.4f}%")
        
        # Save results
        width_results = {
            'config': exp_config,
            'training_results': training_results,
            'test_accuracy': test_accuracy,
            'network_width': width
        }
        
        if config.save_results:
            result_file = output_dir / f"width_{width}_results.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump(width_results, f)
            print(f"Saved results: {result_file}")
        
        results[f"width_{width}"] = width_results
    
    return results


def run_systematic_parameter_study(config: DynamicsExperimentConfig) -> Dict[str, Any]:
    """
    Run systematic parameter studies.
    
    Recreates the systematic experiments from the original systematic_train.py.
    """
    results = {}
    output_dir = Path(config.output_base_dir) / "systematic_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning systematic parameter study...")
    
    for batch_size in config.batch_sizes:
        for learning_rate in [1.0, 0.002]:  # Two main LR settings from original
            batch_results = {}
            
            print(f"\nBatch size: {batch_size}, Learning rate: {learning_rate}")
            
            for w_max in config.w_max_values:
                # Determine which periods to test
                if w_max == 1:
                    periods_to_test = [5000]  # Only test one period for no-oscillation
                else:
                    periods_to_test = config.period_values
                
                for period in periods_to_test:
                    print(f"  w_max: {w_max}, period: {period}")
                    
                    # Create experiment configuration
                    exp_config = create_base_config()
                    exp_config.model_config.nn_width = 50
                    exp_config.training_config.learning_rate = learning_rate
                    exp_config.training_config.batch_size = batch_size
                    exp_config.dynamics_config.w_max = w_max
                    exp_config.dynamics_config.period_length = period
                    exp_config.training_config.total_steps = 15 * 5000 + 1  # Fixed as in original
                    
                    # Create classifier and run experiment
                    classifier = SpiralClassifier(exp_config)
                    
                    # Generate datasets
                    X_train, Y_train = classifier.generate_spiral_data(seed=0)
                    X_test, Y_test = classifier.generate_spiral_data(seed=1)
                    
                    # Run training
                    training_results = classifier.train(X_train, Y_train, steps=exp_config.training_config.total_steps)
                    
                    # Evaluate final performance
                    test_accuracy = classifier.compute_accuracy(X_test, Y_test)
                    print(f"    Test Accuracy: {test_accuracy * 100:.4f}%")
                    
                    # Store results
                    param_key = f"w{w_max}_T{period}"
                    batch_results[param_key] = {
                        'config': exp_config,
                        'training_results': training_results,
                        'test_accuracy': test_accuracy,
                        'w_max': w_max,
                        'period': period
                    }
            
            # Save batch results
            if config.save_results:
                result_file = output_dir / f"batch_{batch_size}_lr_{learning_rate}_results.pkl"
                with open(result_file, 'wb') as f:
                    pickle.dump(batch_results, f)
                print(f"Saved batch results: {result_file}")
            
            results[f"batch_{batch_size}_lr_{learning_rate}"] = batch_results
    
    return results


def create_dynamics_visualization(config: DynamicsExperimentConfig) -> None:
    """
    Create visualization of the dynamic focus mechanism.
    
    Recreates the "Dynamic focus" plots from the original code.
    """
    print("\nCreating dynamics visualization...")
    
    output_dir = Path(config.output_base_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary classifier for dynamics visualization
    temp_config = create_base_config()
    temp_config.dynamics_config.period_length = 5000
    classifier = SpiralClassifier(temp_config)
    
    # Plot class focus dynamics for different w_max values
    plot_class_focus_dynamics(
        period_length=5000,
        num_periods=6,
        w_max_values=[10, 70],
        num_classes=3,
        save_path=output_dir / "dynamic_focus_comparison.png" if config.create_plots else None
    )
    
    print("Dynamics visualization created")


def create_zoomed_analysis_plots(results_file: str, config: DynamicsExperimentConfig) -> None:
    """
    Create zoomed-in analysis plots of training phases.
    
    Recreates the "Zoomed-in" analysis from the original code.
    """
    print(f"\nCreating zoomed analysis from {results_file}...")
    
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found. Skipping zoomed analysis.")
        return
    
    output_dir = Path(config.output_base_dir) / "zoomed_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    if 'training_results' not in results:
        print("No training results found in file")
        return
    
    training_results = results['training_results']
    
    # Extract loss and accuracy histories
    train_losses = training_results.get('train_losses', [])
    test_losses = training_results.get('test_losses', [])
    train_accuracies = training_results.get('train_accuracies', [])
    test_accuracies = training_results.get('test_accuracies', [])
    
    if not train_losses:
        print("No loss data found in training results")
        return
    
    # Create zoomed plots for initial and final phases
    phases = [
        ("initial", 0, 5*5000, "Initial Training Phase"),
        ("final", 10*5000, 15*5000, "Final Training Phase")
    ]
    
    for phase_name, start_idx, end_idx, title in phases:
        fig, axes = plt.subplots(2, 1, figsize=(19, 8))
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(train_losses)-1))
        end_idx = max(start_idx+1, min(end_idx, len(train_losses)))
        
        # Plot losses
        axes[0].plot(test_losses[start_idx:end_idx], ".", ms=0.5, alpha=0.5, 
                    color="red", label='Test Loss')
        axes[0].plot(train_losses[start_idx:end_idx], ".", ms=0.5, alpha=0.5, 
                    color="black", label='Training Loss')
        axes[0].set_title(f'{title} - Loss')
        axes[0].legend(fontsize=12, markerscale=15)
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Add period boundary if relevant
        if phase_name == "initial" and len(train_losses) > 5000:
            axes[0].axvline(x=5000-start_idx, color='black', linestyle='--', alpha=0.7)
        
        # Plot accuracies  
        if train_accuracies and len(train_accuracies) > end_idx:
            axes[1].plot(test_accuracies[start_idx:end_idx], ".", ms=0.5, alpha=0.5, 
                        color='red', label='Test Accuracy')
            axes[1].plot(train_accuracies[start_idx:end_idx], ".", ms=0.5, alpha=0.5, 
                        color='black', label='Training Accuracy')
            axes[1].set_title(f'{title} - Accuracy')
            axes[1].legend(fontsize=12, markerscale=15)
            axes[1].grid(True, alpha=0.3)
            
            if phase_name == "initial":
                axes[1].axvline(x=5000-start_idx, color='black', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if config.create_plots:
            plot_file = output_dir / f"zoomed_{phase_name}_phase.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"Saved zoomed analysis: {plot_file}")
        
        plt.show()


def main():
    """Main function for running dynamics experiments."""
    parser = argparse.ArgumentParser(description='Run dynamics analysis experiments')
    parser.add_argument('--experiment_type', type=str, 
                       choices=['gradient_phase_analysis', 'no_oscillation_comparison',
                               'network_width_comparison', 'systematic_study', 
                               'dynamics_visualization', 'zoomed_analysis', 'all'],
                       default='all', help='Type of experiment to run')
    parser.add_argument('--output_dir', type=str, default='dynamics_experiments',
                       help='Output directory for results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--no_plots', action='store_true', 
                       help='Do not create plots')
    parser.add_argument('--zoomed_results_file', type=str, 
                       help='Results file for zoomed analysis')
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = DynamicsExperimentConfig(
        output_base_dir=args.output_dir,
        save_results=not args.no_save,
        create_plots=not args.no_plots
    )
    
    print("Starting dynamics experiments...")
    print(f"Experiment type: {args.experiment_type}")
    print(f"Output directory: {args.output_dir}")
    
    # Run requested experiments
    if args.experiment_type in ['gradient_phase_analysis', 'all']:
        print("=" * 50)
        gradient_results = run_gradient_phase_analysis(config)
        
    if args.experiment_type in ['no_oscillation_comparison', 'all']:
        print("=" * 50)
        no_osc_results = run_no_oscillation_comparison(config)
        
    if args.experiment_type in ['network_width_comparison', 'all']:
        print("=" * 50)
        width_results = run_network_width_comparison(config)
        
    if args.experiment_type in ['systematic_study', 'all']:
        print("=" * 50)
        systematic_results = run_systematic_parameter_study(config)
        
    if args.experiment_type in ['dynamics_visualization', 'all']:
        print("=" * 50)
        create_dynamics_visualization(config)
        
    if args.experiment_type == 'zoomed_analysis':
        if not args.zoomed_results_file:
            print("Error: --zoomed_results_file required for zoomed analysis")
            return
        print("=" * 50)
        create_zoomed_analysis_plots(args.zoomed_results_file, config)
    
    print("\nDynamics experiments completed!")


if __name__ == "__main__":
    main()