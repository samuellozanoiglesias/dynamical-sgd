#!/bin/bash
# ============================================================================
# Neural Collapse Batch Experiment Runner (Enhanced)
# ============================================================================
# This script runs neural collapse experiments with multiple random seeds
# for statistical analysis. Supports both spiral and MNIST datasets with
# all possible configurations.
#
# Usage:
#   ./run_neural_collapse.sh     # Run all four configs in sequence
#
# The script runs four specific configurations in order:
#   1. MNIST dataset configuration
#   2. Super big neural network configuration  
#   3. Super big neural network (revoluted) configuration
#   4. Baseline spiral configuration
#
# Dynamics Modes (controlled by bumps_before_TPT and bumps_at_TPT):
#   - always:   bumps_before_TPT=true,  bumps_at_TPT=true   (bump throughout)
#   - pre_tpt:  bumps_before_TPT=true,  bumps_at_TPT=false  (stop at TPT)
#   - tpt_only: bumps_before_TPT=false, bumps_at_TPT=true   (start at TPT)
#   - never:    bumps_before_TPT=false, bumps_at_TPT=false  (no bumps)
#
# Expected Neural Collapse behavior comparison:
#   Spiral (3 classes):  Target angle = -0.5   (120° between class centers)
#   MNIST (10 classes):  Target angle = -0.111 (83.6° between class centers)
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================

# Change to project root directory (parent of cuenca/)
cd "$(dirname "$0")/.." || {
    echo "Error: Cannot change to project root directory"
    exit 1
}

# Verify we're in the correct directory by checking for key files
if [[ ! -f "run_nc_experiment.py" || ! -f "run_experiment.py" ]]; then
    echo "Error: Not in project root directory. Missing key files."
    echo "Current directory: $(pwd)"
    echo "Expected files: run_nc_experiment.py, run_experiment.py"
    exit 1
fi

echo "Working from project root: $(pwd)"

# Experiment configuration
CLUSTER="cuenca"
MODE="all"  # Run all dynamics modes: always, pre_tpt, tpt_only, never

# Single seed for each config
SEED=42

# Four configurations to run in sequence
CONFIGS=("nc_config_mnist.yaml" "nc_config_baseline.yaml")
CONFIG_NAMES=("mnist" "baseline")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

run_config_experiment() {
    local config_name=$1
    local config_file=$2
    
    echo ""
    echo "######################################################################"
    echo "STARTING $config_name EXPERIMENT"  
    echo "######################################################################"
    echo "Configuration name: $config_name"
    echo "Configuration file: $config_file"
    echo "Mode: $MODE (all dynamics modes)"
    echo "Cluster: $CLUSTER"
    echo "Seed: $SEED"
    echo ""
    
    # Display configuration details
    echo "Configuration details for $config_name:"
    if [ "$config_name" = "mnist" ]; then
        echo "  - Dataset: MNIST (10 classes, 784-dim)"
        echo "  - Target angle: -0.111 (83.6° between class centers)"
        echo "  - Challenge: High-dimensional, complex data distribution"
    elif [ "$config_name" = "super_big_nn" ]; then
        echo "  - Dataset: Spiral with large neural network"
        echo "  - Focus: Large capacity model behavior"
    elif [ "$config_name" = "super_big_nn_revoluted" ]; then
        echo "  - Dataset: Spiral with large neural network (alternative config)"
        echo "  - Focus: Large capacity model with different parameters"
    elif [ "$config_name" = "baseline" ]; then
        echo "  - Dataset: Spiral (3 classes, 2-dim)"
        echo "  - Target angle: -0.5 (120° between class centers)"
        echo "  - Standard baseline configuration"
    fi
    echo "######################################################################"
    echo ""
    
    echo "======================================================================"
    echo "Running $config_name experiment with seed: $SEED"
    echo "======================================================================"
    
    python run_nc_experiment.py \
        --cluster $CLUSTER \
        --mode $MODE \
        --config_file $config_file \
        --seed $SEED
    
    # Check if the experiment succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ ERROR: $config_name experiment failed!"
        exit 1
    fi
    
    echo ""
    echo "✓ Completed $config_name experiment"
    echo ""
    
    echo ""
    echo "######################################################################"
    echo "✓ $config_name EXPERIMENT COMPLETED SUCCESSFULLY!"
    echo "######################################################################"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "========================================================================"
echo "NEURAL COLLAPSE FOUR-CONFIG EXPERIMENT SUITE"
echo "========================================================================"
echo "Running 4 configurations in sequence:"
echo "  1. MNIST dataset"
echo "  2. Super big neural network"
echo "  3. Super big neural network (revoluted)"
echo "  4. Baseline spiral"
echo "Total experiments: 4 configs × 4 dynamics modes = 16 total runs"
echo "Seed: $SEED"
echo ""
echo "This will generate Neural Collapse analysis comparing different:"
echo "  1. Dataset types (MNIST vs spiral)"
echo "  2. Network architectures (standard vs super big)"
echo "  3. Configuration variations"
echo "  4. Dynamics modes (bumping strategies)"
echo "========================================================================"

# Execute experiments for each configuration
for i in "${!CONFIGS[@]}"; do
    run_config_experiment "${CONFIG_NAMES[$i]}" "${CONFIGS[$i]}"
done

echo ""
echo "========================================================================"
echo "🎉 ALL FOUR-CONFIG EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉"
echo "========================================================================"
echo ""
echo "EXPERIMENT SUMMARY:"
echo "  Configurations run: 4 (mnist, super_big_nn, super_big_nn_revoluted, baseline)"
echo "  Seed used: $SEED"
echo "  Total experiment runs: 16 (4 configs × 4 dynamics modes)"
echo ""
echo "RESULTS LOCATIONS:"
echo "  📁 MNIST results:        outputs/nc_*/nc_config_mnist/experiment_*/"
echo "  📁 Super big NN results: outputs/nc_*/nc_config_super_big_nn/experiment_*/"
echo "  📁 Super big NN rev:     outputs/nc_*/nc_config_super_big_nn_false/experiment_*/"
echo "  📁 Baseline results:     outputs/nc_*/nc_config_baseline/experiment_*/"
echo ""
echo "KEY OUTPUT FILES FOR COMPARISON:"
echo "  📊 nc_metrics_evolution.png - Main NC metrics over time"
echo "  📈 training_curves.png - Loss and accuracy progression"  
echo "  🎬 nc_evolution_*.mp4 - Animation videos (spiral configs only)"
echo "  📋 results.txt - Final metrics summary"
echo "  🔍 nc_snapshots.pkl - Raw data for further analysis"
echo ""
echo "ANALYSIS RECOMMENDATIONS:"
echo "  1. Compare MNIST vs spiral NC metric convergence patterns"
echo "  2. Analyze effect of network size (standard vs super big)"
echo "  3. Compare dynamics modes across all configurations"
echo "  4. Validate architectural effects on Neural Collapse behavior"
echo ""
echo "Next steps: Compare results across the four configurations!"
echo "========================================================================"