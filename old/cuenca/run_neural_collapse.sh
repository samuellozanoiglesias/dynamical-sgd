#!/bin/bash
# ============================================================================
# Neural Collapse MNIST Experiment Runner
# ============================================================================
# Runs neural collapse experiments on MNIST with ResNet-18 architecture
# Supports multiple dynamic bumping modes for comparison.
#
# Usage:
#   ./run_neural_collapse.sh     # Run all four dynamics modes for MNIST
#
# This script runs MNIST configuration with four dynamics modes:
#   1. never:    No bumping (standard training, no class focus)
#   2. pre_tpt:  Bumping ends when reaching TPT accuracy threshold
#   3. tpt_only: Bumping starts only at TPT accuracy threshold
#   4. always:   Bumping throughout entire training
#
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
if [[ ! -f "run_experiment.py" ]]; then
    echo "Error: Not in project root directory. Missing key files."
    echo "Current directory: $(pwd)"
    echo "Expected file: run_experiment.py"
    exit 1
fi

echo "Working from project root: $(pwd)"

# Experiment configuration
CLUSTER="cuenca"
MODE="all"  # Run all dynamics modes: always, pre_tpt, tpt_only, never

# Single seed for each config
SEED=42

# MNIST configuration
CONFIG="spiral_resnet18.yaml"
CONFIG_NAME="${CONFIG%.yaml}"
    
get_training_cfg_value() {
    local key="$1"
    local config_path="config/$CONFIG"
    awk -v key="$key" '
        /^training:/ {in_training=1; next}
        in_training && /^[^[:space:]]/ {in_training=0}
        in_training && $1 == key":" {
            val=$2
            gsub(/"/, "", val)
            print val
            exit
        }
    ' "$config_path"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

run_mnist_experiment() {
    local training_steps
    local total_steps
    local steps_per_epoch
    local derived_epochs

    training_steps="$(get_training_cfg_value training_steps)"
    total_steps="$(get_training_cfg_value total_steps)"
    steps_per_epoch="$(get_training_cfg_value steps_per_epoch)"

    # Support both legacy and new config field names.
    if [[ -z "$training_steps" && -n "$total_steps" ]]; then
        training_steps="$total_steps"
    fi

    if [[ -z "$training_steps" || -z "$steps_per_epoch" ]]; then
        echo "Error: Could not read training.total_steps/training.training_steps or training.steps_per_epoch from config/$CONFIG"
        exit 1
    fi

    derived_epochs=$(( (training_steps + steps_per_epoch - 1) / steps_per_epoch ))

    echo ""
    echo "######################################################################"
    echo "STARTING NEURAL COLLAPSE EXPERIMENTS"  
    echo "######################################################################"
    echo "Dynamics mode: $MODE"
    echo "Cluster: $CLUSTER"
    echo "Seed: $SEED"
    echo ""
    
    echo "Configuration details:"
    echo "  - Dataset: $CONFIG_NAME"
    echo "  - Training steps: $training_steps"
    echo "  - Steps per epoch: $steps_per_epoch"
    echo "  - Derived epochs: $derived_epochs"
    echo "######################################################################"
    echo ""
    
    echo "======================================================================"
    echo "Running experiment with 4 dynamics modes (seed=$SEED)"
    echo "======================================================================"
    
    python run_experiment.py \
        --cluster $CLUSTER \
        --mode $MODE \
        --config_file $CONFIG \
        --seed $SEED
    
    # Check if the experiment succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ ERROR: Experiment failed!"
        exit 1
    fi
    
    echo ""
    echo "✓ Completed experiment"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "========================================================================"
echo "NEURAL COLLAPSE " + $CONFIG_NAME + " EXPERIMENT SUITE"
echo "========================================================================"
echo "Total experiment runs: 4 dynamics modes (never, pre_tpt, tpt_only, always)"
echo "Seed: $SEED"
echo ""
echo "This will test Neural Collapse dynamics with different bumping strategies:"
echo "  1. never:    Standard SGD training without class focus bumps"
echo "  2. pre_tpt:  Dynamic class focusing before Terminal Phase Training"
echo "  3. tpt_only: Dynamic class focusing only during Terminal Phase Training"
echo "  4. always:   Continuous dynamic class focusing throughout training"
echo ""
echo "Results will show how class focus timing affects NC metric evolution"
echo "========================================================================"

# Execute MNIST experiment with all dynamics modes
run_mnist_experiment

echo ""
echo "========================================================================"
echo "🎉 NEURAL COLLAPSE EXPERIMENT COMPLETED SUCCESSFULLY! 🎉"
echo "========================================================================"
echo ""
echo "EXPERIMENT SUMMARY:"
echo "  Dataset: $CONFIG_NAME"
echo "  Seed used: $SEED"
echo "  Total dynamics modes: 4"
echo ""
echo "RESULTS LOCATIONS:"
echo "  📁 Output base: ./outputs/ (created by run_experiment.py)"
echo "  📁 Structure: outputs/nc_*/nc_config_mnist/experiment_*/results/"
echo ""
echo "KEY OUTPUT FILES:"
echo "  📊 training_curves_epoch_*.png - Combined loss/accuracy curves (train vs test)"
echo "  🔍 nc_results.csv - Complete NC metrics in CSV format"
echo "  📋 training_dataset_samples.png - " + $CONFIG_NAME + " sample visualization"
echo "  📊 dataset_statistics.png - Class distribution analysis"
echo "  ⚙️  class_focus_dynamics.png - Weight dynamics during bumping"
echo ""
echo "ANALYSIS RECOMMENDATIONS:"
echo "  1. Compare NC1 (activation collapse) convergence across modes"
echo "  2. Analyze NC2 equiangularity approach to target angle (-0.111)"
echo "  3. Study TPT emergence timing with different bumping schedules"
echo "  4. Evaluate effectiveness of class focus for accelerating NC"
echo ""
echo "Next steps: Analyze NC metrics in nc_results.csv for each mode!"
echo "========================================================================"