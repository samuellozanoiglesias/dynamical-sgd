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
# Dynamics Mode Controls (bumps_before_TPT / bumps_at_TPT):
#   - never:    false / false  (standard SGD, no dynamics)
#   - pre_tpt:  true / false   (stop at TPT accuracy)
#   - tpt_only: false / true   (start at TPT accuracy)
#   - always:   true / true    (continuous throughout)
#
# Expected Neural Collapse behavior for MNIST (10 classes, ResNet-18):
#   Target equiangular angle: -0.111 (83.6° between class centers)
#   TPT emergence: ~2500-3500 training steps (roughly 5-7 epochs, 99% accuracy threshold)
#   NC metrics: Activation collapse, equinorm, equiangularity, self-duality
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
MODE="tpt_only"  # Run all dynamics modes: always, pre_tpt, tpt_only, never

# Single seed for each config
SEED=42

# MNIST configuration
CONFIG="nc_config_mnist_mlp.yaml"
CONFIG_NAME="mnist_mlp"
    
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
    local steps_per_epoch
    local derived_epochs

    training_steps="$(get_training_cfg_value training_steps)"
    steps_per_epoch="$(get_training_cfg_value steps_per_epoch)"

    if [[ -z "$training_steps" || -z "$steps_per_epoch" ]]; then
        echo "Error: Could not read training.training_steps or training.steps_per_epoch from config/$CONFIG"
        exit 1
    fi

    derived_epochs=$(( (training_steps + steps_per_epoch - 1) / steps_per_epoch ))

    echo ""
    echo "######################################################################"
    echo "STARTING MNIST NEURAL COLLAPSE EXPERIMENTS"  
    echo "######################################################################"
    echo "Configuration: MNIST with different architectures and dynamics modes"
    echo "Dynamics mode: $MODE"
    echo "Cluster: $CLUSTER"
    echo "Seed: $SEED"
    echo ""
    
    echo "Configuration details:"
    echo "  - Dataset: MNIST (10 classes, 28×28 grayscale images)"
    echo "  - Training steps: $training_steps"
    echo "  - Steps per epoch: $steps_per_epoch"
    echo "  - Derived epochs: $derived_epochs"
    echo "  - Batch size: 128"
    echo "  - Optimizer: SGD with momentum=0.9, weight_decay=5e-4"
    echo "  - Learning rate: 0.0679 (tuned for CrossEntropyLoss)"
    echo "  - Target equiangular angle: -0.111 (83.6° between class centers)"
    echo "  - TPT threshold: 100% accuracy (zero training error)"
    echo "######################################################################"
    echo ""
    
    echo "======================================================================"
    echo "Running MNIST experiment with 4 dynamics modes (seed=$SEED)"
    echo "======================================================================"
    
    python run_nc_experiment.py \
        --cluster $CLUSTER \
        --mode $MODE \
        --config_file $CONFIG \
        --seed $SEED
    
    # Check if the experiment succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ ERROR: MNIST experiment failed!"
        exit 1
    fi
    
    echo ""
    echo "✓ Completed MNIST experiment"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "========================================================================"
echo "NEURAL COLLAPSE MNIST EXPERIMENT SUITE"
echo "========================================================================"
echo "Running MNIST dataset with ResNet-18 architecture"
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
echo "🎉 MNIST NEURAL COLLAPSE EXPERIMENT COMPLETED SUCCESSFULLY! 🎉"
echo "========================================================================"
echo ""
echo "EXPERIMENT SUMMARY:"
echo "  Dataset: MNIST (10 classes)"
echo "  Architecture: ResNet-18"
echo "  Seed used: $SEED"
echo "  Total dynamics modes: 4"
echo ""
echo "RESULTS LOCATIONS:"
echo "  📁 Output base: ./outputs/ (created by run_nc_experiment.py)"
echo "  📁 Structure: outputs/nc_*/nc_config_mnist/experiment_*/results/"
echo ""
echo "KEY OUTPUT FILES:"
echo "  📊 training_curves_epoch_*.png - Combined loss/accuracy curves (train vs test)"
echo "  🔍 nc_results.csv - Complete NC metrics in CSV format"
echo "  📋 training_dataset_samples.png - MNIST sample visualization"
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