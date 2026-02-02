#!/bin/bash
#SBATCH --job-name=neural_collapse_baseline
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-GPU-%x-%j.out
#SBATCH --error=error-GPU-%x-%j.err

# Activar tu entorno
source ~/.bashrc
conda activate dynamical-sgd

# Change to project directory
cd /mnt/lustre/home/samuloza/dynamical-sgd

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_FILE="nc_config_wide_bump.yaml"
MODE="with_dynamics"  # Options: with_dynamics, without_dynamics, both
CLUSTER="brigit"

# Seeds for multiple runs (for statistics)
# To run multiple seeds sequentially, add them to this array
# Example: SEEDS=(21 42 84 168)
# For single run, use: SEEDS=(42)
SEEDS=(21 42 84 168)

# Create logs directory based on configuration name
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
LOGS_DIR="brigit/logs/${CONFIG_NAME}"
mkdir -p "$LOGS_DIR"
echo "Individual experiment logs will be saved to: $LOGS_DIR"

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
echo "========================================================================"
echo "NEURAL COLLAPSE BATCH EXPERIMENT"
echo "========================================================================"
echo "Configuration: $CONFIG_FILE"
echo "Mode: $MODE"
echo "Cluster: $CLUSTER"
echo "Number of runs: ${#SEEDS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "========================================================================"
echo ""

# Run experiments for each seed
for SEED in "${SEEDS[@]}"; do
    echo "========================================================================"
    echo "Running experiment with seed: $SEED"
    echo "========================================================================"
    
    # Create log file name based on config, mode, and seed
    LOG_FILE="$LOGS_DIR/${CONFIG_NAME}_${MODE}_seed_${SEED}_$(date +"%H%M%S").log"
    
    echo "Experiment log file: $LOG_FILE"
    echo "Starting experiment at: $(date)"
    
    # Run experiment with output redirected to individual log file
    python run_nc_experiment.py \
        --cluster $CLUSTER \
        --mode $MODE \
        --config_file $CONFIG_FILE \
        --seed $SEED \
        > "$LOG_FILE" 2>&1
    
    # Check if the experiment succeeded
    EXPERIMENT_EXIT_CODE=$?
    if [ $EXPERIMENT_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "✗ ERROR: Experiment with seed $SEED failed! (Exit code: $EXPERIMENT_EXIT_CODE)"
        echo "✗ Check log file for details: $LOG_FILE"
        echo "✗ Last 20 lines of log:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    
    echo ""
    echo "✓ Completed experiment with seed $SEED at: $(date)"
    echo "✓ Log saved to: $LOG_FILE"
    echo ""
done

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo "Total runs: ${#SEEDS[@]}"
echo "Seeds used: ${SEEDS[@]}"
echo "Individual experiment logs saved to: $LOGS_DIR"
echo ""
echo "Log files created:"
ls -la "$LOGS_DIR"/*.log
echo ""
echo "To view results:"
echo "  - Individual logs: ls $LOGS_DIR/*.log"
echo "  - Experiment outputs: find brigit/data/samuel_lozano/dynamical-sgd -name 'experiment_*' -type d"
echo "========================================================================"