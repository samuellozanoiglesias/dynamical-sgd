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
CONFIG_FILE="nc_config_baseline.yaml"
MODE="both"  # Options: with_dynamics, without_dynamics, both
CLUSTER="brigit"

# Seeds for multiple runs (for statistics)
# To run multiple seeds sequentially, add them to this array
# Example: SEEDS=(21 42 84 168)
# For single run, use: SEEDS=(42)
SEEDS=(21 42 84 168)

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
    
    python run_nc_experiment.py \
        --cluster $CLUSTER \
        --mode $MODE \
        --config_file $CONFIG_FILE \
        --seed $SEED
    
    # Check if the experiment succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ ERROR: Experiment with seed $SEED failed!"
        exit 1
    fi
    
    echo ""
    echo "✓ Completed experiment with seed $SEED"
    echo ""
done

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "========================================================================"
echo "Total runs: ${#SEEDS[@]}"
echo "Seeds used: ${SEEDS[@]}"
echo "========================================================================"