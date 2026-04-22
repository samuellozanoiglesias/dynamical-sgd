#!/bin/bash
#SBATCH --job-name=nc_config_size_500
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
# Array of config files to run
CONFIG_FILES=(nc_config_size_75.yaml nc_config_size_100.yaml nc_config_size_150.yaml nc_config_size_200.yaml nc_config_size_250.yaml)
# CONFIG_FILES=("nc_config_baseline.yaml" "nc_config_big_nn.yaml"  "nc_config_super_big_nn.yaml" "nc_config_narrow_bump.yaml")

MODE="all"  # Options: with_bumps_TPT, without_bumps_TPT, without_dynamics, with_dynamics, all
CLUSTER="brigit"

# Seeds for multiple runs (for statistics)
SEEDS=(10 20 30 40 50)

# Parallelization settings
MAX_PARALLEL=1  # Maximum number of jobs running in parallel (reduced to avoid GPU contention)
DELAY_SECONDS=15  # Delay between starting each job

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Function to count running background jobs
count_running_jobs() {
    jobs -r | wc -l
}

# Function to wait until we have fewer than MAX_PARALLEL jobs
wait_for_slot() {
    while [ $(count_running_jobs) -ge $MAX_PARALLEL ]; do
        echo "  [$(date +"%H:%M:%S")] Waiting for job slot (${MAX_PARALLEL} jobs running)..."
        sleep 10
    done
}

# Function to run a single experiment in background
run_experiment() {
    local seed=$1
    local config_file=$2
    local mode=$3
    local config_name=$4
    local log_file=$5
    
    echo "  [$(date +"%H:%M:%S")] Starting: seed=$seed, config=$config_name, mode=$mode"
    
    python run_experiment.py \
        --cluster $CLUSTER \
        --mode $mode \
        --config_file $config_file \
        --seed $seed \
        > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "  [$(date +"%H:%M:%S")] ✗ FAILED: seed=$seed, config=$config_name, mode=$mode (Exit code: $exit_code)"
        echo "  Check log: $log_file"
    else
        echo "  [$(date +"%H:%M:%S")] ✓ COMPLETED: seed=$seed, config=$config_name, mode=$mode"
    fi
    
    return $exit_code
}

# ============================================================================
# CALCULATE TOTAL JOBS
# ============================================================================

# Calculate how many runs per seed (mode="all" counts as 3, "with_dynamics" counts as 2)
if [ "$MODE" == "all" ]; then
    RUNS_PER_SEED=3
elif [ "$MODE" == "with_dynamics" ]; then
    RUNS_PER_SEED=2
else
    RUNS_PER_SEED=1
fi

TOTAL_JOBS=$((${#CONFIG_FILES[@]} * ${#SEEDS[@]} * RUNS_PER_SEED))

echo "========================================================================"
echo "NEURAL COLLAPSE PARALLEL BATCH EXPERIMENT"
echo "========================================================================"
echo "Configuration files: ${CONFIG_FILES[@]}"
echo "Mode: $MODE (${RUNS_PER_SEED} runs per seed)"
echo "Cluster: $CLUSTER"
echo "Seeds: ${SEEDS[@]}"
echo "Total jobs to run: $TOTAL_JOBS"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Delay between starts: ${DELAY_SECONDS}s"
echo "========================================================================"
echo ""

# ============================================================================
# CREATE LOG DIRECTORIES
# ============================================================================

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    LOGS_DIR="brigit/logs/${CONFIG_NAME}"
    mkdir -p "$LOGS_DIR"
done

# ============================================================================
# RUN EXPERIMENTS IN PARALLEL
# ============================================================================

JOB_COUNT=0
declare -a PIDS=()
declare -a LOG_FILES=()

for SEED in "${SEEDS[@]}"; do
    
    for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
        CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
        LOGS_DIR="brigit/logs/${CONFIG_NAME}"
        
        if [ "$MODE" == "all" ]; then
            # Run all three experiments
            MODES_TO_RUN=("with_bumps_TPT" "without_bumps_TPT" "without_dynamics")
        elif [ "$MODE" == "with_dynamics" ]; then
            # Run both TPT experiments (with and without bumps after 100% acc)
            MODES_TO_RUN=("with_bumps_TPT" "without_bumps_TPT")
        else
            # Run the specified mode
            MODES_TO_RUN=("$MODE")
        fi
        
        for RUN_MODE in "${MODES_TO_RUN[@]}"; do
            JOB_COUNT=$((JOB_COUNT + 1))
            
            # Wait for an available slot
            wait_for_slot
            
            # Create log file
            LOG_FILE="$LOGS_DIR/${CONFIG_NAME}_${RUN_MODE}_seed_${SEED}_$(date +"%H%M%S").log"
            
            echo ""
            echo "========================================================================"
            echo "Job ${JOB_COUNT}/${TOTAL_JOBS}"
            echo "========================================================================"
            echo "Config: $CONFIG_NAME"
            echo "Mode: $RUN_MODE"
            echo "Seed: $SEED"
            echo "Log: $LOG_FILE"
            echo "========================================================================"
            
            # Run experiment in background
            run_experiment $SEED $CONFIG_FILE $RUN_MODE $CONFIG_NAME "$LOG_FILE" &
            
            # Store PID and log file
            PIDS+=($!)
            LOG_FILES+=("$LOG_FILE")
            
            echo "  [$(date +"%H:%M:%S")] Launched job ${JOB_COUNT}/${TOTAL_JOBS} (PID: ${PIDS[-1]})"
            echo "  Currently running: $(count_running_jobs) jobs"
            
            # Delay before starting next job (except for the last one)
            if [ $JOB_COUNT -lt $TOTAL_JOBS ]; then
                echo "  Waiting ${DELAY_SECONDS}s before next job..."
                sleep $DELAY_SECONDS
            fi
        done
    done
done

echo ""
echo "========================================================================"
echo "All jobs launched! Waiting for completion..."
echo "========================================================================"
echo ""

# ============================================================================
# WAIT FOR ALL JOBS TO COMPLETE
# ============================================================================

FAILED_JOBS=0
SUCCESSFUL_JOBS=0

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    LOG_FILE=${LOG_FILES[$i]}
    
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        FAILED_JOBS=$((FAILED_JOBS + 1))
        echo "✗ Job ${i} (PID: $PID) FAILED with exit code $EXIT_CODE"
        echo "  Log: $LOG_FILE"
    else
        SUCCESSFUL_JOBS=$((SUCCESSFUL_JOBS + 1))
    fi
done

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo "ALL JOBS COMPLETED!"
echo "========================================================================"
echo "Total jobs: $TOTAL_JOBS"
echo "Successful: $SUCCESSFUL_JOBS"
echo "Failed: $FAILED_JOBS"
echo ""

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    LOGS_DIR="brigit/logs/${CONFIG_NAME}"
    echo "Logs for ${CONFIG_NAME}:"
    ls -lh "$LOGS_DIR"/*.log 2>/dev/null | tail -10
    echo ""
done

echo "To view results:"
echo "  - Individual logs: find brigit/logs -name '*.log' -mmin -60"
echo "  - Experiment outputs: find /mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd -name 'experiment_*' -type d -mmin -60"
echo "========================================================================"

if [ $FAILED_JOBS -gt 0 ]; then
    exit 1
else
    exit 0
fi