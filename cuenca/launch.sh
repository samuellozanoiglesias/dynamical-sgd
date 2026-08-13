#USE: bash launch.sh

# =====================================================
# Configuration
# =====================================================

CONFIG_NAME="checkerboard-no_bumps"

DATA_DIR="/home/samuel_lozano/dynamical-sgd/data"
OUTPUT_DIR="/data/samuel_lozano/dynamical-sgd"

# Seeds to run
SEEDS=(
    0
    10
    20
    30
    40
    50
    60
    70
    80
    90
)

# Maximum simultaneous jobs
MAX_PARALLEL=4

# =====================================================
# Launch function
# =====================================================

LOG="launch_${CONFIG_NAME}.log"
CONFIG="config/${CONFIG_NAME}.yaml"

# If we're not already running under nohup, restart ourselves.
if [[ -z "$LAUNCH_INTERNAL" ]]; then
    export LAUNCH_INTERNAL=1
    nohup bash "$0" >> "$LOG" 2>&1 &
    echo "Launched in background. Log: $LOG"
    exit 0
fi

cd /home/samuel_lozano/dynamical-sgd || exit 1

run_job () {
    seed=$1

    echo "Starting seed ${seed}"

    python launch.py \
        --config "${CONFIG}" \
        --override use_gpu=false \
        --override data.data_dir="${DATA_DIR}" \
        --override output.output_dir="${OUTPUT_DIR}" \
        --override training.random_seed="${seed}" \
        --override data.random_seed="${seed}"

    echo "Finished seed ${seed}"
}

# =====================================================
# Parallel execution
# =====================================================

running=0

for seed in "${SEEDS[@]}"; do
    run_job "${seed}" &
    ((running++))

    if (( running >= MAX_PARALLEL )); then
        wait -n
        ((running--))
    fi
done

wait

echo "All jobs finished."