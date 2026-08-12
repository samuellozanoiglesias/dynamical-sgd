#!/bin/bash
#SBATCH --job-name=generate_reduce_dim_plot_launcher
#SBATCH --time=30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-GPU-%x-%j.out
#SBATCH --error=error-GPU-%x-%j.err

# ==========================================
# Environment Setup
# ==========================================
# Activar entorno conda
source ~/.bashrc
conda activate dynamical-sgd

cd /mnt/lustre/home/samuloza/dynamical-sgd/


# ==========================================
# Execution Command
# ==========================================

echo "Starting training job..."

checkpoint_type="cifar10_metaclasses_finetune_intuitive"
checkpoint_id="training_2026_07_24-05_24_13"
pretrained_num_classes=3
metaclass_mapping_type="intuitive"
reduce_method="tsne"
reduce_dims=(2 3)
plot_dims=(3)

for dim in "${reduce_dims[@]}"; do
    for plot_dim in "${plot_dims[@]}"; do
        if [ "$dim" -ge "$plot_dim" ]; then
            echo "Generating plot for checkpoint_type=$checkpoint_type, checkpoint_id=$checkpoint_id, pretrained_num_classes=$pretrained_num_classes, metaclass_mapping_type=$metaclass_mapping_type, reduce_method=$reduce_method, reduce_dim=$dim, plot_dim=$plot_dim"
            python generate_reduce_dim_plot.py \
                --checkpoint_type "$checkpoint_type" \
                --checkpoint_id "$checkpoint_id" \
                --pretrained_num_classes $pretrained_num_classes \
                --metaclass_mapping_type "$metaclass_mapping_type" \
                --reduce_dim $dim \
                --reduce_method $reduce_method \
                --plot_dim $plot_dim
        fi
    done
done

echo "Job finished."