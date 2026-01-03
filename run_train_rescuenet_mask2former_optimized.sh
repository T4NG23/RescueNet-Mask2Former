#!/usr/bin/env bash
set -e

# OPTIMIZED training script for RescueNet Mask2Former (Dual L40S)
NPROC=2

DOCKER_IMAGE="letatanu/semseg_2d:latest"
CONFIG_FILE="/working/nh_datasets/configs/mask2former_rescuenet_optimized.py"
OUTPUT_DIR="/working/runs/rescuenet_mask2former_ceiling"

# CLEANUP
# fuser -k -v /dev/nvidia0 /dev/nvidia1 || true

docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus all \
  -w /working \
  -v /media/volume/Tang_Volume/semseg_2d/:/working \
  -v /media/volume/Tang_Volume:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=8 

    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
      . /opt/conda/etc/profile.d/conda.sh
    elif [ -f \$HOME/miniconda3/etc/profile.d/conda.sh ]; then
      . \$HOME/miniconda3/etc/profile.d/conda.sh
    else
      echo 'conda.sh not found in image' >&2; exit 1
    fi
    conda activate semseg

    echo 'Installing Albumentations...'
    pip install albumentations --no-cache-dir -q

    echo \"Starting Training...\"
    
    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
      train_mask2former_optimized.py \
      --config_file \"${CONFIG_FILE}\" \
      --resume none \
      --fp16 1
    
    echo \"Training complete!\"
    "