#!/usr/bin/env bash
set -e

# BASELINE training script for RescueNet Mask2Former
# Uses Standard resizing strategy (512x512) and standard Loss

# Define GPU devices (Use 1 GPU for baseline)
DEVICES="0"
NPROC=1

DOCKER_IMAGE="letatanu/semseg_2d:latest"
CONFIG_FILE="/working/nh_datasets/configs/mask2former_rescuenet.py"

docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus "device=${DEVICES}" \
  -w /working \
  -v /media/volume/Tang_Volume/semseg_2d/:/working \
  -v /media/volume/Tang_Volume:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    
    # Activate Environment
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
      . /opt/conda/etc/profile.d/conda.sh
    elif [ -f \$HOME/miniconda3/etc/profile.d/conda.sh ]; then
      . \$HOME/miniconda3/etc/profile.d/conda.sh
    else
      echo 'conda.sh not found in image' >&2; exit 1
    fi
    conda activate semseg

    echo 'Starting Baseline Training...'
    
    # Run Training (Not Evaluation)
    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
      train_mask2former.py \
      --config_file \"${CONFIG_FILE}\" \
      --resume none
    
    echo 'Baseline Training complete!'
    "