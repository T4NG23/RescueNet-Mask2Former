#!/usr/bin/env bash
set -e

# Define GPU devices to use (comma-separated list)
DEVICES="0"

# --- FIX: Use robust NPROC calculation ---
# This reliably counts the number of devices (1 for "0", 2 for "0,1", etc.)
NPROC=$(echo "$DEVICES" | awk -F',' '{print NF}')

echo "Using GPUs: ${DEVICES}"
echo "Setting NPROC: ${NPROC}"

## ------------------- DOCKER CONFIG ----------------------- ##
DOCKER_IMAGE="letatanu/semseg_2d:latest"

# --- RESCUENET SPECIFIC PATHS ---
# Define the path to your best model checkpoint *inside the container*
# We are targeting the RescueNet run directory and checkpoint-269700.
# The volume mount /media/volume/Tang_Volume/semseg_2d/ is /working/
CHECKPOINT_PATH_DIR="/working/runs/rescuenet_mask2former/checkpoint-269700"
CHECKPOINT_FILE="${CHECKPOINT_PATH_DIR}/model.safetensors"
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
    export OMP_NUM_THREADS=16

    # Activate conda environment
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
      . /opt/conda/etc/profile.d/conda.sh
    elif [ -f \$HOME/miniconda3/etc/profile.d/conda.sh ]; then
      . \$HOME/miniconda3/etc/profile.d/conda.sh
    else
      echo 'conda.sh not found in image' >&2; exit 1
    fi
    conda activate semseg

    # --- EVALUATION COMMAND (Mask2Former - RescueNet) ---
    
    # Use the defined checkpoint file path
    CHECKPOINT_FILE=\"${CHECKPOINT_FILE}\"
    CONFIG_FILE=\"${CONFIG_FILE}\"

    echo \"Running evaluation on RescueNet using checkpoint: \$CHECKPOINT_FILE\"

    # Check if the file exists before running the evaluation
    if [ ! -f \"\$CHECKPOINT_FILE\" ]; then
      echo \"Error: Checkpoint file \$CHECKPOINT_FILE not found in the container. Double-check your checkpoint folder and file name.\" >&2; exit 1
    fi

    # The --evaluate argument now correctly points to the model file.
    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
      train_mask2former.py \
      --config_file \"\$CONFIG_FILE\" \
      --evaluate \"\$CHECKPOINT_FILE\"
    "
