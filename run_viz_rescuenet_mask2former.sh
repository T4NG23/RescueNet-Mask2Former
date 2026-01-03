#!/usr/bin/env bash
set -e

DEVICES="all"
DOCKER_IMAGE="letatanu/semseg_2d:latest"

# --- RESCUENET SPECIFIC PATHS ---
# Checkpoint is model.safetensors in the run root folder.
CHECKPOINT_PATH="/working/runs/rescuenet_mask2former/model.safetensors" 

# FIX: Set the input folder to the confirmed host location /media/volume/Tang_Volume/RescueNet/test/
# which translates to the container path:
INPUT_FOLDER="/data/RescueNet/test/" 

OUTPUT_FOLDER="/working/runs/rescuenet_mask2former/viz/" 

docker run --rm -it \
  -v /dev/shm:/dev/shm \
  --gpus "all" \
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

    echo 'Starting RescueNet visualization...'
    
    # Create the viz output directory if it doesn't exist
    mkdir -p \"${OUTPUT_FOLDER}\"
    
    # Check if the input folder exists (this check should pass now)
    if [ ! -d \"${INPUT_FOLDER}\" ]; then
        echo \"FATAL ERROR: Input image folder ${INPUT_FOLDER} not found. Check your data path.\" >&2; exit 1
    fi

    # Run the visualization script
    python /working/viz_semseg_2d/viz/viz_mask2former.py \
      --model \"${CHECKPOINT_PATH}\" \
      --folder \"${INPUT_FOLDER}\" \
      --outdir \"${OUTPUT_FOLDER}\" 
"
