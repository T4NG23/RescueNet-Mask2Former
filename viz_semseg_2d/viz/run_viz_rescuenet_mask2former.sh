#!/usr/bin/env bash
set -e

# --- CONFIGURATION ---
DOCKER_IMAGE="nvcr.io/nvidia/pytorch:23.08-py3"

# 1. Point to your BEST Model (Dec 16 Checkpoint)
CHECKPOINT_PATH="/working/runs/rescuenet_mask2former_1024_weighted/checkpoint-19500" 

# 2. Input Images (Use Validation set since we confirmed path exists)
INPUT_FOLDER="/data/RescueNet/val/val-org-img/" 

# 3. Ground Truth Labels (Use Validation set)
GT_FOLDER="/data/RescueNet/val/val-label-img/"

# 4. Output Folder
OUTPUT_FOLDER="/working/runs/rescuenet_mask2former_1024_weighted/visualizations/" 

echo "------------------------------------------------"
echo "Running Visualization for Checkpoint:"
echo "${CHECKPOINT_PATH}"
echo "Input: ${INPUT_FOLDER}"
echo "------------------------------------------------"

docker run --rm -it \
  -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /media/volume/Tang_Volume/semseg_2d/:/working \
  -v /media/volume/Tang_Volume:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    
    # Install dependencies (pinned to avoid numpy/transformers crashes)
    pip install \"numpy<2\" \"transformers==4.35.2\" \"accelerate\" \"soxr\" \"matplotlib\" \"pillow\" \"scikit-learn\" \"opencv-python-headless\" --no-cache-dir -q > /dev/null 2>&1

    mkdir -p \"${OUTPUT_FOLDER}\"

    python /working/viz_semseg_2d/viz/viz_mask2former.py \
      --model \"${CHECKPOINT_PATH}\" \
      --folder \"${INPUT_FOLDER}\" \
      --gt_folder \"${GT_FOLDER}\" \
      --outdir \"${OUTPUT_FOLDER}\" \
      --device cuda
"
