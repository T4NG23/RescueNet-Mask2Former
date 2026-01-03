#!/usr/bin/env bash
set -e

DEVICES="0" 
NPROC=$(( $(tr -cd ',' <<<"$DEVICES" | wc -c) + 1 ))


## --------------------------------------------------------- ##
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -ti\
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /media/volume/data_2d/semseg_2d/:/working \
  -v /media/volume/data_2d/:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
        set -euo pipefail
        export OMP_NUM_THREADS=16
        # Activate conda (adjust if your image uses a different prefix)
        if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
        . /opt/conda/etc/profile.d/conda.sh
        elif [ -f \$HOME/miniconda3/etc/profile.d/conda.sh ]; then
        . \$HOME/miniconda3/etc/profile.d/conda.sh
        else
        echo 'conda.sh not found in image' >&2; exit 1
        fi
        conda activate semseg
        torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC}  \
        train_mask2former.py   --config_file /working/nh_datasets/configs/mask2former_crarsar.py \
        "