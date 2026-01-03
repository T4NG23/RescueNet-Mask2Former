#!/usr/bin/env bash
set -e

## --------------------------------------------------------- ##
DATASET="binalab"
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -it\
  -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /data/nhl224/code/semantic_2D/FloodSemSeg/:/working \
  -v /data/nhl224/code/semantic_2D/data:/data \
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
        python viz_segformer.py --model runs/segformer_floodnet/checkpoint-47060 \
        --folder /data/FloodNet-Supervised_v1.0/test/test-org-img/ \
        --gt_folder /data/FloodNet-Supervised_v1.0/test/test-label-img/ --outdir runs/segformer_floodnet/viz/
        "