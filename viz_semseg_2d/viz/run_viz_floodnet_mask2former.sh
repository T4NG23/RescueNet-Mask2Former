#!/usr/bin/env bash
set -e

## --------------------------------------------------------- ##
DOCKER_IMAGE="letatanu/semseg_2d:latest"

# 1. DOCKER RUN COMMAND: All arguments on one line, including image name and bash -lc block start.
docker run --rm -it -v /dev/shm:/dev/shm --gpus "all" -w /working -v /media/volume/Tang_Volume/semseg_2d/:/working -v /media/volume/Tang_Volume:/data "${DOCKER_IMAGE}" bash -lc "
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

    # 2. PYTHON COMMAND: All arguments on one line, using the correct path to the viz file.
    python viz_semseg_2d/viz/viz_mask2former.py --model /working/runs/mask2former_floodnet/checkpoint-108600 --folder /data/FloodNet-Supervised_v1.0/test/test-org-img/ --gt_folder /data/FloodNet-Supervised_v1.0/test/test-label-img/ --outdir /working/runs/mask2former_floodnet/viz/
"
