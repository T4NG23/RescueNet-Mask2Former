#!/usr/bin/env bash
set -e

# SETTINGS
NPROC=1
DOCKER_IMAGE="nvcr.io/nvidia/pytorch:23.08-py3"
CONFIG_FILE="/working/nh_datasets/configs/mask2former_rescuenet_optimized.py"
DATASET_FILE="/working/nh_datasets/rescuenet_optimized.py"

# RUN DOCKER
docker run --rm -ti \
  -v /dev/shm:/dev/shm \
  --gpus all \
  -w /working \
  -v /media/volume/Tang_Volume/semseg_2d/:/working \
  -v /media/volume/Tang_Volume:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -e
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=4 

    echo '--------------------------------------'
    echo ' [SETUP] Installing System Libs...'
    echo '--------------------------------------'
    apt-get update && apt-get install -y libgl1 libglib2.0-0

    echo ' [SETUP] Checking Environment...'
    if ! python -c 'import transformers; print(transformers.__version__)' | grep -q '4.35.2'; then
        echo ' [INSTALL] Installing Golden Dependencies...'
        pip uninstall -y opencv-python opencv-python-headless numpy numba accelerate transformers || true
        rm -rf /usr/local/lib/python3.10/dist-packages/cv2
        rm -rf /usr/local/lib/python3.10/dist-packages/opencv*
        
        # Golden Set
        pip install \"transformers==4.35.2\" \"accelerate==0.25.0\" \"numpy==1.23.5\" \"numba\" \"soxr\" \"soundfile\" \"opencv-python-headless==4.6.0.66\" albumentations evaluate scikit-learn --no-cache-dir -q
    fi

    echo ' [INFO] Applying Final Optimizations...'
    
    # 1. FIX STABILITY: Disable FP16 in Python (Prevents NaN/Infeasible Matrix)
    sed -i 's/fp16=True,/fp16=False,/g' train_mask2former_optimized.py
    
    # 2. FIX SPEED/MEMORY: Reduce Image Size to 512 in Config
    # 1024 is too big for T4 FP32. 512 will run 4x faster.
    sed -i 's/\"image_size\": 1024/\"image_size\": 512/g' ${CONFIG_FILE}
    
    # 3. FIX MEMORY: Force Batch Size 1
    sed -i 's/batch_size = 2/batch_size = 1/g' ${CONFIG_FILE}

    # 4. FIX CPU: Disable Copy-Paste (High I/O)
    sed -i 's/if random.random() > 0.5/if True/g' ${DATASET_FILE}

    echo ' [INFO] Patching Python Script...'
    # Clean previous edits
    sed -i 's/evaluation_strategy=\"steps\",//g' train_mask2former_optimized.py
    sed -i 's/load_best_model_at_end=True,//g' train_mask2former_optimized.py
    
    # Inject Config
    sed -i 's/save_strategy=\"steps\",/save_strategy=\"steps\", evaluation_strategy=\"steps\", load_best_model_at_end=True,/g' train_mask2former_optimized.py

    # Fix Version Mismatches
    sed -i 's/processing_class=image_processor/tokenizer=image_processor/g' train_mask2former_optimized.py
    sed -i 's/self.processing_class/self.tokenizer/g' train_mask2former_optimized.py
    # Fix Scalar Loss
    sed -i 's/return (outputs.loss, outputs) if return_outputs else outputs.loss/return (outputs.loss.mean(), outputs) if return_outputs else outputs.loss.mean()/g' train_mask2former_optimized.py

    echo ' [INFO] Starting Training...'
    
    # NOTE: --fp16 0 disables mixed precision at the CLI level too
    torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
      train_mask2former_optimized.py \
      --config_file \"${CONFIG_FILE}\" \
      --resume none \
      --fp16 0
    "