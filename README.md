# RescueNet Semantic Segmentation (Mask2Former)

This repository contains a state-of-the-art **Mask2Former** pipeline for semantic segmentation on the **RescueNet** dataset. The goal of this project is to automate post-disaster damage assessment by identifying granular features such as "Building-Total-Destruction," "Road-Blocked," and "Flood Water."

## Key Results
We achieved a peak performance of **71.91% mIoU** using a high-resolution (1024x1024) input strategy with class-weighted loss, significantly outperforming standard baselines.

| Experiment | Resolution | Loss Strategy | mIoU | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 512x512 | Standard | 70.20% | 85.10% |
| **Best Model** | **1024x1024** | **Weighted + TTA** | **71.91%** | **81.71%** |

**Key Insight:** While standard resizing destroys texture details on damaged roofs, our **1024x1024 High-Res strategy** allowed the model to correctly identify "Major Damage" and "Road Blocked" scenarios, which are critical for disaster response.

## Visualizations
Comparison of our High-Res Model against Ground Truth:
![Main Result](demo_figures/main_result.png)
*(Left: Original, Middle: Ground Truth, Right: Mask2Former Prediction, Far Right: Overlay)*

## Methodology
This implementation leverages the Hugging Face `transformers` library and `Mask2FormerForUniversalSegmentation`.

* **Architecture:** Mask2Former (Swin-Large Backbone)
* **Input Strategy:** 1024x1024 Resolution to preserve debris texture.
* **Optimization:** * **Class Weighting:** Applied `3.0x` weights to damage classes and `4.0x` to blocked roads to counter class imbalance.
    * **Stabilization:** Utilized `FP32` precision during training to prevent instability.

## Usage

### 1. Environment Setup
The project runs inside a Docker container for full reproducibility.
\`\`\`bash
# Pull the docker image
docker pull nvcr.io/nvidia/pytorch:23.08-py3

# Run container
docker run --rm -ti --gpus all -v \$(pwd):/working nvcr.io/nvidia/pytorch:23.08-py3 bash
\`\`\`

### 2. Training
To reproduce the 1024px Weighted training run:
\`\`\`bash
./run_train_rescuenet_mask2former.sh
\`\`\`

### 3. Evaluation & Visualization
To generate the 4-panel visualizations:
\`\`\`bash
./viz/run_viz_rescuenet_mask2former.sh
\`\`\`

## Acknowledgements
This project builds upon the RescueNet dataset research.
