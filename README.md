# RescueNet Semantic Segmentation (Mask2Former)

This repository contains a state-of-the-art **Mask2Former** pipeline for semantic segmentation on the **RescueNet** dataset. The goal of this project is to automate post-disaster damage assessment by identifying granular features such as "Building-Total-Destruction," "Road-Blocked," and "Flood Water."

This release represents our **"Highest Ceiling" Optimization Strategy**, achieving a peak performance of **71.91% mIoU** by combining high-resolution texture preservation with aggressive synthetic data generation (Copy-Paste Augmentation).

## Methodology: The Optimization Evolution

Our approach evolved through two distinct phases of engineering to solve the specific challenges of satellite damage assessment.

### Phase 1: Texture Preservation (Passive Strategy)
Initial baselines failed to identify "Minor Damage" because standard resizing (512x512) blurred fine cracks and debris.
* **High-Resolution Input:** We locked the input size to **1024x1024**. This forced the model to process fine-grained details necessary for subtle damage classification.
* **Disabled Distortions:** We explicitly **disabled** shear, random scale, and color jitter. This ensured the model saw "true" building geometry without artificial warping that could mimic (or hide) structural damage.

### Phase 2: "Highest Ceiling" (Active Strategy)
To break past the 70% plateau, we addressed the extreme scarcity of critical classes like "Road Blocked" (Class 8).

* **1. "Smart" Copy-Paste Augmentation:**
    * **Problem:** "Road-Blocked" examples were too rare for the model to learn effectively.
    * **Solution:** We implemented a **Copy-Paste (SCP)** routine that physically "cuts" debris and vehicles from source images and "pastes" them onto clear roads in other training samples. This synthetically generated thousands of new "Blocked Road" scenarios.

* **2. Aggressive Loss Reweighting:**
    We modified the Cross-Entropy loss function to heavily penalize missing safety-critical classes:
    * **Road-Blocked:** Weight increased to **4.0x** (Highest priority).
    * **Building Damage:** Weights maintained at **3.0x** to focus on distinguishing damage levels.
    * **Background:** Weight kept at 1.0.

* **3. Test-Time Augmentation (TTA):**
    We implemented a **3-Pass Voting System** during inference. The model predicts on the original image + horizontally flipped + vertically flipped versions, averaging the results to smooth out noise.

## Visualizations: Training-Aligned Inference

A major challenge was generating accurate visualizations for the massive original images (3000x4000px).

**The Solution:** We engineered a custom visualization pipeline that mirrors our 1024x1024 training logic. Instead of squashing the image (which destroys aspect ratio), we perform **Sliding Window Inference** with overlap handling, ensuring the model sees the same high-resolution features during inference as it did during training.

![Main Result](demo_figures/main_result.png)
*(Left: Original, Middle: Ground Truth, Right: Model Prediction, Far Right: Overlay)*

## Technical Implementation

* **Architecture:** Mask2Former (Swin-Large Backbone)
* **Optimizer:** AdamW with Cosine Annealing scheduler (Warmup ratio 0.1).
* **Stabilization:** Used EMA (Exponential Moving Average) for stable weights and Label Smoothing (0.05) to prevent overconfidence on ambiguous debris boundaries.

## Usage

### 1. Environment Setup
The project runs inside a Docker container for full reproducibility.

```bash
# Pull the docker image
docker pull letatanu/semseg_2d:latest

# Start the container
docker run --rm -ti --gpus all -v $(pwd):/working letatanu/semseg_2d:latest bash
2. Training
To reproduce the Optimized (Highest Ceiling) training run with Copy-Paste and weighted loss:

Bash

./run_train_rescuenet_mask2former_optimized.sh
3. Visualization
To generate the 4-panel visualizations:

Bash

./viz/run_viz_rescuenet_mask2former.sh
### Attribution & Research Team
This project was developed as part of research work at Bina Labs at Lehigh University.

Principal Investigator: Dr. Maryam Rahnemoonfar

Primary Author: Nhut Le, PhD Candidate

Research Lead: William Tang

Modifications by William Tang:

Implementation of Mask2Former Swin-Large pipeline.

Development of Copy-Paste Augmentation Strategy for class imbalance.

Optimization of Loss Landscape (4.0x weighting) for critical infrastructure classes.

License
This code is released for academic and educational use. Please cite the original RescueNet Paper if you use this in your research.
