# RescueNet Semantic Segmentation (Highest Ceiling Strategy)

This repository contains a high-performance **Mask2Former** pipeline for semantic segmentation on the **RescueNet** dataset. The goal of this project is to automate post-disaster damage assessment by identifying granular features such as "Building-Total-Destruction," "Road-Blocked," and "Flood Water."

This release represents our **"Highest Ceiling" Optimization Strategy**, where we achieved a significant performance leap by combining high-resolution texture preservation with aggressive synthetic data generation.

## Baseline vs. Highest Ceiling Comparison

We significantly outperformed our initial baselines by shifting from a passive resizing strategy to an active "hard mining" approach. This phase introduces **Smart Copy-Paste Augmentation** and a **4.0x Weighted Loss Landscape** to resolve class imbalance.

### Global Metrics

| **Highest Ceiling** | **1024x1024 + Weighted Loss + Copy-Paste** | **71.91%** | **81.71%** |

*Key Insight: While the standard baseline (512px) struggled with rare classes, our **1024x1024 strategy** allowed the Mask2Former Swin-L backbone to resolve fine debris textures. Additionally, applying aggressive class weights (4.0x for Blocked Roads) unlocked detection for the most critical disaster features.*

---

### Detailed Performance Analysis
Comparing our **Original Baseline** against our final **Highest Ceiling** model reveals critical improvements in the "Safety Critical" classes, proving that the Copy-Paste strategy allows the model to see rare features.

| Class Category | Impact of Optimization |
|:---|:---|
| **Road-Blocked** | **Major Improvement.** The baseline frequently confused debris-covered roads with clear roads. Our **Copy-Paste Augmentation** synthetically generated thousands of new "Blocked Road" samples, and the **4.0x Class Weight** forced the model to prioritize them. |
| **Damage Levels** | **Texture Recovery.** The "Squash/Resize" method (512px) destroyed the texture of damaged roofs. The **1024px strategy** preserved these textures, allowing the model to distinguish "Minor" from "Major" damage based on subtle cracks. |
| **Small Objects** | **High Precision.** The Swin-Large backbone combined with high-resolution input ensured that small classes like **Pools** and **Vehicles** were not interpolated out of existence. |

**Analysis:**
* **Road-Blocked Sensitivity:** The baseline frequently confused debris-covered roads with clear roads. The **Copy-Paste Augmentation** synthetically generated thousands of new "Blocked Road" samples, and the **4.0x Class Weight** forced the model to prioritize them.
* **Texture Recovery:** The "Squash/Resize" method (Original) destroyed the texture of damaged roofs. The **1024px strategy** preserved these textures, allowing the model to distinguish "Minor" from "Major" damage based on subtle cracks.

---

## The Visualization Challenge & Solution
A major engineering challenge in this project was generating accurate visualizations for high-resolution satellite imagery (3000x4000px).

### The Problem
During initial inference, predictions appeared as **"chaotic blobs"** or low-confidence noise.
* **Root Cause:** The model was trained on **1024x1024 crops**. Standard inference scripts attempted to **squash** the massive full-resolution image into a single square. This destroyed the aspect ratio and scale, presenting the model with distorted features it had never seen during training.

### The Solution: "Training-Aligned Inference"
We engineered a custom visualization pipeline that strictly mirrors the training logic:
1.  **No Squashing:** We perform **Sliding Window Inference** with overlap to handle the full native resolution.
2.  **Palette Alignment:** We mapped the custom 11-class training indices to the correct visual palette, fixing discrepancies where roads appeared as "Debris."
3.  **Result:** Sharp, pixel-perfect segmentation maps that accurately reflect the model's high mIoU score.

![Main Result](demo_figures/main_result.png)
*(Left: Original, Middle: Ground Truth, Right: Model Prediction, Far Right: Overlay)*

---

## Methodology

This implementation builds upon the Hugging Face `transformers` library and utilizes:

* **Architecture:** `facebook/mask2former-swin-large-cityscapes-semantic` (Swin-Large Backbone).
* **Input Strategy (The Key Differentiator):**
    * *Original:* Resizing images (Destroys small debris details).
    * *Highest Ceiling:* **1024x1024 High-Res Input**. This forces the model to learn high-frequency textures, crucial for distinguishing "Rubble" from "Road."
* **Active Optimization Strategy:**
    * **Copy-Paste Augmentation:** Physically "pastes" debris onto clear roads to create synthetic "Road-Blocked" examples.
    * **Loss Reweighting:** Penalizes the model **4.0x** more for missing blocked roads and **3.0x** more for building damage.
* **Stabilization:** AdamW optimizer with Cosine Annealing scheduler (Warmup ratio 0.1) and Label Smoothing (0.05).

---

## Usage

### 1. Environment Setup
The project runs inside a Docker container for full reproducibility.

```bash
# Pull the docker image
docker pull letatanu/semseg_2d:latest

# Start the container
docker run --rm -ti --gpus all -v $(pwd):/working letatanu/semseg_2d:latest bash
```

### Training
To reproduce the Highest Ceiling training run (Weighted Loss + Copy-Paste):

```bash
./run_train_rescuenet_mask2former_optimized.sh
```

### Visualization
To generate the 4-panel visualizations seen above:

```bash
./viz/run_viz_rescuenet_mask2former.sh
```

### Attribution & Research Team
This project was developed as part of research work at Bina Labs at Lehigh University.

Principal Investigator: Dr. Maryam Rahnemoonfar

Primary Author: Nhut Le, PhD Candidate

Research Lead: William Tang

Modifications by William Tang:

Implementation of Mask2Former Swin-Large pipeline.

Development of Copy-Paste Augmentation Strategy to solve class imbalance.

Optimization of Loss Landscape (4.0x weighting) for critical infrastructure classes.

### License
This code is released for academic and educational use. Please cite the original RescueNet Paper if you use this in your research.
