import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from nh_datasets.loader import build_dataset_from_py
from torch.utils.data import DataLoader

def main():
    # 1. SETUP PATHS
    path_a = "/working/runs/rescuenet_mask2former_1024_weighted/checkpoint-19500"
    path_b = "/working/runs/rescuenet_mask2former_ceiling/checkpoint-134500"
    config_file = "/working/nh_datasets/configs/mask2former_rescuenet_optimized.py"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # 2. LOAD MODEL A (1024px Specialist)
    print(f"[Load] Model A (High Res): {path_a}")
    proc_a = AutoImageProcessor.from_pretrained(path_a)
    model_a = Mask2FormerForUniversalSegmentation.from_pretrained(path_a).to(device)
    model_a.eval()

    # 3. LOAD MODEL B (512px Generalist)
    print(f"[Load] Model B (Low Res): {path_b}")
    proc_b = AutoImageProcessor.from_pretrained(path_b)
    model_b = Mask2FormerForUniversalSegmentation.from_pretrained(path_b).to(device)
    model_b.eval()

    # 4. LOAD DATASET (Validation)
    print("[Load] Loading Dataset...")
    # Use Model A processor to enforce 1024x1024 loading
    val_ds = build_dataset_from_py(config_file, split="val", augment=False, image_processor=proc_a)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # 5. METRICS INIT
    num_classes = 11
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    
    print(f"[Run] Ensembling {len(loader)} images...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # A. Prepare Inputs
            inputs_a = batch["pixel_values"].to(device) # 1024x1024
            
            # Input B: Downsample to 512x512 for Model B
            inputs_b = F.interpolate(inputs_a, size=(512, 512), mode="bilinear", align_corners=False)

            # Ground Truth
            labels = batch["labels_semantic"].to(device)
            target_size = (labels.shape[1], labels.shape[2]) 

            # B. Inference Model A
            out_a = model_a(pixel_values=inputs_a)
            pred_map_a = proc_a.post_process_semantic_segmentation(out_a, target_sizes=[target_size])[0]
            
            # C. Inference Model B
            out_b = model_b(pixel_values=inputs_b)
            pred_map_b = proc_b.post_process_semantic_segmentation(out_b, target_sizes=[target_size])[0]

            # D. Ensemble Voting (Average Probabilities)
            probs_a = F.one_hot(pred_map_a.long(), num_classes=num_classes).float()
            probs_b = F.one_hot(pred_map_b.long(), num_classes=num_classes).float()
            
            ensemble_probs = (probs_a + probs_b) / 2.0
            pred_final = torch.argmax(ensemble_probs, dim=-1)

            # E. Update Metrics
            valid_mask = (labels != 255).squeeze(0)
            if valid_mask.any():
                g = labels.squeeze(0)[valid_mask]
                p = pred_final[valid_mask]
                bins = g * num_classes + p
                hist = torch.bincount(bins, minlength=num_classes**2)
                cm += hist.view(num_classes, num_classes)

    # 6. CALCULATE
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = tp + fp + fn
    ious = torch.where(denom > 0, tp / denom, torch.full_like(denom, float('nan')))
    miou = torch.nanmean(ious).item()
    macc = (tp.sum() / cm.sum()).item()

    print("\n" + "="*40)
    print("       FINAL ENSEMBLE SCORE       ")
    print("="*40)
    print(f"Ensemble mIoU:   {miou:.4f} ({miou*100:.2f}%)")
    print(f"Ensemble mAcc:   {macc:.4f} ({macc*100:.2f}%)")
    print("-" * 20)
    
    class_names = [
        "Background", "Water", "No_Damage", "Minor_Damage",
        "Major_Damage", "Total_Destruction", "Vehicle",
        "Road-Clear", "Road-Blocked", "Tree", "Pool"
    ]
    for i, name in enumerate(class_names):
        val = ious[i].item()
        print(f"  {name:<20}: {val:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
