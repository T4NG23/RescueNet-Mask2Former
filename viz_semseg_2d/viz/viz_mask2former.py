#!/usr/bin/env python3
"""
Visualize Mask2Former semantic segmentation (RescueNet palette).
"""

import argparse, os, glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from contextlib import nullcontext
import matplotlib.patches as mpatches

# ---------- RescueNet Labels & Palette ----------
CLASSES = [
    "Background",
    "Water",
    "Building No Damage",
    "Building Minor Damage",
    "Building Major Damage",
    "Building Total Destruction",
    "Vehicle",
    "Road Clear",
    "Road Blocked",
    "Tree",
    "Pool",
]

# Standard RescueNet Color Palette (R, G, B)
PALETTE = np.array([
    [0, 0, 0],       # 0: Background (Black)
    [0, 0, 255],     # 1: Water (Blue)
    [0, 255, 0],     # 2: No Damage (Green)
    [255, 255, 0],   # 3: Minor Damage (Yellow)
    [255, 128, 0],   # 4: Major Damage (Orange)
    [255, 0, 0],     # 5: Total Destruction (Red)
    [255, 0, 255],   # 6: Vehicle (Purple)
    [128, 128, 128], # 7: Road Clear (Gray)
    [50, 50, 50],    # 8: Road Blocked (Dark Gray)
    [0, 128, 0],     # 9: Tree (Dark Green)
    [0, 255, 255]    # 10: Pool (Cyan)
], dtype=np.uint8)

# ---------- small viz helpers ----------
def draw_palette_legend(palette: np.ndarray, class_names: list[str]):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 1)) # Wider for long names
    ax.axis("off")
    handles = [mpatches.Patch(color=np.array(c)/255.0, label=cls)
               for c, cls in zip(palette, class_names)]
    ax.legend(handles=handles, loc="center", ncol=min(n, 6), fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, 0.5))
    return fig

def colorize(mask: np.ndarray, palette: np.ndarray = PALETTE) -> Image.Image:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Ensure indices are within palette range
    valid = (mask >= 0) & (mask < len(palette))
    rgb[valid] = palette[mask[valid]]
    return Image.fromarray(rgb, mode="RGB")

def overlay_image(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> Image.Image:
    img = img_rgb.astype(np.float32) / 255.0
    msk = mask_rgb.astype(np.float32) / 255.0
    out = (1 - alpha) * img + alpha * msk
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), mode="RGB")

def save_panel(img_pil: Image.Image, pred_arr: np.ndarray,
               out_path: Path, gt_arr: np.ndarray | None = None):
    pred_rgb = np.array(colorize(pred_arr))
    overlay = np.array(overlay_image(np.array(img_pil), pred_rgb, alpha=0.5))
    
    # If GT provided, colorize it too
    if gt_arr is not None:
        gt_rgb = np.array(colorize(gt_arr))
        
    h, w = pred_rgb.shape[:2]
    # Determine columns: Image | GT (Optional) | Pred | Overlay
    ncols = 4 if gt_arr is not None else 3
    panel = np.zeros((h, w * ncols, 3), dtype=np.uint8)
    
    # Resize raw image to match prediction dimensions if needed
    img_resized = np.array(img_pil.resize((w, h)))
    
    # Stitch columns
    col = 0
    panel[:, col*w:(col+1)*w, :] = img_resized; col += 1
    
    if gt_arr is not None:
        panel[:, col*w:(col+1)*w, :] = gt_rgb; col += 1
        
    panel[:, col*w:(col+1)*w, :] = pred_rgb; col += 1
    panel[:, col*w:(col+1)*w, :] = overlay
    
    Image.fromarray(panel).save(out_path)

def load_gt_mask(img_path: Path, gt_path: str = None,
                 gt_folder: str = None, gt_suffix: str = "_lab.png") -> np.ndarray | None:
    if gt_path:
        p = Path(gt_path)
    elif gt_folder:
        # Handle both jpg and png variations just in case
        p = Path(gt_folder) / (img_path.stem + gt_suffix)
    else:
        return None
        
    if not p.exists():
        # Fallback check for alternate extensions if not found
        if gt_suffix == "_lab.png":
            p_alt = Path(gt_folder) / (img_path.stem + "_lab.jpg")
            if p_alt.exists(): p = p_alt
            
    if not p.exists():
        return None
        
    return np.array(Image.open(p).convert("L"), dtype=np.int64)

# ---------- Mask2Former prediction ----------
@torch.inference_mode()
def predict_mask(model, processor, image_pil: Image.Image, long_side=1024, use_bf16=False, device="cuda"):
    ow, oh = image_pil.size
    # Resize for inference if image is huge
    if max(oh, ow) > long_side:
        scale = long_side / max(oh, ow)
        nw, nh = int(ow * scale), int(oh * scale)
        image_rs = image_pil.resize((nw, nh), Image.BILINEAR)
    else:
        image_rs = image_pil

    enc = processor(images=image_rs, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(device)

    # Use autocast if requested
    dtype = torch.bfloat16 if use_bf16 and device.startswith("cuda") else torch.float32
    # Fix: Mask2Former sometimes struggles with pure bf16 on older GPUs, float16 is safer usually
    # using standard autocast context
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if device.startswith("cuda") else nullcontext()
    
    with ctx:
        outputs = model(pixel_values=pixel_values)

    target_sizes = [(image_rs.size[1], image_rs.size[0])]
    pred_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    pred_small = np.array(pred_list[0].cpu(), dtype=np.int64)

    # Resize prediction back to original image size using Nearest Neighbor (to keep integer classes)
    if image_rs.size != image_pil.size:
        pred = np.array(Image.fromarray(pred_small.astype(np.uint8), mode="L")
                        .resize(image_pil.size, Image.NEAREST), dtype=np.int64)
    else:
        pred = pred_small
    return pred

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", type=str)
    ap.add_argument("--folder", type=str)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--gt_folder", type=str, help="Folder containing ground truth masks")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    print(f"[Info] Loading model from {args.model}...")
    
    # We load with ignore_mismatched_sizes=True just in case, though for viz it should be exact
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(args.device).eval()

    # Paths setup
    paths = []
    if args.image: paths.append(Path(args.image))
    if args.folder: paths.extend(sorted(Path(args.folder).glob("*.jpg")) + sorted(Path(args.folder).glob("*.png")))
    
    if not paths:
        print("No images found. Check paths.")
        return

    outdir = Path(args.outdir) if args.outdir else Path("viz_output")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save Legend
    legend_path = outdir / "legend.png"
    fig = draw_palette_legend(PALETTE, CLASSES)
    fig.savefig(legend_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Info] Legend saved to {legend_path}")

    print(f"[Info] Processing {len(paths)} images...")
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            
            # Try to find GT
            gt_arr = load_gt_mask(p, gt_folder=args.gt_folder)
            
            # Predict
            pred = predict_mask(model, processor, img, device=args.device)
            
            # Save
            out_path = outdir / (p.stem + "_viz.png")
            save_panel(img, pred, out_path, gt_arr)
            print(f"Saved: {out_path.name}")
            
        except Exception as e:
            print(f"Error processing {p.name}: {e}")

if __name__ == "__main__":
    main()