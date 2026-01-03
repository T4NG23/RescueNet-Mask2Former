#!/usr/bin/env python3
"""
Visualize SegFormer semantic segmentation (FloodNet palette).
- Works with fine-tuned checkpoints saved by HF Trainer (output_dir)
  or directly with a pretrained model id (e.g. nvidia/segformer-b2-...).

Usage examples:
  # visualize one image, show window only
  python viz_segformer.py --model runs/segformer_floodnet_b2_512 --image /path/to/img.jpg

  # visualize all JPGs in a folder and save results
  python viz_segformer.py --model runs/segformer_floodnet_b2_512 \
      --folder /data/FloodNet-Supervised_v1.0/val/val-org-img \
      --outdir viz_out --ext .jpg
"""

import argparse, os, glob
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from contextlib import nullcontext


# ---------- FloodNet labels & palette (edit if yours differ) ----------
CLASSES = [
    "background",
    "building flooded",
    "building non-flooded",
    "road flooded",
    "road non-flooded",
    "water",
    "tree",
    "vehicle",
    "pool",
    "grass",
]

PALETTE = np.array([
    [  0,   0,   0],  # 0 background
    [  255,   0,    0],  # 1 building_flooded
    [  180, 120, 120],  # 2 building_nonflooded
    [  160, 150, 20],  # 3 road_flooded
    [140, 140, 140],  # 4 road_nonflooded
    [61, 230, 250],  # 5 water
    [0, 82, 255],  # 6 tree
    [255, 0, 245],  # 7 vehicle
    [255, 235, 0],  # 8 pool
    [4, 250, 7],  # 9 grass
], dtype=np.uint8)

import matplotlib.patches as mpatches

def draw_palette_legend(palette: np.ndarray, class_names: list[str]):
    """Return a matplotlib Figure containing the color legend."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 1))
    ax.axis("off")

    handles = [
        mpatches.Patch(color=np.array(c)/255.0, label=cls)
        for c, cls in zip(palette, class_names)
    ]
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=min(n, 6),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.5)
    )
    return fig


def colorize(mask: np.ndarray, palette: np.ndarray = PALETTE) -> Image.Image:
    """mask: HxW integer class IDs. returns RGB PIL image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    rgb[valid] = palette[mask[valid]]
    return Image.fromarray(rgb, mode="RGB")


def overlay_image(img_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Alpha blend image (HxWx3 uint8) with mask colors."""
    img = img_rgb.astype(np.float32) / 255.0
    msk = mask_rgb.astype(np.float32) / 255.0
    out = (1 - alpha) * img + alpha * msk
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


@torch.inference_mode()
def predict_mask(model, processor, image_pil, long_side=1024, use_bf16=False):

    ow, oh = image_pil.size
    # resize keep aspect
    if max(oh, ow) > long_side:
        if oh >= ow:
            nh, nw = long_side, int(round(ow * long_side / oh))
        else:
            nw, nh = long_side, int(round(oh * long_side / ow))
        image_rs = image_pil.resize((nw, nh), Image.BILINEAR)
    else:
        image_rs = image_pil

    enc = processor(images=image_rs, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(model.device)

    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and model.device.type=="cuda") else nullcontext()
    with amp:
        out = model(pixel_values=pixel_values)
        logits = out.logits  # [1,C,h,w]

    # upsample to resized image size
    nh, nw = image_rs.size[1], image_rs.size[0]
    logits = F.interpolate(logits, size=(nh, nw), mode="bilinear", align_corners=False)
    pred_small = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)  # [nh, nw], int64

    # if we resized, scale pred back to original size using NEAREST
    if (ow, oh) != (nw, nh):
        # cast to uint8 for PIL, then back to int64 after resize
        pred_uint8 = pred_small.astype(np.uint8)
        pred_resized = Image.fromarray(pred_uint8, mode='L').resize((ow, oh), Image.NEAREST)
        pred = np.array(pred_resized, dtype=np.int64)
    else:
        pred = pred_small

    return pred



def show_panel(img_pil: Image.Image, mask_arr: np.ndarray, title: str = ""):
    mask_rgb = np.array(colorize(mask_arr))
    overlay = np.array(overlay_image(np.array(img_pil), mask_rgb, alpha=0.5))

    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 3, 1); ax1.imshow(img_pil); ax1.set_title("image"); ax1.axis("off")
    ax2 = plt.subplot(1, 3, 2); ax2.imshow(mask_rgb); ax2.set_title("mask"); ax2.axis("off")
    ax3 = plt.subplot(1, 3, 3); ax3.imshow(overlay); ax3.set_title("overlay"); ax3.axis("off")
    if title: fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_panel(img_pil: Image.Image, pred_arr: np.ndarray,
               out_path: Path, gt_arr: np.ndarray | None = None):
    pred_rgb = np.array(colorize(pred_arr))
    overlay = np.array(overlay_image(np.array(img_pil), pred_rgb, alpha=0.5))
    gt_rgb = np.array(colorize(gt_arr)) if gt_arr is not None else np.zeros_like(pred_rgb)

    h, w = pred_rgb.shape[:2]
    ncols = 4 if gt_arr is not None else 3
    panel = np.zeros((h, w * ncols, 3), dtype=np.uint8)
    panel[:, 0:w, :] = np.array(img_pil.resize((w, h)))
    col = 1
    if gt_arr is not None:
        panel[:, col*w:(col+1)*w, :] = gt_rgb
        col += 1
    panel[:, col*w:(col+1)*w, :] = pred_rgb; col += 1
    panel[:, col*w:(col+1)*w, :] = overlay
    Image.fromarray(panel).save(out_path)

def load_gt_mask(img_path: Path, gt_path: str = None,
                 gt_folder: str = None, gt_suffix: str = "_lab.png") -> np.ndarray | None:
    """Try to find or load a ground-truth mask; returns HxW np.int64 or None."""
    if gt_path:
        p = Path(gt_path)
    elif gt_folder:
        p = Path(gt_folder) / (img_path.stem + gt_suffix)
    else:
        return None
    if not p.exists():
        print(f"[warn] no GT mask found for {img_path.name}")
        return None
    gt = np.array(Image.open(p).convert("L"), dtype=np.int64)
    return gt



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_side", type=int, default=1024,
                    help="Resize so max(H,W)=long_side before inference (reduces VRAM).")
    ap.add_argument("--bf16", action="store_true",
                    help="Use bfloat16 for inference (great on A100).")
    ap.add_argument("--model", required=True,
                    help="Path to fine-tuned checkpoint dir OR HF model id (e.g. runs/... or nvidia/segformer-b2...)")
    ap.add_argument("--image", type=str, help="Single image path")
    ap.add_argument("--folder", type=str, help="Folder with images to visualize")
    ap.add_argument("--ext", type=str, default=".jpg", help="Extension to scan in --folder (e.g., .jpg or .png)")
    ap.add_argument("--outdir", type=str, default="/working", help="If set, saves panels to this folder")
    ap.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    ap.add_argument("--no_show", action="store_true", help="Skip on-screen display (useful when batch saving)")
    ap.add_argument("--gt", type=str, help="Path to ground-truth mask (.png) for this image")
    ap.add_argument("--gt_folder", type=str, help="If set, look for GT masks here matching filename + suffix")
    ap.add_argument("--gt_suffix", type=str, default="_lab.png",
                help="Suffix appended to image stem for GT lookup (used with --gt_folder)")
    args = ap.parse_args()

    # label mapping for the model head (important if you changed class order)
    id2label = {i: name for i, name in enumerate(CLASSES)}
    label2id = {v: k for k, v in id2label.items()}

    # load model + processor
    print(f"[load] {args.model}")
    processor = SegformerImageProcessor(do_resize=False, do_normalize=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=len(CLASSES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # ok when loading ADE-pretrained heads
    ).to(args.device).eval()

    paths = []
    if args.image:
        paths.append(Path(args.image))
    if args.folder:
        paths.extend(sorted(Path(args.folder).glob(f"*{args.ext}")))
    if not paths:
        raise SystemExit("Provide --image or --folder")

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        legend_path = outdir / "color_legend.png"
        fig = draw_palette_legend(PALETTE, CLASSES)
        fig.savefig(legend_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[save] {legend_path}")

    for p in paths:
        img = Image.open(p).convert("RGB")
        gt_arr = load_gt_mask(p, args.gt, args.gt_folder, args.gt_suffix)

        pred = predict_mask(model, processor, img,
                    long_side=args.long_side, use_bf16=args.bf16)

        title = p.name
        if outdir:
            out_path = outdir / (p.stem + "_panel.png")
            save_panel(img, pred, out_path, gt_arr)
            print(f"[save] {out_path}")

        if not args.no_show:
            show_panel(img, pred, title)


if __name__ == "__main__":
    main()
