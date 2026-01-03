import os, json, inspect, runpy, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from transformers import (
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to dataset config .py file.")
    parser.add_argument("--data_root", type=str, default="/data/FloodNet-Supervised_v1.0")
    parser.add_argument("--output_dir", type=str, default="/working/runs/mask2former_floodnet")
    parser.add_argument("--model_name", type=str, default="facebook/mask2former-swin-large-cityscapes-semantic")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=bool, default=False, help="use mixed precision")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--resume", type=str, default=None,
        help="Resume from a checkpoint: 'auto' to pick last in output_dir, a path, or 'none' to start fresh.")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False,
        help="Allow training to start even if output_dir is non-empty.")
    
    parser.add_argument("--val_fraction", type=float, default=0.25, help="Use this fraction (0-1) of val images.")
    parser.add_argument("--val_seed", type=int, default=1337, help="Deterministic subset seed.")
    parser.add_argument("--val_limit", type=int, default=200,
                    help="Evaluate on at most this many validation samples.")
    parser.add_argument("--evaluate", type=bool, default=False,
                        help="If set, only run evaluation using the model from --eval_from or best checkpoint in output_dir.")
    parser.add_argument("--eval_from", type=str, default=None,
                        help="Path to a model/checkpoint dir to evaluate. If omitted, the best checkpoint under --output_dir is used.")
    args = parser.parse_args()
    
    cfg = runpy.run_path(args.config_file)
    args.data_root = cfg["DATASET_KWARGS"].get("root", args.data_root)
    args.num_classes = cfg["DATASET_KWARGS"].get("num_classes", args.num_classes)
    args.image_size = cfg["DATASET_KWARGS"].get("image_size", args.image_size)
    args.batch_size = cfg.get("batch_size", args.batch_size)
    args.lr = cfg.get("lr", args.lr)
    args.weight_decay = cfg.get("weight_decay", args.weight_decay)
    args.epochs = cfg.get("num_epochs", args.epochs)
    args.fp16 = cfg.get("fp16", args.fp16)
    args.save_total_limit = cfg.get("save_total_limit", args.save_total_limit)
    args.warmup_ratio = cfg.get("warmup_ratio", args.warmup_ratio)
    args.logging_steps = cfg.get("logging_steps", args.logging_steps)
    args.gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", args.gradient_accumulation_steps) 
    args.output_dir = cfg.get("output_dir", args.output_dir)
    args.model_name = cfg.get("model_name", args.model_name)
    args.resume = cfg.get("resume", args.resume)
    args.overwrite_output_dir = cfg.get("overwrite_output_dir", args.overwrite_output_dir)
    args.eval_from = cfg.get("eval_from", args.eval_from)
    args.eval_limit = cfg.get("eval_limit", args.val_limit)
    args.evaluate = cfg.get("evaluate", args.evaluate)
    args.train_split = cfg.get("train_split", args.train_split)
    args.val_split = cfg.get("val_split", args.val_split)
    args.test_split = cfg.get("test_split", args.test_split)
    return args

def setup_devices_autodetect():
    """
    If launched via torchrun (WORLD_SIZE>1 and a rank set) -> DDP.
    Else -> single process, single GPU (scrub DDP env so Accelerate stays off).
    """
    gpu_count = torch.cuda.device_count()
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    has_rank = ("LOCAL_RANK" in os.environ) or ("RANK" in os.environ)
    ddp_env = env_world > 1 and has_rank

    if ddp_env:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        if torch.cuda.is_available() and gpu_count > 0:
            torch.cuda.set_device(local_rank % gpu_count)
        mode = "ddp"
        world_size = env_world
    else:
        for k in ("LOCAL_RANK","RANK","WORLD_SIZE","MASTER_ADDR","MASTER_PORT",
                  "SLURM_PROCID","SLURM_NTASKS","PMI_RANK","PMI_SIZE","ACCELERATE_USE_DISTRIBUTED"):
            os.environ.pop(k, None)
        os.environ["ACCELERATE_USE_DISTRIBUTED"] = "false"
        if torch.cuda.is_available() and gpu_count > 0:
            torch.cuda.set_device(0)
        mode = "single"
        local_rank = 0
        world_size = 1
    return mode, local_rank, world_size


def take_first_n(dataset, n: int):
    if n is None or n <= 0:
        return dataset
    from torch.utils.data import Subset
    return Subset(dataset, list(range(min(n, len(dataset)))))

def ddp_barrier_safe():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])  # PyTorch >=2.0
        except TypeError:
            dist.barrier()
def safe_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**filtered)


def make_val_subset(dataset, limit: int | None = None, fraction: float | None = None, seed: int = 42):
    n = len(dataset)
    if limit is None and (fraction is None or fraction <= 0 or fraction > 1):
        return dataset
    if fraction is not None and limit is None:
        limit = max(1, int(round(n * fraction)))
    # deterministic pick: shuffle indices with a fixed seed, then take the head
    g = np.random.default_rng(seed)
    indices = np.arange(n)
    g.shuffle(indices)
    indices = indices[:limit]
    indices.sort()  # keep loader cache-friendly ordering
    return Subset(dataset, indices.tolist())

def choose_resume_checkpoint(resume_arg: str, out_dir: str) -> str | bool:
    if resume_arg is None or str(resume_arg).lower() == "none":
        return False
    if resume_arg == "auto":
        last = get_last_checkpoint(out_dir)
        if last is not None:
            return last
        p = Path(out_dir)
        if not p.exists():
            return False
        cks = sorted(
            [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else -1
        )
        return str(cks[-1]) if cks else False
    p = Path(resume_arg)
    return str(p) if p.exists() else False

def discover_best_model_dir(base_dir: str) -> str:
    """
    Resolve a directory that can be passed to from_pretrained():
      1) If base_dir already contains weights, return it.
      2) If trainer_state.json has best_model_checkpoint, return that.
      3) Else fall back to the last checkpoint in base_dir.
      4) Else return base_dir (best effort).
    """
    p = Path(base_dir)

    # Direct weights?
    if (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists():
        return str(p)

    # trainer_state.json → best_model_checkpoint
    ts = p / "trainer_state.json"
    if ts.exists():
        try:
            state = json.loads(ts.read_text())
            best = state.get("best_model_checkpoint")
            if best and Path(best).exists():
                return best
        except Exception:
            pass

    # last checkpoint
    last = get_last_checkpoint(str(p))
    if last:
        return last

    return str(p)

def compute_mIoU(eval_preds, num_classes: int, ignore_index: int = 255):
    """
    eval_preds: (logits, labels)
      - logits: np.ndarray [N, C, h, w]  (often downsampled, e.g., 128x128)
      - labels: np.ndarray [N, H, W]     (your target size, e.g., 512x512)
    """

    logits, labels = eval_preds

    # to torch
    logits_t = torch.from_numpy(logits)       # [N,C,h,w]
    labels_t = torch.from_numpy(labels)       # [N,H,W]
    if labels_t.dtype != torch.long:
        labels_t = labels_t.long()

    # upsample logits to label size
    N, C, h, w = logits_t.shape
    H, W = labels_t.shape[-2], labels_t.shape[-1]
    if (h, w) != (H, W):
        logits_t = F.interpolate(logits_t, size=(H, W), mode="bilinear", align_corners=False)

    preds = logits_t.argmax(dim=1)            # [N,H,W]
    preds_np = preds.cpu().numpy()
    labels_np = labels_t.cpu().numpy()

    # confusion accumulators
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for p, g in zip(preds_np, labels_np):
        mask = (g != ignore_index)
        if not np.any(mask):
            continue
        p = p[mask]
        g = g[mask]
        for cls in range(num_classes):
            p_c = (p == cls)
            g_c = (g == cls)
            tp[cls] += np.logical_and(p_c, g_c).sum()
            fp[cls] += np.logical_and(p_c, ~g_c).sum()
            fn[cls] += np.logical_and(~p_c, g_c).sum()

    iou = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        denom = tp[c] + fp[c] + fn[c]
        iou[c] = float(tp[c]) / denom if denom > 0 else float("nan")

    miou = float(np.nanmean(iou))
    macc = float((tp / np.maximum(tp + fn, 1)).mean())

    metrics = {"mIoU": miou, "mAcc": macc}
    for c, v in enumerate(iou):
        metrics[f"IoU_{c}"] = 0.0 if np.isnan(v) else float(v)
    return metrics

def _to_list_class_labels(x):
    """Return a python list[int] from list/tuple/Tensor/None."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().to(torch.long).tolist()
    raise TypeError(f"Unsupported class_labels type: {type(x)}")

def _to_list_masks(x):
    """Return a list[Tensor(H,W,bool)] from list/tuple/Tensor/None."""
    if x is None:
        return []
    out = []
    if isinstance(x, (list, tuple)):
        it = x
    elif isinstance(x, torch.Tensor):
        # shapes: [N,H,W] or [H,W]
        if x.ndim == 2:
            it = [x]
        elif x.ndim == 3:
            it = [x[i] for i in range(x.shape[0])]
        else:
            raise ValueError(f"mask_labels tensor must be 2D/3D, got {x.shape}")
    else:
        raise TypeError(f"Unsupported mask_labels type: {type(x)}")

    for m in it:
        mt = torch.as_tensor(m)
        if mt.ndim == 3 and mt.shape[0] == 1:  # squeeze channel if [1,H,W]
            mt = mt[0]
        if mt.ndim != 2:
            raise ValueError(f"Each mask must be 2D, got {mt.shape}")
        out.append(mt.to(dtype=torch.bool, copy=False))
    return out

def _masks_to_semantic(class_labels_list, mask_list, ignore_index=0):
    """
    Build an [H,W] long tensor with class ids. Later masks override earlier ones.
    """
    if len(mask_list) == 0:
        raise ValueError("mask_labels is empty; cannot reconstruct semantic map")

    H, W = mask_list[0].shape[-2], mask_list[0].shape[-1]
    canvas = torch.full((H, W), int(ignore_index), dtype=torch.long)
    # last mask wins (stable & simple)
    for c, m in zip(class_labels_list, mask_list):
        canvas[m] = int(c)
    return canvas