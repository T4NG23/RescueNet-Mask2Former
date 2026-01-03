# train_mask2former_optimized.py
# Updated: 2025-12-20 - RescueNet Optimized (Flip-Only TTA + Logit Boosting)

import os, json, math, copy, time, warnings, gc, sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Trainer, set_seed
from torch.utils.data import DataLoader, SequentialSampler
from nh_datasets.loader import build_dataset_from_py
from utils import (setup_devices_autodetect, safe_training_args, choose_resume_checkpoint, parse_args)

# Prevent Thread Conflicts
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ---- SETUP ----
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DDP_FIND_UNUSED_PARAMETERS"] = "false" 
torch.backends.cudnn.benchmark = True 

# ---------------------------------------------------------
# CUSTOM TRAINER with OPTIMIZED TTA
# ---------------------------------------------------------
class Mask2FormerTrainerOptimized(Trainer):
    def __init__(self, *args, num_classes: int, ignore_index: int = 255, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_classes = int(num_classes)
        self._ignore_index = int(ignore_index)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if "labels_semantic" in inputs: inputs.pop("labels_semantic")
        
        outputs = model(
            pixel_values=inputs["pixel_values"],
            class_labels=inputs.get("class_labels"),
            mask_labels=inputs.get("mask_labels"),
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None: eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=SequentialSampler(eval_dataset),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    # Helper for TTA Voting
    def get_vote_tensor(self, outputs, target_size):
        # Convert model output to a One-Hot prediction map
        pred_map_list = self.processing_class.post_process_semantic_segmentation(
            outputs, target_sizes=[target_size]*outputs.class_queries_logits.shape[0]
        )
        pred_stack = torch.stack(pred_map_list).to(self.args.device) # (B, H, W)
        return F.one_hot(pred_stack, num_classes=self._num_classes).permute(0, 3, 1, 2).float() # (B, C, H, W)

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)

        model = self._wrap_model(self.model, training=False)
        model.eval()

        device = self.args.device
        K = self._num_classes
        
        cm = torch.zeros((K, K), dtype=torch.long, device=device)
        n_seen = 0
        start = time.time()

        # [STRATEGY] Logit Boosting Configuration
        # We boost damage classes to overcome the "smoothing" effect of TTA
        # Indices: 3:Minor, 4:Major, 5:Total, 8:Road-Blocked
        BOOST_INDICES = [3, 4, 5, 8]
        BOOST_VALUE = 1.0  # Conservative boost (Start at 1.0, increase if needed)

        print(f"\n[Eval] Starting TTA (2-Pass + LogitBoost) Evaluation on {len(dataloader)} batches...")

        for i, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels_sem = batch["labels_semantic"].to(device, non_blocking=True)
            B, H_lab, W_lab = labels_sem.shape

            with torch.amp.autocast("cuda", enabled=True):
                # --- PASS 1: Original ---
                out_orig = model(pixel_values=pixel_values)
                vote_accum = self.get_vote_tensor(out_orig, (H_lab, W_lab))

                # --- PASS 2: Horizontal Flip Only ---
                # (Removed Vertical Flip to preserve shadow/rubble texture consistency)
                out_flip = model(pixel_values=torch.flip(pixel_values, [3]))
                vote_flip = self.get_vote_tensor(out_flip, (H_lab, W_lab))
                vote_accum += torch.flip(vote_flip, [3]) 

                # --- LOGIT BOOSTING ---
                # Artificially inflate the scores for Damage classes
                # This breaks ties in favor of Damage over "No Damage"
                for idx in BOOST_INDICES:
                    vote_accum[:, idx, :, :] += BOOST_VALUE

                # Final Prediction via Majority Vote
                preds = torch.argmax(vote_accum, dim=1)

            valid = (labels_sem != self._ignore_index)
            if valid.any():
                g = labels_sem[valid]
                p = preds[valid]
                bins = g * K + p
                hist = torch.bincount(bins, minlength=K*K)
                cm += hist.view(K, K)

            n_seen += B
            if i % 10 == 0: gc.collect()

        if self.args.world_size > 1:
            torch.distributed.all_reduce(cm, op=torch.distributed.ReduceOp.SUM)

        cm = cm.to(torch.double).cpu()
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom, torch.full_like(denom, float('nan')))
        miou = torch.nanmean(iou).item()
        
        denom_acc = tp + fn
        acc_c = torch.where(denom_acc > 0, tp / denom_acc, torch.tensor(0.0, dtype=torch.double))
        macc = acc_c.mean().item()

        runtime = time.time() - start
        metrics = {
            f"{metric_key_prefix}_mIoU": float(miou),
            f"{metric_key_prefix}_mAcc": float(macc),
            f"{metric_key_prefix}_runtime": float(runtime),
            f"{metric_key_prefix}_samples_per_second": float(n_seen / max(runtime, 1e-6)),
        }
        
        for c in range(K):
            v = iou[c].item()
            metrics[f"{metric_key_prefix}_IoU_{c}"] = 0.0 if np.isnan(v) else float(v)

        self.log(metrics)
        gc.collect()
        torch.cuda.empty_cache()
        return metrics

# ---------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------
def train(args, ddp_kwargs):
    image_processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
    image_processor.do_resize = False 
    image_processor.do_rescale = True

    if ddp_kwargs.get("local_rank", 0) == 0: print("[Setup] Loading datasets...")
    
    train_ds = build_dataset_from_py(args.config_file, split=args.train_split, image_processor=image_processor)
    val_ds = build_dataset_from_py(args.config_file, split=args.val_split, augment=False, image_processor=image_processor)
    
    if ddp_kwargs.get("local_rank", 0) == 0: print("[Setup] Loading Model...")
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        id2label=train_ds.id2label,
        label2id=train_ds.label2id,
        ignore_mismatched_sizes=True,
    )
    
    if hasattr(model.model.pixel_level_module, "encoder"):
        model.model.pixel_level_module.encoder.gradient_checkpointing = True

    # ---------------------------------------------------------
    # [STRATEGY] AGGRESSIVE CLASS WEIGHTING (TRAINING)
    # ---------------------------------------------------------
    # 8:Road-Blocked (Critical, +4.0), 3,4,5:Damage (High, +3.0)
    # Weights: [Back, Water, NoDmg, Min, Maj, Tot, Veh, RdClr, RdBlk, Tree, Pool]
    weights_list = [1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 4.0, 1.0, 1.0]
    weights_list.append(0.1) # No Object
    
    class_weights_tensor = torch.tensor(weights_list, device=model.device, dtype=torch.float32)
    
    if hasattr(model, "criterion"):
        if ddp_kwargs.get("local_rank", 0) == 0:
            print(f"[Strategy] Injecting Aggressive Class Weights: {weights_list}")
        model.criterion.empty_weight = class_weights_tensor
    else:
        print(f"[WARNING] Could not inject weights. Check model structure.")

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        class_labels = [b["class_labels"] for b in batch]
        mask_labels  = [b["mask_labels"]  for b in batch]
        labels_sem   = torch.stack([b["labels_semantic"] for b in batch], dim=0)
        return {
            "pixel_values": pixel_values,
            "class_labels": class_labels,
            "mask_labels":  mask_labels,
            "labels_semantic": labels_sem, 
        }

    training_args = safe_training_args(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size, 
        learning_rate=args.lr,
        weight_decay=0.05,
        num_train_epochs=args.epochs,
        include_inputs_for_metrics=False, 
        gradient_checkpointing=False,
        fp16=True, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500, 
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",
        greater_is_better=True,
        dataloader_num_workers=4,           
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        eval_accumulation_steps=1,
        remove_unused_columns=False,
        **ddp_kwargs
    )
    
    trainer = Mask2FormerTrainerOptimized(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=image_processor,
        num_classes=args.num_classes,
        ignore_index=train_ds.ignore_index
    )
    
    if hasattr(model, "criterion"):
         model.criterion.empty_weight = model.criterion.empty_weight.to(trainer.args.device)

    trainer.train(resume_from_checkpoint=choose_resume_checkpoint(args.resume, args.output_dir))
    trainer.save_model(args.output_dir)

def main(): 
    args = parse_args()
    mode, local_rank, world_size = setup_devices_autodetect()
    ddp_kwargs = {}
    if world_size > 1: 
        ddp_kwargs.update(dict(ddp_find_unused_parameters=False, ddp_backend="nccl"))
    set_seed(args.seed)
    train(args, ddp_kwargs)

if __name__ == "__main__":
    main()