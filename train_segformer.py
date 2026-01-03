# train_floodnet_segformer.py
# Fine-tune SegFormer for FloodNet-style semantic segmentation
# Images:  data/FloodNet-Supervised_v1.0/<split>/<split>-org-img/*.jpg
# Labels:  data/FloodNet-Supervised_v1.0/<split>/<split>-label-img/*_lab.png

import os, json, math

import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    set_seed,
)
import numpy as np
from PIL import Image
from nh_datasets.loader import build_dataset_from_py
from utils import (
    setup_devices_autodetect,
    safe_training_args,
    compute_mIoU,
    choose_resume_checkpoint,
    parse_args,
    discover_best_model_dir,
    take_first_n,
    ddp_barrier_safe
)


# -----------------------
# 1) Evaluation only (test set)
# -----------------------
@torch.no_grad()
def evaluate_only(args, ddp_kwargs):
    IGNORE = args.ignore_index
    ddp_barrier_safe()
    # pick a model dir to load
    model_dir = args.eval_from or discover_best_model_dir(args.output_dir)
    print(f"[evaluate] loading model from: {model_dir}")
    try:
        image_processor = SegformerImageProcessor.from_pretrained(model_dir)
        # force our choices if missing in config
        image_processor.do_resize = False
        image_processor.do_normalize = True
    except Exception:
        image_processor = SegformerImageProcessor(do_resize=False, do_normalize=True, reduce_labels=False)
    eval_ds_full = build_dataset_from_py(args.config_file, split=args.test_split, augment=False, image_processor=image_processor)
    eval_ds = take_first_n(eval_ds_full, args.eval_limit)
   

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        num_labels=args.num_classes,
        id2label=eval_ds_full.id2label,
        label2id=eval_ds_full.label2id,
        ignore_mismatched_sizes=True,
    )

    # Collate identical to training
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}

    # Metrics wrapper with our ignore index
    def metrics_fn(eval_pred):
        return compute_mIoU(eval_pred, num_classes=args.num_classes, ignore_index=IGNORE)

    # Lean training args for evaluation
    training_args = safe_training_args(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        dataloader_drop_last=False,
        remove_unused_columns=False,  
        **ddp_kwargs,
    )

    # Trainer for eval only
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_ds,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    # Dump predictions as PNGs
    if args.save_preds_dir:
        os.makedirs(args.save_preds_dir, exist_ok=True)
        # Run a forward pass to get logits (HF evaluate already did; we’ll re-run to also get ids cleanly)
        from torch.utils.data import DataLoader
        dl = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with torch.no_grad():
            idx_base = 0
            for batch in dl:
                pv = batch["pixel_values"].to(device)
                logits = model(pixel_values=pv).logits  # [B,C,h,w]
                # upsample to label size
                H, W = batch["labels"].shape[-2], batch["labels"].shape[-1]
                if logits.shape[-2:] != (H, W):
                    logits = torch.nn.functional.interpolate(
                        logits, size=(H, W), mode="bilinear", align_corners=False
                    )
                preds = logits.argmax(1).cpu().numpy().astype(np.uint8)

                # Save as grayscale index PNGs
                for i in range(preds.shape[0]):
                    # get the original id if available (Subset wraps items as indices)
                    sample = eval_ds_full[idx_base + i] if isinstance(eval_ds, torch.utils.data.Subset) else eval_ds[idx_base + i]
                    im_id = sample["id"] if isinstance(sample, dict) else f"img_{idx_base+i:06d}"
                    Image.fromarray(preds[i], mode="L").save(os.path.join(args.save_preds_dir, f"{im_id}_pred.png"))
                idx_base += preds.shape[0]

        print(f"[evaluate] Saved predictions to: {args.save_preds_dir}")

# -----------------------
# 2) Training
# -----------------------
def train(args, ddp_kwargs):
     # Image processor & model
    image_processor = SegformerImageProcessor(
        do_resize=False,              
        do_normalize=True,
        reduce_labels=False,
    )
    train_ds = build_dataset_from_py(
        args.config_file,
        split=args.train_split,
        augment=True,
        image_processor=image_processor)
    val_ds = build_dataset_from_py(
        args.config_file,
        split=args.val_split,
        augment=False,
        image_processor=image_processor)
    
    id2label = train_ds.id2label
    label2id = train_ds.label2id
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,     
    )

    # Collate (pixel_values already tensor; labels LongTensor [H,W])
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}


    # Training arguments
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * max(1, torch.cuda.device_count())))
    save_steps = max(steps_per_epoch, 100)  # save once per epoch (or at least every 100 steps)

    training_args = safe_training_args(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",      
        save_strategy="epoch",
        save_steps=save_steps,         # e.g., ~once per epoch
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,   # based on metric_for_best_model
        metric_for_best_model="mIoU",
        greater_is_better=True,
        overwrite_output_dir=args.overwrite_output_dir,
        **ddp_kwargs,
    )


    def metrics_fn(eval_pred):
        return compute_mIoU(eval_pred, num_classes=args.num_classes, ignore_index=255)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )
    resume_from = choose_resume_checkpoint(args.resume, args.output_dir)
    print(f"[resume] Using checkpoint: {resume_from}")  # False means fresh start

    trainer.train(resume_from_checkpoint=resume_from)
    metrics = trainer.evaluate()
    print(metrics)

    # Save final model + label mappings
    trainer.save_model(args.output_dir)
    with open(os.path.join(args.output_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f, indent=2)
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    print("Training complete. Best checkpoint in:", trainer.state.best_model_checkpoint)

# -----------------------
# 3) Main
# -----------------------
def main():
    args = parse_args()
    mode, local_rank, world_size = setup_devices_autodetect()
    ddp_kwargs = {}
    if world_size > 1:
        ddp_kwargs.update(dict(
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
        ))
    set_seed(args.seed)
    
    # ---- EVAL-ONLY BRANCH -----------------------------------------------
    if args.evaluate:
        evaluate_only(args, ddp_kwargs)
        return
    # ---- END EVAL-ONLY BRANCH -------------------------------------------
    train(args, ddp_kwargs)

if __name__ == "__main__":
    main()
