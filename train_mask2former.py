# train_floodnet_mask2former.py
# Mask2Former fine-tuning on FloodNet with in-loop mIoU eval & best-checkpoint selection.

import os, json, math
import torch

from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Trainer,
    set_seed,
)
from nh_datasets.loader import build_dataset_from_py
from utils import (
    setup_devices_autodetect,
    ddp_barrier_safe,
    safe_training_args,
    make_val_subset,
    choose_resume_checkpoint,
    discover_best_model_dir,
    compute_mIoU,
    _masks_to_semantic,
    _to_list_class_labels,
    _to_list_masks,
    parse_args
)
import time
import warnings
# ---- GLOBAL SILENCE MODE ----
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"  # if you're not using wandb
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hides TensorFlow/TF-like logs if present

# disable all PyTorch user warnings
torch.set_printoptions(profile="default")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch._C._log_api_usage_once = lambda *a, **k: None  # hides HF API usage banner

# -----------------------
# Custom Trainer
# -----------------------

class Mask2FormerTrainer(Trainer):
    def __init__(self, *args, num_classes: int, ignore_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_classes = int(num_classes)
        self._ignore_index = int(ignore_index)

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        # keep labels_semantic on the batch but never pass it to model.forward
        inputs.pop("labels_semantic", None)
        return inputs

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

        autocast_enabled = bool(self.args.fp16 or self.args.bf16)

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)      # [B,3,h_img,w_img]
            labels_sem   = batch["labels_semantic"].to(device, non_blocking=True)   # [B,H_lab,W_lab]
            B, H_lab, W_lab = labels_sem.shape

            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                outputs = model(pixel_values=pixel_values)

            # >>> resize predictions to **label** size <<<
            target_sizes = [(int(H_lab), int(W_lab))] * B
            pred_list = self.processing_class.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )  # list of length B, each [H_lab, W_lab] on CPU

            # stack preds to GPU and ensure shape match
            preds = torch.stack([p.to(device=device, non_blocking=True) for p in pred_list], dim=0)  # [B,H_lab,W_lab]
            if preds.shape[-2:] != (H_lab, W_lab):  # extra safety net
                preds = torch.nn.functional.interpolate(
                    preds.unsqueeze(1).float(), size=(H_lab, W_lab), mode="nearest"
                ).squeeze(1).long()

            valid = (labels_sem != self._ignore_index)
            if valid.any():
                g = labels_sem[valid]
                p = preds[valid]
                bins = g * K + p
                hist = torch.bincount(bins, minlength=K*K)
                cm += hist.view(K, K)

            n_seen += B

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
        acc_c = torch.where(denom_acc > 0, tp / denom_acc, torch.zeros_like(denom_acc))
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
            metrics[f"{metric_key_prefix}_IoU_{c}"] = 0.0 if (v != v) else float(v)

        self.log(metrics)
        return metrics
    
# ---------- Evaluation-only -----------------------
@torch.no_grad()
def evaluate_only(args, ddp_kwargs):
    ddp_barrier_safe()
    # pick a model dir to load
    model_dir = args.eval_from or discover_best_model_dir(args.output_dir)
    print(f"[evaluate] loading model from: {model_dir}")

    # load processor + model from the chosen dir
    image_processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)

    # dataset: val or test
    eval_ds_full = build_dataset_from_py(args.config_file, split=args.test_split, augment=False, image_processor=image_processor)
    
    eval_ds = make_val_subset(eval_ds_full, limit=args.eval_limit, fraction=None, seed=args.val_seed)
    _IGNORE = eval_ds_full.ignore_index
    def collate_fn(batch):
        """
        Works whether samples include 'labels_semantic' or not.
        Returns:
        pixel_values: [B,3,H,W] float
        class_labels: list[list[int]]
        mask_labels:  list[list[H,W] bool tensors]
        labels_semantic: [B,H,W] long
        """
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)

        class_labels = []
        mask_labels  = []
        labels_sem_list = []

        # pick ignore index (0 by default, or capture from args if you prefer)
        

        for b in batch:
            # standardize to lists
            cls_list = _to_list_class_labels(b.get("class_labels", None))
            m_list   = _to_list_masks(b.get("mask_labels", None))
            class_labels.append(cls_list)
            mask_labels.append(m_list)

            if "labels_semantic" in b and b["labels_semantic"] is not None:
                labels_sem_list.append(b["labels_semantic"].to(torch.long))
            else:
                if len(m_list) == 0:
                    raise ValueError("Sample missing both 'labels_semantic' and 'mask_labels'.")
                labels_sem_list.append(_masks_to_semantic(cls_list, m_list, ignore_index=_IGNORE))

        labels_sem = torch.stack(labels_sem_list, dim=0)

        return {
            "pixel_values": pixel_values,
            "class_labels": class_labels,
            "mask_labels":  mask_labels,
            "labels_semantic": labels_sem,
        }
    # lean TrainingArguments for eval only
    eval_args = safe_training_args(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        dataloader_drop_last=False,
        **ddp_kwargs
    )

    trainer = Mask2FormerTrainer(
        model=model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        processing_class=image_processor,   # required for post_process_semantic_segmentation
        num_classes=args.num_classes,
        ignore_index=_IGNORE,
    )
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))
# ----------------------- Training script -----------------------

def train(args, ddp_kwargs):
      # Datasets
    image_processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
    train_ds = build_dataset_from_py(args.config_file, split=args.train_split, image_processor=image_processor)
    val_ds = build_dataset_from_py(args.config_file, split=args.val_split, augment=False, image_processor=image_processor)
    _IGNORE = train_ds.ignore_index
    
    
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        id2label=train_ds.id2label,
        label2id=train_ds.label2id,
        ignore_mismatched_sizes=True,
    )

    # Collate: include semantic GT for metrics; model will not see it (Trainer subclass pops it)
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)        # [B,3,H,W]
        class_labels = [b["class_labels"] for b in batch]                             # list[Long[K]]
        mask_labels  = [b["mask_labels"]  for b in batch]                             # list[Float[K,H,W]]
        labels_sem   = torch.stack([b["labels_semantic"] for b in batch], dim=0)      # [B,H,W]
        return {
            "pixel_values": pixel_values,
            "class_labels": class_labels,
            "mask_labels":  mask_labels,
            "labels_semantic": labels_sem,   # kept for metrics via label_names
        }


    # Steps/sec/save cadence like yours
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * max(1, torch.cuda.device_count())))
    save_steps = max(steps_per_epoch, 100)

    # Training args: in-loop eval & best checkpoint on mIoU
    training_args = safe_training_args(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        
        include_inputs_for_metrics=False, 

        eval_strategy="epoch",  # evaluate every epoch
        save_strategy="epoch",
        eval_steps=save_steps,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",
        greater_is_better=True,

        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        fp16=args.fp16,
        fp16_full_eval=True,
        overwrite_output_dir=args.overwrite_output_dir,

        remove_unused_columns=True,
        label_names=["labels_semantic"],
        dataloader_num_workers=8,          
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        eval_accumulation_steps=8,  
        **ddp_kwargs

    )

    def metrics_fn(eval_pred):
        return compute_mIoU(eval_pred, num_classes=args.num_classes, ignore_index=0)
    # 2) Trainer instantiation
    trainer = Mask2FormerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        processing_class=image_processor,   # <<< needed for post_process_semantic_segmentation
        num_classes=args.num_classes,
        ignore_index=_IGNORE,
    )

    resume_from = choose_resume_checkpoint(args.resume, args.output_dir)
    print(f"[resume] Using checkpoint: {resume_from}")  # False => fresh start

    trainer.train(resume_from_checkpoint=resume_from)
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(args.output_dir)
    with open(os.path.join(args.output_dir, "id2label.json"), "w") as f:
        json.dump(train_ds.id2label, f, indent=2)
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(train_ds.label2id, f, indent=2)

    print("Training complete. Best checkpoint:", trainer.state.best_model_checkpoint)
    
    
# -----------------------
# Main
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
