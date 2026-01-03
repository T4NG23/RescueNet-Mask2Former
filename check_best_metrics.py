#!/usr/bin/env python3
"""
Script to find the best checkpoint based on mIoU and mAcc scores
from trainer_state.json files in checkpoint directories.
Includes detailed per-class IoU breakdown.
"""

import json
import os
import sys
from pathlib import Path

def find_best_metrics(output_dir):
    """Find the best checkpoint based on mIoU metric and show per-class details."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    checkpoints = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    
    if not checkpoints:
        print(f"No checkpoints found in {output_dir}")
        return
    
    best_miou = -1
    best_macc = -1
    best_checkpoint = None
    best_per_class_ious = {}
    
    all_metrics = []
    
    for checkpoint_dir in checkpoints:
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        
        if not trainer_state_path.exists():
            continue
        
        try:
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            
            # Extract metrics from log_history
            for log_entry in trainer_state.get("log_history", []):
                if "eval_mIoU" in log_entry:
                    miou = log_entry.get("eval_mIoU", -1)
                    macc = log_entry.get("eval_mAcc", -1)
                    step = log_entry.get("step", -1)
                    epoch = log_entry.get("epoch", -1)
                    
                    # [NEW] Extract all per-class IoUs dynamically
                    current_per_class = {k: v for k, v in log_entry.items() if k.startswith("eval_IoU_")}
                    
                    all_metrics.append({
                        "checkpoint": checkpoint_dir.name,
                        "step": step,
                        "epoch": epoch,
                        "mIoU": miou,
                        "mAcc": macc,
                    })
                    
                    if miou > best_miou:
                        best_miou = miou
                        best_macc = macc
                        best_checkpoint = checkpoint_dir.name
                        best_per_class_ious = current_per_class
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {trainer_state_path}: {e}")
            continue
    
    # Print results
    print("=" * 80)
    print("BEST CHECKPOINT FOUND")
    print("=" * 80)
    
    if best_checkpoint:
        print(f"\nCheckpoint: {best_checkpoint}")
        print(f"Best mIoU:  {best_miou:.4f} ({best_miou*100:.2f}%)")
        print(f"Best mAcc:  {best_macc:.4f} ({best_macc*100:.2f}%)")
        
        # [NEW] Print Per-Class Breakdown
        print("\nPer-Class IoU Breakdown:")
        print("-" * 40)
        
        # Sort keys by class ID (IoU_0, IoU_1, etc.)
        if best_per_class_ious:
            sorted_keys = sorted(best_per_class_ious.keys(), key=lambda x: int(x.split('_')[-1]))
            
            for key in sorted_keys:
                class_id = key.split('_')[-1]
                score = best_per_class_ious[key]
                print(f"  Class {class_id:>2}: {score:.4f} ({score*100:.2f}%)")
        else:
            print("  No per-class metrics found in logs.")
            
    else:
        print("No checkpoints with metrics found")
        return
    
    # Print all metrics sorted by mIoU
    print("\n" + "=" * 80)
    print("ALL CHECKPOINTS (sorted by mIoU)")
    print("=" * 80)
    
    all_metrics.sort(key=lambda x: x["mIoU"], reverse=True)
    
    print(f"\n{'Checkpoint':<20} {'Step':<10} {'Epoch':<8} {'mIoU':<10} {'mAcc':<10}")
    print("-" * 80)
    
    for metric in all_metrics:
        print(f"{metric['checkpoint']:<20} {int(metric['step']):<10} {metric['epoch']:<8.1f} {metric['mIoU']:<10.4f} {metric['mAcc']:<10.4f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        # Default fallback
        output_dir = "/media/volume/Tang_Volume/semseg_2d/runs/rescuenet_mask2former_optimized"
    
    find_best_metrics(output_dir)