import os
import random
import cv2
import numpy as np
import torch
import albumentations as A
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor
from .registry import register_dataset

@register_dataset("rescuenet_mask2former_optimized")
class RescueNetMask2FormerDatasetOptimized(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
    CLASSES = [
        "Background", "Water", "Building_No_Damage", "Building_Minor_Damage",
        "Building_Major_Damage", "Building_Total_Destruction", "Vehicle",
        "Road-Clear", "Road-Blocked", "Tree", "Pool",
    ]
    
    # Label Mappings
    label2id = {name: i for i, name in enumerate(CLASSES)}
    id2label = {i: name for i, name in enumerate(CLASSES)}
    
    # [STRATEGY] Copy-Paste Sources
    COPY_CLASSES = [6, 8] 

    def __init__(self, root: str, split: str, image_processor: AutoImageProcessor, num_classes: int = 11, image_size: int = 1024, augment: bool = False, ignore_index: int = 255):
        self.root = Path(root)
        self.split = split
        self.ip = image_processor
        self.num_classes = num_classes
        self.image_size = image_size
        self.augment = augment
        self.ignore_index = ignore_index

        self.img_dir = self.root / split / f"{split}-org-img"
        self.lbl_dir = self.root / split / f"{split}-label-img"
        
        self.samples: List[Tuple[Path, Path]] = []
        self.copy_candidates = [] 

        if not self.img_dir.is_dir(): 
            raise FileNotFoundError(f"Missing: {self.img_dir}")

        all_files = sorted(os.listdir(self.img_dir))
        for i, fname in enumerate(all_files):
            stem, ext = os.path.splitext(fname)
            if ext not in self.IMG_EXTS: continue
            
            img_p = self.img_dir / fname
            found_lbl = False
            for ext2 in self.IMG_EXTS:
                lbl_p = self.lbl_dir / f"{stem}_lab{ext2}"
                if lbl_p.exists():
                    self.samples.append((img_p, lbl_p))
                    if split == "train":
                        self.copy_candidates.append(len(self.samples)-1)
                    found_lbl = True
                    break
            if not found_lbl and split == "train":
                print(f"[Warning] No label for {fname}")
        
        # [STRATEGY] HIGHEST CEILING PIPELINE
        # We MUST use geometric augmentations because aerial imagery has no "correct" orientation.
        # We MUST use photometric augmentations to handle different times of day/lighting.
        if self.augment:
            self.transform = A.Compose([
                # Geometric (Crucial for Trees/Buildings)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Texture/Color (Crucial for Damage Detection)
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # Prevents overfitting to camera grain
                
                A.Resize(height=self.image_size, width=self.image_size, interpolation=cv2.INTER_LINEAR),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.image_size, width=self.image_size),
            ])

    def __len__(self):
        return len(self.samples)

    def load_sample(self, idx):
        img_path, lbl_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(lbl_path).convert("L"))
        return image, mask

    def copy_paste(self, image, mask):
        """
        [HIGHEST CEILING] Copy-Paste Augmentation
        Randomly pastes 'Road-Blocked' or 'Vehicle' pixels from another image.
        """
        if True or len(self.copy_candidates) < 1: 
            return image, mask
            
        src_idx = random.choice(self.copy_candidates)
        src_img, src_mask = self.load_sample(src_idx)
        
        if src_img.shape[:2] != image.shape[:2]:
            src_img = cv2.resize(src_img, (image.shape[1], image.shape[0]))
            src_mask = cv2.resize(src_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        target_pixels = np.isin(src_mask, self.COPY_CLASSES)
        
        if not np.any(target_pixels):
            return image, mask

        image[target_pixels] = src_img[target_pixels]
        mask[target_pixels] = src_mask[target_pixels]
        
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.load_sample(idx)
        
        # [STEP 1] Apply Copy-Paste
        if self.augment:
            image, mask = self.copy_paste(image, mask)

        # [STEP 2] Standard Augmentations
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        img_pil = Image.fromarray(image)
        mask = mask.astype(np.int64)

        encoded = self.ip(images=img_pil, do_resize=False, do_rescale=True, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)
        mask_tensor, class_tensor = self._classwise_masks(mask)

        return {
            "pixel_values": pixel_values,
            "class_labels": class_tensor,
            "mask_labels": mask_tensor,
            "labels_semantic": torch.from_numpy(mask),
            "id": self.samples[idx][0].stem
        }

    def _classwise_masks(self, lab_np):
        mask_list = []
        class_ids = []
        unique_classes = np.unique(lab_np)
        for c in unique_classes:
            if c == self.ignore_index: continue
            m = (lab_np == c).astype(np.float32)
            mask_list.append(torch.from_numpy(m))
            class_ids.append(int(c))
        
        if not class_ids:
            m = np.zeros_like(lab_np, dtype=np.float32)
            mask_list.append(torch.from_numpy(m))
            class_ids.append(0)

        return torch.stack(mask_list, dim=0), torch.tensor(class_ids, dtype=torch.long)