import nh_datasets.rescuenet_optimized

# -----------------------
# Configuration Variables
# -----------------------

data_root = "/data/RescueNet"
output_dir = "/working/runs/rescuenet_mask2former_ceiling"

num_classes = 11
ignore_index = 255
DATASET_NAME = "rescuenet_mask2former_optimized"
model_name = "facebook/mask2former-swin-large-cityscapes-semantic"

epochs = 300
batch_size = 1  # Fits 1024x1024 on L40S
gradient_accumulation_steps = 8

lr = 6e-5 
weight_decay = 0.05 

train_split = "train"
val_split = "val"
test_split = "test"

fp16 = True 

save_total_limit = 3
warmup_ratio = 0.05
logging_steps = 25

DATASET_KWARGS = {
    "root": data_root,
    "image_size": 512, # High Resolution
    "augment": True,
    "ignore_index": ignore_index,
    "num_classes": num_classes
}