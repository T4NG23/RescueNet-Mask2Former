# 1. Set the correct data path (inside the docker, /data/ maps to your volume)
data_root = "/data/RescueNet"

# 2. Set the new output directory (Must match your Mask2Former script name)
output_dir = "/working/runs/rescuenet_mask2former"

# 3. Set the number of classes to 11
num_classes = 11

# 4. Set the ignore_index (255 is a safe value)
ignore_index = 255

# 5. Set the new dataset name (Must match the name used in loader.py)
DATASET_NAME = "rescuenet_mask2former"

# --- Other settings ---
# Using the confirmed public Cityscapes checkpoint
model_name = "facebook/mask2former-swin-large-cityscapes-semantic" 
epochs = 300
train_split = "train"
val_split = "val"
test_split = "test"
batch_size = 4
fp16=True

DATASET_KWARGS = {
    "root": data_root,
    "image_size": 512,
    "augment": True,
    "ignore_index": ignore_index,
    "num_classes": num_classes
}
