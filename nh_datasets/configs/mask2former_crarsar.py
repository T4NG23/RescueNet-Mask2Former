data_root = "/data/CRASAR-tiles"
output_dir = "/working/runs/mask2former_crarsar"
model_name = "facebook/mask2former-swin-large-cityscapes-semantic"
epochs = 300
num_classes = 6
train_split = "train"
val_split = "test"
test_split = "test"
batch_size = 4
fp16=True
ignore_index = 0

DATASET_NAME = "crarsar_mask2former"
DATASET_KWARGS = {
    "root": data_root,
    "image_size": 512,
    "augment": True,
    "ignore_index": ignore_index,
    "num_classes": num_classes
}