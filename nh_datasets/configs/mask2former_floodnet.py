data_root = "/data/FloodNet-Supervised_v1.0"
output_dir = "/working/runs/mask2former_floodnet"
model_name = "facebook/mask2former-swin-large-cityscapes-semantic"
epochs = 300
train_split = "train"
val_split = "val"
test_split = "test"
batch_size = 4
fp16=True
ignore_index = 0
num_classes = 10

DATASET_NAME = "floodnet_mask2former"
DATASET_KWARGS = {
    "root": data_root,
    "image_size": 512,
    "augment": True,
    "ignore_index": ignore_index,
    "num_classes": num_classes
}