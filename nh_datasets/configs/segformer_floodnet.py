data_root = "/data/FloodNet-Supervised_v1.0"
output_dir = "/working/runs/segformer_floodnet"
model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
epochs = 300
num_classes = 10
train_split = "train"
val_split = "val"
test_split = "test"
batch_size = 4
fp16=True
ignore_index = 0

DATASET_NAME = "floodnet_segformer"
DATASET_KWARGS = {
    "root": data_root,
    "image_size": 512,
    "augment": True,
    "ignore_index": ignore_index,
    "num_classes": num_classes
}