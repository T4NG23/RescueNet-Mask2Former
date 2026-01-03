data_root = "/data/CRASAR-tiles"
output_dir = "/working/runs/segformer_crarsar"
model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
epochs = 300
num_classes = 6
train_split = "train"
val_split = "test"
test_split = "test"
batch_size = 4
fp16=True
ignore_index = 0

DATASET_NAME = "crarsar_segformer"
DATASET_KWARGS = {
    "root": data_root,
    "image_size": 512,
    "augment": True,
    "ignore_index": ignore_index,
    "num_classes": num_classes
}