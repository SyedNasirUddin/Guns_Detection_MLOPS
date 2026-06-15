import torch
from src.data_processing import GunDataset
from src.model_architecture import FasterRCNNModel

dataset = GunDataset("artifacts/raw")

image, target = dataset[0]

model = FasterRCNNModel(
    num_classes=2,
    device="cpu"
).model

model.train()

print("BEFORE FORWARD")

losses = model(
    [image],
    [target]
)


print("AFTER FORWARD")

print(losses)