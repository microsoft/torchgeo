import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchgeo.datasets import VHR10
from torchgeo.trainers import InstanceSegmentationTask


# Custom collate function for DataLoader (required for Mask R-CNN models)
def collate_fn(batch):
    return tuple(zip(*batch))

# Initialize the VHR10 dataset
train_dataset = VHR10(root="data", split="positive", transforms=None, download=True)
val_dataset = VHR10(root="data", split="positive", transforms=None)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Initialize the InstanceSegmentationTask
task = InstanceSegmentationTask(
    model="mask_rcnn",         # Use Mask R-CNN as the model
    backbone="resnet50",       # ResNet-50 as the backbone
    weights=True,              # Use pretrained weights 
    num_classes=11,            # 10 object classes in VHR10 + 1 background class
    lr=1e-3,                   # Learning rate
    freeze_backbone=False      # Allow training the backbone
)

# Set up PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

# Train the model
trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Evaluate the model
trainer.test(task, dataloaders=val_loader)

# Example inference
test_sample = train_dataset[0]
test_image = test_sample["image"].unsqueeze(0)  # Add batch dimension
predictions = task.predict_step({"image": test_image}, batch_idx=0)
print(predictions)
