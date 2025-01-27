import torch
import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
from torchgeo.datasets import VHR10
from torchgeo.trainers import InstanceSegmentationTask
import torch.nn.functional as F
from pycocotools import mask as coco_mask
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from matplotlib.patches import Rectangle
from torchvision.transforms.functional import to_pil_image

# Custom collate function for DataLoader (required for Mask R-CNN models)
def collate_fn(batch):
    max_height = max(sample['image'].shape[1] for sample in batch)
    max_width = max(sample['image'].shape[2] for sample in batch)

    images = torch.stack([
        F.pad(sample['image'], (0, max_width - sample['image'].shape[2], 0, max_height - sample['image'].shape[1]))
        for sample in batch
    ])

    targets = [
        {
            "labels": sample["labels"].to(torch.int64),
            "boxes": sample["boxes"].to(torch.float32),
            "masks": F.pad(
                sample["masks"],
                (0, max_width - sample["masks"].shape[2], 0, max_height - sample["masks"].shape[1]),
            ).to(torch.uint8),
        }
        for sample in batch
    ]

    return {"image": images, "target": targets}

# Visualization function
def visualize_predictions(image, predictions, targets):
    """Visualize model predictions and ground truth."""
    image = to_pil_image(image)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    
    # Plot predictions
    for box, label in zip(predictions['boxes'], predictions['labels']):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, str(label.item()), color='red', fontsize=12)
    
    # Plot ground truth
    for box, label in zip(targets['boxes'], targets['labels']):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, str(label.item()), color='blue', fontsize=12)
    
    plt.show()

# Initialize the VHR10 dataset
train_dataset = VHR10(root="data", split="positive", transforms=None, download=True)
val_dataset = VHR10(root="data", split="positive", transforms=None)

# Select a small subset of the dataset 
N = 100  # Number of samples to use
train_subset = Subset(train_dataset, list(range(N)))
val_subset = Subset(val_dataset, list(range(N)))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn, persistent_workers=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn, persistent_workers=True)

    print('\nDEBUG TRAIN LOADER\n')
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Target: {batch['target']}")
        break

    for batch in train_loader:
        print(batch)
        break

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )

    task = InstanceSegmentationTask(
        model="mask_rcnn",         
        backbone="resnet50",       
        weights=True,              
        num_classes=11,            
        lr=1e-3,                   
        freeze_backbone=False      
    )

    print('\nTRAIN THE MODEL\n')

    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print('\nEVALUATE THE MODEL\n')

    trainer.test(task, dataloaders=val_loader)

    print('\nINFERENCE AND VISUALIZATION\n')

    test_sample = train_dataset[0]
    test_image = test_sample["image"].unsqueeze(0)  # Add batch dimension
    predictions = task.predict_step({"image": test_image}, batch_idx=0)

    visualize_predictions(test_image, predictions[0], test_sample)



