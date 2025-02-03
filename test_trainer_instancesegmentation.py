import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import VHR10
from torchvision.transforms.functional import to_pil_image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchgeo.trainers import InstanceSegmentationTask  

def collate_fn(batch):
    """Custom collate function for DataLoader."""
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

def visualize_predictions(image, predictions, targets):
    """Visualize predictions and ground truth."""
    image = to_pil_image(image)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)

    # Predictions
    for box, label in zip(predictions['boxes'], predictions['labels']):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"Pred: {label.item()}", color='red', fontsize=12)

    # Ground truth
    for box, label in zip(targets['boxes'], targets['labels']):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"GT: {label.item()}", color='blue', fontsize=12)

    plt.show()

def plot_losses(train_losses, val_losses):
    """Plot training and validation losses over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

# Initialize VHR-10 dataset
train_dataset = VHR10(root="data", split="positive", transforms=None, download=True)
val_dataset = VHR10(root="data", split="positive", transforms=None)

# Subset for quick experimentation (adjust N as needed)
N = 100
train_subset = Subset(train_dataset, list(range(N)))
val_subset = Subset(val_dataset, list(range(N)))


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn)

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=5, 
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )

    task = InstanceSegmentationTask(
        model="mask_rcnn",          
        backbone="resnet50",        
        weights="imagenet",         # Pretrained on ImageNet
        num_classes=11,             # VHR-10 has 10 classes + 1 background
        lr=1e-3,                    
        freeze_backbone=False       
    )

    print('\nSTART TRAINING\n')
    # trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_losses, val_losses = [], []
    for epoch in range(5):
        trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)
        train_loss = task.trainer.callback_metrics.get("train_loss")
        val_loss = task.trainer.callback_metrics.get("val_loss")
        if train_loss is not None:
            train_losses.append(train_loss.item())
        if val_loss is not None:
            val_losses.append(val_loss.item())
    
    plot_losses(train_losses, val_losses)

    #trainer.test(task, dataloaders=val_loader)

    # Inference and Visualization
    sample = train_dataset[1]
    image = sample['image'].unsqueeze(0)  
    predictions = task.predict_step({"image": image}, batch_idx=0)
    visualize_predictions(image[0], predictions[0], sample)
