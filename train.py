import argparse
import copy
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.nn.modules import Module
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
from tqdm import tqdm

from torchgeo.datasets import TropicalCycloneWindEstimation

ROOT_DIR = os.path.expanduser("~/mount/data/")


def fit(
    model: Module,
    device: torch.device,  # type: ignore[name-defined]
    data_loader: DataLoader[Any],
    optimizer: Optimizer,
    criterion: Module,
    epoch: int,
    memo: str = "",
) -> np.number[Any]:
    model.train()

    losses = []
    tic = time.time()
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        images = batch["image"].to(device)
        targets = batch["wind_speed"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)

    elapsed_time = time.time() - tic
    print(
        f"[{memo}] Training Epoch: {epoch}\t Time elapsed: {elapsed_time:.2f} seconds"
        + "\t Loss: {avg_loss:.2f}"
    )

    return avg_loss


def evaluate(
    model: Module,
    device: torch.device,  # type: ignore[name-defined]
    data_loader: DataLoader[Any],
    criterion: Module,
    epoch: int,
    memo: str = "",
) -> Dict[str, np.number[Any]]:
    model.eval()

    losses = []

    batch_size = data_loader.batch_size
    assert batch_size is not None
    all_predictions = np.zeros((len(data_loader),), dtype=np.float32)
    all_targets = np.zeros((len(data_loader),), dtype=np.float32)

    tic = time.time()
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        images = batch["image"].to(device)
        targets = batch["wind_speed"].to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

            predictions = outputs.argmax(axis=1).cpu().numpy()

        start_index = batch_idx * batch_size
        end_index = (batch_idx * batch_size) + predictions.shape[0]
        all_predictions[start_index:end_index] = predictions
        all_targets[start_index:end_index] = batch["wind_speed"].numpy()

    avg_loss = np.mean(losses)
    val_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

    elapsed_time = time.time() - tic
    print(
        f"[{memo}] Validation Epoch: {epoch}\t Time elapsed: {elapsed_time:.2f} "
        + "seconds\t Loss: {avg_loss:.2f}"
    )

    return {"loss": avg_loss, "rmse": val_rmse}


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description="Training script for the NASA Cyclone dataset"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "resnet50"],
        help="Model to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use in training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use in the Dataloaders",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random number generator seed for numpy and torch",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of this experiment in TensorBoard",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Flag to enable verbose output",
    )

    return parser


def custom_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms a single sample from the Dataset"""
    sample["image"] = sample["image"] / 255.0  # scale to [0,1]
    sample["image"] = (
        sample["image"].unsqueeze(0).repeat(3, 1, 1)
    )  # convert to 3 channel
    sample["wind_speed"] = sample["wind_speed"] // 5

    return sample


def get_cyclone_datasets(
    root_dir: str, seed: int
) -> Tuple[
    Dataset[Any], Dataset[Any], Dataset[Any]
]:  # returns Dataset[Any]'s to account for the Subsets

    # Create datasets
    all_train_dataset = TropicalCycloneWindEstimation(
        root_dir, split="train", transforms=custom_transform, download=False
    )

    test_dataset = TropicalCycloneWindEstimation(
        root_dir, split="test", transforms=custom_transform, download=False
    )

    df = pd.read_csv("cyclone.csv")
    df = df[df["train"]]

    train_indices, val_indices = next(
        GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=seed).split(
            df, groups=df["storm_id"]
        )
    )

    train_dataset = Subset(all_train_dataset, train_indices)
    val_dataset = Subset(all_train_dataset, val_indices)

    return train_dataset, val_dataset, test_dataset


def main(args: argparse.Namespace) -> None:

    device = torch.device(  # type: ignore[attr-defined]
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ######################################
    # Setup datasets and dataloaders
    ######################################
    train_dataset, val_dataset, test_dataset = get_cyclone_datasets(ROOT_DIR, args.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    ######################################
    # Setup model, optimizer, losses, etc.
    ######################################
    if args.model == "resnet18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=186)
    elif args.model == "resnet50":
        model = torchvision.models.resnet50(pretrained=False, num_classes=186)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    criterion = nn.modules.loss.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=4)

    writer = SummaryWriter(  # type: ignore[no-untyped-call]
        log_dir=args.experiment_name
    )

    ######################################
    # Training loop
    ######################################
    training_losses = []
    validation_losses = []
    model_checkpoints = []

    for epoch in range(args.num_epochs):

        training_loss = fit(
            model,
            device,
            train_dataloader,
            optimizer,
            criterion,
            epoch,
        )

        validation_results = evaluate(model, device, val_dataloader, criterion, epoch)
        validation_loss = validation_results["loss"]
        validation_rmse = validation_results["rmse"]

        scheduler.step(validation_loss)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        # Update tensorboard
        writer.add_scalar(  # type: ignore[no-untyped-call]
            "Loss/train", training_loss, epoch
        )
        writer.add_scalar(  # type: ignore[no-untyped-call]
            "Loss/val", validation_loss, epoch
        )
        writer.add_scalar(  # type: ignore[no-untyped-call]
            "RMSE/val", validation_rmse, epoch
        )
        writer.flush()  # type: ignore[no-untyped-call]

        model_checkpoints.append(copy.deepcopy(model.state_dict()))
    writer.close()  # type: ignore[no-untyped-call]

    # Evaluate model on the test dataset
    test_results = evaluate(model, device, test_dataloader, criterion, 0)
    print(test_results)

    ######################################
    # TODO: Write out results, tear down
    ######################################
    pass


if __name__ == "__main__":
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    # Set random seed for reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Main training procedure
    main(args)
