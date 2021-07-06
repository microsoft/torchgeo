import argparse
import copy
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.nn.modules import Module
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
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
) -> float:
    model.train()

    losses = []
    tic = time.time()
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        images = batch["image"].to(device)
        targets = batch["wind_speed"].float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # TODO: what is the correct way to return a numpy function result without
    # making it a float?
    avg_loss = float(np.mean(losses))

    elapsed_time = time.time() - tic
    print(
        f"[{memo}] Training Epoch: {epoch}\t Time elapsed: {elapsed_time:.2f} seconds"
        + f"\t Loss: {avg_loss:.2f}"
    )

    return avg_loss


def evaluate(
    model: Module,
    device: torch.device,  # type: ignore[name-defined]
    data_loader: DataLoader[Dict[str, Any]],
    criterion: Module,
    epoch: int,
    memo: str = "",
) -> Dict[str, float]:
    model.eval()

    losses = []

    batch_size = data_loader.batch_size
    # TODO: not sure how to convince mypy that data_loader.dataset will have __len__
    dataset_size = len(data_loader.dataset)  # type: ignore[arg-type]
    assert batch_size is not None
    assert dataset_size is not None
    all_predictions = np.zeros((dataset_size,), dtype=np.float32)
    all_targets = np.zeros((dataset_size,), dtype=np.float32)

    tic = time.time()
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        images = batch["image"].to(device)
        targets = batch["wind_speed"].float().to(device)

        with torch.no_grad():
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            # predictions = outputs.argmax(axis=1).cpu().numpy()
            predictions = outputs.cpu().numpy()

        start_index = batch_idx * batch_size
        end_index = (batch_idx * batch_size) + batch_size
        all_predictions[start_index:end_index] = predictions
        all_targets[start_index:end_index] = batch["wind_speed"].numpy()

    # TODO: what is the correct way to return a numpy function result without
    # making it a float?
    avg_loss = float(np.mean(losses))
    val_rmse = float(np.sqrt(mean_squared_error(all_targets, all_predictions)))

    elapsed_time = time.time() - tic
    print(
        f"[{memo}] Validation Epoch: {epoch}\t Time elapsed: {elapsed_time:.2f} "
        + f"seconds\t Loss: {avg_loss:.2f}"
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
        "--loss",
        default="ce",
        choices=["ce", "mse"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use in training",
    )
    parser.add_argument(
        "--initial_lr",
        type=float,
        default=0.01,
        help="Initial learning rate",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store output files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Flag to enable overwriting existing output",
    )

    return parser


def custom_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms a single sample from the Dataset"""
    sample["image"] = sample["image"] / 255.0  # scale to [0,1]
    sample["image"] = (
        sample["image"].unsqueeze(0).repeat(3, 1, 1)
    )  # convert to 3 channel
    sample["wind_speed"] = sample["wind_speed"]

    return sample


def get_cyclone_datasets(
    root_dir: str, seed: int
) -> Tuple[Subset[Dict[str, Any]], Subset[Dict[str, Any]], Subset[Dict[str, Any]]]:

    # Create datasets
    all_train_dataset = TropicalCycloneWindEstimation(
        root_dir, split="train", transforms=custom_transform, download=False
    )

    all_test_dataset = TropicalCycloneWindEstimation(
        root_dir, split="test", transforms=custom_transform, download=False
    )

    # Extract the `storm_id`s from each sample in the training dataset
    storm_ids = []
    for item in all_train_dataset.collection:
        storm_id = item["href"].split("/")[0].split("_")[-2]
        storm_ids.append(storm_id)

    train_indices, val_indices = next(
        GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=seed).split(
            storm_ids, groups=storm_ids
        )
    )

    train_dataset = Subset(all_train_dataset, train_indices)
    val_dataset = Subset(all_train_dataset, val_indices)
    test_dataset = Subset(all_test_dataset, range(len(all_test_dataset)))

    return train_dataset, val_dataset, test_dataset


def main(args: argparse.Namespace) -> None:

    device = torch.device(  # type: ignore[attr-defined]
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ######################################
    # Setup output directory
    ######################################
    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)):
        if args.overwrite:
            print(
                f"WARNING! The output directory, {args.output_dir}, already exists, "
                + "we might overwrite data in it!"
            )
        else:
            print(
                f"The output directory, {args.output_dir}, already exists and isn't "
                + "empty. We don't want to overwrite and existing results, exiting..."
            )
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    # TODO: overwrite tensorboard output location too

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

    # TODO: this is horrible
    # 186 is the largest windspeed
    if args.model == "resnet18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=1)
    elif args.model == "resnet50":
        model = torchvision.models.resnet50(pretrained=False, num_classes=1)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, amsgrad=True)
    # criterion = nn.modules.loss.CrossEntropyLoss()
    criterion = nn.modules.loss.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=4, verbose=True
    )

    writer = SummaryWriter(  # type: ignore[no-untyped-call]
        log_dir=os.path.join("logs/", args.experiment_name)
    )

    ######################################
    # Training loop
    ######################################
    metrics_per_epoch: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
    }
    best_val: float = float("inf")

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

        metrics_per_epoch["train_loss"].append(training_loss)
        metrics_per_epoch["val_loss"].append(validation_loss)
        metrics_per_epoch["val_rmse"].append(validation_rmse)

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

        save_obj = {
            "epoch": epoch,
            "optimizer_checkpoint": copy.deepcopy(optimizer.state_dict()),
            "model_checkpoint": copy.deepcopy(model.state_dict()),
        }
        torch.save(
            save_obj, os.path.join(args.output_dir, "checkpoint_epoch_%d.pt" % (epoch))
        )
        if validation_rmse < best_val:
            print(f"New best! Validation RMSE: {validation_rmse:0.4f}")
            best_val = validation_rmse
            torch.save(save_obj, os.path.join(args.output_dir, "checkpoint_best.pt"))

        torch.save(
            {"metrics_per_epoch": metrics_per_epoch, "args": args},
            os.path.join(args.output_dir, "results.pt"),
        )

    writer.close()  # type: ignore[no-untyped-call]

    # Evaluate model on the test dataset
    test_results = evaluate(model, device, test_dataloader, criterion, 0)
    print(test_results)
    # TODO: save test results ot args.output_dir


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
