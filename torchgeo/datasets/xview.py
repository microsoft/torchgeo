# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""xView2 dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets.utils import dataset_split, draw_semantic_segmentation_masks
from .geo import VisionDataset


class XView2(VisionDataset):
    """xView2 dataset.

    The `xView2 <https://xview2.org/>`_
    dataset is a dataset for building disaster change detection.

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs where the pixel values represent the class

    Dataset classes:

    0. background
    1. no damage
    2. minor damage
    3. major damage
    4. destroyed

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1911.09296

    """

    metadata = {
        "train": {
            "filename": "train_images_labels_targets.tar",
            "md5": "a20ebbfb7eb3452785b63ad02ffd1e16",
            "directory": "train_images_labels_targets",
        },
        "test": {
            "filename": "test_images_labels_targets.tar",
            "md5": "1b39c47e05d1319c17cc8763cee6fe0c",
            "directory": "test_images_labels_targets",
        },
    }
    classes = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
    colormap = ["green", "blue", "orange", "red"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new xView2 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        assert split in self.metadata
        self.root = root
        self.split = split
        self.transforms = transforms
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files(root, split)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files["image1"])
        image2 = self._load_image(files["image2"])
        mask1 = self._load_target(files["mask1"])
        mask2 = self._load_target(files["mask2"])

        image = torch.stack(tensors=[image1, image2], dim=0)
        mask = torch.stack(tensors=[mask1, mask2], dim=0)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self, root: str, split: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            split: subset of dataset, one of [train, test]

        Returns:
            list of dicts containing paths for each pair of images and masks
        """
        files = []
        directory = self.metadata[split]["directory"]
        image_root = os.path.join(root, directory, split, "images")
        mask_root = os.path.join(root, directory, split, "targets")
        images = glob.glob(os.path.join(image_root, "*.png"))
        basenames = [os.path.basename(f) for f in images]
        basenames = ["_".join(f.split("_")[:-2]) for f in basenames]
        for name in set(basenames):
            image1 = os.path.join(image_root, f"{name}_pre_disaster.png")
            image2 = os.path.join(image_root, f"{name}_post_disaster.png")
            mask1 = os.path.join(mask_root, f"{name}_pre_disaster_target.png")
            mask2 = os.path.join(mask_root, f"{name}_post_disaster_target.png")
            files.append(dict(image1=image1, image2=image2, mask1=mask1, mask2=mask2))
        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("RGB"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("L"))
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
            return tensor

    def plot(self, index: int, alpha: float = 0.5) -> plt.Figure:
        """Plot a data sample.

        Args:
            index: the index of the sample to plot

        Returns:

        """
        sample = self[index]
        image1 = draw_semantic_segmentation_masks(
            sample["image"][0], sample["mask"][0], alpha=alpha, colors=self.colormap
        )
        image2 = draw_semantic_segmentation_masks(
            sample["image"][1], sample["mask"][1], alpha=alpha, colors=self.colormap
        )
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        fig.set_size_inches((25, 25))
        ax1.imshow(image1)
        ax1.set_axis_off()
        ax2.imshow(image2)
        ax2.set_axis_off()
        plt.tight_layout()
        return fig


class XView2DataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the xView2 dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for xView2 based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the xView2 Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        dataset = XView2(self.root_dir, "train", transforms=transforms)

        if self.val_split_pct > 0.0:
            self.train_dataset, self.val_dataset, _ = dataset_split(
                dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
        else:
            self.train_dataset = dataset  # type: ignore[assignment]
            self.val_dataset = None  # type: ignore[assignment]

        self.test_dataset = XView2(self.root_dir, "test", transforms=transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        if self.val_split_pct == 0.0:
            return self.train_dataloader()
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
