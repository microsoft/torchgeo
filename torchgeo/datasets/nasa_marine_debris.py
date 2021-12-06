# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NASA Marine Debris dataset."""

import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.utils import draw_bounding_boxes

from .geo import VisionDataset
from .utils import dataset_split, download_radiant_mlhub_dataset, extract_archive

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Any]:
    """Custom object detection collate fn to handle variable boxes.

    Args:
        batch: list of sample dicts return by dataset

    Returns:
        batch dict output
    """
    output: Dict[str, Any] = {}
    output["image"] = torch.stack([sample["image"] for sample in batch])
    output["boxes"] = [sample["boxes"] for sample in batch]
    return output


class NASAMarineDebris(VisionDataset):
    """NASA Marine Debris dataset.

    The `NASA Marine Debris <https://mlhub.earth/data/nasa_marine_debris>`_
    dataset is a dataset for detection of floating marine debris in satellite imagery.

    Dataset features:

    * 707 patches with 3 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 1 object class: marine_debris
    * images taken by Planet Labs PlanetScope satellites
    * imagery taken from 2016-2019 from coasts of Greece, Honduras, and Ghana

    Dataset format:

    * images are three-channel geotiffs in uint8 format
    * labels are numpy files (.npy) containing bounding box (xyxy) coordinates
    * additional: images in jpg format and labels in geojson format

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/rdnt.9r6ekg

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub
    """

    dataset_id = "nasa_marine_debris"
    directories = ["nasa_marine_debris_source", "nasa_marine_debris_labels"]
    filenames = ["nasa_marine_debris_source.tar.gz", "nasa_marine_debris_labels.tar.gz"]
    md5s = ["fe8698d1e68b3f24f0b86b04419a797d", "d8084f5a72778349e07ac90ec1e1d990"]
    class_label = "marine_debris"

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize a new NASA Marine Debris Dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            verbose: if True, print messages when new tiles are loaded
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.api_key = api_key
        self.checksum = checksum
        self.verbose = verbose
        self._verify()
        self.files = self._load_files()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        image = self._load_image(self.files[index]["image"])
        boxes = self._load_target(self.files[index]["target"])
        sample = {"image": image, "boxes": boxes}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array = f.read()
        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the target bounding boxes for a single image.

        Args:
            path: path to the labels

        Returns:
            the target boxes
        """
        array = np.load(path)  # type: ignore[no-untyped-call]
        # boxes contain unecessary value of 1 after xyxy coords
        array = array[:, :4]
        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        return tensor

    def _load_files(self) -> List[Dict[str, str]]:
        """Load a image and label files.

        Returns:
            list of dicts containing image and label files
        """
        image_root = os.path.join(self.root, self.directories[0])
        target_root = os.path.join(self.root, self.directories[1])
        image_folders = sorted(
            [f for f in os.listdir(image_root) if not f.endswith("json")]
        )

        files = []
        for folder in image_folders:
            files.append(
                {
                    "image": os.path.join(image_root, folder, "image_geotiff.tif"),
                    "target": os.path.join(
                        target_root,
                        folder.replace("source", "labels"),
                        "pixel_bounds.npy",
                    ),
                }
            )
        return files

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        exists = [
            os.path.exists(os.path.join(self.root, directory))
            for directory in self.directories
        ]
        if all(exists):
            return

        # Check if zip file already exists (if so then extract)
        exists = []
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            if os.path.exists(filepath):
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` directory and "
                "`download=False`, either specify a different `root` directory"
                "or use `download=True` to automaticaly download the dataset."
            )

        # TODO: need a checksum check in here post downloading
        # Download and extract the dataset
        download_radiant_mlhub_dataset(self.dataset_id, self.root, self.api_key)
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            extract_archive(filepath)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 1

        image = draw_bounding_boxes(image=sample["image"], boxes=sample["boxes"])
        image = image.permute((1, 2, 0)).numpy()

        if "prediction_boxes" in sample:
            ncols += 1
            preds = draw_bounding_boxes(
                image=sample["image"], boxes=sample["prediction_boxes"]
            )
            preds = preds.permute((1, 2, 0)).numpy()

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if ncols < 2:
            axs.imshow(image)
            axs.axis("off")
            if show_titles:
                axs.set_title("Ground Truth")
        else:
            axs[0].imshow(image)
            axs[0].axis("off")
            axs[1].imshow(preds)
            axs[1].axis("off")

            if show_titles:
                axs[0].set_title("Ground Truth")
                axs[1].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class NASAMarineDebrisDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NASA Marine Debris dataset."""

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NASA Marine Debris based DataLoaders.

        Args:
            root_dir: The ``root`` argument to pass to the Dataset class
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

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

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        NASAMarineDebris(self.root_dir, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        dataset = NASAMarineDebris(self.root_dir, transforms=transforms)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )

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
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
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
            collate_fn=collate_fn,
        )
