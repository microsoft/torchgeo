# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from matplotlib.figure import Figure
from numpy import ndarray as Array
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets.utils import dataset_split, draw_semantic_segmentation_masks
from .geo import VisionDataset
from .utils import download_url, extract_archive, sort_sentinel2_bands


class OSCD(VisionDataset):
    """OSCD dataset.

    The `Onera Satellite Change Detection <https://rcdaudt.github.io/oscd/>`_
    dataset addresses the issue of detecting changes between
    satellite images from different dates. Imagery comes from
    Sentinel-2 which contains varying resolutions per band.

    Dataset format:

    * images are 13-channel tifs
    * masks are single-channel pngs where no change = 0, change = 255

    Dataset classes:

    0. no change
    1. change

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2018.8518015

    .. versionadded:: 0.2
    """

    folder_prefix = "Onera Satellite Change Detection dataset - "
    urls = {
        "Onera Satellite Change Detection dataset - Images.zip": (
            "https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download"
        ),
        "Onera Satellite Change Detection dataset - Train Labels.zip": (
            "https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu/download"
        ),
        "Onera Satellite Change Detection dataset - Test Labels.zip": (
            "https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download"
        ),
    }

    md5 = "7383412da7ece1dca1c12dc92ac77f09"

    zipfile_glob = "*Onera*.zip"
    filename_glob = "*Onera*"
    splits = ["train", "test"]

    colormap = ["blue"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new OSCD dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in self.splits
        assert bands in ["rgb", "all"]

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files["images1"])
        image2 = self._load_image(files["images2"])
        mask = self._load_target(str(files["mask"]))

        image = torch.stack(tensors=[image1, image2], dim=0)
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

    def _load_files(self) -> List[Dict[str, Union[str, Sequence[str]]]]:
        regions = []
        labels_root = os.path.join(
            self.root,
            f"Onera Satellite Change Detection dataset - {self.split.capitalize()} "
            + "Labels",
        )
        images_root = os.path.join(
            self.root, "Onera Satellite Change Detection dataset - Images"
        )
        folders = glob.glob(os.path.join(labels_root, "*/"))
        for folder in folders:
            region = folder.split(os.sep)[-2]
            mask = os.path.join(labels_root, region, "cm", "cm.png")

            def get_image_paths(ind: int) -> List[str]:
                return sorted(
                    glob.glob(
                        os.path.join(images_root, region, f"imgs_{ind}_rect", "*.tif")
                    ),
                    key=sort_sentinel2_bands,
                )

            images1, images2 = get_image_paths(1), get_image_paths(2)
            if self.bands == "rgb":
                images1, images2 = images1[1:4][::-1], images2[1:4][::-1]

            with open(os.path.join(images_root, region, "dates.txt")) as f:
                dates = tuple(
                    [line.split()[-1] for line in f.read().strip().splitlines()]
                )

            regions.append(
                dict(
                    region=region,
                    images1=images1,
                    images2=images2,
                    mask=mask,
                    dates=dates,
                )
            )

        return regions

    def _load_image(self, paths: Sequence[str]) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        images = []
        for path in paths:
            with rasterio.open(path) as f:
                images.append(f.read())
        array = np.stack(images, axis=0).astype(np.int_).squeeze()
        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
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
            tensor = torch.clamp(tensor, min=0, max=1)  # type: ignore[attr-defined]
            tensor = tensor.to(torch.long)  # type: ignore[attr-defined]
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "**", self.filename_glob)
        for fname in glob.iglob(pathname, recursive=True):
            if not fname.endswith(".zip"):
                return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_glob)
        if glob.glob(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for f_name in self.urls:
            download_url(
                self.urls[f_name],
                self.root,
                filename=f_name,
                md5=self.md5 if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.zipfile_glob)
        for zipfile in glob.iglob(pathname):
            extract_archive(zipfile)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        alpha: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            alpha: opacity with which to render predictions on top of the imagery

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2

        rgb_inds = [3, 2, 1] if self.bands == "all" else [0, 1, 2]

        def get_masked(img: Tensor) -> Array:  # type: ignore[type-arg]
            rgb_img = img[rgb_inds].float().numpy()
            per02 = np.percentile(rgb_img, 2)  # type: ignore[no-untyped-call]
            per98 = np.percentile(rgb_img, 98)  # type: ignore[no-untyped-call]
            rgb_img = (np.clip((rgb_img - per02) / (per98 - per02), 0, 1) * 255).astype(
                np.uint8
            )
            array: Array = draw_semantic_segmentation_masks(  # type: ignore[type-arg]
                torch.from_numpy(rgb_img),  # type: ignore[attr-defined]
                sample["mask"],
                alpha=alpha,
                colors=self.colormap,  # type: ignore[arg-type]
            )
            return array

        image1, image2 = get_masked(sample["image"][0]), get_masked(sample["image"][1])
        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs[0].imshow(image1)
        axs[0].axis("off")
        axs[1].imshow(image2)
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Pre change")
            axs[1].set_title("Post change")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


# TODO: add validation split
class OSCDDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the OSCD dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded: 0.2
    """

    # NOTE: For some reason this doesn't have the B10 band so for now I'll insert
    # a value based on it's neighbor while I figure out what to put there.

    # (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12)
    # min/max band statistics computed on 100k random samples
    band_mins_raw = torch.tensor(  # type: ignore[attr-defined]
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]  # B10 = mode(all)
    )
    band_maxs_raw = torch.tensor(  # type: ignore[attr-defined]
        [
            18556.0,
            20528.0,
            18976.0,
            17874.0,
            16611.0,
            16512.0,
            16394.0,
            16672.0,
            16141.0,
            16097.0,
            15716.0,  # (16097 + 15336)/2
            15336.0,
            15203.0,
        ]
    )

    # min/max band statistics computed by percentile clipping the
    # above to samples to [2, 98]
    band_mins = torch.tensor(  # type: ignore[attr-defined]
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]  # B10 = mode(all)
    )
    band_maxs = torch.tensor(  # type: ignore[attr-defined]
        [
            9859.0,
            12872.0,
            13163.0,
            14445.0,
            12477.0,
            12563.0,
            12289.0,
            15596.0,
            12183.0,
            9458.0,
            7677.0,  # (9458 + 5897)/2
            5897.0,
            5544.0,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        bands: str = "all",
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for OSCD based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the OSCD Dataset classes
            bands: "rgb" or "all"
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct

        if bands == "rgb":
            self.mins = self.band_mins[[3, 2, 1], None, None]
            self.maxs = self.band_maxs[[3, 2, 1], None, None]
        else:
            self.mins = self.band_mins[:, None, None]
            self.maxs = self.band_maxs[:, None, None]

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"].float()
        sample["image"] = (sample["image"] - self.mins) / (self.maxs - self.mins)
        sample["image"] = torch.clip(  # type: ignore[attr-defined]
            sample["image"], min=0.0, max=1.0
        )
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        OSCD(self.root_dir, split="train", bands=self.bands, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        transforms = Compose([self.preprocess])

        dataset = OSCD(
            self.root_dir, split="train", bands=self.bands, transforms=transforms
        )

        # TODO: maybe we can remove this if statement?
        # include this functionality in dataset_split?
        if self.val_split_pct > 0.0:
            self.train_dataset, self.val_dataset, _ = dataset_split(
                dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
        else:
            self.train_dataset = dataset  # type: ignore[assignment]
            self.val_dataset = None  # type: ignore[assignment]

        self.test_dataset = OSCD(
            self.root_dir, split="test", bands=self.bands, transforms=transforms
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
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
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
