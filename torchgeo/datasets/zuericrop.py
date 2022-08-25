# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ZueriCrop dataset."""

import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, percentile_normalization


class ZueriCrop(NonGeoDataset):
    """ZueriCrop dataset.

    The `ZueriCrop <https://github.com/0zgur0/ms-convSTAR>`__
    dataset is a dataset for time-series instance segmentation of crops.

    Dataset features:

    * Sentinel-2 multispectral imagery
    * instance masks of 48 crop categories
    * nine multispectral bands
    * 116k images with 10 m per pixel resolution (24x24 px)
    * ~28k time-series containing 142 images each

    Dataset format:

    * single hdf5 dataset containing images, semantic masks, and instance masks
    * data is parsed into images and instance masks, boxes, and labels
    * one mask per time-series

    Dataset classes:

    * 48 fine-grained hierarchical crop
      `categories <https://github.com/0zgur0/ms-convSTAR/blob/master/labels.csv>`_

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.rse.2021.112603

    .. note::

       This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset
    """

    urls = [
        "https://polybox.ethz.ch/index.php/s/uXfdr2AcXE3QNB6/download",
        "https://raw.githubusercontent.com/0zgur0/ms-convSTAR/master/labels.csv",
    ]
    md5s = ["1635231df67f3d25f4f1e62c98e221a4", "5118398c7a5bbc246f5f6bb35d8d529b"]
    filenames = ["ZueriCrop.hdf5", "labels.csv"]

    band_names = ("NIR", "B03", "B02", "B04", "B05", "B06", "B07", "B11", "B12")
    RGB_BANDS = ["B04", "B03", "B02"]

    def __init__(
        self,
        root: str = "data",
        bands: Sequence[str] = band_names,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new ZueriCrop dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self._validate_bands(bands)
        self.band_indices = torch.tensor(
            [self.band_names.index(b) for b in bands]
        ).long()

        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.filepath = os.path.join(root, "ZueriCrop.hdf5")

        self._verify()

        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError(
                "h5py is not installed and is required to use this dataset"
            )

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample containing image, mask, bounding boxes, and target label
        """
        image = self._load_image(index)
        mask, boxes, label = self._load_target(index)

        sample = {"image": image, "mask": mask, "boxes": boxes, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        import h5py

        with h5py.File(self.filepath, "r") as f:
            length: int = f["data"].shape[0]
        return length

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        import h5py

        with h5py.File(self.filepath, "r") as f:
            array = f["data"][index, ...]

        tensor = torch.from_numpy(array)
        # Convert from TxHxWxC to TxCxHxW
        tensor = tensor.permute((0, 3, 1, 2))
        tensor = torch.index_select(tensor, dim=1, index=self.band_indices)
        return tensor

    def _load_target(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask and label for each mask
        """
        import h5py

        with h5py.File(self.filepath, "r") as f:
            mask_array = f["gt"][index, ...]
            instance_array = f["gt_instance"][index, ...]

        mask_tensor = torch.from_numpy(mask_array)
        instance_tensor = torch.from_numpy(instance_array)

        # Convert from HxWxC to CxHxW
        mask_tensor = mask_tensor.permute((2, 0, 1))
        instance_tensor = instance_tensor.permute((2, 0, 1))

        # Convert instance mask of N instances to N binary instance masks
        instance_ids = torch.unique(instance_tensor)
        # Exclude a mask for unknown/background
        instance_ids = instance_ids[instance_ids != 0]
        instance_ids = instance_ids[:, None, None]
        masks = instance_tensor == instance_ids

        # Parse labels for each instance
        labels_list = []
        for mask in masks:
            label = mask_tensor[mask[None, :, :]]
            label = torch.unique(label)[0]
            labels_list.append(label)

        # Get bounding boxes for each instance
        boxes_list = []
        for mask in masks:
            pos = torch.where(mask)
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes_list.append([xmin, ymin, xmax, ymax])

        masks = masks.to(torch.uint8)
        boxes = torch.tensor(boxes_list).to(torch.float)
        labels = torch.tensor(labels_list).to(torch.long)

        return masks, boxes, labels

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        exists = []
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            exists.append(os.path.exists(filepath))

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        for url, filename, md5 in zip(self.urls, self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                download_url(
                    url,
                    self.root,
                    filename=filename,
                    md5=md5 if self.checksum else None,
                )

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided sequence of bands to load
        Raises:
            AssertionError: if ``bands`` is not a sequence
            ValueError: if an invalid band name is provided

        .. versionadded:: 0.2
        """
        assert isinstance(bands, Sequence), "'bands' must be a sequence"
        for band in bands:
            if band not in self.band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def plot(
        self,
        sample: Dict[str, Tensor],
        time_step: int = 0,
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            time_step: time step at which to access image, beginning with 0
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        ncols = 2
        image, mask = sample["image"][time_step, rgb_indices], sample["mask"]

        image = torch.tensor(
            percentile_normalization(image.numpy()) * 255, dtype=torch.uint8
        )

        mask = torch.argmax(mask, dim=0)

        if "prediction" in sample:
            ncols += 1
            preds = torch.argmax(sample["prediction"], dim=0)

        fig, axs = plt.subplots(ncols=ncols, figsize=(10, 10 * ncols))

        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if "prediction" in sample:
            axs[2].imshow(preds)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
