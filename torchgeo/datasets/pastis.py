# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PASTIS dataset."""

import os
from typing import Callable, Optional, Sequence

import fiona
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class PASTIS(NonGeoDataset):
    """PASTIS Panoptic Agricultural Satellite TIme Series dataset.

    The `PASTIS <https://doi.org/10.1109/ICCV48922.2021.00483>`__
    dataset is a benchmark dataset for panoptic and semantic segmentation of
    agricultural parcels from satellite time series. See the
    `GitHub repo <https://github.com/VSainteuf/pastis-benchmark/>`__ for details.

    Dataset features:

    * 2,433 patches (128x128 pixels) with time series of Sentinel-2 for each
    * 10 spectral bands (all Sentinel-2 L2A bands except B01, B09 and B10)
    * Variable number of time steps -- all available S2 images from 9/2018 to 11/2019
    * 18 crop classes with instance segmentation labels

    Dataset format:

    * images are three-channel jpgs

    Dataset classes:

    0. Background
    1. Meadow
    2. Soft winter wheat
    3. Corn
    4. Winter barley
    5. Winter rapeseed
    6. Spring barley
    7. Sunflower
    8. Grapevine
    9. Beet
    10. Winter triticale
    11. Winter durum wheat
    12. Fruits, vegetables, flowers
    13. Potatoes
    14. Leguminous fodder
    15. Soybeans
    16. Orchard
    17. Mixed cereal
    18. Sorghum
    19. Void label

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/ICCV48922.2021.00483

    .. versionadded:: 0.5
    """

    url = "https://zenodo.org/record/5012942/files/PASTIS.zip?download=1"
    md5 = "cfc441bf18137ff0bbf4fad58828fb98 "
    filename = "PASTIS.zip"
    directory = "PASTIS"

    targets = ["semantic", "instance"]
    classes = [
        "Background",
        "Meadow",
        "Soft winter wheat",
        "Corn",
        "Winter barley",
        "Winter rapeseed",
        "Spring barley",
        "Sunflower",
        "Grapevine",
        "Beet",
        "Winter triticale",
        "Winter durum wheat",
        "Fruits, vegetables, flowers",
        "Potatoes",
        "Leguminous fodder",
        "Soybeans",
        "Orchard",
        "Mixed cereal",
        "Sorghum",
        "Void label",
    ]

    # From https://github.com/VSainteuf/pastis-benchmark/blob/main/documentation/colormap.txt  # noqa: E501
    cmap = ListedColormap(
        [
            (0, 0, 0),
            (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
            (1.0, 0.4980392156862745, 0.054901960784313725),
            (1.0, 0.7333333333333333, 0.47058823529411764),
            (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
            (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            (1.0, 0.596078431372549, 0.5882352941176471),
            (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
            (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
            (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
            (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
            (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
            (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
            (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
            (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
            (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
            (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
            (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
            (1, 1, 1),
        ]
    )

    def __init__(
        self,
        root: str = "data",
        folds: Sequence[int] = [0, 1, 2, 3, 4],
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new RESISC45 dataset instance.

        Args:
            root: root directory where dataset can be found
            folds: a sequence of integers from 0 to 4 specifying which of the five
                dataset folds to include
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        for i in folds:
            assert 0 <= i < 5, f"Fold {i} is not in [0, 4]"
        self.root = root
        self.download = download
        self.checksum = checksum
        self.transforms = transforms
        self._verify()

        self.idxs = []
        metadata_fn = os.path.join(self.root, self.directory, "metadata.geojson")
        with fiona.open(metadata_fn) as f:
            for row in f:
                fold = int(row["properties"]["Fold"])
                if fold in folds:
                    self.idxs.append(row["properties"]["ID_PATCH"])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index

        Raises:
            IndexError: if index is out of range of the dataset
        """
        idx = self.idxs[index]
        img_fn = os.path.join(self.root, self.directory, "DATA_S2", f"S2_{idx}.npy")
        target_fn = os.path.join(
            self.root, self.directory, "ANNOTATIONS", f"TARGET_{idx}.npy"
        )
        # parcel_ids_fn = os.path.join(
        #     self.root, self.directory, "ANNOTATIONS", f"ParcelIDs_{idx}.npy"
        # )
        # heatmap_fn = os.path.join(
        #     self.root, self.directory, "INSTANCE_ANNOTATIONS", f"HEATMAP_{idx}.npy"
        # )
        instances_fn = os.path.join(
            self.root, self.directory, "INSTANCE_ANNOTATIONS", f"INSTANCES_{idx}.npy"
        )
        # zones_fn = os.path.join(
        #     self.root, self.directory, "INSTANCE_ANNOTATIONS", f"ZONES_{idx}.npy"
        # )

        img = torch.from_numpy(np.load(img_fn))  # int16 by default

        # See https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader.py#L201  # noqa: E501
        mask = torch.from_numpy(np.load(target_fn)[0]).long()
        instance_mask = torch.from_numpy(np.load(instances_fn)).long()

        sample = {"image": img, "mask": mask, "instance_mask": instance_mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.idxs)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        filepath = os.path.join(self.root, self.directory)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NonGeoClassificationDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # Keep the RGB bands and convert to T x H x W x C format
        images = sample["image"][:, [2, 1, 0], :, :].numpy().transpose(0, 2, 3, 1)
        mask = sample["mask"].numpy()

        num_panels = 3
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 4))
        axs[0].imshow(images[0] / 5000)
        axs[1].imshow(images[1] / 5000)
        axs[2].imshow(mask, vmin=0, vmax=19, cmap=self.cmap, interpolation="none")
        axs[0].axis("off")
        axs[1].axis("off")
        axs[2].axis("off")
        if showing_predictions:
            axs[3].imshow(
                predictions, vmin=0, vmax=19, cmap=self.cmap, interpolation="none"
            )
            axs[3].axis("off")

        if show_titles:
            axs[0].set_title("Image 0")
            axs[1].set_title("Image 1")
            axs[2].set_title("Mask")
            if showing_predictions:
                axs[3].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
