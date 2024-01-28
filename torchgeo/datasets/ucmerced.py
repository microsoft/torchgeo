# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""UC Merced dataset."""
import os
from typing import Callable, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoClassificationDataset
from .utils import DatasetNotFoundError, check_integrity, download_url, extract_archive


class UCMerced(NonGeoClassificationDataset):
    """UC Merced Land Use dataset.

    The `UC Merced Land Use <http://weegee.vision.ucmerced.edu/datasets/landuse.html>`_
    dataset is a land use classification dataset of 2.1k 256x256 1ft resolution RGB
    images of urban locations around the U.S. extracted from the USGS National Map Urban
    Area Imagery collection with 21 land use classes (100 images per class).

    Dataset features:

    * land use class labels from around the U.S.
    * three spectral bands - RGB
    * 21 classes

    Dataset classes:

    * agricultural
    * airplane
    * baseballdiamond
    * beach
    * buildings
    * chaparral
    * denseresidential
    * forest
    * freeway
    * golfcourse
    * harbor
    * intersection
    * mediumresidential
    * mobilehomepark
    * overpass
    * parkinglot
    * river
    * runway
    * sparseresidential
    * storagetanks
    * tenniscourt

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following paper:

    * https://dl.acm.org/doi/10.1145/1869790.1869829
    """

    url = "https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip"  # noqa: E501
    filename = "UCMerced_LandUse.zip"
    md5 = "5b7ec56793786b6dc8a908e8854ac0e4"

    base_dir = os.path.join("UCMerced_LandUse", "Images")

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-train.txt",  # noqa: E501
        "val": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-val.txt",  # noqa: E501
        "test": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-test.txt",  # noqa: E501
    }
    split_md5s = {
        "train": "f2fb12eb2210cfb53f93f063a35ff374",
        "val": "11ecabfc52782e5ea6a9c7c0d263aca0",
        "test": "046aff88472d8fc07c4678d03749e28d",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new UC Merced dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, f"uc_merced-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transforms,
            is_valid_file=is_in_split,
        )

    def _load_image(self, index: int) -> tuple[Tensor, Tensor]:
        """Load a single image and its class label.

        Args:
            index: index to return

        Returns:
            the image and class label
        """
        img, label = super()._load_image(index)
        img = F.resize(img, size=(256, 256), antialias=True)
        return img, label

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename), self.md5 if self.checksum else None
        )
        return integrity

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        filepath = os.path.join(self.root, self.base_dir)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        if self._check_integrity():
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

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
        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                filename=f"uc_merced-{split}.txt",
                md5=self.split_md5s[split] if self.checksum else None,
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
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NonGeoClassificationDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)

        # Normalize the image if the max value is greater than 1
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0  # Scale to [0, 1]

        label = cast(int, sample["label"].item())
        label_class = self.classes[label]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = cast(int, sample["prediction"].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {label_class}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_class}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
