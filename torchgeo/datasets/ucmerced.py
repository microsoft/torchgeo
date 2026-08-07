# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""UC Merced dataset."""

import os
from collections.abc import Callable
from typing import ClassVar, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoClassificationDataset
from .utils import Path, Sample, check_integrity, download_url, extract_archive


class UCMerced(NonGeoClassificationDataset):
    """UC Merced Land Use dataset.

    The `UC Merced Land Use <https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset>`_
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

    url = 'https://hf.co/datasets/torchgeo/ucmerced/resolve/7c5ef3454d9b1cccfa7ccde0c01fc8f00a45909a/'
    filename = 'UCMerced_LandUse.zip'
    sha256 = '06c539ef28703a58fb07bd2837991ac7c48b813b00bb12ac197efd813a18daeb'

    base_dir = os.path.join('UCMerced_LandUse', 'Images')

    splits = ('train', 'val', 'test')
    split_filenames: ClassVar[dict[str, str]] = {
        'train': 'uc_merced-train.txt',
        'val': 'uc_merced-val.txt',
        'test': 'uc_merced-test.txt',
    }
    split_sha256s: ClassVar[dict[str, str]] = {
        'train': 'd625ea884cb2870e007774c1d64e904e9ae71ba3e7b4b92a9e8aa6065e2cd8cc',
        'val': '4459da518e2c1486471b4ea57734950c9c7c449611fe3468c0d1ca34a4bd3a56',
        'test': '0d3b64706fd2a9f9faaa1d7dade934ec8b4fd258a9853f037983b1f2db239220',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        transforms: Callable[[Sample], Sample] | None = None,
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
            checksum: if True, verify the checksum of the downloaded files (may be slow)

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
        with open(os.path.join(self.root, self.split_filenames[split])) as f:
            for fn in f:
                valid_fns.add(fn.strip())

        def is_in_split(x: Path) -> bool:
            return os.path.basename(x) in valid_fns

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
        img = F.resize(img, size=[256, 256], antialias=True)
        return img, label

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or checksums match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename),
            sha256=self.sha256 if self.checksum else None,
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
            self.url + self.filename,
            self.root,
            sha256=self.sha256 if self.checksum else None,
        )
        for split in self.splits:
            download_url(
                self.url + self.split_filenames[split],
                self.root,
                sha256=self.split_sha256s[split] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
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
        image = np.rollaxis(sample['image'].numpy(), 0, 3)

        # Normalize the image if the max value is greater than 1
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0  # Scale to [0, 1]

        label = cast(int, sample['label'].item())
        label_class = self.classes[label]

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = cast(int, sample['prediction'].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')
        if show_titles:
            title = f'Label: {label_class}'
            if showing_predictions:
                title += f'\nPrediction: {prediction_class}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
