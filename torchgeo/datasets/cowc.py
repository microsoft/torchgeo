# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""COWC datasets."""

import abc
import csv
import os
from typing import Callable, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, download_and_extract_archive


class COWC(NonGeoDataset, abc.ABC):
    """Abstract base class for the COWC dataset.

    The `Cars Overhead With Context (COWC) <https://gdo152.llnl.gov/cowc/>`_ data set
    is a large set of annotated cars from overhead. It is useful for training a device
    such as a deep neural network to learn to detect and/or count cars.

    The dataset has the following attributes:

    1. Data from overhead at 15 cm per pixel resolution at ground (all data is EO).
    2. Data from six distinct locations: Toronto, Canada; Selwyn, New Zealand;
       Potsdam and Vaihingen, Germany; Columbus, Ohio and Utah, United States.
    3. 32,716 unique annotated cars. 58,247 unique negative examples.
    4. Intentional selection of hard negative examples.
    5. Established baseline for detection and counting tasks.
    6. Extra testing scenes for use after validation.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1007/978-3-319-46487-9_48
    """

    @property
    @abc.abstractmethod
    def base_url(self) -> str:
        """Base URL to download dataset from."""

    @property
    @abc.abstractmethod
    def filenames(self) -> List[str]:
        """List of files to download."""

    @property
    @abc.abstractmethod
    def md5s(self) -> List[str]:
        """List of MD5 checksums of files to download."""

    @property
    @abc.abstractmethod
    def filename(self) -> str:
        """Filename containing train/test split and target labels."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new COWC dataset instance.

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
        assert split in ["train", "test"]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.images = []
        self.targets = []
        with open(
            os.path.join(self.root, self.filename.format(split)),
            encoding="utf-8-sig",
            newline="",
        ) as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                self.images.append(row[0])
                self.targets.append(row[1])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample = {"image": self._load_image(index), "label": self._load_target(index)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.targets)

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        filename = os.path.join(self.root, self.images[index])
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img)
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load a single target.

        Args:
            index: index to return

        Returns:
            the target
        """
        target = int(self.targets[index])
        tensor = torch.tensor(target)
        return tensor

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        for filename, md5 in zip(self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for filename, md5 in zip(self.filenames, self.md5s):
            download_and_extract_archive(
                self.base_url + filename,
                self.root,
                filename=filename,
                md5=md5 if self.checksum else None,
            )

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

        .. versionadded:: 0.2
        """
        image = sample["image"]
        label = cast(str, sample["label"].item())

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = cast(str, sample["prediction"].item())
        else:
            prediction = None

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image.permute(1, 2, 0))
        ax.axis("off")

        if show_titles:
            title = f"Label: {label}"
            if prediction is not None:
                title += f"\nPrediction: {prediction}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class COWCCounting(COWC):
    """COWC Dataset for car counting."""

    base_url = (
        "https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/"
    )
    filenames = [
        "COWC_train_list_64_class.txt.bz2",
        "COWC_test_list_64_class.txt.bz2",
        "COWC_Counting_Toronto_ISPRS.tbz",
        "COWC_Counting_Selwyn_LINZ.tbz",
        "COWC_Counting_Potsdam_ISPRS.tbz",
        "COWC_Counting_Vaihingen_ISPRS.tbz",
        "COWC_Counting_Columbus_CSUAV_AFRL.tbz",
        "COWC_Counting_Utah_AGRC.tbz",
    ]
    md5s = [
        "187543d20fa6d591b8da51136e8ef8fb",
        "930cfd6e160a7b36db03146282178807",
        "bc2613196dfa93e66d324ae43e7c1fdb",
        "ea842ae055f5c74d0d933d2194764545",
        "19a77ab9932b722ef52b197d70e68ce7",
        "4009c1e420566390746f5b4db02afdb9",
        "daf8033c4e8ceebbf2c3cac3fabb8b10",
        "777ec107ed2a3d54597a739ce74f95ad",
    ]
    filename = "COWC_{}_list_64_class.txt"


class COWCDetection(COWC):
    """COWC Dataset for car detection."""

    base_url = (
        "https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/detection/"
    )
    filenames = [
        "COWC_train_list_detection.txt.bz2",
        "COWC_test_list_detection.txt.bz2",
        "COWC_Detection_Toronto_ISPRS.tbz",
        "COWC_Detection_Selwyn_LINZ.tbz",
        "COWC_Detection_Potsdam_ISPRS.tbz",
        "COWC_Detection_Vaihingen_ISPRS.tbz",
        "COWC_Detection_Columbus_CSUAV_AFRL.tbz",
        "COWC_Detection_Utah_AGRC.tbz",
    ]
    md5s = [
        "c954a5a3dac08c220b10cfbeec83893c",
        "c6c2d0a78f12a2ad88b286b724a57c1a",
        "11af24f43b198b0f13c8e94814008a48",
        "22fd37a86961010f5d519a7da0e1fc72",
        "bf053545cc1915d8b6597415b746fe48",
        "23945d5b22455450a938382ccc2a8b27",
        "f40522dc97bea41b10117d4a5b946a6f",
        "195da7c9443a939a468c9f232fd86ee3",
    ]
    filename = "COWC_{}_list_detection.txt"


# TODO: add COCW-M datasets:
#
# * https://gdo152.llnl.gov/cowc/download/cowc-m/datasets/
# * https://github.com/LLNL/cowc
#
# Same as COCW datasets, but instead of binary classification there are 4 car classes:
#
# 1. Sedan
# 2. Pickup
# 3. Other
# 4. Unknown
#
# May need new abstract base class. Will need subclasses for different patch sizes.
