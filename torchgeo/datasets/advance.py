# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ADVANCE dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_and_extract_archive


class ADVANCE(NonGeoDataset):
    """ADVANCE dataset.

    The `ADVANCE <https://akchen.github.io/ADVANCE-DATASET/>`__
    dataset is a dataset for audio visual scene recognition.

    Dataset features:

    * 5,075 pairs of geotagged audio recordings and images
    * three spectral bands - RGB (512x512 px)
    * 10-second audio recordings

    Dataset format:

    * images are three-channel jpgs
    * audio files are in wav format

    Dataset classes:

    0. airport
    1. beach
    2. bridge
    3. farmland
    4. forest
    5. grassland
    6. harbour
    7. lake
    8. orchard
    9. residential
    10. sparse shrub land
    11. sports land
    12. train station

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1007/978-3-030-58586-0_5

    .. note::
        This dataset requires the following additional library to be installed:

        * `scipy <https://pypi.org/project/scipy/>`_ to load the audio files to tensors
    """

    urls = [
        "https://zenodo.org/record/3828124/files/ADVANCE_vision.zip?download=1",
        "https://zenodo.org/record/3828124/files/ADVANCE_sound.zip?download=1",
    ]
    filenames = ["ADVANCE_vision.zip", "ADVANCE_sound.zip"]
    md5s = ["a9e8748219ef5864d3b5a8979a67b471", "a2d12f2d2a64f5c3d3a9d8c09aaf1c31"]
    directories = ["vision", "sound"]
    classes = [
        "airport",
        "beach",
        "bridge",
        "farmland",
        "forest",
        "grassland",
        "harbour",
        "lake",
        "orchard",
        "residential",
        "sparse shrub land",
        "sports land",
        "train station",
    ]

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new ADVANCE dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.files = self._load_files(self.root)
        self.classes = sorted({f["cls"] for f in self.files})
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image = self._load_image(files["image"])
        audio = self._load_target(files["audio"])
        cls_label = self.class_to_idx[files["cls"]]
        label = torch.tensor(cls_label, dtype=torch.long)
        sample = {"image": image, "audio": audio, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self, root: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image, audio, label
        """
        images = sorted(glob.glob(os.path.join(root, "vision", "**", "*.jpg")))
        wavs = sorted(glob.glob(os.path.join(root, "sound", "**", "*.wav")))
        labels = [image.split(os.sep)[-2] for image in images]
        files = [
            dict(image=image, audio=wav, cls=label)
            for image, wav, label in zip(images, wavs, labels)
        ]
        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the target audio for a single image.

        Args:
            path: path to the target

        Returns:
            the target audio
        """
        try:
            from scipy.io import wavfile
        except ImportError:
            raise ImportError(
                "scipy is not installed and is required to use this dataset"
            )

        array = wavfile.read(path, mmap=True)[1]
        tensor = torch.from_numpy(array)
        tensor = tensor.unsqueeze(0)
        return tensor

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories are found, else False
        """
        for directory in self.directories:
            filepath = os.path.join(self.root, directory)
            if not os.path.exists(filepath):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for filename, url, md5 in zip(self.filenames, self.urls, self.md5s):
            download_and_extract_archive(
                url, self.root, filename=filename, md5=md5 if self.checksum else None
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
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
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
