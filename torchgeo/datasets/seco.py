# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Sentinel 2 imagery from the Seasonal Contrast paper."""

import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, extract_archive, percentile_normalization


class SeasonalContrastS2(NonGeoDataset):
    """Sentinel 2 imagery from the Seasonal Contrast paper.

    The `Seasonal Contrast imagery <https://github.com/ElementAI/seasonal-contrast/>`_
    dataset contains Sentinel 2 imagery patches sampled from different points in time
    around the 10k most populated cities on Earth.

    Dataset features:

    * Two versions: 100K and 1M patches
    * 12 band Sentinel 2 imagery from 5 points in time at each location

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/pdf/2103.16607.pdf
    """

    ALL_BANDS = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ]
    RGB_BANDS = ["B4", "B3", "B2"]

    urls = {
        # 7.3 GB
        "100k": "https://zenodo.org/record/4728033/files/seco_100k.zip?download=1",
        # 36.3 GB
        "1m": "https://zenodo.org/record/4728033/files/seco_1m.zip?download=1",
    }
    filenames = {"100k": "seco_100k.zip", "1m": "seco_1m.zip"}
    md5s = {
        "100k": "ebf2d5e03adc6e657f9a69a20ad863e0",
        "1m": "187963d852d4d3ce6637743ec3a4bd9e",
    }
    directory_names = {"100k": "seasonal_contrast_100k", "1m": "seasonal_contrast_1m"}

    def __init__(
        self,
        root: str = "data",
        version: str = "100k",
        bands: List[str] = RGB_BANDS,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SeCo dataset instance.

        Args:
            root: root directory where dataset can be found
            version: one of "100k" or "1m" for the version of the dataset to use
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``version`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert version in ["100k", "1m"]
        for band in bands:
            assert band in self.ALL_BANDS

        self.root = root
        self.bands = bands
        self.url = self.urls[version]
        self.filename = self.filenames[version]
        self.md5 = self.md5s[version]
        self.directory_name = self.directory_names[version]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        # TODO: This is slow, I think this should be generated on download and then
        # loaded in the constructor
        self.scene_to_patches = defaultdict(list)
        for root_directory, directories, fns in os.walk(
            os.path.join(self.root, self.directory_name)
        ):
            if len(directories) == 0 and len(fns) > 0:
                root_directory, patch_name = os.path.split(root_directory)
                _, scene_name = os.path.split(root_directory)
                self.scene_to_patches[scene_name].append(patch_name)

        self.scenes = sorted(self.scene_to_patches.keys())
        for scene_name in self.scenes:
            self.scene_to_patches[scene_name] = sorted(
                self.scene_to_patches[scene_name]
            )

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample with an "image" in 5xCxHxW format where the 5 indexes over the same
                patch sampled from different points in time by the SeCo method
        """
        scene_name = self.scenes[index]
        patch_names = self.scene_to_patches[scene_name]

        imagery = [
            self._load_patch(scene_name, patch_name) for patch_name in patch_names
        ]

        sample = {"image": torch.stack(imagery, dim=0)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.scenes)

    def _load_patch(self, scene_name: str, patch_name: str) -> Tensor:
        """Load a single image patch.

        Args:
            scene_name: the name of the scene to load from, e.g. '019999'
            patch_name: the name of the patch to load, e.g.
                '20200713T075609_20200713T081050_T36QZH'

        Returns:
            the image with the subset of bands specified by ``self.bands``
        """
        all_data = []
        for band in self.bands:
            fn = os.path.join(
                self.root, self.directory_name, scene_name, patch_name, f"{band}.tif"
            )
            with rasterio.open(fn) as f:
                band_data = f.read(1)
                height, width = band_data.shape
                size = min(height, width)
                if size < 264:
                    # TODO: PIL resize is much slower than cv2, we should check to see
                    # what could be sped up throughout later. There is also a potential
                    # slowdown here from converting to/from a PIL Image just to resize.
                    # https://gist.github.com/calebrob6/748045ac8d844154067b2eefa47de92f
                    pil_image = Image.fromarray(band_data)
                    # Moved in PIL 9.1.0
                    try:
                        resample = Image.Resampling.BILINEAR
                    except AttributeError:
                        resample = Image.BILINEAR
                    band_data = np.array(
                        pil_image.resize((264, 264), resample=resample)
                    )
                all_data.append(band_data)
        image = torch.from_numpy(np.stack(all_data, axis=0))
        return image

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        directory_path = os.path.join(self.root, self.directory_name)
        if os.path.exists(directory_path):
            return

        # Check if the zip files have already been downloaded
        zip_path = os.path.join(self.root, self.filename)
        if os.path.exists(zip_path):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
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
        extract_archive(os.path.join(self.root, self.filename))

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

        Raises:
            ValueError: if the RGB bands are included in ``self.bands`` or the sample
                contains a "prediction" key

        .. versionadded:: 0.2
        """
        if "prediction" in sample:
            raise ValueError("This dataset doesn't support plotting predictions")

        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        images = []
        for i in range(5):
            image = np.rollaxis(sample["image"][i, rgb_indices].numpy(), 0, 3)
            image = percentile_normalization(image, 0, 100)
            images.append(image)

        fig, axs = plt.subplots(ncols=5, figsize=(20, 4))
        for i in range(5):
            axs[i].imshow(images[i])
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title(f"t={i+1}")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
