# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SeasoNet dataset."""

import os
import random
from typing import Callable, Collection, Iterable, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from rasterio.enums import Resampling
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, extract_archive, percentile_normalization


class SeasoNet(NonGeoDataset):  # TODO: Docs
    """SeasoNet Semantic Segmentation dataset.

    The `SeasoNet <https://doi.org/10.5281/zenodo.5850306>`__ dataset consists of
    1,759,830 multi-spectral Sentinel-2 image patches, taken from 519,547 unique
    locations, covering the whole surface area of Germany. Annotations are
    provided in the form of pixel-level land cover and land usage segmentation
    masks from the German land cover model LBM-DE2018 with land cover classes
    based on the CORINE Land Cover database (CLC) 2018. The set is split into
    two overlapping grids, consisting of roughly 880,000 samples each, which are
    shifted by half the patch size in both dimensions. The images in each of the
    both grids themselves do not overlap.

    Dataset format:

    * images are 16-bit GeoTiffs, split into seperate files based on resolution
    * images include 12 spectral bands with 10, 20 and 60 m per pixel resolutions
    * masks are single-channel 8-bit GeoTiffs

    Dataset classes:

    0. Continuous urban fabric
    1. Discontinuous urban fabric
    2. Industrial or commercial units
    3. Road and rail networks and associated land
    4. Port areas
    5. Airports
    6. Mineral extraction sites
    7. Dump sites
    8. Construction sites
    9. Green urban areas
    10. Sport and leisure facilities
    11. Non-irrigated arable land
    12. Vineyards
    13. Fruit trees and berry plantations
    14. Pastures
    15. Broad-leaved forest
    16. Coniferous forest
    17. Mixed forest
    18. Natural grasslands
    19. Moors and heathland
    20. Transitional woodland/shrub
    21. Beaches, dunes, sands
    22. Bare rock
    23. Sparsely vegetated areas
    24. Inland marshes
    25. Peat bogs
    26. Salt marshes
    27. Intertidal flats
    28. Water courses
    29. Water bodies
    30. Coastal lagoons
    31. Estuaries
    32. Sea and ocean

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS46834.2022.9884079

    .. note::
       This dataset requires the following additional library to be installed:

       * `pandas <https://pypi.org/project/pandas/>`_ to load CSV files

    .. versionadded:: 0.5
    """

    url = "https://zenodo.org/api/records/5850306"
    file_names = [
        "spring.zip",
        "summer.zip",
        "fall.zip",
        "winter.zip",
        "snow.zip",
        "splits.zip",
        "meta.csv",
    ]
    extracted_names = [
        "spring",
        "summer",
        "fall",
        "winter",
        "snow",
        "splits",
        "meta.csv",
    ]
    classes = [
        "Continuous urban fabric",
        "Discontinuous urban fabric",
        "Industrial or commercial units",
        "Road and rail networks and associated land",
        "Port areas",
        "Airports",
        "Mineral extraction sites",
        "Dump sites",
        "Construction sites",
        "Green urban areas",
        "Sport and leisure facilities",
        "Non-irrigated arable land",
        "Vineyards",
        "Fruit trees and berry plantations",
        "Pastures",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grasslands",
        "Moors and heathland",
        "Transitional woodland/shrub",
        "Beaches, dunes, sands",
        "Bare rock",
        "Sparsely vegetated areas",
        "Inland marshes",
        "Peat bogs",
        "Salt marshes",
        "Intertidal flats",
        "Water courses",
        "Water bodies",
        "Coastal lagoons",
        "Estuaries",
        "Sea and ocean",
    ]
    all_seasons = {"Spring", "Summer", "Fall", "Winter", "Snow"}
    all_bands = ("10m_RGB", "10m_IR", "20m", "60m")
    band_nums = {"10m_RGB": 3, "10m_IR": 1, "20m": 6, "60m": 2}
    splits = ["train", "val", "test"]
    colormap = [
        (230, 000, 77, 255),
        (255, 000, 000, 255),
        (204, 77, 242, 255),
        (204, 000, 000, 255),
        (230, 204, 204, 255),
        (230, 204, 230, 255),
        (166, 000, 204, 255),
        (166, 77, 000, 255),
        (255, 77, 255, 255),
        (255, 166, 255, 255),
        (255, 230, 255, 255),
        (255, 255, 168, 255),
        (230, 128, 000, 255),
        (242, 166, 77, 255),
        (230, 230, 77, 255),
        (128, 255, 000, 255),
        (000, 166, 000, 255),
        (77, 255, 000, 255),
        (204, 242, 77, 255),
        (166, 255, 128, 255),
        (166, 242, 000, 255),
        (230, 230, 230, 255),
        (204, 204, 204, 255),
        (204, 255, 204, 255),
        (166, 166, 255, 255),
        (77, 77, 255, 255),
        (204, 204, 255, 255),
        (166, 166, 230, 255),
        (000, 204, 242, 255),
        (128, 242, 230, 255),
        (000, 255, 166, 255),
        (166, 255, 230, 255),
        (230, 242, 255, 255),
    ]
    image_size = (120, 120)

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        seasons: Collection[str] = all_seasons,
        bands: Iterable[str] = all_bands,
        grids: Iterable[int] = [1, 2],
        concat_seasons: int = 1,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SeasoNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val" or "test"
            seasons: list of seasons to load
            bands: list of bands to load
            grids: which of the overlapping grids to load
            concat_seasons: number of seasonal images to return per sample.
                if 1, each seasonal image is returned as its own sample,
                otherwise seasonal images are randomly picked from the seasons
                specified in `seasons` and returned as stacked tensors
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            ImportError: if pandas is not installed
        """
        assert split in self.splits
        for season in seasons:
            assert season in self.all_seasons
        for band in bands:
            assert band in self.all_bands
        for grid in grids:
            assert grid in [1, 2]
        assert concat_seasons in range(1, len(seasons) + 1)

        self.root = root
        self.bands = bands
        self.concat_seasons = concat_seasons
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is not installed and is required to use this dataset"
            )

        self._verify()

        self.channels = 0
        for b in bands:
            self.channels += self.band_nums[b]

        csv = pd.read_csv(os.path.join(self.root, "meta.csv"), index_col="Index")

        if split is not None:
            split_csv = pd.read_csv(
                os.path.join(self.root, f"splits/{split}.csv"), header=None
            )[0]
            csv = csv.iloc[split_csv]

        csv = csv[csv["Grid"].isin(grids)]
        csv = csv[csv["Season"].isin(seasons)]
        csv["Path"] = csv["Path"].apply(
            lambda p: [os.path.join(self.root, p, os.path.basename(p))]
        )

        if self.concat_seasons > 1:
            self.files = csv.groupby(["Latitude", "Longitude"])
            self.files = self.files["Path"].agg(sum)
            self.files = self.files[
                self.files.apply(lambda d: len(d) >= self.concat_seasons)
            ]
        else:
            self.files = csv["Path"]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample at that index containing the image with shape SCxHxW,
            where ``S = self.concat_seasons``, and the mask
        """
        image = self._load_image(index)
        mask = self._load_target(index)
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

    def _load_image(self, index: int) -> Tensor:
        """Load image(s) for a single location.

        Args:
            index: index to return

        Returns:
            the stacked seasonal images
        """
        paths = self.files.iloc[index]
        if self.concat_seasons > 1:
            paths = random.sample(paths, self.concat_seasons)
        tensor = torch.empty(self.concat_seasons * self.channels, *self.image_size)
        for img_idx, path in enumerate(paths):
            bnd_idx = 0
            for band in self.bands:
                with rasterio.open(f"{path}_{band}.tif") as f:
                    array = f.read(
                        out_shape=[f.count] + list(self.image_size),
                        out_dtype="int32",
                        resampling=Resampling.bilinear,
                    )
                image = torch.from_numpy(array).float()
                c = img_idx * self.channels + bnd_idx
                tensor[c : c + image.shape[0]] = image
                bnd_idx += image.shape[0]
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single location.

        Args:
            index: index to return

        Returns:
            the target mask
        """
        path = self.files.iloc[index][0]
        with rasterio.open(f"{path}_labels.tif") as f:
            array = f.read() - 1
        tensor = torch.from_numpy(array).squeeze().long()
        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
            ImportError: if ``download=True`` but requests is not installed
        """
        # Check if all files already exist
        if all(
            os.path.exists(os.path.join(self.root, name))
            for name in self.extracted_names
        ):
            return

        # Check for downloaded files and extract existing zips
        missing = []
        for file_name in self.file_names:
            file_path = os.path.join(self.root, file_name)
            if not os.path.exists(file_path):
                missing.append(file_name)
            elif file_name[-3:] == "zip":
                extract_archive(file_path)

        if not missing:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            missing_str = (
                f"{', '.join(missing[:-1])} and {missing[-1]}"
                if len(missing) > 1
                else missing[0]
            )
            raise RuntimeError(
                f"{missing_str} not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True`"
                " to automatically download the dataset."
            )

        # Download missing files
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError(
                "requests is not installed and is required to download this dataset"
            )
        record_info = requests.get(self.url).json()["files"]
        extractable = []
        for file_info in record_info:
            file_path = os.path.join(self.root, file_info["key"])
            if not os.path.exists(file_path):
                download_url(
                    file_info["links"]["self"],
                    self.root,
                    filename=file_info["key"],
                    md5=file_info["checksum"].split(":")[1] if self.checksum else None,
                )
                if file_info["type"] == "zip":
                    extractable.append(file_path)

        # Extract downloaded files
        for file_path in extractable:
            extract_archive(file_path)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        show_legend: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            show_legend: flag indicating whether to show a legend for
                the segmentation masks
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        if "10m_RGB" not in self.bands:
            raise ValueError("Dataset does not contain RGB bands")

        ncols = self.concat_seasons + 1

        images, mask = sample["image"], sample["mask"]
        show_predictions = "prediction" in sample
        if show_predictions:
            prediction = sample["prediction"]
            ncols += 1

        start = 0
        for b in self.bands:
            if b == "10m_RGB":
                break
            start += self.band_nums[b]
        rgb_indices = [start + s * self.channels for s in range(self.concat_seasons)]

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4.5, 5))
        fig.subplots_adjust(wspace=0.05)
        for ax, index in enumerate(rgb_indices):
            image = images[index : index + 3].permute(1, 2, 0).numpy()
            image = percentile_normalization(image)
            axs[ax].imshow(image)
            axs[ax].axis("off")
            if show_titles:
                axs[ax].set_title(f"Image {ax+1}")

        axs[ax + 1].imshow(mask, vmin=0, vmax=32, cmap=self.cmap)
        axs[ax + 1].axis("off")
        if show_titles:
            axs[ax + 1].set_title("Mask")

        if show_predictions:
            axs[ax + 2].imshow(prediction, vmin=0, vmax=32, cmap=self.cmap)
            axs[ax + 2].axis("off")
            if show_titles:
                axs[ax + 2].set_title("Prediction")

        if show_legend:
            lgd = np.unique(mask)

            if show_predictions:
                lgd = np.union1d(lgd, np.unique(prediction))
            patches = [
                mpatches.Patch(color=self.cmap(i), label=self.classes[i]) for i in lgd
            ]
            plt.legend(
                handles=patches, bbox_to_anchor=(1.05, 1), borderaxespad=0, loc=2
            )

        if suptitle is not None:
            plt.suptitle(suptitle, size="xx-large")

        return fig
