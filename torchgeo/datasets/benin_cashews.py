# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Smallholder Cashew Plantations in Benin dataset."""

import json
import os
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import torch
from rasterio.crs import CRS
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, download_radiant_mlhub_dataset, extract_archive


# TODO: read geospatial information from stac.json files
class BeninSmallHolderCashews(NonGeoDataset):
    r"""Smallholder Cashew Plantations in Benin dataset.

    This dataset contains labels for cashew plantations in a 120 km\ :sup:`2`\  area
    in the center of Benin. Each pixel is classified for Well-managed plantation,
    Poorly-managed plantation, No plantation and other classes. The labels are
    generated using a combination of ground data collection with a handheld GPS device,
    and final corrections based on Airbus Pléiades imagery. See
    `this website <https://doi.org/10.34911/rdnt.hfv20i>`__ for dataset details.

    Specifically, the data consists of Sentinel 2 imagery from a 120 km\ :sup:`2`\  area
    in the center of Benin over 71 points in time from 11/05/2019 to 10/30/2020
    and polygon labels for 6 classes:

    0. No data
    1. Well-managed plantation
    2. Poorly-managed planatation
    3. Non-plantation
    4. Residential
    5. Background
    6. Uncertain

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.34911/rdnt.hfv20i

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub
    """

    dataset_id = "ts_cashew_benin"
    image_meta = {
        "filename": "ts_cashew_benin_source.tar.gz",
        "md5": "957272c86e518a925a4e0d90dab4f92d",
    }
    target_meta = {
        "filename": "ts_cashew_benin_labels.tar.gz",
        "md5": "f9d3f0c671427d852fae9b52a0ae0051",
    }
    dates = (
        "2019_11_05",
        "2019_11_10",
        "2019_11_15",
        "2019_11_20",
        "2019_11_30",
        "2019_12_05",
        "2019_12_10",
        "2019_12_15",
        "2019_12_20",
        "2019_12_25",
        "2019_12_30",
        "2020_01_04",
        "2020_01_09",
        "2020_01_14",
        "2020_01_19",
        "2020_01_24",
        "2020_01_29",
        "2020_02_08",
        "2020_02_13",
        "2020_02_18",
        "2020_02_23",
        "2020_02_28",
        "2020_03_04",
        "2020_03_09",
        "2020_03_14",
        "2020_03_19",
        "2020_03_24",
        "2020_03_29",
        "2020_04_03",
        "2020_04_08",
        "2020_04_13",
        "2020_04_18",
        "2020_04_23",
        "2020_04_28",
        "2020_05_03",
        "2020_05_08",
        "2020_05_13",
        "2020_05_18",
        "2020_05_23",
        "2020_05_28",
        "2020_06_02",
        "2020_06_07",
        "2020_06_12",
        "2020_06_17",
        "2020_06_22",
        "2020_06_27",
        "2020_07_02",
        "2020_07_07",
        "2020_07_12",
        "2020_07_17",
        "2020_07_22",
        "2020_07_27",
        "2020_08_01",
        "2020_08_06",
        "2020_08_11",
        "2020_08_16",
        "2020_08_21",
        "2020_08_26",
        "2020_08_31",
        "2020_09_05",
        "2020_09_10",
        "2020_09_15",
        "2020_09_20",
        "2020_09_25",
        "2020_09_30",
        "2020_10_10",
        "2020_10_15",
        "2020_10_20",
        "2020_10_25",
        "2020_10_30",
    )

    ALL_BANDS = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "CLD",
    )
    RGB_BANDS = ("B04", "B03", "B02")

    classes = [
        "No data",
        "Well-managed planatation",
        "Poorly-managed planatation",
        "Non-planatation",
        "Residential",
        "Background",
        "Uncertain",
    ]

    # Same for all tiles
    tile_height = 1186
    tile_width = 1122

    def __init__(
        self,
        root: str = "data",
        chip_size: int = 256,
        stride: int = 128,
        bands: Tuple[str, ...] = ALL_BANDS,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize a new Benin Smallholder Cashew Plantations Dataset instance.

        Args:
            root: root directory where dataset can be found
            chip_size: size of chips
            stride: spacing between chips, if less than chip_size, then there
                will be overlap between chips
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            verbose: if True, print messages when new tiles are loaded

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self._validate_bands(bands)

        self.root = os.path.expanduser(root)
        self.chip_size = chip_size
        self.stride = stride
        self.bands = bands
        self.transforms = transforms
        self.checksum = checksum
        self.verbose = verbose

        if download:
            self._download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        # Calculate the indices that we will use over all tiles
        self.chips_metadata = []
        for y in list(range(0, self.tile_height - self.chip_size, stride)) + [
            self.tile_height - self.chip_size
        ]:
            for x in list(range(0, self.tile_width - self.chip_size, stride)) + [
                self.tile_width - self.chip_size
            ]:
                self.chips_metadata.append((y, x))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            a dict containing image, mask, transform, crs, and metadata at index.
        """
        y, x = self.chips_metadata[index]

        img, transform, crs = self._load_all_imagery(self.bands)
        labels = self._load_mask(transform)

        img = img[:, :, y : y + self.chip_size, x : x + self.chip_size]
        labels = labels[y : y + self.chip_size, x : x + self.chip_size]

        sample = {
            "image": img,
            "mask": labels,
            "x": torch.tensor(x),
            "y": torch.tensor(y),
            "transform": transform,
            "crs": crs,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of chips in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.chips_metadata)

    def _validate_bands(self, bands: Tuple[str, ...]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided tuple of bands to load

        Raises:
            AssertionError: if ``bands`` is not a tuple
            ValueError: if an invalid band name is provided
        """
        assert isinstance(bands, tuple), "The list of bands must be a tuple"
        for band in bands:
            if band not in self.ALL_BANDS:
                raise ValueError(f"'{band}' is an invalid band name.")

    @lru_cache(maxsize=128)
    def _load_all_imagery(
        self, bands: Tuple[str, ...] = ALL_BANDS
    ) -> Tuple[Tensor, rasterio.Affine, CRS]:
        """Load all the imagery (across time) for the dataset.

        Optionally allows for subsetting of the bands that are loaded.

        Args:
            bands: tuple of bands to load

        Returns:
            imagery of shape (70, number of bands, 1186, 1122) where 70 is the number
            of points in time, 1186 is the tile height, and 1122 is the tile width
            rasterio affine transform, mapping pixel coordinates to geo coordinates
            coordinate reference system of transform
        """
        if self.verbose:
            print("Loading all imagery")

        img = torch.zeros(
            len(self.dates),
            len(bands),
            self.tile_height,
            self.tile_width,
            dtype=torch.float32,
        )

        for date_index, date in enumerate(self.dates):
            single_scene, transform, crs = self._load_single_scene(date, self.bands)
            img[date_index] = single_scene

        return img, transform, crs

    @lru_cache(maxsize=128)
    def _load_single_scene(
        self, date: str, bands: Tuple[str, ...]
    ) -> Tuple[Tensor, rasterio.Affine, CRS]:
        """Load the imagery for a single date.

        Optionally allows for subsetting of the bands that are loaded.

        Args:
            date: date of the imagery to load
            bands: bands to load

        Returns:
            Tensor containing a single image tile, rasterio affine transform,
            mapping pixel coordinates to geo coordinates, and coordinate
            reference system of transform.

        Raises:
            AssertionError: if  ``date`` is invalid
        """
        assert date in self.dates

        if self.verbose:
            print(f"Loading imagery at {date}")

        img = torch.zeros(
            len(bands), self.tile_height, self.tile_width, dtype=torch.float32
        )
        for band_index, band_name in enumerate(self.bands):
            filepath = os.path.join(
                self.root,
                "ts_cashew_benin_source",
                f"ts_cashew_benin_source_00_{date}",
                f"{band_name}.tif",
            )
            with rasterio.open(filepath) as src:
                transform = src.transform  # same transform for every bands
                crs = src.crs
                array = src.read().astype(np.float32)
                img[band_index] = torch.from_numpy(array)

        return img, transform, crs

    @lru_cache()
    def _load_mask(self, transform: rasterio.Affine) -> Tensor:
        """Rasterizes the dataset's labels (in geojson format)."""
        # Create a mask layer out of the geojson
        mask_geojson_fn = os.path.join(
            self.root, "ts_cashew_benin_labels", "_common", "labels.geojson"
        )
        with open(mask_geojson_fn) as f:
            geojson = json.load(f)

        labels = [
            (feature["geometry"], feature["properties"]["class"])
            for feature in geojson["features"]
        ]

        mask_data = rasterio.features.rasterize(
            labels,
            out_shape=(self.tile_height, self.tile_width),
            fill=0,  # nodata value
            transform=transform,
            all_touched=False,
            dtype=np.uint8,
        )

        mask = torch.from_numpy(mask_data).long()
        return mask

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        images: bool = check_integrity(
            os.path.join(self.root, self.image_meta["filename"]),
            self.image_meta["md5"] if self.checksum else None,
        )

        targets: bool = check_integrity(
            os.path.join(self.root, self.target_meta["filename"]),
            self.target_meta["md5"] if self.checksum else None,
        )

        return images and targets

    def _download(self, api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_radiant_mlhub_dataset(self.dataset_id, self.root, api_key)

        image_archive_path = os.path.join(self.root, self.image_meta["filename"])
        target_archive_path = os.path.join(self.root, self.target_meta["filename"])
        for fn in [image_archive_path, target_archive_path]:
            extract_archive(fn, self.root)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        time_step: int = 0,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            time_step: time step at which to access image, beginning with 0
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            ValueError: if the RGB bands are not included in ``self.bands``

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        num_time_points = sample["image"].shape[0]
        assert time_step < num_time_points

        image = np.rollaxis(sample["image"][time_step, rgb_indices].numpy(), 0, 3)
        image = np.clip(image / 3000, 0, 1)
        mask = sample["mask"].numpy()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_panels += 1

        fig, axs = plt.subplots(ncols=num_panels, figsize=(4 * num_panels, 4))

        axs[0].imshow(image)
        axs[0].axis("off")
        if show_titles:
            axs[0].set_title(f"t={time_step}")

        axs[1].imshow(mask, vmin=0, vmax=6, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions, vmin=0, vmax=6, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
