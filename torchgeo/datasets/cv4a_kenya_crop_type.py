# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CV4A Kenya Crop Type dataset."""

import csv
import os
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, download_radiant_mlhub_dataset, extract_archive


# TODO: read geospatial information from stac.json files
class CV4AKenyaCropType(NonGeoDataset):
    """CV4A Kenya Crop Type dataset.

    Used in a competition in the Computer NonGeo for Agriculture (CV4A) workshop in
    ICLR 2020. See `this website <https://registry.mlhub.earth/10.34911/rdnt.dw605x/>`__
    for dataset details.

    Consists of 4 tiles of Sentinel 2 imagery from 13 different points in time.

    Each tile has:

    * 13 multi-band observations throughout the growing season. Each observation
      includes 12 bands from Sentinel-2 L2A product, and a cloud probability layer.
      The twelve bands are [B01, B02, B03, B04, B05, B06, B07, B08, B8A,
      B09, B11, B12] (refer to Sentinel-2 documentation for more information about
      the bands). The cloud probability layer is a product of the
      Sentinel-2 atmospheric correction algorithm (Sen2Cor) and provides an estimated
      cloud probability (0-100%) per pixel. All of the bands are mapped to a common
      10 m spatial resolution grid.
    * A raster layer indicating the crop ID for the fields in the training set.
    * A raster layer indicating field IDs for the fields (both training and test sets).
      Fields with a crop ID 0 are the test fields.

    There are 3,286 fields in the train set and 1,402 fields in the test set.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/RDNT.DW605X

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub
    """

    dataset_id = "ref_african_crops_kenya_02"
    image_meta = {
        "filename": "ref_african_crops_kenya_02_source.tar.gz",
        "md5": "9c2004782f6dc83abb1bf45ba4d0da46",
    }
    target_meta = {
        "filename": "ref_african_crops_kenya_02_labels.tar.gz",
        "md5": "93949abd0ae82ba564f5a933cefd8215",
    }

    tile_names = [
        "ref_african_crops_kenya_02_tile_00",
        "ref_african_crops_kenya_02_tile_01",
        "ref_african_crops_kenya_02_tile_02",
        "ref_african_crops_kenya_02_tile_03",
    ]
    dates = [
        "20190606",
        "20190701",
        "20190706",
        "20190711",
        "20190721",
        "20190805",
        "20190815",
        "20190825",
        "20190909",
        "20190919",
        "20190924",
        "20191004",
        "20191103",
    ]
    band_names = (
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

    RGB_BANDS = ["B04", "B03", "B02"]

    # Same for all tiles
    tile_height = 3035
    tile_width = 2016

    def __init__(
        self,
        root: str = "data",
        chip_size: int = 256,
        stride: int = 128,
        bands: Tuple[str, ...] = band_names,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize a new CV4A Kenya Crop Type Dataset instance.

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

        self.root = root
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
        for tile_index in range(len(self.tile_names)):
            for y in list(range(0, self.tile_height - self.chip_size, stride)) + [
                self.tile_height - self.chip_size
            ]:
                for x in list(range(0, self.tile_width - self.chip_size, stride)) + [
                    self.tile_width - self.chip_size
                ]:
                    self.chips_metadata.append((tile_index, y, x))

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        tile_index, y, x = self.chips_metadata[index]
        tile_name = self.tile_names[tile_index]

        img = self._load_all_image_tiles(tile_name, self.bands)
        labels, field_ids = self._load_label_tile(tile_name)

        img = img[:, :, y : y + self.chip_size, x : x + self.chip_size]
        labels = labels[y : y + self.chip_size, x : x + self.chip_size]
        field_ids = field_ids[y : y + self.chip_size, x : x + self.chip_size]

        sample = {
            "image": img,
            "mask": labels,
            "field_ids": field_ids,
            "tile_index": torch.tensor(tile_index),
            "x": torch.tensor(x),
            "y": torch.tensor(y),
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

    @lru_cache(maxsize=128)
    def _load_label_tile(self, tile_name: str) -> Tuple[Tensor, Tensor]:
        """Load a single _tile_ of labels and field_ids.

        Args:
            tile_name: name of tile to load

        Returns:
            tuple of labels and field ids

        Raises:
            AssertionError: if ``tile_name`` is invalid
        """
        assert tile_name in self.tile_names

        if self.verbose:
            print(f"Loading labels/field_ids for {tile_name}")

        directory = os.path.join(
            self.root, "ref_african_crops_kenya_02_labels", tile_name + "_label"
        )

        with Image.open(os.path.join(directory, "labels.tif")) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img)
            labels = torch.from_numpy(array)

        with Image.open(os.path.join(directory, "field_ids.tif")) as img:
            array = np.array(img)
            field_ids = torch.from_numpy(array)

        return (labels, field_ids)

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
            if band not in self.band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    @lru_cache(maxsize=128)
    def _load_all_image_tiles(
        self, tile_name: str, bands: Tuple[str, ...] = band_names
    ) -> Tensor:
        """Load all the imagery (across time) for a single _tile_.

        Optionally allows for subsetting of the bands that are loaded.

        Args:
            tile_name: name of tile to load
            bands: tuple of bands to load

        Returns
            imagery of shape (13, number of bands, 3035, 2016) where 13 is the number of
                points in time, 3035 is the tile height, and 2016 is the tile width

        Raises:
            AssertionError: if ``tile_name`` is invalid
        """
        assert tile_name in self.tile_names

        if self.verbose:
            print(f"Loading all imagery for {tile_name}")

        img = torch.zeros(
            len(self.dates),
            len(bands),
            self.tile_height,
            self.tile_width,
            dtype=torch.float32,
        )

        for date_index, date in enumerate(self.dates):
            img[date_index] = self._load_single_image_tile(tile_name, date, self.bands)

        return img

    @lru_cache(maxsize=128)
    def _load_single_image_tile(
        self, tile_name: str, date: str, bands: Tuple[str, ...]
    ) -> Tensor:
        """Load the imagery for a single tile for a single date.

        Optionally allows for subsetting of the bands that are loaded.

        Args:
            tile_name: name of tile to load
            date: date of tile to load
            bands: bands to load

        Returns:
            array containing a single image tile

        Raises:
            AssertionError: if ``tile_name`` or ``date`` is invalid
        """
        assert tile_name in self.tile_names
        assert date in self.dates

        if self.verbose:
            print(f"Loading imagery for {tile_name} at {date}")

        img = torch.zeros(
            len(bands), self.tile_height, self.tile_width, dtype=torch.float32
        )
        for band_index, band_name in enumerate(self.bands):
            filepath = os.path.join(
                self.root,
                "ref_african_crops_kenya_02_source",
                f"{tile_name}_{date}",
                f"{band_name}.tif",
            )
            with Image.open(filepath) as band_img:
                array: "np.typing.NDArray[np.int_]" = np.array(band_img)
                img[band_index] = torch.from_numpy(array)

        return img

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

    def get_splits(self) -> Tuple[List[int], List[int]]:
        """Get the field_ids for the train/test splits from the dataset directory.

        Returns:
            list of training field_ids and list of testing field_ids
        """
        train_field_ids = []
        test_field_ids = []
        splits_fn = os.path.join(
            self.root,
            "ref_african_crops_kenya_02_labels",
            "_common",
            "field_train_test_ids.csv",
        )

        with open(splits_fn, newline="") as f:
            reader = csv.reader(f)

            # Skip header row
            next(reader)

            for row in reader:
                train_field_ids.append(int(row[0]))
                if row[1]:
                    test_field_ids.append(int(row[1]))

        return train_field_ids, test_field_ids

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

        if "prediction" in sample:
            prediction = sample["prediction"]
            n_cols = 3
        else:
            n_cols = 2

        image, mask = sample["image"], sample["mask"]

        assert time_step <= image.shape[0] - 1, (
            "The specified time step"
            " does not exist, image only contains {} time"
            " instances."
        ).format(image.shape[0])

        image = image[time_step, rgb_indices, :, :]

        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))

        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if "prediction" in sample:
            axs[2].imshow(prediction)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
