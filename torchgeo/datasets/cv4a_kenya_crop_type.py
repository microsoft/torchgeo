import os
from typing import Any, Callable, Dict, Optional, Tuple, List
from functools import lru_cache

import numpy as np

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity


class CV4AKenyaCropType(VisionDataset):
    """CV4A Kenya Crop Type dataset.

    Used in a competition in the Computer Vision for Agriculture (CV4A) workshop in
    ICLR 2020.  See this website <https://registry.mlhub.earth/10.34911/rdnt.dw605x/>
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

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
       imagery and labels from the Radiant Earth MLHub
    """

    base_folder = "ref_african_crops_kenya_02"
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

    # Same for all tiles
    tile_height = 3035
    tile_width = 2016

    def __init__(
        self,
        root: str = "data",
        chip_size: int = 256,
        stride: int = 128,
        bands: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[Image.Image], Any]] = None,
        transforms: Optional[Callable[[Image.Image, Image.Image], Any]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize a new CV4A Kenya Crop Type Dataset instance.

        Parameters:
            root: root directory where dataset can be found
            chip_size (int): size of chips
            stride (int): spacing between chips, if less than chip_size, then there
                will be overlap between chips
            bands (tuple): the subset of bands to load
            transform: a function/transform that takes in a PIL image and returns a
                transformed version
            target_transform: a function/transform that takes in the target and
                transforms it
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
        """
        super().__init__(root, transforms, transform, target_transform)
        self.verbose = verbose

        if download:
            if api_key is None:
                raise RuntimeError(
                    "You must pass an MLHub API key if download=True. "
                    + "See https://www.mlhub.earth/ to register for API access."
                )
            else:
                self.download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        # Calculate the indices that we will use over all tiles
        self.bands = self._validate_bands(bands)
        self.chip_size = chip_size
        self.chips_metadata = []
        for tile_index in range(len(self.tile_names)):
            for y in list(range(0, self.tile_height - self.chip_size, stride)) + [
                self.tile_height - self.chip_size
            ]:
                for x in list(range(0, self.tile_width - self.chip_size, stride)) + [
                    self.tile_width - self.chip_size
                ]:
                    self.chips_metadata.append((tile_index, y, x))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data, label, and field ids at that index
        """
        assert index < len(self)

        tile_index, y, x = self.chips_metadata[index]
        tile_name = self.tile_names[tile_index]

        img = self._load_all_image_tiles(tile_name, self.bands)
        labels, field_ids = self._load_label_tile(tile_name)

        img = img[:, :, y : y + self.chip_size, x : x + self.chip_size]
        labels = labels[y : y + self.chip_size, x : x + self.chip_size]
        field_ids = field_ids[y : y + self.chip_size, x : x + self.chip_size]

        return {
            "img": img,
            "labels": labels,
            "field_ids": field_ids,
            "metadata": (tile_index, y, x),
        }

    def __len__(self) -> int:
        """Return the number of chips in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.chips_metadata)

    @lru_cache
    def _load_label_tile(self, tile_name_: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads a single _tile_ of labels and field_ids"""
        assert tile_name_ in self.tile_names

        if self.verbose:
            print(f"Loading labels/field_ids for {tile_name_}")

        labels = np.array(
            Image.open(
                os.path.join(
                    self.root,
                    self.base_folder,
                    "ref_african_crops_kenya_02_labels",
                    tile_name_ + "_label",
                    "labels.tif",
                )
            )
        )

        field_ids = np.array(
            Image.open(
                os.path.join(
                    self.root,
                    self.base_folder,
                    "ref_african_crops_kenya_02_labels",
                    tile_name_ + "_label",
                    "field_ids.tif",
                )
            )
        )

        return (labels, field_ids)

    def _validate_bands(self, bands: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
        """Routine for validating a list of bands / filling in a default value"""

        if bands is None:
            return self.band_names
        else:
            assert isinstance(bands, tuple), "The list of bands must be a tuple"
            for band in bands:
                if band not in self.band_names:
                    raise ValueError(f"'{band}' is an invalid band name.")
            return bands

    @lru_cache
    def _load_all_image_tiles(
        self, tile_name_: str, bands: Optional[Tuple[str, ...]] = None
    ) -> np.ndarray:
        """Load all the imagery (across time) for a single _tile_. Optionally allows
        for subsetting of the bands that are loaded.

        Returns
            imagery of shape (13, number of bands, 3035, 2016) where 13 is the number of
                points in time, 3035 is the tile height, and 2016 is the tile width.
        """
        assert tile_name_ in self.tile_names
        bands = self._validate_bands(bands)

        if self.verbose:
            print(f"Loading all imagery for {tile_name_}")

        img = np.zeros(
            (len(self.dates), len(bands), self.tile_height, self.tile_width),
            dtype=np.float32,
        )

        for date_index, date in enumerate(self.dates):
            img[date_index] = self._load_single_image_tile(tile_name_, date, bands)

        return img

    @lru_cache
    def _load_single_image_tile(
        self, tile_name_: str, date_: str, bands: Optional[Tuple[str, ...]] = None
    ) -> np.ndarray:
        """Loads the imagery for a single tile for a single date. Optionally allows
        for subsetting of the bands that are loaded."""
        assert tile_name_ in self.tile_names
        assert date_ in self.dates
        bands = self._validate_bands(bands)

        if self.verbose:
            print(f"Loading imagery for {tile_name_} at {date_}")

        img = np.zeros(
            (len(bands), self.tile_height, self.tile_width), dtype=np.float32
        )
        for band_index, band_name in enumerate(bands):
            img_fn = os.path.join(
                self.root,
                self.base_folder,
                "ref_african_crops_kenya_02_source",
                f"{tile_name_}_{date_}",
                f"{band_name}.tif",
            )
            band_img = np.array(Image.open(img_fn))
            img[band_index] = band_img

        return img

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if the MD5s of the dataset's archives match, else False
        """
        images: bool = check_integrity(
            os.path.join(self.root, self.base_folder, self.image_meta["filename"]),
            self.image_meta["md5"],
        )

        targets: bool = check_integrity(
            os.path.join(self.root, self.base_folder, self.target_meta["filename"]),
            self.target_meta["md5"],
        )

        return images and targets

    def get_splits(self) -> Tuple[List[int], List[int]]:
        """Gets the field_ids for the train/test splits from the dataset directory

        Returns:
            list of training field_ids and list of testing field_ids
        """

        train_field_ids = []
        test_field_ids = []
        splits_fn = os.path.join(
            self.root,
            self.base_folder,
            "ref_african_crops_kenya_02_labels",
            "_common",
            "field_train_test_ids.csv",
        )

        with open(splits_fn, "r") as f:
            lines = f.read().strip().split("\n")
            for line in lines[1:]:  # we skip the first line as it is a header
                parts = line.split(",")
                train_field_ids.append(int(parts[0]))
                if parts[1] != "":
                    test_field_ids.append(int(parts[1]))

        return train_field_ids, test_field_ids

    def download(self, api_key: str) -> None:
        """Download the dataset and extract it.

        Parameters:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
        """

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        # Download from MLHub and check integrity
        import radiant_mlhub  # To download from MLHub, could use `requests` instead

        dataset = radiant_mlhub.Dataset.fetch(
            "ref_african_crops_kenya_02", api_key=api_key
        )
        dataset.download(
            output_dir=os.path.join(self.root, self.base_folder), api_key=api_key
        )  # NOTE: Will not work with library versions < 0.2.1

        if not self._check_integrity():
            raise RuntimeError("Dataset files not found or corrupted.")

        # Extract archives
        import tarfile  # To extract .tar.gz archives

        image_archive_path = os.path.join(
            self.root, self.base_folder, self.image_meta["filename"]
        )
        target_archive_path = os.path.join(
            self.root, self.base_folder, self.target_meta["filename"]
        )
        for fn in [image_archive_path, target_archive_path]:
            with tarfile.open(fn) as tfile:
                tfile.extractall(path=os.path.join(self.root, self.base_folder))
